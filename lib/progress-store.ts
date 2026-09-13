import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { neon } from '@neondatabase/serverless';

/**
 * Server-side progress store.
 *
 * Two backends, chosen at runtime by whether DATABASE_URL is set:
 *
 *   Postgres (Neon)  - used in production. Serverless hosts have a read-only
 *                      filesystem, so files are not an option there.
 *   Local files      - used when no DATABASE_URL is present, so `npm run dev`
 *                      still works with zero setup, exactly as before.
 *
 * In both backends the key is a SHA-256 hash of the lowercased email, never the
 * address itself: a database dump contains no user identities.
 */

export interface ProgressState {
  completed: string[];
  bookmarks: string[];
  updatedAt: string;
}

const EMPTY: ProgressState = { completed: [], bookmarks: [], updatedAt: new Date(0).toISOString() };

const DATABASE_URL = process.env.DATABASE_URL ?? process.env.POSTGRES_URL;
export const usingDatabase = Boolean(DATABASE_URL);

/** Hashed so raw email addresses are never stored, in files or in Postgres. */
function userHash(userKey: string): string {
  return crypto.createHash('sha256').update(userKey).digest('hex').slice(0, 32);
}

function normalise(state: Partial<ProgressState> | undefined): ProgressState {
  return {
    completed: Array.isArray(state?.completed) ? state.completed : [],
    bookmarks: Array.isArray(state?.bookmarks) ? state.bookmarks : [],
    updatedAt: state?.updatedAt ?? EMPTY.updatedAt,
  };
}

/* -------------------------------------------------------------------------- */
/* Postgres backend                                                            */
/* -------------------------------------------------------------------------- */

const sql = DATABASE_URL ? neon(DATABASE_URL) : null;
let schemaReady: Promise<void> | null = null;

/**
 * Created on first use rather than by a migration step: one table, no columns
 * that will ever need altering, and it keeps deployment to "paste the
 * connection string and go".
 */
function ensureSchema(): Promise<void> {
  if (!sql) return Promise.resolve();
  schemaReady ??= (async () => {
    await sql`
      CREATE TABLE IF NOT EXISTS progress (
        user_hash  TEXT PRIMARY KEY,
        completed  JSONB       NOT NULL DEFAULT '[]'::jsonb,
        bookmarks  JSONB       NOT NULL DEFAULT '[]'::jsonb,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
      )
    `;
  })().catch((error) => {
    // Reset so a transient failure at boot does not poison every later request.
    schemaReady = null;
    throw error;
  });
  return schemaReady;
}

async function dbRead(hash: string): Promise<ProgressState> {
  await ensureSchema();
  const rows = (await sql!`
    SELECT completed, bookmarks, updated_at FROM progress WHERE user_hash = ${hash}
  `) as Array<{ completed: string[]; bookmarks: string[]; updated_at: string | Date }>;

  const row = rows[0];
  if (!row) return { ...EMPTY };
  return normalise({
    completed: row.completed,
    bookmarks: row.bookmarks,
    updatedAt: new Date(row.updated_at).toISOString(),
  });
}

async function dbWrite(
  hash: string,
  next: Pick<ProgressState, 'completed' | 'bookmarks'>,
): Promise<ProgressState> {
  await ensureSchema();
  const completed = Array.from(new Set(next.completed));
  const bookmarks = Array.from(new Set(next.bookmarks));

  const rows = (await sql!`
    INSERT INTO progress (user_hash, completed, bookmarks, updated_at)
    VALUES (${hash}, ${JSON.stringify(completed)}::jsonb, ${JSON.stringify(bookmarks)}::jsonb, now())
    ON CONFLICT (user_hash) DO UPDATE
      SET completed = EXCLUDED.completed,
          bookmarks = EXCLUDED.bookmarks,
          updated_at = now()
    RETURNING completed, bookmarks, updated_at
  `) as Array<{ completed: string[]; bookmarks: string[]; updated_at: string | Date }>;

  const row = rows[0];
  return normalise({
    completed: row.completed,
    bookmarks: row.bookmarks,
    updatedAt: new Date(row.updated_at).toISOString(),
  });
}

/* -------------------------------------------------------------------------- */
/* File backend (local development)                                            */
/* -------------------------------------------------------------------------- */

function storeDir(): string {
  return path.join(process.cwd(), process.env.PROGRESS_DIR || '.data/progress');
}

async function fileRead(hash: string): Promise<ProgressState> {
  try {
    const raw = await fs.readFile(path.join(storeDir(), `${hash}.json`), 'utf8');
    return normalise(JSON.parse(raw) as Partial<ProgressState>);
  } catch {
    return { ...EMPTY };
  }
}

async function fileWrite(
  hash: string,
  next: Pick<ProgressState, 'completed' | 'bookmarks'>,
): Promise<ProgressState> {
  const state: ProgressState = {
    completed: Array.from(new Set(next.completed)),
    bookmarks: Array.from(new Set(next.bookmarks)),
    updatedAt: new Date().toISOString(),
  };
  await fs.mkdir(storeDir(), { recursive: true });
  await fs.writeFile(path.join(storeDir(), `${hash}.json`), JSON.stringify(state, null, 2), 'utf8');
  return state;
}

/* -------------------------------------------------------------------------- */
/* Public API - unchanged signatures, so callers did not have to move          */
/* -------------------------------------------------------------------------- */

export async function readProgress(userKey: string): Promise<ProgressState> {
  const hash = userHash(userKey);
  return usingDatabase ? dbRead(hash) : fileRead(hash);
}

/** Union merge: a lesson completed on any device stays completed. */
export async function writeProgress(
  userKey: string,
  incoming: Pick<ProgressState, 'completed' | 'bookmarks'>,
): Promise<ProgressState> {
  const hash = userHash(userKey);
  const current = usingDatabase ? await dbRead(hash) : await fileRead(hash);
  const merged = {
    completed: [...current.completed, ...incoming.completed],
    bookmarks: [...current.bookmarks, ...incoming.bookmarks],
  };
  return usingDatabase ? dbWrite(hash, merged) : fileWrite(hash, merged);
}

/** Replace outright - used when the learner resets progress deliberately. */
export async function replaceProgress(
  userKey: string,
  next: Pick<ProgressState, 'completed' | 'bookmarks'>,
): Promise<ProgressState> {
  const hash = userHash(userKey);
  return usingDatabase ? dbWrite(hash, next) : fileWrite(hash, next);
}
