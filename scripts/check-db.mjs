#!/usr/bin/env node
/**
 * Verifies DATABASE_URL end to end: connects, creates the progress table,
 * round-trips a row, and deletes it again.
 *
 *   npm run check:db
 *
 * Never prints the connection string or password - only the host, so the
 * output is safe to paste into a chat or an issue.
 */
import { readFileSync, existsSync } from 'node:fs';
import { neon } from '@neondatabase/serverless';

// Minimal .env.local reader: this script runs outside Next.js, which is what
// loads env files normally.
if (existsSync('.env.local')) {
  for (const line of readFileSync('.env.local', 'utf8').split(/\r?\n/)) {
    const match = /^([A-Z_][A-Z0-9_]*)=(.*)$/.exec(line.trim());
    if (match && !process.env[match[1]]) process.env[match[1]] = match[2];
  }
}

const url = process.env.DATABASE_URL || process.env.POSTGRES_URL;

if (!url) {
  console.log('DATABASE_URL is not set.');
  console.log('The app will use JSON files under .data/progress (fine locally,');
  console.log('but it CANNOT work on Vercel - the filesystem is read-only there).');
  process.exit(0);
}

let host;
try {
  host = new URL(url).host;
} catch {
  console.error('FAIL  DATABASE_URL is not a valid URL.');
  console.error('      It should start with postgresql:// and end with ?sslmode=require');
  process.exit(1);
}

console.log(`host    ${host}`);

if (!host.includes('-pooler')) {
  console.warn('WARN  This is the UNPOOLED connection string.');
  console.warn('      Serverless functions open a connection per invocation and will');
  console.warn('      exhaust Neon\'s limit. Use the string with "-pooler" in the host.');
}

const sql = neon(url);
const probe = `__selftest_${Date.now()}`;

try {
  const [{ version }] = await sql`SELECT version()`;
  console.log(`server  ${version.split(',')[0]}`);

  await sql`
    CREATE TABLE IF NOT EXISTS progress (
      user_hash  TEXT PRIMARY KEY,
      completed  JSONB       NOT NULL DEFAULT '[]'::jsonb,
      bookmarks  JSONB       NOT NULL DEFAULT '[]'::jsonb,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )`;
  console.log('table   progress ready');

  await sql`
    INSERT INTO progress (user_hash, completed, bookmarks)
    VALUES (${probe}, '["selftest"]'::jsonb, '[]'::jsonb)
    ON CONFLICT (user_hash) DO UPDATE SET completed = EXCLUDED.completed`;

  const rows = await sql`SELECT completed FROM progress WHERE user_hash = ${probe}`;
  if (rows[0]?.completed?.[0] !== 'selftest') throw new Error('round-trip mismatch');
  console.log('write   ok');
  console.log('read    ok');

  await sql`DELETE FROM progress WHERE user_hash = ${probe}`;
  console.log('cleanup ok');

  const [{ count }] = await sql`SELECT count(*)::int AS count FROM progress`;
  console.log(`\nPASS    database is ready. ${count} real user row(s) stored.`);
} catch (error) {
  console.error(`\nFAIL    ${error.message}`);
  console.error('\nCommon causes:');
  console.error('  - password copied incomplete (Neon shows it only once)');
  console.error('  - missing ?sslmode=require at the end');
  console.error('  - project still provisioning; wait a moment and retry');
  await sql`DELETE FROM progress WHERE user_hash = ${probe}`.catch(() => {});
  process.exit(1);
}
