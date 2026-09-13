import { NextResponse } from 'next/server';
import { z } from 'zod';
import { auth } from '@/auth';
import { readProgress, writeProgress, replaceProgress } from '@/lib/progress-store';

export const dynamic = 'force-dynamic';

const BodySchema = z.object({
  completed: z.array(z.string().max(200)).max(5000),
  bookmarks: z.array(z.string().max(200)).max(5000),
  mode: z.enum(['merge', 'replace']).default('merge'),
});

async function userKey(): Promise<string | null> {
  const session = await auth();
  const email = session?.user?.email;
  return email ? email.toLowerCase() : null;
}

export async function GET() {
  const key = await userKey();
  if (!key) return NextResponse.json({ signedIn: false, completed: [], bookmarks: [] });

  try {
    const state = await readProgress(key);
    return NextResponse.json({ signedIn: true, ...state });
  } catch (error) {
    // A storage outage must not break the lesson pages: the client falls back
    // to its localStorage copy when the server cannot answer.
    console.error('progress read failed', error);
    return NextResponse.json(
      { signedIn: true, completed: [], bookmarks: [], degraded: true },
      { status: 200 },
    );
  }
}

export async function POST(request: Request) {
  const key = await userKey();
  if (!key) {
    return NextResponse.json({ error: 'Not signed in' }, { status: 401 });
  }

  let json: unknown;
  try {
    json = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const parsed = BodySchema.safeParse(json);
  if (!parsed.success) {
    return NextResponse.json(
      { error: 'Invalid payload', issues: parsed.error.issues.slice(0, 5) },
      { status: 422 },
    );
  }

  const { completed, bookmarks, mode } = parsed.data;

  try {
    const state =
      mode === 'replace'
        ? await replaceProgress(key, { completed, bookmarks })
        : await writeProgress(key, { completed, bookmarks });
    return NextResponse.json({ signedIn: true, ...state });
  } catch (error) {
    console.error('progress write failed', error);
    return NextResponse.json(
      { error: 'Progress storage unavailable', degraded: true },
      { status: 503 },
    );
  }
}
