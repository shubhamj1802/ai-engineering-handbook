import { NextResponse } from 'next/server';
import { getSearchIndex } from '@/lib/content';

/**
 * The search index is built from markdown on the server and fetched once, the
 * first time a learner opens the command palette. Keeping it out of the initial
 * HTML keeps first paint small.
 */
export async function GET() {
  return NextResponse.json(
    { docs: getSearchIndex() },
    { headers: { 'cache-control': 'public, max-age=60, stale-while-revalidate=600' } },
  );
}
