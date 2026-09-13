'use client';

import Link from 'next/link';
import { useSession, signIn } from 'next-auth/react';
import { BookmarkCheck, Cloud, RotateCcw, Trophy } from 'lucide-react';
import type { CourseStats, GroupWithPhases } from '@/lib/content';
import { useProgress } from './progress-provider';
import { GoogleMark } from './top-bar';

export function ProgressDashboard({
  nav,
  stats,
}: {
  nav: GroupWithPhases[];
  stats: CourseStats;
}) {
  const { completed, bookmarks, resetAll, ready, syncing } = useProgress();
  const { data: session } = useSession();

  const allLessons = nav.flatMap((g) => g.phases.flatMap((p) => p.lessons));
  const bookmarked = allLessons.filter((l) => bookmarks.has(l.id));
  const pct = stats.lessons === 0 ? 0 : Math.round((completed.size / stats.lessons) * 100);
  const minutesDone = allLessons
    .filter((l) => completed.has(l.id))
    .reduce((s, l) => s + l.duration, 0);

  return (
    <div className="px-5 py-10 sm:px-9 lg:px-12">
      <div className="mx-auto max-w-4xl">
        <h1 className="text-3xl font-semibold tracking-tight">My progress</h1>

        {!session?.user && (
          <div className="panel mt-6 flex flex-col gap-3 p-5 sm:flex-row sm:items-center">
            <div className="flex-1">
              <p className="text-[14.5px] font-medium">Progress is saved in this browser only</p>
              <p className="mt-1 text-[13px] text-[var(--text-dim)]">
                Sign in with Google to keep it across browsers and clean installs.
              </p>
            </div>
            <button
              type="button"
              onClick={() => signIn('google', { callbackUrl: '/progress' })}
              className="btn shrink-0 !py-2.5"
            >
              <GoogleMark size={16} /> Sign in with Google
            </button>
          </div>
        )}

        <div className="mt-6 grid gap-3 sm:grid-cols-3">
          <div className="panel p-5">
            <p className="flex items-center gap-2 text-[11px] uppercase tracking-[0.14em] text-[var(--text-dim)]">
              <Trophy size={12} /> Completion
            </p>
            <p className="mt-2 text-3xl font-semibold text-[var(--accent)]">{pct}%</p>
            <p className="text-[12.5px] text-[var(--text-dim)]">
              {ready ? completed.size : 0} of {stats.lessons} lessons
            </p>
          </div>
          <div className="panel p-5">
            <p className="flex items-center gap-2 text-[11px] uppercase tracking-[0.14em] text-[var(--text-dim)]">
              <Cloud size={12} /> Time covered
            </p>
            <p className="mt-2 text-3xl font-semibold">
              {Math.floor(minutesDone / 60)}
              <span className="text-base font-normal text-[var(--text-dim)]">h</span>{' '}
              {minutesDone % 60}
              <span className="text-base font-normal text-[var(--text-dim)]">m</span>
            </p>
            <p className="text-[12.5px] text-[var(--text-dim)]">
              of ~{Math.round(stats.minutes / 60)}h total
            </p>
          </div>
          <div className="panel p-5">
            <p className="flex items-center gap-2 text-[11px] uppercase tracking-[0.14em] text-[var(--text-dim)]">
              <BookmarkCheck size={12} /> Bookmarks
            </p>
            <p className="mt-2 text-3xl font-semibold">{bookmarked.length}</p>
            <p className="text-[12.5px] text-[var(--text-dim)]">
              {session?.user ? (syncing ? 'syncing…' : 'synced to your account') : 'local only'}
            </p>
          </div>
        </div>

        {bookmarked.length > 0 && (
          <section className="mt-10">
            <h2 className="mb-3 text-lg font-semibold tracking-tight">Bookmarked lessons</h2>
            <div className="grid gap-2 sm:grid-cols-2">
              {bookmarked.map((l) => (
                <Link key={l.id} href={l.href} className="panel p-3.5 text-[13.5px] hover:border-[var(--border-strong)]">
                  {l.title}
                </Link>
              ))}
            </div>
          </section>
        )}

        <section className="mt-10">
          <h2 className="mb-4 text-lg font-semibold tracking-tight">By phase</h2>
          <div className="space-y-2">
            {nav.flatMap((g) =>
              g.phases.map(({ phase, lessons }) => {
                const done = lessons.filter((l) => completed.has(l.id)).length;
                const phasePct = Math.round((done / lessons.length) * 100);
                return (
                  <Link
                    key={phase.id}
                    href={`/learn/${phase.id}`}
                    className="panel flex items-center gap-4 p-3.5 transition hover:border-[var(--border-strong)]"
                  >
                    <span className="font-mono text-[11px] text-[var(--text-dim)]">
                      {String(phase.number).padStart(2, '0')}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-[14px]">{phase.title}</span>
                    <span className="hidden w-40 sm:block">
                      <span className="block h-1.5 overflow-hidden rounded-full bg-[var(--panel-2)]">
                        <span
                          className="block h-full rounded-full transition-all"
                          style={{
                            width: `${phasePct}%`,
                            background:
                              phasePct === 100
                                ? 'var(--ok)'
                                : 'linear-gradient(90deg,var(--accent),var(--accent-2))',
                          }}
                        />
                      </span>
                    </span>
                    <span className="w-16 text-right font-mono text-[11px] text-[var(--text-dim)]">
                      {done}/{lessons.length}
                    </span>
                  </Link>
                );
              }),
            )}
          </div>
        </section>

        <section className="mt-10 border-t border-[var(--border)] pt-6">
          <button
            type="button"
            onClick={() => {
              if (confirm('Reset all completion and bookmark data? This cannot be undone.')) {
                resetAll();
              }
            }}
            className="btn text-[13px] text-[var(--danger)]"
          >
            <RotateCcw size={14} /> Reset all progress
          </button>
        </section>
      </div>
    </div>
  );
}
