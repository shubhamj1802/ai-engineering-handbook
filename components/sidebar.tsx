'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import { Check, ChevronRight, Circle, CircleDot } from 'lucide-react';
import type { CourseStats, GroupWithPhases } from '@/lib/content';
import { useProgress } from './progress-provider';

function PhaseProgress({ done, total }: { done: number; total: number }) {
  const pct = total === 0 ? 0 : Math.round((done / total) * 100);
  return (
    <span className="flex items-center gap-1.5" title={`${done} of ${total} lessons complete`}>
      <span className="h-1 w-8 overflow-hidden rounded-full bg-[var(--panel-2)]">
        <span
          className="block h-full rounded-full transition-all"
          style={{
            width: `${pct}%`,
            background:
              pct === 100
                ? 'var(--ok)'
                : 'linear-gradient(90deg, var(--accent), var(--accent-3))',
          }}
        />
      </span>
      <span className="w-7 text-right font-mono text-[10px] text-[var(--text-dim)]">{pct}%</span>
    </span>
  );
}

export function Sidebar({
  nav,
  stats,
  mobileOpen,
  onNavigate,
}: {
  nav: GroupWithPhases[];
  stats: CourseStats;
  mobileOpen: boolean;
  onNavigate: () => void;
}) {
  const pathname = usePathname();
  const { completed, toggleComplete, ready } = useProgress();
  const [openPhases, setOpenPhases] = useState<Record<string, boolean>>({});

  const activePhaseId = useMemo(() => {
    const match = pathname?.match(/^\/learn\/([^/]+)/);
    return match ? match[1] : null;
  }, [pathname]);

  // The phase you are reading is always expanded.
  useEffect(() => {
    if (activePhaseId) setOpenPhases((prev) => ({ ...prev, [activePhaseId]: true }));
  }, [activePhaseId]);

  const overallPct = stats.lessons === 0 ? 0 : Math.round((completed.size / stats.lessons) * 100);

  return (
    <>
      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={onNavigate}
          aria-hidden
        />
      )}

      <aside
        className={clsx(
          'no-print z-40 w-[310px] shrink-0 border-r border-[var(--border)] bg-[var(--bg-elev)]',
          'fixed inset-y-0 left-0 top-14 transform transition-transform duration-200 lg:sticky lg:top-14 lg:translate-x-0',
          'h-[calc(100vh-3.5rem)] overflow-y-auto',
          mobileOpen ? 'translate-x-0' : '-translate-x-full',
        )}
        aria-label="Handbook navigation"
      >
        <div className="border-b border-[var(--border)] px-4 py-4">
          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-dim)]">
              Your progress
            </span>
            <span className="font-mono text-xs text-[var(--accent)]">{overallPct}%</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--panel-2)]">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${overallPct}%`,
                background: 'linear-gradient(90deg, var(--accent), var(--accent-2))',
              }}
            />
          </div>
          <p className="mt-2 text-[11px] text-[var(--text-dim)]">
            {ready ? completed.size : 0} of {stats.lessons} lessons · {Math.round(stats.minutes / 60)}h
            of material
          </p>
        </div>

        <nav className="px-2.5 pb-24 pt-3">
          {nav.map((group) => (
            <div key={group.group} className="mb-5">
              <p className="px-2.5 pb-1.5 text-[10.5px] font-semibold uppercase tracking-[0.16em] text-[var(--text-dim)]">
                {group.group}
              </p>

              {group.phases.map(({ phase, lessons }) => {
                const done = lessons.filter((l) => completed.has(l.id)).length;
                const isOpen = openPhases[phase.id] ?? false;
                const isActive = activePhaseId === phase.id;

                return (
                  <div key={phase.id} className="mb-0.5">
                    <button
                      type="button"
                      onClick={() =>
                        setOpenPhases((prev) => ({ ...prev, [phase.id]: !prev[phase.id] }))
                      }
                      className={clsx(
                        'group flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-[13px] transition',
                        isActive
                          ? 'bg-[color-mix(in_srgb,var(--accent)_10%,transparent)] text-[var(--text)]'
                          : 'text-[var(--text-muted)] hover:bg-[var(--panel-2)] hover:text-[var(--text)]',
                      )}
                      aria-expanded={isOpen}
                    >
                      <ChevronRight
                        size={13}
                        className={clsx(
                          'shrink-0 text-[var(--text-dim)] transition-transform',
                          isOpen && 'rotate-90',
                        )}
                      />
                      <span className="font-mono text-[10px] text-[var(--text-dim)]">
                        {String(phase.number).padStart(2, '0')}
                      </span>
                      <span className="flex-1 truncate font-medium">{phase.title}</span>
                      <PhaseProgress done={done} total={lessons.length} />
                    </button>

                    {isOpen && (
                      <ul className="ml-[13px] border-l border-[var(--border)] pl-1.5">
                        {lessons.map((lesson) => {
                          const isDone = completed.has(lesson.id);
                          const active = pathname === lesson.href;
                          return (
                            <li key={lesson.id} className="flex items-center gap-1">
                              <button
                                type="button"
                                aria-label={
                                  isDone ? `Mark ${lesson.title} incomplete` : `Mark ${lesson.title} complete`
                                }
                                onClick={() => toggleComplete(lesson.id)}
                                className="shrink-0 rounded p-1 text-[var(--text-dim)] transition hover:text-[var(--accent)]"
                              >
                                {isDone ? (
                                  <Check size={12} className="text-[var(--ok)]" />
                                ) : active ? (
                                  <CircleDot size={12} className="text-[var(--accent)]" />
                                ) : (
                                  <Circle size={12} />
                                )}
                              </button>
                              <Link
                                href={lesson.href}
                                onClick={onNavigate}
                                className={clsx(
                                  'block flex-1 truncate rounded-md px-1.5 py-1 text-[12.5px] transition',
                                  active
                                    ? 'bg-[color-mix(in_srgb,var(--accent)_12%,transparent)] font-medium text-[var(--text)]'
                                    : isDone
                                      ? 'text-[var(--text-dim)] hover:text-[var(--text)]'
                                      : 'text-[var(--text-muted)] hover:bg-[var(--panel-2)] hover:text-[var(--text)]',
                                )}
                                title={lesson.title}
                              >
                                {lesson.title}
                              </Link>
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </nav>
      </aside>
    </>
  );
}
