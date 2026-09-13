import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { ArrowRight, Clock } from 'lucide-react';
import { getPhaseLessons } from '@/lib/content';
import { PHASES, PHASE_BY_ID, DIFFICULTY_STYLES, BADGE_STYLES } from '@/lib/curriculum';

interface Params {
  params: Promise<{ phaseId: string }>;
}

export function generateStaticParams() {
  return PHASES.map((p) => ({ phaseId: p.id }));
}

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { phaseId } = await params;
  const phase = PHASE_BY_ID.get(phaseId);
  return phase
    ? { title: `Phase ${phase.number} · ${phase.title}`, description: phase.blurb }
    : { title: 'Not found' };
}

export default async function PhasePage({ params }: Params) {
  const { phaseId } = await params;
  const phase = PHASE_BY_ID.get(phaseId);
  if (!phase) notFound();
  const lessons = getPhaseLessons(phaseId);
  if (lessons.length === 0) notFound();

  const totalMinutes = lessons.reduce((s, l) => s + l.duration, 0);

  return (
    <div className="px-5 py-10 sm:px-9 lg:px-12">
      <div className="mx-auto max-w-4xl">
        <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-[var(--text-dim)]">
          {phase.group}
        </p>
        <h1 className="mt-2 flex items-baseline gap-3 text-3xl font-semibold tracking-tight">
          <span className="font-mono text-xl text-[var(--accent)]">
            {String(phase.number).padStart(2, '0')}
          </span>
          {phase.title}
        </h1>
        <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-[var(--text-muted)]">
          {phase.blurb}
        </p>

        <div className="mt-5 flex flex-wrap gap-2">
          <span className={`chip ${DIFFICULTY_STYLES[phase.difficulty]}`}>{phase.difficulty}</span>
          <span className="chip chip-muted">{lessons.length} lessons</span>
          <span className="chip chip-muted">
            <Clock size={11} /> ~{Math.round(totalMinutes / 60)}h {totalMinutes % 60}m
          </span>
        </div>

        <ol className="mt-9 space-y-2.5">
          {lessons.map((lesson, i) => (
            <li key={lesson.id}>
              <Link
                href={lesson.href}
                className="panel group flex items-start gap-4 p-4 transition hover:border-[var(--border-strong)]"
              >
                <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg border border-[var(--border)] bg-[var(--panel-2)] font-mono text-[11px] text-[var(--text-dim)]">
                  {String(i + 1).padStart(2, '0')}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex flex-wrap items-center gap-2">
                    <span className="text-[15px] font-medium text-[var(--text)] group-hover:text-[var(--accent)]">
                      {lesson.title}
                    </span>
                    {(lesson.badges ?? []).map((b) => (
                      <span key={b} className={`chip ${BADGE_STYLES[b] ?? 'chip-muted'}`}>
                        {b}
                      </span>
                    ))}
                  </span>
                  {lesson.summary && (
                    <span className="mt-1 block text-[13.5px] leading-relaxed text-[var(--text-dim)]">
                      {lesson.summary}
                    </span>
                  )}
                </span>
                <span className="hidden shrink-0 items-center gap-3 pt-1 sm:flex">
                  <span className="font-mono text-[11px] text-[var(--text-dim)]">
                    {lesson.duration}m
                  </span>
                  <ArrowRight
                    size={14}
                    className="text-[var(--text-dim)] transition group-hover:translate-x-0.5 group-hover:text-[var(--accent)]"
                  />
                </span>
              </Link>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
