import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { Clock, GraduationCap, Layers, Lightbulb, ListChecks } from 'lucide-react';
import { getAllLessons, getLesson, getSiblings, getPhaseLessons } from '@/lib/content';
import { renderMarkdown } from '@/lib/markdown';
import { DIFFICULTY_STYLES, BADGE_STYLES } from '@/lib/curriculum';
import { ArticleBody } from '@/components/article-body';
import { Toc } from '@/components/toc';
import { BookmarkButton, CompleteButton, LessonFooterNav } from '@/components/lesson-actions';

interface Params {
  params: Promise<{ phaseId: string; slug: string }>;
}

export function generateStaticParams() {
  return getAllLessons().map((l) => ({ phaseId: l.phaseId, slug: l.slug }));
}

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { phaseId, slug } = await params;
  const lesson = getLesson(phaseId, slug);
  if (!lesson) return { title: 'Not found' };
  return { title: lesson.title, description: lesson.summary };
}

export default async function LessonPage({ params }: Params) {
  const { phaseId, slug } = await params;
  const lesson = getLesson(phaseId, slug);
  if (!lesson) notFound();

  const { html, toc } = renderMarkdown(lesson.body);
  const { prev, next } = getSiblings(lesson);
  const phaseLessons = getPhaseLessons(phaseId);
  const indexInPhase = phaseLessons.findIndex((l) => l.id === lesson.id) + 1;

  return (
    <div className="flex">
      <article className="min-w-0 flex-1 px-5 py-8 sm:px-9 lg:px-12">
        <div className="mx-auto max-w-prose">
          {/* Breadcrumb ------------------------------------------------- */}
          <nav className="mb-5 flex flex-wrap items-center gap-1.5 font-mono text-[11px] text-[var(--text-dim)]">
            <Link href="/roadmap" className="transition hover:text-[var(--accent)]">
              {lesson.phase.group}
            </Link>
            <span>/</span>
            <Link href={`/learn/${phaseId}`} className="transition hover:text-[var(--accent)]">
              Phase {String(lesson.phase.number).padStart(2, '0')} · {lesson.phase.title}
            </Link>
            <span>/</span>
            <span className="text-[var(--text-muted)]">
              Lesson {indexInPhase} of {phaseLessons.length}
            </span>
          </nav>

          {/* Title block ------------------------------------------------ */}
          <header className="mb-8">
            <h1 className="text-[2.1rem] font-semibold leading-[1.15] tracking-tight text-[var(--text)]">
              {lesson.title}
            </h1>
            {lesson.summary && (
              <p className="mt-3 text-[15.5px] leading-relaxed text-[var(--text-muted)]">
                {lesson.summary}
              </p>
            )}

            <div className="mt-5 flex flex-wrap items-center gap-2">
              <span className={`chip ${DIFFICULTY_STYLES[lesson.difficulty]}`}>
                <GraduationCap size={11} />
                {lesson.difficulty}
              </span>
              <span className="chip chip-muted">
                <Clock size={11} />
                {lesson.duration} min read
              </span>
              {(lesson.badges ?? []).map((b) => (
                <span key={b} className={`chip ${BADGE_STYLES[b] ?? 'chip-muted'}`}>
                  {b}
                </span>
              ))}
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <CompleteButton id={lesson.id} />
              <BookmarkButton id={lesson.id} />
            </div>
          </header>

          {/* Body ------------------------------------------------------- */}
          <ArticleBody html={html} />

          <LessonFooterNav
            id={lesson.id}
            prev={prev ? { title: prev.title, href: prev.href } : undefined}
            next={next ? { title: next.title, href: next.href } : undefined}
          />
        </div>
      </article>

      {/* Right context panel ------------------------------------------- */}
      <aside className="no-print sticky top-14 hidden h-[calc(100vh-3.5rem)] w-[262px] shrink-0 overflow-y-auto border-l border-[var(--border)] px-5 py-8 xl:block">
        <Toc entries={toc} />

        {(lesson.prereqs ?? []).length > 0 && (
          <section className="mt-8">
            <p className="mb-2 flex items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-[0.16em] text-[var(--text-dim)]">
              <ListChecks size={11} /> Prerequisites
            </p>
            <ul className="space-y-1 text-[12.5px] text-[var(--text-muted)]">
              {lesson.prereqs!.map((p) => (
                <li key={p} className="flex gap-1.5">
                  <span className="text-[var(--text-dim)]">›</span>
                  {p}
                </li>
              ))}
            </ul>
          </section>
        )}

        {(lesson.keyConcepts ?? []).length > 0 && (
          <section className="mt-8">
            <p className="mb-2 flex items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-[0.16em] text-[var(--text-dim)]">
              <Lightbulb size={11} /> Key concepts
            </p>
            <div className="flex flex-wrap gap-1.5">
              {lesson.keyConcepts!.map((c) => (
                <span
                  key={c}
                  className="rounded-md border border-[var(--border)] bg-[var(--panel)] px-2 py-0.5 font-mono text-[10.5px] text-[var(--text-muted)]"
                >
                  {c}
                </span>
              ))}
            </div>
          </section>
        )}

        <section className="mt-8">
          <p className="mb-2 flex items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-[0.16em] text-[var(--text-dim)]">
            <Layers size={11} /> In this phase
          </p>
          <ul className="space-y-0.5">
            {phaseLessons.map((l) => (
              <li key={l.id}>
                <Link
                  href={l.href}
                  className={`block truncate rounded px-2 py-1 text-[12.5px] transition ${
                    l.id === lesson.id
                      ? 'bg-[color-mix(in_srgb,var(--accent)_12%,transparent)] text-[var(--text)]'
                      : 'text-[var(--text-dim)] hover:text-[var(--text-muted)]'
                  }`}
                >
                  {l.title}
                </Link>
              </li>
            ))}
          </ul>
        </section>

        {next && (
          <section className="mt-8 border-t border-[var(--border)] pt-5">
            <p className="mb-1 text-[10.5px] font-semibold uppercase tracking-[0.16em] text-[var(--text-dim)]">
              Next lesson
            </p>
            <Link
              href={next.href}
              className="block text-[13px] font-medium text-[var(--accent)] hover:underline"
            >
              {next.title} →
            </Link>
          </section>
        )}
      </aside>
    </div>
  );
}
