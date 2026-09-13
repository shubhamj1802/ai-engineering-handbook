import type { Metadata } from 'next';
import Link from 'next/link';
import { getNavigationTree, getCourseStats } from '@/lib/content';
import { renderMarkdown } from '@/lib/markdown';
import { ArticleBody } from '@/components/article-body';
import { DIFFICULTY_STYLES } from '@/lib/curriculum';

export const metadata: Metadata = {
  title: 'Roadmap',
  description: 'The full learning path from zero Python to production agentic AI systems.',
};

const DEPENDENCY_MAP = `
\`\`\`mermaid
flowchart TD
  P0["Phase 0<br/>Environment"] --> P1["Phase 1-2<br/>Python"]
  P1 --> P3["Phase 3-5<br/>NumPy · Pandas · Viz"]
  P1 --> P22["Phase 22<br/>Tool use"]
  P3 --> P6["Phase 6-7<br/>Machine learning"]
  P6 --> P8["Phase 8-9<br/>AI · Deep learning"]
  P8 --> P10["Phase 10<br/>LLM fundamentals"]
  P10 --> P11["Phase 11<br/>Embeddings · Vector DBs"]
  P11 --> P12["Phase 12-13<br/>RAG · Advanced RAG"]
  P10 --> P14["Phase 14-15<br/>Agents · Agentic patterns"]
  P22 --> P14
  P12 --> P16["Phase 16-18<br/>LangChain · LangGraph · CrewAI"]
  P14 --> P16
  P16 --> P19["Phase 19-24<br/>Guardrails · HITL · Memory · Observability"]
  P19 --> P25["Phase 25<br/>Production systems"]
  P25 --> P26["Phase 26<br/>Capstones"]
\`\`\`
`;

export default function RoadmapPage() {
  const nav = getNavigationTree();
  const stats = getCourseStats();
  const { html } = renderMarkdown(DEPENDENCY_MAP);

  return (
    <div className="px-5 py-10 sm:px-9 lg:px-12">
      <div className="mx-auto max-w-4xl">
        <h1 className="text-3xl font-semibold tracking-tight">The roadmap</h1>
        <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-[var(--text-muted)]">
          {stats.phases} phases, {stats.lessons} lessons, roughly{' '}
          {Math.round(stats.minutes / 60)} hours of reading plus the time you spend writing code.
          Follow it in order: nothing later assumes anything you have not already built.
        </p>

        <section className="mt-9">
          <h2 className="mb-3 text-lg font-semibold tracking-tight">Technology dependency map</h2>
          <ArticleBody html={html} />
        </section>

        <section className="mt-12">
          <h2 className="mb-4 text-lg font-semibold tracking-tight">Phase by phase</h2>
          <div className="relative border-l border-[var(--border)] pl-6">
            {nav.map((group) => (
              <div key={group.group} className="mb-8">
                <p className="-ml-6 mb-3 inline-block rounded-full border border-[var(--border)] bg-[var(--panel)] px-3 py-1 text-[10.5px] font-semibold uppercase tracking-[0.16em] text-[var(--text-dim)]">
                  {group.group}
                </p>
                <div className="space-y-2.5">
                  {group.phases.map(({ phase, lessons }) => (
                    <Link
                      key={phase.id}
                      href={`/learn/${phase.id}`}
                      className="panel group relative block p-4 transition hover:border-[var(--border-strong)]"
                    >
                      <span className="absolute -left-[31px] top-6 h-2.5 w-2.5 rounded-full border-2 border-[var(--bg)] bg-[var(--accent)]" />
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-[11px] text-[var(--accent)]">
                          PHASE {String(phase.number).padStart(2, '0')}
                        </span>
                        <span className="text-[15px] font-medium group-hover:text-[var(--accent)]">
                          {phase.title}
                        </span>
                        <span className={`chip ${DIFFICULTY_STYLES[phase.difficulty]}`}>
                          {phase.difficulty}
                        </span>
                        <span className="ml-auto font-mono text-[10.5px] text-[var(--text-dim)]">
                          {lessons.length} lessons
                        </span>
                      </div>
                      <p className="mt-1.5 text-[13px] leading-relaxed text-[var(--text-dim)]">
                        {phase.blurb}
                      </p>
                    </Link>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
