import Link from 'next/link';
import {
  ArrowRight,
  BookOpen,
  Boxes,
  Brain,
  Clock,
  Code2,
  Database,
  GitBranch,
  Network,
  ShieldCheck,
  Sparkles,
  Workflow,
} from 'lucide-react';
import { getCourseStats, getNavigationTree, getAllLessons } from '@/lib/content';
import { PHASES } from '@/lib/curriculum';
import { ContinueLearning } from '@/components/continue-learning';

const TRACKS = [
  {
    icon: Code2,
    title: 'Python, properly',
    body: 'Zero to production Python: data structures, functions, OOP, typing, async, tests and packaging.',
    phases: '0–2',
    href: '/learn/phase-01-python-fundamentals',
  },
  {
    icon: Database,
    title: 'Data & classical ML',
    body: 'NumPy, Pandas, visualization, then supervised and unsupervised learning with scikit-learn.',
    phases: '3–7',
    href: '/learn/phase-03-numpy',
  },
  {
    icon: Brain,
    title: 'Deep learning & LLMs',
    body: 'Neural networks in PyTorch, transformers and attention, then real LLM API engineering.',
    phases: '8–10',
    href: '/learn/phase-08-ai-fundamentals',
  },
  {
    icon: Network,
    title: 'Embeddings & RAG',
    body: 'Vector search from first principles, hand-built RAG, then hybrid retrieval and reranking.',
    phases: '11–13',
    href: '/learn/phase-11-embeddings-vector-db',
  },
  {
    icon: Workflow,
    title: 'Agents & frameworks',
    body: 'Agents from scratch, agentic patterns, then LangChain, LangGraph and CrewAI.',
    phases: '14–18',
    href: '/learn/phase-14-ai-agents',
  },
  {
    icon: ShieldCheck,
    title: 'Production systems',
    body: 'Guardrails, human-in-the-loop, memory, observability, evaluation, FastAPI and Docker.',
    phases: '19–26',
    href: '/learn/phase-19-guardrails',
  },
];

export default function HomePage() {
  const stats = getCourseStats();
  const nav = getNavigationTree();
  const first = getAllLessons()[0];

  return (
    <div className="px-5 py-10 sm:px-9 lg:px-12">
      <div className="mx-auto max-w-5xl">
        {/* Hero ------------------------------------------------------------ */}
        <section className="relative overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-7 sm:p-10">
          <div className="surface-grid pointer-events-none absolute inset-0 opacity-[0.35]" />
          <div className="pointer-events-none absolute -right-20 -top-24 h-64 w-64 rounded-full bg-[var(--accent)] opacity-[0.08] blur-3xl" />
          <div className="pointer-events-none absolute -bottom-24 -left-16 h-64 w-64 rounded-full bg-[var(--accent-2)] opacity-[0.08] blur-3xl" />

          <div className="relative">
            <span className="chip chip-muted">
              <Sparkles size={11} className="text-[var(--accent)]" />
              {stats.phases} phases · {stats.lessons} lessons · ~{Math.round(stats.minutes / 60)}h
            </span>

            <h1 className="mt-5 text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">
              AI Engineering
              <br />
              <span className="bg-gradient-to-r from-[var(--accent)] via-[var(--accent-3)] to-[var(--accent-2)] bg-clip-text text-transparent">
                Handbook
              </span>
            </h1>

            <p className="mt-4 max-w-2xl font-mono text-[13px] text-[var(--text-dim)]">
              Python → Data Science → ML → Deep Learning → LLMs → RAG → Agents → Agentic AI →
              Production
            </p>

            <p className="mt-4 max-w-2xl text-[15.5px] leading-relaxed text-[var(--text-muted)]">
              Learn the foundations. Build the systems. Ship production-grade AI. Every concept is
              explained from first principles, implemented by hand, and only then rebuilt with a
              framework — so you understand what the framework is doing for you.
            </p>

            <div className="mt-7 flex flex-wrap gap-3">
              <Link href={first?.href ?? '/roadmap'} className="btn btn-primary !px-4 !py-2.5">
                Start with Phase 0 <ArrowRight size={16} />
              </Link>
              <Link href="/roadmap" className="btn !px-4 !py-2.5">
                <BookOpen size={15} /> See the full roadmap
              </Link>
            </div>
          </div>
        </section>

        <ContinueLearning />

        {/* Tracks ---------------------------------------------------------- */}
        <section className="mt-12">
          <h2 className="text-lg font-semibold tracking-tight">The six tracks</h2>
          <p className="mt-1 text-[14px] text-[var(--text-muted)]">
            Sequential by design. Each track assumes only what came before it.
          </p>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {TRACKS.map((t) => (
              <Link
                key={t.title}
                href={t.href}
                className="panel group relative overflow-hidden p-5 transition hover:border-[var(--border-strong)]"
              >
                <div className="flex items-start justify-between">
                  <span className="grid h-9 w-9 place-items-center rounded-lg border border-[var(--border)] bg-[var(--panel-2)] text-[var(--accent)]">
                    <t.icon size={16} />
                  </span>
                  <span className="font-mono text-[10px] text-[var(--text-dim)]">
                    PHASES {t.phases}
                  </span>
                </div>
                <h3 className="mt-3 text-[15px] font-medium text-[var(--text)] group-hover:text-[var(--accent)]">
                  {t.title}
                </h3>
                <p className="mt-1.5 text-[13px] leading-relaxed text-[var(--text-dim)]">{t.body}</p>
              </Link>
            ))}
          </div>
        </section>

        {/* Teaching method -------------------------------------------------- */}
        <section className="mt-12 grid gap-3 md:grid-cols-3">
          {[
            {
              icon: GitBranch,
              title: 'Concept before framework',
              body: 'RAG by hand before LangChain. A state machine by hand before LangGraph. An agent loop by hand before any agent SDK.',
            },
            {
              icon: Boxes,
              title: 'Every example is runnable',
              body: 'File name, folder path, full imports, how to run it, expected output, common errors, and what changes in production.',
            },
            {
              icon: ShieldCheck,
              title: 'Production is not an epilogue',
              body: 'Guardrails, evaluation, tracing, cost control and human approval are taught as first-class engineering, not afterthoughts.',
            },
          ].map((c) => (
            <div key={c.title} className="panel p-5">
              <c.icon size={16} className="text-[var(--accent-2)]" />
              <h3 className="mt-3 text-[14.5px] font-medium">{c.title}</h3>
              <p className="mt-1.5 text-[13px] leading-relaxed text-[var(--text-dim)]">{c.body}</p>
            </div>
          ))}
        </section>

        {/* Full phase index -------------------------------------------------- */}
        <section className="mt-12">
          <h2 className="text-lg font-semibold tracking-tight">All {PHASES.length} phases</h2>
          <div className="mt-5 space-y-7">
            {nav.map((group) => (
              <div key={group.group}>
                <p className="mb-2.5 text-[10.5px] font-semibold uppercase tracking-[0.16em] text-[var(--text-dim)]">
                  {group.group}
                </p>
                <div className="grid gap-2 sm:grid-cols-2">
                  {group.phases.map(({ phase, lessons }) => (
                    <Link
                      key={phase.id}
                      href={`/learn/${phase.id}`}
                      className="panel group flex items-start gap-3 p-3.5 transition hover:border-[var(--border-strong)]"
                    >
                      <span className="mt-0.5 font-mono text-[11px] text-[var(--accent)]">
                        {String(phase.number).padStart(2, '0')}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block text-[14px] font-medium text-[var(--text)] group-hover:text-[var(--accent)]">
                          {phase.title}
                        </span>
                        <span className="mt-0.5 block text-[12.5px] leading-relaxed text-[var(--text-dim)]">
                          {phase.blurb}
                        </span>
                        <span className="mt-1.5 flex items-center gap-2 font-mono text-[10px] text-[var(--text-dim)]">
                          <Clock size={10} />
                          {lessons.reduce((s, l) => s + l.duration, 0)}m · {lessons.length} lessons
                        </span>
                      </span>
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
