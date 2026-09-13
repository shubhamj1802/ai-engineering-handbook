'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { ArrowRight, PlayCircle } from 'lucide-react';
import { useProgress } from './progress-provider';

interface LiteDoc {
  id: string;
  title: string;
  href: string;
  phase: string;
}

/**
 * "Pick up where you left off": the first lesson in curriculum order that is
 * not yet marked complete. Hidden until progress has hydrated so the server and
 * client markup match.
 */
export function ContinueLearning() {
  const { completed, ready } = useProgress();
  const [docs, setDocs] = useState<LiteDoc[] | null>(null);

  useEffect(() => {
    fetch('/api/search')
      .then((r) => r.json())
      .then((d: { docs: LiteDoc[] }) => setDocs(d.docs))
      .catch(() => setDocs([]));
  }, []);

  if (!ready || !docs || docs.length === 0 || completed.size === 0) return null;

  const next = docs.find((d) => !completed.has(d.id));
  const pct = Math.round((completed.size / docs.length) * 100);

  return (
    <section className="panel mt-6 flex flex-col gap-4 p-5 sm:flex-row sm:items-center">
      <div className="flex-1">
        <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--accent)]">
          <PlayCircle size={12} /> Continue learning
        </p>
        <p className="mt-2 text-[15px] font-medium text-[var(--text)]">
          {next ? next.title : 'You have completed every lesson.'}
        </p>
        <p className="mt-0.5 text-[12.5px] text-[var(--text-dim)]">
          {next ? next.phase : 'Time for the capstones and the production checklist.'} ·{' '}
          {completed.size}/{docs.length} complete ({pct}%)
        </p>
        <div className="mt-3 h-1.5 w-full max-w-sm overflow-hidden rounded-full bg-[var(--panel-2)]">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${pct}%`,
              background: 'linear-gradient(90deg, var(--accent), var(--accent-2))',
            }}
          />
        </div>
      </div>
      {next && (
        <Link href={next.href} className="btn btn-primary !py-2.5 shrink-0">
          Resume <ArrowRight size={15} />
        </Link>
      )}
    </section>
  );
}
