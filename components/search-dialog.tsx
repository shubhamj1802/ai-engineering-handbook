'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import clsx from 'clsx';
import { CornerDownLeft, Loader2, Search } from 'lucide-react';
import type { SearchDoc } from '@/lib/content';

interface Scored {
  doc: SearchDoc;
  score: number;
  snippet: string;
}

/**
 * Small, dependency-free ranked search.
 * Weights: title > headings > summary > body. Every query term must appear
 * somewhere in the document (AND semantics) which keeps results tight.
 */
function search(docs: SearchDoc[], query: string): Scored[] {
  const terms = query
    .toLowerCase()
    .split(/\s+/)
    .map((t) => t.trim())
    .filter(Boolean);
  if (terms.length === 0) return [];

  const results: Scored[] = [];

  for (const doc of docs) {
    const title = doc.title.toLowerCase();
    const headings = doc.headings.join('  ').toLowerCase();
    const summary = doc.summary.toLowerCase();
    const text = doc.text.toLowerCase();
    const phase = `${doc.phase} ${doc.group}`.toLowerCase();

    let score = 0;
    let matchedAll = true;

    for (const term of terms) {
      let termScore = 0;
      if (title.startsWith(term)) termScore += 60;
      if (title.includes(term)) termScore += 40;
      if (phase.includes(term)) termScore += 12;
      if (headings.includes(term)) termScore += 18;
      if (summary.includes(term)) termScore += 10;
      const occurrences = text.split(term).length - 1;
      termScore += Math.min(occurrences, 6) * 3;
      if (termScore === 0) matchedAll = false;
      score += termScore;
    }

    if (!matchedAll || score === 0) continue;

    const first = terms[0];
    const at = text.indexOf(first);
    const snippet =
      at >= 0
        ? `…${doc.text.slice(Math.max(0, at - 60), Math.min(doc.text.length, at + 120)).trim()}…`
        : doc.summary;

    results.push({ doc, score, snippet });
  }

  return results.sort((a, b) => b.score - a.score).slice(0, 24);
}

export function SearchDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const router = useRouter();
  const [docs, setDocs] = useState<SearchDoc[] | null>(null);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // The index is fetched once, lazily, the first time the palette opens.
  useEffect(() => {
    if (!open || docs) return;
    fetch('/api/search')
      .then((r) => r.json())
      .then((d: { docs: SearchDoc[] }) => setDocs(d.docs))
      .catch(() => setDocs([]));
  }, [open, docs]);

  useEffect(() => {
    if (open) {
      setActive(0);
      const t = setTimeout(() => inputRef.current?.focus(), 20);
      return () => clearTimeout(t);
    }
  }, [open]);

  const results = useMemo(() => (docs ? search(docs, query) : []), [docs, query]);

  useEffect(() => {
    setActive(0);
  }, [query]);

  if (!open) return null;

  const go = (href: string) => {
    onClose();
    setQuery('');
    router.push(href);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter' && results[active]) {
      e.preventDefault();
      go(results[active].doc.href);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[60] flex items-start justify-center bg-black/70 px-4 pt-[10vh] backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Search the handbook"
    >
      <div
        className="panel w-full max-w-2xl overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <div className="flex items-center gap-3 border-b border-[var(--border)] px-4 py-3">
          <Search size={16} className="text-[var(--text-dim)]" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search lessons, concepts, code…  (try: reducer, HyDE, pytest, guardrail)"
            className="w-full bg-transparent text-[15px] text-[var(--text)] outline-none placeholder:text-[var(--text-dim)]"
          />
          {docs === null && <Loader2 size={15} className="animate-spin text-[var(--text-dim)]" />}
          <kbd className="rounded border border-[var(--border)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-dim)]">
            esc
          </kbd>
        </div>

        <div ref={listRef} className="max-h-[55vh] overflow-y-auto p-2">
          {query.length === 0 && (
            <p className="px-3 py-6 text-center text-[13px] text-[var(--text-dim)]">
              Type to search across every phase of the handbook.
            </p>
          )}

          {query.length > 0 && results.length === 0 && docs !== null && (
            <p className="px-3 py-6 text-center text-[13px] text-[var(--text-dim)]">
              No lesson matches <span className="text-[var(--text)]">{query}</span>.
            </p>
          )}

          {results.map((r, i) => (
            <button
              key={r.doc.id}
              type="button"
              onMouseEnter={() => setActive(i)}
              onClick={() => go(r.doc.href)}
              className={clsx(
                'block w-full rounded-lg px-3 py-2.5 text-left transition',
                i === active ? 'bg-[color-mix(in_srgb,var(--accent)_12%,transparent)]' : 'hover:bg-[var(--panel-2)]',
              )}
            >
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] text-[var(--text-dim)]">
                  {String(r.doc.phaseNumber).padStart(2, '0')}
                </span>
                <span className="truncate text-[14px] font-medium text-[var(--text)]">
                  {r.doc.title}
                </span>
                <span className="ml-auto shrink-0 font-mono text-[10px] text-[var(--text-dim)]">
                  {r.doc.phase}
                </span>
                {i === active && <CornerDownLeft size={12} className="text-[var(--accent)]" />}
              </div>
              <p className="mt-1 line-clamp-2 text-[12px] leading-relaxed text-[var(--text-dim)]">
                {r.snippet}
              </p>
            </button>
          ))}
        </div>

        <div className="flex items-center justify-between border-t border-[var(--border)] px-4 py-2 font-mono text-[10px] text-[var(--text-dim)]">
          <span>↑↓ navigate · ↵ open · esc close</span>
          <span>{docs ? `${docs.length} lessons indexed` : 'loading index…'}</span>
        </div>
      </div>
    </div>
  );
}
