'use client';

import { useEffect, useState } from 'react';
import clsx from 'clsx';
import type { TocEntry } from '@/lib/markdown';

export function Toc({ entries }: { entries: TocEntry[] }) {
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    if (entries.length === 0) return;
    const headings = entries
      .map((e) => document.getElementById(e.id))
      .filter((el): el is HTMLElement => Boolean(el));

    const observer = new IntersectionObserver(
      (records) => {
        const visible = records
          .filter((r) => r.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActiveId(visible[0].target.id);
      },
      { rootMargin: '-80px 0px -70% 0px', threshold: [0, 1] },
    );

    headings.forEach((h) => observer.observe(h));
    return () => observer.disconnect();
  }, [entries]);

  if (entries.length === 0) return null;

  return (
    <nav aria-label="On this page">
      <p className="mb-2 text-[10.5px] font-semibold uppercase tracking-[0.16em] text-[var(--text-dim)]">
        On this page
      </p>
      <ul className="space-y-0.5 border-l border-[var(--border)]">
        {entries.map((e) => (
          <li key={e.id}>
            <a
              href={`#${e.id}`}
              className={clsx(
                'block py-1 text-[12.5px] leading-snug transition',
                e.level === 3 ? 'pl-5' : 'pl-3',
                activeId === e.id
                  ? 'border-l-2 border-[var(--accent)] -ml-px font-medium text-[var(--text)]'
                  : 'text-[var(--text-dim)] hover:text-[var(--text-muted)]',
              )}
            >
              {e.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
