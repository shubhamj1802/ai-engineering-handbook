'use client';

import Link from 'next/link';
import clsx from 'clsx';
import { Bookmark, BookmarkCheck, Check, ArrowRight, ArrowLeft } from 'lucide-react';
import { useProgress } from './progress-provider';

export function BookmarkButton({ id }: { id: string }) {
  const { isBookmarked, toggleBookmark } = useProgress();
  const on = isBookmarked(id);
  return (
    <button
      type="button"
      onClick={() => toggleBookmark(id)}
      className={clsx('btn !py-1.5 text-[12.5px]', on && 'text-[var(--accent)]')}
      title={on ? 'Remove bookmark' : 'Bookmark this lesson'}
    >
      {on ? <BookmarkCheck size={14} /> : <Bookmark size={14} />}
      {on ? 'Bookmarked' : 'Bookmark'}
    </button>
  );
}

export function CompleteButton({ id }: { id: string }) {
  const { isComplete, toggleComplete } = useProgress();
  const done = isComplete(id);
  return (
    <button
      type="button"
      onClick={() => toggleComplete(id)}
      className={clsx(
        'btn !py-1.5 text-[12.5px]',
        done && 'border-[var(--ok)]/40 bg-[var(--ok)]/10 text-[var(--ok)]',
      )}
    >
      <Check size={14} />
      {done ? 'Completed' : 'Mark complete'}
    </button>
  );
}

export function LessonFooterNav({
  id,
  prev,
  next,
}: {
  id: string;
  prev?: { title: string; href: string };
  next?: { title: string; href: string };
}) {
  const { isComplete, setComplete } = useProgress();
  const done = isComplete(id);

  return (
    <div className="mt-14 border-t border-[var(--border)] pt-8">
      <div className="panel mb-8 flex flex-col items-start justify-between gap-3 p-4 sm:flex-row sm:items-center">
        <div>
          <p className="text-sm font-medium text-[var(--text)]">
            {done ? 'Lesson complete.' : 'Finished this lesson?'}
          </p>
          <p className="text-[12.5px] text-[var(--text-dim)]">
            {done
              ? 'Your progress is saved on this device, and synced when you sign in with Google.'
              : 'Mark it complete to track your position in the roadmap.'}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setComplete(id, !done)}
            className={clsx('btn !py-2', done ? '' : 'btn-primary')}
          >
            <Check size={15} />
            {done ? 'Mark as not done' : 'Mark complete'}
          </button>
          {next && (
            <Link href={next.href} className="btn !py-2" onClick={() => setComplete(id, true)}>
              Complete & continue <ArrowRight size={15} />
            </Link>
          )}
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {prev ? (
          <Link href={prev.href} className="panel group p-4 transition hover:border-[var(--border-strong)]">
            <span className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-[var(--text-dim)]">
              <ArrowLeft size={12} /> Previous
            </span>
            <span className="mt-1 block text-[14px] font-medium text-[var(--text)] group-hover:text-[var(--accent)]">
              {prev.title}
            </span>
          </Link>
        ) : (
          <div />
        )}
        {next && (
          <Link
            href={next.href}
            className="panel group p-4 text-right transition hover:border-[var(--border-strong)]"
          >
            <span className="flex items-center justify-end gap-1.5 text-[11px] uppercase tracking-wider text-[var(--text-dim)]">
              Next <ArrowRight size={12} />
            </span>
            <span className="mt-1 block text-[14px] font-medium text-[var(--text)] group-hover:text-[var(--accent)]">
              {next.title}
            </span>
          </Link>
        )}
      </div>
    </div>
  );
}
