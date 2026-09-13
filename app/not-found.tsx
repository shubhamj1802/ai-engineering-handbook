import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="grid min-h-[70vh] place-items-center px-5 text-center">
      <div>
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-[var(--text-dim)]">
          404
        </p>
        <h1 className="mt-3 text-2xl font-semibold tracking-tight">This lesson does not exist</h1>
        <p className="mt-2 text-[14px] text-[var(--text-muted)]">
          It may have been renamed. Try the roadmap or search with ⌘K.
        </p>
        <Link href="/roadmap" className="btn btn-primary mt-6 !py-2.5">
          Back to the roadmap
        </Link>
      </div>
    </div>
  );
}
