'use client';

import Link from 'next/link';
import Image from 'next/image';
import { useEffect, useState } from 'react';
import { signOut, useSession } from 'next-auth/react';
import { Menu, Moon, Search, Sun, LogOut, Cloud, CloudOff } from 'lucide-react';
import type { CourseStats } from '@/lib/content';
import { useProgress } from './progress-provider';

function ThemeToggle() {
  const [light, setLight] = useState(false);

  useEffect(() => {
    setLight(document.documentElement.classList.contains('light'));
  }, []);

  const toggle = () => {
    const next = !light;
    setLight(next);
    document.documentElement.classList.toggle('light', next);
    document.documentElement.style.colorScheme = next ? 'light' : 'dark';
    try {
      localStorage.setItem('aieh.theme', next ? 'light' : 'dark');
    } catch {
      /* storage blocked: theme resets on reload, nothing else breaks */
    }
  };

  return (
    <button
      type="button"
      onClick={toggle}
      className="rounded-lg border border-[var(--border)] p-2 text-[var(--text-muted)] transition hover:border-[var(--border-strong)] hover:text-[var(--text)]"
      aria-label={light ? 'Switch to dark mode' : 'Switch to light mode'}
      title={light ? 'Dark mode' : 'Light mode'}
    >
      {light ? <Moon size={15} /> : <Sun size={15} />}
    </button>
  );
}

function AuthButton() {
  const { data: session, status } = useSession();
  const { syncing } = useProgress();
  const [menuOpen, setMenuOpen] = useState(false);

  if (status === 'loading') {
    return <div className="h-8 w-24 animate-pulse rounded-lg bg-[var(--panel-2)]" />;
  }

  if (!session?.user) {
    // Routed through /signin rather than calling signIn() directly: that page
    // explains the Google OAuth setup when credentials are not configured yet.
    return (
      <Link href="/signin" className="btn btn-primary !py-1.5 text-[13px]">
        <GoogleMark />
        <span className="hidden sm:inline">Sign in with Google</span>
        <span className="sm:hidden">Sign in</span>
      </Link>
    );
  }

  const name = session.user.name ?? session.user.email ?? 'Learner';

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setMenuOpen((v) => !v)}
        className="flex items-center gap-2 rounded-lg border border-[var(--border)] py-1 pl-1 pr-2.5 transition hover:border-[var(--border-strong)]"
      >
        {session.user.image ? (
          <Image
            src={session.user.image}
            alt=""
            width={24}
            height={24}
            className="rounded-full"
            unoptimized
          />
        ) : (
          <span className="grid h-6 w-6 place-items-center rounded-full bg-[var(--accent)] text-[11px] font-bold text-black">
            {name.charAt(0).toUpperCase()}
          </span>
        )}
        <span className="hidden max-w-[140px] truncate text-[13px] text-[var(--text-muted)] sm:inline">
          {name}
        </span>
        {syncing ? (
          <Cloud size={13} className="animate-pulse text-[var(--accent)]" />
        ) : (
          <Cloud size={13} className="text-[var(--text-dim)]" />
        )}
      </button>

      {menuOpen && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setMenuOpen(false)} aria-hidden />
          <div className="panel absolute right-0 z-50 mt-2 w-64 p-3 shadow-xl">
            <p className="truncate text-sm font-medium text-[var(--text)]">{name}</p>
            <p className="truncate text-xs text-[var(--text-dim)]">{session.user.email}</p>
            <p className="mt-3 flex items-center gap-1.5 text-[11px] text-[var(--text-muted)]">
              <Cloud size={12} className="text-[var(--ok)]" />
              Progress syncs to this machine
            </p>
            <Link
              href="/progress"
              onClick={() => setMenuOpen(false)}
              className="btn mt-3 w-full !py-1.5 text-[13px]"
            >
              My progress
            </Link>
            <button
              type="button"
              onClick={() => signOut({ callbackUrl: '/' })}
              className="btn mt-2 w-full !py-1.5 text-[13px]"
            >
              <LogOut size={13} /> Sign out
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export function GoogleMark({ size = 15 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" aria-hidden>
      <path
        fill="#FFC107"
        d="M43.6 20.5H42V20H24v8h11.3C33.7 32.9 29.3 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 13 4 4 13 4 24s9 20 20 20 20-9 20-20c0-1.2-.1-2.3-.4-3.5z"
      />
      <path
        fill="#FF3D00"
        d="m6.3 14.7 6.6 4.8C14.7 15.1 19 12 24 12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z"
      />
      <path
        fill="#4CAF50"
        d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35.1 26.7 36 24 36c-5.2 0-9.6-3.1-11.3-7.5l-6.5 5C9.5 39.6 16.2 44 24 44z"
      />
      <path
        fill="#1976D2"
        d="M43.6 20.5H42V20H24v8h11.3c-.8 2.3-2.3 4.2-4.1 5.6l6.2 5.2C36.9 40.2 44 35 44 24c0-1.2-.1-2.3-.4-3.5z"
      />
    </svg>
  );
}

export function TopBar({
  onOpenSearch,
  onToggleNav,
  stats,
}: {
  onOpenSearch: () => void;
  onToggleNav: () => void;
  stats: CourseStats;
}) {
  const { status } = useSession();

  return (
    <header className="no-print sticky top-0 z-50 h-14 border-b border-[var(--border)] bg-[color-mix(in_srgb,var(--bg)_86%,transparent)] backdrop-blur-xl">
      <div className="mx-auto flex h-full w-full max-w-[1680px] items-center gap-3 px-3 sm:px-4">
        <button
          type="button"
          onClick={onToggleNav}
          className="rounded-lg border border-[var(--border)] p-2 text-[var(--text-muted)] lg:hidden"
          aria-label="Toggle navigation"
        >
          <Menu size={15} />
        </button>

        <Link href="/" className="group flex items-center gap-2.5">
          <span className="grid h-7 w-7 place-items-center rounded-lg bg-gradient-to-br from-[var(--accent)] to-[var(--accent-2)] font-mono text-[11px] font-bold text-black">
            AI
          </span>
          <span className="hidden leading-tight sm:block">
            <span className="block text-[13px] font-semibold tracking-tight text-[var(--text)]">
              AI Engineering Handbook
            </span>
            <span className="block font-mono text-[10px] text-[var(--text-dim)]">
              Python → ML → LLMs → RAG → Agents
            </span>
          </span>
        </Link>

        <div className="flex-1" />

        <button
          type="button"
          onClick={onOpenSearch}
          className="flex items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--panel)] px-3 py-1.5 text-[13px] text-[var(--text-dim)] transition hover:border-[var(--border-strong)] hover:text-[var(--text-muted)]"
        >
          <Search size={14} />
          <span className="hidden md:inline">Search {stats.lessons} lessons…</span>
          <kbd className="ml-1 hidden rounded border border-[var(--border)] px-1.5 py-0.5 font-mono text-[10px] md:inline">
            ⌘K
          </kbd>
        </button>

        <Link
          href="/roadmap"
          className="hidden rounded-lg px-2.5 py-1.5 text-[13px] text-[var(--text-muted)] transition hover:text-[var(--text)] md:block"
        >
          Roadmap
        </Link>

        <ThemeToggle />

        {status === 'unauthenticated' ? (
          <span className="hidden items-center gap-1 text-[11px] text-[var(--text-dim)] xl:flex">
            <CloudOff size={12} /> local only
          </span>
        ) : null}

        <AuthButton />
      </div>
    </header>
  );
}
