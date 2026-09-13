'use client';

import { useCallback, useEffect, useState } from 'react';
import type { CourseStats, GroupWithPhases } from '@/lib/content';
import { Sidebar } from './sidebar';
import { TopBar } from './top-bar';
import { SearchDialog } from './search-dialog';

export function AppShell({
  nav,
  stats,
  children,
}: {
  nav: GroupWithPhases[];
  stats: CourseStats;
  children: React.ReactNode;
}) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  // Cmd/Ctrl+K anywhere opens the palette; "/" works outside inputs.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.isContentEditable);
      if ((e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setSearchOpen((v) => !v);
      } else if (e.key === '/' && !typing) {
        e.preventDefault();
        setSearchOpen(true);
      } else if (e.key === 'Escape') {
        setMobileNavOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const closeMobileNav = useCallback(() => setMobileNavOpen(false), []);

  return (
    <div className="min-h-screen">
      <TopBar
        onOpenSearch={() => setSearchOpen(true)}
        onToggleNav={() => setMobileNavOpen((v) => !v)}
        stats={stats}
      />

      <div className="mx-auto flex w-full max-w-[1680px]">
        <Sidebar nav={nav} stats={stats} mobileOpen={mobileNavOpen} onNavigate={closeMobileNav} />
        <main className="min-w-0 flex-1">{children}</main>
      </div>

      <SearchDialog open={searchOpen} onClose={() => setSearchOpen(false)} />
    </div>
  );
}
