'use client';

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useSession } from 'next-auth/react';

const STORAGE_KEY = 'aieh.progress.v1';

interface StoredProgress {
  completed: string[];
  bookmarks: string[];
}

interface ProgressContextValue {
  completed: Set<string>;
  bookmarks: Set<string>;
  ready: boolean;
  syncing: boolean;
  isComplete: (id: string) => boolean;
  isBookmarked: (id: string) => boolean;
  toggleComplete: (id: string) => void;
  setComplete: (id: string, value: boolean) => void;
  toggleBookmark: (id: string) => void;
  resetAll: () => void;
  completedCount: number;
}

const ProgressContext = createContext<ProgressContextValue | null>(null);

function readLocal(): StoredProgress {
  if (typeof window === 'undefined') return { completed: [], bookmarks: [] };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { completed: [], bookmarks: [] };
    const parsed = JSON.parse(raw) as Partial<StoredProgress>;
    return {
      completed: Array.isArray(parsed.completed) ? parsed.completed : [],
      bookmarks: Array.isArray(parsed.bookmarks) ? parsed.bookmarks : [],
    };
  } catch {
    return { completed: [], bookmarks: [] };
  }
}

function writeLocal(state: StoredProgress) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    /* private mode or blocked storage: progress is simply not persisted */
  }
}

export function ProgressProvider({ children }: { children: React.ReactNode }) {
  const { status } = useSession();
  const [completed, setCompleted] = useState<Set<string>>(new Set());
  const [bookmarks, setBookmarks] = useState<Set<string>>(new Set());
  const [ready, setReady] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const pendingSync = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 1. Hydrate from localStorage immediately (works signed out).
  useEffect(() => {
    const local = readLocal();
    setCompleted(new Set(local.completed));
    setBookmarks(new Set(local.bookmarks));
    setReady(true);
  }, []);

  // 2. When signed in, pull the server copy and union-merge it.
  useEffect(() => {
    if (status !== 'authenticated') return;
    let cancelled = false;
    (async () => {
      try {
        setSyncing(true);
        const res = await fetch('/api/progress', { cache: 'no-store' });
        if (!res.ok) return;
        const data = (await res.json()) as { completed?: string[]; bookmarks?: string[] };
        if (cancelled) return;
        setCompleted((prev) => new Set([...prev, ...(data.completed ?? [])]));
        setBookmarks((prev) => new Set([...prev, ...(data.bookmarks ?? [])]));
      } catch {
        /* offline: local progress still works */
      } finally {
        if (!cancelled) setSyncing(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [status]);

  const persist = useCallback(
    (nextCompleted: Set<string>, nextBookmarks: Set<string>) => {
      const payload = {
        completed: Array.from(nextCompleted),
        bookmarks: Array.from(nextBookmarks),
      };
      writeLocal(payload);
      if (status !== 'authenticated') return;
      if (pendingSync.current) clearTimeout(pendingSync.current);
      pendingSync.current = setTimeout(() => {
        setSyncing(true);
        fetch('/api/progress', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ ...payload, mode: 'replace' }),
        })
          .catch(() => undefined)
          .finally(() => setSyncing(false));
      }, 800);
    },
    [status],
  );

  const setComplete = useCallback(
    (id: string, value: boolean) => {
      setCompleted((prev) => {
        const next = new Set(prev);
        if (value) next.add(id);
        else next.delete(id);
        persist(next, bookmarks);
        return next;
      });
    },
    [bookmarks, persist],
  );

  const toggleComplete = useCallback(
    (id: string) => {
      setCompleted((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        persist(next, bookmarks);
        return next;
      });
    },
    [bookmarks, persist],
  );

  const toggleBookmark = useCallback(
    (id: string) => {
      setBookmarks((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        persist(completed, next);
        return next;
      });
    },
    [completed, persist],
  );

  const resetAll = useCallback(() => {
    setCompleted(new Set());
    setBookmarks(new Set());
    persist(new Set(), new Set());
  }, [persist]);

  const value = useMemo<ProgressContextValue>(
    () => ({
      completed,
      bookmarks,
      ready,
      syncing,
      isComplete: (id) => completed.has(id),
      isBookmarked: (id) => bookmarks.has(id),
      toggleComplete,
      setComplete,
      toggleBookmark,
      resetAll,
      completedCount: completed.size,
    }),
    [completed, bookmarks, ready, syncing, toggleComplete, setComplete, toggleBookmark, resetAll],
  );

  return <ProgressContext.Provider value={value}>{children}</ProgressContext.Provider>;
}

export function useProgress(): ProgressContextValue {
  const ctx = useContext(ProgressContext);
  if (!ctx) throw new Error('useProgress must be used inside <ProgressProvider>');
  return ctx;
}
