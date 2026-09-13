'use client';

import { SessionProvider } from 'next-auth/react';
import { ProgressProvider } from './progress-provider';

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <ProgressProvider>{children}</ProgressProvider>
    </SessionProvider>
  );
}
