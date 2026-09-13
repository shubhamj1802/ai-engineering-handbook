import type { Metadata } from 'next';
import Link from 'next/link';
import { googleConfigured } from '@/auth';
import { SignInPanel } from '@/components/sign-in-panel';

export const metadata: Metadata = {
  title: 'Sign in',
  description: 'Sign in with your Google account to sync handbook progress.',
};

export default async function SignInPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string; callbackUrl?: string }>;
}) {
  const { error, callbackUrl } = await searchParams;

  return (
    <div className="grid min-h-[calc(100vh-3.5rem)] place-items-center px-5 py-10">
      <div className="w-full max-w-md">
        <SignInPanel
          configured={googleConfigured}
          error={error}
          callbackUrl={callbackUrl ?? '/'}
        />

        <p className="mt-6 text-center text-[12.5px] text-[var(--text-dim)]">
          Signing in is optional.{' '}
          <Link href="/" className="text-[var(--accent)] hover:underline">
            Keep reading without an account
          </Link>{' '}
          — progress is still stored in this browser.
        </p>
      </div>
    </div>
  );
}
