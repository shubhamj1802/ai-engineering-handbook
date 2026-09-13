'use client';

import { useEffect, useState } from 'react';
import { signIn, useSession } from 'next-auth/react';
import Link from 'next/link';
import { AlertTriangle, CheckCircle2, Cloud, Laptop, ShieldCheck } from 'lucide-react';
import { GoogleMark } from './top-bar';

const ERROR_COPY: Record<string, string> = {
  Configuration:
    'Sign-in could not start. This usually means the page was opened directly or reloaded mid-flow — use the button below rather than a bookmarked link.',
  AccessDenied: 'That account was not allowed to sign in.',
  Verification: 'The sign-in link expired. Try again.',
  OAuthCallback:
    'Google rejected the callback. The redirect URI registered for this app does not match where you are signed in from.',
  OAuthSignin: 'Could not reach Google. Check your connection and try again.',
};

export function SignInPanel({
  configured,
  error,
  callbackUrl,
}: {
  configured: boolean;
  error?: string;
  callbackUrl: string;
}) {
  const { data: session } = useSession();
  // Read after mount: the server has no window, and hardcoding localhost would
  // print the wrong redirect URI on a deployed instance.
  const [origin, setOrigin] = useState('');
  useEffect(() => setOrigin(window.location.origin), []);

  if (session?.user) {
    return (
      <div className="panel p-7 text-center">
        <CheckCircle2 size={28} className="mx-auto text-[var(--ok)]" />
        <h1 className="mt-4 text-xl font-semibold">You are signed in</h1>
        <p className="mt-1.5 text-[13.5px] text-[var(--text-muted)]">
          {session.user.email} — progress now syncs whenever you complete a lesson.
        </p>
        <Link href="/progress" className="btn btn-primary mt-5 w-full !py-2.5">
          View my progress
        </Link>
      </div>
    );
  }

  return (
    <div className="panel overflow-hidden">
      <div className="border-b border-[var(--border)] bg-[var(--panel-2)] px-7 py-6 text-center">
        <span className="mx-auto grid h-11 w-11 place-items-center rounded-xl bg-gradient-to-br from-[var(--accent)] to-[var(--accent-2)] font-mono text-[13px] font-bold text-black">
          AI
        </span>
        <h1 className="mt-4 text-xl font-semibold tracking-tight">Sign in to the handbook</h1>
        <p className="mt-1.5 text-[13.5px] leading-relaxed text-[var(--text-muted)]">
          Use your Gmail or Google Workspace account to keep lesson progress and bookmarks across
          every browser and device you read on.
        </p>
      </div>

      <div className="p-7">
        {error && (
          <div className="mb-5 flex gap-2.5 rounded-lg border border-[var(--danger)]/40 bg-[var(--danger)]/10 px-3.5 py-3">
            <AlertTriangle size={15} className="mt-0.5 shrink-0 text-[var(--danger)]" />
            <p className="text-[12.5px] leading-relaxed text-[var(--text-muted)]">
              {ERROR_COPY[error] ?? `Sign-in failed (${error}).`}
            </p>
          </div>
        )}

        {configured ? (
          <button
            type="button"
            onClick={() => signIn('google', { callbackUrl })}
            className="btn w-full !justify-center !py-3 text-[14px] font-medium"
          >
            <GoogleMark size={17} />
            Continue with Google
          </button>
        ) : (
          <div className="rounded-lg border border-[var(--warn)]/40 bg-[var(--warn)]/10 p-4">
            <p className="flex items-center gap-2 text-[13px] font-semibold text-[var(--warn)]">
              <AlertTriangle size={14} /> Google OAuth is not configured yet
            </p>
            <p className="mt-2 text-[12.5px] leading-relaxed text-[var(--text-muted)]">
              Create an OAuth client in the Google Cloud Console, then add the credentials to{' '}
              <code className="rounded bg-black/30 px-1 py-0.5 font-mono text-[11px]">
                .env.local
              </code>
              :
            </p>
            <pre className="mt-3 overflow-x-auto rounded-lg bg-black/40 p-3 font-mono text-[11px] leading-relaxed text-[var(--text-muted)]">
{`AUTH_SECRET=<openssl rand -base64 32>
GOOGLE_CLIENT_ID=...apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=...`}
            </pre>
            <p className="mt-3 text-[12px] text-[var(--text-dim)]">
              Authorised redirect URI:{' '}
              <code className="font-mono">{origin}/api/auth/callback/google</code>
            </p>
          </div>
        )}

        <ul className="mt-6 space-y-3 text-[12.5px] text-[var(--text-muted)]">
          <li className="flex gap-2.5">
            <Cloud size={14} className="mt-0.5 shrink-0 text-[var(--accent)]" />
            Completed lessons and bookmarks are stored per account.
          </li>
          <li className="flex gap-2.5">
            <Laptop size={14} className="mt-0.5 shrink-0 text-[var(--accent)]" />
            Progress follows you across browsers and devices. Without signing in it is kept in
            this browser only.
          </li>
          <li className="flex gap-2.5">
            <ShieldCheck size={14} className="mt-0.5 shrink-0 text-[var(--accent)]" />
            Only your email, name and avatar are requested. Nothing is sent anywhere else.
          </li>
        </ul>
      </div>
    </div>
  );
}
