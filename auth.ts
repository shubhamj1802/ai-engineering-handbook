import NextAuth from 'next-auth';
import Google from 'next-auth/providers/google';

/**
 * Gmail / Google sign-in.
 *
 * Credentials come from environment variables only - never commit them.
 * Auth.js reads AUTH_GOOGLE_ID / AUTH_GOOGLE_SECRET by convention; the more
 * familiar GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET names are accepted too.
 *
 *   .env.local
 *     AUTH_SECRET=<openssl rand -base64 32>
 *     GOOGLE_CLIENT_ID=...apps.googleusercontent.com
 *     GOOGLE_CLIENT_SECRET=...
 *
 * Authorised redirect URI in Google Cloud Console:
 *     http://localhost:3000/api/auth/callback/google
 */
const googleId = process.env.AUTH_GOOGLE_ID ?? process.env.GOOGLE_CLIENT_ID;
const googleSecret = process.env.AUTH_GOOGLE_SECRET ?? process.env.GOOGLE_CLIENT_SECRET;

export const googleConfigured = Boolean(googleId && googleSecret);

export const { handlers, auth, signIn, signOut } = NextAuth({
  // Without credentials the provider list stays empty: the app still runs in
  // local-only mode instead of crashing on every auth route.
  providers: googleConfigured
    ? [
        Google({
          clientId: googleId,
          clientSecret: googleSecret,
          authorization: {
            params: { prompt: 'select_account', access_type: 'offline', scope: 'openid email profile' },
          },
        }),
      ]
    : [],
  session: { strategy: 'jwt', maxAge: 60 * 60 * 24 * 30 },
  trustHost: true,
  pages: { signIn: '/signin', error: '/signin' },
  callbacks: {
    async jwt({ token, profile }) {
      if (profile?.email) token.email = profile.email;
      if (profile?.picture && typeof profile.picture === 'string') token.picture = profile.picture;
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = (token.sub as string) ?? session.user.email ?? 'anonymous';
      }
      return session;
    },
  },
});
