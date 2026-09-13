import type { Metadata, Viewport } from 'next';
import './globals.css';
import { getNavigationTree, getCourseStats } from '@/lib/content';
import { Providers } from '@/components/providers';
import { AppShell } from '@/components/app-shell';
import { SITE_URL, SITE_NAME, SITE_DESCRIPTION } from '@/lib/site';

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: 'AI Engineering Handbook',
    template: '%s · AI Engineering Handbook',
  },
  description:
    'Python → ML → LLMs → RAG → Agents → Agentic AI. Learn the foundations. Build the systems. Ship production-grade AI.',
  keywords: [
    'AI engineering',
    'LLM',
    'RAG',
    'AI agents',
    'agentic AI',
    'LangChain',
    'LangGraph',
    'CrewAI',
    'Python',
    'machine learning',
  ],
  applicationName: SITE_NAME,
  openGraph: {
    type: 'website',
    siteName: SITE_NAME,
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
    url: SITE_URL,
  },
  twitter: {
    card: 'summary_large_image',
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
  },
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  themeColor: '#07080c',
  width: 'device-width',
  initialScale: 1,
};

const THEME_BOOTSTRAP = `
(function () {
  try {
    var saved = localStorage.getItem('aieh.theme');
    var mode = saved || 'dark';
    document.documentElement.classList.toggle('light', mode === 'light');
    document.documentElement.style.colorScheme = mode;
  } catch (e) {}
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const nav = getNavigationTree();
  const stats = getCourseStats();

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP }} />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <Providers>
          <AppShell nav={nav} stats={stats}>
            {children}
          </AppShell>
        </Providers>
      </body>
    </html>
  );
}
