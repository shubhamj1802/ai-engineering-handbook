/**
 * The canonical public URL of this deployment.
 *
 * Resolution order matters: an explicit NEXT_PUBLIC_SITE_URL wins so that a
 * custom domain is used in sitemaps and metadata rather than the per-deployment
 * *.vercel.app hostname, which changes on every push.
 */
export const SITE_URL = (
  process.env.NEXT_PUBLIC_SITE_URL ||
  (process.env.VERCEL_PROJECT_PRODUCTION_URL
    ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
    : '') ||
  (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : '') ||
  'http://localhost:3000'
).replace(/\/$/, '');

export const SITE_NAME = 'AI Engineering Handbook';

export const SITE_DESCRIPTION =
  'A free, open handbook: Python → Data Science → ML → Deep Learning → LLMs → RAG → Agents → Agentic AI → LangChain, LangGraph and CrewAI → production systems. 82 hands-on lessons with runnable code.';
