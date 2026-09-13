import type { MetadataRoute } from 'next';
import { getAllLessons, getPhaseLessons } from '@/lib/content';
import { PHASES } from '@/lib/curriculum';
import { SITE_URL } from '@/lib/site';

const BUILT_AT = new Date();

/**
 * Every lesson is a static page worth indexing - the whole point of making this
 * public is that someone searching "how does RAG reranking work" can land on it.
 * /progress and /signin are deliberately absent: they are per-user, not content.
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const lessons = getAllLessons().map((lesson) => ({
    url: `${SITE_URL}/learn/${lesson.phaseId}/${lesson.slug}`,
    lastModified: BUILT_AT,
    changeFrequency: 'monthly' as const,
    priority: 0.8,
  }));

  const phases = PHASES.filter((phase) => getPhaseLessons(phase.id).length > 0).map((phase) => ({
    url: `${SITE_URL}/learn/${phase.id}`,
    lastModified: BUILT_AT,
    changeFrequency: 'monthly' as const,
    priority: 0.6,
  }));

  return [
    { url: SITE_URL, lastModified: BUILT_AT, changeFrequency: 'weekly', priority: 1 },
    { url: `${SITE_URL}/roadmap`, lastModified: BUILT_AT, changeFrequency: 'monthly', priority: 0.7 },
    ...phases,
    ...lessons,
  ];
}
