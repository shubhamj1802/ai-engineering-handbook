import type { MetadataRoute } from 'next';
import { SITE_URL } from '@/lib/site';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: '*',
        allow: '/',
        // Per-user routes: nothing to index, and /api/auth must never be crawled.
        disallow: ['/api/', '/progress', '/signin'],
      },
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
