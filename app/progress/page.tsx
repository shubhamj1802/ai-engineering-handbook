import type { Metadata } from 'next';
import { getNavigationTree, getCourseStats } from '@/lib/content';
import { ProgressDashboard } from '@/components/progress-dashboard';

export const metadata: Metadata = {
  title: 'My progress',
  description: 'Track completed lessons, bookmarks and remaining phases.',
};

export default function ProgressPage() {
  return <ProgressDashboard nav={getNavigationTree()} stats={getCourseStats()} />;
}
