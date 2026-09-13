import fs from 'node:fs';
import path from 'node:path';
import matter from 'gray-matter';
import { PHASES, PHASE_BY_ID, GROUPS, type Difficulty, type PhaseMeta } from './curriculum';

const CONTENT_DIR = path.join(process.cwd(), 'content');

export interface LessonFrontmatter {
  title: string;
  order: number;
  difficulty: Difficulty;
  duration: number; // minutes
  badges?: string[];
  summary?: string;
  prereqs?: string[];
  keyConcepts?: string[];
  related?: string[];
}

export interface Lesson extends LessonFrontmatter {
  /** phase folder id, e.g. phase-01-python-fundamentals */
  phaseId: string;
  /** file slug, e.g. variables-and-types */
  slug: string;
  /** route path, e.g. /learn/phase-01-python-fundamentals/variables-and-types */
  href: string;
  /** stable id used for progress tracking */
  id: string;
  body: string;
  phase: PhaseMeta;
  words: number;
}

/** Trimmed lesson shape for the sidebar - never ships markdown bodies to the client. */
export interface NavLesson {
  id: string;
  title: string;
  href: string;
  duration: number;
  difficulty: Difficulty;
  badges: string[];
  order: number;
}

export interface PhaseWithLessons {
  phase: PhaseMeta;
  lessons: NavLesson[];
}

export interface GroupWithPhases {
  group: string;
  phases: PhaseWithLessons[];
}

let cache: Lesson[] | null = null;

function readLessonFile(phaseId: string, file: string): Lesson | null {
  const full = path.join(CONTENT_DIR, phaseId, file);
  const raw = fs.readFileSync(full, 'utf8');
  const { data, content } = matter(raw);
  const phase = PHASE_BY_ID.get(phaseId);
  if (!phase) return null;

  const slug = file.replace(/\.md$/, '');
  const fm = data as Partial<LessonFrontmatter>;

  return {
    title: fm.title ?? slug,
    order: typeof fm.order === 'number' ? fm.order : 999,
    difficulty: (fm.difficulty ?? phase.difficulty) as Difficulty,
    duration: typeof fm.duration === 'number' ? fm.duration : estimateMinutes(content),
    badges: fm.badges ?? [],
    summary: fm.summary ?? '',
    prereqs: fm.prereqs ?? [],
    keyConcepts: fm.keyConcepts ?? [],
    related: fm.related ?? [],
    phaseId,
    slug,
    href: `/learn/${phaseId}/${slug}`,
    id: `${phaseId}/${slug}`,
    body: content,
    phase,
    words: content.split(/\s+/).length,
  };
}

function estimateMinutes(text: string): number {
  const words = text.split(/\s+/).length;
  return Math.max(3, Math.round(words / 200));
}

/** All lessons in curriculum order. Cached per process. */
export function getAllLessons(): Lesson[] {
  if (cache && process.env.NODE_ENV === 'production') return cache;

  const lessons: Lesson[] = [];
  for (const phase of PHASES) {
    const dir = path.join(CONTENT_DIR, phase.id);
    if (!fs.existsSync(dir)) continue;
    const files = fs.readdirSync(dir).filter((f) => f.endsWith('.md'));
    const parsed = files
      .map((f) => readLessonFile(phase.id, f))
      .filter((l): l is Lesson => Boolean(l))
      .sort((a, b) => a.order - b.order || a.slug.localeCompare(b.slug));
    lessons.push(...parsed);
  }
  cache = lessons;
  return lessons;
}

export function getLesson(phaseId: string, slug: string): Lesson | undefined {
  return getAllLessons().find((l) => l.phaseId === phaseId && l.slug === slug);
}

export function getLessonById(id: string): Lesson | undefined {
  return getAllLessons().find((l) => l.id === id);
}

export function getPhaseLessons(phaseId: string): Lesson[] {
  return getAllLessons().filter((l) => l.phaseId === phaseId);
}

/** Sidebar tree: group -> phase -> lessons. Phases with no content are omitted. */
export function getNavigationTree(): GroupWithPhases[] {
  const lessons = getAllLessons();
  const toNav = (l: Lesson): NavLesson => ({
    id: l.id,
    title: l.title,
    href: l.href,
    duration: l.duration,
    difficulty: l.difficulty,
    badges: l.badges ?? [],
    order: l.order,
  });
  return GROUPS.map((group) => ({
    group,
    phases: PHASES.filter((p) => p.group === group)
      .map((phase) => ({
        phase,
        lessons: lessons.filter((l) => l.phaseId === phase.id).map(toNav),
      }))
      .filter((p) => p.lessons.length > 0),
  })).filter((g) => g.phases.length > 0);
}

export function getSiblings(lesson: Lesson): { prev?: Lesson; next?: Lesson } {
  const all = getAllLessons();
  const i = all.findIndex((l) => l.id === lesson.id);
  return { prev: all[i - 1], next: all[i + 1] };
}

export interface CourseStats {
  lessons: number;
  phases: number;
  minutes: number;
  words: number;
}

export function getCourseStats(): CourseStats {
  const all = getAllLessons();
  const phases = new Set(all.map((l) => l.phaseId));
  return {
    lessons: all.length,
    phases: phases.size,
    minutes: all.reduce((sum, l) => sum + l.duration, 0),
    words: all.reduce((sum, l) => sum + l.words, 0),
  };
}

export interface SearchDoc {
  id: string;
  title: string;
  href: string;
  phase: string;
  phaseNumber: number;
  group: string;
  difficulty: string;
  summary: string;
  headings: string[];
  text: string;
}

/** Lightweight search index, generated on the server and shipped to the client once. */
export function getSearchIndex(): SearchDoc[] {
  return getAllLessons().map((l) => {
    const headings = Array.from(l.body.matchAll(/^#{2,4}\s+(.+)$/gm)).map((m) =>
      m[1].replace(/[`*_]/g, '').trim(),
    );
    const text = l.body
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/[#>*`_|-]/g, ' ')
      .replace(/\s+/g, ' ')
      .slice(0, 2400);
    return {
      id: l.id,
      title: l.title,
      href: l.href,
      phase: l.phase.title,
      phaseNumber: l.phase.number,
      group: l.phase.group,
      difficulty: l.difficulty,
      summary: l.summary ?? '',
      headings,
      text,
    };
  });
}
