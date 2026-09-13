/**
 * Content linter for the handbook.
 *
 * Catches the mistakes that silently degrade the site instead of crashing it:
 *   - YAML frontmatter that fails to parse (an unquoted colon is the usual cause)
 *   - missing required fields
 *   - duplicate `order` values inside a phase
 *   - lesson folders that are not declared in lib/curriculum.ts
 *   - malformed ```quiz blocks
 *
 * Run: npm run check:content
 */
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const CONTENT = 'content';
const REQUIRED = ['title', 'order', 'difficulty', 'duration', 'summary'];

const curriculum = readFileSync('lib/curriculum.ts', 'utf8');
const declaredPhases = new Set([...curriculum.matchAll(/id: '([^']+)'/g)].map((m) => m[1]));

let errors = 0;
let warnings = 0;
let lessons = 0;

const fail = (file, message) => {
  errors += 1;
  console.error(`  ERROR  ${file}: ${message}`);
};
const warn = (file, message) => {
  warnings += 1;
  console.warn(`  WARN   ${file}: ${message}`);
};

/** Minimal frontmatter parser - only needs to agree with gray-matter on validity. */
function parseFrontmatter(raw, file) {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
  if (!match) {
    fail(file, 'missing frontmatter block');
    return null;
  }
  const fields = {};
  for (const line of match[1].split(/\r?\n/)) {
    if (!line.trim() || line.startsWith('#') || /^\s/.test(line)) continue;
    const kv = line.match(/^(\w+):\s*(.*)$/);
    if (!kv) {
      fail(file, `unparseable frontmatter line: ${line.slice(0, 60)}`);
      continue;
    }
    const [, key, valueRaw] = kv;
    const value = valueRaw.trim();
    if (value && !/^[["']/.test(value) && (value.includes(': ') || value.endsWith(':'))) {
      fail(
        file,
        `field "${key}" contains an unquoted colon - YAML will drop the whole frontmatter. Wrap the value in double quotes.`,
      );
    }
    fields[key] = value;
  }
  return fields;
}

for (const phaseDir of readdirSync(CONTENT)) {
  const dir = join(CONTENT, phaseDir);
  if (!existsSync(dir)) continue;
  const files = readdirSync(dir).filter((f) => f.endsWith('.md'));
  if (files.length === 0) continue;

  if (!declaredPhases.has(phaseDir)) {
    fail(dir, 'phase folder is not declared in lib/curriculum.ts - its lessons will not appear');
  }

  const orders = new Map();

  for (const file of files) {
    lessons += 1;
    const path = join(dir, file);
    const raw = readFileSync(path, 'utf8');
    const fields = parseFrontmatter(raw, path);
    if (!fields) continue;

    for (const key of REQUIRED) {
      if (!fields[key]) fail(path, `missing required frontmatter field: ${key}`);
    }

    if (fields.order) {
      const seen = orders.get(fields.order);
      if (seen) warn(path, `duplicate order ${fields.order} (also ${seen})`);
      else orders.set(fields.order, file);
    }

    for (const block of raw.matchAll(/```quiz\r?\n([\s\S]*?)```/g)) {
      try {
        const parsed = JSON.parse(block[1]);
        const questions = Array.isArray(parsed) ? parsed : [parsed];
        for (const q of questions) {
          if (typeof q.question !== 'string' || !Array.isArray(q.options)) {
            fail(path, 'quiz question missing `question` or `options`');
          } else if (typeof q.answer !== 'number' || q.answer < 0 || q.answer >= q.options.length) {
            fail(path, `quiz answer index out of range: ${q.answer}`);
          }
        }
      } catch (err) {
        fail(path, `invalid JSON in quiz block: ${err.message}`);
      }
    }

    const openFences = (raw.match(/^```/gm) || []).length;
    if (openFences % 2 !== 0) fail(path, 'unbalanced ``` code fences');

    // ::: containers must be balanced, ignoring anything inside code fences
    const KINDS = new Set([
      'note', 'tip', 'warning', 'danger', 'info', 'exercise', 'challenge', 'interview',
      'production', 'security', 'performance', 'mistake', 'solution', 'details', 'answer',
    ]);
    let depth = 0;
    let inFence = false;
    raw.split(String.fromCharCode(10)).forEach((rawLine, i) => {
      const line = rawLine.split(String.fromCharCode(13))[0];
      if (/^\s*(```|~~~)/.test(line)) {
        inFence = !inFence;
        return;
      }
      if (inFence) return;
      const match = line.match(/^:::\s*([a-zA-Z]*)\s*(.*)$/);
      if (!match) return;
      const kind = match[1].toLowerCase();
      if (kind === '') {
        depth -= 1;
        if (depth < 0) {
          fail(path, `stray closing ::: at line ${i + 1}`);
          depth = 0;
        }
      } else if (KINDS.has(kind)) {
        depth += 1;
      } else {
        warn(path, `unknown container kind ":::${kind}" at line ${i + 1}`);
      }
    });
    if (depth !== 0) fail(path, `${depth} unclosed ::: container(s)`);
  }
}

console.log(`\nchecked ${lessons} lessons: ${errors} errors, ${warnings} warnings`);
process.exit(errors > 0 ? 1 : 0);
