import MarkdownIt from 'markdown-it';
import hljs from 'highlight.js';

export interface TocEntry {
  id: string;
  text: string;
  level: number;
}

export interface RenderedMarkdown {
  html: string;
  toc: TocEntry[];
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`*_~]/g, '')
    .replace(/[^a-z0-9\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
    .slice(0, 80);
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * `:::` containers are rewritten to HTML before markdown parsing so the inner
 * content is still rendered as markdown by markdown-it (html: true).
 *
 *   :::note Title            -> callout
 *   :::exercise Title        -> callout
 *   :::solution Title        -> collapsed <details>
 *   :::details Title         -> collapsed <details>
 */
const CALLOUT_KINDS = new Set([
  'note',
  'tip',
  'warning',
  'danger',
  'info',
  'exercise',
  'challenge',
  'interview',
  'production',
  'security',
  'performance',
  'mistake',
]);

const COLLAPSIBLE_KINDS = new Set(['solution', 'details', 'answer']);

function preprocessContainers(src: string): string {
  // Normalise line endings first: a stray \r makes the container regexes below
  // fail to match a titled `:::note Title` line, silently dropping the callout.
  const lines = src.replace(/\r\n?/g, '\n').split('\n');
  const out: string[] = [];
  const stack: string[] = [];
  let inFence = false;

  for (const line of lines) {
    if (/^\s*(```|~~~)/.test(line)) {
      inFence = !inFence;
      out.push(line);
      continue;
    }
    if (inFence) {
      out.push(line);
      continue;
    }

    const open = line.match(/^:::\s*([a-zA-Z]+)\s*(.*)$/);
    if (open) {
      const kind = open[1].toLowerCase();
      const title = open[2].trim();
      if (CALLOUT_KINDS.has(kind)) {
        stack.push('div');
        out.push(
          `<div class="callout callout-${kind}"><p class="callout-title">${escapeHtml(
            title || kind.charAt(0).toUpperCase() + kind.slice(1),
          )}</p>`,
          '',
        );
        continue;
      }
      if (COLLAPSIBLE_KINDS.has(kind)) {
        stack.push('details');
        out.push(
          `<details class="collapsible collapsible-${kind}"><summary>${escapeHtml(
            title || (kind === 'solution' ? 'Show solution' : 'Show more'),
          )}</summary><div class="collapsible-body">`,
          '',
        );
        continue;
      }
    }

    if (/^:::\s*$/.test(line) && stack.length > 0) {
      const kind = stack.pop();
      out.push('', kind === 'details' ? '</div></details>' : '</div>', '');
      continue;
    }

    out.push(line);
  }

  while (stack.length > 0) {
    const kind = stack.pop();
    out.push(kind === 'details' ? '</div></details>' : '</div>');
  }

  return out.join('\n');
}

function createRenderer(): MarkdownIt {
  const md = new MarkdownIt({
    html: true,
    linkify: true,
    typographer: false,
    breaks: false,
  });

  // --- fenced code: mermaid, quiz and highlighted code blocks -----------------
  md.renderer.rules.fence = (tokens, idx) => {
    const token = tokens[idx];
    const info = (token.info || '').trim();
    const [langRaw, ...rest] = info.split(/\s+/);
    const lang = (langRaw || '').toLowerCase();
    const meta = rest.join(' ');
    const code = token.content;

    if (lang === 'mermaid') {
      return `<div class="mermaid-block" data-chart="${escapeHtml(code)}"><pre class="mermaid-fallback">${escapeHtml(
        code,
      )}</pre></div>`;
    }

    if (lang === 'quiz') {
      return `<div class="quiz-block" data-quiz="${escapeHtml(code)}"></div>`;
    }

    // `title=` metadata renders a filename chip above the block.
    const titleMatch = meta.match(/title="([^"]+)"|title=([^\s]+)/);
    const filename = titleMatch ? titleMatch[1] || titleMatch[2] : '';

    let highlighted: string;
    if (lang && hljs.getLanguage(lang)) {
      try {
        highlighted = hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
      } catch {
        highlighted = escapeHtml(code);
      }
    } else {
      highlighted = escapeHtml(code);
    }

    const header = filename
      ? `<div class="code-header"><span class="code-file">${escapeHtml(filename)}</span><span class="code-lang">${escapeHtml(
          lang || 'text',
        )}</span></div>`
      : `<div class="code-header code-header-plain"><span class="code-lang">${escapeHtml(
          lang || 'text',
        )}</span></div>`;

    return `<div class="code-block" data-lang="${escapeHtml(lang)}">${header}<pre class="hljs"><code class="language-${escapeHtml(
      lang,
    )}">${highlighted}</code></pre></div>`;
  };

  // --- tables scroll horizontally on small screens ---------------------------
  const defaultTableOpen =
    md.renderer.rules.table_open ||
    ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.table_open = (tokens, idx, options, env, self) =>
    `<div class="table-wrap">${defaultTableOpen(tokens, idx, options, env, self)}`;
  md.renderer.rules.table_close = () => '</table></div>';

  // --- external links open in a new tab --------------------------------------
  const defaultLinkOpen =
    md.renderer.rules.link_open ||
    ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    const href = tokens[idx].attrGet('href') || '';
    if (/^https?:\/\//.test(href)) {
      tokens[idx].attrSet('target', '_blank');
      tokens[idx].attrSet('rel', 'noopener noreferrer');
    }
    return defaultLinkOpen(tokens, idx, options, env, self);
  };

  return md;
}

const md = createRenderer();

export function renderMarkdown(source: string): RenderedMarkdown {
  const processed = preprocessContainers(source);
  const tokens = md.parse(processed, {});
  const toc: TocEntry[] = [];
  const used = new Map<string, number>();

  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i];
    if (token.type !== 'heading_open') continue;
    const level = Number(token.tag.slice(1));
    const inline = tokens[i + 1];
    const text = (inline?.content || '').replace(/[`*_]/g, '').trim();
    let id = slugify(text) || `section-${i}`;
    const seen = used.get(id) ?? 0;
    used.set(id, seen + 1);
    if (seen > 0) id = `${id}-${seen}`;
    token.attrSet('id', id);
    if (level >= 2 && level <= 3) toc.push({ id, text, level });
  }

  return { html: md.renderer.render(tokens, md.options, {}), toc };
}
