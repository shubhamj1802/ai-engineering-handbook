'use client';

import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Quiz, type QuizQuestion } from './quiz';

interface QuizMount {
  el: HTMLElement;
  questions: QuizQuestion[];
}

/**
 * Renders server-generated HTML and then progressively enhances it:
 *   - copy-to-clipboard buttons on every code block
 *   - Mermaid diagrams (lazy-loaded, re-themed when the theme toggles)
 *   - interactive quizzes mounted into ```quiz placeholders via portals
 */
export function ArticleBody({ html }: { html: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [quizzes, setQuizzes] = useState<QuizMount[]>([]);

  // --- copy buttons ---------------------------------------------------------
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const blocks = Array.from(root.querySelectorAll<HTMLElement>('.code-block'));
    const cleanups: Array<() => void> = [];

    for (const block of blocks) {
      if (block.querySelector('.code-copy')) continue;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'code-copy';
      button.textContent = 'copy';
      const onClick = async () => {
        const code = block.querySelector('code')?.textContent ?? '';
        try {
          await navigator.clipboard.writeText(code);
          button.textContent = 'copied';
          button.dataset.copied = 'true';
        } catch {
          button.textContent = 'press ⌘C';
        }
        setTimeout(() => {
          button.textContent = 'copy';
          delete button.dataset.copied;
        }, 1600);
      };
      button.addEventListener('click', onClick);
      block.appendChild(button);
      cleanups.push(() => {
        button.removeEventListener('click', onClick);
        button.remove();
      });
    }

    return () => cleanups.forEach((fn) => fn());
  }, [html]);

  // --- mermaid --------------------------------------------------------------
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const blocks = Array.from(root.querySelectorAll<HTMLElement>('.mermaid-block'));
    if (blocks.length === 0) return;

    let disposed = false;

    const render = async () => {
      const mermaid = (await import('mermaid')).default;
      if (disposed) return;
      const light = document.documentElement.classList.contains('light');
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: 'base',
        fontFamily: 'JetBrains Mono, ui-monospace, monospace',
        themeVariables: light
          ? {
              background: '#ffffff',
              primaryColor: '#e8fbf6',
              primaryTextColor: '#11141d',
              primaryBorderColor: '#0d9488',
              lineColor: '#64748b',
              secondaryColor: '#ede9fe',
              tertiaryColor: '#f1f5f9',
            }
          : {
              background: '#10131c',
              primaryColor: '#152029',
              primaryTextColor: '#e8ebf2',
              primaryBorderColor: '#5eead4',
              lineColor: '#64748b',
              secondaryColor: '#1b1a2e',
              tertiaryColor: '#151926',
              clusterBkg: '#0f1320',
              clusterBorder: '#232936',
            },
      });

      for (let i = 0; i < blocks.length; i += 1) {
        const block = blocks[i];
        const chart = block.dataset.chart;
        if (!chart) continue;
        try {
          const { svg } = await mermaid.render(`mmd-${Date.now()}-${i}`, chart);
          if (disposed) return;
          block.innerHTML = svg;
          block.dataset.rendered = 'true';
        } catch {
          // Leave the <pre> fallback visible: a broken diagram must not hide content.
          block.dataset.rendered = 'false';
        }
      }
    };

    render();

    // Re-render when the theme class flips so diagrams match the palette.
    const observer = new MutationObserver(() => {
      blocks.forEach((b) => {
        const chart = b.dataset.chart;
        if (chart) b.innerHTML = `<pre class="mermaid-fallback">${chart}</pre>`;
      });
      render();
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

    return () => {
      disposed = true;
      observer.disconnect();
    };
  }, [html]);

  // --- quizzes --------------------------------------------------------------
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const mounts: QuizMount[] = [];
    root.querySelectorAll<HTMLElement>('.quiz-block').forEach((el) => {
      const raw = el.dataset.quiz;
      if (!raw) return;
      try {
        const parsed = JSON.parse(raw) as QuizQuestion | QuizQuestion[];
        mounts.push({ el, questions: Array.isArray(parsed) ? parsed : [parsed] });
      } catch {
        el.innerHTML =
          '<p class="text-[13px] text-[var(--danger)]">Quiz block contains invalid JSON.</p>';
      }
    });
    setQuizzes(mounts);
  }, [html]);

  return (
    <>
      <div ref={ref} className="prose-doc" dangerouslySetInnerHTML={{ __html: html }} />
      {quizzes.map((q, i) => createPortal(<Quiz questions={q.questions} />, q.el, `quiz-${i}`))}
    </>
  );
}
