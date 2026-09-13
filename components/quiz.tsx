'use client';

import { useState } from 'react';
import clsx from 'clsx';
import { Check, HelpCircle, X } from 'lucide-react';

export interface QuizQuestion {
  question: string;
  options: string[];
  answer: number;
  explanation?: string;
}

export function Quiz({ questions }: { questions: QuizQuestion[] }) {
  const [picked, setPicked] = useState<Record<number, number>>({});

  return (
    <div className="panel overflow-hidden">
      <div className="flex items-center gap-2 border-b border-[var(--border)] bg-[var(--panel-2)] px-4 py-2.5">
        <HelpCircle size={14} className="text-[var(--accent-2)]" />
        <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[var(--accent-2)]">
          Check yourself
        </span>
        <span className="ml-auto font-mono text-[10px] text-[var(--text-dim)]">
          {Object.keys(picked).length}/{questions.length} answered
        </span>
      </div>

      <div className="divide-y divide-[var(--border)]">
        {questions.map((q, qi) => {
          const choice = picked[qi];
          const answered = choice !== undefined;
          return (
            <div key={qi} className="px-4 py-4">
              <p className="mb-3 text-[14.5px] font-medium text-[var(--text)]">
                <span className="mr-2 font-mono text-[11px] text-[var(--text-dim)]">
                  Q{qi + 1}
                </span>
                {q.question}
              </p>

              <div className="space-y-1.5">
                {q.options.map((opt, oi) => {
                  const isCorrect = oi === q.answer;
                  const isPicked = choice === oi;
                  return (
                    <button
                      key={oi}
                      type="button"
                      disabled={answered}
                      onClick={() => setPicked((p) => ({ ...p, [qi]: oi }))}
                      className={clsx(
                        'flex w-full items-start gap-2.5 rounded-lg border px-3 py-2 text-left text-[13.5px] transition',
                        !answered &&
                          'border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--border-strong)] hover:text-[var(--text)]',
                        answered && isCorrect && 'border-[var(--ok)]/50 bg-[var(--ok)]/10 text-[var(--text)]',
                        answered &&
                          isPicked &&
                          !isCorrect &&
                          'border-[var(--danger)]/50 bg-[var(--danger)]/10 text-[var(--text)]',
                        answered && !isPicked && !isCorrect && 'border-[var(--border)] text-[var(--text-dim)]',
                      )}
                    >
                      <span className="mt-0.5 shrink-0 font-mono text-[11px] text-[var(--text-dim)]">
                        {String.fromCharCode(65 + oi)}
                      </span>
                      <span className="flex-1">{opt}</span>
                      {answered && isCorrect && <Check size={14} className="mt-0.5 text-[var(--ok)]" />}
                      {answered && isPicked && !isCorrect && (
                        <X size={14} className="mt-0.5 text-[var(--danger)]" />
                      )}
                    </button>
                  );
                })}
              </div>

              {answered && q.explanation && (
                <p className="mt-3 rounded-lg border border-[var(--border)] bg-[var(--panel-2)] px-3 py-2 text-[13px] leading-relaxed text-[var(--text-muted)]">
                  <span
                    className={clsx(
                      'mr-1.5 font-semibold',
                      choice === q.answer ? 'text-[var(--ok)]' : 'text-[var(--warn)]',
                    )}
                  >
                    {choice === q.answer ? 'Correct.' : 'Not quite.'}
                  </span>
                  {q.explanation}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
