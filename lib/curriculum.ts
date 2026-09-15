/**
 * The curriculum skeleton: groups -> phases.
 * Lesson pages live as markdown files in /content/<phaseId>/<slug>.md and are
 * discovered at build time by lib/content.ts.
 */

export type Difficulty =
  | 'Beginner'
  | 'Intermediate'
  | 'Advanced'
  | 'Expert'
  | 'Production'
  | 'Architect';

export interface PhaseMeta {
  id: string;
  number: number;
  title: string;
  group: string;
  blurb: string;
  difficulty: Difficulty;
}

export const GROUPS = [
  'GETTING STARTED',
  'PYTHON',
  'DATA',
  'MACHINE LEARNING',
  'AI & DEEP LEARNING',
  'LLM ENGINEERING',
  'RAG',
  'AGENTS',
  'FRAMEWORKS',
  'PRODUCTION',
  'PROJECTS',
  'REFERENCE',
] as const;

export type Group = (typeof GROUPS)[number];

export const PHASES: PhaseMeta[] = [
  {
    id: 'phase-00-roadmap',
    number: 0,
    title: 'Roadmap & Environment',
    group: 'GETTING STARTED',
    blurb:
      'What AI engineering actually is, how the pieces fit together, and a development environment you will not have to redo.',
    difficulty: 'Beginner',
  },
  {
    id: 'phase-01-python-fundamentals',
    number: 1,
    title: 'Python Fundamentals',
    group: 'PYTHON',
    blurb:
      'From zero: values, types, collections, control flow, functions, errors, files, modules.',
    difficulty: 'Beginner',
  },
  {
    id: 'phase-02-advanced-python',
    number: 2,
    title: 'Advanced Python',
    group: 'PYTHON',
    blurb:
      'OOP, dataclasses, decorators, generators, context managers, typing, async, testing, packaging.',
    difficulty: 'Intermediate',
  },
  {
    id: 'phase-03-numpy',
    number: 3,
    title: 'NumPy',
    group: 'DATA',
    blurb:
      'Arrays, shapes, broadcasting and vectorisation: the mental model behind every tensor library.',
    difficulty: 'Intermediate',
  },
  {
    id: 'phase-04-pandas',
    number: 4,
    title: 'Pandas',
    group: 'DATA',
    blurb: 'Load, clean, reshape, group and join real datasets without writing loops.',
    difficulty: 'Intermediate',
  },
  {
    id: 'phase-05-visualization',
    number: 5,
    title: 'Matplotlib & Visualization',
    group: 'DATA',
    blurb: 'Charts that answer a question: Matplotlib mechanics plus Seaborn and Plotly.',
    difficulty: 'Intermediate',
  },
  {
    id: 'phase-06-ml-fundamentals',
    number: 6,
    title: 'ML Fundamentals',
    group: 'MACHINE LEARNING',
    blurb:
      'Learning from data from first principles, the core algorithms, and honest evaluation.',
    difficulty: 'Intermediate',
  },
  {
    id: 'phase-07-practical-ml',
    number: 7,
    title: 'Practical Machine Learning',
    group: 'MACHINE LEARNING',
    blurb:
      'Two end-to-end projects: data, to tuned model, to saved artifact, to inference API.',
    difficulty: 'Advanced',
  },
  {
    id: 'phase-08-ai-fundamentals',
    number: 8,
    title: 'AI Fundamentals',
    group: 'AI & DEEP LEARNING',
    blurb:
      'How we got from symbolic AI to foundation models, and the vocabulary of generative AI.',
    difficulty: 'Intermediate',
  },
  {
    id: 'phase-09-deep-learning',
    number: 9,
    title: 'Deep Learning',
    group: 'AI & DEEP LEARNING',
    blurb:
      'Neurons, backprop, optimisers and a real PyTorch training loop: as deep as an AI engineer needs.',
    difficulty: 'Advanced',
  },
  {
    id: 'phase-10-llm-fundamentals',
    number: 10,
    title: 'LLM Fundamentals',
    group: 'LLM ENGINEERING',
    blurb:
      'Transformers, attention, tokenisation, sampling, and production-grade LLM API calls.',
    difficulty: 'Advanced',
  },
  {
    id: 'phase-11-embeddings-vector-db',
    number: 11,
    title: 'Embeddings & Vector DBs',
    group: 'LLM ENGINEERING',
    blurb:
      'Semantic similarity, chunking, and choosing between FAISS, Chroma, Qdrant, Pinecone and pgvector.',
    difficulty: 'Advanced',
  },
  {
    id: 'phase-12-rag',
    number: 12,
    title: 'RAG',
    group: 'RAG',
    blurb:
      'Retrieval-augmented generation built by hand first, then with a framework, with citations.',
    difficulty: 'Advanced',
  },
  {
    id: 'phase-13-advanced-rag',
    number: 13,
    title: 'Advanced RAG',
    group: 'RAG',
    blurb:
      'Query rewriting, HyDE, hybrid search, reranking, parent-child retrieval, routing and evaluation.',
    difficulty: 'Expert',
  },
  {
    id: 'phase-14-ai-agents',
    number: 14,
    title: 'AI Agents',
    group: 'AGENTS',
    blurb:
      'What an agent really is. Build a tool-calling agent from scratch before touching a framework.',
    difficulty: 'Expert',
  },
  {
    id: 'phase-15-agentic-ai',
    number: 15,
    title: 'Agentic AI Patterns',
    group: 'AGENTS',
    blurb:
      'ReAct, planner/executor, reflection, routing, supervisors, and when not to use an agent at all.',
    difficulty: 'Expert',
  },
  {
    id: 'phase-16-langchain',
    number: 16,
    title: 'LangChain',
    group: 'FRAMEWORKS',
    blurb:
      'Models, messages, structured output, tools, LCEL runnables, retrievers and streaming.',
    difficulty: 'Expert',
  },
  {
    id: 'phase-17-langgraph',
    number: 17,
    title: 'LangGraph',
    group: 'FRAMEWORKS',
    blurb:
      'State machines for LLM applications: nodes, edges, reducers, checkpoints, interrupts, subgraphs.',
    difficulty: 'Expert',
  },
  {
    id: 'phase-18-crewai',
    number: 18,
    title: 'CrewAI',
    group: 'FRAMEWORKS',
    blurb:
      'Role-based multi-agent crews, tasks and processes, and how they compare to LangGraph.',
    difficulty: 'Expert',
  },
  {
    id: 'phase-19-guardrails',
    number: 19,
    title: 'Guardrails & AI Safety',
    group: 'PRODUCTION',
    blurb:
      'Validate input and output, contain prompt injection, sandbox tools, cap budgets and loops.',
    difficulty: 'Production',
  },
  {
    id: 'phase-20-human-in-the-loop',
    number: 20,
    title: 'Human-in-the-Loop',
    group: 'PRODUCTION',
    blurb: 'Approval gates, escalation, confidence thresholds, interrupt and resume.',
    difficulty: 'Production',
  },
  {
    id: 'phase-21-memory-state',
    number: 21,
    title: 'Memory & State',
    group: 'PRODUCTION',
    blurb:
      'Short-term vs long-term memory, checkpointing, user profiles, and what must never be stored.',
    difficulty: 'Production',
  },
  {
    id: 'phase-22-tool-use',
    number: 22,
    title: 'Tool Use & Function Calling',
    group: 'PRODUCTION',
    blurb: 'Tool schemas, validation, permissions, retries, timeouts and tool observability.',
    difficulty: 'Production',
  },
  {
    id: 'phase-23-multi-agent',
    number: 23,
    title: 'Multi-Agent Systems',
    group: 'PRODUCTION',
    blurb: 'Supervisor, router, hierarchical and peer topologies, with their real failure modes.',
    difficulty: 'Architect',
  },
  {
    id: 'phase-24-observability-eval',
    number: 24,
    title: 'Observability & Evaluation',
    group: 'PRODUCTION',
    blurb:
      'Tracing, token/cost/latency metrics, RAGAS, DeepEval, LLM-as-judge and regression suites.',
    difficulty: 'Production',
  },
  {
    id: 'phase-25-production',
    number: 25,
    title: 'Production AI Systems',
    group: 'PRODUCTION',
    blurb: 'FastAPI, Docker, queues, caching, auth, rate limits, fallbacks, cost control and CI/CD.',
    difficulty: 'Architect',
  },
  {
    id: 'phase-26-capstones',
    number: 26,
    title: 'Capstone Projects',
    group: 'PROJECTS',
    blurb:
      'Five substantial builds, from a document-intelligence RAG service to a full agent platform.',
    difficulty: 'Architect',
  },
  {
    id: 'phase-28-multi-agent-labs',
    number: 27,
    title: 'Multi-Agent Labs',
    group: 'PROJECTS',
    blurb:
      'Six multi-agent systems you will not find in tutorials: adversarial hardening, consensus extraction, synthetic user populations, self-healing pipelines, debate, and task markets.',
    difficulty: 'Architect',
  },
  {
    id: 'phase-27-reference',
    number: 28,
    title: 'Reference & Maps',
    group: 'REFERENCE',
    blurb: 'Roadmaps, dependency maps, cheat sheets, interview prep and the production checklist.',
    difficulty: 'Production',
  },
];

export const PHASE_BY_ID = new Map(PHASES.map((p) => [p.id, p]));

export const DIFFICULTY_STYLES: Record<Difficulty, string> = {
  Beginner: 'text-emerald-300 border-emerald-400/30 bg-emerald-400/10',
  Intermediate: 'text-sky-300 border-sky-400/30 bg-sky-400/10',
  Advanced: 'text-violet-300 border-violet-400/30 bg-violet-400/10',
  Expert: 'text-amber-300 border-amber-400/30 bg-amber-400/10',
  Production: 'text-rose-300 border-rose-400/30 bg-rose-400/10',
  Architect: 'text-fuchsia-300 border-fuchsia-400/30 bg-fuchsia-400/10',
};

export const BADGE_STYLES: Record<string, string> = {
  'Start here': 'text-emerald-300 border-emerald-400/30 bg-emerald-400/10',
  'Hands-on': 'text-sky-300 border-sky-400/30 bg-sky-400/10',
  'Read once, refer often': 'text-violet-300 border-violet-400/30 bg-violet-400/10',
  Project: 'text-fuchsia-300 border-fuchsia-400/30 bg-fuchsia-400/10',
  Reference: 'text-slate-300 border-slate-400/30 bg-slate-400/10',
  'Deep dive': 'text-amber-300 border-amber-400/30 bg-amber-400/10',
  Theory: 'text-cyan-300 border-cyan-400/30 bg-cyan-400/10',
  Security: 'text-rose-300 border-rose-400/30 bg-rose-400/10',
};
