import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: 'class',
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './lib/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          950: '#07080c',
          900: '#0b0d13',
          850: '#0f121a',
          800: '#141821',
          750: '#1a1f2b',
          700: '#232936',
          600: '#333b4d',
          500: '#4a5468',
          400: '#6b7688',
        },
        accent: {
          DEFAULT: '#5eead4',
          soft: '#2dd4bf',
          deep: '#0d9488',
        },
        violetx: '#a78bfa',
        amberx: '#fbbf24',
        rosex: '#fb7185',
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      maxWidth: {
        prose: '78ch',
      },
      keyframes: {
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        fadeUp: 'fadeUp .28s ease-out both',
      },
    },
  },
  plugins: [],
};

export default config;
