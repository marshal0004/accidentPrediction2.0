/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'pbi': {
          'bg': '#1B1B2F',
          'bg2': '#162447',
          'card': '#1F2940',
          'card-hover': '#253553',
          'sidebar': '#0F1923',
          'sidebar-active': '#2563EB',
          'border': '#2D3748',
          'text': '#FFFFFF',
          'text2': '#8899A6',
          'muted': '#5C6B7A',
          'blue': '#2563EB',
          'green': '#10B981',
          'red': '#EF4444',
          'yellow': '#F59E0B',
          'purple': '#8B5CF6',
          'orange': '#F97316',
          'cyan': '#06B6D4',
          'pink': '#EC4899',
        }
      },
      fontFamily: {
        'segoe': ['"Segoe UI"', 'Inter', '-apple-system', 'sans-serif'],
      },
      boxShadow: {
        'card': '0 4px 20px rgba(0, 0, 0, 0.3)',
        'card-hover': '0 8px 30px rgba(0, 0, 0, 0.4)',
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
