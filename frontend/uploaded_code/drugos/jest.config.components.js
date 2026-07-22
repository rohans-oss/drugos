// Jest config for React component tests (audit #295, #296, #297).
// Separate from jest.config.js (backend API tests).
// Uses jsdom env, matches tests/components/**, polyfills ResizeObserver
// and HTMLCanvasElement.getContext so Canvas-based components render.

module.exports = {
  preset: "ts-jest",
  testEnvironment: "jsdom",
  testMatch: [
    "<rootDir>/tests/components/**/*.test.tsx",
    "<rootDir>/tests/components/**/*.test.ts",
  ],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
    "\\.(css|less|scss|sass)$": "<rootDir>/tests/components/__mocks__/styleMock.js",
  },
  transform: {
    "^.+\\.tsx?$": [
      "ts-jest",
      {
        tsconfig: {
          target: "ES2022",
          module: "CommonJS",
          esModuleInterop: true,
          skipLibCheck: true,
          resolveJsonModule: true,
          jsx: "react-jsx",
          lib: ["dom", "dom.iterable", "ES2022"],
        },
      },
    ],
  },
  transformIgnorePatterns: [
    "/node_modules/(?!(next|@prisma/client|@radix-ui|lucide-react|framer-motion|recharts|@tanstack|@dnd-kit|react-syntax-highlighter))",
  ],
  setupFilesAfterEnv: ["<rootDir>/tests/components/setup.ts"],
  testTimeout: 30000,
};
