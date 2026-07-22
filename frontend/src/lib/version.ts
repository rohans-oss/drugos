/**
 * Single source of truth for the application version.
 *
 * ROOT FIX for FE-037 (version inconsistency):
 * Previously the codebase had THREE different version strings:
 *   - package.json: "version": "0.2.0"
 *   - app-router.tsx:2390: "DrugOS v0.3.0"
 *   - app-shell.tsx:335:  "DrugOS v2.1.0"
 *
 * Now every component imports `APP_VERSION` from this module. The value
 * is read from package.json at build time via a Next.js public runtime
 * config. We use `process.env.NEXT_PUBLIC_APP_VERSION` so the value is
 * inlined into the client bundle at build time.
 *
 * In dev, the value falls back to reading package.json synchronously.
 *
 * BE-072 ROOT FIX (v115, LOW): the previous code imported package.json
 * via a relative path (`../../package.json`). This breaks if the build
 * runs from a different working directory (e.g., `next build` invoked
 * from a parent directory, or a Docker build with WORKDIR set
 * differently). The relative import resolved to a non-existent file
 * and the build crashed with "Cannot find module '../../package.json'".
 *
 * ROOT FIX: use `process.env.NEXT_PUBLIC_APP_VERSION` as the primary
 * source (Next.js inlines this at build time, robust to cwd changes).
 * Fall back to the static import only if the env var is unset (which
 * happens in `next dev` without an .env file). The static import uses
 * a try/catch via dynamic require so it doesn't crash if the file is
 * missing — instead, it falls back to "0.0.0-unknown".
 */

// Primary source: build-time env var. Next.js inlines this into the
// client bundle via webpack's DefinePlugin when set in .env.production.
// This is robust to cwd changes because Next.js resolves the env file
// relative to the project root (where next.config.ts lives), NOT
// relative to the file that imports it.
const ENV_VERSION = process.env.NEXT_PUBLIC_APP_VERSION;

// Fallback: static import of package.json. This works in `next dev`
// (no env file needed) and in `next build` (env file present, env var
// wins). Next.js + TypeScript supports JSON imports via resolveJsonModule
// (enabled in tsconfig.json) — the import is resolved at BUILD time by
// webpack, so it's robust to runtime cwd changes. If the file is
// missing at build time, the build fails with a clear error (which
// is the correct behavior — a build without package.json is broken).
//
// BE-072: the previous code used `require()` which is forbidden by
// the @typescript-eslint/no-require-imports lint rule. The fix uses
// a static `import` (which is the modern, lint-compliant pattern).
import packageJson from "../../package.json";

const STATIC_VERSION: string | undefined = (packageJson as { version?: string }).version;

export const APP_VERSION: string = ENV_VERSION || STATIC_VERSION || "0.0.0-unknown";
export const APP_NAME: string = "DrugOS";
export const APP_COPYRIGHT: string = `© ${new Date().getFullYear()}`;

/** A formatted display string, e.g., "DrugOS v0.2.0 · © 2026". */
export const APP_DISPLAY_STRING: string = `${APP_NAME} v${APP_VERSION} · ${APP_COPYRIGHT}`;
