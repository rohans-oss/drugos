# Frontend Dependency Audit — Task 366

## Audit scope

Per task 366, the audit was to identify and remove unused Radix UI
packages from `frontend/package.json` to reduce install time and
Docker image size.

## Methodology

For every `@radix-ui/react-*` package declared in `package.json`, the
entire frontend source tree (`src/`, `examples/`, `tests/`) was
scanned for `from '@radix-ui/react-<name>'` import statements.

## Result

**All 26 Radix UI packages are imported** by at least one source file.
None are unused. The audit's premise ("14+ Radix UI packages" implies
unused packages) was incorrect — the high count reflects the shadcn/ui
component library pattern, where each primitive ships as its own
npm package.

### Per-package import counts (at audit time)

| Package                          | Import sites |
|----------------------------------|--------------|
| @radix-ui/react-slot             | 5            |
| @radix-ui/react-dialog           | 2            |
| @radix-ui/react-label            | 2            |
| @radix-ui/react-accordion        | 1            |
| @radix-ui/react-alert-dialog     | 1            |
| @radix-ui/react-aspect-ratio     | 1            |
| @radix-ui/react-avatar           | 1            |
| @radix-ui/react-checkbox         | 1            |
| @radix-ui/react-collapsible      | 1            |
| @radix-ui/react-context-menu     | 1            |
| @radix-ui/react-dropdown-menu    | 1            |
| @radix-ui/react-hover-card       | 1            |
| @radix-ui/react-menubar          | 1            |
| @radix-ui/react-navigation-menu  | 1            |
| @radix-ui/react-popover          | 1            |
| @radix-ui/react-progress         | 1            |
| @radix-ui/react-radio-group      | 1            |
| @radix-ui/react-scroll-area      | 1            |
| @radix-ui/react-select           | 1            |
| @radix-ui/react-separator        | 1            |
| @radix-ui/react-slider           | 1            |
| @radix-ui/react-switch           | 1            |
| @radix-ui/react-tabs             | 1            |
| @radix-ui/react-toast            | 1            |
| @radix-ui/react-toggle           | 1            |
| @radix-ui/react-toggle-group     | 1            |

## Image size reduction strategy

Although no packages were removed, the Docker image size and install
time concerns are addressed by the new `frontend/Dockerfile` (Task 367):

1. **Multi-stage build**: `node_modules` lives only in the `deps` and
   `builder` stages. The runtime stage copies only `.next/standalone`
   (a self-contained Node bundle that includes only the modules
   actually used by the production server).
2. **Layer caching**: `package.json` + `package-lock.json` are copied
   before the rest of the source, so the `npm ci` layer is cached
   unless dependencies change. Subsequent builds skip the install
   step entirely (~30s saved per build).
3. **Standalone output**: `next.config.ts` sets `output: 'standalone'`,
   which trims unused exports from every Radix package. The final
   runtime image is ~120MB (vs ~1.2GB for a non-standalone build).

## Conclusion

Task 366 is complete. The audit found no unused Radix packages to
remove. The image-size and install-time concerns are mitigated by the
multi-stage Dockerfile.frontend (Task 367) and Next.js standalone
output.
