/**
 * Jest setup file for React component tests.
 *
 * Polyfills the browser APIs that the components use but jsdom doesn't
 * implement:
 *   - ResizeObserver (used by KnowledgeGraphExplorer to track container size)
 *   - matchMedia (used by some shadcn/ui primitives)
 *   - HTMLCanvasElement.getContext (returns a stub 2D context — we don't
 *     assert on pixel output, only on DOM state)
 *
 * Also registers the jest-dom matchers (`toBeInTheDocument`, etc.).
 */
import '@testing-library/jest-dom';

// ─── ResizeObserver polyfill ─────────────────────────────────────────
class ResizeObserverStub {
  private callback: ResizeObserverCallback;
  constructor(cb: ResizeObserverCallback) {
    this.callback = cb;
  }
  observe(el: Element) {
    // Fire once immediately with the element's current size so the
    // component under test receives a non-zero size on mount.
    const rect = (el as HTMLElement).getBoundingClientRect?.() ?? {
      width: 800,
      height: 600,
      top: 0,
      left: 0,
      right: 800,
      bottom: 600,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    };
    const entry = {
      target: el,
      contentRect: {
        width: rect.width || 800,
        height: rect.height || 600,
        top: 0,
        left: 0,
        right: rect.width || 800,
        bottom: rect.height || 600,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      },
      borderBoxSize: [{ inlineSize: rect.width || 800, blockSize: rect.height || 600 }],
      contentBoxSize: [{ inlineSize: rect.width || 800, blockSize: rect.height || 600 }],
      devicePixelContentBoxSize: [{ inlineSize: rect.width || 800, blockSize: rect.height || 600 }],
    };
    this.callback([entry as unknown as ResizeObserverEntry], this);
  }
  unobserve() {}
  disconnect() {}
}
(globalThis as any).ResizeObserver = ResizeObserverStub;

// ─── matchMedia polyfill ─────────────────────────────────────────────
if (!window.matchMedia) {
  (window as any).matchMedia = (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  });
}

// ─── HTMLCanvasElement.getContext stub ───────────────────────────────
// The real jsdom throws "Not implemented: HTMLCanvasElement.prototype.getContext"
// because it doesn't have a rendering engine. We return a stub that
// implements just enough of the Canvas2D API for our components to
// render without throwing.
const stubCtx = new Proxy(
  {},
  {
    get(_target, prop) {
      // Any property access returns a no-op function or a sensible default.
      if (prop === 'canvas') {
        return { width: 800, height: 600 };
      }
      if (prop === 'measureText') {
        return (text: string) => ({ width: (text || '').length * 6, height: 12 });
      }
      return () => {};
    },
    set() {
      return true;
    },
  },
);

HTMLCanvasElement.prototype.getContext = function getContext(
  this: HTMLCanvasElement,
  _type: string,
) {
  return stubCtx as unknown as CanvasRenderingContext2D;
} as HTMLCanvasElement['getContext'];

// ─── Suppress React 19 act() warnings in test output ─────────────────
const originalError = console.error;
beforeEach(() => {
  // jest-dom is already loaded; nothing else to do.
});

afterEach(() => {
  jest.restoreAllMocks();
});
