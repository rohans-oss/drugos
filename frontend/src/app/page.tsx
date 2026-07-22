'use client'

import DrugOSApp from '@/components/drugos/app-router'
import React, { Suspense } from 'react'

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean, error: Error | null }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center p-12 text-center h-full">
          <h2 className="text-xl font-bold text-red-600 mb-2">Something went wrong.</h2>
          <pre className="text-xs bg-slate-100 p-4 rounded text-left overflow-auto max-w-full">
            {this.state.error?.message}
          </pre>
          <button 
            className="mt-4 px-4 py-2 bg-slate-800 text-white rounded hover:bg-slate-700"
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function Home() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<div className="p-8">Loading application...</div>}>
        <DrugOSApp />
      </Suspense>
    </ErrorBoundary>
  );
}
