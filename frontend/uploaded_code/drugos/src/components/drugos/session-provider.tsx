'use client'

/**
 * Session provider for DruGOS.
 *
 * - Tracks the current authenticated user (or null when logged out).
 * - Exposes `loading` so the app shell can show a splash while /api/auth/me
 *   resolves on first mount.
 * - Exposes `signIn`, `signOut`, `refresh` helpers used by the auth screens.
 * - Listens for `drugos:unauthorized` events dispatched by the API client
 *   when any request returns 401, and forces a re-fetch of /me.
 *
 * IMPORTANT — Hydration + auth-guard safety:
 * We use a `mounted` flag that starts `false` on both server and first client
 * render (so they agree → no hydration mismatch), then flips to `true` inside
 * a `useEffect`. The `loading` flag is `true` whenever `mounted === false`
 * OR a `/api/auth/me` fetch is in flight. This means the AppShell auth guard
 * sees `loading: true` immediately after mount and waits for the session
 * fetch to resolve before deciding whether to redirect to /login.
 */

import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react'
import { api, type AuthMeResponse } from '@/lib/api-client'

interface SessionState {
  user: AuthMeResponse['user'] | null
  organizations: AuthMeResponse['organizations']
  activeOrganizationId: string | null
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
  signOut: () => Promise<void>
}

const SessionContext = createContext<SessionState>({
  user: null,
  organizations: [],
  activeOrganizationId: null,
  loading: true,
  error: null,
  refresh: async () => {},
  signOut: async () => {},
})

export function useSession() {
  return useContext(SessionContext)
}

export function SessionProvider({ children }: { children: React.ReactNode }) {
  // `mounted` is false on the server and during the first client render,
  // then becomes true inside a useEffect. While `mounted === false`, we
  // report `loading: true` so the AppShell auth guard doesn't prematurely
  // redirect to /login before we've had a chance to check the auth cookie.
  const [mounted, setMounted] = useState(false)
  const [user, setUser] = useState<AuthMeResponse['user'] | null>(null)
  const [organizations, setOrganizations] = useState<AuthMeResponse['organizations']>([])
  const [activeOrganizationId, setActiveOrganizationId] = useState<string | null>(null)
  const [fetching, setFetching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inFlight = useRef(false)

  const refresh = useCallback(async () => {
    if (inFlight.current) return
    inFlight.current = true
    setFetching(true)
    try {
      const data = await api.me()
      setUser(data.user)
      setOrganizations(data.organizations || [])
      setActiveOrganizationId(data.activeOrganizationId)
      setError(null)
    } catch (e: any) {
      // 401 means not logged in — that's not an error, just no session.
      if (e?.status !== 401) {
        setError(e?.message || 'Failed to load session')
      }
      setUser(null)
      setOrganizations([])
      setActiveOrganizationId(null)
    } finally {
      setFetching(false)
      inFlight.current = false
    }
  }, [])

  const signOut = useCallback(async () => {
    try {
      await api.logout()
    } catch {
      // Best-effort
    }
    setUser(null)
    setOrganizations([])
    setActiveOrganizationId(null)
  }, [])

  // Mount effect: flip `mounted` to true and kick off the first /me fetch.
  useEffect(() => {
    setMounted(true)
    refresh()
  }, [refresh])

  // Listen for 401 events from the API client — refresh session state.
  useEffect(() => {
    const handler = () => {
      setUser(null)
      setOrganizations([])
      setActiveOrganizationId(null)
    }
    window.addEventListener('drugos:unauthorized', handler)
    return () => window.removeEventListener('drugos:unauthorized', handler)
  }, [])

  // `loading` is true if we haven't mounted yet OR a fetch is in flight.
  // This is the value consumers use to decide whether to show a splash or
  // redirect to /login.
  const loading = !mounted || fetching

  return (
    <SessionContext.Provider
      value={{ user, organizations, activeOrganizationId, loading, error, refresh, signOut }}
    >
      {children}
    </SessionContext.Provider>
  )
}
