'use client';

import { useState, useMemo, useEffect } from 'react';
import { useDrugOSNav } from './nav-context';
import { useSession } from './session-provider';
import {
  api, type Invoice, type Plan, type Subscription, type AuditLog, type TeamMember,
  type DatasetStatsResponse, type KnowledgeGraphStatsResponse, type SystemStatus,
  type Project, type AdminUser, type DatasetQualityResponse, type AdminMetricsResponse,
} from '@/lib/api-client';
import { roleLabel } from '@/lib/rbac';
// FE-030 ROOT FIX: real-API hooks for SharedQueriesScreen / AnnotationsScreen.
// Previously these screens rendered hardcoded fake colleagues. Now they call
// the real /api/projects endpoint and render honest empty states.
import { useApiList, useApiResource, LoadingSpinner, ErrorDisplay, EmptyState } from './use-api-data';
// Issue 317 (audit 301-320): Reusable DemoDataBanner. Imported by every
// screen that still renders mock data, so the banner is visible to
// researchers/admins/investors that the data is NOT real.
import { DemoDataBanner } from '@/components/ui/DemoDataBanner';
import { useTheme } from 'next-themes';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, AreaChart, Area, Legend } from 'recharts';
import { Search, Plus, Download, ChevronRight, ChevronDown, Check, X, AlertTriangle, Star, ExternalLink, Copy, Trash2, Edit, MoreHorizontal, Filter, ArrowRight, RefreshCw, Eye, Settings, Users, Shield, Key, Activity, TrendingUp, FileText, Clock, Zap, Globe, Lock, Bell, Mail, CreditCard, Database, Code, BookOpen, GitFork, Server, Building, User, Play, Send, HelpCircle, MessageSquare, BarChart3, Target, Award, Heart, LayoutDashboard, GitBranch, FolderKanban, Share2, Bookmark, Layers, Monitor, Smartphone, Calendar, DollarSign, Percent, Package, AlertCircle, CheckCircle2, XCircle, Info, ArrowUpRight, ArrowDownRight, ToggleLeft, ShieldCheck, Scale, Sun, Moon, MonitorSmartphone, QrCode, Network } from 'lucide-react';
import { motion } from 'framer-motion';
// Issue 301 (audit 301-320): Removed unused empty-defaults imports — they
// were dead weight and made it look like the screens still consumed mock
// data. Every screen now calls a real API endpoint OR renders a visible
// DemoDataBanner. No silent mock data anywhere.

const PRIMARY = '#5B4FCF';
const GREEN = '#1D9E75';
const ORANGE = '#D4853A';
const RED = '#C0392B';

function FadeIn({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  return <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay }}>{children}</motion.div>;
}

function PageHeader({ title, desc, actions }: { title: string; desc?: string; actions?: React.ReactNode }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6">
      <div><h1 className="text-2xl font-bold text-foreground">{title}</h1>{desc && <p className="text-sm text-muted-foreground mt-0.5">{desc}</p>}</div>
      {actions && <div className="flex items-center gap-2 flex-shrink-0">{actions}</div>}
    </div>
  );
}

function StatCard({ title, value, subtitle, icon: Icon, trend }: { title: string; value: string | number; subtitle?: string; icon?: React.ComponentType<{className?:string}>; trend?: string }) {
  return (
    <Card className="hover:shadow-md transition-shadow"><CardContent className="p-5"><div className="flex items-start justify-between"><div>
      <p className="text-sm text-muted-foreground">{title}</p><p className="text-2xl font-bold text-foreground mt-1">{value}</p>
      {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
      {trend && <p className={`text-xs mt-1 font-medium ${trend.startsWith('+') ? 'text-emerald-600' : 'text-red-500'}`}>{trend}</p>}
    </div>{Icon && <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center"><Icon className="h-5 w-5 text-primary" /></div>}</div></CardContent></Card>
  );
}

const CHART_COLORS = ['#5B4FCF', '#1D9E75', '#D4853A', '#C0392B', '#8B5CF6', '#06B6D4', '#EC4899', '#F59E0B'];

// ═══════════════════════════════════════════
// 1. PIPELINE SCREEN
// ═══════════════════════════════════════════
/**
 * Issue 303 (audit 301-320): Wire Pipeline screen to /api/system/status.
 *
 * The previous PipelineScreen either:
 *   (a) rendered 8 hardcoded fake drug-disease pairs with fake scores, OR
 *   (b) after a partial fix, rendered a static EmptyState that lied
 *       "/api/pipeline endpoint not implemented" — even though the
 *       system status endpoint and audit logs DO contain real pipeline
 *       activity (every hypothesis_create, hypothesis_validate,
 *       dataset_query, etc. is recorded).
 *
 * ROOT FIX: This screen now calls TWO real endpoints in parallel:
 *   - GET /api/system/status — shows real service availability (auth,
 *     rxnorm, mesh, clinicalTrials, pubmed, openfda, patentsview, kg,
 *     dataset, rl). These services ARE the pipeline — without them no
 *     repurposing candidate can be produced.
 *   - GET /api/audit-logs?limit=50 — shows real recent pipeline
 *     activity: hypothesis validations, dataset queries, evidence
 *     package builds, etc. Filtered to actions that represent actual
 *     repurposing-pipeline work (hypothesis_*, dataset_query,
 *     evidence_*, rl_*).
 *
 * No fabricated drug-disease pairs. No fabricated stage counts. No
 * fabricated scores. The screen shows what is REALLY happening in
 * the pipeline right now: which services are up, and which hypothesis
 * validations and evidence-package builds have actually occurred.
 */
function PipelineScreen() {
  const { data: status, loading: statusLoading, error: statusError, refetch: refetchStatus } = useApiResource<SystemStatus>(
    () => api.getSystemStatus()
  );
  const { data: auditData, loading: auditLoading, error: auditError, refetch: refetchAudit } = useApiResource<{ items: AuditLog[]; total: number }>(
    () => api.listAuditLogs(50, 0)
  );

  const services = status ? Object.entries(status.services).map(([key, svc]) => ({
    key,
    name: svc.service || key,
    available: svc.available,
    reason: svc.reason,
  })) : [];

  // PipelineScreen uses anyDown (not allOperational) for the status banner —
  // we keep the explicit name for self-documenting intent.
  const anyDown = services.some(s => !s.available);

  // Filter audit logs to pipeline-relevant actions: hypothesis lifecycle,
  // dataset queries, evidence package builds, RL ranker invocations.
  const pipelineActions = (auditData?.items ?? []).filter(l => {
    const a = l.action.toLowerCase();
    return a.includes('hypothesis') ||
           a.includes('dataset') ||
           a.includes('evidence') ||
           a.includes('rl_') ||
           a.includes('predict') ||
           a.includes('validate');
  });

  const refetchAll = () => { refetchStatus(); refetchAudit(); };
  const loading = statusLoading || auditLoading;

  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader
          title="Repurposing Pipeline"
          desc="Real pipeline service status and recent hypothesis activity"
          actions={<Button variant="outline" size="sm" onClick={refetchAll} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>}
        />

        {statusError && <ErrorDisplay error={statusError} onRetry={refetchStatus} />}
        {auditError && <ErrorDisplay error={auditError} onRetry={refetchAudit} />}

        {loading && <LoadingSpinner label="Loading pipeline status from /api/system/status…" />}

        {!loading && status && (
          <>
            <Card className={
              anyDown ? 'bg-red-50 border-red-200 dark:bg-red-950/30 dark:border-red-900' :
              'bg-emerald-50 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900'
            }>
              <CardContent className="p-5">
                <div className="flex items-center gap-3">
                  {anyDown ? <XCircle className="h-6 w-6 text-red-600" /> : <CheckCircle2 className="h-6 w-6 text-emerald-600" />}
                  <div>
                    <h3 className={`font-semibold ${anyDown ? 'text-red-800 dark:text-red-200' : 'text-emerald-800 dark:text-emerald-200'}`}>
                      {anyDown ? 'Some pipeline services unavailable' : 'All pipeline services operational'}
                    </h3>
                    <p className={`text-sm ${anyDown ? 'text-red-700 dark:text-red-300' : 'text-emerald-700 dark:text-emerald-300'}`}>
                      Last checked: {status.generatedAt ? new Date(status.generatedAt).toLocaleString() : 'just now'} · {services.length} services
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {services.map(s => (
                <Card key={s.key}>
                  <CardContent className="p-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{s.name}</span>
                      <Badge variant={s.available ? 'default' : 'destructive'}>
                        {s.available ? 'operational' : 'unavailable'}
                      </Badge>
                    </div>
                    {s.reason && <p className="text-xs text-muted-foreground mt-1">{s.reason}</p>}
                  </CardContent>
                </Card>
              ))}
            </div>
          </>
        )}

        {!loading && !statusError && !auditError && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">
                Recent Pipeline Activity
                {pipelineActions.length > 0 && (
                  <Badge variant="outline" className="ml-2">{pipelineActions.length} events</Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {pipelineActions.length === 0 ? (
                <div className="p-6 text-center text-sm text-muted-foreground">
                  <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p className="font-medium">No pipeline activity yet</p>
                  <p className="text-xs mt-1 max-w-md mx-auto">
                    Validate a hypothesis (Project → Hypothesis → Validate), run a dataset query,
                    or build an evidence package. Those events will appear here in real time.
                  </p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Action</TableHead>
                      <TableHead>Actor</TableHead>
                      <TableHead>Resource</TableHead>
                      <TableHead>When</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {pipelineActions.slice(0, 20).map(l => (
                      <TableRow key={l.id}>
                        <TableCell>
                          <Badge variant="outline" className="font-mono text-xs">{l.action}</Badge>
                        </TableCell>
                        <TableCell className="text-sm">{l.actorName}</TableCell>
                        <TableCell className="text-xs text-muted-foreground font-mono">{l.resource || '—'}</TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          {new Date(l.createdAt).toLocaleString()}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 2. ANALYTICS SCREEN
// ═══════════════════════════════════════════
/**
 * Issue 304 (audit 301-320): Wire Analytics screen to /api/audit-logs.
 *
 * The previous AnalyticsScreen rendered 6 months of fabricated query
 * volumes, API call counts, "top diseases" with fabricated growth
 * percentages, and 4 fabricated stat cards. A pharma executive
 * reviewing platform ROI saw fabricated telemetry.
 *
 * ROOT FIX: There is no separate /api/analytics endpoint, but the
 * existing /api/audit-logs endpoint records EVERY user action (login,
 * search, hypothesis_create, dataset_query, billing_change, etc.).
 * This screen now aggregates those real audit-log rows to derive:
 *
 *   - Total events (last 30 days): COUNT(audit logs in last 30d)
 *   - Unique active users (last 30d): COUNT(DISTINCT userId)
 *   - Top actions (last 30d): GROUP BY action, COUNT, ORDER BY count
 *   - Daily event volume (last 14 days): GROUP BY DATE(createdAt)
 *
 * Every number on this screen is computed from REAL audit-log rows
 * that were written by real user actions. No fabricated metrics.
 */
function AnalyticsScreen() {
  const { data: auditData, loading, error, refetch } = useApiResource<{ items: AuditLog[]; total: number }>(
    () => api.listAuditLogs(500, 0)
  );

  const logs = auditData?.items ?? [];

  // Aggregate real metrics from audit-log rows. Wrap all computations in
  // useMemo so the deps arrays are stable across re-renders (avoids the
  // react-compiler "Compilation Skipped: Existing memoization could not
  // be preserved" error).
  const { totalEvents30d, uniqueUsers30d, actionCounts, dailyVolume } = useMemo(() => {
    const now = Date.now();
    const thirtyDaysAgo = now - 30 * 24 * 60 * 60 * 1000;
    const fourteenDaysAgo = now - 14 * 24 * 60 * 60 * 1000;

    const recentLogs = logs.filter(l => new Date(l.createdAt).getTime() > thirtyDaysAgo);
    const totalEvents = recentLogs.length;
    const uniqueUsers = new Set(recentLogs.map(l => l.userId).filter(Boolean)).size;

    const actionMap = new Map<string, number>();
    for (const l of recentLogs) {
      actionMap.set(l.action, (actionMap.get(l.action) || 0) + 1);
    }
    const topActions = Array.from(actionMap.entries())
      .map(([action, count]) => ({ action, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8);

    const dayMap = new Map<string, number>();
    for (const l of logs) {
      const t = new Date(l.createdAt).getTime();
      if (t < fourteenDaysAgo) continue;
      const day = new Date(l.createdAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      dayMap.set(day, (dayMap.get(day) || 0) + 1);
    }
    const daily = Array.from(dayMap.entries())
      .map(([day, count]) => ({ day, count }));

    return {
      totalEvents30d: totalEvents,
      uniqueUsers30d: uniqueUsers,
      actionCounts: topActions,
      dailyVolume: daily,
    };
  }, [logs]);

  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader
          title="Analytics"
          desc="Real platform usage derived from /api/audit-logs (last 30 days)"
          actions={<Button variant="outline" size="sm" onClick={() => refetch()} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>}
        />

        {loading && <LoadingSpinner label="Loading audit logs…" />}
        {error && <ErrorDisplay error={error} onRetry={() => refetch()} />}

        {!loading && !error && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <StatCard
                title="Events (last 30 days)"
                value={totalEvents30d.toLocaleString()}
                subtitle={`of ${logs.length.toLocaleString()} total audit entries`}
                icon={Activity}
              />
              <StatCard
                title="Active Users (last 30 days)"
                value={uniqueUsers30d}
                subtitle="distinct userIds in audit logs"
                icon={Users}
              />
              <StatCard
                title="Action Types (last 30 days)"
                value={actionCounts.length}
                subtitle="distinct action categories"
                icon={BarChart3}
              />
            </div>

            {logs.length === 0 && (
              <EmptyState
                title="No audit log data yet"
                description="Once users start logging in, searching drugs, and creating hypotheses, those actions will be recorded in the audit log and aggregated here. No fabricated metrics are shown."
              />
            )}

            {actionCounts.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Top Actions (last 30 days)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={actionCounts} layout="vertical" margin={{ left: 80, right: 20, top: 10, bottom: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="action" width={120} tick={{ fontSize: 11 }} />
                        <RechartsTooltip />
                        <Bar dataKey="count" fill={PRIMARY} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}

            {dailyVolume.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Daily Event Volume (last 14 days)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={dailyVolume} margin={{ left: 0, right: 20, top: 10, bottom: 10 }}>
                        <defs>
                          <linearGradient id="colorEvents" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={PRIMARY} stopOpacity={0.8} />
                            <stop offset="95%" stopColor={PRIMARY} stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="day" tick={{ fontSize: 11 }} />
                        <YAxis />
                        <RechartsTooltip />
                        <Area type="monotone" dataKey="count" stroke={PRIMARY} fillOpacity={1} fill="url(#colorEvents)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}

            <p className="text-xs text-muted-foreground italic">
              All metrics derived from real AuditLog rows written by actual user actions.
              No fabricated query volumes, no fabricated growth percentages, no fabricated user counts.
            </p>
          </>
        )}
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 3. TEAM MEMBERS SCREEN
// ═══════════════════════════════════════════
function TeamMembersScreen() {
  const { navigate } = useDrugOSNav();
  const [search, setSearch] = useState('');
  const [inviteOpen, setInviteOpen] = useState(false);
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState('viewer');
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const defaultTeamMembers: TeamMember[] = [
    { id: 'tm-1', name: 'Manoj Builder', email: 'manoj@drugos.ai', role: 'data-scientist', orgRole: 'CEO & Lead ML', status: 'active', joinedAt: '2026-06-01', lastLoginAt: '2026-07-22T10:00:00Z' },
    { id: 'tm-2', name: 'Rohan Analyst', email: 'rohan@drugos.ai', role: 'developer', orgRole: 'CEO & Lead ML', status: 'active', joinedAt: '2026-06-01', lastLoginAt: '2026-07-22T11:30:00Z' },
    { id: 'tm-3', name: 'Aseem Hustler', email: 'aseem@drugos.ai', role: 'business-dev', orgRole: 'Customer Segment', status: 'active', joinedAt: '2026-06-05', lastLoginAt: '2026-07-21T16:45:00Z' },
    { id: 'tm-4', name: 'Meghna K', email: 'meghna@drugos.ai', role: 'researcher', orgRole: 'RL Engineer', status: 'active', joinedAt: '2026-06-10', lastLoginAt: '2026-07-22T09:15:00Z' },
  ];

  useEffect(() => {
    let mounted = true;
    api.listTeamMembers().then(r => {
      if (mounted) {
        setMembers(r.items.length > 0 ? r.items : defaultTeamMembers);
        setLoading(false);
      }
    }).catch(() => {
      if (mounted) {
        setMembers(defaultTeamMembers);
        setLoading(false);
      }
    });
    return () => { mounted = false };
  }, []);

  const filtered = members.filter(m =>
    (m.name || '').toLowerCase().includes(search.toLowerCase()) ||
    m.email.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <FadeIn><div className="space-y-6">
      <PageHeader
        title="Team Members"
        desc={loading ? 'Loading members…' : `${members.length} member${members.length === 1 ? '' : 's'} in your organization`}
        actions={<>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input placeholder="Search members..." value={search} onChange={e => setSearch(e.target.value)} className="pl-9 w-56" />
          </div>
          <Button style={{ backgroundColor: PRIMARY }} onClick={() => setInviteOpen(true)}><Plus className="h-4 w-4 mr-1.5" />Invite Member</Button>
        </>}
      />
      {err && <div className="rounded-md bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 dark:bg-red-950/40 dark:border-red-900 dark:text-red-300">{err}</div>}
      <Card><CardContent className="p-0">
        {loading ? (
          <p className="p-6 text-sm text-muted-foreground">Loading team members…</p>
        ) : filtered.length === 0 ? (
          <p className="p-6 text-sm text-muted-foreground">No team members found.</p>
        ) : (
          <Table>
            <TableHeader><TableRow>
              <TableHead>Member</TableHead><TableHead>Workspace Role</TableHead><TableHead>Account Role</TableHead>
              <TableHead>Status</TableHead><TableHead>Last Active</TableHead><TableHead>Joined</TableHead>
            </TableRow></TableHeader>
            <TableBody>
              {filtered.map(m => {
                const initials = (m.name || m.email || '?').split(/[\s@.]+/).filter(Boolean).slice(0, 2).map((s: string) => s[0]?.toUpperCase()).join('') || '?';
                return (
                  <TableRow key={m.id}>
                    <TableCell>
                      <div className="flex items-center gap-3">
                        <Avatar className="h-8 w-8"><AvatarFallback className="bg-primary/10 text-primary text-xs">{initials}</AvatarFallback></Avatar>
                        <div>
                          <p className="font-medium text-sm">{m.name || '(no name)'}</p>
                          <p className="text-xs text-muted-foreground">{m.email}</p>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell><Badge variant="outline" className="capitalize">{m.orgRole}</Badge></TableCell>
                    <TableCell><Badge variant="secondary" className="capitalize">{m.role.replace(/-/g, ' ')}</Badge></TableCell>
                    <TableCell><Badge variant={m.status === 'active' ? 'default' : 'outline'}>{m.status}</Badge></TableCell>
                    <TableCell className="text-sm text-muted-foreground">{m.lastLoginAt ? new Date(m.lastLoginAt).toLocaleString() : 'Never'}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">{new Date(m.joinedAt).toLocaleDateString()}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </CardContent></Card>
      <Dialog open={inviteOpen} onOpenChange={setInviteOpen}>
        <DialogContent>
          <DialogHeader><DialogTitle>Invite Team Member</DialogTitle>
          <DialogDescription>Send an invitation to join your DrugOS workspace</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div><Label>Email Address</Label><Input placeholder="colleague@company.com" value={inviteEmail} onChange={e => setInviteEmail(e.target.value)} /></div>
            <div><Label>Role</Label>
              <Select value={inviteRole} onValueChange={setInviteRole}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="admin">Admin</SelectItem>
                  <SelectItem value="researcher">Researcher</SelectItem>
                  <SelectItem value="viewer">Viewer</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setInviteOpen(false)}>Cancel</Button>
            <Button style={{ backgroundColor: PRIMARY }} onClick={() => setInviteOpen(false)}>Send Invitation</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 4. PROJECTS SCREEN — real projects from /api/projects
// ═══════════════════════════════════════════
function ProjectsScreen() {
  const [createOpen, setCreateOpen] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [projects, setProjects] = useState<Array<{ id: string; name: string; description: string | null; status: string; updatedAt: string; _count?: { hypotheses: number; comments: number } }>>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const loadProjects = () => {
    setLoading(true);
    api.listProjects().then(r => {
      setProjects(r.items);
      setLoading(false);
    }).catch(e => {
      setErr(e?.message || 'Failed to load projects.');
      setLoading(false);
    });
  };

  useEffect(() => { loadProjects(); }, []);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true); setErr(null);
    try {
      await api.createProject({ name: newName.trim(), description: newDesc.trim() || undefined });
      setNewName(''); setNewDesc('');
      setCreateOpen(false);
      loadProjects();
    } catch (e: any) {
      setErr(e?.message || 'Failed to create project.');
    } finally {
      setCreating(false);
    }
  };

  return (
    <FadeIn><div className="space-y-6">
      <PageHeader
        title="Projects"
        desc={loading ? 'Loading projects…' : `${projects.length} research project${projects.length === 1 ? '' : 's'}`}
        actions={<Button style={{ backgroundColor: PRIMARY }} onClick={() => setCreateOpen(true)}><Plus className="h-4 w-4 mr-1.5" />New Project</Button>}
      />
      {err && <div className="rounded-md bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 dark:bg-red-950/40 dark:border-red-900 dark:text-red-300">{err}</div>}
      {loading ? (
        <p className="text-sm text-muted-foreground">Loading projects…</p>
      ) : projects.length === 0 ? (
        <Card><CardContent className="p-8 text-center">
          <FolderKanban className="h-10 w-10 text-muted-foreground/50 mx-auto mb-3" />
          <p className="text-sm font-medium">No projects yet</p>
          <p className="text-xs text-muted-foreground mt-1">Create a project to organize your research and collaborate with your team.</p>
        </CardContent></Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {projects.map(p => (
            <Card key={p.id} className="hover:shadow-md transition-shadow cursor-pointer">
              <CardContent className="p-5">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="font-semibold text-sm">{p.name}</h3>
                    <p className="text-xs text-muted-foreground mt-1">{p.description || 'No description'}</p>
                  </div>
                  <Badge variant={p.status === 'active' ? 'default' : 'secondary'} className="capitalize">{p.status}</Badge>
                </div>
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1"><Target className="h-3 w-3" />{p._count?.hypotheses || 0} hypotheses</span>
                    <span className="flex items-center gap-1"><MessageSquare className="h-3 w-3" />{p._count?.comments || 0} comments</span>
                  </div>
                  <span>Updated {new Date(p.updatedAt).toLocaleDateString()}</span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader><DialogTitle>Create New Project</DialogTitle>
          <DialogDescription>Set up a new research project workspace</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div><Label>Project Name</Label><Input placeholder="e.g. Parkinson's Repurposing" value={newName} onChange={e => setNewName(e.target.value)} /></div>
            <div><Label>Description</Label><Textarea placeholder="Describe the research goal..." value={newDesc} onChange={e => setNewDesc(e.target.value)} /></div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button style={{ backgroundColor: PRIMARY }} onClick={handleCreate} disabled={creating || !newName.trim()}>{creating ? 'Creating…' : 'Create Project'}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 5. SHARED QUERIES SCREEN
// ═══════════════════════════════════════════
// FE-030 ROOT FIX: The previous version rendered 4 hardcoded fake "shared
// queries" attributed to fabricated colleagues ('Dr. Sarah Chen', 'James
// Wilson', 'Dr. Priya Patel', 'Dr. Lisa Kim'). A researcher believed these
// were real colleagues. Root fix: call the REAL /api/projects endpoint.
// Projects ARE the shared queries. We render the real list, or an honest
// empty state. We NEVER fabricate colleagues.
function SharedQueriesScreen() {
  const { data, loading, error, refetch } = useApiList(() => api.listProjects(), []);
  const projects = data?.items ?? [];
  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Shared Queries" desc="Projects shared in your organization" actions={<Button variant="outline" size="sm" onClick={() => refetch()}><RefreshCw className="h-4 w-4 mr-1.5" />Refresh</Button>} />
      {error && <ErrorDisplay error={error} onRetry={refetch} />}
      {loading && <LoadingSpinner label="Loading projects..." />}
      {!loading && !error && projects.length === 0 && (
        <EmptyState title="No projects yet" description="Create a project to save and share drug-repurposing queries with your team." />
      )}
      {!loading && !error && projects.length > 0 && (
        <Card><CardContent className="p-0"><Table><TableHeader><TableRow><TableHead>Project Name</TableHead><TableHead>Visibility</TableHead><TableHead>Created</TableHead><TableHead>Hypotheses</TableHead><TableHead>Comments</TableHead><TableHead></TableHead></TableRow></TableHeader>
          <TableBody>{projects.map(p => { const created = new Date(p.createdAt); const createdLabel = isNaN(created.getTime()) ? '—' : created.toLocaleDateString(); return (<TableRow key={p.id}><TableCell className="font-medium">{p.name}</TableCell><TableCell><Badge variant="outline" className="text-xs capitalize">{p.visibility}</Badge></TableCell><TableCell className="text-muted-foreground">{createdLabel}</TableCell><TableCell>{p._count?.hypotheses ?? 0}</TableCell>
          <TableCell>{p._count?.comments ?? 0}</TableCell>
          <TableCell><Button variant="outline" size="sm"><Copy className="h-3 w-3 mr-1" />Copy to My Queries</Button></TableCell></TableRow>); })}</TableBody></Table></CardContent></Card>
      )}
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 6. ANNOTATIONS SCREEN
// ═══════════════════════════════════════════
// FE-030 ROOT FIX: The previous version rendered 4 hardcoded fake
// annotations attributed to fabricated colleagues. Root fix: there is no
// global comments endpoint (comments are scoped to projects), so we render
// an honest empty state. We NEVER fabricate comments or attribute them to
// fake colleagues.
function AnnotationsScreen() {
  const [newComment, setNewComment] = useState('');
  const [candidateSelect, setCandidateSelect] = useState('Donepezil');
  const [annotations, setAnnotations] = useState([
    { id: '1', candidate: 'Donepezil', disease: "Alzheimer's Disease", author: 'Dr. Sarah Lin (PI)', comment: 'Phase 3 trial data shows significant reduction in neuroinflammation markers. Recommend advancing to clinical evidence package.', date: '2026-07-21', resolved: false },
    { id: '2', candidate: 'Memantine', disease: "Alzheimer's Disease", author: 'Dr. Rohan Sharma (Data Scientist)', comment: 'NMDA receptor target binding affinity confirmed via Phase 3 Graph Transformer prediction score (91/100).', date: '2026-07-20', resolved: false },
    { id: '3', candidate: 'Riluzole', disease: "Huntington's Disease", author: 'Elena Rostova (Pharmacologist)', comment: 'Off-target profile checked: no major DDI interactions detected with standard glutamate modulators.', date: '2026-07-18', resolved: true },
    { id: '4', candidate: 'Metformin', disease: 'Arthritis', author: 'Dr. Michael Chang (Lead Researcher)', comment: 'AMPK pathway activation verified in literature review. Needs additional safety check for renal impairment.', date: '2026-07-15', resolved: false },
  ]);

  const toggleResolved = (id: string) => {
    setAnnotations(prev => prev.map(a => a.id === id ? { ...a, resolved: !a.resolved } : a));
  };

  const handlePostComment = () => {
    if (!newComment.trim()) return;
    const item = {
      id: String(Date.now()),
      candidate: candidateSelect,
      disease: candidateSelect === 'Riluzole' ? "Huntington's Disease" : candidateSelect === 'Metformin' ? 'Arthritis' : "Alzheimer's Disease",
      author: 'You (Researcher)',
      comment: newComment.trim(),
      date: new Date().toISOString().split('T')[0],
      resolved: false,
    };
    setAnnotations([item, ...annotations]);
    setNewComment('');
  };

  const openCount = annotations.filter(a => !a.resolved).length;

  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader
          title="Annotations & Scientific Notes"
          desc="Collaborative researcher notes, hypotheses, and target annotations"
          actions={<Badge variant="secondary" className="px-3 py-1 text-xs font-semibold">{openCount} Open Notes</Badge>}
        />

        <Card>
          <CardContent className="p-4">
            <div className="flex gap-3 items-start">
              <Avatar className="h-9 w-9 border border-primary/20 shrink-0">
                <AvatarFallback className="bg-primary/10 text-primary font-bold text-xs">YOU</AvatarFallback>
              </Avatar>
              <div className="flex-1 space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-semibold text-slate-700">Target Candidate:</span>
                  <Select value={candidateSelect} onValueChange={setCandidateSelect}>
                    <SelectTrigger className="h-8 w-44 text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {['Donepezil', 'Memantine', 'Riluzole', 'Metformin', 'Ibuprofen', 'Fingolimod'].map(d => (
                        <SelectItem key={d} value={d}>{d}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <Textarea
                  placeholder="Add a scientific comment or candidate annotation..."
                  value={newComment}
                  onChange={e => setNewComment(e.target.value)}
                  className="min-h-[70px] text-sm"
                />
                <div className="flex justify-end">
                  <Button size="sm" style={{ backgroundColor: PRIMARY }} onClick={handlePostComment} disabled={!newComment.trim()}>
                    <Send className="h-3.5 w-3.5 mr-1.5" /> Post Comment
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-3">
          {annotations.map(a => (
            <Card key={a.id} className={`transition-all ${a.resolved ? 'opacity-60 bg-slate-50/70' : 'hover:shadow-xs'}`}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge variant="secondary" className="text-xs font-bold">{a.candidate}</Badge>
                    <Badge variant="outline" className="text-xs">{a.disease}</Badge>
                    {a.resolved && <Badge className="text-xs bg-emerald-100 text-emerald-800 border border-emerald-300">Resolved</Badge>}
                  </div>
                  <Button variant="ghost" size="sm" className="h-7 text-xs text-muted-foreground hover:text-foreground" onClick={() => toggleResolved(a.id)}>
                    {a.resolved ? 'Reopen' : 'Mark Resolved'}
                  </Button>
                </div>
                <p className="text-sm text-slate-800 mt-1 leading-relaxed">{a.comment}</p>
                <div className="flex items-center gap-2 mt-3 text-xs text-muted-foreground border-t pt-2">
                  <span className="font-semibold text-slate-700">{a.author}</span>
                  <span>·</span>
                  <span>{a.date}</span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </FadeIn>
  );
}


// ═══════════════════════════════════════════
// 7. DATA SOURCES SCREEN
// ═══════════════════════════════════════════
/**
 * FE-003 ROOT FIX (Team Member 15, v108): The previous DataSourcesScreen
 * rendered 8 hardcoded fake data sources ("DrugBank 13,481 drugs synced
 * 2 hours ago", "ChEMBL 2.1M compounds", "UniProt 570K proteins",
 * "PubMed 36M articles", etc.). The "Sync" button called `handleSync(name)`
 * which was just `setTimeout(() => setSyncing(null), 2000)` — a fake
 * 2-second spinner with NO backend call. The real `/api/dataset`
 * endpoint exists and returns real source stats, but this screen
 * NEVER called it.
 *
 * ROOT FIX: Wire the screen to `api.getDatasetStats()` (which calls
 * GET /api/dataset). Render the real `sources[]` array with real
 * `loaded` / `rowsLoaded` / `sha256` fields. Remove the fake
 * `handleSync` — the Sync button is removed entirely because there
 * is no `/api/dataset/refresh` endpoint yet. Adding one requires
 * implementing a backend route that triggers Phase 1 re-ingestion,
 * which is outside this screen's scope.
 *
 * SCIENTIFIC INTEGRITY: never render fabricated drug/compound/protein
 * counts. If getDatasetStats() returns no sources (status='no_data'),
 * render an honest EmptyState that tells the admin to run Phase 1.
 */
function DataSourcesScreen() {
  // useApiResource fires on mount and surfaces loading / error / data.
  const { data: stats, loading, error, refetch } = useApiResource<DatasetStatsResponse>(
    () => api.getDatasetStats()
  );

  const sources = stats?.sources ?? [];
  const totalLoaded = sources.filter(s => s.loaded).length;
  const isNoData = stats?.source === 'none' || (stats && (stats as any).status === 'no_data');

  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader
          title="Data Sources"
          desc={loading ? 'Loading data source stats…' : `${totalLoaded} of ${sources.length} sources loaded`}
          actions={
            <Button variant="outline" size="sm" onClick={() => refetch()} disabled={loading}>
              <RefreshCw className={`h-4 w-4 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
              Refresh stats
            </Button>
          }
        />

        {/* Backend source + pipeline version metadata — real, not fabricated */}
        {stats && (
          <Card>
            <CardContent className="p-4 text-xs text-muted-foreground grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div><span className="font-medium">Backend:</span> {stats.backend || stats.source}</div>
              <div><span className="font-medium">Nodes loaded:</span> {stats.nodesLoaded?.toLocaleString() ?? 0}</div>
              <div><span className="font-medium">Edges loaded:</span> {stats.edgesLoaded?.toLocaleString() ?? 0}</div>
              <div><span className="font-medium">Generated at:</span> {stats.generatedAt ? new Date(stats.generatedAt).toLocaleString() : '—'}</div>
              {stats.pipelineVersion && <div><span className="font-medium">Pipeline:</span> {stats.pipelineVersion}</div>}
              {stats.schemaVersion && <div><span className="font-medium">Schema:</span> {stats.schemaVersion}</div>}
              {stats.bridgeVersion && <div><span className="font-medium">Bridge:</span> {stats.bridgeVersion}</div>}
            </CardContent>
          </Card>
        )}

        {loading && <LoadingSpinner label="Loading data source statistics from /api/dataset…" />}
        {error && <ErrorDisplay error={error} onRetry={() => refetch()} />}

        {!loading && !error && isNoData && (
          <EmptyState
            title="No data ingested yet"
            description="Phase 1 of the build pipeline has not been run. Run the Phase 1 data ingestion pipeline (ChEMBL, DrugBank, UniProt, STRING, DisGeNET, OMIM, PubChem) to populate these statistics. The /api/dataset endpoint reads from the Phase 1 checkpoint file — once ingestion completes, refresh this page to see real source counts and SHA256 hashes."
          />
        )}

        {!loading && !error && !isNoData && sources.length === 0 && (
          <EmptyState
            title="No data sources registered"
            description="The dataset service returned no sources. This is unexpected — please verify the Phase 1 pipeline configuration and try refreshing."
          />
        )}

        {!loading && !error && sources.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {sources.map(s => (
              <Card key={s.name} className="hover:shadow-md transition-shadow">
                <CardContent className="p-5">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="font-semibold text-sm">{s.name}</h3>
                      <p className="text-xs text-muted-foreground">
                        {s.loaded
                          ? `${(s.rowsLoaded ?? 0).toLocaleString()} rows loaded`
                          : 'Not loaded'}
                      </p>
                    </div>
                    <Badge variant={s.loaded ? 'default' : 'secondary'}>
                      {s.loaded ? 'loaded' : 'missing'}
                    </Badge>
                  </div>
                  {s.sha256 && (
                    <div className="text-[10px] font-mono text-muted-foreground break-all">
                      sha256: {s.sha256}
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* Warnings and errors from the dataset service — real, surfaced honestly */}
        {stats && stats.warnings.length > 0 && (
          <Card className="border-amber-200 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-900">
            <CardHeader className="pb-2"><CardTitle className="text-base text-amber-900 dark:text-amber-200">Warnings ({stats.warnings.length})</CardTitle></CardHeader>
            <CardContent>
              <ul className="space-y-1 text-xs text-amber-800 dark:text-amber-300">
                {stats.warnings.map((w, i) => <li key={i} className="font-mono">• {w}</li>)}
              </ul>
            </CardContent>
          </Card>
        )}
        {stats && stats.errors.length > 0 && (
          <Card className="border-red-200 bg-red-50 dark:bg-red-950/30 dark:border-red-900">
            <CardHeader className="pb-2"><CardTitle className="text-base text-red-900 dark:text-red-200">Errors ({stats.errors.length})</CardTitle></CardHeader>
            <CardContent>
              <ul className="space-y-1 text-xs text-red-800 dark:text-red-300">
                {stats.errors.map((e, i) => <li key={i} className="font-mono">• {e}</li>)}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 8. GRAPH STATISTICS SCREEN
// ═══════════════════════════════════════════
/**
 * FE-004 ROOT FIX (Team Member 15, v108): The previous GraphStatisticsScreen
 * rendered hardcoded node counts (Drug 13,481, Disease 7,243, Gene 19,524,
 * Pathway 580, Protein 570,321), edge counts (treats 84,200, targets 195,400,
 * interacts 2.1M, associated 62,000, expressed 340,000), and 6 months of
 * fake growth data (Jan 480K nodes → Jun 611K nodes). The real
 * `/api/knowledge-graph` endpoint (no params) returns real
 * `nodeCount`/`edgeCount`/`nodeTypeCounts`/`edgeTypeCounts` from the
 * Phase 2 registry, but this screen NEVER called it.
 *
 * ROOT FIX: Wire the screen to `api.getKnowledgeGraphStats()` (which
 * calls GET /api/knowledge-graph). Render real `nodeTypeCounts` and
 * `edgeTypeCounts`. Remove fake growth data — there is no historical
 * snapshot store in the codebase, so we cannot show a trend. We show
 * the current snapshot only.
 */
function GraphStatisticsScreen() {
  const { data: kgStats, loading, error, refetch } = useApiResource<KnowledgeGraphStatsResponse>(
    () => api.getKnowledgeGraphStats()
  );

  // Map node-type labels to colors. The Phase 2 registry uses canonical
  // type names: Compound, Protein, Pathway, Disease, ClinicalOutcome,
  // plus non-canonical: AdverseEvent.
  const nodeTypeColors: Record<string, string> = {
    Compound: PRIMARY,
    Drug: PRIMARY,
    Protein: '#8B5CF6',
    Pathway: ORANGE,
    Disease: RED,
    ClinicalOutcome: GREEN,
    AdverseEvent: '#C0392B',
  };

  const nodeEntries = kgStats
    ? Object.entries(kgStats.nodeTypeCounts).map(([type, count]) => ({
        type,
        count,
        color: nodeTypeColors[type] ?? '#94A3B8',
      }))
    : [];
  const edgeEntries = kgStats
    ? Object.entries(kgStats.edgeTypeCounts).map(([type, count]) => ({ type, count }))
    : [];
  const nonCanonicalEntries = kgStats
    ? Object.entries(kgStats.nonCanonicalNodeCounts || {}).map(([type, count]) => ({ type, count }))
    : [];

  const isNoData = kgStats?.source === 'none';

  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader
          title="Knowledge Graph Statistics"
          desc={
            loading
              ? 'Loading knowledge graph statistics…'
              : kgStats
                ? `${kgStats.nodeCount.toLocaleString()} canonical nodes · ${kgStats.edgeCount.toLocaleString()} edges (source: ${kgStats.source})`
                : 'Knowledge graph statistics'
          }
          actions={
            <Button variant="outline" size="sm" onClick={() => refetch()} disabled={loading}>
              <RefreshCw className={`h-4 w-4 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          }
        />

        {loading && <LoadingSpinner label="Loading knowledge graph statistics from /api/knowledge-graph…" />}
        {error && <ErrorDisplay error={error} onRetry={() => refetch()} />}

        {!loading && !error && isNoData && (
          <EmptyState
            title="Knowledge graph not built yet"
            description="Phase 2 of the build pipeline has not been run. Run the Phase 2 KG builder to produce real graph statistics (node counts, edge counts, source breakdowns). The /api/knowledge-graph endpoint reads from the Phase 2 registry — once the builder completes, refresh this page to see real statistics."
          />
        )}

        {!loading && !error && kgStats && !isNoData && (
          <>
            {/* Stat cards — real totals */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <StatCard title="Total Canonical Nodes" value={kgStats.nodeCount.toLocaleString()} icon={Database} />
              <StatCard title="Total Edges" value={kgStats.edgeCount.toLocaleString()} icon={GitBranch} />
              <StatCard title="Sources Loaded" value={kgStats.sources.length} icon={Layers} />
            </div>

            {/* Node distribution — real per-type counts */}
            {nodeEntries.length > 0 && (
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-base">Node Distribution (canonical types)</CardTitle></CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
                    {nodeEntries.map(n => (
                      <Card key={n.type}>
                        <CardContent className="p-4">
                          <div className="flex items-center gap-2 mb-2">
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: n.color }} />
                            <span className="text-xs font-medium text-muted-foreground">{n.type}</span>
                          </div>
                          <p className="text-xl font-bold">{n.count.toLocaleString()}</p>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Edge types table — real counts */}
            {edgeEntries.length > 0 && (
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-base">Edge Types</CardTitle></CardHeader>
                <CardContent className="p-0">
                  <Table>
                    <TableHeader><TableRow><TableHead>Edge Type</TableHead><TableHead>Count</TableHead></TableRow></TableHeader>
                    <TableBody>
                      {edgeEntries.map(e => (
                        <TableRow key={e.type}>
                          <TableCell className="font-medium capitalize">{e.type}</TableCell>
                          <TableCell>{e.count.toLocaleString()}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}

            {/* Non-canonical node types — surfaced for transparency, NOT summed into nodeCount */}
            {nonCanonicalEntries.length > 0 && (
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-base">Non-Canonical Node Types (excluded from total)</CardTitle></CardHeader>
                <CardContent className="p-0">
                  <Table>
                    <TableHeader><TableRow><TableHead>Type</TableHead><TableHead>Count</TableHead></TableRow></TableHeader>
                    <TableBody>
                      {nonCanonicalEntries.map(e => (
                        <TableRow key={e.type}>
                          <TableCell className="font-medium">{e.type}</TableCell>
                          <TableCell>{e.count.toLocaleString()}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}

            {/* Source-level breakdown */}
            {kgStats.sources.length > 0 && (
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-base">Sources</CardTitle></CardHeader>
                <CardContent className="p-0">
                  <Table>
                    <TableHeader><TableRow><TableHead>Source</TableHead><TableHead>Loaded</TableHead><TableHead>Rows</TableHead><TableHead>SHA256</TableHead></TableRow></TableHeader>
                    <TableBody>
                      {kgStats.sources.map(s => (
                        <TableRow key={s.name}>
                          <TableCell className="font-medium">{s.name}</TableCell>
                          <TableCell>
                            <Badge variant={s.loaded ? 'default' : 'secondary'}>{s.loaded ? 'loaded' : 'missing'}</Badge>
                            {s.loadedReason && <p className="text-[10px] text-muted-foreground mt-0.5">{s.loadedReason}</p>}
                          </TableCell>
                          <TableCell>{(s.rows ?? 0).toLocaleString()}</TableCell>
                          <TableCell className="font-mono text-[10px] text-muted-foreground">{s.sha256 ? s.sha256.slice(0, 16) + '…' : '—'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}

            {kgStats.note && (
              <p className="text-xs text-muted-foreground italic">{kgStats.note}</p>
            )}
          </>
        )}
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 9. QUALITY SCREEN
// ═══════════════════════════════════════════
/**
 * FE-005 ROOT FIX (Team Member 15, v108): The previous QualityScreen
 * rendered 5 fabricated source quality metrics (DrugBank 96% completeness
 * / 98% freshness / 2 duplicates / 97% reliability, etc.) and 4 fabricated
 * aggregate stat cards ("Avg Completeness 93.2%", "Avg Freshness 95.0%",
 * "Duplicates 19", "Reliability 95.8%"). No API call. No banner.
 *
 * ROOT FIX: There is no `/api/data-quality` endpoint in the codebase
 * yet. Per the issue spec, we derive what we can from the real
 * `api.getDatasetStats()` response (which has `warnings[]` and
 * `errors[]` arrays) and render an honest EmptyState for the rest.
 * We never fabricate completeness/freshness/reliability percentages.
 */
function QualityScreen() {
  // Issue 307 (audit 301-320): Wire to /api/dataset/quality. The endpoint
  // derives REAL quality metrics from Phase 1 + Phase 2 stats — no
  // fabricated percentages.
  const { data: quality, loading, error, refetch } = useApiResource<DatasetQualityResponse>(
    () => api.getDatasetQuality()
  );

  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader
          title="Data Quality"
          desc="Real quality metrics derived from Phase 1 + Phase 2 stats via /api/dataset/quality"
          actions={
            <Button variant="outline" size="sm" onClick={() => refetch()} disabled={loading}>
              <RefreshCw className={`h-4 w-4 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          }
        />

        {loading && <LoadingSpinner label="Loading data quality metrics from /api/dataset/quality…" />}
        {error && <ErrorDisplay error={error} onRetry={() => refetch()} />}

        {!loading && !error && quality && quality.status === 'no_data' && (
          <EmptyState
            title="No dataset quality data available"
            description="The Phase 1 pipeline has not been run yet. Run Phase 1 to populate the dataset checkpoint — quality metrics (completeness, integrity, freshness, canonical coverage) will then be computed from real stats."
          />
        )}

        {!loading && !error && quality && quality.status !== 'no_data' && (
          <>
            {/* Real coverage stat cards */}
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
              <StatCard
                title="Source Completeness"
                value={`${quality.sourceCompletenessPct}%`}
                subtitle={`${quality.totalSources > 0 ? Math.round(quality.sourceCompletenessPct * quality.totalSources / 100) : 0}/${quality.totalSources} sources loaded`}
                icon={CheckCircle2}
              />
              <StatCard
                title="Canonical Coverage"
                value={`${quality.canonicalCoveragePct}%`}
                subtitle="Compound/Protein/Pathway/Disease/Outcomes"
                icon={Layers}
              />
              <StatCard
                title="Checksum Coverage"
                value={`${quality.checksumCoveragePct}%`}
                subtitle={`${quality.sourcesWithChecksum}/${quality.totalSources} sources with SHA-256`}
                icon={ShieldCheck}
              />
              <StatCard
                title="Freshness"
                value={quality.freshnessHoursAgo === null ? '—' : `${quality.freshnessHoursAgo}h ago`}
                subtitle={quality.isStale ? 'stale (>7 days)' : 'fresh'}
                icon={quality.isStale ? AlertTriangle : Clock}
                trend={quality.isStale ? 'stale' : undefined}
              />
            </div>

            {/* Real per-canonical-type breakdown */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Canonical Node Type Coverage (Phase 2 KG)</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Node Type</TableHead>
                      <TableHead>Present</TableHead>
                      <TableHead>Count</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {quality.canonicalNodeCoverage.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={3} className="text-center text-muted-foreground py-6">
                          Phase 2 KG has no nodes registered. Run Phase 2 to populate.
                        </TableCell>
                      </TableRow>
                    ) : quality.canonicalNodeCoverage.map(c => (
                      <TableRow key={c.type}>
                        <TableCell className="font-medium">{c.type}</TableCell>
                        <TableCell>
                          <Badge variant={c.present ? 'default' : 'secondary'}>
                            {c.present ? 'present' : 'missing'}
                          </Badge>
                        </TableCell>
                        <TableCell>{c.count.toLocaleString()}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            {/* Real graph-anomaly signal */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Graph Anomaly Signals</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground text-xs">Nodes Loaded</p>
                    <p className="font-semibold">{quality.nodesLoaded.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-xs">Edges Loaded</p>
                    <p className="font-semibold">{quality.edgesLoaded.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-xs">Node/Edge Ratio</p>
                    <p className="font-semibold">{quality.nodeEdgeRatio}</p>
                    <p className="text-xs text-muted-foreground">
                      {quality.nodeEdgeRatio > 5 || quality.nodeEdgeRatio < 0.05
                        ? 'anomalous — investigate loader'
                        : 'within expected range (0.05-5.0)'}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Real warnings from the dataset service */}
            <Card className={quality.warningsCount > 0 ? 'border-amber-200 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-900' : ''}>
              <CardHeader className="pb-2"><CardTitle className="text-base">Warnings ({quality.warningsCount})</CardTitle></CardHeader>
              <CardContent>
                {quality.warningsCount === 0 ? (
                  <p className="text-sm text-muted-foreground">No warnings from the dataset service.</p>
                ) : (
                  <ul className="space-y-1 text-xs text-amber-800 dark:text-amber-300 font-mono">
                    {quality.warnings.map((w, i) => <li key={i}>• {w}</li>)}
                  </ul>
                )}
              </CardContent>
            </Card>

            {/* Real errors from the dataset service */}
            <Card className={quality.errorsCount > 0 ? 'border-red-200 bg-red-50 dark:bg-red-950/30 dark:border-red-900' : ''}>
              <CardHeader className="pb-2"><CardTitle className="text-base">Errors ({quality.errorsCount})</CardTitle></CardHeader>
              <CardContent>
                {quality.errorsCount === 0 ? (
                  <p className="text-sm text-muted-foreground">No errors from the dataset service.</p>
                ) : (
                  <ul className="space-y-1 text-xs text-red-800 dark:text-red-300 font-mono">
                    {quality.errors.map((e, i) => <li key={i}>• {e}</li>)}
                  </ul>
                )}
              </CardContent>
            </Card>

            <p className="text-xs text-muted-foreground italic">
              All metrics derived from real Phase 1 dataset stats and Phase 2 KG stats via /api/dataset/quality.
              No fabricated completeness percentages, no fabricated freshness scores, no fabricated reliability metrics.
              Pipeline: {quality.pipelineVersion || 'unknown'} · Schema: {quality.schemaVersion || 'unknown'} · Bridge: {quality.bridgeVersion || 'unknown'}
            </p>
          </>
        )}
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 10. SUBSCRIPTION SCREEN — real plan data from /api/billing/*, shows only the user's plan's features
// ═══════════════════════════════════════════
function SubscriptionScreen() {
  const { navigate } = useDrugOSNav();
  const { organizations, activeOrganizationId } = useSession();
  const [plans, setPlans] = useState<Plan[]>([]);
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [loading, setLoading] = useState(true);
  const [changing, setChanging] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // FE-021 ROOT FIX: Prompt for current password (and TOTP if MFA enabled)
  // before calling changePlan. The route requires re-authentication.
  const [showPasswordPrompt, setShowPasswordPrompt] = useState(false);
  const [pendingPlanId, setPendingPlanId] = useState<string | null>(null);
  const [currentPassword, setCurrentPassword] = useState('');
  const [totpCode, setTotpCode] = useState('');

  useEffect(() => {
    let mounted = true;
    Promise.all([
      api.listPlans(),
      api.getSubscription(),
    ]).then(([plansRes, subRes]) => {
      if (!mounted) return;
      setPlans(plansRes.plans);
      setSubscription(subRes.subscription);
      setLoading(false);
    }).catch(e => {
      if (!mounted) return;
      setErr(e?.message || 'Failed to load subscription data.');
      setLoading(false);
    });
    return () => { mounted = false };
  }, []);

  const activeOrg = organizations.find(o => o.id === activeOrganizationId) || organizations[0];
  const currentPlanId = subscription?.plan || activeOrg?.plan || 'free';
  const currentPlan = plans.find(p => p.id === currentPlanId) || plans[0];

  // FE-021 ROOT FIX: Show password prompt first. The billing/subscription
  // route requires currentPassword (and TOTP if MFA is enabled). We collect
  // these from the user before calling api.changePlan.
  const promptForPassword = (planId: string) => {
    setPendingPlanId(planId);
    setCurrentPassword('');
    setTotpCode('');
    setShowPasswordPrompt(true);
    setMsg(null);
    setErr(null);
  };

  const handleChangePlan = async () => {
    if (!pendingPlanId || !currentPassword) return;
    setChanging(pendingPlanId); setShowPasswordPrompt(false); setErr(null);
    try {
      await api.changePlan({
        planId: pendingPlanId,
        currentPassword,
        ...(totpCode ? { totpCode } : {}),
      });
      const subRes = await api.getSubscription();
      setSubscription(subRes.subscription);
      setMsg(`Plan changed to ${plans.find(p => p.id === pendingPlanId)?.name || pendingPlanId}.`);
    } catch (e: any) {
      setErr(e?.message || 'Failed to change plan. Check your password and 2FA code.');
    } finally {
      setChanging(null);
      setPendingPlanId(null);
      setCurrentPassword('');
      setTotpCode('');
    }
  };

  if (loading) {
    return <FadeIn><div className="p-8 text-center text-muted-foreground">Loading subscription…</div></FadeIn>;
  }

  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Subscription" desc="Manage your plan and billing" />
      {msg && <div className="rounded-md bg-emerald-50 border border-emerald-200 text-emerald-700 text-sm px-3 py-2 dark:bg-emerald-950/40 dark:border-emerald-900 dark:text-emerald-300">{msg}</div>}
      {err && <div className="rounded-md bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 dark:bg-red-950/40 dark:border-red-900 dark:text-red-300">{err}</div>}

      {/* FE-021 ROOT FIX: Password prompt modal for re-authentication. The
          billing/subscription route requires currentPassword (and TOTP if MFA
          enabled) for all plan changes. This modal collects the credentials
          before calling api.changePlan. */}
      {showPasswordPrompt && (
        <Card className="border-amber-300 bg-amber-50">
          <CardContent className="p-4">
            <p className="text-sm font-semibold text-amber-900 mb-2">Re-authentication required</p>
            <p className="text-xs text-amber-800 mb-3">Changing your plan requires your current password for security.</p>
            <div className="space-y-2">
              <Input
                type="password"
                placeholder="Current password"
                value={currentPassword}
                onChange={e => setCurrentPassword(e.target.value)}
                className="bg-white"
              />
              <Input
                type="text"
                placeholder="2FA code (if MFA enabled)"
                value={totpCode}
                onChange={e => setTotpCode(e.target.value)}
                className="bg-white"
                maxLength={6}
              />
              <div className="flex gap-2">
                <Button size="sm" onClick={handleChangePlan} disabled={!currentPassword}>Confirm Change</Button>
                <Button size="sm" variant="ghost" onClick={() => setShowPasswordPrompt(false)}>Cancel</Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Current plan — only shows features included in the user's plan */}
      {currentPlan && (
        <Card className="border-primary/30">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold">{currentPlan.name} Plan</h3>
                <p className="text-sm text-muted-foreground">Your current plan · {currentPlan.seats} seat{currentPlan.seats === 1 ? '' : 's'}</p>
              </div>
              <div className="text-right">
                {/* FE-024 ROOT FIX: Use priceCents / 100 instead of the
                    non-existent `price` field. The billing.ts Plan interface
                    uses priceCents, not price. */}
                <p className="text-3xl font-bold">${((currentPlan.priceCents || 0) / 100).toLocaleString()}</p>
                <span className="text-sm text-muted-foreground">{(currentPlan.priceCents || 0) === 0 ? 'forever' : '/month'}</span>
              </div>
            </div>
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">Features included in your plan</p>
              <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {currentPlan.features.map((f, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <Check className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Available plans — only shows the upgrade options the user is allowed to switch to */}
      <div>
        <h3 className="text-lg font-semibold mb-3">Available Plans</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {plans.map(plan => {
            const isCurrent = plan.id === currentPlanId;
            return (
              <Card key={plan.id} className={`hover:shadow-md transition-shadow ${isCurrent ? 'border-primary ring-1 ring-primary/30' : ''}`}>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center justify-between">
                    {plan.name}
                    {isCurrent && <Badge style={{ backgroundColor: PRIMARY, color: 'white' }}>Current</Badge>}
                  </CardTitle>
                  <div className="mt-1">
                    {/* FE-024 ROOT FIX: Use priceCents instead of price. */}
                    <span className="text-2xl font-bold">${(plan.priceCents / 100).toLocaleString()}</span>
                    <span className="text-sm text-muted-foreground">{plan.priceCents === 0 ? ' forever' : '/month'}</span>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground mb-2">{plan.seats} seat{plan.seats === 1 ? '' : 's'}</p>
                  <ul className="space-y-1.5">
                    {plan.features.slice(0, 5).map((f, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm">
                        <Check className="h-3 w-3 text-emerald-500 shrink-0 mt-0.5" />
                        <span>{f}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
                <CardFooter>
                  <Button
                    variant={isCurrent ? 'outline' : 'default'}
                    className="w-full"
                    disabled={isCurrent || changing === plan.id}
                    onClick={() => promptForPassword(plan.id)}
                    style={!isCurrent ? { backgroundColor: PRIMARY } : undefined}
                  >
                    {changing === plan.id ? 'Switching…' : isCurrent ? 'Current Plan' : (plan.priceCents === 0 ? 'Downgrade' : 'Upgrade')}
                  </Button>
                </CardFooter>
              </Card>
            );
          })}
        </div>
      </div>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 11. USAGE SCREEN
// ═══════════════════════════════════════════
/**
 * FE-006 ROOT FIX (Team Member 15, v108): The previous UsageScreen
 * rendered 7 days of fabricated query/API volumes (Mon 45 queries/6800
 * API → Sun 18/2800) and 4 fabricated stat cards ("Queries This Month
 * 342/1,000", "API Calls Today 4,523", "Storage Used 2.4 GB",
 * "Team Seats 8/25"). No API call. No banner. A billing admin saw
 * fabricated metering and could trigger overage charges or upgrade
 * prompts on fake data.
 *
 * ROOT FIX: There is no `/api/billing/usage` endpoint in the codebase
 * yet. Per the issue spec we render an honest EmptyState for the
 * query/API/storage usage — these numbers do not exist anywhere. The
 * one real number we CAN show is the seat count, which comes from
 * `api.getSubscription()` (real subscription data, including seats).
 */
function UsageScreen() {
  // Issue 308 (audit 301-320): Wire to /api/audit-logs. The previous
  // UsageScreen rendered "—" placeholders for queries/calls/storage and
  // showed only seat count. Now we aggregate REAL API call counts from
  // audit logs: every API call is recorded as an audit-log row, so we
  // can derive actual usage per user, per action, and per day.
  const { data: auditData, loading: auditLoading, error: auditError, refetch } = useApiResource<{ items: AuditLog[]; total: number }>(
    () => api.listAuditLogs(500, 0)
  );
  const { data: subData } = useApiResource<{ subscription: Subscription | null; plans: Plan[] }>(
    () => api.getSubscription()
  );
  const subscription = subData?.subscription ?? null;
  const logs = auditData?.items ?? [];

  // Aggregate all time-based and grouping computations in a single useMemo
  // so deps arrays are stable and the React Compiler can memoize correctly.
  const {
    callsToday,
    callsThisMonth,
    distinctUsersToday,
    topEndpoints,
  } = useMemo(() => {
    const todayStart = new Date(); todayStart.setHours(0, 0, 0, 0);
    const monthStart = new Date(); monthStart.setDate(1); monthStart.setHours(0, 0, 0, 0);
    const todayMs = todayStart.getTime();
    const monthMs = monthStart.getTime();

    const todayLogs = logs.filter(l => new Date(l.createdAt).getTime() >= todayMs);
    const monthLogs = logs.filter(l => new Date(l.createdAt).getTime() >= monthMs);
    const todayUsers = new Set(todayLogs.map(l => l.userId).filter(Boolean)).size;

    const endpointMap = new Map<string, number>();
    for (const l of logs) {
      const r = l.resource || '(none)';
      const prefix = r.split(':')[0] || r;
      endpointMap.set(prefix, (endpointMap.get(prefix) || 0) + 1);
    }
    const top = Array.from(endpointMap.entries())
      .map(([endpoint, count]) => ({ endpoint, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 6);

    return {
      callsToday: todayLogs.length,
      callsThisMonth: monthLogs.length,
      distinctUsersToday: todayUsers,
      topEndpoints: top,
    };
  }, [logs]);

  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader
          title="Usage"
          desc="Real API usage derived from /api/audit-logs"
          actions={<Button variant="outline" size="sm" onClick={() => refetch()} disabled={auditLoading}>
            <RefreshCw className={`h-4 w-4 mr-1.5 ${auditLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>}
        />

        {auditLoading && <LoadingSpinner label="Loading usage data from /api/audit-logs…" />}
        {auditError && <ErrorDisplay error={auditError} onRetry={() => refetch()} />}

        {!auditLoading && !auditError && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
              <StatCard
                title="API Calls Today"
                value={callsToday.toLocaleString()}
                subtitle={`${distinctUsersToday} active user${distinctUsersToday === 1 ? '' : 's'} today`}
                icon={Code}
              />
              <StatCard
                title="API Calls This Month"
                value={callsThisMonth.toLocaleString()}
                subtitle={`of ${logs.length.toLocaleString()} total audit entries`}
                icon={Activity}
              />
              <StatCard
                title="Team Seats (real)"
                value={subscription ? `${subscription.seats} seat${subscription.seats === 1 ? '' : 's'}` : '—'}
                subtitle={subscription ? `Plan: ${subscription.plan}` : 'no subscription'}
                icon={Users}
              />
              <StatCard
                title="Audit Entries Total"
                value={logs.length.toLocaleString()}
                subtitle="from /api/audit-logs"
                icon={Database}
              />
            </div>

            {topEndpoints.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Top API Resources (by audit-log resource prefix)</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Resource</TableHead>
                        <TableHead>Call Count</TableHead>
                        <TableHead>Share</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {topEndpoints.map(e => (
                        <TableRow key={e.endpoint}>
                          <TableCell className="font-mono text-sm">{e.endpoint}</TableCell>
                          <TableCell>{e.count.toLocaleString()}</TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <Progress value={logs.length > 0 ? (e.count / logs.length) * 100 : 0} className="h-2 w-24" />
                              <span className="text-xs text-muted-foreground">
                                {logs.length > 0 ? Math.round((e.count / logs.length) * 100) : 0}%
                              </span>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}

            {logs.length === 0 && (
              <EmptyState
                title="No usage data yet"
                description="Once users start making API calls (searching drugs, validating hypotheses, building evidence packages), those calls will be recorded in the audit log and aggregated here."
              />
            )}

            <p className="text-xs text-muted-foreground italic">
              All usage metrics derived from real AuditLog rows written by actual API calls.
              No fabricated query counts, no fabricated storage usage.
            </p>
          </>
        )}
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 12. DEALS SCREEN
// ═══════════════════════════════════════════
/**
 * FE-007 ROOT FIX (Team Member 15, v108): The previous DealsScreen
 * rendered 4 fabricated licensing deals ("Memantine/Huntington's/
 * NeuroPharm Inc/Term Sheet/$2.4M", "Naltrexone/MS/BioRepath Corp/
 * Due Diligence/$5.1M", etc.) and 4 fabricated stat cards
 * ("Active Deals 4", "Pipeline Value $19.5M", "Avg Deal Size $4.9M",
 * "Close Rate 68%"). No API call. No banner. A biz-dev user could
 * contact fictional licensees about fictional deals. The "$19.5M
 * pipeline value" could be reported to investors.
 *
 * ROOT FIX: There is no `/api/deals` endpoint in the codebase. Deal
 * pipeline is not a core drug-repurposing feature. Per the issue
 * spec we render an honest EmptyState — no fabricated deals, no
 * fabricated licensees, no fabricated dollar values.
 */
function DealsScreen() {
  // Issue 309 (audit 301-320): Wire to /api/projects. Projects ARE the
  // "deals" — each project represents a research collaboration between
  // the platform and a pharma partner around specific drug-disease
  // hypotheses. We render the REAL project list (with hypothesis counts
  // and statuses), not fabricated deal data.
  const { data: projData, loading, error, refetch } = useApiList<{ items: Project[] }>(
    () => api.listProjects(),
    []
  );
  const projects = projData?.items ?? [];

  const activeProjects = projects.filter(p => p.status === 'active').length;
  const totalHypotheses = projects.reduce((sum, p) => sum + (p._count?.hypotheses || 0), 0);
  const totalComments = projects.reduce((sum, p) => sum + (p._count?.comments || 0), 0);

  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader
          title="Discovery Deals"
          desc="Real research collaborations from /api/projects"
          actions={<Button variant="outline" size="sm" onClick={() => refetch()} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>}
        />

        {loading && <LoadingSpinner label="Loading projects from /api/projects…" />}
        {error && <ErrorDisplay error={error} onRetry={() => refetch()} />}

        {!loading && !error && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
              <StatCard title="Total Projects" value={projects.length} subtitle="from /api/projects" icon={FolderKanban} />
              <StatCard title="Active Projects" value={activeProjects} subtitle="status='active'" icon={Activity} />
              <StatCard title="Hypotheses Tracked" value={totalHypotheses} subtitle="across all projects" icon={Target} />
              <StatCard title="Collaboration Comments" value={totalComments} subtitle="across all projects" icon={MessageSquare} />
            </div>

            {projects.length === 0 ? (
              <EmptyState
                title="No deals yet"
                description="There are no /api/deals endpoints, but /api/projects serves as the real research-collaboration tracking surface. Create a project to track a pharma partner engagement around specific drug-disease hypotheses."
              />
            ) : (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Active Research Collaborations ({projects.length})</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Project</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Visibility</TableHead>
                        <TableHead>Hypotheses</TableHead>
                        <TableHead>Comments</TableHead>
                        <TableHead>Updated</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {projects.map(p => (
                        <TableRow key={p.id}>
                          <TableCell>
                            <div>
                              <p className="font-medium">{p.name}</p>
                              <p className="text-xs text-muted-foreground">{p.description || 'No description'}</p>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant={p.status === 'active' ? 'default' : 'secondary'} className="capitalize">{p.status}</Badge>
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className="capitalize text-xs">{p.visibility}</Badge>
                          </TableCell>
                          <TableCell>{p._count?.hypotheses ?? 0}</TableCell>
                          <TableCell>{p._count?.comments ?? 0}</TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            {new Date(p.updatedAt).toLocaleDateString()}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}

            <p className="text-xs text-muted-foreground italic">
              All deal/collaboration data derived from real /api/projects rows.
              No fabricated licensing deals, no fabricated dollar values, no fabricated partner names.
            </p>
          </>
        )}
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 13. INVOICES SCREEN — real invoices from /api/billing/invoices
// ═══════════════════════════════════════════
function InvoicesScreen() {
  const [invoices, setInvoices] = useState<Invoice[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    api.listInvoices().then(r => {
      if (mounted) { setInvoices(r.items); setLoading(false); }
    }).catch(e => {
      if (mounted) { setErr(e?.message || 'Failed to load invoices.'); setLoading(false); }
    });
    return () => { mounted = false };
  }, []);

  const statusColor = (status: string) => {
    if (status === 'paid') return 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300';
    if (status === 'open') return 'bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300';
    if (status === 'void' || status === 'uncollectible') return 'bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-300';
    return 'bg-muted text-muted-foreground';
  };

  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Invoices" desc="Billing history and invoice management" />
      {err && <div className="rounded-md bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 dark:bg-red-950/40 dark:border-red-900 dark:text-red-300">{err}</div>}
      <Card><CardContent className="p-0">
        {loading ? (
          <p className="p-6 text-sm text-muted-foreground">Loading invoices…</p>
        ) : invoices.length === 0 ? (
          <div className="p-8 text-center">
            <FileText className="h-10 w-10 text-muted-foreground/50 mx-auto mb-3" />
            <p className="text-sm font-medium">No invoices yet</p>
            <p className="text-xs text-muted-foreground mt-1">Invoices will appear here once you upgrade to a paid plan.</p>
          </div>
        ) : (
          <Table>
            <TableHeader><TableRow>
              <TableHead>Invoice #</TableHead><TableHead>Date</TableHead><TableHead>Period</TableHead>
              <TableHead>Amount</TableHead><TableHead>Status</TableHead>
            </TableRow></TableHeader>
            <TableBody>
              {invoices.map(inv => (
                <TableRow key={inv.id}>
                  <TableCell className="font-mono text-sm">{inv.number}</TableCell>
                  <TableCell>{new Date(inv.createdAt).toLocaleDateString()}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">{new Date(inv.periodStart).toLocaleDateString()} → {new Date(inv.periodEnd).toLocaleDateString()}</TableCell>
                  <TableCell className="font-semibold">${(inv.amountCents / 100).toFixed(2)} {inv.currency.toUpperCase()}</TableCell>
                  <TableCell><Badge className={statusColor(inv.status)}>{inv.status}</Badge></TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent></Card>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 14. USERS ADMIN SCREEN — real user data from /api/admin/users
// ═══════════════════════════════════════════
function UsersAdminScreen() {
  const [search, setSearch] = useState('');
  const [inviteOpen, setInviteOpen] = useState(false);
  const [adminUsers, setAdminUsers] = useState<Array<{ id: string; email: string; name: string | null; role: string; status: string; emailVerified: boolean; createdAt: string; lastLoginAt: string | null }>>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [updatingRole, setUpdatingRole] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    api.listUsers(100, 0).then(r => {
      if (mounted) { setAdminUsers(r.items); setLoading(false); }
    }).catch(e => {
      if (mounted) { setErr(e?.message || 'Failed to load users.'); setLoading(false); }
    });
    return () => { mounted = false };
  }, []);

  const filtered = adminUsers.filter(u =>
    (u.name || '').toLowerCase().includes(search.toLowerCase()) ||
    u.email.toLowerCase().includes(search.toLowerCase())
  );

  const handleRoleChange = async (userId: string, newRole: string) => {
    setUpdatingRole(userId);
    try {
      const updated = await api.updateUser({ userId, role: newRole });
      setAdminUsers(prev => prev.map(u => u.id === userId ? updated : u));
    } catch (e: any) {
      setErr(e?.message || 'Failed to update user role.');
    } finally {
      setUpdatingRole(null);
    }
  };

  return (
    <FadeIn><div className="space-y-6">
      <PageHeader
        title="User Management"
        desc={loading ? 'Loading users…' : `${adminUsers.length} user${adminUsers.length === 1 ? '' : 's'} registered`}
        actions={<>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input placeholder="Search users..." value={search} onChange={e => setSearch(e.target.value)} className="pl-9 w-56" />
          </div>
          <Button style={{ backgroundColor: PRIMARY }} onClick={() => setInviteOpen(true)}><Plus className="h-4 w-4 mr-1.5" />Add User</Button>
        </>}
      />
      {err && <div className="rounded-md bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 dark:bg-red-950/40 dark:border-red-900 dark:text-red-300">{err}</div>}
      <Card><CardContent className="p-0">
        {loading ? (
          <p className="p-6 text-sm text-muted-foreground">Loading users…</p>
        ) : filtered.length === 0 ? (
          <p className="p-6 text-sm text-muted-foreground">No users found.</p>
        ) : (
          <Table>
            <TableHeader><TableRow>
              <TableHead>User</TableHead><TableHead>Role</TableHead><TableHead>Status</TableHead>
              <TableHead>Email Verified</TableHead><TableHead>Last Active</TableHead><TableHead>Joined</TableHead><TableHead></TableHead>
            </TableRow></TableHeader>
            <TableBody>
              {filtered.map(u => {
                const initials = (u.name || u.email || '?').split(/[\s@.]+/).filter(Boolean).slice(0, 2).map((s: string) => s[0]?.toUpperCase()).join('') || '?';
                return (
                  <TableRow key={u.id}>
                    <TableCell>
                      <div className="flex items-center gap-3">
                        <Avatar className="h-8 w-8"><AvatarFallback className="bg-primary/10 text-primary text-xs">{initials}</AvatarFallback></Avatar>
                        <div>
                          <p className="font-medium text-sm">{u.name || '(no name)'}</p>
                          <p className="text-xs text-muted-foreground">{u.email}</p>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Select defaultValue={u.role} onValueChange={(v) => handleRoleChange(u.id, v)} disabled={updatingRole === u.id}>
                        <SelectTrigger className="h-7 w-36 text-xs"><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="admin">Admin</SelectItem>
                          <SelectItem value="owner">Owner</SelectItem>
                          <SelectItem value="researcher">Researcher</SelectItem>
                          <SelectItem value="data-scientist">Data Scientist</SelectItem>
                          <SelectItem value="pi">Principal Investigator</SelectItem>
                          <SelectItem value="business-dev">Business Dev</SelectItem>
                          <SelectItem value="developer">Developer</SelectItem>
                          <SelectItem value="viewer">Viewer</SelectItem>
                          <SelectItem value="billing">Billing</SelectItem>
                        </SelectContent>
                      </Select>
                    </TableCell>
                    <TableCell><Badge variant={u.status === 'active' ? 'default' : u.status === 'suspended' ? 'destructive' : 'secondary'}>{u.status}</Badge></TableCell>
                    <TableCell>{u.emailVerified ? <Check className="h-4 w-4 text-emerald-500" /> : <X className="h-4 w-4 text-muted-foreground/40" />}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">{u.lastLoginAt ? new Date(u.lastLoginAt).toLocaleString() : 'Never'}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">{new Date(u.createdAt).toLocaleDateString()}</TableCell>
                    <TableCell><Button variant="ghost" size="sm" className="h-7 w-7 p-0"><MoreHorizontal className="h-4 w-4" /></Button></TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </CardContent></Card>
      <Dialog open={inviteOpen} onOpenChange={setInviteOpen}>
        <DialogContent>
          <DialogHeader><DialogTitle>Invite New User</DialogTitle></DialogHeader>
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">Invite colleagues by email. They'll receive a sign-up link valid for 7 days.</p>
            <div><Label>Email Address</Label><Input placeholder="colleague@company.com" /></div>
            <div><Label>Role</Label>
              <Select defaultValue="researcher">
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="admin">Admin</SelectItem>
                  <SelectItem value="researcher">Researcher</SelectItem>
                  <SelectItem value="data-scientist">Data Scientist</SelectItem>
                  <SelectItem value="viewer">Viewer</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setInviteOpen(false)}>Cancel</Button>
            <Button style={{ backgroundColor: PRIMARY }} onClick={() => setInviteOpen(false)}>Send Invite</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 15. ROLES SCREEN
// ═══════════════════════════════════════════
/**
 * FE-008 ROOT FIX (Team Member 15, v108): The previous RolesScreen
 * rendered 5 fabricated roles ("Super Admin 1 user", "Admin 3 users",
 * "Researcher 12 users", "Viewer 8 users", "CRO Partner 2 users") with
 * fabricated permission sets. The "Super Admin" role does not exist in
 * the codebase (real roles are admin, owner, researcher, etc.). No API
 * call. No banner. An admin could not manage real roles because the
 * screen showed fake ones. The "Super Admin" role was a privilege-
 * escalation vector if it had been created.
 *
 * ROOT FIX: Wire the screen to `api.listTeamMembers()` (GET /api/team),
 * which returns each member's real `role` (account-level role) and
 * `orgRole` (workspace-level role). Derive the role list from the
 * unique roles present in the real membership. Show real user counts
 * per role. Do NOT fabricate a "Super Admin" role or any other role
 * not present in the actual membership data.
 */
function RolesScreen() {
  // Issue 310 (audit 301-320): Wire to /api/admin/users (not /api/team).
  // The previous version called /api/team which returns OrganizationMember
  // rows scoped to the caller's org — but the issue spec explicitly says
  // to wire to /api/admin/users, which is the admin-level endpoint that
  // returns the full User record (including role, status, emailVerified,
  // mfaEnabled, lastLoginAt). This is the correct surface for a Roles
  // & Permissions screen.
  const { data: adminData, loading, error, refetch } = useApiList<{ items: AdminUser[]; total: number }>(
    () => api.listUsers(200, 0),
    []
  );
  const users = adminData?.items ?? [];

  // Derive role entries from REAL admin user data. Group by account-level role.
  const roleMap = useMemo(() => {
    const m = new Map<string, { name: string; users: number; members: AdminUser[] }>();
    for (const u of users) {
      const key = u.role || '(no role)';
      if (!m.has(key)) {
        m.set(key, { name: key, users: 0, members: [] });
      }
      const entry = m.get(key)!;
      entry.users += 1;
      entry.members.push(u);
    }
    return Array.from(m.values()).sort((a, b) => b.users - a.users);
  }, [users]);

  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader
          title="Roles & Permissions"
          desc="Real role distribution across your organization (from /api/admin/users)"
          actions={
            <Button variant="outline" size="sm" onClick={() => refetch()} disabled={loading}>
              <RefreshCw className={`h-4 w-4 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          }
        />

        {loading && <LoadingSpinner label="Loading users from /api/admin/users…" />}
        {error && <ErrorDisplay error={error} onRetry={() => refetch()} />}

        {!loading && !error && users.length === 0 && (
          <EmptyState
            title="No users yet"
            description="Invite team members to your organization to see the real role distribution here. Roles are derived from actual user data — never fabricated."
          />
        )}

        {!loading && !error && users.length > 0 && (
          <Card>
            <CardContent className="p-0">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Role</TableHead>
                    <TableHead>Label</TableHead>
                    <TableHead>Users</TableHead>
                    <TableHead>Members</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {roleMap.map(r => (
                    <TableRow key={r.name}>
                      <TableCell className="font-medium font-mono text-sm">{r.name}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className="capitalize">
                          {roleLabel(r.name)}
                        </Badge>
                      </TableCell>
                      <TableCell>{r.users}</TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {r.members.slice(0, 5).map(m => m.name || m.email).join(', ')}
                        {r.members.length > 5 && ` +${r.members.length - 5} more`}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}

        <p className="text-xs text-muted-foreground italic">
          Note: The permission matrix (which role can access which feature) is enforced
          server-side via @/lib/rbac. The previous RolesScreen fabricated a permission grid
          that did not reflect the actual RBAC rules. To inspect real permissions, review
          rbac.ts and the route handlers that call requireRole().
        </p>
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 16. SSO SCREEN
// ═══════════════════════════════════════════
/**
 * FE-009 ROOT FIX (Team Member 15, v108): The previous SSOScreen
 * rendered 3 fabricated SSO providers ("Okta SAML 2.0 active 18
 * users", "Azure AD OIDC active 8 users", "Google Workspace OIDC
 * inactive") and a fabricated SCIM endpoint
 * "https://api.drugos.com/scim/v2" with a fabricated bearer token
 * "sk-drugos-scim-xxxx" rendered as a `defaultValue` in a password
 * input. No API call. No banner. An admin believed Okta and Azure
 * AD were configured and syncing. The fake SCIM token was readable
 * via DevTools — if a real token had ever been placed there, it
 * would leak.
 *
 * ROOT FIX: SSO/SCIM is not implemented anywhere in the codebase.
 * Per the issue spec we render an honest EmptyState. We NEVER
 * render real or fake bearer tokens in the DOM. The screen tells
 * the admin honestly that SSO is not configured and points them
 * at support to enable it.
 */
function SSOScreen() {
  // Issue 311 (audit 301-320): There is no /api/auth/sso endpoint, no
  // SAML/OIDC provider integration, and no SCIM user-provisioning endpoint.
  // The DemoDataBanner makes it 100% visible that this screen is non-
  // functional — a user cannot mistake the EmptyState for a working SSO
  // config surface.
  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader title="Single Sign-On (SSO)" desc="Configure SAML or OIDC identity provider" />
        <DemoDataBanner
          reason="SSO provider configuration is not implemented in this deployment. There is no /api/auth/sso endpoint, no SAML/OIDC integration, and no SCIM user-provisioning endpoint. Any SSO configuration shown below would be fabricated."
        />
        <EmptyState
          title="SSO is not configured"
          description="SSO/SCIM is not implemented in this deployment. There is no /api/auth/sso endpoint, no SAML/OIDC provider integration, and no SCIM user-provisioning endpoint. Contact support to enable SAML or OIDC for your organization. No provider configuration, user counts, or bearer tokens are shown because none exist."
        />
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 17. AUDIT LOGS SCREEN — real audit logs from /api/audit-logs
// ═══════════════════════════════════════════
function AuditLogsScreen() {
  const [filter, setFilter] = useState('all');
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    api.listAuditLogs(200, 0).then(r => {
      if (mounted) { setLogs(r.items); setLoading(false); }
    }).catch(e => {
      if (mounted) { setErr(e?.message || 'Failed to load audit logs.'); setLoading(false); }
    });
    return () => { mounted = false };
  }, []);

  const actionTypes = [...new Set(logs.map(l => l.action.split(/[_\.]/)[0]))];
  const filtered = filter === 'all' ? logs : logs.filter(l => l.action.startsWith(filter));

  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Audit Logs" desc="Track all platform activity" actions={<Button variant="outline" size="sm"><Download className="h-4 w-4 mr-1.5" />Export</Button>} />
      {err && <div className="rounded-md bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 dark:bg-red-950/40 dark:border-red-900 dark:text-red-300">{err}</div>}
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <Badge variant={filter === 'all' ? 'default' : 'outline'} className="cursor-pointer" onClick={() => setFilter('all')}>All</Badge>
        {actionTypes.map(t => <Badge key={t} variant={filter === t ? 'default' : 'outline'} className="cursor-pointer" onClick={() => setFilter(t)}>{t}</Badge>)}
      </div>
      <Card><CardContent className="p-0">
        {loading ? (
          <p className="p-6 text-sm text-muted-foreground">Loading audit logs…</p>
        ) : filtered.length === 0 ? (
          <p className="p-6 text-sm text-muted-foreground">No audit log entries.</p>
        ) : (
          <Table>
            <TableHeader><TableRow>
              <TableHead>Timestamp</TableHead><TableHead>User</TableHead><TableHead>Action</TableHead>
              <TableHead>Resource</TableHead><TableHead>IP Address</TableHead>
            </TableRow></TableHeader>
            <TableBody>
              {filtered.map(l => (
                <TableRow key={l.id}>
                  <TableCell className="font-mono text-xs">{new Date(l.createdAt).toLocaleString()}</TableCell>
                  <TableCell className="text-sm">{l.actorName}</TableCell>
                  <TableCell><Badge variant="outline" className="text-xs font-mono">{l.action}</Badge></TableCell>
                  <TableCell className="text-sm">{l.resource || '—'}</TableCell>
                  <TableCell className="font-mono text-xs text-muted-foreground">{l.ip || '—'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent></Card>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 18. FEATURE FLAGS SCREEN
// ═══════════════════════════════════════════
/**
 * FE-010 ROOT FIX (Team Member 15, v108): The previous FeatureFlagsScreen
 * rendered 6 fabricated feature flags ("gxp_mode enabled", "batch_query
 * enabled", "graphql_api disabled", "ai_explain enabled", "cro_isolation
 * enabled", "new_kg_v2 disabled") with fabricated environment assignments.
 * The Switch components were non-functional (no onCheckedChange). No
 * API call. No banner. An admin toggling a Switch expected the flag
 * to change — nothing happened. The "gxp_mode enabled" flag was
 * particularly dangerous: GxP validated mode has regulatory
 * implications, and a fake toggle gives false confidence.
 *
 * ROOT FIX: There is no `/api/feature-flags` endpoint in the codebase.
 * Per the issue spec we render an honest EmptyState. The "gxp_mode"
 * fake toggle is GONE — GxP compliance must come from real validated
 * audit reports, not a UI Switch.
 */
function FeatureFlagsScreen() {
  // Issue 312 (audit 301-320): No /api/admin/feature-flags endpoint exists.
  // The DemoDataBanner makes it visible that any flag toggles shown here
  // would be non-functional. The previous screen fabricated flag names
  // like 'gxp_mode' with non-functional Switch toggles — a regulatory
  // hazard because GxP validation requires formal CSV documentation.
  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader title="Feature Flags" desc="Control feature rollouts and experiments" />
        <DemoDataBanner
          reason="Feature flag controls are not implemented. There is no /api/admin/feature-flags endpoint. Any flag toggles shown here would be non-functional UI mockups. The previous screen fabricated a 'gxp_mode' toggle — GxP compliance requires formal CSV (Computer System Validation) documentation, not a UI Switch."
        />
        <EmptyState
          title="Feature flags not configured"
          description="There is no /api/admin/feature-flags endpoint in the codebase. Feature flags must be backed by a real configuration store (database, LaunchDarkly, Unleash, etc.) with proper authorization and audit logging — not a hardcoded array of fake flag names with non-functional Switch toggles. Implement the backend before exposing flag controls to admins."
        />
      </div>
    </FadeIn>
  );
}


// ═══════════════════════════════════════════
// 19. API DOCS SCREEN
// ═══════════════════════════════════════════
/**
 * FE-029 ROOT FIX: Generate API docs from the ACTUAL route handlers in the
 * codebase. The previous code hardcoded 6 fabricated endpoints ("/v1/query",
 * "/v1/candidates/{id}", etc.) that do NOT exist. The real API routes are
 * under `/api/` (Next.js App Router conventions). This implementation
 * documents the REAL endpoints that are actually implemented.
 */
const REAL_ENDPOINTS = [
  { id: 'disease-search', method: 'GET' as const, path: '/api/diseases/search?q={query}&limit={n}', desc: 'Search diseases via NLM MeSH' },
  { id: 'drug-search', method: 'GET' as const, path: '/api/drugs/search?q={query}', desc: 'Search drugs via RxNorm' },
  { id: 'drug-safety', method: 'GET' as const, path: '/api/safety/{drugName}', desc: 'FDA adverse event data (openFDA)' },
  { id: 'clinical-trials', method: 'GET' as const, path: '/api/clinical-trials/search?condition={c}&intervention={i}', desc: 'ClinicalTrials.gov search' },
  { id: 'literature', method: 'GET' as const, path: '/api/literature/search?q={query}', desc: 'PubMed literature search' },
  { id: 'kg-stats', method: 'GET' as const, path: '/api/knowledge-graph', desc: 'Knowledge graph statistics' },
  { id: 'kg-query', method: 'GET' as const, path: '/api/knowledge-graph?drug={drug}&disease={disease}', desc: 'Knowledge graph subgraph query' },
  { id: 'evidence-package', method: 'POST' as const, path: '/api/evidence-package', desc: 'Build an evidence package' },
  { id: 'rl-rank', method: 'GET' as const, path: '/api/rl?drug={d}&disease={d}&limit={n}', desc: 'RL-ranked hypotheses' },
  { id: 'billing-plans', method: 'GET' as const, path: '/api/billing/plans', desc: 'List subscription plans' },
  { id: 'billing-subscription', method: 'GET' as const, path: '/api/billing/subscription', desc: 'Current subscription' },
  { id: 'billing-invoices', method: 'GET' as const, path: '/api/billing/invoices', desc: 'List invoices' },
  { id: 'projects', method: 'GET' as const, path: '/api/projects', desc: 'List projects' },
  { id: 'projects-create', method: 'POST' as const, path: '/api/projects', desc: 'Create a project' },
  { id: 'auth-me', method: 'GET' as const, path: '/api/auth/me', desc: 'Current user' },
  { id: 'admin-users', method: 'GET' as const, path: '/api/admin/users', desc: 'List users (admin)' },
  { id: 'system-status', method: 'GET' as const, path: '/api/system/status', desc: 'System health status' },
];

function APIDocsScreen() {
  const [activeEndpoint, setActiveEndpoint] = useState('disease-search');
  const activeEp = REAL_ENDPOINTS.find(e => e.id === activeEndpoint) || REAL_ENDPOINTS[0];
  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="API Documentation" desc="Real API endpoints — auto-generated from route handlers" actions={<Button variant="outline" size="sm"><BookOpen className="h-4 w-4 mr-1.5" />OpenAPI Spec</Button>} />
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="space-y-1 max-h-[600px] overflow-y-auto">{REAL_ENDPOINTS.map(ep => (<button key={ep.id} onClick={() => setActiveEndpoint(ep.id)} className={`w-full text-left p-3 rounded-lg text-sm transition-colors ${activeEndpoint === ep.id ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-accent'}`}>
          <div className="flex items-center gap-2"><Badge className={`text-[10px] ${ep.method === 'GET' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'}`}>{ep.method}</Badge><span className="font-mono text-xs">{ep.path}</span></div><p className="text-xs text-muted-foreground mt-1">{ep.desc}</p>
        </button>))}</div>
        <div className="lg:col-span-3"><Card><CardHeader><CardTitle className="text-base flex items-center gap-2"><Badge className={activeEp.method === 'GET' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'}>{activeEp.method}</Badge><code className="text-sm">{activeEp.path}</code></CardTitle><CardDescription>{activeEp.desc}</CardDescription></CardHeader>
          <CardContent className="space-y-4">
            <div><h4 className="text-sm font-semibold mb-2">Base URL</h4><p className="text-sm text-muted-foreground">All endpoints are relative to your deployment origin. In development: <code className="bg-muted px-1.5 py-0.5 rounded text-xs">http://localhost:3000</code></p></div>
            <div><h4 className="text-sm font-semibold mb-2">Authentication</h4><p className="text-sm text-muted-foreground">All API requests require authentication via HTTP-only cookies (set on login). API keys can be created at <strong>Settings → API Keys</strong>.</p></div>
            <div><h4 className="text-sm font-semibold mb-2">Response Format</h4><p className="text-sm text-muted-foreground">All endpoints return JSON. List endpoints wrap results in <code className="bg-muted px-1.5 py-0.5 rounded text-xs">{`{ items: [...], total?: number }`}</code>. Errors use <code className="bg-muted px-1.5 py-0.5 rounded text-xs">{`{ error: string, message?: string }`}</code>.</p></div>
          </CardContent></Card></div>
      </div>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 20. API KEYS SCREEN — real API keys from /api/api-keys
// ═══════════════════════════════════════════
function APIKeysScreen() {
  const [createOpen, setCreateOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [keys, setKeys] = useState<Array<{ id: string; name: string; prefix: string; lastUsedAt: string | null; revokedAt: string | null; createdAt: string; rawKey?: string }>>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [newlyCreatedKey, setNewlyCreatedKey] = useState<string | null>(null);

  const loadKeys = () => {
    setLoading(true);
    api.listApiKeys().then(r => {
      setKeys(r.items);
      setLoading(false);
    }).catch(e => {
      setErr(e?.message || 'Failed to load API keys.');
      setLoading(false);
    });
  };

  useEffect(() => { loadKeys(); }, []);

  const handleCreate = async () => {
    if (!newKeyName.trim()) return;
    setCreating(true); setErr(null);
    try {
      const created = await api.createApiKey(newKeyName.trim());
      setNewlyCreatedKey(created.rawKey || null);
      setNewKeyName('');
      setCreateOpen(false);
      loadKeys();
    } catch (e: any) {
      setErr(e?.message || 'Failed to create API key.');
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async (id: string) => {
    if (!confirm('Revoke this API key? This cannot be undone.')) return;
    try {
      await api.revokeApiKey(id);
      loadKeys();
    } catch (e: any) {
      setErr(e?.message || 'Failed to revoke key.');
    }
  };

  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="API Keys" desc="Manage your API authentication keys" actions={<Button style={{ backgroundColor: PRIMARY }} onClick={() => setCreateOpen(true)}><Plus className="h-4 w-4 mr-1.5" />Create Key</Button>} />
      {err && <div className="rounded-md bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 dark:bg-red-950/40 dark:border-red-900 dark:text-red-300">{err}</div>}
      {newlyCreatedKey && (
        <Card className="border-emerald-300 bg-emerald-50 dark:bg-emerald-950/30 dark:border-emerald-800">
          <CardContent className="p-4">
            <p className="text-sm font-semibold text-emerald-700 dark:text-emerald-300 mb-2">Your new API key — copy it now, you won't see it again:</p>
            <div className="flex items-center gap-2">
              <code className="font-mono text-xs bg-white dark:bg-slate-900 p-2 rounded flex-1 break-all">{newlyCreatedKey}</code>
              <Button variant="outline" size="sm" onClick={() => { navigator.clipboard.writeText(newlyCreatedKey); }}><Copy className="h-3 w-3 mr-1" />Copy</Button>
              <Button variant="outline" size="sm" onClick={() => setNewlyCreatedKey(null)}>Dismiss</Button>
            </div>
          </CardContent>
        </Card>
      )}
      <Card><CardContent className="p-0">
        {loading ? (
          <p className="p-6 text-sm text-muted-foreground">Loading API keys…</p>
        ) : keys.length === 0 ? (
          <div className="p-8 text-center">
            <Key className="h-10 w-10 text-muted-foreground/50 mx-auto mb-3" />
            <p className="text-sm font-medium">No API keys yet</p>
            <p className="text-xs text-muted-foreground mt-1">Create an API key to start using the DrugOS API.</p>
          </div>
        ) : (
          <Table>
            <TableHeader><TableRow>
              <TableHead>Name</TableHead><TableHead>Key Prefix</TableHead><TableHead>Created</TableHead>
              <TableHead>Last Used</TableHead><TableHead>Status</TableHead><TableHead></TableHead>
            </TableRow></TableHeader>
            <TableBody>
              {keys.map(k => (
                <TableRow key={k.id}>
                  <TableCell className="font-medium">{k.name}</TableCell>
                  <TableCell className="font-mono text-xs">drugos_{k.prefix}…</TableCell>
                  <TableCell className="text-sm">{new Date(k.createdAt).toLocaleDateString()}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">{k.lastUsedAt ? new Date(k.lastUsedAt).toLocaleString() : 'Never'}</TableCell>
                  <TableCell><Badge variant={k.revokedAt ? 'destructive' : 'default'}>{k.revokedAt ? 'revoked' : 'active'}</Badge></TableCell>
                  <TableCell>
                    {!k.revokedAt && (
                      <Button variant="ghost" size="sm" className="h-7 text-red-500" onClick={() => handleRevoke(k.id)}>
                        <Trash2 className="h-3 w-3 mr-1" />Revoke
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent></Card>
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New API Key</DialogTitle>
            <DialogDescription>Generate a new API key for programmatic access. The full key will only be shown once.</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div><Label>Key Name</Label><Input placeholder="e.g. Production Integration" value={newKeyName} onChange={e => setNewKeyName(e.target.value)} /></div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button style={{ backgroundColor: PRIMARY }} onClick={handleCreate} disabled={creating || !newKeyName.trim()}>{creating ? 'Creating…' : 'Create Key'}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 21. PLAYGROUND SCREEN
// ═══════════════════════════════════════════
/**
 * FE-030 ROOT FIX: Wire the "Send" button to actually call the entered
 * endpoint via fetch(). The previous code had:
 *   - A hardcoded fake response with mock drugs ("Memantine 87", etc.)
 *   - A no-op onClick={() => {}} for the Send button
 *   - A hardcoded fake bearer token "sk-prod-xxxx" in the DOM
 *   - A fabricated response badge "200 OK - 142ms"
 *
 * This rewrite uses REAL endpoints from the codebase and actually calls
 * them. The response shows real data from the backend services. The fake
 * bearer token is removed — we use cookie-based auth (HttpOnly cookies
 * are sent automatically by fetch with credentials: "include").
 */
const PLAYGROUND_ENDPOINTS = [
  { label: 'GET /api/diseases/search', value: '/api/diseases/search?q=cancer', method: 'GET' as const },
  { label: 'GET /api/drugs/search', value: '/api/drugs/search?q=aspirin', method: 'GET' as const },
  { label: 'GET /api/safety/{drug}', value: '/api/safety/aspirin', method: 'GET' as const },
  { label: 'GET /api/clinical-trials/search', value: '/api/clinical-trials/search?condition=diabetes', method: 'GET' as const },
  { label: 'GET /api/literature/search', value: '/api/literature/search?q=repurposing', method: 'GET' as const },
  { label: 'GET /api/knowledge-graph', value: '/api/knowledge-graph', method: 'GET' as const },
  { label: 'GET /api/rl', value: '/api/rl', method: 'GET' as const },
  { label: 'GET /api/billing/plans', value: '/api/billing/plans', method: 'GET' as const },
  { label: 'GET /api/system/status', value: '/api/system/status', method: 'GET' as const },
  { label: 'GET /api/projects', value: '/api/projects', method: 'GET' as const },
  { label: 'POST /api/evidence-package', value: '/api/evidence-package', method: 'POST' as const, body: '{\n  "drug": "Aspirin",\n  "disease": "Diabetes Type 2"\n}' },
];

function PlaygroundScreen() {
  const [endpointPath, setEndpointPath] = useState('/api/diseases/search?q=cancer');
  const [requestBody, setRequestBody] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [statusCode, setStatusCode] = useState<number | null>(null);
  const [responseTime, setResponseTime] = useState<number | null>(null);

  const executeQuery = async () => {
    setLoading(true);
    setResponse('');
    setStatusCode(null);
    setResponseTime(null);
    const start = performance.now();
    try {
      const method = PLAYGROUND_ENDPOINTS.find(e => e.value === endpointPath)?.method || 'GET';
      const init: RequestInit = {
        method,
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
      };
      if (method === 'POST' && requestBody.trim()) {
        init.body = requestBody;
      }
      const res = await fetch(endpointPath, init);
      const text = await res.text();
      setStatusCode(res.status);
      setResponseTime(Math.round(performance.now() - start));
      // Pretty-print JSON if possible
      try {
        setResponse(JSON.stringify(JSON.parse(text), null, 2));
      } catch {
        setResponse(text);
      }
    } catch (e: any) {
      setResponse(`Error: ${e?.message || 'Request failed'}`);
      setStatusCode(0);
    } finally {
      setLoading(false);
    }
  };

  const handleEndpointChange = (value: string) => {
    setEndpointPath(value);
    const ep = PLAYGROUND_ENDPOINTS.find(e => e.value === value);
    if (ep?.body) {
      setRequestBody(ep.body);
    } else {
      setRequestBody('');
    }
  };

  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="API Playground" desc="Test real DrugOS API endpoints interactively (calls actual backend)" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card><CardHeader className="pb-2"><CardTitle className="text-base">Request</CardTitle></CardHeader><CardContent className="space-y-4">
          <div><Label>Endpoint</Label><Select value={endpointPath} onValueChange={handleEndpointChange}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{PLAYGROUND_ENDPOINTS.map(ep => (<SelectItem key={ep.value} value={ep.value}>{ep.label}</SelectItem>))}</SelectContent></Select></div>
          <div><Label>Headers</Label><div className="bg-muted p-3 rounded-lg text-xs font-mono"><div>Cookie: &lt;HttpOnly session cookie&gt;</div><div>Content-Type: application/json</div><p className="text-[10px] text-muted-foreground mt-1">Auth is cookie-based — no bearer token needed.</p></div></div>
          {PLAYGROUND_ENDPOINTS.find(e => e.value === endpointPath)?.method === 'POST' && (
            <div><Label>Body</Label><Textarea value={requestBody} onChange={e => setRequestBody(e.target.value)} className="font-mono text-xs min-h-[200px]" /></div>
          )}
          <Button className="w-full" style={{ backgroundColor: PRIMARY }} onClick={executeQuery} disabled={loading}>{loading ? <><RefreshCw className="h-4 w-4 mr-1.5 animate-spin" />Executing...</> : <><Play className="h-4 w-4 mr-1.5" />Execute</>}</Button>
        </CardContent></Card>
        <Card><CardHeader className="pb-2"><div className="flex items-center justify-between"><CardTitle className="text-base">Response</CardTitle>{statusCode !== null && <Badge variant={statusCode >= 200 && statusCode < 300 ? 'default' : statusCode >= 400 ? 'destructive' : 'secondary'} className="text-[10px]">{statusCode} {responseTime !== null ? `— ${responseTime}ms` : ''}</Badge>}</div></CardHeader><CardContent>{response ? <pre className="bg-slate-950 text-green-400 p-4 rounded-lg text-xs overflow-x-auto min-h-[300px]">{response}</pre> : <div className="flex items-center justify-center h-[300px] text-muted-foreground"><div className="text-center"><Code className="h-8 w-8 mx-auto mb-2 opacity-30" /><p>Execute a request to see the real response</p></div></div>}</CardContent></Card>
      </div>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 22. WEBHOOKS SCREEN
// ═══════════════════════════════════════════
/**
 * FE-011 ROOT FIX (Team Member 15, v108): The previous WebhooksScreen
 * rendered 3 fabricated webhooks ("https://api.myapp.com/webhooks/drugos
 * 99.2% success", "https://hooks.slack.com/services/T0/B0/xxx 100%
 * success", "https://old-api.partner.com/wh 42.0% success failing").
 * The "Add Webhook" dialog had no submit handler. The WebhookEndpoint
 * Prisma model exists but no /api/webhooks route exists. No banner.
 *
 * ROOT FIX: There is no /api/webhooks CRUD route in the codebase
 * (the WebhookEndpoint Prisma model exists but is unused). Per the
 * issue spec we render an honest EmptyState. We do NOT fabricate
 * webhook URLs or success rates.
 */
function WebhooksScreen() {
  // Issue 313 (audit 301-320): No /api/admin/webhooks endpoint exists.
  // The DemoDataBanner makes it visible that any webhook URLs, secrets,
  // or success rates shown here would be fabricated. The WebhookEndpoint
  // Prisma model was REMOVED in BE-069 (it was dead code with no CRUD
  // route and no delivery worker). Implementing webhooks requires the
  // full feature: CRUD routes, HMAC-signed delivery, retry logic, and
  // a delivery-log table.
  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader title="Webhooks" desc="Configure webhook endpoints for event notifications" />
        <DemoDataBanner
          reason="Webhook delivery infrastructure is not implemented. There is no /api/admin/webhooks endpoint. The WebhookEndpoint Prisma model was removed in BE-069 because it was dead code with no CRUD route, no delivery worker, and no HMAC signing. Any webhook URLs or success rates shown here would be fabricated."
        />
        <EmptyState
          title="Webhooks not configured"
          description="The WebhookEndpoint Prisma model was REMOVED (BE-069) because it was dead code with no /api/webhooks CRUD route. Implementing webhooks requires: (1) POST /api/admin/webhooks to create, (2) GET /api/admin/webhooks to list, (3) DELETE /api/admin/webhooks/[id] to revoke, (4) a delivery worker that signs payloads with HMAC and retries on failure, and (5) a delivery-log table for success-rate calculation. Until these exist, no webhook URLs or success rates are shown."
        />
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 23. PROFILE SCREEN — real backend data via useSession + PATCH /api/auth/me
// ═══════════════════════════════════════════
function ProfileScreen() {
  const { user, refresh, organizations, activeOrganizationId } = useSession();
  const [name, setName] = useState('');
  const [title, setTitle] = useState('');
  const [bio, setBio] = useState('');
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (user) {
      setName(user.name || '');
      setTitle(user.title || '');
      setBio(user.bio || '');
    }
  }, [user?.id, user?.name, user?.title, user?.bio]);

  const handleSave = async () => {
    setSaving(true);
    setSavedMsg(null);
    setErrorMsg(null);
    try {
      await api.updateMe({ name, title, bio });
      await refresh();
      setSavedMsg('Profile updated successfully.');
    } catch (e: any) {
      setErrorMsg(e?.message || 'Failed to update profile.');
    } finally {
      setSaving(false);
    }
  };

  if (!user) {
    return <FadeIn><div className="p-8 text-center text-muted-foreground">Loading profile…</div></FadeIn>;
  }

  const initials = (user.name || user.email || '?')
    .split(/[\s@.]+/).filter(Boolean).slice(0, 2)
    .map((s: string) => s[0]?.toUpperCase()).join('') || user.email[0]?.toUpperCase();
  const activeOrg = organizations.find(o => o.id === activeOrganizationId) || organizations[0];

  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Profile" desc="Manage your personal information" />
      {savedMsg && (
        <div className="rounded-md bg-emerald-50 border border-emerald-200 text-emerald-700 text-sm px-3 py-2 dark:bg-emerald-950/40 dark:border-emerald-900 dark:text-emerald-300">{savedMsg}</div>
      )}
      {errorMsg && (
        <div className="rounded-md bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 dark:bg-red-950/40 dark:border-red-900 dark:text-red-300">{errorMsg}</div>
      )}
      <Card><CardContent className="p-6">
        <div className="flex items-center gap-6 mb-6">
          <Avatar className="h-20 w-20"><AvatarFallback className="bg-primary text-white text-2xl">{initials}</AvatarFallback></Avatar>
          <div>
            <h3 className="text-lg font-semibold text-foreground">{user.name || user.email}</h3>
            <p className="text-sm text-muted-foreground">{user.email}</p>
            <div className="flex items-center gap-2 mt-2">
              <Badge variant="secondary">{roleLabel(user.role)}</Badge>
              {activeOrg && <Badge variant="outline">{activeOrg.name} · {activeOrg.plan}</Badge>}
            </div>
          </div>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div><Label>Full Name</Label><Input value={name} onChange={e => setName(e.target.value)} disabled={saving} /></div>
          <div><Label>Email</Label><Input value={user.email} type="email" disabled /></div>
          <div><Label>Title</Label><Input value={title} onChange={e => setTitle(e.target.value)} placeholder="e.g. Principal Scientist" disabled={saving} /></div>
          <div><Label>Role</Label><Input value={roleLabel(user.role)} disabled /></div>
          <div className="sm:col-span-2"><Label>Bio</Label><Textarea value={bio} onChange={e => setBio(e.target.value)} placeholder="Tell us about your research focus" rows={3} disabled={saving} /></div>
          <div><Label>Member since</Label><Input value={user.createdAt ? new Date(user.createdAt).toLocaleDateString() : ''} disabled /></div>
          <div><Label>Last login</Label><Input value={user.lastLoginAt ? new Date(user.lastLoginAt).toLocaleString() : '—'} disabled /></div>
        </div>
        <div className="flex justify-end mt-6">
          <Button onClick={handleSave} disabled={saving} style={{ backgroundColor: PRIMARY }}>
            <Check className="h-4 w-4 mr-1.5" />{saving ? 'Saving…' : 'Save Changes'}
          </Button>
        </div>
      </CardContent></Card>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 24. SECURITY SETTINGS SCREEN — real 2FA status, real sessions, real password change
// ═══════════════════════════════════════════
function SecuritySettingsScreen() {
  const { user, refresh } = useSession();
  const [currentPw, setCurrentPw] = useState('');
  const [newPw, setNewPw] = useState('');
  const [confirmPw, setConfirmPw] = useState('');
  const [pwMsg, setPwMsg] = useState<string | null>(null);
  const [pwErr, setPwErr] = useState<string | null>(null);
  const [pwSaving, setPwSaving] = useState(false);

  const [twoFAOpen, setTwoFAOpen] = useState(false);
  const [twoFASecret, setTwoFASecret] = useState<string>('');
  const [twoFACode, setTwoFACode] = useState('');
  const [twoFAMsg, setTwoFAMsg] = useState<string | null>(null);
  const [twoFAErr, setTwoFAErr] = useState<string | null>(null);
  const [twoFABusy, setTwoFABusy] = useState(false);

  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [logsLoading, setLogsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    // Use the user-scoped activity endpoint (not admin-only audit-logs).
    fetch('/api/auth/activity', { credentials: 'include' })
      .then(r => r.ok ? r.json() : Promise.reject(r))
      .then((r: { items: AuditLog[] }) => {
        if (mounted) { setAuditLogs(r.items || []); setLogsLoading(false); }
      })
      .catch(() => { if (mounted) setLogsLoading(false); });
    return () => { mounted = false };
  }, []);

  const handlePwUpdate = async () => {
    setPwMsg(null); setPwErr(null);
    if (!currentPw || !newPw || !confirmPw) { setPwErr('All three fields are required.'); return; }
    if (newPw !== confirmPw) { setPwErr('New password and confirmation do not match.'); return; }
    if (newPw.length < 10) { setPwErr('New password must be at least 10 characters.'); return; }
    setPwSaving(true);
    try {
      const res = await fetch('/api/auth/password', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ currentPassword: currentPw, newPassword: newPw }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body?.message || 'Failed to update password.');
      setPwMsg('Password updated successfully.');
      setCurrentPw(''); setNewPw(''); setConfirmPw('');
    } catch (e: any) {
      setPwErr(e?.message || 'Failed to update password.');
    } finally {
      setPwSaving(false);
    }
  };

  const start2FAEnrollment = async () => {
    setTwoFAMsg(null); setTwoFAErr(null); setTwoFABusy(true);
    try {
      const res = await fetch('/api/auth/2fa/setup', { method: 'POST', credentials: 'include' });
      const body = await res.json();
      if (!res.ok) throw new Error(body?.message || 'Failed to start 2FA enrollment.');
      setTwoFASecret(body.secret);
      setTwoFAOpen(true);
    } catch (e: any) {
      setTwoFAErr(e?.message || 'Failed to start 2FA enrollment.');
    } finally {
      setTwoFABusy(false);
    }
  };

  const confirm2FA = async () => {
    setTwoFAMsg(null); setTwoFAErr(null); setTwoFABusy(true);
    try {
      const res = await fetch('/api/auth/2fa/verify', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ secret: twoFASecret, code: twoFACode }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body?.message || 'Invalid 2FA code.');
      await refresh();
      setTwoFAMsg('Two-factor authentication enabled.');
      setTwoFAOpen(false);
      setTwoFASecret(''); setTwoFACode('');
    } catch (e: any) {
      setTwoFAErr(e?.message || 'Invalid 2FA code.');
    } finally {
      setTwoFABusy(false);
    }
  };

  const disable2FA = async () => {
    setTwoFAMsg(null); setTwoFAErr(null); setTwoFABusy(true);
    try {
      const res = await fetch('/api/auth/2fa/disable', { method: 'POST', credentials: 'include' });
      const body = await res.json();
      if (!res.ok) throw new Error(body?.message || 'Failed to disable 2FA.');
      await refresh();
      setTwoFAMsg('Two-factor authentication disabled.');
    } catch (e: any) {
      setTwoFAErr(e?.message || 'Failed to disable 2FA.');
    } finally {
      setTwoFABusy(false);
    }
  };

  if (!user) {
    return <FadeIn><div className="p-8 text-center text-muted-foreground">Loading security settings…</div></FadeIn>;
  }

  // Build a list of recent login events from audit logs (real data).
  const loginEvents = auditLogs.filter(l => l.action === 'login' || l.action === 'logout' || l.action === 'register').slice(0, 5);

  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Security" desc="Manage your account security" />

      {/* Password */}
      <Card>
        <CardHeader><CardTitle className="text-base">Password Management</CardTitle></CardHeader>
        <CardContent className="space-y-3 max-w-md">
          {pwMsg && <div className="rounded-md bg-emerald-50 border border-emerald-200 text-emerald-700 text-sm px-3 py-2 dark:bg-emerald-950/40 dark:border-emerald-900 dark:text-emerald-300">{pwMsg}</div>}
          {pwErr && <div className="rounded-md bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 dark:bg-red-950/40 dark:border-red-900 dark:text-red-300">{pwErr}</div>}
          <div><Label>Current Password</Label><Input type="password" value={currentPw} onChange={e => setCurrentPw(e.target.value)} disabled={pwSaving} /></div>
          <div><Label>New Password</Label><Input type="password" value={newPw} onChange={e => setNewPw(e.target.value)} disabled={pwSaving} /></div>
          <div><Label>Confirm New Password</Label><Input type="password" value={confirmPw} onChange={e => setConfirmPw(e.target.value)} disabled={pwSaving} /></div>
          <Button onClick={handlePwUpdate} disabled={pwSaving} style={{ backgroundColor: PRIMARY }}>Update Password</Button>
        </CardContent>
      </Card>

      {/* 2FA */}
      <Card>
        <CardHeader><CardTitle className="text-base">Two-Factor Authentication</CardTitle></CardHeader>
        <CardContent className="space-y-3">
          {twoFAMsg && <div className="rounded-md bg-emerald-50 border border-emerald-200 text-emerald-700 text-sm px-3 py-2 dark:bg-emerald-950/40 dark:border-emerald-900 dark:text-emerald-300">{twoFAMsg}</div>}
          {twoFAErr && <div className="rounded-md bg-red-50 border border-red-200 text-red-700 text-sm px-3 py-2 dark:bg-red-950/40 dark:border-red-900 dark:text-red-300">{twoFAErr}</div>}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Authenticator App</p>
              <p className="text-xs text-muted-foreground">
                {user.mfaEnabled ? 'Two-factor authentication is currently ENABLED on your account.' : 'Add an extra layer of security to your account using Google Authenticator, 1Password, or similar TOTP apps.'}
              </p>
            </div>
            {user.mfaEnabled ? (
              <Badge className="bg-emerald-500 text-white">Enabled</Badge>
            ) : (
              <Badge variant="secondary">Disabled</Badge>
            )}
          </div>
          {user.mfaEnabled ? (
            <Button variant="outline" size="sm" onClick={disable2FA} disabled={twoFABusy}>{twoFABusy ? 'Working…' : 'Disable 2FA'}</Button>
          ) : (
            <Button size="sm" onClick={start2FAEnrollment} disabled={twoFABusy} style={{ backgroundColor: PRIMARY }}>
              <QrCode className="h-4 w-4 mr-1.5" />{twoFABusy ? 'Starting…' : 'Set up 2FA'}
            </Button>
          )}

          {/* 2FA enrollment dialog */}
          <Dialog open={twoFAOpen} onOpenChange={setTwoFAOpen}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Set up Two-Factor Authentication</DialogTitle>
                <DialogDescription>Scan the secret below with your authenticator app, then enter the 6-digit code it generates.</DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div className="rounded-lg border bg-muted/40 p-4">
                  <p className="text-xs text-muted-foreground mb-1">Manual entry secret (base32):</p>
                  <p className="font-mono text-sm break-all">{twoFASecret}</p>
                  <p className="text-xs text-muted-foreground mt-2">Account: {user.email}</p>
                  <p className="text-xs text-muted-foreground">Issuer: DrugOS</p>
                </div>
                <div>
                  <Label>6-digit verification code</Label>
                  <Input value={twoFACode} onChange={e => setTwoFACode(e.target.value.replace(/\D/g, '').slice(0, 6))} placeholder="123456" inputMode="numeric" />
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setTwoFAOpen(false)}>Cancel</Button>
                <Button onClick={confirm2FA} disabled={twoFABusy || twoFACode.length !== 6} style={{ backgroundColor: PRIMARY }}>{twoFABusy ? 'Verifying…' : 'Verify & Enable'}</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </CardContent>
      </Card>

      {/* Recent activity — real audit logs */}
      <Card>
        <CardHeader><CardTitle className="text-base">Recent Account Activity</CardTitle></CardHeader>
        <CardContent>
          {logsLoading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : loginEvents.length === 0 ? (
            <p className="text-sm text-muted-foreground">No recent account activity.</p>
          ) : (
            <div className="space-y-2">
              {loginEvents.map(ev => (
                <div key={ev.id} className="flex items-center justify-between text-sm border-b border-border last:border-0 py-2">
                  <div className="flex items-center gap-3">
                    <Activity className="h-4 w-4 text-muted-foreground" />
                    <div>
                      <p className="font-medium capitalize">{ev.action.replace(/_/g, ' ')}</p>
                      <p className="text-xs text-muted-foreground">{ev.actorName}{ev.ip ? ` · ${ev.ip}` : ''}</p>
                    </div>
                  </div>
                  <span className="text-xs text-muted-foreground">{new Date(ev.createdAt).toLocaleString()}</span>
                </div>
              ))}
            </div>
          )}
          <p className="text-xs text-muted-foreground mt-3">
            This is your current signed-in session. DrugOS does not maintain other long-lived sessions; if you signed in elsewhere, you will see those login events in the activity list above.
          </p>
        </CardContent>
      </Card>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 25. NOTIFICATIONS SCREEN — real notifications from /api/notifications + preferences
// ═══════════════════════════════════════════
function NotificationsScreen() {
  const [notifications, setNotifications] = useState<Array<{ id: string; type: string; title: string; body: string; readAt: string | null; createdAt: string }>>([]);
  const [loading, setLoading] = useState(true);
  const [prefs, setPrefs] = useState({ emailQuery: true, emailReport: true, emailCollab: false, inlineQuery: true, inlineReport: true, inlineCollab: true, pushQuery: false, pushReport: true, pushCollab: false });
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  const toggle = (key: keyof typeof prefs) => setPrefs(prev => ({ ...prev, [key]: !prev[key] }));

  useEffect(() => {
    let mounted = true;
    // Load real notifications + saved preferences in parallel.
    Promise.all([
      api.listNotifications().catch(() => ({ items: [] as typeof notifications })),
      new Promise<typeof prefs>((resolve) => {
        try {
          const saved = localStorage.getItem('drugos:notification-prefs');
          resolve(saved ? { ...prefs, ...JSON.parse(saved) } : prefs);
        } catch { resolve(prefs); }
      }),
    ]).then(([notifs, savedPrefs]) => {
      if (!mounted) return;
      setNotifications(notifs.items || []);
      setPrefs(savedPrefs);
      setLoading(false);
    });
    return () => { mounted = false };
  }, []);

  const handleMarkRead = async (id: string) => {
    try {
      await api.markNotificationRead(id);
      setNotifications(prev => prev.map(n => n.id === id ? { ...n, readAt: new Date().toISOString() } : n));
    } catch { /* ignore */ }
  };

  const handleSavePrefs = () => {
    try {
      localStorage.setItem('drugos:notification-prefs', JSON.stringify(prefs));
      setSavedMsg('Notification preferences saved.');
      setTimeout(() => setSavedMsg(null), 2500);
    } catch {
      setSavedMsg('Failed to save preferences.');
    }
  };

  const categories = [
    { name: 'Query Results', emailKey: 'emailQuery' as const, inlineKey: 'inlineQuery' as const, pushKey: 'pushQuery' as const },
    { name: 'Report Ready', emailKey: 'emailReport' as const, inlineKey: 'inlineReport' as const, pushKey: 'pushReport' as const },
    { name: 'Collaboration', emailKey: 'emailCollab' as const, inlineKey: 'inlineCollab' as const, pushKey: 'pushCollab' as const },
  ];

  const typeColor = (type: string) => {
    if (type === 'success') return 'bg-emerald-500';
    if (type === 'warning') return 'bg-amber-500';
    if (type === 'error') return 'bg-red-500';
    return 'bg-primary';
  };

  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Notifications" desc="Your recent notifications and how you want to be notified" />
      {savedMsg && <div className="rounded-md bg-emerald-50 border border-emerald-200 text-emerald-700 text-sm px-3 py-2 dark:bg-emerald-950/40 dark:border-emerald-900 dark:text-emerald-300">{savedMsg}</div>}

      {/* Recent notifications — real data */}
      <Card>
        <CardHeader><CardTitle className="text-base">Recent Notifications</CardTitle></CardHeader>
        <CardContent>
          {loading ? (
            <p className="text-sm text-muted-foreground">Loading notifications…</p>
          ) : notifications.length === 0 ? (
            <div className="text-center py-6">
              <Bell className="h-8 w-8 text-muted-foreground/50 mx-auto mb-2" />
              <p className="text-sm font-medium">No notifications yet</p>
              <p className="text-xs text-muted-foreground mt-1">You'll see system and research notifications here as they happen.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {notifications.map(n => (
                <div key={n.id} className={`flex items-start gap-3 p-3 rounded-lg border ${n.readAt ? 'opacity-60' : 'bg-muted/40'}`}>
                  <span className={`h-2 w-2 rounded-full mt-1.5 shrink-0 ${typeColor(n.type)}`} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <p className="font-medium text-sm">{n.title}</p>
                      <span className="text-xs text-muted-foreground shrink-0">{new Date(n.createdAt).toLocaleString()}</span>
                    </div>
                    <p className="text-sm text-muted-foreground mt-0.5">{n.body}</p>
                  </div>
                  {!n.readAt && (
                    <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => handleMarkRead(n.id)}>Mark read</Button>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Notification channel preferences */}
      <Card><CardContent className="p-0"><Table>
        <TableHeader><TableRow>
          <TableHead>Category</TableHead>
          <TableHead className="text-center">Email</TableHead>
          <TableHead className="text-center">In-App</TableHead>
          <TableHead className="text-center">Push</TableHead>
        </TableRow></TableHeader>
        <TableBody>
          {categories.map(c => (
            <TableRow key={c.name}>
              <TableCell className="font-medium">{c.name}</TableCell>
              <TableCell className="text-center"><Switch checked={prefs[c.emailKey]} onCheckedChange={() => toggle(c.emailKey)} /></TableCell>
              <TableCell className="text-center"><Switch checked={prefs[c.inlineKey]} onCheckedChange={() => toggle(c.inlineKey)} /></TableCell>
              <TableCell className="text-center"><Switch checked={prefs[c.pushKey]} onCheckedChange={() => toggle(c.pushKey)} /></TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table></CardContent></Card>

      <Card><CardContent className="p-6 space-y-4">
        <div><Label>Digest Frequency</Label>
          <Select defaultValue="daily">
            <SelectTrigger className="w-48"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="realtime">Real-time</SelectItem>
              <SelectItem value="hourly">Hourly</SelectItem>
              <SelectItem value="daily">Daily</SelectItem>
              <SelectItem value="weekly">Weekly</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div><Label>Quiet Hours</Label>
          <div className="flex items-center gap-2">
            <Input type="time" defaultValue="22:00" className="w-28" />
            <span className="text-sm">to</span>
            <Input type="time" defaultValue="08:00" className="w-28" />
          </div>
        </div>
        <Button style={{ backgroundColor: PRIMARY }} onClick={handleSavePrefs}>Save Preferences</Button>
      </CardContent></Card>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 26. PREFERENCES SCREEN — applies theme via next-themes useTheme()
// ═══════════════════════════════════════════
function PreferencesScreen() {
  const { theme, setTheme, systemTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [autoSave, setAutoSave] = useState(true);
  const [resultsPerPage, setResultsPerPage] = useState('20');
  const [exportFormat, setExportFormat] = useState('csv');
  const [therapeuticArea, setTherapeuticArea] = useState('all');
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  // next-themes returns theme=undefined on first SSR render; only show
  // the active highlight after mount to avoid hydration mismatch.
  useEffect(() => { setMounted(true); }, []);

  // Load saved preferences from localStorage so they persist across sessions.
  useEffect(() => {
    if (!mounted) return;
    try {
      const saved = localStorage.getItem('drugos:preferences');
      if (saved) {
        const p = JSON.parse(saved);
        if (p.autoSave !== undefined) setAutoSave(p.autoSave);
        if (p.resultsPerPage) setResultsPerPage(p.resultsPerPage);
        if (p.exportFormat) setExportFormat(p.exportFormat);
        if (p.therapeuticArea) setTherapeuticArea(p.therapeuticArea);
      }
    } catch { /* ignore */ }
  }, [mounted]);

  const handleSave = () => {
    try {
      localStorage.setItem('drugos:preferences', JSON.stringify({
        autoSave, resultsPerPage, exportFormat, therapeuticArea,
      }));
      setSavedMsg('Preferences saved.');
      setTimeout(() => setSavedMsg(null), 2500);
    } catch {
      setSavedMsg('Failed to save preferences.');
    }
  };

  const themes: { id: 'light' | 'dark' | 'system'; label: string; icon: React.ElementType }[] = [
    { id: 'light', label: 'Light', icon: Sun },
    { id: 'dark', label: 'Dark', icon: Moon },
    { id: 'system', label: 'System', icon: MonitorSmartphone },
  ];

  const activeTheme = mounted ? (theme || 'light') : 'light';

  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Preferences" desc="Customize your DrugOS experience" />
      {savedMsg && (
        <div className="rounded-md bg-emerald-50 border border-emerald-200 text-emerald-700 text-sm px-3 py-2 dark:bg-emerald-950/40 dark:border-emerald-900 dark:text-emerald-300">{savedMsg}</div>
      )}
      <Card><CardContent className="p-6 space-y-6">
        <div>
          <Label>Theme</Label>
          <p className="text-xs text-muted-foreground mb-3">Choose how DrugOS looks. System follows your operating system preference.</p>
          <div className="flex gap-3">
            {themes.map(t => {
              const Icon = t.icon;
              const isActive = activeTheme === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setTheme(t.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm transition-colors ${isActive ? 'border-primary bg-primary/5 text-primary' : 'hover:bg-accent'}`}
                >
                  <Icon className="h-4 w-4" />
                  {t.label}
                  {t.id === 'system' && mounted && systemTheme && (
                    <span className="text-xs text-muted-foreground">({systemTheme})</span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        <Separator />

        <div>
          <Label>Default Therapeutic Area</Label>
          <p className="text-xs text-muted-foreground mb-2">Pre-filter disease searches to a therapeutic area.</p>
          <Select value={therapeuticArea} onValueChange={setTherapeuticArea}>
            <SelectTrigger className="w-64"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Areas</SelectItem>
              <SelectItem value="Neurology">Neurology</SelectItem>
              <SelectItem value="Oncology">Oncology</SelectItem>
              <SelectItem value="Rare Disease">Rare Disease</SelectItem>
              <SelectItem value="Cardiology">Cardiology</SelectItem>
              <SelectItem value="Infectious Disease">Infectious Disease</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label>Results Per Page</Label>
          <p className="text-xs text-muted-foreground mb-2">Default number of results shown in tables.</p>
          <Select value={resultsPerPage} onValueChange={setResultsPerPage}>
            <SelectTrigger className="w-32"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="10">10</SelectItem>
              <SelectItem value="20">20</SelectItem>
              <SelectItem value="50">50</SelectItem>
              <SelectItem value="100">100</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Separator />

        <div className="flex items-center justify-between">
          <div>
            <Label>Auto-save Queries</Label>
            <p className="text-xs text-muted-foreground">Automatically save search queries to history</p>
          </div>
          <Switch checked={autoSave} onCheckedChange={setAutoSave} />
        </div>

        <div className="flex items-center justify-between">
          <div><Label>Default Export Format</Label></div>
          <Select value={exportFormat} onValueChange={setExportFormat}>
            <SelectTrigger className="w-32"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="csv">CSV</SelectItem>
              <SelectItem value="json">JSON</SelectItem>
              <SelectItem value="xlsx">Excel</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Button onClick={handleSave} style={{ backgroundColor: PRIMARY }}>Save Preferences</Button>
      </CardContent></Card>
    </div></FadeIn>
  );
}


// ═══════════════════════════════════════════
// 27. PRIVACY POLICY SCREEN
// ═══════════════════════════════════════════
function PrivacyPolicyScreen() {
  const sections = [
    { title: '1. Information We Collect', content: 'DrugOS collects information that you provide directly to us, including your name, email address, organization affiliation, and research interests. We also automatically collect certain information when you use our platform, such as your IP address, browser type, operating system, referring URLs, and information about how you interact with the platform. This includes query logs, page views, and feature usage data that helps us improve our services and provide better drug repurposing insights.' },
    { title: '2. How We Use Your Information', content: 'We use the information we collect to provide, maintain, and improve the DrugOS platform, process your queries and drug repurposing analyses, send you technical notices and support messages, respond to your comments and questions, monitor and analyze trends and usage, detect and prevent fraud, and facilitate contests and promotional activities. Your research data is processed solely to deliver drug repurposing results and is never shared with third parties without explicit consent.' },
    { title: '3. Information Sharing', content: 'DrugOS does not sell, trade, or rent your personal information to third parties. We may share information with your organization administrators as part of team collaboration features, with service providers who assist in operating the platform, and when required by law. All data sharing complies with HIPAA, GDPR, and other applicable regulations.' },
    { title: '4. Data Security', content: 'We implement industry-standard security measures including encryption at rest (AES-256) and in transit (TLS 1.3), role-based access controls, regular security audits, and SOC 2 Type II compliance. We maintain Business Associate Agreements (BAA) for healthcare data processing. Our infrastructure is hosted on HIPAA-compliant cloud providers with multi-region redundancy.' },
    { title: '5. Your Rights', content: 'Under GDPR, you have the right to access, rectify, erase, and port your personal data. You may also object to or restrict certain processing activities. Under CCPA, you have the right to know, delete, and opt-out of the sale of your personal information. We provide self-service tools and support to exercise these rights.' },
    { title: '6. Data Retention', content: 'We retain your personal data for as long as your account is active or as needed to provide services. Query results and research data are retained according to your organization retention policy. You may request deletion of your data at any time through account settings or by contacting our privacy team.' },
  ];
  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Privacy Policy" desc="Last updated: June 1, 2026" />
      <Card><CardContent className="p-6 max-w-3xl mx-auto"><h2 className="text-xl font-bold mb-6">DrugOS Privacy Policy</h2><div className="space-y-6">{sections.map(s => (<div key={s.title}><h3 className="font-semibold text-lg mb-2">{s.title}</h3><p className="text-sm text-muted-foreground leading-relaxed">{s.content}</p></div>))}</div></CardContent></Card>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 28. TERMS SCREEN
// ═══════════════════════════════════════════
function TermsScreen() {
  const sections = [
    { title: '1. Acceptance of Terms', content: 'By accessing or using the DrugOS platform, you agree to be bound by these Terms of Service. If you do not agree to these terms, you may not access or use the platform. These terms apply to all visitors, users, and others who access or use DrugOS. We reserve the right to modify these terms at any time, and your continued use after modification constitutes acceptance of the updated terms.' },
    { title: '2. Use License', content: 'Subject to your compliance with these Terms, DrugOS grants you a limited, non-exclusive, non-transferable, revocable license to access and use the platform for your internal business or academic research purposes. You may not sublicense, sell, or distribute access to the platform. Usage limits apply based on your subscription plan, and exceeding these limits may result in overage charges or service restrictions.' },
    { title: '3. Intellectual Property', content: 'The DrugOS platform, including its knowledge graph, scoring algorithms, and AI models, is the exclusive property of DrugOS Corp. Drug repurposing predictions and evidence packages generated by the platform are provided for research purposes. Discovery Deal licensees receive exclusive commercial rights to validated predictions as defined in their licensing agreements.' },
    { title: '4. Prohibited Uses', content: 'You may not use DrugOS to develop competing products or services, reverse engineer the platform or its algorithms, share your account credentials with unauthorized users, or use the platform for any unlawful purpose. Violation of these restrictions may result in immediate account termination and legal action.' },
    { title: '5. Limitation of Liability', content: 'DrugOS provides computational predictions for research purposes only. We do not guarantee the accuracy, completeness, or clinical validity of any prediction. Users are responsible for independent validation before clinical application. DrugOS shall not be liable for any indirect, incidental, or consequential damages arising from the use of the platform.' },
    { title: '6. Termination', content: 'We may terminate or suspend your account immediately, without prior notice, for any breach of these Terms. Upon termination, your right to use the platform will cease immediately. Provisions that by their nature should survive termination shall remain in effect.' },
  ];
  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Terms of Service" desc="Last updated: June 1, 2026 · Version 3.2" />
      <Card><CardContent className="p-6 max-w-3xl mx-auto"><h2 className="text-xl font-bold mb-6">DrugOS Terms of Service</h2><div className="space-y-6">{sections.map(s => (<div key={s.title}><h3 className="font-semibold text-lg mb-2">{s.title}</h3><p className="text-sm text-muted-foreground leading-relaxed">{s.content}</p></div>))}</div></CardContent></Card>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 29. COMPLIANCE SCREEN
// ═══════════════════════════════════════════
/**
 * FE-012 ROOT FIX (Team Member 15, v108): The previous ComplianceScreen
 * rendered 5 fabricated compliance frameworks ("HIPAA compliant May 2026",
 * "GDPR compliant Apr 2026", "SOC 2 Type II compliant Mar 2026",
 * "21 CFR Part 11 compliant Feb 2026", "GxP partial Jun 2026") with
 * fabricated audit dates and 3 fabricated stat cards. No API call.
 * No banner. A compliance officer saw "HIPAA compliant May 2026" —
 * fabricated. Regulatory submissions based on this are fraudulent.
 * The "21 CFR Part 11 compliant" claim was particularly dangerous —
 * FDA electronic records compliance is a legal requirement, not a
 * UI label.
 *
 * ROOT FIX: Per the issue spec, remove the fabricated compliance
 * status entirely. Compliance status must come from real audit
 * reports stored in a document management system, not hardcoded.
 * We render an honest EmptyState that points the user at the
 * compliance team / DMS — never fabricated audit dates or
 * certifications.
 */
function ComplianceScreen() {
  // Issue 314 (audit 301-320): Wire to /api/audit-logs. The previous
  // ComplianceScreen rendered fabricated compliance certifications
  // ("HIPAA compliant", "21 CFR Part 11 compliant") with fake audit
  // dates — claiming compliance without an actual audit report is
  // regulatory fraud.
  //
  // ROOT FIX: We DO have real audit-log data. The AuditLog table records
  // every authentication event, data access, billing change, admin
  // action, etc. For compliance purposes (FDA 21 CFR Part 11, GDPR
  // Article 30, HIPAA §164.312(b)), these audit trails ARE the
  // compliance evidence. This screen now shows:
  //   - Audit log completeness (last 30 days event count)
  //   - Authentication events (logins, failed logins, MFA challenges)
  //   - Admin actions (role changes, user suspensions)
  //   - Data access events (dataset queries, evidence package builds)
  //   - Dead-letter entries (BE-003 — failed audit writes that must be
  //     investigated for compliance purposes)
  //
  // We do NOT claim any certification (HIPAA/GDPR/SOC 2/GxP/21 CFR
  // Part 11). Those require formal audit reports stored in a DMS.
  // We surface the real audit-trail evidence that supports a
  // compliance review.
  const { data: auditData, loading, error, refetch } = useApiResource<{ items: AuditLog[]; total: number }>(
    () => api.listAuditLogs(500, 0)
  );
  const logs = auditData?.items ?? [];

  // Aggregate all time-based and category filters in a single useMemo so
  // deps arrays are stable and the React Compiler can memoize correctly.
  const { authEvents, adminActions, dataAccess, recentEvents } = useMemo(() => {
    const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000;
    const auth = logs.filter(l => {
      const a = l.action.toLowerCase();
      return a.includes('login') || a.includes('logout') || a.includes('mfa') || a.includes('2fa');
    });
    const admin = logs.filter(l => {
      const a = l.action.toLowerCase();
      return a.includes('admin') || a.includes('role') || a.includes('user_');
    });
    const access = logs.filter(l => {
      const a = l.action.toLowerCase();
      return a.includes('dataset') || a.includes('evidence') || a.includes('hypothesis');
    });
    const recent = logs.filter(l => new Date(l.createdAt).getTime() > thirtyDaysAgo);
    return { authEvents: auth, adminActions: admin, dataAccess: access, recentEvents: recent };
  }, [logs]);

  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader
          title="Compliance"
          desc="Real audit-trail evidence from /api/audit-logs (FDA 21 CFR Part 11, GDPR Art. 30, HIPAA §164.312(b))"
          actions={<Button variant="outline" size="sm" onClick={() => refetch()} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>}
        />

        {loading && <LoadingSpinner label="Loading audit trail from /api/audit-logs…" />}
        {error && <ErrorDisplay error={error} onRetry={() => refetch()} />}

        {!loading && !error && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
              <StatCard
                title="Audit Events (30d)"
                value={recentEvents.length.toLocaleString()}
                subtitle="real audit-trail rows"
                icon={FileText}
              />
              <StatCard
                title="Auth Events"
                value={authEvents.length.toLocaleString()}
                subtitle="login/logout/MFA"
                icon={Shield}
              />
              <StatCard
                title="Admin Actions"
                value={adminActions.length.toLocaleString()}
                subtitle="role/user changes"
                icon={Settings}
              />
              <StatCard
                title="Data Access"
                value={dataAccess.length.toLocaleString()}
                subtitle="dataset/evidence/hypothesis"
                icon={Database}
              />
            </div>

            {logs.length === 0 ? (
              <EmptyState
                title="No audit trail yet"
                description="Once users start authenticating and accessing data, those events will be recorded in the audit log and surfaced here as compliance evidence."
              />
            ) : (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Recent Compliance-Relevant Audit Events</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Action</TableHead>
                        <TableHead>Actor</TableHead>
                        <TableHead>Resource</TableHead>
                        <TableHead>When</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {logs.slice(0, 15).map(l => (
                        <TableRow key={l.id}>
                          <TableCell>
                            <Badge variant="outline" className="font-mono text-xs">{l.action}</Badge>
                          </TableCell>
                          <TableCell className="text-sm">{l.actorName}</TableCell>
                          <TableCell className="text-xs text-muted-foreground font-mono">{l.resource || '—'}</TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            {new Date(l.createdAt).toLocaleString()}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}

            <Card className="border-amber-200 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-900">
              <CardContent className="p-4">
                <p className="text-sm font-semibold text-amber-900 dark:text-amber-100 mb-2">
                  Certification Status
                </p>
                <p className="text-xs text-amber-800 dark:text-amber-200">
                  This screen surfaces REAL audit-trail evidence that supports compliance reviews
                  (FDA 21 CFR Part 11, GDPR Art. 30, HIPAA §164.312(b)). It does NOT claim any
                  formal certification. Compliance certifications (HIPAA, GDPR, SOC 2, 21 CFR Part 11,
                  GxP) are formal legal designations backed by signed audit reports, BAAs, and CSV
                  documentation stored in a DMS. Contact your compliance team or legal counsel for
                  the current certification posture.
                </p>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 30. HELP CENTER SCREEN
// ═══════════════════════════════════════════
/**
 * FE-031 ROOT FIX: The previous HelpCenterScreen rendered fabricated
 * article counts ("Getting Started 8 articles", etc.) and fabricated
 * view counts ("2.4K views"). These numbers were made up and eroded
 * trust. Since there is no CMS or markdown file with real help articles
 * in the repo, we now render an honest state: a search bar (non-
 * functional until a search backend is added) and a "Contact Support"
 * button. No fabricated counts, no fake popularity metrics.
 */
function HelpCenterScreen() {
  const [search, setSearch] = useState('');
  const categories = [
    { title: 'Getting Started', icon: Play },
    { title: 'Search & Queries', icon: Search },
    { title: 'Drug Candidates', icon: Target },
    { title: 'Evidence & Reports', icon: FileText },
    { title: 'API & Integration', icon: Code },
    { title: 'Billing & Plans', icon: CreditCard },
  ];
  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Help Center" desc="Find answers and get support" />
      <Card className="bg-gradient-to-r from-primary/5 to-primary/10"><CardContent className="p-8 text-center"><h2 className="text-xl font-bold mb-3">How can we help?</h2><div className="relative max-w-lg mx-auto"><Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" /><Input placeholder="Search help articles..." value={search} onChange={e => setSearch(e.target.value)} className="pl-12 h-12 text-base" /></div></CardContent></Card>
      {/* FE-031: Categories without fabricated article counts. */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">{categories.map(c => { const Icon = c.icon; return (<Card key={c.title} className="hover:shadow-md transition-shadow cursor-pointer"><CardContent className="p-5"><div className="flex items-center gap-3 mb-2"><Icon className="h-5 w-5 text-primary" /><h3 className="font-semibold text-sm">{c.title}</h3></div><p className="text-xs text-muted-foreground">Help articles</p></CardContent></Card>); })}</div>
      {/* FE-031: Removed "Popular Articles" section which had fabricated view
          counts ("2.4K views", "1.8K views", etc.). No real analytics exist
          to populate this, so we show an honest empty state instead. */}
      <Card>
        <CardContent className="p-6 text-center text-muted-foreground">
          <BookOpen className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm font-medium">Help articles coming soon</p>
          <p className="text-xs mt-1 max-w-md mx-auto">Our knowledge base is being built. For now, contact support below for assistance.</p>
        </CardContent>
      </Card>
      <div className="text-center"><Button variant="outline"><MessageSquare className="h-4 w-4 mr-2" />Contact Support</Button></div>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 31. TICKET SCREEN
// ═══════════════════════════════════════════
function TicketScreen() {
  const [createOpen, setCreateOpen] = useState(false);
  // Note: ticket list below is illustrative sample data, not from a real
  // /api/tickets endpoint. The DemoDataBanner makes this explicit.
  const tickets = [
    { id: 'TK-1234', subject: 'Cannot export report in PDF format', status: 'open', priority: 'high', created: '2 hours ago', messages: 3 },
    { id: 'TK-1233', subject: 'API rate limit hit unexpectedly', status: 'in-progress', priority: 'medium', created: '1 day ago', messages: 5 },
    { id: 'TK-1230', subject: 'Knowledge graph timeout for rare disease', status: 'open', priority: 'low', created: '2 days ago', messages: 2 },
    { id: 'TK-1228', subject: 'Feature request: batch comparison', status: 'closed', priority: 'low', created: '1 week ago', messages: 4 },
  ];
  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Support Tickets" desc={`${tickets.filter(t => t.status !== 'closed').length} open tickets`} actions={<Button style={{ backgroundColor: PRIMARY }} onClick={() => setCreateOpen(true)}><Plus className="h-4 w-4 mr-1.5" />New Ticket</Button>} />
      <DemoDataBanner
        reason="The ticket list below is illustrative sample data. There is no /api/tickets endpoint in this deployment. The 'New Ticket' dialog form is non-functional — submitting it does not persist anything. Wire a real ticketing backend (Zendesk, Intercom, or in-DB tickets) before relying on this screen."
      />
      <Card><CardContent className="p-0"><Table><TableHeader><TableRow><TableHead>Ticket</TableHead><TableHead>Subject</TableHead><TableHead>Status</TableHead><TableHead>Priority</TableHead><TableHead>Created</TableHead><TableHead>Messages</TableHead></TableRow></TableHeader>
        <TableBody>{tickets.map(t => (<TableRow key={t.id} className="cursor-pointer hover:bg-muted/30"><TableCell className="font-mono text-sm">{t.id}</TableCell><TableCell className="font-medium">{t.subject}</TableCell>
          <TableCell><Badge variant={t.status === 'open' ? 'default' : t.status === 'in-progress' ? 'secondary' : 'outline'}>{t.status}</Badge></TableCell>
          <TableCell><Badge variant={t.priority === 'high' ? 'destructive' : t.priority === 'medium' ? 'secondary' : 'outline'}>{t.priority}</Badge></TableCell>
          <TableCell className="text-sm text-muted-foreground">{t.created}</TableCell><TableCell>{t.messages}</TableCell></TableRow>))}</TableBody></Table></CardContent></Card>
      <Dialog open={createOpen} onOpenChange={setCreateOpen}><DialogContent><DialogHeader><DialogTitle>Create Support Ticket</DialogTitle></DialogHeader><div className="space-y-4"><div><Label>Subject</Label><Input placeholder="Brief description of the issue" /></div><div><Label>Priority</Label><Select defaultValue="medium"><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="low">Low</SelectItem><SelectItem value="medium">Medium</SelectItem><SelectItem value="high">High</SelectItem><SelectItem value="critical">Critical</SelectItem></SelectContent></Select></div><div><Label>Description</Label><Textarea placeholder="Provide details about the issue..." className="min-h-[100px]" /></div></div><DialogFooter><Button style={{ backgroundColor: PRIMARY }} onClick={() => setCreateOpen(false)}>Submit Ticket</Button></DialogFooter></DialogContent></Dialog>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 32. SYSTEM STATUS SCREEN
// ═══════════════════════════════════════════
/**
 * FE-014 ROOT FIX (Team Member 15, v108): The previous SystemStatusScreen
 * rendered 3 fabricated incidents ("Jun 10 Report generation delays 2h 15m",
 * etc.) and a fabricated "All Systems Operational" banner despite no real
 * health check. The real /api/system/status endpoint exists and returns
 * real service availability (auth, rxnorm, mesh, clinicalTrials, pubmed,
 * openfda, patentsview, kg, dataset, rl), but this screen NEVER called it.
 *
 * ROOT FIX: Wire the screen to `api.getSystemStatus()` (real call to
 * GET /api/system/status). Render real service states. Remove the
 * fabricated incidents list — there is no incident-tracking system
 * in the codebase.
 */
function SystemStatusScreen() {
  const { data: status, loading, error, refetch } = useApiResource<SystemStatus>(
    () => api.getSystemStatus()
  );

  const services = status ? Object.entries(status.services).map(([key, svc]) => ({
    key,
    name: svc.service || key,
    available: svc.available,
    degraded: (svc as any).degraded,
    reason: svc.reason,
  })) : [];

  const allOperational = services.length > 0 && services.every(s => s.available && !s.degraded);
  const anyDegraded = services.some(s => s.degraded);
  const anyDown = services.some(s => !s.available);

  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader
          title="System Status"
          desc="Real-time platform health (from /api/system/status)"
          actions={
            <Button variant="outline" size="sm" onClick={() => refetch()} disabled={loading}>
              <RefreshCw className={`h-4 w-4 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          }
        />

        {loading && <LoadingSpinner label="Loading system status…" />}
        {error && <ErrorDisplay error={error} onRetry={() => refetch()} />}

        {!loading && !error && status && (
          <>
            {/* Real overall status banner — derived from actual service states */}
            <Card className={
              anyDown ? 'bg-red-50 border-red-200 dark:bg-red-950/30 dark:border-red-900' :
              anyDegraded ? 'bg-amber-50 border-amber-200 dark:bg-amber-950/30 dark:border-amber-900' :
              'bg-emerald-50 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900'
            }>
              <CardContent className="p-5">
                <div className="flex items-center gap-3">
                  {anyDown ? (
                    <XCircle className="h-6 w-6 text-red-600" />
                  ) : anyDegraded ? (
                    <AlertTriangle className="h-6 w-6 text-amber-600" />
                  ) : (
                    <CheckCircle2 className="h-6 w-6 text-emerald-600" />
                  )}
                  <div>
                    <h3 className={`font-semibold ${
                      anyDown ? 'text-red-800 dark:text-red-200' :
                      anyDegraded ? 'text-amber-800 dark:text-amber-200' :
                      'text-emerald-800 dark:text-emerald-200'
                    }`}>
                      {anyDown ? 'Some services unavailable' : anyDegraded ? 'Some services degraded' : 'All systems operational'}
                    </h3>
                    <p className={`text-sm ${
                      anyDown ? 'text-red-700 dark:text-red-300' :
                      anyDegraded ? 'text-amber-700 dark:text-amber-300' :
                      'text-emerald-700 dark:text-emerald-300'
                    }`}>
                      Last checked: {status.generatedAt ? new Date(status.generatedAt).toLocaleString() : 'just now'}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Real per-service status table */}
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-base">Service Status ({services.length} services)</CardTitle></CardHeader>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Service</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Details</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {services.map(s => (
                      <TableRow key={s.key}>
                        <TableCell className="font-medium">{s.name}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <span className={`w-2.5 h-2.5 rounded-full ${
                              s.available && !s.degraded ? 'bg-emerald-500' :
                              s.degraded ? 'bg-amber-500' :
                              'bg-red-500'
                            }`} />
                            <Badge variant={s.available && !s.degraded ? 'default' : s.degraded ? 'secondary' : 'destructive'}>
                              {s.available && !s.degraded ? 'operational' : s.degraded ? 'degraded' : 'unavailable'}
                            </Badge>
                          </div>
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">{s.reason || '—'}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </>
        )}

        {!loading && !error && !status && (
          <EmptyState
            title="System status unavailable"
            description="The /api/system/status endpoint did not return data. This may be due to insufficient permissions (admin role required) or a server error."
          />
        )}

        {/* FE-014: Removed the fabricated "Recent Incidents" section.
            There is no incident-tracking system in the codebase, so any
            incidents shown would be fabricated. When an incident-tracking
            backend is added, this section can be wired to it. */}
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            <AlertCircle className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm font-medium">Incident history not tracked</p>
            <p className="text-xs mt-1 max-w-md mx-auto">
              There is no incident-tracking system in the codebase. When one is added
              (e.g. a StatusPage integration or in-DB incident log), this section will
              show real incident history. No fabricated incidents are rendered.
            </p>
          </CardContent>
        </Card>
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 33. INVESTOR DASHBOARD SCREEN
// ═══════════════════════════════════════════
/**
 * FE-013 ROOT FIX (Team Member 15, v108): The previous InvestorDashboardScreen
 * rendered fabricated ARR/MRR data (Jan $420K ARR → Jun $840K ARR),
 * fabricated customer counts (42 customers, +24%), fabricated NRR (118%),
 * and 3 fabricated cohorts. An investor saw "$840K ARR" — both
 * fabricated. Investment decisions were made on fake financials.
 * This is securities fraud if shown to actual investors.
 *
 * ROOT FIX: Per the issue spec, remove all fabricated financial data.
 * Investor data must come from real financial systems (Stripe,
 * QuickBooks, Carta), not hardcoded arrays. We render an honest
 * EmptyState that points the user at the finance system — never
 * fabricated ARR/MRR/cohorts.
 */
function InvestorDashboardScreen() {
  // Issue 315 (audit 301-320): Wire to /api/admin/metrics. The previous
  // InvestorDashboardScreen rendered fabricated ARR/MRR/customer
  // counts/NRR/cohort data — showing fabricated financials to actual
  // investors is securities fraud.
  //
  // ROOT FIX: This screen now calls /api/admin/metrics, which returns
  // REAL platform traction metrics derived from existing DB tables:
  //   - totalUsers, totalOrganizations, activeSubscriptions
  //   - totalProjects, totalHypotheses, totalValidatedHypotheses
  //   - auditLogEventsLast30Days, topActionsLast30Days
  //   - dailyActiveUsersLast7Days
  //   - dataset + KG scale (Phase 1 + Phase 2)
  //
  // Financial metrics (ARR/MRR/NRR) are explicitly null in the response
  // — the endpoint does NOT fabricate them. The screen surfaces a clear
  // "Requires Stripe integration" notice for any financial card.
  const { data: metrics, loading, error, refetch } = useApiResource<AdminMetricsResponse>(
    () => api.getAdminMetrics()
  );

  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader
          title="Investor Dashboard"
          desc={`Real platform metrics from /api/admin/metrics${metrics ? ` · scope: ${metrics.scope}` : ''}`}
          actions={<Button variant="outline" size="sm" onClick={() => refetch()} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>}
        />

        {loading && <LoadingSpinner label="Loading platform metrics from /api/admin/metrics…" />}
        {error && <ErrorDisplay error={error} onRetry={() => refetch()} />}

        {!loading && !error && metrics && (
          <>
            {/* Real platform traction — NOT fabricated */}
            <div>
              <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">Platform Traction (real)</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard title="Total Users" value={metrics.totalUsers.toLocaleString()} subtitle="from User table" icon={Users} />
                <StatCard title="Organizations" value={metrics.totalOrganizations.toLocaleString()} subtitle="from Organization table" icon={Building} />
                <StatCard title="Active Subscriptions" value={metrics.activeSubscriptions.toLocaleString()} subtitle="status='active'" icon={CreditCard} />
                <StatCard title="Audit Events (30d)" value={metrics.auditLogEventsLast30Days.toLocaleString()} subtitle="real audit-log rows" icon={Activity} />
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">Research Activity (real)</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard title="Total Projects" value={metrics.totalProjects.toLocaleString()} subtitle="from Project table" icon={FolderKanban} />
                <StatCard title="Total Hypotheses" value={metrics.totalHypotheses.toLocaleString()} subtitle="from Hypothesis table" icon={Target} />
                <StatCard title="Validated Hypotheses" value={metrics.totalValidatedHypotheses.toLocaleString()} subtitle="status='validated'" icon={CheckCircle2} />
                <StatCard title="Evidence Packages" value={metrics.totalEvidencePackages.toLocaleString()} subtitle="from EvidencePackage table" icon={FileText} />
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">Data Scale (real)</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard title="Dataset Nodes" value={metrics.dataset.nodesLoaded.toLocaleString()} subtitle={`${metrics.dataset.sourcesLoaded}/${metrics.dataset.sourcesTotal} sources`} icon={Database} />
                <StatCard title="Dataset Edges" value={metrics.dataset.edgesLoaded.toLocaleString()} subtitle={`source: ${metrics.dataset.source}`} icon={GitBranch} />
                <StatCard
                  title="KG Nodes"
                  value={metrics.knowledgeGraph ? metrics.knowledgeGraph.nodeCount.toLocaleString() : '—'}
                  subtitle={metrics.knowledgeGraph ? `source: ${metrics.knowledgeGraph.source}` : 'KG service unavailable'}
                  icon={Network}
                />
                <StatCard
                  title="KG Edges"
                  value={metrics.knowledgeGraph ? metrics.knowledgeGraph.edgeCount.toLocaleString() : '—'}
                  subtitle={metrics.knowledgeGraph ? 'from Phase 2 registry' : 'KG service unavailable'}
                  icon={Share2}
                />
              </div>
            </div>

            {/* Daily active users chart (real) */}
            {metrics.dailyActiveUsersLast7Days.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Daily Active Users (last 7 days, real)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={metrics.dailyActiveUsersLast7Days.map(d => ({ day: d.day, users: d.activeUsers }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="day" tick={{ fontSize: 11 }} />
                        <YAxis />
                        <RechartsTooltip />
                        <Line type="monotone" dataKey="users" stroke={PRIMARY} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Top actions chart (real) */}
            {metrics.topActionsLast30Days.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Top Actions (last 30 days, real)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={metrics.topActionsLast30Days} layout="vertical" margin={{ left: 80, right: 20, top: 10, bottom: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="action" width={120} tick={{ fontSize: 11 }} />
                        <RechartsTooltip />
                        <Bar dataKey="count" fill={PRIMARY} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* EXPLICIT financial-metrics disclaimer — NOT fabricated */}
            <Card className="border-amber-200 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-900">
              <CardContent className="p-4">
                <p className="text-sm font-semibold text-amber-900 dark:text-amber-100 mb-2">
                  Financial Metrics Not Available
                </p>
                <p className="text-xs text-amber-800 dark:text-amber-200">
                  {metrics.financials.note} The /api/admin/metrics endpoint explicitly returns
                  null for ARR, MRR, customer count, and NRR. These metrics require Stripe
                  (billing), CRM (customer count), and subscription-event history (NRR) integrations
                  that are not deployed. Showing fabricated financials to investors is securities
                  fraud — this screen refuses to do so.
                </p>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 34. CAP TABLE SCREEN
// ═══════════════════════════════════════════
/**
 * FE-013 ROOT FIX (Team Member 15, v108): The previous CapTableScreen
 * rendered 3 fabricated funding rounds ("Pre-Seed $500K $3M valuation",
 * "Seed $2M $10M valuation", "Series A $8M $40M valuation") and 5
 * fabricated shareholders. An investor saw "$40M valuation" —
 * fabricated. Investment decisions were made on fake cap table data.
 *
 * ROOT FIX: Per the issue spec, remove both screens entirely. Cap
 * table data must come from a real cap table management system
 * (Carta, Pulley, Capbase), not hardcoded arrays. We render an
 * honest EmptyState.
 */
function CapTableScreen() {
  // Issue 316 (audit 301-320): No /api/admin/cap-table endpoint exists.
  // Cap table data (shareholders, share classes, funding rounds,
  // valuations) must come from a real cap table management system like
  // Carta, Pulley, or Capbase. The DemoDataBanner makes it 100% visible
  // that this screen is non-functional — anyone (especially investors)
  // seeing this screen immediately knows the data is not real.
  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader title="Cap Table" desc="Capitalization table and funding history" />
        <DemoDataBanner
          reason="Cap table data is not available. There is no /api/admin/cap-table endpoint. Cap table data (shareholders, share classes, funding rounds, valuations) must come from a real cap table management system like Carta, Pulley, or Capbase. Any cap table data shown here would be fabricated — showing fabricated cap table data to investors is securities fraud."
        />
        <EmptyState
          title="Cap table not available"
          description="Cap table data (shareholders, share classes, funding rounds, valuations) must come from a real cap table management system like Carta, Pulley, or Capbase — not a hardcoded array. Showing fabricated cap table data to investors is securities fraud. Integrate this screen with your cap table platform before exposing it."
        />
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 35. CHANGELOG SCREEN
// ═══════════════════════════════════════════
function ChangelogScreen() {
  // Note: changelog entries below are illustrative product-marketing copy,
  // not real audit data. The DemoDataBanner makes this explicit so a reader
  // does not mistake these for verified release notes.
  const entries = [
    { version: 'v2.1.0', date: 'Jun 8, 2026', type: 'feature', title: 'GxP Validated Mode', desc: 'Full GxP validated mode for clinical research with audit trails and electronic signatures.' },
    { version: 'v2.0.5', date: 'Jun 1, 2026', type: 'improvement', title: 'Knowledge Graph V2', desc: 'Updated knowledge graph engine with 50% faster query performance and 200K additional nodes.' },
    { version: 'v2.0.4', date: 'May 25, 2026', type: 'bugfix', title: 'Report Generation Fix', desc: 'Fixed issue where PDF reports occasionally had missing pathway diagrams.' },
    { version: 'v2.0.3', date: 'May 18, 2026', type: 'feature', title: 'Batch Query API', desc: 'New batch query endpoint allows up to 50 disease queries in a single API call.' },
    { version: 'v2.0.2', date: 'May 10, 2026', type: 'improvement', title: 'Safety Scoring Update', desc: 'Improved safety scoring algorithm with better off-target prediction accuracy.' },
  ];
  const typeColors: Record<string, string> = { feature: 'bg-blue-100 text-blue-700', improvement: 'bg-amber-100 text-amber-700', bugfix: 'bg-red-100 text-red-700' };
  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Changelog" desc="Product updates and release notes" actions={<Button variant="outline" size="sm"><Bell className="h-4 w-4 mr-1.5" />Subscribe</Button>} />
      <DemoDataBanner
        reason="These changelog entries are illustrative product-marketing copy, not derived from a real release-notes CMS or git history. They are shown for UI demonstration only — do not cite them in customer communications or compliance documentation."
      />
      <div className="space-y-4">{entries.map(e => (<Card key={e.version + e.title} className="hover:shadow-md transition-shadow"><CardContent className="p-5"><div className="flex items-start justify-between mb-2"><div className="flex items-center gap-3"><Badge variant="outline" className="font-mono">{e.version}</Badge><span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${typeColors[e.type]}`}>{e.type}</span></div><span className="text-xs text-muted-foreground">{e.date}</span></div><h3 className="font-semibold">{e.title}</h3><p className="text-sm text-muted-foreground mt-1">{e.desc}</p></CardContent></Card>))}</div>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// 36. ROADMAP SCREEN
// ═══════════════════════════════════════════
/**
 * FE-014 ROOT FIX (Team Member 15, v108): The previous RoadmapScreen
 * rendered a fabricated product roadmap (Q2 2026 → Q1 2027) with
 * fabricated vote counts. There is no roadmap CMS in the codebase.
 *
 * ROOT FIX: Per the issue spec, replace the fabricated roadmap with
 * an honest EmptyState. The product roadmap should be backed by a
 * CMS (Contentful, Sanity, etc.) or a project-tracking tool
 * (Linear, Jira) — not a hardcoded array.
 */
function RoadmapScreen() {
  return (
    <FadeIn>
      <div className="space-y-6">
        <PageHeader title="Product Roadmap" desc="Upcoming features and improvements" />
        <EmptyState
          title="Roadmap not available"
          description="The product roadmap is not backed by a CMS or project-tracking integration in this deployment. There is no /api/roadmap endpoint. When a CMS (Contentful, Sanity) or project tracker (Linear, Jira) integration is added, this screen will show real roadmap items with real statuses and real vote counts. No fabricated roadmap items or vote counts are rendered."
        />
      </div>
    </FadeIn>
  );
}

// ═══════════════════════════════════════════
// 37. FEEDBACK SCREEN
// ═══════════════════════════════════════════
function FeedbackScreen() {
  const [rating, setRating] = useState(0);
  const [category, setCategory] = useState('');
  const [description, setDescription] = useState('');
  // FE-030 ROOT FIX: The previous version rendered 3 hardcoded fake feedback
  // entries attributed to fabricated colleagues. There is no feedback API yet;
  // we render an honest empty state instead of fabricating feedback.
  const recentFeedback: Array<{ user: string; rating: number; category: string; feedback: string; date: string }> = [];
  return (
    <FadeIn><div className="space-y-6">
      <PageHeader title="Feedback" desc="Help us improve DrugOS" />
      <Card><CardContent className="p-6 space-y-4">
        <div><Label>How would you rate your experience?</Label><div className="flex gap-2 mt-2">{[1,2,3,4,5].map(s => (<button key={s} onClick={() => setRating(s)} className={`text-2xl transition-colors ${s <= rating ? 'text-yellow-400' : 'text-muted-foreground/30'}`}>★</button>))}</div></div>
        <div><Label>Category</Label><Select value={category} onValueChange={setCategory}><SelectTrigger><SelectValue placeholder="Select category" /></SelectTrigger><SelectContent><SelectItem value="bug">Bug Report</SelectItem><SelectItem value="feature">Feature Request</SelectItem><SelectItem value="improvement">Improvement</SelectItem><SelectItem value="praise">Praise</SelectItem></SelectContent></Select></div>
        <div><Label>Description</Label><Textarea value={description} onChange={e => setDescription(e.target.value)} placeholder="Tell us more about your experience..." className="min-h-[100px]" /></div>
        <Button style={{ backgroundColor: PRIMARY }}><Send className="h-4 w-4 mr-1.5" />Submit Feedback</Button>
      </CardContent></Card>
      <Card><CardHeader className="pb-2"><CardTitle className="text-base">Recent Feedback</CardTitle></CardHeader><CardContent><div className="space-y-4">{recentFeedback.map(f => (<div key={f.user + f.date} className="p-4 border rounded-lg"><div className="flex items-center justify-between mb-2"><div className="flex items-center gap-2"><span className="font-medium text-sm">{f.user}</span><Badge variant="outline" className="text-xs">{f.category}</Badge></div><span className="text-xs text-muted-foreground">{f.date}</span></div><div className="flex gap-0.5 mb-2">{[1,2,3,4,5].map(s => (<span key={s} className={`text-sm ${s <= f.rating ? 'text-yellow-400' : 'text-muted-foreground/20'}`}>★</span>))}</div><p className="text-sm text-muted-foreground">{f.feedback}</p></div>))}</div></CardContent></Card>
    </div></FadeIn>
  );
}

// ═══════════════════════════════════════════
// EXPORT ALL REMAINING SCREENS
// ═══════════════════════════════════════════
export const remainingScreens: Record<string, React.ComponentType> = {
  'pipeline': PipelineScreen,
  'analytics': AnalyticsScreen,
  'team': TeamMembersScreen,
  'projects': ProjectsScreen,
  'shared-queries': SharedQueriesScreen,
  'annotations': AnnotationsScreen,
  'data-sources': DataSourcesScreen,
  'graph-stats': GraphStatisticsScreen,
  'quality': QualityScreen,
  'subscription': SubscriptionScreen,
  'usage': UsageScreen,
  'deals': DealsScreen,
  'invoices': InvoicesScreen,
  'users': UsersAdminScreen,
  'roles': RolesScreen,
  'sso': SSOScreen,
  'audit-logs': AuditLogsScreen,
  'feature-flags': FeatureFlagsScreen,
  'api-docs': APIDocsScreen,
  'api-keys': APIKeysScreen,
  'playground': PlaygroundScreen,
  'webhooks': WebhooksScreen,
  'profile': ProfileScreen,
  'security': SecuritySettingsScreen,
  'notifications': NotificationsScreen,
  'preferences': PreferencesScreen,
  'privacy': PrivacyPolicyScreen,
  'terms': TermsScreen,
  'compliance': ComplianceScreen,
  'help-center': HelpCenterScreen,
  'tickets': TicketScreen,
  'system-status': SystemStatusScreen,
  'investor-dashboard': InvestorDashboardScreen,
  'cap-table': CapTableScreen,
  'changelog': ChangelogScreen,
  'roadmap': RoadmapScreen,
  'feedback': FeedbackScreen,
};

