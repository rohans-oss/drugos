'use client';

import { useState, useMemo } from 'react';
import { useDrugOSNav } from './nav-context';
import {
  CreditCard, TrendingUp, Shield, Scale, Code, Settings, HelpCircle,
  FileQuestion, Users, CheckCircle, AlertTriangle, Clock, Download,
  Plus, Search, Filter, X, ChevronRight, ChevronDown, ExternalLink,
  Key, Globe, Lock, Eye, Mail, Bell, Zap, RefreshCw, Copy, Trash2,
  Edit, MoreVertical, ArrowUpRight, ArrowDownRight, DollarSign,
  BarChart3, PieChart, Target, BookOpen, MessageSquare, Send,
  FileText, Database, Server, Activity, Info, Save, Upload,
  ChevronLeft, Check, Minus, PlusCircle, AlertCircle, ShieldCheck,
  Building, UserCog, MapPin, Palette, Plug, BellRing, ScrollText,
  Calculator, Gauge, Workflow, Percent, ShoppingCart,
  ToggleRight, FolderLock, KeyRound, ScanSearch, Cookie, Archive,
  BookMarked, Play, Webhook, Braces, Upload as UploadIcon,
  Settings2, User, Smartphone, Headphones, Ticket, Library,
  MessageCircle, Lightbulb, RotateCcw, Heart as HeartIcon, ShieldAlert,
  LogIn, UserCheck, Hourglass, Siren
} from 'lucide-react';
import {
  Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Separator } from '@/components/ui/separator';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table';
import {
  Dialog, DialogContent, DialogDescription, DialogFooter,
  DialogHeader, DialogTitle, DialogTrigger,
} from '@/components/ui/dialog';
import {
  Accordion, AccordionContent, AccordionItem, AccordionTrigger,
} from '@/components/ui/accordion';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area, RadarChart,
  Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, Legend, PieChart as RechartsPieChart, Pie, Cell,
} from 'recharts';
// FE-034 ROOT FIX: `mock-data.ts` was deleted (dangerous name invited future
// engineers to add fabricated data). Empty defaults now live in
// `@/lib/empty-defaults` — the name makes it unambiguous that these are
// EMPTY placeholders, NOT sample data. Components render empty states until
// migrated to real API calls (see use-api-data.tsx / use-account-data.tsx).
import {
  users, auditLogs, subscriptionPlans, billingHistory,
  apiKeys, webhooks, usageMetrics, dealPipeline, organization,
  featureFlags, dataSources,
} from '@/lib/empty-defaults';

// ─── Design Tokens ───
const C = { primary: '#5B4FCF', green: '#1D9E75', orange: '#D4853A', red: '#C0392B', bg: '#F8F8FA' };

// ─── Shared Helpers ───
function PageHeader({ title, desc, actions }: { title: string; desc?: string; actions?: React.ReactNode }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6">
      <div><h1 className="text-2xl font-bold text-foreground">{title}</h1>{desc && <p className="text-sm text-muted-foreground mt-1">{desc}</p>}</div>
      {actions && <div className="flex items-center gap-2 flex-shrink-0">{actions}</div>}
    </div>
  );
}

function StatCard({ title, value, subtitle, icon: Icon, trend }: { title: string; value: string | number; subtitle?: string; icon?: React.ComponentType<{className?:string}>; trend?: string }) {
  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold text-foreground mt-1">{value}</p>
            {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
            {trend && <p className={`text-xs mt-1 font-medium ${trend.startsWith('+') ? 'text-emerald-600' : 'text-red-500'}`}>{trend}</p>}
          </div>
          {Icon && <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center"><Icon className="h-5 w-5 text-primary" /></div>}
        </div>
      </CardContent>
    </Card>
  );
}

function ColorProgress({ value, max, label }: { value: number; max: number; label: string }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = pct > 90 ? C.red : pct > 75 ? C.orange : pct > 50 ? '#EAB308' : C.green;
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-sm"><span className="text-muted-foreground">{label}</span><span className="font-medium">{value.toLocaleString()}/{max.toLocaleString()}</span></div>
      <div className="h-2 w-full bg-muted rounded-full overflow-hidden"><div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} /></div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, 'default' | 'secondary' | 'outline' | 'destructive'> = {
    active: 'default', operational: 'default', healthy: 'default', Paid: 'default',
    inactive: 'secondary', degraded: 'secondary', pending: 'outline', invited: 'outline',
    suspended: 'destructive', error: 'destructive',
  };
  return <Badge variant={map[status] ?? 'outline'}>{status}</Badge>;
}

// ─── Chart Data ───
// FE-064 ROOT FIX: All hardcoded chart data arrays have been DELETED.
//
// Previously this file defined 10 fabricated datasets:
//   usageTrendData, endpointData, revenueProjectionData, marketSizingData,
//   radarData, comparableData, pipelinePredictData, royaltyData,
//   apiUsageTimeData, moatData
//
// These contained invented numbers (e.g. revenue projections of $12M → $350M
// over 5 years, pipeline success rates of 15%/30%/45%/65%/90% by phase,
// competitive intelligence scores for BenevolentAI/Recursion/Insilico).
// In a pharma platform, an admin/investor/executive viewing these screens
// would believe they were seeing real analytics — leading to decisions made
// on fabricated data, potential investor fraud, and strategic misdirection.
//
// ROOT FIX (per issue spec): "Replace every hardcoded dataset with a real
// API call. If the underlying data doesn't exist yet, render an empty state
// with 'No data available' — never fabricated numbers."
//
// These constants were not exported and not consumed by any rendering
// component in the codebase (verified by grep for all 10 names across
// frontend/src/). They were dead code — fabricated numbers sitting in the
// source with no UI path to them. The actual admin/billing/investor screens
// in remaining-screens.tsx and all-screens.tsx already use the real
// api-client (api.listInvoices, api.listPlans, api.listTeamMembers, etc.)
// with proper loading/error/empty states via the useApiList / useApiResource
// hooks from use-api-data.tsx.
//
// DO NOT re-add hardcoded chart data here. If a screen needs analytics:
//   1. Create a real API endpoint (e.g. /api/admin/analytics) backed by DB
//      queries or a real analytics service.
//   2. Fetch it via useApiList / useApiResource.
//   3. Render an EmptyState ("No data available") when the API returns [].
//
// The recharts imports above are retained because the helper components
// (PageHeader, StatCard, etc.) in this file may be used by future screens
// that fetch real analytics. If this file ends up with no consumers, it
// should be deleted entirely rather than repopulated with fake data.

