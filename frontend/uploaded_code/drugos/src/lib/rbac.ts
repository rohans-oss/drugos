/**
 * Role-Based Access Control (RBAC) for DruGOS.
 *
 * Each sidebar section has a list of roles allowed to view it. The AppShell
 * sidebar filters out sections the current user's role cannot access, and
 * the AppSectionRenderer redirects to the dashboard if a user tries to
 * navigate directly to a forbidden section.
 *
 * Role hierarchy:
 *   viewer        — read-only access to research outputs
 *   researcher    — search diseases, build evidence packages, save queries
 *   data-scientist — researcher + datasets, graph statistics, quality metrics
 *   pi            — researcher + projects, team oversight
 *   business-dev  — researcher + deals, market opportunity data
 *   developer     — developer platform: API keys, webhooks, playground
 *   billing       — billing & invoices only
 *   admin         — full access (users, roles, SSO, audit logs, feature flags)
 *   owner         — full access (same as admin + billing)
 */

export type Role =
  | "viewer"
  | "researcher"
  | "data-scientist"
  | "pi"
  | "business-dev"
  | "developer"
  | "billing"
  | "admin"
  | "owner";

/** Sections every authenticated user can see. */
const BASE_SECTIONS = [
  "dashboard",
  "search",
  "search-results",
  "candidate-detail",
  "knowledge-graph",
  "pathways",
  "safety",
  "patents",
  "clinical-trials",
  "literature",
  "admet",
  "interactions",
  "evidence-builder",
  "reports",
  "saved-queries",
  "shortlists",
  "profile",
  "security",
  "notifications",
  "preferences",
  "help-center",
  "tickets",
  "system-status",
  "privacy",
  "terms",
  "compliance",
];

/** Sections restricted to specific roles. */
const ROLE_SECTIONS: Record<string, string[]> = {
  // Team & collaboration
  team: ["pi", "admin", "owner"],
  projects: ["researcher", "data-scientist", "pi", "admin", "owner", "business-dev"],
  "shared-queries": ["researcher", "data-scientist", "pi", "admin", "owner"],
  annotations: ["researcher", "data-scientist", "pi", "admin", "owner"],

  // Data science — datasets, graph stats, quality
  "data-sources": ["data-scientist", "pi", "admin", "owner"],
  "graph-stats": ["data-scientist", "pi", "admin", "owner"],
  quality: ["data-scientist", "pi", "admin", "owner"],

  // Billing
  subscription: ["owner", "admin", "billing"],
  usage: ["owner", "admin", "billing"],
  deals: ["owner", "admin", "business-dev"],
  invoices: ["owner", "admin", "billing"],

  // Admin console
  users: ["admin", "owner"],
  roles: ["admin", "owner"],
  sso: ["admin", "owner"],
  "audit-logs": ["admin", "owner"],
  "feature-flags": ["admin", "owner"],

  // Developer platform
  "api-docs": ["developer", "admin", "owner"],
  "api-keys": ["developer", "admin", "owner"],
  playground: ["developer", "admin", "owner"],
  webhooks: ["developer", "admin", "owner"],

  // Investor relations
  "investor-dashboard": ["owner"],
  "cap-table": ["owner"],

  // More
  changelog: ["researcher", "data-scientist", "pi", "admin", "owner", "business-dev", "developer", "viewer"],
  roadmap: ["researcher", "data-scientist", "pi", "admin", "owner", "business-dev", "developer", "viewer"],
  feedback: ["researcher", "data-scientist", "pi", "admin", "owner", "business-dev", "developer", "viewer"],
};

/**
 * Returns true if the given role is allowed to access the given section.
 * Base sections are accessible to everyone; restricted sections require
 * the role to appear in the ROLE_SECTIONS list.
 */
export function canAccessSection(role: string | undefined | null, sectionId: string): boolean {
  if (!role) return false;
  if (BASE_SECTIONS.includes(sectionId)) return true;
  const allowed = ROLE_SECTIONS[sectionId];
  if (!allowed) return false; // unknown section — default deny
  return allowed.includes(role);
}

/**
 * Returns the list of section IDs the given role can access. Used to
 * filter the sidebar navigation.
 */
export function visibleSectionsForRole(role: string | undefined | null): string[] {
  if (!role) return BASE_SECTIONS;
  const all = [...BASE_SECTIONS];
  for (const [section, roles] of Object.entries(ROLE_SECTIONS)) {
    if (roles.includes(role)) all.push(section);
  }
  return all;
}

/** Human-readable label for each role. */
export const ROLE_LABELS: Record<string, string> = {
  viewer: "Viewer",
  researcher: "Researcher",
  "data-scientist": "Data Scientist",
  pi: "Principal Investigator",
  "business-dev": "Business Development",
  developer: "Developer",
  billing: "Billing",
  admin: "Administrator",
  owner: "Owner",
};

/** Returns a friendly label for a role, or the raw role string if unknown. */
export function roleLabel(role: string | undefined | null): string {
  if (!role) return "Unknown";
  return ROLE_LABELS[role] || role;
}
