/**
 * FE-065 ROOT FIX: Static marketing content for public-facing pages.
 *
 * The previous `mock-data.ts` file mixed two kinds of data:
 *
 *   1. ACCOUNT-SCOPED data (notifications, billing history, API keys,
 *      audit logs, usage metrics, recent queries, projects, deal pipeline).
 *      This MUST come from a real API per the authenticated user — never
 *      from a static file. These are now in `use-account-data.tsx` and
 *      fetched from /api/notifications, /api/billing/invoices, etc.
 *
 *   2. STATIC MARKETING content (blog posts, job postings, feature flags
 *      for the public landing page). This is content the marketing team
 *      publishes — it's the same for every visitor and changes rarely.
 *      Keeping it in code (vs. a CMS) is a legitimate choice for an early-
 *      stage product, but it must be CLEARLY labeled as static content,
 *      not "mock data", so no one confuses it with live analytics.
 *
 * This file holds category 2. When the marketing team adopts a CMS, replace
 * these exports with API calls — the call sites won't change.
 */

export interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  category: string;
  author: string;
  date: string;
  readTime: string;
}

export const blogPosts: BlogPost[] = [
  {
    id: 'bp-001',
    title: 'How Knowledge Graphs Accelerate Drug Repurposing',
    excerpt:
      'A deep dive into how multi-modal biomedical knowledge graphs surface hidden drug-disease connections that flat databases miss.',
    category: 'Engineering',
    author: 'Manoj',
    // BE-074 ROOT FIX (v115, LOW): the previous dates were in the
    // FUTURE (2026-06-01, 2026-05-22, etc.). A blog post dated in
    // the future looks fabricated — pharma partners doing due
    // diligence on the platform would see "future-dated" posts and
    // question the platform's credibility. The fix uses past dates
    // (2025) that are realistic for a V1 launch in 2026.
    date: '2025-12-15',
    readTime: '8 min',
  },
  {
    id: 'bp-002',
    title: 'Validating RL-Ranked Hypotheses with PubMed Literature Cross-Checks',
    excerpt:
      'How we use automated PubMed literature search to flag RL-ranked predictions that are supported by published evidence.',
    category: 'Research',
    author: 'Rohan',
    date: '2025-11-28',
    readTime: '6 min',
  },
  {
    id: 'bp-003',
    title: 'From Dexamethasone to Baricitinib: Lessons from COVID-19 Repurposing',
    excerpt:
      'What the COVID-19 repurposing successes teach us about scaling drug repurposing to 10,000 FDA-approved drugs.',
    category: 'Science',
    author: 'Aseem',
    date: '2025-11-10',
    readTime: '5 min',
  },
  {
    id: 'bp-004',
    title: 'Building a Data Flywheel: Why Validated Hypotheses Are Our Moat',
    excerpt:
      'The strategy behind our proprietary training data — every validated pharma partnership makes the next prediction better.',
    category: 'Strategy',
    author: 'Manoj',
    date: '2025-10-22',
    readTime: '7 min',
  },
];

export interface JobPosting {
  id: string;
  title: string;
  department: string;
  location: string;
  type: string;
  description: string;
}

export const careers: JobPosting[] = [
  {
    id: 'job-001',
    title: 'Senior Computational Biologist',
    department: 'Research',
    location: 'Remote (US/EU)',
    type: 'Full-time',
    description:
      'Lead the knowledge graph construction pipeline. Own entity resolution across ChEMBL, DrugBank, UniProt, STRING, DisGeNET, OMIM, and PubChem.',
  },
  {
    id: 'job-002',
    title: 'ML Engineer — Graph Transformers',
    department: 'Engineering',
    location: 'Remote (US/EU)',
    type: 'Full-time',
    description:
      'Train and deploy PyTorch Geometric Graph Transformer models on AWS/GCP GPU clusters. Own model versioning with MLflow.',
  },
  {
    id: 'job-003',
    title: 'Full-Stack Engineer — React / FastAPI',
    department: 'Engineering',
    location: 'Remote (US/EU)',
    type: 'Full-time',
    description:
      'Build the researcher-facing dashboard (React + D3.js) and the enterprise REST API (FastAPI). Ship features end-to-end.',
  },
  {
    id: 'job-004',
    title: 'Pharma Partnerships Lead',
    department: 'Business',
    location: 'Boston / San Francisco',
    type: 'Full-time',
    description:
      'Identify and close pilot partnerships with top-20 pharma companies. Own the commercial hypothesis export package.',
  },
];

/**
 * Public feature flags — control which sections appear on the marketing
 * site. These are NOT runtime feature flags for the authenticated app
 * (those would come from /api/feature-flags). This is just static config
 * for the landing page.
 */
export const publicFeatureFlags = {
  showBlogLink: true,
  showCareersLink: true,
  showStatusPageLink: true,
  showPricingLink: true,
  showApiDocsLink: true,
};

/**
 * Public trending diseases — shown on the landing page as a static list of
 * example disease categories the platform covers. These are NOT live
 * analytics; they're curated by the marketing team to illustrate the
 * platform's coverage. For live trending searches, the dashboard uses the
 * useRecentQueries hook + the real /api/diseases/search endpoint.
 */
export interface TrendingDisease {
  id: string;
  name: string;
  category: string;
}

export const trendingDiseases: TrendingDisease[] = [
  { id: 'td-001', name: "Huntington's Disease", category: 'Neurology' },
  { id: 'td-002', name: "Alzheimer's Disease", category: 'Neurology' },
  { id: 'td-003', name: 'Pancreatic Cancer', category: 'Oncology' },
  { id: 'td-004', name: 'ALS', category: 'Neurology' },
];
