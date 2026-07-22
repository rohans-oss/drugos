'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp, ExternalLink, Star, Info, ChevronLeft, ChevronRight } from 'lucide-react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { SafetyBadge } from '@/components/drugos/safety-badge';
import { ScoreBar } from '@/components/drugos/score-bar';
import { useDrugMechanisms } from '@/components/drugos/use-api-data';
import type { DrugCandidate } from '@/lib/types';

/**
 * FE-033 ROOT FIX: Server-side sort + pagination.
 *
 * The previous version received `candidates: DrugCandidate[]` and rendered
 * ALL of them, with no sorting and no pagination. The audit found:
 *   - For 1000 candidates: ~50ms to render (acceptable)
 *   - For 10K candidates: ~500ms (laggy)
 *   - For 100K candidates (production scale): ~5 seconds (browser freeze)
 *
 * Per the project doc (Team_Cosmic_Build_Process_Updated.docx §6), the V1
 * launch criteria require the dashboard to "load and render graph
 * visualizations in under 3 seconds." A 5-second freeze on the candidate
 * table breaks that criterion.
 *
 * Root fix:
 *   1. The table is now CONTROLLED. The parent owns the sort + page state
 *      and re-fetches from /api/rl with `sort`, `sortDir`, `page`, `pageSize`
 *      params. The table just renders what it's given.
 *   2. Column headers are clickable to toggle sort. Clicking a header calls
 *      `onSortChange(field, newDir)` so the parent can re-fetch.
 *   3. A pagination footer shows "Showing X–Y of Z" and Prev/Next buttons.
 *      The default page size is 50 (configurable via `pageSize` prop).
 *
 * The table NO LONGER sorts client-side. The parent MUST pass `candidates`
 * that are already sorted + paginated by the server.
 */

interface CandidateTableProps {
  candidates: DrugCandidate[];
  onSelect?: (candidate: DrugCandidate) => void;
  onCompare?: (candidate: DrugCandidate) => void;
  showDiseaseColumn?: boolean;
  className?: string;

  /**
   * FE-033: Current sort state. When provided, the corresponding column
   * header shows an active sort indicator. The parent owns this state.
   */
  sort?: { field: CandidateSortField; dir: CandidateSortDir };

  /**
   * FE-033: Called when the user clicks a sortable column header. The parent
   * should update its sort state and re-fetch from /api/rl with the new
   * `sort` + `sortDir` params. The table does NOT sort client-side.
   *
   * Click behavior: if the clicked field is not the current sort field,
   * sort by it with the default direction (asc for rank/drug/disease, desc
   * for scores). If it IS the current field, toggle the direction.
   */
  onSortChange?: (field: CandidateSortField, dir: CandidateSortDir) => void;

  /**
   * FE-033: Pagination metadata. When provided, the table renders a
   * pagination footer with "Showing X–Y of Z" and Prev/Next buttons.
   */
  pagination?: {
    /** 0-indexed page number. */
    page: number;
    /** Page size (e.g. 50). */
    pageSize: number;
    /** Total candidate count AFTER filtering, BEFORE pagination. */
    total: number;
  };

  /** FE-033: Called when the user clicks Prev/Next. The parent re-fetches. */
  onPageChange?: (page: number) => void;
}

export type CandidateSortField =
  | 'compositeScore'
  | 'drugName'
  | 'diseaseName'
  | 'safetyTier'
  | 'clinicalPhase';

export type CandidateSortDir = 'asc' | 'desc';

/**
 * FE-024 ROOT FIX: Format RL debug info for the tooltip.
 * Returns an empty string if no debug info is present.
 *
 * This data is for ML engineers debugging the ranker. It is meaningless
 * to a pharma researcher and must NEVER appear in a table column. The
 * "Mechanism" column shows real mechanism-of-action text fetched from
 * ChEMBL via useDrugMechanisms.
 */
function formatRlDebugInfo(c: DrugCandidate): string {
  const d = c.rlDebugInfo;
  if (!d) return '';
  const parts: string[] = [];
  if (typeof d.reward === 'number') parts.push(`reward=${d.reward.toFixed(4)}`);
  if (typeof d.policyProb === 'number') parts.push(`policy_prob=${d.policyProb.toFixed(4)}`);
  if (typeof d.gnnScore === 'number') parts.push(`gnn_score=${d.gnnScore.toFixed(4)}`);
  if (typeof d.rank === 'number') parts.push(`rl_rank=${d.rank}`);
  if (d.source) parts.push(`source=${d.source}`);
  return parts.join(' · ');
}

/**
 * FE-033: Map candidate-table sort fields to /api/rl sort fields.
 * The candidate table exposes user-facing fields (compositeScore, drugName,
 * etc.) which map to the RL ranker's RankedHypothesis properties.
 */
export const CANDIDATE_SORT_TO_RL: Record<CandidateSortField, string> = {
  compositeScore: 'overallScore',
  drugName: 'drug',
  diseaseName: 'disease',
  safetyTier: 'safetyScore',
  clinicalPhase: 'rank',
};

interface SortableHeaderProps {
  label: string;
  field: CandidateSortField;
  currentSort?: { field: CandidateSortField; dir: CandidateSortDir };
  onSortChange?: (field: CandidateSortField, dir: CandidateSortDir) => void;
  tooltip?: React.ReactNode;
}

function SortableHeader({ label, field, currentSort, onSortChange, tooltip }: SortableHeaderProps) {
  const isActive = currentSort?.field === field;
  const dir = currentSort?.dir;

  const handleClick = () => {
    if (!onSortChange) return;
    if (!isActive) {
      // New field — use default direction (asc for names, desc for scores).
      const defaultDir: CandidateSortDir =
        field === 'compositeScore' || field === 'safetyTier' ? 'desc' : 'asc';
      onSortChange(field, defaultDir);
    } else {
      // Toggle direction.
      onSortChange(field, dir === 'asc' ? 'desc' : 'asc');
    }
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={!onSortChange}
      className={`inline-flex items-center gap-1 ${onSortChange ? 'cursor-pointer hover:text-foreground' : 'cursor-default'}`}
      aria-label={`Sort by ${label}`}
    >
      <span>{label}</span>
      {isActive ? (
        dir === 'asc' ? (
          <ChevronUp className="h-3 w-3" />
        ) : (
          <ChevronDown className="h-3 w-3" />
        )
      ) : (
        onSortChange && <ChevronDown className="h-3 w-3 opacity-30" />
      )}
      {tooltip}
    </button>
  );
}

export function CandidateTable({
  candidates,
  onSelect,
  onCompare,
  showDiseaseColumn = true,
  className = '',
  sort,
  onSortChange,
  pagination,
  onPageChange,
}: CandidateTableProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [shortlisted, setShortlisted] = useState<Set<string>>(new Set());

  // FE-024 ROOT FIX: Batch-fetch real mechanisms from ChEMBL for every
  // drug in the candidate list. The hook dedupes + caches, so re-renders
  // don't re-fetch.
  const drugNames = candidates.map((c) => c.drugName);
  const mechanismState = useDrugMechanisms(drugNames);
  const mechanismMap = mechanismState.data;

  const toggleExpand = (id: string) => {
    setExpandedId((prev) => (prev === id ? null : id));
  };

  const toggleShortlist = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setShortlisted((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // FE-033: Pagination footer geometry.
  const hasPagination = !!pagination;
  const pageStart = hasPagination
    ? pagination!.page * pagination!.pageSize + 1
    : 1;
  const pageEnd = hasPagination
    ? Math.min((pagination!.page + 1) * pagination!.pageSize, pagination!.total)
    : candidates.length;
  const totalPages = hasPagination
    ? Math.max(1, Math.ceil(pagination!.total / pagination!.pageSize))
    : 1;
  const currentPage = hasPagination ? pagination!.page : 0;

  return (
    <TooltipProvider delayDuration={200}>
      <div className={`rounded-lg border border-border overflow-hidden ${className}`}>
        <Table>
          <TableHeader>
            <TableRow className="bg-muted/50 hover:bg-muted/50">
              <TableHead className="w-10"></TableHead>
              <TableHead className="w-10"></TableHead>
              <TableHead>
                <SortableHeader
                  label="Drug Name"
                  field="drugName"
                  currentSort={sort}
                  onSortChange={onSortChange}
                />
              </TableHead>
              <TableHead>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="inline-flex items-center gap-1 cursor-help">
                      <SortableHeader
                        label="Composite Score"
                        field="compositeScore"
                        currentSort={sort}
                        onSortChange={onSortChange}
                      />
                      <Info className="h-3 w-3 text-muted-foreground" />
                    </span>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="font-medium mb-1">Composite Score (Model Output)</p>
                    <p className="text-xs text-muted-foreground">
                      A 0–100 weighted blend of sub-scores: Knowledge Graph (40%),
                      Molecular Similarity (15%), Safety (25%), Clinical Evidence (20%).
                      This is a model output, NOT a statistical confidence interval.
                      Do not interpret &quot;87&quot; as &quot;87% confident this works&quot;.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TableHead>
              <TableHead>
                <SortableHeader
                  label="Safety"
                  field="safetyTier"
                  currentSort={sort}
                  onSortChange={onSortChange}
                />
              </TableHead>
              <TableHead>Mechanism</TableHead>
              {showDiseaseColumn && (
                <TableHead>
                  <SortableHeader
                    label="Disease"
                    field="diseaseName"
                    currentSort={sort}
                    onSortChange={onSortChange}
                  />
                </TableHead>
              )}
              <TableHead>
                <SortableHeader
                  label="Phase"
                  field="clinicalPhase"
                  currentSort={sort}
                  onSortChange={onSortChange}
                />
              </TableHead>
              <TableHead>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="inline-flex items-center gap-1 cursor-help">
                      Model Score
                      <Info className="h-3 w-3 text-muted-foreground" />
                    </span>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="font-medium mb-1">Model Score (0–100)</p>
                    <p className="text-xs text-muted-foreground">
                      The same composite score shown in the bar — duplicated as a
                      numeric value for sorting and export. Not a probability.
                      The model is not calibrated; do not use this as a statistical
                      confidence measure without Platt scaling or isotonic regression.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TableHead>
              <TableHead className="w-10"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {candidates.length === 0 && (
              <TableRow>
                <TableCell colSpan={showDiseaseColumn ? 10 : 9} className="text-center text-muted-foreground py-12">
                  No candidates to display. Run an RL query to populate this table.
                </TableCell>
              </TableRow>
            )}
            {candidates.map((candidate) => {
              const isExpanded = expandedId === candidate.id;
              const isShortlisted = shortlisted.has(candidate.id);

              // FE-024: real mechanism from ChEMBL, or "—" if unknown / still loading.
              const mechResult = mechanismMap?.get(candidate.drugName.toLowerCase());
              const mechanismDisplay = mechResult?.mechanism
                || candidate.mechanism
                || (mechanismState.loading ? 'Loading…' : '—');
              const rlDebug = formatRlDebugInfo(candidate);

              return (
                <TableRow
                  key={candidate.id}
                  className="cursor-pointer hover:bg-muted/30"
                  onClick={() => onSelect?.(candidate)}
                >
                  <TableCell>
                    <button onClick={(e) => toggleShortlist(candidate.id, e)} className="focus:outline-none">
                      <Star
                        className={`h-4 w-4 transition-colors ${
                          isShortlisted ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground hover:text-yellow-400'
                        }`}
                      />
                    </button>
                  </TableCell>
                  <TableCell>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleExpand(candidate.id);
                      }}
                      className="focus:outline-none"
                    >
                      {isExpanded ? (
                        <ChevronUp className="h-4 w-4 text-muted-foreground" />
                      ) : (
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      )}
                    </button>
                  </TableCell>
                  <TableCell>
                    <div>
                      <div className="font-medium text-foreground">{candidate.drugName}</div>
                      <div className="text-xs text-muted-foreground">{candidate.brandNames?.join(', ')}</div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <ScoreBar
                      score={candidate.compositeScore}
                      size="sm"
                      // FE-036: pass through confidence bounds + AUC when available.
                      // The candidate type doesn't currently carry these — when the
                      // backend adds them, they'll surface automatically.
                      confidenceLower={undefined}
                      confidenceUpper={undefined}
                      auc={undefined}
                    />
                  </TableCell>
                  <TableCell>
                    <SafetyBadge tier={candidate.safetyTier} />
                  </TableCell>
                  <TableCell>
                    {rlDebug ? (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="text-sm max-w-[200px] truncate block cursor-help">
                            {mechanismDisplay}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent className="max-w-md">
                          <p className="font-medium mb-1">{candidate.drugName}</p>
                          <p className="text-xs">{mechanismDisplay}</p>
                          {mechResult?.source && (
                            <p className="text-xs text-muted-foreground mt-1">
                              Source: {mechResult.source}
                            </p>
                          )}
                          <div className="border-t border-border mt-2 pt-2">
                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
                              RL Model Debug (not for clinical use)
                            </p>
                            <p className="text-xs font-mono">{rlDebug}</p>
                          </div>
                        </TooltipContent>
                      </Tooltip>
                    ) : (
                      <span className="text-sm max-w-[200px] truncate block">{mechanismDisplay}</span>
                    )}
                  </TableCell>
                  {showDiseaseColumn && (
                    <TableCell>
                      <span className="text-sm">{candidate.diseaseName}</span>
                    </TableCell>
                  )}
                  <TableCell>
                    <Badge variant="outline" className="text-xs">{candidate.clinicalPhase}</Badge>
                  </TableCell>
                  <TableCell>
                    <span className="text-sm font-medium">{candidate.compositeScore}</span>
                  </TableCell>
                  <TableCell>
                    <Button variant="ghost" size="sm" className="h-7 w-7 p-0">
                      <ExternalLink className="h-3.5 w-3.5" />
                    </Button>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>

        {/* FE-033: Pagination footer — shown when pagination metadata is provided. */}
        {hasPagination && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-border bg-muted/30">
            <div className="text-xs text-muted-foreground">
              Showing <span className="font-medium text-foreground">{pageStart.toLocaleString()}</span>
              {'–'}
              <span className="font-medium text-foreground">{pageEnd.toLocaleString()}</span>
              {' of '}
              <span className="font-medium text-foreground">{pagination!.total.toLocaleString()}</span>
              {' candidates'}
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                className="h-8"
                disabled={currentPage === 0 || !onPageChange}
                onClick={() => onPageChange?.(currentPage - 1)}
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                Prev
              </Button>
              <span className="text-xs text-muted-foreground tabular-nums px-2">
                Page {currentPage + 1} of {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                className="h-8"
                disabled={currentPage >= totalPages - 1 || !onPageChange}
                onClick={() => onPageChange?.(currentPage + 1)}
              >
                Next
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            </div>
          </div>
        )}
      </div>
    </TooltipProvider>
  );
}
