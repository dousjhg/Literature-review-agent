export interface Evidence {
  paper_id: string;
  quote: string;
}
export interface Paper {
  id: string;
  title: string;
  source: string;
  year?: number | null;
  abstract?: string | null;
  full_text?: string | null;
  authors?: string[];
  doi?: string | null;
}
export interface Alternative {
  cluster_id: string;
  similarity: number;
  min_similarity: number;
  meets_membership_criteria: boolean;
  reason: string;
}
export interface Detail {
  paper_id: string;
  question_similarity: number;
  relevance: string;
  reason: string;
  evidence: Evidence[];
  keywords: string[];
  assigned_similarity?: number | null;
  also_related: Alternative[];
}
export interface Cluster {
  id: string;
  label: string;
  kind?: "main" | "side";
  paper_ids: string[];
  rationale: string;
  paper_details?: Detail[];
}
export interface Connection {
  id: string;
  kind: string;
  paper_ids: string[];
  cluster_ids: string[];
  description: string;
  evidence: Evidence[];
  relation?: string;
}
export interface Gap {
  id: string;
  cluster_id: string;
  cluster_ids?: string[];
  scope?: string;
  between_paper_ids: string[];
  description: string;
  why_it_matters: string;
  evidence: Evidence[];
  proposed_test?: string;
  hypothesis_status?: string;
  coverage?: {
    status: string;
    reason: string;
    scope: string;
    searches: { query: string; status: string; papers: Paper[] }[];
  };
}
export interface Grade {
  accuracy: number;
  appropriateness: number;
  actionability: number;
  reasoning: string;
}
export interface MapResult {
  run_id: string;
  root_run_id: string;
  parent_run_id: string | null;
  revision: number;
  version_token: string;
  status: "done" | "partial";
  question: string;
  sample_mode: boolean;
  stage: string;
  clusters: Cluster[];
  connections: Connection[];
  gaps: Gap[];
  warnings: string[];
  grade?: Grade | null;
  analysis?: { pending_stages: string[]; grades: Record<string, Grade> } | null;
}
export interface CatalogItem {
  run_id: string;
  root_run_id: string;
  question: string;
  status: string;
  stage: string;
  revision: number;
  active: boolean;
  sample_mode: boolean;
  paper_count: number;
}
export interface Head {
  root_run_id: string;
  run_id: string;
  version_token: string;
}
export interface Versions {
  root_run_id: string;
  active_head: Head;
  versions: {
    run_id: string;
    parent_run_id: string | null;
    revision: number;
    status: string;
    version_token: string;
  }[];
}
export interface Candidate {
  paper_id: string;
  source_cluster_id: string;
  target_cluster_id: string;
  eligible: boolean;
  target_similarity: number;
  target_min_similarity: number;
  reason: string;
}
export interface Move {
  paper_id: string;
  target_cluster_id: string;
}
export interface Plan {
  preview_id: string;
  applicable: boolean;
  errors: string[];
  changes: { cluster_id: string; before: string[]; after: string[] }[];
  relabel_cluster_ids: string[];
  reused_label_cluster_ids: string[];
}
export interface Annotation {
  verdict: string;
  reviewer: string;
  rationale: string;
  evidence: Evidence[];
  saw_model_assessment: boolean;
}
export interface ReviewItem {
  id: string;
  kind: "connection" | "gap";
  scope: string;
  paper_ids: string[];
  cluster_ids: string[];
  claim: string | null;
  proposed_test: string | null;
  papers: {
    id: string;
    title: string;
    year: number | null;
    source_field: string;
    source_text: string;
  }[];
  annotation: Annotation | null;
  model_assessment?: { decision: string; reason?: string };
}
export interface ReviewPack {
  run_id: string;
  version_token: string;
  question: string;
  sample_mode: boolean;
  items: ReviewItem[];
  summary: {
    total: number;
    reviewed: number;
    pending: number;
    supported: number;
    unsupported: number;
    uncertain: number;
    semantic_accuracy: null;
  };
}
export type Selection = {
  type: "paper" | "cluster" | "connection" | "gap";
  id: string;
};
export type Filter = "all" | "main_side" | "connections" | "gaps";
