use serde::Serialize;
use serde_json::Value;
use std::path::PathBuf;

use crate::retrieval::{RetrievalBudget, RetrievalStageReport, ScopeSpec};
use crate::tools::errors::ToolErrorBody;

/// Response from the search tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SearchResponse {
    pub schema: &'static str,
    pub phrase: String,
    pub mode: String,
    pub brief: bool,
    pub expanded_phrases: Vec<String>,
    pub hits: Vec<SearchHit>,
    /// Only when nothing matched exactly and the collection holds English
    /// (word-layout) text: the passages that share the most words with the
    /// phrase, best first, each with a `score`. They are leads, not matches:
    /// read one before quoting it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub related_passages: Option<Vec<SearchHit>>,
    /// Why there are no hits and what to try, when that is known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clusters: Option<Vec<ClusterHitsCluster>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_groups: Option<Vec<TermUsageGroup>>,
    pub search_strategy: SearchStrategy,
    /// Next-step candidates derived from the shape of *this* result set
    /// (hit count, work concentration, etc.) — varies call to call rather
    /// than repeating a fixed tip, and already-tried tools are filtered out.
    pub suggested_next_tools: Vec<SuggestedToolCall>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SearchHit {
    pub passage_id: String,
    pub source_work_id: Option<String>,
    pub main_title: Option<String>,
    pub heading: Option<String>,
    pub zh_quote: String,
    pub score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SearchStrategy {
    pub method: String,
    pub filters: Value,
    pub candidate_count: Option<usize>,
    pub verified_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verification_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub used_phrase_index: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ComponentStatus {
    Ok,
    SkippedNotRequested,
    SkippedUnavailable,
    SkippedBudgetExhausted,
    TimedOut,
    Failed,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct WorkflowComponent {
    pub name: String,
    pub tool: String,
    pub status: ComponentStatus,
    pub used: bool,
    pub elapsed_ms: Option<u128>,
    pub summary: Option<String>,
    pub error: Option<ToolErrorBody>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SuggestedToolCall {
    pub tool: String,
    pub args: serde_json::Value,
    pub reason: String,
}

/// Response from the heading-search tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct HeadingSearchResponse {
    pub schema: &'static str,
    pub query: String,
    pub brief: bool,
    pub returned_count: usize,
    pub sections: Vec<HeadingSearchHit>,
    pub search_strategy: SearchStrategy,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct HeadingSearchHit {
    pub source_work_id: Option<String>,
    pub main_title: Option<String>,
    pub heading: Option<String>,
    pub heading_path: Option<String>,
    pub passage_id: String,
    pub sample: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Response from the tool-docs tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ToolDocsResponse {
    pub schema: &'static str,
    pub tool: Option<String>,
    pub docs: serde_json::Value,
}

/// Response from the tool-log-summary tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ToolLogSummaryResponse {
    pub schema: &'static str,
    pub summary: serde_json::Value,
}

/// Response from the passage tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PassageResponse {
    pub schema: &'static str,
    pub passage_id: String,
    pub zh_quote: String,
    pub source_work_id: Option<String>,
    pub main_title: Option<String>,
    pub heading: Option<String>,
}

/// Response from the source-read tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SourceReadResponse {
    pub schema: &'static str,
    pub source_work_id: String,
    pub work_title: Option<String>,
    pub cursor: SourceReadCursorInfo,
    pub position: SourceReadPosition,
    pub segments: Vec<SourceReadSegment>,
    pub metadata: Option<serde_json::Value>,
    pub reading_state: SourceReadingState,
    pub suggested_next_tools: Vec<SuggestedToolCall>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SourceReadCursorInfo {
    pub current: String,
    pub next: Option<String>,
    pub prev: Option<String>,
    pub has_next: bool,
    pub has_prev: bool,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SourceReadPosition {
    pub char_start: usize,
    pub char_end: usize,
    pub total_chars: usize,
    pub estimated_percent: f64,
    pub passage_start: Option<String>,
    pub passage_end: Option<String>,
    pub section_path: Vec<String>,
    pub boundary_policy: String,
    pub boundary_quality: String,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SourceReadSegment {
    pub kind: String,
    pub citeable: bool,
    pub char_start: usize,
    pub char_end: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SourceReadingState {
    pub work_title: Option<String>,
    pub current_location: Option<String>,
    pub progress: serde_json::Value,
    pub running_summary_prompt: String,
}

/// Response from the canonical-source tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CanonicalSourceResponse {
    pub schema: &'static str,
    pub phrase: String,
    pub hits: Vec<CanonicalSourceHit>,
    pub search_strategy: SearchStrategy,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CanonicalSourceHit {
    pub passage_id: String,
    pub source_work_id: Option<String>,
    pub main_title: Option<String>,
    pub heading: Option<String>,
    pub zh_quote: String,
    pub is_canon_side: bool,
}

/// Response from the status tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct StatusResponse {
    pub schema: &'static str,
    pub data_root: String,
    pub passages_parquet_exists: bool,
    pub phrase_index_exists: bool,
    pub tfidf_index_exists: bool,
    pub vector_index_exists: bool,
    pub catalog_index_exists: bool,
    pub doc_table_exists: bool,
    pub registry_exists: bool,
}

/// Response from the validate-adjudication tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ValidateAdjudicationResponse {
    pub schema: &'static str,
    pub path: PathBuf,
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Response from the graph-build tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct GraphBuildResponse {
    pub schema: &'static str,
    pub out: PathBuf,
    pub node_count: usize,
    pub edge_count: usize,
}

/// Response from the report-build tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ReportBuildResponse {
    pub schema: &'static str,
    pub out: PathBuf,
    pub section_count: usize,
}

/// Response from the pdf-build tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PdfBuildResponse {
    pub schema: &'static str,
    pub out: PathBuf,
    pub source_format: String,
    pub section_count: usize,
    pub side_by_side: bool,
}

/// Response from the works tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct WorksResponse {
    pub schema: &'static str,
    pub works: Vec<WorkInfo>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct WorkInfo {
    pub work_id: String,
    pub main_title: String,
    pub author: Option<String>,
    pub period: Option<String>,
    pub canon: Option<String>,
    pub traditions: Vec<String>,
    pub passage_count: usize,
}

/// Response from the work-metadata tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct WorkMetadataResponse {
    pub schema: &'static str,
    pub requested_count: usize,
    pub found_count: usize,
    pub works: Vec<WorkInfo>,
    pub missing_work_ids: Vec<String>,
}

/// Response from the catalog-index-info tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CatalogIndexInfoResponse {
    pub schema: &'static str,
    #[serde(flatten)]
    pub info: serde_json::Value,
}

/// Response from the similar tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SimilarResponse {
    pub schema: &'static str,
    pub seed: String,
    pub similar_passages: Vec<serde_json::Value>,
}

/// Response from the frontier tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct FrontierResponse {
    pub schema: &'static str,
    pub seed_passage_id: String,
    #[serde(flatten)]
    pub payload: serde_json::Value,
}

/// Response from the first-attestation tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct FirstAttestationResponse {
    pub schema: &'static str,
    pub phrase: String,
    pub first: Option<serde_json::Value>,
    pub next_earlier: Vec<serde_json::Value>,
    pub scope: ScopeInfo,
    pub search_strategy: SearchStrategyInfo,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ScopeInfo {
    pub canon: Vec<String>,
    pub period: Vec<String>,
    pub source_work_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SearchStrategyInfo {
    pub used_phrase_index: bool,
    pub candidates_verified: usize,
    pub after_scope_and_sort: usize,
    pub limit: usize,
    pub max_candidates: usize,
}

/// Response from the phrase-history tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PhraseHistoryResponse {
    pub schema: &'static str,
    #[serde(flatten)]
    pub payload: serde_json::Value,
}

/// Response from the phrase-index-search tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PhraseIndexSearchResponse {
    pub schema: &'static str,
    pub phrase: String,
    pub returned_count: usize,
    pub limit: usize,
    pub search_strategy: serde_json::Value,
    pub results: Vec<serde_json::Value>,
}

/// Response from the seed-pick tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SeedPickResponse {
    pub schema: &'static str,
    pub limit: usize,
    pub already_worked_count: usize,
    pub filters: FilterInfo,
    pub candidates: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct FilterInfo {
    pub tradition: Vec<String>,
    pub period: Vec<String>,
}

/// Response from the expand-context-adaptive tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ExpandContextAdaptiveResponse {
    pub schema: &'static str,
    pub seed_passage_id: String,
    pub selected_node_id: u32,
    pub selected_node_kind: String,
    pub selected_label: String,
    pub heading_path: Vec<String>,
    pub work_id: Option<String>,
    pub passage_count: usize,
    pub char_count: usize,
    /// True when the selected node was larger than the budget and only the
    /// passages around the seed are returned.
    pub truncated: bool,
    /// What was left out and how to read on, when `truncated`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    pub passages: Vec<serde_json::Value>,
    pub search_strategy: SearchStrategyInfoAdaptive,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SearchStrategyInfoAdaptive {
    pub budget: usize,
    pub climbed_levels: u32,
    pub leaf_kind: String,
    pub mode: String,
}

/// Response from the trace-term-usage tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct TraceTermUsageResponse {
    pub schema: &'static str,
    pub phrase: String,
    pub group_by: String,
    pub groups: Vec<TermUsageGroup>,
    pub search_strategy: TermUsageSearchStrategy,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct TermUsageGroup {
    pub key: String,
    pub hit_count: u32,
    pub work_count: usize,
    pub top_works: Vec<String>,
    pub representative_passages: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct TermUsageSearchStrategy {
    pub used_phrase_index: bool,
    pub total_hits: usize,
    pub limit_total: usize,
    pub limit_per_group: usize,
    pub max_candidates: usize,
}

/// Response from the query-expand-terms tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct QueryExpandTermsResponse {
    pub schema: &'static str,
    pub input: String,
    pub expanded: Vec<String>,
    pub by_source: ExpandTermsBySource,
    pub search_strategy: ExpandTermsSearchStrategy,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ExpandTermsBySource {
    pub variants: Vec<String>,
    pub orthographic: Vec<String>,
    pub persons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ExpandTermsSearchStrategy {
    pub mode: String,
    pub max: usize,
    pub input_lang_guess: String,
}

/// Response from the compare-usage tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CompareUsageResponse {
    pub schema: &'static str,
    pub scope_a: CompareUsageScope,
    pub scope_b: CompareUsageScope,
    pub distinctive_to_a: Vec<CompareUsageTerm>,
    pub distinctive_to_b: Vec<CompareUsageTerm>,
    pub search_strategy: CompareUsageSearchStrategy,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CompareUsageScope {
    pub node_id: Option<u32>,
    pub work_id: Option<String>,
    pub canon: Option<String>,
    pub period: Option<String>,
    pub passage_count: usize,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CompareUsageTerm {
    pub term: Option<String>,
    pub term_hash: u64,
    pub score: f32,
    pub a_count: u32,
    pub b_count: u32,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CompareUsageSearchStrategy {
    pub gram_len: usize,
    pub limit_passages: usize,
    pub limit_terms: usize,
}

/// Response from the collocation-search tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CollocationSearchResponse {
    pub schema: &'static str,
    pub phrase: String,
    pub window_chars: usize,
    pub gram_len: usize,
    pub total_passages: usize,
    pub near_ngram_count: u32,
    pub background_ngram_count: u32,
    pub collocates: Vec<CollocateTerm>,
    pub search_strategy: CollocationSearchStrategy,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CollocateTerm {
    pub term: Option<String>,
    pub term_hash: u64,
    pub score: f32,
    pub near_count: u32,
    pub background_count: u32,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CollocationSearchStrategy {
    pub phrase: serde_json::Value,
    pub limit_total: usize,
    pub limit_collocates: usize,
}

/// Response from the pair-appearance tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PairAppearanceResponse {
    pub schema: &'static str,
    pub candidate_budget: RetrievalBudget,
    pub scope: ScopeSpec,
    pub term1: String,
    pub term2: String,
    pub unit: String,
    pub window_chars: usize,
    pub ordered: bool,
    pub total_term1_hits: usize,
    pub total_term2_hits: usize,
    pub pair_hit_count: usize,
    pub hits: Vec<PairAppearanceHit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_summary: Option<PairAppearanceNegativeSummary>,
    pub stages: Vec<RetrievalStageReport>,
    pub search_strategy: PairAppearanceSearchStrategy,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PairAppearanceHit {
    pub passage_id: String,
    pub source_work_id: Option<String>,
    pub main_title: Option<String>,
    pub heading: Option<String>,
    pub distance_chars: Option<usize>,
    pub term1_offsets: Vec<usize>,
    pub term2_offsets: Vec<usize>,
    pub zh_quote: String,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PairAppearanceNegativeSummary {
    pub term1_only_count: usize,
    pub term2_only_count: usize,
    pub sample_term1_only_passage_ids: Vec<String>,
    pub sample_term2_only_passage_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PairAppearanceSearchStrategy {
    pub term1: serde_json::Value,
    pub term2: serde_json::Value,
    pub unit: String,
    pub supported_units: Vec<&'static str>,
    pub used_variant_expansion: bool,
    pub max_candidates_per_term: usize,
}

/// Response from the outline-search tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct OutlineSearchResponse {
    pub schema: &'static str,
    pub phrase: String,
    pub start_node_id: u32,
    pub start_label: String,
    pub group_by: String,
    pub total_hits: usize,
    pub group_count: usize,
    pub groups: Vec<OutlineSearchGroup>,
    pub search_strategy: OutlineSearchStrategy,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct OutlineSearchGroup {
    pub node_id: u32,
    pub label: String,
    pub heading_path: String,
    pub node_kind: String,
    pub hit_count: u32,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct OutlineSearchStrategy {
    pub phrase: serde_json::Value,
    pub limit_total: usize,
    pub limit_per_group: usize,
}

/// Response from the cluster-hits tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ClusterHitsResponse {
    pub schema: &'static str,
    pub phrase: String,
    pub cluster_by: String,
    pub total_hits: usize,
    pub cluster_count: usize,
    pub clusters: Vec<ClusterHitsCluster>,
    pub search_strategy: ClusterHitsSearchStrategy,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ClusterHitsCluster {
    pub node_id: u32,
    pub label: String,
    pub heading_path: String,
    pub node_kind: String,
    pub hit_count: u32,
    pub representative_passages: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ClusterHitsSearchStrategy {
    pub phrase: serde_json::Value,
    pub limit_total: usize,
    pub limit_per_cluster: usize,
    pub max_candidates: usize,
}

/// Response from the absence-check tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct AbsenceCheckResponse {
    pub schema: &'static str,
    pub phrase: String,
    pub scope: AbsenceCheckScope,
    pub found: bool,
    pub hit_count: usize,
    pub sample_hits: Vec<serde_json::Value>,
    pub search_strategy: AbsenceCheckSearchStrategy,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct AbsenceCheckScope {
    pub work_id: Option<String>,
    pub canon: Vec<String>,
    pub period: Option<String>,
    pub node_id: Option<u32>,
    pub doc_range: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct AbsenceCheckSearchStrategy {
    pub phrase: serde_json::Value,
    pub limit: usize,
    pub max_candidates: usize,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct VectorInfoResponse {
    pub schema: &'static str,
    pub index_path: String,
    pub info: serde_json::Value,
    pub doc_table_fingerprint_match: bool,
    pub coverage: VectorCoverage,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct VectorCoverage {
    pub kind: String,
    pub covered_docs: u32,
    pub total_docs: u32,
    pub coverage_ratio: f64,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct VectorNeighborsResponse {
    pub schema: &'static str,
    pub query_mode: String,
    pub seed_passage_id: Option<String>,
    pub model_id: String,
    pub model_revision: String,
    pub embedding_dim: u32,
    pub distance: String,
    pub normalized: bool,
    pub rerank_requested: bool,
    pub rerank_applied: bool,
    pub score_interpretation: String,
    pub loading_index_ms: Option<u128>,
    pub hnsw_build_ms: Option<u128>,
    pub warnings: Vec<String>,
    pub hits: Vec<VectorNeighborHit>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct VectorNeighborHit {
    pub passage_id: String,
    pub doc_id: u32,
    pub ann_distance: f32,
    pub ann_score: f32,
    pub vector_score: f32,
    pub source_work_id: Option<String>,
    pub main_title: Option<String>,
    pub heading: Option<String>,
    pub period: Option<String>,
    pub snippet: Option<String>,
    pub warning: String,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct EvidenceSearchResponse {
    pub schema: &'static str,
    pub workflow: &'static str,
    pub candidate_budget: RetrievalBudget,
    pub scope: ScopeSpec,
    pub phrase: String,
    pub expanded_terms: Vec<String>,
    pub expanded_terms_used: bool,
    pub variant_policy: String,
    pub exact: SearchResponse,
    pub absence_check: Option<AbsenceCheckResponse>,
    pub first_attestation: Option<FirstAttestationResponse>,
    pub phrase_history: Option<PhraseHistoryResponse>,
    pub usage: Option<TraceTermUsageResponse>,
    pub clusters: Option<ClusterHitsResponse>,
    pub stages: Vec<RetrievalStageReport>,
    pub components: Vec<WorkflowComponent>,
    pub suggested_next_tools: Vec<SuggestedToolCall>,
    pub indexes_used: Vec<String>,
    pub fallbacks: Vec<String>,
    pub evidence_status: String,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct HybridDiscoverResponse {
    pub schema: &'static str,
    pub workflow: &'static str,
    pub mode: String,
    pub mode_reason: String,
    pub candidate_budget: RetrievalBudget,
    pub merged_candidate_count: usize,
    pub returned_count: usize,
    pub seed_passage_id: Option<String>,
    pub vector_neighbors: Option<VectorNeighborsResponse>,
    pub tfidf_similar: Option<SimilarResponse>,
    pub context: Option<ExpandContextAdaptiveResponse>,
    pub groups: HybridDiscoverGroups,
    pub merged_hits: Vec<HybridDiscoverHit>,
    pub stages: Vec<RetrievalStageReport>,
    pub components: Vec<WorkflowComponent>,
    pub suggested_next_tools: Vec<SuggestedToolCall>,
    pub indexes_used: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct HybridDiscoverGroups {
    pub semantic_candidates: Vec<String>,
    pub lexical_parallels: Vec<String>,
    pub overlap_candidates: Vec<String>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct HybridDiscoverHit {
    pub passage_id: String,
    pub labels: Vec<String>,
    pub candidate_sources: Vec<String>,
    pub evidence_status: String,
    pub vector_score: Option<f32>,
    pub vector_rank: Option<usize>,
    pub tfidf_score: Option<f32>,
    pub tfidf_rank: Option<usize>,
    pub semantic_score: Option<f32>,
    pub lexical_score: Option<f32>,
    pub final_score: f32,
    pub merged_rank_reason: String,
    pub title: Option<String>,
    pub snippet: Option<String>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct SourceInvestigateResponse {
    pub schema: &'static str,
    pub workflow: &'static str,
    pub seed_passage_id: String,
    pub seed: PassageResponse,
    pub context: Option<ExpandContextAdaptiveResponse>,
    pub frontier: Option<FrontierResponse>,
    pub similar: Option<SimilarResponse>,
    pub vector_neighbors: Option<VectorNeighborsResponse>,
    pub phrase_histories: Vec<PhraseHistoryResponse>,
    pub components: Vec<WorkflowComponent>,
    pub suggested_next_tools: Vec<SuggestedToolCall>,
    pub risk_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ScopeProfileResponse {
    pub schema: &'static str,
    pub workflow: &'static str,
    pub phrase: Option<String>,
    pub comparison: CompareUsageResponse,
    pub term_usage: Option<TraceTermUsageResponse>,
    pub components: Vec<WorkflowComponent>,
    pub suggested_next_tools: Vec<SuggestedToolCall>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ReportFromEvidenceResponse {
    pub schema: &'static str,
    pub workflow: &'static str,
    pub validation: ValidateAdjudicationResponse,
    pub graph: Option<GraphBuildResponse>,
    pub report: Option<ReportBuildResponse>,
    pub components: Vec<WorkflowComponent>,
    pub suggested_next_tools: Vec<SuggestedToolCall>,
    pub warnings: Vec<String>,
}

/// Response from the batch-evidence-search tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct BatchEvidenceSearchResponse {
    pub schema: &'static str,
    pub results: Vec<BatchEvidenceSearchResult>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct BatchEvidenceSearchResult {
    pub phrase: String,
    /// Backward-compatible alias for `returned_count`; this is not guaranteed
    /// to be the corpus-wide total when it reaches the requested limit.
    pub hit_count: usize,
    pub returned_count: usize,
    pub possibly_truncated: bool,
    pub sample_passage_ids: Vec<String>,
    pub error: Option<String>,
}

/// Response from the pair-profile tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PairProfileResponse {
    pub schema: &'static str,
    pub term1: String,
    pub term2: String,
    pub unit: String,
    pub group_by: String,
    pub total_term1_hits: usize,
    pub total_term2_hits: usize,
    pub total_pair_hits: usize,
    pub groups: Vec<PairProfileGroup>,
    pub search_strategy: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PairProfileGroup {
    pub group_label: String,
    pub term1_count: usize,
    pub term2_count: usize,
    pub pair_count: usize,
    /// pair_count / term1_count (0 if no term1 hits in group)
    pub pair_rate_given_term1: f64,
    /// pair_count / term2_count (0 if no term2 hits in group)
    pub pair_rate_given_term2: f64,
    pub representative_hits: Vec<PairAppearanceHit>,
}

/// Response from the person-resolve tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PersonResolveResponse {
    pub schema: &'static str,
    pub name: String,
    pub aliases: Vec<String>,
    pub canonical_candidate: String,
    /// DDBC authority record, if found (null if not in authority database).
    pub authority: Option<serde_json::Value>,
    pub name_forms: Vec<serde_json::Value>,
    pub ambiguity_notes: Vec<String>,
    pub evidence: Vec<serde_json::Value>,
    pub caveats: Vec<String>,
    pub suggested_next: Vec<String>,
}

/// Response from the place-resolve tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PlaceResolveResponse {
    pub schema: &'static str,
    pub name: String,
    pub aliases: Vec<String>,
    /// DDBC authority record, if found (null if not in authority database).
    pub authority: Option<serde_json::Value>,
    pub name_forms: Vec<serde_json::Value>,
    pub ambiguity_notes: Vec<String>,
    pub evidence: Vec<serde_json::Value>,
    pub caveats: Vec<String>,
    pub suggested_next: Vec<String>,
}

/// Response from the person-history tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PersonHistoryResponse {
    pub schema: &'static str,
    pub name: String,
    pub aliases: Vec<String>,
    /// All name forms actually searched (primary name + aliases).
    pub aliases_searched: Vec<String>,
    pub canonical_candidate: String,
    pub total_mentions: usize,
    pub limit: usize,
    /// Full mention rows. Empty when compact=true (use compact_summary instead).
    pub mentions: Vec<serde_json::Value>,
    /// Populated when compact=true: hits grouped by mention_class × period with counts and samples.
    pub compact_summary: Option<serde_json::Value>,
    pub earliest_unambiguous: serde_json::Value,
    pub ambiguous_earlier_hits: Vec<serde_json::Value>,
    pub evidence: Vec<serde_json::Value>,
    pub caveats: Vec<String>,
    /// Non-empty when the name is ≤2 chars and may be a common Chinese compound.
    pub false_positive_warning: Option<String>,
    pub suggested_next: Vec<String>,
}

/// Response from the person-profile tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PersonProfileResponse {
    pub schema: &'static str,
    pub name: String,
    /// "fully_attested" | "partially_attested" | "needs_filtered_search" | "not_found"
    pub status: String,
    /// "none" | "low" | "high"
    pub false_positive_risk: String,
    /// Populated when false_positive_risk is "high".
    pub false_positive_warning: Option<String>,
    /// DDBC authority record (bio, birth/death years, dynasty, teachers, students, etc.).
    pub authority: Option<serde_json::Value>,
    pub known_aliases: Vec<String>,
    /// Compact corpus mention summary grouped by mention_class.
    pub corpus_summary: Option<serde_json::Value>,
    pub caveats: Vec<String>,
}

/// Response from the citation-verify tool
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CitationVerifyResponse {
    pub schema: &'static str,
    pub quote: String,
    pub claimed_attribution: Option<String>,
    pub scope_source_work_id: Option<String>,
    pub scope_canon: Option<String>,
    pub verified: bool,
    pub exact_hit_count: usize,
    pub exact_hits: Vec<serde_json::Value>,
    pub near_matches: Vec<CitationNearMatch>,
    pub verdict: String,
    pub search_strategy: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CitationNearMatch {
    pub passage_id: String,
    pub source_work_id: Option<String>,
    pub main_title: Option<String>,
    pub heading: Option<String>,
    pub overlap_score: f64,
    pub zh_quote: String,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct RunBatchResponse {
    pub out: PathBuf,
    pub jobs_total: usize,
    pub jobs_ok: usize,
    pub jobs_failed: usize,
    pub elapsed_ms: u128,
}
