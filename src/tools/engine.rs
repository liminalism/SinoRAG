use crate::catalog_index::CorpusCatalogIndex;
use crate::datafusion_store::DataFusionStore;
use crate::document_table::DocumentTable;
use crate::phrase_index::PhraseIndex;
use crate::tfidf::TfidfIndex;
use crate::tools::errors::ToolError;
use crate::vector_index::VectorIndex;
use anyhow::Result;
use rustc_hash::FxHashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{OnceCell, Semaphore};

/// Configuration for the ToolEngine
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub pack: Option<PathBuf>,
    pub passages_parquet: Option<PathBuf>,
    pub phrase_index: Option<PathBuf>,
    pub tfidf_index: Option<PathBuf>,
    pub vector_index: Option<PathBuf>,
    pub catalog_index: Option<PathBuf>,
    pub doc_table: Option<PathBuf>,
    pub registry: Option<PathBuf>,
    pub readonly: bool,
    pub allow_admin_tools: bool,
    pub output_root: Option<PathBuf>,
    pub max_heavy_concurrency: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            pack: None,
            passages_parquet: None,
            phrase_index: None,
            tfidf_index: None,
            vector_index: None,
            catalog_index: None,
            doc_table: None,
            registry: None,
            readonly: false,
            allow_admin_tools: false,
            output_root: None,
            max_heavy_concurrency: 1,
        }
    }
}

/// The ToolEngine owns shared state and manages lazy-loaded resources
pub struct ToolEngine {
    pub config: EngineConfig,

    // Lazy-loaded heavy resources
    passages: OnceCell<Arc<DataFusionStore>>,
    phrase: OnceCell<Arc<PhraseIndex>>,
    tfidf: OnceCell<Arc<TfidfIndex>>,
    vector: OnceCell<Arc<VectorIndex>>,
    catalog: OnceCell<Arc<CorpusCatalogIndex>>,
    doc_table: OnceCell<Arc<DocumentTable>>,
    registry: OnceCell<Arc<()>>, // Placeholder - Registry may not exist yet

    heavy_slots: Semaphore,
}

fn expand_optional_filter(value: Option<&str>) -> Vec<String> {
    value
        .map(|v| crate::commands::search::expand_values(&[v.to_string()]))
        .unwrap_or_default()
}

/// Like `expand_optional_filter` but also resolves numeric tradition IDs.
fn expand_tradition_filter(value: Option<&str>) -> Vec<String> {
    expand_optional_filter(value)
        .into_iter()
        .map(|t| crate::taxonomy_legend::resolve_tradition(&t).to_string())
        .collect()
}

/// Like `expand_optional_filter` but also resolves numeric period IDs.
fn expand_period_filter(value: Option<&str>) -> Vec<String> {
    expand_optional_filter(value)
        .into_iter()
        .map(|p| crate::taxonomy_legend::resolve_period(&p).to_string())
        .collect()
}

/// Like `expand_optional_filter` but also resolves numeric origin IDs.
fn expand_origin_filter(value: Option<&str>) -> Vec<String> {
    expand_optional_filter(value)
        .into_iter()
        .map(|o| crate::taxonomy_legend::resolve_origin(&o).to_string())
        .collect()
}

fn exact_any_sql(where_parts: &mut Vec<String>, column: &str, values: &[String]) {
    if values.is_empty() {
        return;
    }
    let quoted = values
        .iter()
        .map(|v| crate::datafusion_store::sql_literal(v))
        .collect::<Vec<_>>()
        .join(", ");
    where_parts.push(format!("{column} IN ({quoted})"));
}

fn tradition_contains_sql(column: &str, value: &str) -> String {
    let token = serde_json::to_string(value).unwrap_or_else(|_| format!("\"{value}\""));
    crate::datafusion_store::string_contains_sql(column, &token)
}

fn brief_row(row: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "passage_id": row.get("passage_id").and_then(|v| v.as_str()).unwrap_or(""),
        "source_work_id": row.get("source_work_id").and_then(|v| v.as_str()).unwrap_or(""),
        "main_title": row.get("main_title").and_then(|v| v.as_str()).unwrap_or(""),
        "heading": row.get("heading").and_then(|v| v.as_str()).unwrap_or(""),
        "period": row.get("period").and_then(|v| v.as_str()).unwrap_or(""),
        "zh_quote": row.get("zh_text_raw").and_then(|v| v.as_str()).unwrap_or(""),
    })
}

fn opt_str(row: &serde_json::Value, key: &str) -> Option<String> {
    row.get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
}

fn looks_like_source_work_id(value: &str) -> bool {
    if value.contains('/') || value.contains('#') || value.chars().any(char::is_whitespace) {
        return false;
    }
    let Some((prefix, suffix)) = value.split_once('n') else {
        return false;
    };
    !prefix.is_empty()
        && prefix.chars().any(|ch| ch.is_ascii_digit())
        && prefix
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit())
        && suffix.chars().next().is_some_and(|ch| ch.is_ascii_digit())
}

fn validate_passage_input(tool: &str, value: &str) -> Result<()> {
    let detected_kind = if looks_like_source_work_id(value) {
        Some("source_work_id")
    } else if value.trim().is_empty() || value.chars().any(char::is_whitespace) {
        Some("search_text")
    } else {
        None
    };
    if let Some(detected_kind) = detected_kind {
        return Err(ToolError::InvalidPassageId {
            tool: tool.to_string(),
            provided: value.to_string(),
            detected_kind: detected_kind.to_string(),
        }
        .into_anyhow());
    }
    Ok(())
}

fn passage_not_found(passage_id: &str) -> anyhow::Error {
    ToolError::PassageNotFound {
        passage_id: passage_id.to_string(),
    }
    .into_anyhow()
}

fn resolve_source_read_direction(
    requested: &str,
    has_passage_id: bool,
    has_cursor: bool,
) -> (&str, Option<String>) {
    match requested {
        "auto" if has_passage_id => ("around", None),
        "auto" if has_cursor => ("next", None),
        "auto" => ("start", None),
        "start" | "next" | "prev" | "at" | "around" => (requested, None),
        other => (
            "start",
            Some(format!("unknown_direction_{other}_treated_as_start")),
        ),
    }
}

fn char_prefix(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn char_len(value: &str) -> usize {
    value.chars().count()
}

fn char_slice(value: &str, start: usize, end: usize) -> String {
    value
        .chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

fn find_char_offsets(haystack: &str, needle: &str) -> Vec<usize> {
    if needle.is_empty() {
        return Vec::new();
    }
    let mut offsets = Vec::new();
    let mut search_start = 0usize;
    while search_start <= haystack.len() {
        let Some(rel_pos) = haystack[search_start..].find(needle) else {
            break;
        };
        let byte_pos = search_start + rel_pos;
        offsets.push(haystack[..byte_pos].chars().count());
        search_start = byte_pos + needle.len();
    }
    offsets
}

fn find_offsets_for_terms(haystack: &str, terms: &[String]) -> Vec<usize> {
    // Terms come in both stored layouts (see `pair_term_variants`); use the
    // ones for this passage's layout. CJK-layout text never contains a space,
    // so there only the original `normalize_zh` terms are tried, as before.
    let word_layout = crate::normalize::has_word_layout(haystack);
    let mut offsets = terms
        .iter()
        .filter(|term| {
            let spaced = term.contains(' ');
            if word_layout {
                spaced || term.chars().all(crate::normalize::is_dense_char)
            } else {
                !spaced
            }
        })
        .flat_map(|term| find_char_offsets(haystack, term))
        .collect::<Vec<_>>();
    offsets.sort_unstable();
    offsets.dedup();
    offsets
}

fn min_pair_distance(a_offsets: &[usize], b_offsets: &[usize], ordered: bool) -> Option<usize> {
    let mut best: Option<usize> = None;
    for &a in a_offsets {
        for &b in b_offsets {
            if ordered && b < a {
                continue;
            }
            let distance = a.abs_diff(b);
            best = Some(best.map_or(distance, |current| current.min(distance)));
        }
    }
    best
}

fn has_sentence_pair(text: &str, a_offsets: &[usize], b_offsets: &[usize], ordered: bool) -> bool {
    let sentence_break =
        |ch: char| matches!(ch, '。' | '！' | '？' | '；' | '\n' | '.' | '!' | '?' | ';');
    let mut start = 0usize;
    let mut current = 0usize;
    for ch in text.chars() {
        current += 1;
        if sentence_break(ch) {
            if span_has_pair(start, current, a_offsets, b_offsets, ordered) {
                return true;
            }
            start = current;
        }
    }
    span_has_pair(start, current, a_offsets, b_offsets, ordered)
}

fn span_has_pair(
    start: usize,
    end: usize,
    a_offsets: &[usize],
    b_offsets: &[usize],
    ordered: bool,
) -> bool {
    a_offsets.iter().any(|&a| {
        a >= start
            && a < end
            && b_offsets
                .iter()
                .any(|&b| b >= start && b < end && (!ordered || b >= a))
    })
}

fn pair_snippet(
    text: &str,
    a_offsets: &[usize],
    b_offsets: &[usize],
    context_chars: usize,
) -> String {
    let Some(first) = a_offsets.iter().chain(b_offsets.iter()).min().copied() else {
        return char_prefix(text, context_chars.saturating_mul(2));
    };
    let last = a_offsets
        .iter()
        .chain(b_offsets.iter())
        .max()
        .copied()
        .unwrap_or(first);
    let start = first.saturating_sub(context_chars);
    let end = (last + context_chars).min(char_len(text));
    char_slice(text, start, end)
}

fn pair_term_variants(term: &str, allow_variants: bool) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::<String>::new();
    let mut terms = Vec::new();
    let normalized = crate::normalize::normalize_zh(term);
    if !normalized.is_empty() && seen.insert(normalized.clone()) {
        terms.push(normalized);
    }
    // Word-layout form, for alphabetic-script passages.
    let words = crate::normalize::normalize_words(term);
    if !words.trim().is_empty() && seen.insert(words.clone()) {
        terms.push(words);
    }
    if allow_variants {
        let tables = crate::templates::variants::VariantTables::load();
        for variant in tables.term_variants(term) {
            let normalized = crate::normalize::normalize_zh(&variant);
            if !normalized.is_empty() && seen.insert(normalized.clone()) {
                terms.push(normalized);
            }
        }
        for variant in tables.orthographic_flips(term, 20) {
            let normalized = crate::normalize::normalize_zh(&variant);
            if !normalized.is_empty() && seen.insert(normalized.clone()) {
                terms.push(normalized);
            }
        }
    }
    terms.truncate(20);
    terms
}

fn pair_profile_unit_key(
    row: &serde_json::Value,
    unit: &str,
    doc_table: &DocumentTable,
    catalog: Option<&CorpusCatalogIndex>,
) -> Option<String> {
    match unit {
        "work" => opt_str(row, "source_work_id"),
        "section" => {
            let catalog = catalog?;
            let pid = row.get("passage_id").and_then(|v| v.as_str())?;
            let doc_id = doc_table.doc_id(pid)?;
            let mut node_id = *catalog.doc_parent.get(&doc_id)?;
            let mut fallback = node_id;
            while let Some(node) = catalog.get_node(node_id) {
                fallback = node.node_id;
                if matches!(
                    node.node_kind,
                    crate::catalog_index::OutlineNodeKind::Section
                        | crate::catalog_index::OutlineNodeKind::Chapter
                        | crate::catalog_index::OutlineNodeKind::Division
                        | crate::catalog_index::OutlineNodeKind::Fascicle
                ) {
                    return Some(format!("section:{}", node.node_id));
                }
                let Some(parent) = node.parent_id else {
                    break;
                };
                node_id = parent;
            }
            Some(format!("section:{fallback}"))
        }
        _ => row
            .get("passage_id")
            .and_then(|v| v.as_str())
            .map(ToString::to_string),
    }
}

fn split_heading_path(value: Option<&str>) -> Vec<String> {
    value
        .unwrap_or("")
        .split('/')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn round_percent(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn refresh_hybrid_scores(hit: &mut crate::tools::responses::HybridDiscoverHit) {
    let (semantic_score, lexical_score, final_score, sources) =
        crate::retrieval::refresh_hybrid_scores(hit.vector_rank, hit.tfidf_rank);
    hit.semantic_score = semantic_score;
    hit.lexical_score = lexical_score;
    hit.final_score = final_score;
    hit.candidate_sources = sources
        .into_iter()
        .map(|source| source.as_str().to_string())
        .collect();
}

fn optional_one(value: &Option<String>) -> Option<Vec<String>> {
    value
        .as_ref()
        .filter(|v| !v.is_empty())
        .map(|v| vec![v.clone()])
}

fn evidence_scope_spec(
    req: &crate::tools::requests::EvidenceSearchRequest,
) -> crate::retrieval::ScopeSpec {
    crate::retrieval::ScopeSpec {
        canon: optional_one(&req.scope_canon),
        period: optional_one(&req.scope_period),
        source_work_id: optional_one(&req.scope_source_work_id),
        catalog_node_id: req.scope_node_id,
        author: optional_one(&req.author),
        source_corpus: None,
    }
}

fn pair_scope_spec(
    req: &crate::tools::requests::PairAppearanceRequest,
) -> crate::retrieval::ScopeSpec {
    crate::retrieval::ScopeSpec {
        canon: optional_one(&req.scope_canon),
        period: optional_one(&req.scope_period),
        source_work_id: optional_one(&req.scope_source_work_id),
        catalog_node_id: req.scope_node_id,
        author: None,
        source_corpus: None,
    }
}

fn adjust_read_end(text: &str, start: usize, target_end: usize, total: usize) -> (usize, String) {
    let end = target_end.min(total);
    if end >= total || end <= start {
        return (end, "edge".to_string());
    }

    let min_end = start + ((end - start) * 8 / 10).max(1);
    let punctuation = ['。', '！', '？', '；', '\n'];
    let mut clean_end = None;
    for (idx, ch) in text
        .chars()
        .enumerate()
        .skip(start)
        .take(end.saturating_sub(start))
    {
        if idx >= min_end && punctuation.contains(&ch) {
            clean_end = Some((idx + 1).min(total));
        }
    }
    match clean_end {
        Some(end) => (end, "clean".to_string()),
        None => (end, "rough".to_string()),
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SourceReadCursor {
    version: u8,
    source_work_id: String,
    char_start: usize,
    unit: String,
    max_chars: usize,
    overlap_chars: usize,
    heading_path_prefix: Option<String>,
}

fn encode_source_read_cursor(cursor: &SourceReadCursor) -> Result<String> {
    use base64::Engine;
    let bytes = serde_json::to_vec(cursor)?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

fn decode_source_read_cursor(cursor: &str) -> Result<SourceReadCursor> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(cursor)
        .map_err(|err| anyhow::anyhow!("invalid source-read cursor: {err}"))?;
    let decoded: SourceReadCursor = serde_json::from_slice(&bytes)
        .map_err(|err| anyhow::anyhow!("invalid source-read cursor payload: {err}"))?;
    if decoded.version != 1 {
        return Err(anyhow::anyhow!(
            "unsupported source-read cursor version: {}",
            decoded.version
        ));
    }
    Ok(decoded)
}

fn component_ok(
    name: &str,
    tool: &str,
    elapsed_ms: u128,
    summary: impl Into<String>,
) -> crate::tools::responses::WorkflowComponent {
    crate::tools::responses::WorkflowComponent {
        name: name.to_string(),
        tool: tool.to_string(),
        status: crate::tools::responses::ComponentStatus::Ok,
        used: true,
        elapsed_ms: Some(elapsed_ms),
        summary: Some(summary.into()),
        error: None,
    }
}

fn component_skipped(
    name: &str,
    tool: &str,
    status: crate::tools::responses::ComponentStatus,
    summary: impl Into<String>,
) -> crate::tools::responses::WorkflowComponent {
    crate::tools::responses::WorkflowComponent {
        name: name.to_string(),
        tool: tool.to_string(),
        status,
        used: false,
        elapsed_ms: None,
        summary: Some(summary.into()),
        error: None,
    }
}

#[derive(Debug, Clone)]
struct WorkflowBudget {
    started: std::time::Instant,
    max_elapsed_ms: Option<u64>,
    max_component_ms: Option<u64>,
}

impl WorkflowBudget {
    fn new(max_elapsed_ms: Option<u64>, max_component_ms: Option<u64>) -> Self {
        Self {
            started: std::time::Instant::now(),
            max_elapsed_ms,
            max_component_ms,
        }
    }

    fn elapsed_ms(&self) -> u128 {
        self.started.elapsed().as_millis()
    }

    fn remaining_ms(&self) -> Option<u64> {
        let max = self.max_elapsed_ms?;
        let elapsed = self.elapsed_ms() as u64;
        Some(max.saturating_sub(elapsed))
    }

    fn exhausted(&self) -> bool {
        self.remaining_ms().is_some_and(|remaining| remaining == 0)
    }

    fn component_timeout(&self) -> Option<std::time::Duration> {
        let mut limit = self.max_component_ms;
        if let Some(remaining) = self.remaining_ms() {
            limit = Some(limit.map_or(remaining, |component| component.min(remaining)));
        }
        limit.map(std::time::Duration::from_millis)
    }
}

fn component_timed_out(
    name: &str,
    tool: &str,
    elapsed_ms: u128,
    timeout_ms: u64,
) -> crate::tools::responses::WorkflowComponent {
    crate::tools::responses::WorkflowComponent {
        name: name.to_string(),
        tool: tool.to_string(),
        status: crate::tools::responses::ComponentStatus::TimedOut,
        used: false,
        elapsed_ms: Some(elapsed_ms),
        summary: Some(format!("timed out after {timeout_ms} ms")),
        error: Some(crate::tools::errors::ToolErrorBody {
            code: "timeout".to_string(),
            message: format!("{tool} timed out after {timeout_ms} ms"),
            suggested_command: None,
            details: Some(serde_json::json!({ "timeout_ms": timeout_ms })),
        }),
    }
}

enum ComponentOutcome<T> {
    Ok(T),
    Failed(anyhow::Error),
    TimedOut { elapsed_ms: u128, timeout_ms: u64 },
    BudgetExhausted,
}

const MIN_ENFORCEABLE_COMPONENT_TIMEOUT_MS: u64 = 10;

async fn run_budgeted_component<T, F>(budget: &WorkflowBudget, fut: F) -> ComponentOutcome<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    if budget.exhausted() {
        return ComponentOutcome::BudgetExhausted;
    }

    let started = std::time::Instant::now();
    if let Some(timeout) = budget.component_timeout() {
        let timeout_ms = timeout.as_millis() as u64;
        if timeout_ms < MIN_ENFORCEABLE_COMPONENT_TIMEOUT_MS {
            return ComponentOutcome::TimedOut {
                elapsed_ms: started.elapsed().as_millis(),
                timeout_ms,
            };
        }
        match tokio::time::timeout(timeout, fut).await {
            Ok(Ok(value)) => ComponentOutcome::Ok(value),
            Ok(Err(err)) => ComponentOutcome::Failed(err),
            Err(_) => ComponentOutcome::TimedOut {
                elapsed_ms: started.elapsed().as_millis(),
                timeout_ms,
            },
        }
    } else {
        match fut.await {
            Ok(value) => ComponentOutcome::Ok(value),
            Err(err) => ComponentOutcome::Failed(err),
        }
    }
}

fn record_component_outcome<T>(
    outcome: ComponentOutcome<T>,
    components: &mut Vec<crate::tools::responses::WorkflowComponent>,
    warnings: &mut Vec<String>,
    name: &str,
    tool: &str,
    started: std::time::Instant,
    ok_summary: impl FnOnce(&T) -> String,
    failed_warning: &str,
    timeout_warning: &str,
) -> Option<T> {
    match outcome {
        ComponentOutcome::Ok(value) => {
            components.push(component_ok(
                name,
                tool,
                started.elapsed().as_millis(),
                ok_summary(&value),
            ));
            Some(value)
        }
        ComponentOutcome::Failed(err) => {
            components.push(component_failed(
                name,
                tool,
                started.elapsed().as_millis(),
                &err,
            ));
            warnings.push(failed_warning.to_string());
            None
        }
        ComponentOutcome::TimedOut {
            elapsed_ms,
            timeout_ms,
        } => {
            components.push(component_timed_out(name, tool, elapsed_ms, timeout_ms));
            warnings.push(timeout_warning.to_string());
            None
        }
        ComponentOutcome::BudgetExhausted => {
            components.push(component_skipped(
                name,
                tool,
                crate::tools::responses::ComponentStatus::SkippedBudgetExhausted,
                "budget exhausted",
            ));
            None
        }
    }
}

fn component_failed(
    name: &str,
    tool: &str,
    elapsed_ms: u128,
    err: &anyhow::Error,
) -> crate::tools::responses::WorkflowComponent {
    crate::tools::responses::WorkflowComponent {
        name: name.to_string(),
        tool: tool.to_string(),
        status: crate::tools::responses::ComponentStatus::Failed,
        used: false,
        elapsed_ms: Some(elapsed_ms),
        summary: None,
        error: Some(crate::tools::errors::classify_tool_error(err)),
    }
}

fn suggested_tool(
    tool: &str,
    args: serde_json::Value,
    reason: &str,
) -> crate::tools::responses::SuggestedToolCall {
    crate::tools::responses::SuggestedToolCall {
        tool: tool.to_string(),
        args,
        reason: reason.to_string(),
    }
}

/// Build ranked next-step candidates from the shape of *this* search result
/// — zero hits, a single hit, a tight cluster, or a corpus-wide spread each
/// point somewhere different. Reasons cite the concrete numbers/passage IDs
/// observed in this call so they read as an observation about this result
/// rather than a fixed tip repeated on every call. Callers should drop
/// candidates whose tool the agent has already pivoted to recently (see
/// `tools::log::recent_tool_names`) so the same nudge doesn't linger once
/// it's been acted on.
fn search_suggestion_candidates(
    phrase: &str,
    hits: &[crate::tools::responses::SearchHit],
    verified_count: usize,
) -> Vec<(&'static str, serde_json::Value, String)> {
    if hits.is_empty() {
        return vec![
            (
                "query-expand-terms",
                serde_json::json!({"phrase": phrase}),
                format!(
                    "no exact hits for \"{phrase}\" — query-expand-terms can surface variant forms and orthographic flips to retry with"
                ),
            ),
            (
                "heading-search",
                serde_json::json!({"query": phrase, "limit": 10}),
                format!(
                    "\"{phrase}\" may be a title or section heading rather than body text — heading-search checks heading/section metadata directly"
                ),
            ),
            (
                "canonical-source",
                serde_json::json!({"phrase": phrase}),
                format!(
                    "canonical-source looks for canon-side passages that may phrase \"{phrase}\" differently"
                ),
            ),
        ];
    }

    let top = &hits[0];
    let mut work_ids = std::collections::BTreeSet::new();
    for hit in hits {
        if let Some(ref wid) = hit.source_work_id {
            work_ids.insert(wid.as_str());
        }
    }
    let work_count = work_ids.len();

    if verified_count == 1 {
        return vec![
            (
                "source-investigate",
                serde_json::json!({"seed_passage_id": top.passage_id}),
                format!(
                    "exactly one exact hit, at {} — source-investigate gathers context, frontier, similarity, and phrase history from this single anchor in one call",
                    top.passage_id
                ),
            ),
            (
                "source-read",
                serde_json::json!({"passage_id": top.passage_id, "direction": "around", "max_chars": 4000}),
                format!("read {} in its surrounding source context before pivoting outward", top.passage_id),
            ),
            (
                "frontier",
                serde_json::json!({"seed": top.passage_id}),
                format!("expand the single hit at {} into a discovery frontier of lexically/phrasally similar passages corpus-wide", top.passage_id),
            ),
        ];
    }

    if work_count > 0 && work_count <= 2 {
        let works: Vec<&str> = work_ids.iter().copied().collect();
        let works_label = works.join(", ");
        return vec![
            (
                "frontier",
                serde_json::json!({"seed": top.passage_id}),
                format!(
                    "{verified_count} hits concentrate in {work_count} work(s) ({works_label}) — frontier from {} would expand this cluster into a ranked discovery packet across the rest of the corpus",
                    top.passage_id
                ),
            ),
            (
                "search",
                serde_json::json!({"phrase": phrase, "mode": "clusters", "group_by": "division"}),
                format!("re-run with mode=\"clusters\", group_by=\"division\" to see where within {works_label} these hits concentrate"),
            ),
            (
                "source-read",
                serde_json::json!({"passage_id": top.passage_id, "direction": "around", "max_chars": 4000}),
                format!("read around the top hit {} in its citation-aware source stream", top.passage_id),
            ),
        ];
    }

    if work_count >= 5 {
        return vec![
            (
                "phrase-history",
                serde_json::json!({"phrase": phrase}),
                format!(
                    "{verified_count} hits span {work_count} different works — phrase-history maps this phrase's distribution across periods, canons, and traditions"
                ),
            ),
            (
                "search",
                serde_json::json!({"phrase": phrase, "mode": "trace", "group_by": "period"}),
                format!("re-run with mode=\"trace\", group_by=\"period\" to see how these {verified_count} hits concentrate by historical period directly"),
            ),
            (
                "evidence-search",
                serde_json::json!({"phrase": phrase, "include_attestation": true, "include_history": true}),
                format!("evidence-search would wrap \"{phrase}\" with attestation and history summaries in one call"),
            ),
        ];
    }

    // Moderate spread: neither tightly concentrated nor corpus-wide — offer
    // both a discovery pivot and a navigational view of the concentration.
    vec![
        (
            "frontier",
            serde_json::json!({"seed": top.passage_id}),
            format!(
                "{verified_count} hits across {work_count} works — frontier from the top hit {} would expand into discovery candidates beyond exact matches",
                top.passage_id
            ),
        ),
        (
            "cluster-hits",
            serde_json::json!({"phrase": phrase, "cluster_by": "work"}),
            format!("cluster-hits would group these {verified_count} hits by work/division to show where they concentrate"),
        ),
    ]
}

fn count_unique_ngrams_with_terms(
    text: &str,
    gram_len: usize,
    counts: &mut FxHashMap<u64, u32>,
    terms: &mut FxHashMap<u64, String>,
) -> u32 {
    if gram_len == 0 {
        return 0;
    }
    let mut seen = rustc_hash::FxHashSet::default();
    let mut add = |term: String| {
        let hash = xxhash_rust::xxh3::xxh3_64(term.as_bytes());
        if seen.insert(hash) {
            *counts.entry(hash).or_insert(0) += 1;
            terms.entry(hash).or_insert(term);
        }
    };

    if !crate::normalize::has_word_layout(text) {
        let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
        for window in chars.windows(gram_len) {
            add(window.iter().collect());
        }
        return seen.len() as u32;
    }

    // Word-layout text (English and other alphabetic scripts): a term is
    // `gram_len` consecutive content words. Counting characters here would
    // report single letters as the vocabulary. CJK runs inside such text are
    // still counted by character.
    let mut run: Vec<&str> = Vec::new();
    for token in text.split(' ').filter(|t| !t.is_empty()) {
        if token.chars().any(crate::normalize::is_dense_char) {
            run.clear();
            let chars: Vec<char> = token.chars().collect();
            for window in chars.windows(gram_len) {
                add(window.iter().collect());
            }
        } else if crate::tfidf::ngram::keep_word(token) {
            run.push(token);
            if run.len() >= gram_len {
                add(run[run.len() - gram_len..].join(" "));
            }
        } else {
            // A stopword or a stray letter ends the run, as in the TF-IDF
            // word features: "pump of coolant" is not the term "pump coolant".
            run.clear();
        }
    }
    seen.len() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(passage_id: &str, work_id: &str) -> crate::tools::responses::SearchHit {
        crate::tools::responses::SearchHit {
            passage_id: passage_id.to_string(),
            source_work_id: Some(work_id.to_string()),
            main_title: None,
            heading: None,
            zh_quote: String::new(),
            score: None,
        }
    }

    fn terms(text: &str, gram_len: usize) -> Vec<String> {
        let mut counts = FxHashMap::default();
        let mut display = FxHashMap::default();
        count_unique_ngrams_with_terms(text, gram_len, &mut counts, &mut display);
        let mut terms: Vec<String> = display.into_values().collect();
        terms.sort();
        terms
    }

    #[test]
    fn terms_of_english_text_are_words_not_letters() {
        let text = crate::normalize::normalize_words("The coolant pump of the Coolant Pump leaks.");
        assert_eq!(terms(&text, 1), vec!["coolant", "leaks", "pump"]);
        // A stopword ends a run: "pump ... coolant" across "of the" is no term.
        assert_eq!(terms(&text, 2), vec!["coolant pump", "pump leaks"]);
    }

    #[test]
    fn terms_of_cjk_text_are_still_characters() {
        assert_eq!(terms("如是我聞", 2), vec!["如是", "我聞", "是我"]);
        // CJK inside word-layout text keeps the character treatment.
        let mixed = crate::normalize::normalize_words("pump 如是我 valve");
        assert_eq!(terms(&mixed, 2), vec!["如是", "是我"]);
    }

    #[test]
    fn overlap_of_english_text_is_counted_in_words() {
        let quote = crate::normalize::normalize_words("Use only approved words");
        let related = crate::normalize::normalize_words("Rule 1.2: use approved words from the dictionary only.");
        let unrelated = crate::normalize::normalize_words("The quick brown fox jumps over a lazy dog, why not?");
        assert_eq!(character_overlap_score(&quote, &related), 1.0);
        // Every letter of the quote is in the pangram; none of its words are.
        assert_eq!(character_overlap_score(&quote, &unrelated), 0.0);
        // CJK is compared by character, as before.
        assert_eq!(character_overlap_score("如是我聞", "我聞如是一時"), 1.0);
    }

    #[test]
    fn search_suggestions_steer_toward_expansion_on_zero_hits() {
        let candidates = search_suggestion_candidates("無一物", &[], 0);
        assert!(candidates
            .iter()
            .any(|(tool, _, _)| *tool == "query-expand-terms"));
    }

    #[test]
    fn search_suggestions_seed_investigation_from_lone_hit() {
        let hits = vec![hit("T/T08/T08n0235.xml#p1", "T08n0235")];
        let candidates = search_suggestion_candidates("無一物", &hits, 1);
        let (tool, args, _) = &candidates[0];
        assert_eq!(*tool, "source-investigate");
        assert_eq!(args["seed_passage_id"], "T/T08/T08n0235.xml#p1");
    }

    #[test]
    fn search_suggestions_point_at_frontier_when_concentrated() {
        let hits = vec![
            hit("T/T08/T08n0235.xml#p1", "T08n0235"),
            hit("T/T08/T08n0235.xml#p2", "T08n0235"),
            hit("T/T08/T08n0235.xml#p3", "T08n0235"),
        ];
        let candidates = search_suggestion_candidates("無一物", &hits, 3);
        let (tool, args, reason) = &candidates[0];
        assert_eq!(*tool, "frontier");
        assert_eq!(args["seed"], "T/T08/T08n0235.xml#p1");
        assert!(reason.contains("T08n0235"));
    }

    #[test]
    fn search_suggestions_point_at_phrase_history_when_spread_wide() {
        let hits = (0..6)
            .map(|i| hit(&format!("W{i}#p1"), &format!("W{i}")))
            .collect::<Vec<_>>();
        let candidates = search_suggestion_candidates("無一物", &hits, 6);
        assert!(candidates
            .iter()
            .any(|(tool, _, _)| *tool == "phrase-history"));
    }

    #[test]
    fn optional_filter_expands_csv_values() {
        assert_eq!(expand_optional_filter(None), Vec::<String>::new());
        assert_eq!(
            expand_optional_filter(Some("T, X, T")),
            vec!["T".to_string(), "X".to_string()]
        );
    }

    #[test]
    fn exact_any_uses_in_clause_for_multi_value_filters() {
        let mut where_parts = Vec::new();
        exact_any_sql(
            &mut where_parts,
            "period",
            &["Tang".to_string(), "Song".to_string()],
        );
        assert_eq!(where_parts, vec!["period IN ('Tang', 'Song')".to_string()]);
    }

    #[test]
    fn tradition_match_targets_json_array_token() {
        assert_eq!(
            tradition_contains_sql("traditions", "canon"),
            "strpos(traditions, '\"canon\"') > 0"
        );
    }

    #[test]
    fn passage_input_distinguishes_work_ids_and_search_text() {
        assert!(looks_like_source_work_id("T48n2005_001"));
        assert!(!looks_like_source_work_id(
            "T/T48/T48n2005.xml#pT48p0292a0101"
        ));

        let work_err = validate_passage_input("frontier", "T48n2005_001").unwrap_err();
        let work_body = crate::tools::errors::classify_tool_error(&work_err);
        assert_eq!(work_body.code, "invalid_passage_id");
        assert_eq!(
            work_body.details.unwrap()["detected_kind"],
            "source_work_id"
        );

        let text_err = validate_passage_input("frontier", "摩訶迦葉 拈花").unwrap_err();
        let text_body = crate::tools::errors::classify_tool_error(&text_err);
        assert_eq!(text_body.code, "invalid_passage_id");
        assert_eq!(text_body.details.unwrap()["detected_kind"], "search_text");
    }

    #[test]
    fn source_read_accepts_hash_anchor_and_chooses_contextual_defaults() {
        let request: crate::tools::requests::SourceReadRequest =
            serde_json::from_value(serde_json::json!({
                "passage_id": "B/B24/B24n0137_005.xml#pB24p0459b3001",
                "max_chars": 8000,
                "include_metadata": true
            }))
            .expect("hash-bearing passage ID is valid JSON input");
        assert_eq!(request.direction, "auto");
        assert_eq!(
            resolve_source_read_direction(
                &request.direction,
                request.passage_id.is_some(),
                request.cursor.is_some()
            ),
            ("around", None)
        );
        assert_eq!(
            resolve_source_read_direction("auto", false, true),
            ("next", None)
        );
        assert_eq!(
            resolve_source_read_direction("auto", false, false),
            ("start", None)
        );
    }

    #[test]
    fn pair_profile_work_unit_keys_by_source_work_id() {
        let doc_table = DocumentTable::new();
        let row = serde_json::json!({
            "passage_id": "p1",
            "source_work_id": "T01n0001"
        });
        assert_eq!(
            pair_profile_unit_key(&row, "work", &doc_table, None),
            Some("T01n0001".to_string())
        );
    }

    #[test]
    fn pair_profile_section_unit_climbs_catalog_parent() {
        let mut doc_table = DocumentTable::new();
        doc_table.passage_ids = vec!["p1".to_string()];
        doc_table.passage_lookup_order = vec![0];

        let mut catalog = CorpusCatalogIndex::new();
        let section = catalog.push_node(crate::catalog_index::OutlineNode::leaf(
            crate::catalog_index::OutlineNodeKind::Section,
            None,
            "test".to_string(),
            "W1".to_string(),
            "w1.xml".to_string(),
            "Section".to_string(),
        ));
        let passage = catalog.push_node(crate::catalog_index::OutlineNode::leaf(
            crate::catalog_index::OutlineNodeKind::PassageRange,
            Some(section),
            "test".to_string(),
            "W1".to_string(),
            "w1.xml".to_string(),
            "Passage".to_string(),
        ));
        catalog.add_child(section, passage);
        catalog.doc_parent.insert(0, passage);

        let row = serde_json::json!({"passage_id": "p1"});
        assert_eq!(
            pair_profile_unit_key(&row, "section", &doc_table, Some(&catalog)),
            Some(format!("section:{section}"))
        );
    }
}

impl ToolEngine {
    pub async fn open(config: EngineConfig) -> Result<Self> {
        let max_heavy = config.max_heavy_concurrency.max(1);

        // Tell the dict/entity annotation layer where to find parquet stores.
        let dict_path = Self::resolve_dict_path_static(&config);
        crate::dict::set_dict_path(dict_path);
        let person_path = Self::resolve_authority_path_static(
            &config,
            crate::pack::DEFAULT_PERSONS,
            "data/persons.parquet",
        );
        crate::dict::set_person_path(person_path);
        let place_path = Self::resolve_authority_path_static(
            &config,
            crate::pack::DEFAULT_PLACES,
            "data/places.parquet",
        );
        crate::dict::set_place_path(place_path);

        Ok(Self {
            config,
            passages: OnceCell::new(),
            phrase: OnceCell::new(),
            tfidf: OnceCell::new(),
            vector: OnceCell::new(),
            catalog: OnceCell::new(),
            doc_table: OnceCell::new(),
            registry: OnceCell::new(),
            heavy_slots: Semaphore::new(max_heavy),
        })
    }

    fn resolve_dict_path_static(config: &EngineConfig) -> Option<PathBuf> {
        if let Some(ref pack) = config.pack {
            let p = pack.join(crate::pack::DEFAULT_DICT);
            if p.is_dir() {
                return Some(p);
            }
        }
        let default = PathBuf::from("data/dict.parquet");
        if default.is_dir() {
            return Some(default);
        }
        None
    }

    fn resolve_authority_path_static(
        config: &EngineConfig,
        pack_rel: &str,
        default_rel: &str,
    ) -> Option<PathBuf> {
        if let Some(ref pack) = config.pack {
            let p = pack.join(pack_rel);
            if p.is_dir() {
                return Some(p);
            }
        }
        let default = PathBuf::from(default_rel);
        if default.is_dir() {
            return Some(default);
        }
        None
    }

    /// Resolve the passages parquet path
    pub fn resolve_passages_path(&self) -> Result<PathBuf> {
        if let Some(ref path) = self.config.passages_parquet {
            return Ok(path.clone());
        }

        if let Some(ref pack) = self.config.pack.as_ref() {
            let pack_path = pack.join(crate::pack::DEFAULT_PASSAGES);
            if pack_path.exists() {
                return Ok(pack_path);
            }
        }

        // Default path
        let default = PathBuf::from("data/passages.parquet");
        if default.exists() {
            return Ok(default);
        }

        // Auto-heal: pack-prep workflow renames passages.parquet → passages-raw.parquet;
        // rename it back so all tools work without manual intervention.
        match crate::storage::heal_raw_parquet(&default) {
            Ok(true) => {
                eprintln!("[sinorag] Auto-renamed passages-raw.parquet → passages.parquet");
                return Ok(default);
            }
            Ok(false) => {}
            Err(e) => {
                eprintln!("[sinorag] WARNING: passages-raw.parquet found but rename failed: {e}")
            }
        }

        Err(ToolError::MissingPassages { path: default }.into_anyhow())
    }

    /// Resolve the phrase index path
    pub fn resolve_phrase_path(&self) -> Result<PathBuf> {
        if let Some(ref path) = self.config.phrase_index {
            return Ok(path.clone());
        }

        if let Some(pack) = self.config.pack.as_ref() {
            let pack_path = pack.join(crate::pack::DEFAULT_PHRASE);
            if pack_path.exists() {
                return Ok(pack_path);
            }
        }

        let default = PathBuf::from("data/derived/phrase.index");
        if default.exists() {
            return Ok(default);
        }

        Err(ToolError::MissingPhraseIndex { path: default }.into_anyhow())
    }

    /// Resolve the tfidf index path
    pub fn resolve_tfidf_path(&self) -> Result<PathBuf> {
        if let Some(ref path) = self.config.tfidf_index {
            return Ok(path.clone());
        }

        if let Some(pack) = self.config.pack.as_ref() {
            let pack_path = pack.join(crate::pack::DEFAULT_TFIDF);
            if pack_path.exists() {
                return Ok(pack_path);
            }
        }

        let default = PathBuf::from("data/derived/tfidf.index");
        if default.exists() {
            return Ok(default);
        }

        Err(ToolError::MissingTfidfIndex { path: default }.into_anyhow())
    }

    /// Resolve the vector index path.
    pub fn resolve_vector_path(&self) -> Result<PathBuf> {
        if let Some(ref path) = self.config.vector_index {
            return Ok(path.clone());
        }

        if let Some(pack) = self.config.pack.as_ref() {
            let pack_path = pack.join(crate::pack::DEFAULT_VECTOR);
            if pack_path.exists() {
                return Ok(pack_path);
            }
        }

        let default = PathBuf::from("data/derived/vector.index");
        if default.exists() {
            return Ok(default);
        }

        Err(ToolError::MissingVectorIndex { path: default }.into_anyhow())
    }

    /// Resolve a phrase index path if present, and validate it against the
    /// active doc table before any doc_id-bearing lookup uses it.
    pub async fn optional_phrase_path(&self) -> Result<Option<PathBuf>> {
        let Ok(path) = self.resolve_phrase_path() else {
            return Ok(None);
        };
        self.ensure_phrase_index_matches_doc_table(&path).await?;
        Ok(Some(path))
    }

    /// Resolve a TF-IDF path if present, and validate it against the active
    /// doc table before any doc_id-bearing lookup uses it.
    pub async fn optional_tfidf_path(&self) -> Result<Option<PathBuf>> {
        let Ok(path) = self.resolve_tfidf_path() else {
            return Ok(None);
        };
        self.ensure_tfidf_index_matches_doc_table(&path).await?;
        Ok(Some(path))
    }

    async fn ensure_phrase_index_matches_doc_table(&self, path: &Path) -> Result<()> {
        let info = PhraseIndex::header_info(path)?;
        let fingerprint = info
            .get("doc_table_fingerprint")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        self.ensure_index_matches_doc_table("phrase index", path, fingerprint)
            .await
    }

    async fn ensure_tfidf_index_matches_doc_table(&self, path: &Path) -> Result<()> {
        let info = TfidfIndex::header_info(path)?;
        let fingerprint = info
            .get("doc_table_fingerprint")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        self.ensure_index_matches_doc_table("TF-IDF index", path, fingerprint)
            .await
    }

    async fn ensure_vector_index_matches_doc_table(&self, index: &VectorIndex) -> Result<()> {
        let doc_table_path = self.resolve_doc_table_path()?;
        let doc_table = self.doc_table().await?;
        crate::vector_index::ensure_matches_doc_table(index, &doc_table, &doc_table_path)
    }

    async fn ensure_index_matches_doc_table(
        &self,
        index_name: &str,
        index_path: &Path,
        fingerprint: &str,
    ) -> Result<()> {
        let doc_table_path = self.resolve_doc_table_path()?;
        let doc_table = self.doc_table().await?;
        let coverage = crate::document_table::match_index_fingerprint(
            &doc_table,
            &doc_table_path,
            fingerprint,
        )?;
        if coverage.is_none() {
            anyhow::bail!(
                "{} fingerprint does not match active doc_table; rebuild {} for {}",
                index_name,
                index_name,
                doc_table_path.display()
            );
        }
        if !index_path.exists() {
            anyhow::bail!("{} not found at {}", index_name, index_path.display());
        }
        Ok(())
    }

    /// Resolve the catalog index path
    pub fn resolve_catalog_path(&self) -> Result<PathBuf> {
        if let Some(ref path) = self.config.catalog_index {
            return Ok(path.clone());
        }

        if let Some(pack) = self.config.pack.as_ref() {
            let pack_path = pack.join(crate::pack::DEFAULT_CATALOG);
            if pack_path.exists() {
                return Ok(pack_path);
            }
        }

        let default = PathBuf::from("data/derived/catalog.index");
        if default.exists() {
            return Ok(default);
        }

        Err(ToolError::MissingCatalogIndex { path: default }.into_anyhow())
    }

    /// Resolve the doc table path
    pub fn resolve_doc_table_path(&self) -> Result<PathBuf> {
        if let Some(ref path) = self.config.doc_table {
            return Ok(path.clone());
        }

        if let Some(pack) = self.config.pack.as_ref() {
            let pack_path = pack.join(crate::pack::DEFAULT_DOC_TABLE);
            if pack_path.exists() {
                return Ok(pack_path);
            }
        }

        let default = PathBuf::from("data/derived/doc_table.bin");
        if default.exists() {
            return Ok(default);
        }

        Err(ToolError::MissingDocTable { path: default }.into_anyhow())
    }

    /// Resolve the registry path
    pub fn resolve_registry_path(&self) -> Result<PathBuf> {
        if let Some(ref path) = self.config.registry {
            return Ok(path.clone());
        }

        if let Some(pack) = self.config.pack.as_ref() {
            let pack_path = pack.join(crate::pack::DEFAULT_REGISTRY);
            if pack_path.exists() {
                return Ok(pack_path);
            }
        }

        let default = PathBuf::from("data/derived/registry.sqlite");
        if default.exists() {
            return Ok(default);
        }

        Err(anyhow::anyhow!("Cannot resolve registry.sqlite path"))
    }

    /// Resolve where registry.sqlite should live, even when it does not exist yet.
    pub fn registry_path_or_default(&self) -> PathBuf {
        if let Some(ref path) = self.config.registry {
            return path.clone();
        }

        if let Some(pack) = self.config.pack.as_ref() {
            return pack.join(crate::pack::DEFAULT_REGISTRY);
        }

        PathBuf::from("data/derived/registry.sqlite")
    }

    /// Get or load the passages store
    pub async fn passages(&self) -> Result<Arc<DataFusionStore>> {
        self.passages
            .get_or_try_init(|| async {
                let path = self.resolve_passages_path()?;
                Ok(Arc::new(DataFusionStore::open(&path).await?))
            })
            .await
            .map(Clone::clone)
    }

    /// Get or load the phrase index
    pub async fn phrase(&self) -> Result<Arc<PhraseIndex>> {
        self.phrase
            .get_or_try_init(|| async {
                let path = self.resolve_phrase_path()?;
                self.ensure_phrase_index_matches_doc_table(&path).await?;
                Ok(Arc::new(PhraseIndex::open(&path)?))
            })
            .await
            .map(Clone::clone)
    }

    /// Get or load the tfidf index
    pub async fn tfidf(&self) -> Result<Arc<TfidfIndex>> {
        self.tfidf
            .get_or_try_init(|| async {
                let path = self.resolve_tfidf_path()?;
                self.ensure_tfidf_index_matches_doc_table(&path).await?;
                Ok(Arc::new(TfidfIndex::open(&path)?))
            })
            .await
            .map(Clone::clone)
    }

    /// Get or load the vector index.
    pub async fn vector(&self) -> Result<Arc<VectorIndex>> {
        self.vector
            .get_or_try_init(|| async {
                let path = self.resolve_vector_path()?;
                let index = VectorIndex::open(&path)?;
                self.ensure_vector_index_matches_doc_table(&index).await?;
                Ok(Arc::new(index))
            })
            .await
            .map(Clone::clone)
    }

    /// Get or load the catalog index
    pub async fn catalog(&self) -> Result<Arc<CorpusCatalogIndex>> {
        self.catalog
            .get_or_try_init(|| async {
                let path = self.resolve_catalog_path()?;
                Ok(Arc::new(CorpusCatalogIndex::load(&path)?))
            })
            .await
            .map(Clone::clone)
    }

    /// Get or load the doc table
    pub async fn doc_table(&self) -> Result<Arc<DocumentTable>> {
        self.doc_table
            .get_or_try_init(|| async {
                let path = self.resolve_doc_table_path()?;
                Ok(Arc::new(DocumentTable::load(&path)?))
            })
            .await
            .map(Clone::clone)
    }

    /// Get or load the registry
    pub async fn registry(&self) -> Result<Arc<()>> {
        self.registry
            .get_or_try_init(|| async {
                // Placeholder - Registry may not exist yet
                Ok(Arc::new(()))
            })
            .await
            .map(Clone::clone)
    }

    /// Ensure write operations are allowed
    pub fn ensure_write_allowed(&self, tool: &str, output_path: &Path) -> Result<()> {
        if self.config.readonly {
            return Err(crate::tools::errors::ToolError::ReadonlyViolation {
                tool: tool.to_string(),
            }
            .into_anyhow());
        }

        // If output_root is set, ensure output path is under it
        if let Some(ref root) = self.config.output_root {
            Self::ensure_under_root(root, output_path)?;
        }

        Ok(())
    }

    /// Ensure a path is under a root directory
    fn ensure_under_root(root: &Path, path: &Path) -> Result<()> {
        let root = root
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("Cannot canonicalize root {}: {}", root.display(), e))?;

        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Output path has no parent: {}", path.display()))?;

        std::fs::create_dir_all(parent).map_err(|e| {
            anyhow::anyhow!("Cannot create parent directory {}: {}", parent.display(), e)
        })?;

        let parent = parent.canonicalize().map_err(|e| {
            anyhow::anyhow!("Cannot canonicalize parent {}: {}", parent.display(), e)
        })?;

        if !parent.starts_with(&root) {
            return Err(crate::tools::errors::ToolError::OutputPathViolation {
                path: path.to_path_buf(),
                root,
            }
            .into_anyhow());
        }

        Ok(())
    }

    /// Run a future with a heavy slot (for concurrency control)
    pub async fn with_heavy_slot<T, F>(&self, fut: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        let _permit = self.heavy_slots.acquire().await?;
        fut.await
    }

    /// Implement the status tool
    pub async fn status_impl(&self) -> Result<crate::tools::responses::StatusResponse> {
        use crate::tools::responses::StatusResponse;

        let data_root = self
            .config
            .pack
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "data".to_string());

        let passages_parquet_exists = self.resolve_passages_path().is_ok();
        let phrase_index_exists = self.resolve_phrase_path().is_ok();
        let tfidf_index_exists = self.resolve_tfidf_path().is_ok();
        let vector_index_exists = self.resolve_vector_path().is_ok();
        let catalog_index_exists = self.resolve_catalog_path().is_ok();
        let doc_table_exists = self.resolve_doc_table_path().is_ok();
        let registry_exists = self.resolve_registry_path().is_ok();

        Ok(StatusResponse {
            schema: "sinorag-status-v1",
            data_root,
            passages_parquet_exists,
            phrase_index_exists,
            tfidf_index_exists,
            vector_index_exists,
            catalog_index_exists,
            doc_table_exists,
            registry_exists,
        })
    }

    /// Implement the passage tool
    pub async fn passage_impl(
        &self,
        req: crate::tools::requests::PassageRequest,
    ) -> Result<crate::tools::responses::PassageResponse> {
        use crate::datafusion_store::sql_literal;
        use crate::tools::responses::PassageResponse;

        validate_passage_input("passage", &req.id)?;
        let passages = self.passages().await?;
        let sql = format!(
            "SELECT passage_id, zh_text_raw, source_work_id, main_title, heading, \
                    canon, period, traditions, origin, author, source_rel_path, xml_id \
             FROM passages WHERE passage_id = {} LIMIT 1",
            sql_literal(&req.id)
        );
        let mut rows = passages.query_json(&sql).await?;
        let row = rows
            .drain(..)
            .next()
            .ok_or_else(|| passage_not_found(&req.id))?;

        Ok(PassageResponse {
            schema: "sinorag-passage-v1",
            passage_id: req.id.clone(),
            zh_quote: row
                .get("zh_text_raw")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            source_work_id: row
                .get("source_work_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            main_title: row
                .get("main_title")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            heading: row
                .get("heading")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        })
    }

    /// Implement the source-read tool
    pub async fn source_read_impl(
        &self,
        req: crate::tools::requests::SourceReadRequest,
    ) -> Result<crate::tools::responses::SourceReadResponse> {
        use crate::datafusion_store::sql_literal;
        use crate::tools::responses::{
            SourceReadCursorInfo, SourceReadPosition, SourceReadResponse, SourceReadSegment,
            SourceReadingState,
        };

        if let Some(passage_id) = &req.passage_id {
            validate_passage_input("source-read", passage_id)?;
        }

        let passages = self.passages().await?;
        let mut warnings = Vec::new();
        let (direction, direction_warning) = resolve_source_read_direction(
            &req.direction,
            req.passage_id.is_some(),
            req.cursor.is_some(),
        );
        warnings.extend(direction_warning);
        let unit = match req.unit.as_str() {
            "chunk" => "chunk",
            other => {
                warnings.push(format!("unsupported_unit_{other}_treated_as_chunk"));
                "chunk"
            }
        };

        let decoded_cursor = req
            .cursor
            .as_deref()
            .map(decode_source_read_cursor)
            .transpose()?;
        let cursor_max_chars = decoded_cursor.as_ref().map(|cursor| cursor.max_chars);
        let cursor_overlap_chars = decoded_cursor.as_ref().map(|cursor| cursor.overlap_chars);
        let max_chars = if req.cursor.is_some() && req.max_chars == 4000 {
            cursor_max_chars.unwrap_or(req.max_chars)
        } else {
            req.max_chars
        }
        .clamp(500, 12000);
        let overlap_chars = if req.cursor.is_some() && req.overlap_chars == 400 {
            cursor_overlap_chars.unwrap_or(req.overlap_chars)
        } else {
            req.overlap_chars
        }
        .min(max_chars / 2);

        let mut source_work_id = decoded_cursor
            .as_ref()
            .map(|c| c.source_work_id.clone())
            .or(req.source_work_id.clone());
        let mut heading_path_prefix = decoded_cursor
            .as_ref()
            .and_then(|c| c.heading_path_prefix.clone());

        if let Some(node_id) = req.node_id {
            let catalog = self.catalog().await?;
            let node = catalog
                .get_node(node_id)
                .ok_or_else(|| anyhow::anyhow!("Catalog node not found: {node_id}"))?;
            source_work_id.get_or_insert_with(|| node.work_id.clone());
            if !node.heading_path.is_empty() {
                heading_path_prefix.get_or_insert_with(|| node.heading_path.clone());
            }
        }

        let anchor_passage = if let Some(passage_id) = &req.passage_id {
            let row = passages.get_passage(passage_id).await.map_err(|err| {
                if err.to_string().contains("Passage not found") {
                    passage_not_found(passage_id)
                } else {
                    err
                }
            })?;
            if let Some(anchor_work_id) = opt_str(&row, "source_work_id") {
                if let Some(requested_work_id) = source_work_id.as_deref() {
                    if requested_work_id != anchor_work_id {
                        return Err(ToolError::InvalidArgs(format!(
                            "passage_id belongs to source_work_id={anchor_work_id}, not {requested_work_id}"
                        ))
                        .into_anyhow());
                    }
                    if req.source_work_id.is_some() {
                        warnings.push(
                            "source_work_id_redundant_with_passage_id; omit it on future calls"
                                .to_string(),
                        );
                    }
                }
                source_work_id = Some(anchor_work_id);
            }
            Some(row)
        } else {
            None
        };

        let source_work_id = source_work_id.ok_or_else(|| {
            ToolError::InvalidArgs(
                "source-read needs one anchor: source_work_id (start of a work), passage_id (read \
                 around a hit), node_id, or cursor (continue a previous read)"
                    .to_string(),
            )
            .into_anyhow()
        })?;

        let mut where_parts = vec![format!("source_work_id = {}", sql_literal(&source_work_id))];
        if let Some(prefix) = &heading_path_prefix {
            where_parts.push(format!(
                "heading_path LIKE {}",
                sql_literal(&format!("{prefix}%"))
            ));
        }

        let sql = format!(
            "SELECT passage_id, source_work_id, main_title, heading, heading_path, \
                    from_lb, to_lb, zh_text_raw, canon, period, author, source_rel_path, xml_id, period_rank \
             FROM passages \
             WHERE {} \
             ORDER BY period_rank ASC, source_rel_path ASC, from_lb ASC, xml_id ASC",
            where_parts.join(" AND ")
        );
        let rows = passages.query_json(&sql).await?;
        if rows.is_empty() {
            return Err(anyhow::anyhow!(
                "No passages found for source_work_id={source_work_id}"
            ));
        }

        let mut combined = String::new();
        let mut spans = Vec::with_capacity(rows.len());
        let mut combined_chars = 0usize;
        for row in &rows {
            if !combined.is_empty() {
                combined.push('\n');
                combined_chars += 1;
            }
            let start = combined_chars;
            let text = row
                .get("zh_text_raw")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            combined.push_str(text);
            combined_chars += char_len(text);
            let end = combined_chars;
            spans.push((start, end));
        }
        let total_chars = combined_chars;

        let anchor_passage_id = anchor_passage
            .as_ref()
            .and_then(|anchor| anchor.get("passage_id").and_then(|v| v.as_str()));
        let anchor_idx = anchor_passage_id.and_then(|pid| {
            rows.iter()
                .position(|row| row.get("passage_id").and_then(|v| v.as_str()) == Some(pid))
        });
        let anchor_start = anchor_idx.map(|idx| spans[idx].0);

        let requested_start = match direction {
            "next" => decoded_cursor.as_ref().map(|c| c.char_start).unwrap_or(0),
            "prev" => decoded_cursor.as_ref().map(|c| c.char_start).unwrap_or(0),
            "at" => decoded_cursor.as_ref().map(|c| c.char_start).unwrap_or(0),
            "around" => anchor_start
                .map(|start| start.saturating_sub(req.before_chars.unwrap_or(overlap_chars)))
                .or_else(|| decoded_cursor.as_ref().map(|c| c.char_start))
                .unwrap_or(0),
            _ => decoded_cursor
                .as_ref()
                .filter(|_| req.cursor.is_some() && direction != "start")
                .map(|c| c.char_start)
                .unwrap_or(0),
        }
        .min(total_chars);

        let around_extra = if direction == "around" {
            anchor_start
                .and_then(|start| anchor_idx.map(|idx| spans[idx].1.saturating_sub(start)))
                .unwrap_or(0)
                .saturating_add(req.before_chars.unwrap_or(overlap_chars))
                .saturating_add(req.after_chars.unwrap_or(overlap_chars))
        } else {
            max_chars
        };
        let target_end = requested_start.saturating_add(around_extra.max(max_chars));
        let (char_end, boundary_quality) =
            adjust_read_end(&combined, requested_start, target_end, total_chars);
        let char_start = requested_start.min(char_end);

        let prev_start = char_start.saturating_sub(overlap_chars);
        let next_end = char_end.saturating_add(overlap_chars).min(total_chars);

        let mut segments = Vec::new();
        if req.include_previous_tail && prev_start < char_start {
            segments.push(SourceReadSegment {
                kind: "previous_overlap".to_string(),
                citeable: false,
                char_start: prev_start,
                char_end: char_start,
                text: char_slice(&combined, prev_start, char_start),
            });
        }
        segments.push(SourceReadSegment {
            kind: "main".to_string(),
            citeable: true,
            char_start,
            char_end,
            text: char_slice(&combined, char_start, char_end),
        });
        if req.include_next_head && char_end < next_end {
            segments.push(SourceReadSegment {
                kind: "next_preview".to_string(),
                citeable: false,
                char_start: char_end,
                char_end: next_end,
                text: char_slice(&combined, char_end, next_end),
            });
        }

        let start_idx = spans
            .iter()
            .position(|(_, end)| *end > char_start)
            .unwrap_or(0);
        let end_idx = spans
            .iter()
            .rposition(|(start, _)| *start < char_end)
            .unwrap_or(start_idx);
        let start_row = &rows[start_idx];
        let end_row = &rows[end_idx];
        let work_title = opt_str(start_row, "main_title");
        let section_path =
            split_heading_path(start_row.get("heading_path").and_then(|v| v.as_str()));

        let current_cursor = SourceReadCursor {
            version: 1,
            source_work_id: source_work_id.clone(),
            char_start,
            unit: unit.to_string(),
            max_chars,
            overlap_chars,
            heading_path_prefix: heading_path_prefix.clone(),
        };
        let next_start = char_end.saturating_sub(overlap_chars).min(total_chars);
        let prev_start_cursor = char_start.saturating_sub(max_chars.saturating_sub(overlap_chars));
        let next_cursor = (char_end < total_chars)
            .then(|| {
                encode_source_read_cursor(&SourceReadCursor {
                    char_start: next_start,
                    ..current_cursor.clone()
                })
            })
            .transpose()?;
        let prev_cursor = (char_start > 0)
            .then(|| {
                encode_source_read_cursor(&SourceReadCursor {
                    char_start: prev_start_cursor,
                    ..current_cursor.clone()
                })
            })
            .transpose()?;

        let current_location = start_row
            .get("heading_path")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .or_else(|| opt_str(start_row, "heading"));

        let mut suggested_next_tools = Vec::new();
        if let Some(cursor) = &next_cursor {
            suggested_next_tools.push(suggested_tool(
                "source-read",
                serde_json::json!({"cursor": cursor}),
                "continue reading the source stream",
            ));
        }
        suggested_next_tools.push(suggested_tool(
            "evidence-search",
            serde_json::json!({"phrase": "<exact phrase from current chunk>", "include_attestation": true}),
            "verify important phrases from this chunk as exact evidence",
        ));

        let estimated_percent = if total_chars == 0 {
            0.0
        } else {
            round_percent((char_end as f64 / total_chars as f64) * 100.0)
        };
        let metadata = req.include_metadata.then(|| {
            serde_json::json!({
                "source_work_id": source_work_id,
                "work_title": work_title,
                "canon": start_row.get("canon"),
                "period": start_row.get("period"),
                "author": start_row.get("author"),
                "passage_start_metadata": {
                    "passage_id": start_row.get("passage_id"),
                    "heading": start_row.get("heading"),
                    "heading_path": start_row.get("heading_path"),
                    "from_lb": start_row.get("from_lb"),
                    "to_lb": start_row.get("to_lb"),
                    "source_rel_path": start_row.get("source_rel_path"),
                    "xml_id": start_row.get("xml_id"),
                },
                "passage_end_metadata": {
                    "passage_id": end_row.get("passage_id"),
                    "heading": end_row.get("heading"),
                    "heading_path": end_row.get("heading_path"),
                    "from_lb": end_row.get("from_lb"),
                    "to_lb": end_row.get("to_lb"),
                    "source_rel_path": end_row.get("source_rel_path"),
                    "xml_id": end_row.get("xml_id"),
                }
            })
        });

        Ok(SourceReadResponse {
            schema: "sinorag-source-read-v1",
            source_work_id: source_work_id.clone(),
            work_title: work_title.clone(),
            cursor: SourceReadCursorInfo {
                current: encode_source_read_cursor(&current_cursor)?,
                next: next_cursor,
                prev: prev_cursor,
                has_next: char_end < total_chars,
                has_prev: char_start > 0,
            },
            position: SourceReadPosition {
                char_start,
                char_end,
                total_chars,
                estimated_percent,
                passage_start: opt_str(start_row, "passage_id"),
                passage_end: opt_str(end_row, "passage_id"),
                section_path,
                boundary_policy: "punctuation_or_fixed_char".to_string(),
                boundary_quality,
            },
            segments,
            metadata,
            reading_state: SourceReadingState {
                work_title,
                current_location,
                progress: serde_json::json!({
                    "char_start": char_start,
                    "char_end": char_end,
                    "total_chars": total_chars,
                    "estimated_percent": estimated_percent,
                }),
                running_summary_prompt: "Summarize this chunk and update prior notes, preserving named persons, dates, places, claims, exact phrases, and citation locations.".to_string(),
            },
            suggested_next_tools,
            warnings,
        })
    }

    /// For a search that matched nothing: the word-layout passages sharing
    /// the most words with the phrase, and a note on why exact matching
    /// missed. Both are `None` for a data root without word-layout text, so
    /// CJK-only corpora answer exactly as before.
    async fn related_by_words(
        &self,
        req: &crate::tools::requests::SearchRequest,
    ) -> (
        Option<Vec<crate::tools::responses::SearchHit>>,
        Option<String>,
    ) {
        const RELATED_LIMIT: usize = 5;
        let has_word_corpus = self
            .resolve_passages_path()
            .is_ok_and(|p| crate::normalize::data_root_has_word_corpus(&p));
        let forms = crate::normalize::PhraseForms::new(&req.phrase);
        if !has_word_corpus || forms.is_single() || forms.words.trim().is_empty() {
            return (None, None);
        }
        let no_stemming = "Exact search matches whole words and does not stem: try a shorter \
                           phrase, or other word forms (singular/plural, -ing, -ed) in one \
                           batch-evidence-search call.";

        let related = async {
            let tfidf = self.tfidf().await.ok()?;
            let doc_table = self.doc_table().await.ok()?;
            let passages = self.passages().await.ok()?;
            let ranked = tfidf.query_top_k(&forms.words, RELATED_LIMIT * 8);
            let ids: Vec<String> = ranked
                .iter()
                .filter_map(|(doc_id, _)| doc_table.passage_id(*doc_id).map(str::to_string))
                .collect();
            if ids.is_empty() {
                return None;
            }
            let rows = passages
                .passages_by_ids(
                    &ids,
                    "passage_id, source_work_id, main_title, heading, zh_text_raw, zh_text_normalized",
                )
                .await
                .ok()?;
            let mut hits: Vec<crate::tools::responses::SearchHit> = rows
                .iter()
                .filter_map(|row| {
                    let passage_id = row.get("passage_id")?.as_str()?;
                    let normalized = row.get("zh_text_normalized")?.as_str()?;
                    if !crate::normalize::has_word_layout(normalized) {
                        return None;
                    }
                    let work = opt_str(row, "source_work_id");
                    if req.source_work_id.is_some() && work != req.source_work_id {
                        return None;
                    }
                    let score = ranked
                        .iter()
                        .find(|(doc_id, _)| doc_table.passage_id(*doc_id) == Some(passage_id))
                        .map(|(_, score)| *score)?;
                    Some(crate::tools::responses::SearchHit {
                        passage_id: passage_id.to_string(),
                        source_work_id: work,
                        main_title: opt_str(row, "main_title"),
                        heading: opt_str(row, "heading"),
                        zh_quote: row
                            .get("zh_text_raw")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .chars()
                            .take(if req.brief { 120 } else { 400 })
                            .collect(),
                        score: Some(score),
                    })
                })
                .collect();
            hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            hits.truncate(RELATED_LIMIT);
            (!hits.is_empty()).then_some(hits)
        }
        .await;

        // Passages holding one of the phrase's rarer words are leads of a
        // second kind, and the kind the ranking above cannot give: a word
        // has to be in several passages before it is a TF-IDF feature at
        // all. Neither list is reliably the better one, so take from each in
        // turn.
        let mut by_rare_word = self
            .passages_with_rare_words(req, &forms.words)
            .await
            .into_iter();
        let mut by_similarity = related.unwrap_or_default().into_iter();
        let mut leads: Vec<crate::tools::responses::SearchHit> = Vec::new();
        while leads.len() < RELATED_LIMIT {
            let pair = [by_rare_word.next(), by_similarity.next()];
            if pair.iter().all(Option::is_none) {
                break;
            }
            for hit in pair.into_iter().flatten() {
                if !leads.iter().any(|lead| lead.passage_id == hit.passage_id) {
                    leads.push(hit);
                }
            }
        }
        leads.truncate(RELATED_LIMIT);
        let related = (!leads.is_empty()).then_some(leads);

        let note = match &related {
            Some(hits) => format!(
                "No passage contains this exact phrase. related_passages lists {} that share \
                 several of its words, or one of its rarer words: leads to read with `passage`, \
                 not matches. {no_stemming}",
                hits.len()
            ),
            None => format!("No passage contains this exact phrase. {no_stemming}"),
        };
        (related, Some(note))
    }

    /// Passages that contain a rare content word of a multi-word phrase,
    /// those matching the most (and the rarest) words first.
    async fn passages_with_rare_words(
        &self,
        req: &crate::tools::requests::SearchRequest,
        words: &str,
    ) -> Vec<crate::tools::responses::SearchHit> {
        const MAX_WORDS: usize = 8;
        /// A word in more passages than this says little about any of them.
        const RARE: usize = 40;
        let mut content: Vec<&str> = Vec::new();
        for word in words.split_whitespace() {
            if crate::tfidf::ngram::keep_word(word) && !content.contains(&word) {
                content.push(word);
            }
        }
        // One word is what `search` just looked for and did not find.
        if words.split_whitespace().count() < 2 {
            return Vec::new();
        }
        content.truncate(MAX_WORDS);

        let mut tally: Vec<(f32, crate::tools::responses::SearchHit)> = Vec::new();
        for word in content {
            let search = crate::tools::requests::SearchRequest {
                phrase: word.to_string(),
                limit: RARE + 1,
                mode: "hits".to_string(),
                depth: "exact".to_string(),
                group_by: "work".to_string(),
                include_variants: false,
                limit_per_group: 1,
                brief: req.brief,
                canon: None,
                source_work_id: req.source_work_id.clone(),
                tradition: None,
                period: None,
                origin: None,
                author: None,
                title: None,
                heading_path_prefix: req.heading_path_prefix.clone(),
            };
            let Ok(found) = Box::pin(self.search_with(search, false)).await else {
                continue;
            };
            if found.hits.is_empty() || found.hits.len() > RARE {
                continue;
            }
            // 1 for a word in one passage, about 0.2 at the limit: two
            // shared words outweigh one rarer word.
            let weight = 1.0 / (1.0 + (found.hits.len() as f32).ln());
            for hit in found.hits {
                match tally
                    .iter_mut()
                    .find(|(_, seen)| seen.passage_id == hit.passage_id)
                {
                    Some((score, _)) => *score += weight,
                    None => tally.push((weight, hit)),
                }
            }
        }
        tally.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        tally
            .into_iter()
            .map(|(_, mut hit)| {
                // An excerpt, like the other leads; the score of those is a
                // similarity, which this is not.
                if !req.brief {
                    hit.zh_quote = hit.zh_quote.chars().take(400).collect();
                }
                hit.score = None;
                hit
            })
            .collect()
    }

    /// Implement the search tool
    pub async fn search_impl(
        &self,
        req: crate::tools::requests::SearchRequest,
    ) -> Result<crate::tools::responses::SearchResponse> {
        self.search_with(req, true).await
    }

    /// `search`, with or without the leads it offers when nothing matches.
    /// Finding those leads runs `search` on single words; those inner
    /// searches pass `false`, so they cannot go looking for leads in turn.
    async fn search_with(
        &self,
        req: crate::tools::requests::SearchRequest,
        offer_leads: bool,
    ) -> Result<crate::tools::responses::SearchResponse> {
        use crate::datafusion_store::sql_literal;
        use crate::normalize::normalize_zh;
        use crate::research_tools::phrase::phrase_rows_with_explicit_doc_table;
        use crate::research_tools::scopes::{group_hits_by_outline_node, OutlineSearchLevel};
        use crate::tools::responses::{
            ClusterHitsCluster, SearchHit, SearchResponse, SearchStrategy, TermUsageGroup,
        };

        // An unknown value used to fall back to the default without a word,
        // so `mode: "cluster"` returned plain hits that looked like the answer.
        let one_of = |name: &str, value: &str, allowed: &[&str]| -> Result<()> {
            if allowed.contains(&value) {
                return Ok(());
            }
            Err(ToolError::InvalidArgs(format!(
                "search: `{name}` is \"{value}\"; use one of: {}",
                allowed.join(", ")
            ))
            .into_anyhow())
        };
        one_of("mode", &req.mode, &["hits", "clusters", "trace", "all"])?;
        one_of("depth", &req.depth, &["exact", "expanded", "reuse"])?;
        one_of("group_by", &req.group_by, &["work", "division"])?;

        let passages = self.passages().await?;
        let canon = expand_optional_filter(req.canon.as_deref());
        let tradition = expand_tradition_filter(req.tradition.as_deref());
        let period = expand_period_filter(req.period.as_deref());
        let origin = expand_origin_filter(req.origin.as_deref());
        let normalized = normalize_zh(&req.phrase);
        let mode = req.mode.clone();
        let depth = req.depth.as_str();

        let mut phrases = vec![req.phrase.clone()];
        if req.include_variants || matches!(depth, "expanded" | "reuse") {
            let tables = crate::templates::variants::VariantTables::load();
            let mut seen = std::collections::BTreeSet::<String>::new();
            seen.insert(req.phrase.clone());
            for v in tables.term_variants(&req.phrase) {
                if seen.insert(v.clone()) {
                    phrases.push(v);
                }
            }
            let cur = phrases.clone();
            for p in cur {
                for v in tables.orthographic_flips(&p, 20) {
                    if seen.insert(v.clone()) {
                        phrases.push(v);
                    }
                }
            }
            phrases.truncate(20);
        }

        let doc_table = self.doc_table().await.ok();
        let phrase_index_path = if doc_table.is_some() {
            self.optional_phrase_path().await?
        } else {
            None
        };
        let catalog = self.catalog().await.ok();

        let doc_range = if let (Some(catalog), Some(work_id)) =
            (catalog.as_deref(), req.source_work_id.as_deref())
        {
            self.resolve_doc_range(catalog, None, Some(work_id))?
        } else {
            None
        };

        let canon_for_index = if canon.is_empty() {
            None
        } else {
            Some(canon.as_slice())
        };
        let period_for_index = if period.len() == 1 {
            Some(period[0].as_str())
        } else {
            None
        };
        let limit = req.limit.max(1);
        let per_phrase_limit = if phrases.len() > 1 {
            limit.saturating_mul(2).max(limit)
        } else {
            limit
        };

        let mut rows = Vec::<serde_json::Value>::new();
        let mut strategies = Vec::<serde_json::Value>::new();

        for phrase in &phrases {
            let phrase_rows = if let Some(doc_table) = doc_table.as_deref() {
                let (candidate_rows, strategy) = phrase_rows_with_explicit_doc_table(
                    &passages,
                    doc_table,
                    phrase_index_path.as_deref(),
                    phrase,
                    per_phrase_limit,
                    doc_range,
                    canon_for_index,
                    period_for_index,
                )
                .await?;
                strategies.push(serde_json::json!({
                    "phrase": phrase,
                    "normalized_phrase": normalize_zh(phrase),
                    "strategy": strategy,
                }));
                candidate_rows
            } else {
                let normalized_phrase = normalize_zh(phrase);
                let phrase_forms = crate::normalize::PhraseForms::new(phrase);
                let mut where_parts = Vec::new();
                if !phrase_forms.is_empty() {
                    where_parts.push(crate::datafusion_store::phrase_match_sql(
                        "zh_text_normalized",
                        &phrase_forms,
                    ));
                }
                if let Some(canon) = canon_for_index {
                    let quoted = canon
                        .iter()
                        .map(|c| sql_literal(c))
                        .collect::<Vec<_>>()
                        .join(", ");
                    where_parts.push(format!("canon IN ({quoted})"));
                }
                if let Some(period) = period_for_index {
                    where_parts.push(format!("period = {}", sql_literal(period)));
                }
                let where_sql = if where_parts.is_empty() {
                    "true".to_string()
                } else {
                    where_parts.join(" AND ")
                };
                let sql = format!(
                    "SELECT passage_id, source_rel_path, xml_id, div_path, heading, heading_path, \
                            from_lb, to_lb, zh_text_raw, zh_text_normalized, text_type, \
                            contains_person, contains_term, contains_foreign, canon, canon_name, \
                            traditions, period, origin, author, main_title, period_rank, \
                            source_corpus, source_work_id, source_section_id, source_locator, \
                            source_url, edition_siglum, edition_label, rights_id, rights_notes, \
                            retrieval_method, snapshot_id, quality_flags_json \
                     FROM passages \
                     WHERE {where_sql} \
                     ORDER BY period_rank ASC, source_rel_path ASC, from_lb ASC, xml_id ASC \
                     LIMIT {per_phrase_limit}"
                );
                strategies.push(serde_json::json!({
                    "phrase": phrase,
                    "normalized_phrase": normalized_phrase,
                    "strategy": {
                        "used_phrase_index": false,
                        "scope_scan": "parquet_global_no_doc_table",
                        "limit": per_phrase_limit,
                    },
                }));
                passages.query_json(&sql).await?
            };
            rows.extend(phrase_rows);
        }

        rows.retain(|row| {
            if !canon.is_empty() {
                let value = row.get("canon").and_then(|v| v.as_str()).unwrap_or("");
                if !canon.iter().any(|c| c == value) {
                    return false;
                }
            }
            if !period.is_empty() {
                let value = row.get("period").and_then(|v| v.as_str()).unwrap_or("");
                if !period.iter().any(|p| p == value) {
                    return false;
                }
            }
            if !origin.is_empty() {
                let value = row.get("origin").and_then(|v| v.as_str()).unwrap_or("");
                if !origin.iter().any(|o| o == value) {
                    return false;
                }
            }
            if !tradition.is_empty() {
                let values = row.get("traditions").and_then(|v| v.as_array());
                let has_match = values
                    .map(|vals| {
                        vals.iter()
                            .filter_map(|v| v.as_str())
                            .any(|v| tradition.iter().any(|t| t == v))
                    })
                    .unwrap_or(false);
                if !has_match {
                    return false;
                }
            }
            if let Some(ref author) = req.author {
                let value = row
                    .get("author")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                if !value.contains(&author.to_lowercase()) {
                    return false;
                }
            }
            if let Some(ref title) = req.title {
                let value = row
                    .get("main_title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                if !value.contains(&title.to_lowercase()) {
                    return false;
                }
            }
            if let Some(ref work_id) = req.source_work_id {
                let value = row
                    .get("source_work_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if value != work_id {
                    return false;
                }
            }
            if let Some(ref prefix) = req.heading_path_prefix {
                let value = row
                    .get("heading_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if !value.starts_with(prefix) {
                    return false;
                }
            }
            true
        });

        let mut deduped = Vec::new();
        let mut seen = std::collections::BTreeSet::<String>::new();
        for row in rows {
            let key = row
                .get("passage_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if key.is_empty() || seen.insert(key) {
                deduped.push(row);
            }
        }
        if let Some(doc_table) = doc_table.as_deref() {
            deduped.sort_by_key(|row| {
                let pid = row.get("passage_id").and_then(|v| v.as_str()).unwrap_or("");
                doc_table.doc_id(pid).unwrap_or(u32::MAX)
            });
        } else {
            deduped.sort_by(|a, b| {
                let ak = (
                    a.get("period_rank").and_then(|v| v.as_i64()).unwrap_or(99),
                    a.get("source_rel_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                    a.get("from_lb").and_then(|v| v.as_str()).unwrap_or(""),
                    a.get("xml_id").and_then(|v| v.as_str()).unwrap_or(""),
                );
                let bk = (
                    b.get("period_rank").and_then(|v| v.as_i64()).unwrap_or(99),
                    b.get("source_rel_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                    b.get("from_lb").and_then(|v| v.as_str()).unwrap_or(""),
                    b.get("xml_id").and_then(|v| v.as_str()).unwrap_or(""),
                );
                ak.cmp(&bk)
            });
        }
        let verified_count = deduped.len();
        deduped.truncate(limit);

        let hits: Vec<SearchHit> = deduped
            .iter()
            .map(|row| SearchHit {
                passage_id: row
                    .get("passage_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                source_work_id: row
                    .get("source_work_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                main_title: row
                    .get("main_title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                heading: row
                    .get("heading")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                zh_quote: row
                    .get("zh_text_raw")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .chars()
                    .take(if req.brief { 120 } else { usize::MAX })
                    .collect(),
                score: None,
            })
            .collect();

        // Nothing matched exactly. In English text that is the usual result
        // of a question typed as a phrase, so say how matching works and
        // offer the passages that share the most words instead of a dead end.
        let (related_passages, note) = if hits.is_empty() && offer_leads {
            self.related_by_words(&req).await
        } else {
            (None, None)
        };

        let clusters = if matches!(mode.as_str(), "clusters" | "all") {
            if let (Some(catalog), Some(doc_table)) = (catalog.as_deref(), doc_table.as_deref()) {
                let target = match req.group_by.as_str() {
                    "division" => OutlineSearchLevel::Division,
                    _ => OutlineSearchLevel::Work,
                };
                let doc_rows: Vec<(u32, serde_json::Value)> = deduped
                    .iter()
                    .filter_map(|row| {
                        let pid = row.get("passage_id").and_then(|v| v.as_str())?;
                        Some((doc_table.doc_id(pid)?, row.clone()))
                    })
                    .collect();
                let doc_ids: Vec<u32> = doc_rows.iter().map(|(did, _)| *did).collect();
                let mut sorted_groups: Vec<(u32, u32)> =
                    group_hits_by_outline_node(catalog, &doc_ids, target)
                        .into_iter()
                        .collect();
                sorted_groups.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                Some(
                    sorted_groups
                        .into_iter()
                        .take(req.limit_per_group)
                        .map(|(node_id, count)| {
                            let node = catalog.get_node(node_id);
                            let node_doc_range =
                                node.and_then(|n| n.first_doc_id.zip(n.last_doc_id));
                            let representative_passages = doc_rows
                                .iter()
                                .filter(|(did, _)| {
                                    if let Some((lo, hi)) = node_doc_range {
                                        *did >= lo && *did <= hi
                                    } else {
                                        false
                                    }
                                })
                                .take(3)
                                .map(|(did, row)| {
                                    let mut r = row.clone();
                                    if let Some(obj) = r.as_object_mut() {
                                        obj.insert("doc_id".to_string(), serde_json::json!(*did));
                                    }
                                    if req.brief {
                                        brief_row(&r)
                                    } else {
                                        r
                                    }
                                })
                                .collect();
                            ClusterHitsCluster {
                                node_id,
                                label: node.map(|n| n.label.clone()).unwrap_or_default(),
                                heading_path: node
                                    .map(|n| n.heading_path.clone())
                                    .unwrap_or_default(),
                                node_kind: node
                                    .map(|n| format!("{:?}", &n.node_kind))
                                    .unwrap_or_default(),
                                hit_count: count,
                                representative_passages,
                            }
                        })
                        .collect(),
                )
            } else {
                let key_field = match req.group_by.as_str() {
                    "division" => "heading_path",
                    _ => "source_work_id",
                };
                let mut groups =
                    std::collections::BTreeMap::<String, (u32, Vec<serde_json::Value>)>::new();
                for row in &deduped {
                    let key = row
                        .get(key_field)
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .or_else(|| row.get("source_work_id").and_then(|v| v.as_str()))
                        .unwrap_or("(unknown)")
                        .to_string();
                    let acc = groups.entry(key).or_insert_with(|| (0, Vec::new()));
                    acc.0 += 1;
                    if acc.1.len() < 3 {
                        acc.1.push(if req.brief {
                            brief_row(row)
                        } else {
                            row.clone()
                        });
                    }
                }
                let mut sorted_groups: Vec<(String, u32, Vec<serde_json::Value>)> = groups
                    .into_iter()
                    .map(|(key, (count, reps))| (key, count, reps))
                    .collect();
                sorted_groups.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                Some(
                    sorted_groups
                        .into_iter()
                        .take(req.limit_per_group)
                        .enumerate()
                        .map(
                            |(idx, (label, count, representative_passages))| ClusterHitsCluster {
                                node_id: idx as u32,
                                label: label.clone(),
                                heading_path: label,
                                node_kind: "MetadataFallback".to_string(),
                                hit_count: count,
                                representative_passages,
                            },
                        )
                        .collect(),
                )
            }
        } else {
            None
        };

        let trace_groups = if matches!(mode.as_str(), "trace" | "all") {
            let key_field = match req.group_by.as_str() {
                "canon" => "canon",
                "author" => "author",
                "work" | "division" => "source_work_id",
                _ => "period",
            };
            struct GroupAcc {
                hit_count: u32,
                work_ids: std::collections::BTreeSet<String>,
                reps: Vec<(i32, u32, serde_json::Value)>,
            }
            impl Default for GroupAcc {
                fn default() -> Self {
                    Self {
                        hit_count: 0,
                        work_ids: std::collections::BTreeSet::new(),
                        reps: Vec::new(),
                    }
                }
            }
            let mut groups = std::collections::BTreeMap::<String, GroupAcc>::new();
            for row in &deduped {
                let key = row
                    .get(key_field)
                    .and_then(|v| v.as_str())
                    .unwrap_or("(unknown)")
                    .to_string();
                let acc = groups.entry(key).or_default();
                acc.hit_count += 1;
                if let Some(wid) = row.get("source_work_id").and_then(|v| v.as_str()) {
                    acc.work_ids.insert(wid.to_string());
                }
                let did = doc_table
                    .as_deref()
                    .and_then(|dt| {
                        row.get("passage_id")
                            .and_then(|v| v.as_str())
                            .and_then(|pid| dt.doc_id(pid))
                    })
                    .unwrap_or(u32::MAX);
                let pr = row.get("period_rank").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                acc.reps.push((pr, did, row.clone()));
            }
            Some(
                groups
                    .into_iter()
                    .map(|(key, mut acc)| {
                        acc.reps.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
                        let mut top_works: Vec<String> = acc.work_ids.into_iter().collect();
                        top_works.sort();
                        top_works.truncate(req.limit_per_group);
                        TermUsageGroup {
                            key,
                            hit_count: acc.hit_count,
                            work_count: top_works.len(),
                            top_works,
                            representative_passages: acc
                                .reps
                                .into_iter()
                                .take(req.limit_per_group)
                                .map(|(_, _, row)| row)
                                .collect(),
                        }
                    })
                    .collect(),
            )
        } else {
            None
        };

        let used_phrase_index = strategies.iter().any(|s| {
            s.pointer("/strategy/used_phrase_index")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        });

        // Suggest next steps based on the shape of *this* result (zero hits,
        // a single anchor, a tight cluster, or a corpus-wide spread each call
        // for something different) rather than a fixed tip on every search.
        // Drop candidates whose tool the agent already pivoted to recently so
        // a followed suggestion doesn't keep resurfacing.
        let suggested_next_tools = {
            let candidates = search_suggestion_candidates(&req.phrase, &hits, verified_count);
            let recent_tools =
                crate::tools::log::recent_tool_names(&crate::tools::log::default_log_path(), 12)
                    .unwrap_or_default();
            candidates
                .into_iter()
                .filter(|(tool, _, _)| !recent_tools.iter().any(|r| r == tool))
                .take(2)
                .map(|(tool, args, reason)| suggested_tool(tool, args, &reason))
                .collect::<Vec<_>>()
        };

        Ok(SearchResponse {
            schema: "sinorag-search-v1",
            phrase: req.phrase,
            mode,
            brief: req.brief,
            expanded_phrases: phrases,
            hits,
            related_passages,
            note,
            clusters,
            trace_groups,
            search_strategy: SearchStrategy {
                method: if used_phrase_index {
                    "phrase_index_verified_by_parquet".to_string()
                } else {
                    "parquet_strpos_scan".to_string()
                },
                filters: serde_json::json!({
                    "canon": canon,
                    "tradition": tradition,
                    "period": period,
                    "origin": origin,
                    "author": req.author,
                    "title": req.title,
                    "source_work_id": req.source_work_id,
                    "heading_path_prefix": req.heading_path_prefix,
                    "mode": req.mode,
                    "depth": req.depth,
                    "group_by": req.group_by,
                    "include_variants": req.include_variants,
                    "brief": req.brief,
                    "normalized_phrase": normalized,
                    "limit": limit,
                    "limit_per_group": req.limit_per_group,
                    "layers": strategies,
                }),
                candidate_count: None,
                verified_count: Some(verified_count),
                candidate_source: Some(if used_phrase_index {
                    "phrase_index".to_string()
                } else {
                    "parquet_scan".to_string()
                }),
                verification_source: Some("parquet_text".to_string()),
                used_phrase_index: Some(used_phrase_index),
                fallback_reason: if used_phrase_index {
                    None
                } else {
                    Some("phrase_index_unavailable_or_not_applicable".to_string())
                },
            },
            suggested_next_tools,
        })
    }

    /// Implement the canonical-source tool
    pub async fn canonical_source_impl(
        &self,
        req: crate::tools::requests::CanonicalSourceRequest,
    ) -> Result<crate::tools::responses::CanonicalSourceResponse> {
        use crate::normalize::normalize_zh;
        use crate::tools::responses::{
            CanonicalSourceHit, CanonicalSourceResponse, SearchStrategy,
        };

        let passages = self.passages().await?;

        let normalized = normalize_zh(&req.phrase);
        let phrase_forms = crate::normalize::PhraseForms::new(&req.phrase);
        let mut where_parts = Vec::new();
        if !phrase_forms.is_empty() {
            where_parts.push(crate::datafusion_store::phrase_match_sql(
                "zh_text_normalized",
                &phrase_forms,
            ));
        }

        let canon = expand_optional_filter(req.canon.as_deref());
        if canon.is_empty() {
            where_parts.push("canon IS NOT NULL AND canon != ''".to_string());
        } else {
            exact_any_sql(&mut where_parts, "canon", &canon);
        }

        let where_sql = if where_parts.is_empty() {
            "true".to_string()
        } else {
            where_parts.join(" AND ")
        };
        let limit = req.limit.max(1);
        let sql = format!(
            "SELECT passage_id, source_work_id, main_title, heading, zh_text_raw, \
                    canon, traditions, period, origin, period_rank, source_rel_path, from_lb, xml_id \
             FROM passages \
             WHERE {where_sql} \
             ORDER BY period_rank ASC, source_rel_path ASC, from_lb ASC, xml_id ASC \
             LIMIT {limit}"
        );

        let rows = passages.query_json(&sql).await?;

        let hits: Vec<CanonicalSourceHit> = rows
            .iter()
            .take(req.limit)
            .map(|row| CanonicalSourceHit {
                passage_id: row
                    .get("passage_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                source_work_id: row
                    .get("source_work_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                main_title: row
                    .get("main_title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                heading: row
                    .get("heading")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                zh_quote: row
                    .get("zh_text_raw")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                is_canon_side: true,
            })
            .collect();

        let hit_count = hits.len();

        Ok(CanonicalSourceResponse {
            schema: "sinorag-canonical-source-v1",
            phrase: req.phrase,
            hits,
            search_strategy: SearchStrategy {
                method: "full_text_with_tradition_filter".to_string(),
                filters: serde_json::json!({
                    "canon": req.canon,
                    "normalized_phrase": normalized,
                    "canonical_filter": if canon.is_empty() { "canon != ''" } else { "canon IN (...)" }
                }),
                candidate_count: Some(rows.len()),
                verified_count: Some(hit_count),
                candidate_source: Some("parquet_scan".to_string()),
                verification_source: Some("parquet_text".to_string()),
                used_phrase_index: Some(false),
                fallback_reason: None,
            },
        })
    }

    /// Implement the heading-search tool
    pub async fn heading_search_impl(
        &self,
        req: crate::tools::requests::HeadingSearchRequest,
    ) -> Result<crate::tools::responses::HeadingSearchResponse> {
        use crate::datafusion_store::sql_literal;
        use crate::tools::responses::{HeadingSearchHit, HeadingSearchResponse, SearchStrategy};

        let passages = self.passages().await?;
        let canon = expand_optional_filter(req.canon.as_deref());
        let period = expand_period_filter(req.period.as_deref());
        let normalized_query = crate::normalize::normalize_zh(&req.query);
        let mut where_parts = Vec::new();
        if !req.query.is_empty() {
            where_parts.push(format!(
                "(strpos(lower(heading), lower({q})) > 0 OR strpos(lower(heading_path), lower({q})) > 0 OR {norm})",
                q = sql_literal(&req.query),
                norm = crate::datafusion_store::phrase_match_sql(
                    "zh_text_normalized",
                    &crate::normalize::PhraseForms::new(&req.query)
                ),
            ));
        }
        exact_any_sql(&mut where_parts, "canon", &canon);
        exact_any_sql(&mut where_parts, "period", &period);
        if let Some(ref work_id) = req.source_work_id {
            where_parts.push(format!("source_work_id = {}", sql_literal(work_id)));
        }

        let where_sql = if where_parts.is_empty() {
            "true".to_string()
        } else {
            where_parts.join(" AND ")
        };
        let limit = req.limit.max(1);
        let sql = format!(
            "SELECT passage_id, source_rel_path, xml_id, div_path, heading, heading_path, \
                    from_lb, to_lb, zh_text_raw, zh_text_normalized, text_type, canon, \
                    canon_name, traditions, period, origin, author, main_title, period_rank, \
                    source_work_id, source_section_id, source_locator, source_url \
             FROM passages \
             WHERE {where_sql} \
             ORDER BY period_rank ASC, source_rel_path ASC, from_lb ASC, xml_id ASC \
             LIMIT {limit}"
        );
        let rows = passages.query_json(&sql).await?;
        let sections: Vec<HeadingSearchHit> = rows
            .iter()
            .map(|row| HeadingSearchHit {
                source_work_id: row
                    .get("source_work_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                main_title: row
                    .get("main_title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                heading: row
                    .get("heading")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                heading_path: row
                    .get("heading_path")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                passage_id: row
                    .get("passage_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                sample: row
                    .get("zh_text_raw")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .chars()
                    .take(if req.brief { 80 } else { 240 })
                    .collect(),
                metadata: if req.brief {
                    None
                } else {
                    Some(serde_json::json!({
                        "canon": row.get("canon"),
                        "period": row.get("period"),
                        "author": row.get("author"),
                        "source_rel_path": row.get("source_rel_path"),
                        "from_lb": row.get("from_lb"),
                        "to_lb": row.get("to_lb"),
                    }))
                },
            })
            .collect();
        Ok(HeadingSearchResponse {
            schema: "sinorag-heading-search-v1",
            query: req.query,
            brief: req.brief,
            returned_count: sections.len(),
            sections,
            search_strategy: SearchStrategy {
                method: "heading_path_metadata_scan".to_string(),
                filters: serde_json::json!({
                    "canon": canon,
                    "period": period,
                    "source_work_id": req.source_work_id,
                    "normalized_query": normalized_query,
                    "limit": limit,
                    "brief": req.brief,
                }),
                candidate_count: None,
                verified_count: Some(rows.len()),
                candidate_source: Some("metadata_scan".to_string()),
                verification_source: Some("parquet_metadata".to_string()),
                used_phrase_index: Some(false),
                fallback_reason: None,
            },
        })
    }

    /// Implement the tool-docs tool
    pub async fn tool_docs_impl(
        &self,
        req: crate::tools::requests::ToolDocsRequest,
    ) -> Result<crate::tools::responses::ToolDocsResponse> {
        let docs = crate::tools::docs::docs_payload(req.tool.as_deref());
        Ok(crate::tools::responses::ToolDocsResponse {
            schema: "sinorag-tool-docs-v1",
            tool: req.tool,
            docs,
        })
    }

    /// Implement the tool-log-summary tool.
    pub async fn tool_log_summary_impl(
        &self,
        req: crate::tools::requests::ToolLogSummaryRequest,
    ) -> Result<crate::tools::responses::ToolLogSummaryResponse> {
        let path = req.path.unwrap_or_else(crate::tools::log::default_log_path);
        let summary = crate::tools::log::summarize(&path, req.recent)?;
        Ok(crate::tools::responses::ToolLogSummaryResponse {
            schema: "sinorag-tool-log-summary-v1",
            summary: serde_json::to_value(summary)?,
        })
    }

    /// Implement the validate-adjudication tool
    pub async fn validate_adjudication_impl(
        &self,
        req: crate::tools::requests::ValidateAdjudicationRequest,
    ) -> Result<crate::tools::responses::ValidateAdjudicationResponse> {
        use crate::commands::validate;
        use crate::tools::responses::ValidateAdjudicationResponse;

        // validate::run returns Result<()> and prints to stdout
        // For now, we'll just call it and assume success if it doesn't error
        validate::run(req.path.clone())?;

        Ok(ValidateAdjudicationResponse {
            schema: "sinorag-validate-adjudication-v1",
            path: req.path,
            valid: true,
            errors: vec![],
            warnings: vec![],
        })
    }

    /// Implement the graph-build tool
    pub async fn graph_build_impl(
        &self,
        req: crate::tools::requests::GraphBuildRequest,
    ) -> Result<crate::tools::responses::GraphBuildResponse> {
        use crate::commands::export;
        use crate::tools::responses::GraphBuildResponse;

        self.ensure_write_allowed("graph-build", &req.out)?;

        let graph_kind = match req.kind.as_str() {
            "evidence" => export::GraphKind::Evidence,
            "timeline" => export::GraphKind::Timeline,
            "lineage" => export::GraphKind::Lineage,
            other => anyhow::bail!(
                "unknown graph kind `{}`; expected evidence, timeline, or lineage",
                other
            ),
        };

        export::graph(
            req.input.clone(),
            Some(req.out.clone()),
            graph_kind,
            Some(req.name.clone()),
        )?;

        // Read back the graph to get counts
        let graph_content = std::fs::read_to_string(&req.out)?;
        let graph: serde_json::Value = serde_json::from_str(&graph_content)?;

        let node_count = graph["nodes"].as_array().map(|v| v.len()).unwrap_or(0);
        let edge_count = graph["edges"].as_array().map(|v| v.len()).unwrap_or(0);

        Ok(GraphBuildResponse {
            schema: "sinorag-graph-build-v1",
            out: req.out,
            node_count,
            edge_count,
        })
    }

    /// Implement the report-build tool
    pub async fn report_build_impl(
        &self,
        req: crate::tools::requests::ReportBuildRequest,
    ) -> Result<crate::tools::responses::ReportBuildResponse> {
        use crate::commands::export;
        use crate::tools::responses::ReportBuildResponse;

        self.ensure_write_allowed("report-build", &req.out)?;

        export::report_build(
            req.inputs.clone(),
            req.out.clone(),
            req.title.clone(),
            req.essay_max_pages,
        )?;

        // Count sections in the generated markdown
        let content = std::fs::read_to_string(&req.out)?;
        let section_count = content.matches("##").count();

        Ok(ReportBuildResponse {
            schema: "sinorag-report-build-v1",
            out: req.out,
            section_count,
        })
    }

    /// Implement the pdf-build tool
    pub async fn pdf_build_impl(
        &self,
        req: crate::tools::requests::PdfBuildRequest,
    ) -> Result<crate::tools::responses::PdfBuildResponse> {
        use crate::commands::export;
        use crate::templates;
        use crate::tools::responses::PdfBuildResponse;

        self.ensure_write_allowed("pdf-build", &req.out)?;

        let source_count = usize::from(req.markdown.is_some())
            + usize::from(req.input_markdown.is_some())
            + usize::from(req.input_json.is_some());
        if source_count != 1 {
            return Err(ToolError::InvalidArgs(
                "pdf-build expects exactly one of markdown, input_markdown, or input_json"
                    .to_string(),
            )
            .into_anyhow());
        }

        let (source_format, section_count) = if let Some(markdown) = req.markdown {
            let section_count = markdown
                .lines()
                .filter(|line| line.trim_start().starts_with('#'))
                .count()
                .max(1);
            export::pdf_markdown(&markdown, req.out.clone(), req.side_by_side)?;
            ("inline_markdown".to_string(), section_count)
        } else if let Some(input) = req.input_markdown {
            let markdown = std::fs::read_to_string(&input)?;
            let section_count = markdown
                .lines()
                .filter(|line| line.trim_start().starts_with('#'))
                .count()
                .max(1);
            export::pdf(input, req.out.clone(), req.side_by_side)?;
            ("markdown".to_string(), section_count)
        } else if let Some(input) = req.input_json {
            let payload = templates::read_json(&input)?;
            let sections =
                templates::pdf_report::render(&payload, req.title.as_deref(), req.essay_max_pages);
            let section_count = sections.english.len();
            export::pdf_report(
                input,
                req.out.clone(),
                req.side_by_side,
                req.title,
                req.essay_max_pages,
            )?;
            ("json_basic_report".to_string(), section_count)
        } else {
            unreachable!("source_count validation ensures one input is present");
        };

        Ok(PdfBuildResponse {
            schema: "sinorag-pdf-build-v1",
            out: req.out,
            source_format,
            section_count,
            side_by_side: req.side_by_side,
        })
    }

    /// Implement the work-metadata tool
    pub async fn work_metadata_impl(
        &self,
        req: crate::tools::requests::WorkMetadataRequest,
    ) -> Result<crate::tools::responses::WorkMetadataResponse> {
        use std::collections::HashSet;

        use crate::tools::responses::{WorkInfo, WorkMetadataResponse};

        if req.work_ids.is_empty() {
            return Err(ToolError::InvalidArgs(
                "work-metadata requires at least one work_id".to_string(),
            )
            .into_anyhow());
        }
        if req.work_ids.len() > 100 {
            return Err(ToolError::InvalidArgs(
                "work-metadata accepts at most 100 work_ids per call".to_string(),
            )
            .into_anyhow());
        }

        let catalog = self.catalog().await?;
        let mut seen = HashSet::new();
        let mut works = Vec::new();
        let mut missing_work_ids = Vec::new();

        for raw_work_id in req.work_ids {
            let work_id = raw_work_id.trim();
            if work_id.is_empty() {
                return Err(ToolError::InvalidArgs(
                    "work-metadata work_ids must not contain empty strings".to_string(),
                )
                .into_anyhow());
            }
            if !seen.insert(work_id.to_string()) {
                continue;
            }

            if let Some(work) = catalog.get_work(work_id) {
                works.push(WorkInfo {
                    work_id: work.work_id.clone(),
                    main_title: work.main_title.clone(),
                    author: Some(work.author.clone()),
                    period: Some(work.period.clone()),
                    canon: Some(work.canon.clone()),
                    traditions: work.traditions.clone(),
                    passage_count: work.passage_count as usize,
                });
            } else {
                missing_work_ids.push(work_id.to_string());
            }
        }

        Ok(WorkMetadataResponse {
            schema: "sinorag-work-metadata-v1",
            requested_count: seen.len(),
            found_count: works.len(),
            works,
            missing_work_ids,
        })
    }

    /// Implement the works tool
    pub async fn works_impl(
        &self,
        req: crate::tools::requests::WorksRequest,
    ) -> Result<crate::tools::responses::WorksResponse> {
        use crate::tools::responses::{WorkInfo, WorksResponse};

        let catalog = self.catalog().await?;

        if let Some(ref work_id) = req.work_id {
            let works: Vec<WorkInfo> = catalog
                .get_work(work_id)
                .map(|w| WorkInfo {
                    work_id: w.work_id.clone(),
                    main_title: w.main_title.clone(),
                    author: Some(w.author.clone()),
                    period: Some(w.period.clone()),
                    canon: Some(w.canon.clone()),
                    traditions: w.traditions.clone(),
                    passage_count: w.passage_count as usize,
                })
                .into_iter()
                .collect();
            return Ok(WorksResponse {
                schema: "sinorag-works-v1",
                works,
            });
        }

        let mut filtered: Vec<_> = catalog.works.iter().collect();

        if let Some(ref tradition) = req.tradition {
            filtered.retain(|w| w.traditions.iter().any(|tr| tr == tradition));
        }
        if let Some(ref period) = req.period {
            filtered.retain(|w| &w.period == period);
        }
        if let Some(ref canon) = req.canon {
            filtered.retain(|w| &w.canon == canon);
        }
        if let Some(ref author) = req.author {
            let author = crate::normalize::normalize_zh(author);
            filtered.retain(|w| crate::normalize::normalize_zh(&w.author).contains(&author));
        }
        if let Some(ref title) = req.title {
            let title = crate::normalize::normalize_zh(title);
            filtered.retain(|w| crate::normalize::normalize_zh(&w.main_title).contains(&title));
        }

        filtered.truncate(req.limit);

        let works: Vec<WorkInfo> = filtered
            .iter()
            .map(|w| WorkInfo {
                work_id: w.work_id.clone(),
                main_title: w.main_title.clone(),
                author: Some(w.author.clone()),
                period: Some(w.period.clone()),
                canon: Some(w.canon.clone()),
                traditions: w.traditions.clone(),
                passage_count: w.passage_count as usize,
            })
            .collect();

        Ok(WorksResponse {
            schema: "sinorag-works-v1",
            works,
        })
    }

    /// Implement the catalog-index-info tool
    pub async fn catalog_index_info_impl(
        &self,
        _req: crate::tools::requests::CatalogIndexInfoRequest,
    ) -> Result<crate::tools::responses::CatalogIndexInfoResponse> {
        use crate::tools::responses::CatalogIndexInfoResponse;

        let catalog = self.catalog().await?;
        let info = catalog.info_payload();

        Ok(CatalogIndexInfoResponse {
            schema: "sinorag-catalog-index-info-v1",
            info,
        })
    }

    /// Implement the similar tool
    pub async fn similar_impl(
        &self,
        req: crate::tools::requests::SimilarRequest,
    ) -> Result<crate::tools::responses::SimilarResponse> {
        use crate::commands::tfidf::similar_passages_with_index;
        use crate::tools::responses::SimilarResponse;

        validate_passage_input("similar", &req.seed)?;
        let passages = self.passages().await?;
        let tfidf = self.tfidf().await?;
        let doc_table = self.doc_table().await?;

        let similar_passages = similar_passages_with_index(
            &passages,
            &tfidf,
            &req.seed,
            req.limit,
            req.shared_ngram_limit,
            req.shared_phrase_limit,
            req.min_shared_phrase_len,
            &doc_table,
        )
        .await?;

        Ok(SimilarResponse {
            schema: "sinorag-similar-v1",
            seed: req.seed,
            similar_passages,
        })
    }

    /// Implement the frontier tool
    pub async fn frontier_impl(
        &self,
        req: crate::tools::requests::FrontierRequest,
    ) -> Result<crate::tools::responses::FrontierResponse> {
        use crate::commands::frontier;
        use crate::commands::tfidf::similar_passages_with_index;
        use crate::registry;
        use crate::tools::responses::FrontierResponse;

        validate_passage_input("frontier", &req.seed)?;

        let passages = self.passages().await?;
        let tfidf = self.tfidf().await?;
        let doc_table = self.doc_table().await?;
        let registry_path = self.registry_path_or_default();
        if !registry_path.exists() {
            registry::init_registry(&registry_path)?;
        }

        // Get seed passage
        let seed_row = passages.get_passage(&req.seed).await.map_err(|err| {
            if err.to_string().contains("Passage not found") {
                passage_not_found(&req.seed)
            } else {
                err
            }
        })?;

        // Get similar passages
        let mut similar = similar_passages_with_index(
            &passages, &tfidf, &req.seed, req.limit, 12, // shared_ngram_limit
            8,  // shared_phrase_limit
            4,  // min_shared_phrase_len
            &doc_table,
        )
        .await?;

        // Apply optional post-filters to similar_passages
        if req.min_similarity.is_some()
            || !req.scope_canon.is_empty()
            || !req.scope_period.is_empty()
            || req.scope_source_work_id.is_some()
        {
            similar.retain(|row| {
                if let Some(min_sim) = req.min_similarity {
                    let score = row
                        .get("tfidf_cosine")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    if score < min_sim {
                        return false;
                    }
                }
                if !req.scope_canon.is_empty() {
                    let canon = row.get("canon").and_then(|v| v.as_str()).unwrap_or("");
                    if !req.scope_canon.iter().any(|c| c == canon) {
                        return false;
                    }
                }
                if !req.scope_period.is_empty() {
                    let period = row.get("period").and_then(|v| v.as_str()).unwrap_or("");
                    if !req.scope_period.iter().any(|p| p == period) {
                        return false;
                    }
                }
                if let Some(work) = &req.scope_source_work_id {
                    let w = row
                        .get("source_work_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if w != work {
                        return false;
                    }
                }
                true
            });
        }

        // Get phrase frontiers
        let phrase_frontiers =
            frontier::phrase_frontiers(&passages, &seed_row, req.phrase_limit).await?;

        // Get prior work
        let prior_work = if registry_path.exists() {
            registry::prior_work(&registry_path, &req.seed, 10)?
        } else {
            Vec::new()
        };

        // Build active_filters summary for output transparency
        let mut active_filters = serde_json::Map::new();
        if let Some(ms) = req.min_similarity {
            active_filters.insert("min_similarity".into(), ms.into());
        }
        if !req.scope_canon.is_empty() {
            active_filters.insert("scope_canon".into(), req.scope_canon.clone().into());
        }
        if !req.scope_period.is_empty() {
            active_filters.insert("scope_period".into(), req.scope_period.clone().into());
        }
        if let Some(w) = &req.scope_source_work_id {
            active_filters.insert("scope_source_work_id".into(), w.clone().into());
        }

        let payload = serde_json::json!({
            "schema": "readzen-sinorag-frontier-v1",
            "seed_passage_id": req.seed,
            "seed": seed_row,
            "active_filters": active_filters,
            "similar_passages": similar,
            "phrase_frontiers": phrase_frontiers,
            "facet_summary": frontier::facet_summary(&similar),
            "next_seed_candidates": frontier::next_seed_candidates(&similar),
            "prior_work": prior_work,
        });

        Ok(FrontierResponse {
            schema: "sinorag-frontier-v1",
            seed_passage_id: req.seed,
            payload,
        })
    }

    /// Implement the first-attestation tool
    pub async fn first_attestation_impl(
        &self,
        req: crate::tools::requests::FirstAttestationRequest,
    ) -> Result<crate::tools::responses::FirstAttestationResponse> {
        use crate::research_tools::phrase::phrase_rows_with_explicit_doc_table;
        use crate::tools::responses::{FirstAttestationResponse, ScopeInfo, SearchStrategyInfo};

        let passages = self.passages().await?;
        let doc_table = self.doc_table().await?;
        let phrase_index_path = self.optional_phrase_path().await?;

        let candidate_limit = req.max_candidates.max(req.limit);
        let (raw_hits, _) = phrase_rows_with_explicit_doc_table(
            &passages,
            &doc_table,
            phrase_index_path.as_deref(),
            &req.phrase,
            candidate_limit,
            None,
            None,
            None,
        )
        .await?;

        // Apply scope_period and scope_source_work_id post-hoc
        let hits: Vec<serde_json::Value> = raw_hits
            .into_iter()
            .filter(|row| {
                if !req.scope_canon.is_empty() {
                    let c = row.get("canon").and_then(|v| v.as_str()).unwrap_or("");
                    if !req.scope_canon.iter().any(|s| s == c) {
                        return false;
                    }
                }
                if !req.scope_period.is_empty() {
                    let p = row.get("period").and_then(|v| v.as_str()).unwrap_or("");
                    if !req.scope_period.iter().any(|s| s == p) {
                        return false;
                    }
                }
                if let Some(work) = &req.scope_source_work_id {
                    let w = row
                        .get("source_work_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if w != work {
                        return false;
                    }
                }
                true
            })
            .collect();
        let verified = hits.len();

        // Sort by (period_rank, doc_id)
        let mut scored: Vec<(i32, u32, serde_json::Value)> = Vec::with_capacity(hits.len());
        for row in hits {
            let pid = row.get("passage_id").and_then(|v| v.as_str()).unwrap_or("");
            if let Some(did) = doc_table.doc_id(pid) {
                let pr = doc_table
                    .period_ranks
                    .get(did as usize)
                    .copied()
                    .unwrap_or(0);
                scored.push((pr, did, row));
            }
        }
        scored.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let total = scored.len();
        let take = req.limit.min(total);
        let mut iter = scored.into_iter().take(take);
        let first = iter.next().map(|(pr, did, mut row)| {
            if let Some(obj) = row.as_object_mut() {
                obj.insert("period_rank".to_string(), serde_json::json!(pr));
                obj.insert("doc_id".to_string(), serde_json::json!(did));
            }
            row
        });
        let next_earlier: Vec<serde_json::Value> = iter
            .map(|(pr, did, mut row)| {
                if let Some(obj) = row.as_object_mut() {
                    obj.insert("period_rank".to_string(), serde_json::json!(pr));
                    obj.insert("doc_id".to_string(), serde_json::json!(did));
                }
                row
            })
            .collect();

        Ok(FirstAttestationResponse {
            schema: "sinorag-first-attestation-v1",
            phrase: req.phrase,
            first,
            next_earlier,
            scope: ScopeInfo {
                canon: req.scope_canon,
                period: req.scope_period,
                source_work_id: req.scope_source_work_id,
            },
            search_strategy: SearchStrategyInfo {
                used_phrase_index: phrase_index_path.is_some(),
                candidates_verified: verified,
                after_scope_and_sort: total,
                limit: req.limit,
                max_candidates: candidate_limit,
            },
        })
    }

    /// Implement the phrase-history tool
    pub async fn phrase_history_impl(
        &self,
        req: crate::tools::requests::PhraseHistoryRequest,
    ) -> Result<crate::tools::responses::PhraseHistoryResponse> {
        use crate::commands::phrase_history;
        use crate::tools::responses::PhraseHistoryResponse;

        let passages = self.passages().await?;
        let phrase_index_path = self.optional_phrase_path().await?;

        let payload = phrase_history::phrase_history(
            req.phrase,
            &passages,
            req.include_variants,
            req.timeline,
            phrase_index_path,
        )
        .await?;

        Ok(PhraseHistoryResponse {
            schema: "sinorag-phrase-history-v1",
            payload,
        })
    }

    /// Implement the phrase-index-search tool
    pub async fn phrase_index_search_impl(
        &self,
        req: crate::tools::requests::PhraseIndexSearchRequest,
    ) -> Result<crate::tools::responses::PhraseIndexSearchResponse> {
        use crate::phrase_index::PhraseIndex;
        use crate::research_tools::phrase::phrase_rows_with_explicit_doc_table;
        use crate::tools::errors::ToolError;
        use crate::tools::responses::PhraseIndexSearchResponse;

        let passages = self.passages().await?;
        let doc_table = self.doc_table().await?;
        let phrase_index_path = self.optional_phrase_path().await?;

        if phrase_index_path.is_none() {
            return Err(ToolError::MissingPhraseIndex {
                path: self
                    .config
                    .phrase_index
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("data/derived/phrase.index")),
            }
            .into_anyhow());
        }
        let phrase_index_path_ref = phrase_index_path.as_deref().expect("checked above");
        let phrase_index = PhraseIndex::load(phrase_index_path_ref)?;
        let gram_len = phrase_index.gram_len();
        let phrase_len = crate::normalize::PhraseForms::new(&req.phrase).indexed_len();
        if phrase_len < gram_len {
            return Err(ToolError::InvalidArgs(format!(
                "phrase-index-search requires at least {gram_len} normalized characters for this phrase index; got {phrase_len}. Use search for short phrases so it can fall back to Parquet."
            ))
            .into_anyhow());
        }

        let (rows, search_strategy) = phrase_rows_with_explicit_doc_table(
            &passages,
            &doc_table,
            Some(phrase_index_path_ref),
            &req.phrase,
            req.limit,
            None,
            None,
            None,
        )
        .await?;

        Ok(PhraseIndexSearchResponse {
            schema: "sinorag-phrase-index-search-v1",
            phrase: req.phrase,
            returned_count: rows.len(),
            limit: req.limit.max(1),
            search_strategy,
            results: rows,
        })
    }

    /// Implement the seed-pick tool
    pub async fn seed_pick_impl(
        &self,
        req: crate::tools::requests::SeedPickRequest,
    ) -> Result<crate::tools::responses::SeedPickResponse> {
        use crate::datafusion_store::sql_literal;
        use crate::tools::responses::{FilterInfo, SeedPickResponse};

        let passages = self.passages().await?;
        let registry_path = self.resolve_registry_path().ok();

        // Get already worked passage IDs from registry
        let mut already_worked: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        if let Some(ref path) = registry_path {
            if path.exists() {
                if let Ok(con) = rusqlite::Connection::open(path) {
                    if let Ok(mut stmt) = con.prepare(
                        "SELECT DISTINCT seed_passage_id FROM seed_observations WHERE seed_passage_id != ''",
                    ) {
                        if let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(0)) {
                            for row in rows.flatten() {
                                already_worked.insert(row);
                            }
                        }
                    }
                }
            }
        }

        // Build WHERE clauses
        let mut where_clauses = vec!["true".to_string()];
        for t in &req.tradition {
            let t = crate::taxonomy_legend::resolve_tradition(t);
            where_clauses.push(tradition_contains_sql("traditions", t));
        }
        for p in &req.period {
            let p = crate::taxonomy_legend::resolve_period(p);
            where_clauses.push(format!("period = {}", sql_literal(p)));
        }
        if !already_worked.is_empty() {
            let id_list = already_worked
                .iter()
                .map(|pid| sql_literal(pid))
                .collect::<Vec<_>>()
                .join(", ");
            where_clauses.push(format!("passage_id NOT IN ({})", id_list));
        }

        let sql = format!(
            r#"
            SELECT passage_id, source_rel_path, xml_id, heading, from_lb, to_lb,
                   zh_text_raw, canon, canon_name, traditions, period, origin, author, main_title,
                   period_rank
            FROM passages
            WHERE {}
            ORDER BY period_rank ASC, source_rel_path ASC, from_lb ASC
            LIMIT {}
            "#,
            where_clauses.join(" AND "),
            req.limit.max(1)
        );

        let results = passages.query_json(&sql).await?;

        Ok(SeedPickResponse {
            schema: "sinorag-seed-pick-v1",
            limit: req.limit,
            already_worked_count: already_worked.len(),
            filters: FilterInfo {
                tradition: req.tradition,
                period: req.period,
            },
            candidates: results,
        })
    }

    /// Implement the expand-context-adaptive tool
    pub async fn expand_context_adaptive_impl(
        &self,
        req: crate::tools::requests::ExpandContextAdaptiveRequest,
    ) -> Result<crate::tools::responses::ExpandContextAdaptiveResponse> {
        use crate::catalog_index::OutlineNodeKind;
        use crate::tools::responses::{ExpandContextAdaptiveResponse, SearchStrategyInfoAdaptive};

        let catalog = self.catalog().await?;
        let doc_table = self.doc_table().await?;
        let passages = self.passages().await?;

        // doc_id lookup
        validate_passage_input("expand-context-adaptive", &req.passage_id)?;
        let doc_id = doc_table
            .doc_id(&req.passage_id)
            .ok_or_else(|| passage_not_found(&req.passage_id))?;
        let mut node_id = *catalog.doc_parent.get(&doc_id).ok_or_else(|| {
            anyhow::anyhow!("doc_id {} has no catalog node (rebuild catalog?)", doc_id)
        })?;

        let leaf_kind = catalog
            .get_node(node_id)
            .map(|n| format!("{:?}", n.node_kind))
            .unwrap_or_default();
        let mut climbed = 0u32;

        // Climb until the node's cjk_char_count fits the budget or we reach Work
        let mut prev_node_id = node_id;
        loop {
            let node = catalog
                .get_node(node_id)
                .ok_or_else(|| anyhow::anyhow!("bad node_id"))?;
            let fits = (node.cjk_char_count as usize) <= req.max_chars;
            let at_work = matches!(node.node_kind, OutlineNodeKind::Work);
            if fits || at_work {
                break;
            }
            match node.parent_id {
                Some(parent) => {
                    let parent_node = catalog
                        .get_node(parent)
                        .ok_or_else(|| anyhow::anyhow!("bad parent_id"))?;
                    if matches!(
                        parent_node.node_kind,
                        OutlineNodeKind::Canon | OutlineNodeKind::Corpus
                    ) {
                        node_id = prev_node_id;
                        break;
                    }
                    prev_node_id = node_id;
                    node_id = parent;
                    climbed += 1;
                }
                None => break,
            }
        }

        let selected = catalog
            .get_node(node_id)
            .ok_or_else(|| anyhow::anyhow!("no selected node"))?;
        let first = selected
            .first_doc_id
            .ok_or_else(|| anyhow::anyhow!("node has no doc range"))?;
        let last = selected
            .last_doc_id
            .ok_or_else(|| anyhow::anyhow!("node has no doc range"))?;

        // The selected node need not fit the budget. The climb stops at the
        // work, and a document with no divisions (a PDF from document
        // intake) is one node however long it is; the catalog also sizes
        // nodes by CJK characters, so it reports an English document as
        // empty. The budget is therefore kept against the text itself: when
        // the node is over it, the passages around the seed are returned
        // instead of the whole node.
        const WINDOW_REACH: u32 = 64;
        // A node the catalog measured as mostly CJK and within the budget is
        // returned whole, exactly as before.
        let catalog_fits = (selected.cjk_char_count as usize) <= req.max_chars
            && selected.cjk_char_count >= selected.passage_count.saturating_mul(20);
        let whole_node = catalog_fits || last - first <= 2 * WINDOW_REACH;
        let (lo, hi) = if whole_node {
            (first, last)
        } else {
            (
                doc_id.saturating_sub(WINDOW_REACH).max(first),
                doc_id.saturating_add(WINDOW_REACH).min(last),
            )
        };
        let mut wanted: Vec<String> = Vec::with_capacity((hi - lo + 1) as usize);
        for did in lo..=hi {
            if let Some(pid) = doc_table.passage_id(did) {
                wanted.push(pid.to_string());
            }
        }

        let fetched = passages
            .passages_by_ids(
                &wanted,
                "passage_id, main_title, source_work_id, source_rel_path, \
             from_lb, to_lb, period, zh_text_normalized as zh_text",
            )
            .await?;
        let text_len = |row: &serde_json::Value| -> usize {
            row.get("zh_text")
                .and_then(|v| v.as_str())
                .map_or(0, |t| t.chars().count())
        };

        let node_fits = catalog_fits
            || (whole_node && fetched.iter().map(&text_len).sum::<usize>() <= req.max_chars);
        let rows: Vec<serde_json::Value> = if node_fits {
            fetched
        } else {
            // Put the window in reading order, then grow outwards from the
            // seed, one passage after and one before in turn, while the
            // budget allows. The seed is always returned.
            let mut by_id: FxHashMap<String, serde_json::Value> = fetched
                .into_iter()
                .filter_map(|row| {
                    let pid = row.get("passage_id")?.as_str()?.to_string();
                    Some((pid, row))
                })
                .collect();
            let ordered: Vec<serde_json::Value> =
                wanted.iter().filter_map(|pid| by_id.remove(pid)).collect();
            let seed_at = ordered
                .iter()
                .position(|row| {
                    row.get("passage_id").and_then(|v| v.as_str()) == Some(req.passage_id.as_str())
                })
                .ok_or_else(|| {
                    ToolError::PassageNotFound {
                        passage_id: req.passage_id.clone(),
                    }
                    .into_anyhow()
                })?;
            let (mut start, mut end) = (seed_at, seed_at + 1);
            let mut used = text_len(&ordered[seed_at]);
            let (mut grow_back, mut grow_on) = (true, true);
            while grow_back || grow_on {
                if grow_on {
                    match ordered.get(end).map(&text_len) {
                        Some(len) if used + len <= req.max_chars => {
                            used += len;
                            end += 1;
                        }
                        _ => grow_on = false,
                    }
                }
                if grow_back {
                    match start.checked_sub(1).map(|i| text_len(&ordered[i])) {
                        Some(len) if used + len <= req.max_chars => {
                            used += len;
                            start -= 1;
                        }
                        _ => grow_back = false,
                    }
                }
            }
            ordered.into_iter().skip(start).take(end - start).collect()
        };

        let char_count: usize = rows.iter().map(&text_len).sum();
        let truncation = (!node_fits).then(|| {
            format!(
                "{} has {} passages, more than the budget of {} characters holds: only the {} \
                 around the seed are returned. Use source-read with this passage_id to read on \
                 in either direction.",
                if selected.label.is_empty() {
                    "This section"
                } else {
                    selected.label.as_str()
                },
                selected.passage_count,
                req.max_chars,
                rows.len(),
            )
        });

        Ok(ExpandContextAdaptiveResponse {
            schema: "sinorag-expand-context-adaptive-v1",
            seed_passage_id: req.passage_id,
            selected_node_id: selected.node_id,
            selected_node_kind: format!("{:?}", selected.node_kind),
            selected_label: selected.label.clone(),
            heading_path: vec![selected.heading_path.clone()],
            work_id: Some(selected.work_id.clone()),
            passage_count: rows.len(),
            char_count,
            truncated: truncation.is_some(),
            truncation,
            passages: rows,
            search_strategy: SearchStrategyInfoAdaptive {
                budget: req.max_chars,
                climbed_levels: climbed,
                leaf_kind,
                mode: "auto".to_string(),
            },
        })
    }

    /// Implement the trace-term-usage tool
    pub async fn trace_term_usage_impl(
        &self,
        req: crate::tools::requests::TraceTermUsageRequest,
    ) -> Result<crate::tools::responses::TraceTermUsageResponse> {
        use crate::research_tools::phrase::phrase_rows_with_explicit_doc_table;
        use crate::tools::responses::{
            TermUsageGroup, TermUsageSearchStrategy, TraceTermUsageResponse,
        };

        let passages = self.passages().await?;
        let doc_table = self.doc_table().await?;
        let phrase_index_path = self.optional_phrase_path().await?;

        let key_field = match req.group_by.as_str() {
            "period" => "period",
            "canon" => "canon",
            "author" => "author",
            "work" => "source_work_id",
            _ => "period",
        };

        let candidate_limit = req.max_candidates.max(req.limit_total);
        let (hits, _) = phrase_rows_with_explicit_doc_table(
            &passages,
            &doc_table,
            phrase_index_path.as_deref(),
            &req.phrase,
            candidate_limit,
            None,
            None,
            None,
        )
        .await?;
        let total_hits = hits.len();

        // Group by the chosen field
        use std::collections::BTreeMap;
        struct GroupAcc {
            hit_count: u32,
            work_ids: std::collections::BTreeSet<String>,
            reps: Vec<(i32, u32, serde_json::Value)>,
        }
        impl Default for GroupAcc {
            fn default() -> Self {
                Self {
                    hit_count: 0,
                    work_ids: std::collections::BTreeSet::new(),
                    reps: Vec::new(),
                }
            }
        }

        let mut groups: BTreeMap<String, GroupAcc> = BTreeMap::new();
        for row in hits {
            let key = row
                .get(key_field)
                .and_then(|v| v.as_str())
                .unwrap_or("(unknown)")
                .to_string();
            let acc = groups.entry(key).or_insert_with(GroupAcc::default);
            acc.hit_count += 1;
            if let Some(wid) = row.get("source_work_id").and_then(|v| v.as_str()) {
                acc.work_ids.insert(wid.to_string());
            }
            let pid = row.get("passage_id").and_then(|v| v.as_str()).unwrap_or("");
            let did = doc_table.doc_id(pid).unwrap_or(u32::MAX);
            let pr = if did != u32::MAX {
                doc_table
                    .period_ranks
                    .get(did as usize)
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            };
            acc.reps.push((pr, did, row));
        }

        let mut out_groups: Vec<TermUsageGroup> = Vec::with_capacity(groups.len());
        for (key, mut acc) in groups {
            acc.reps.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            let reps: Vec<serde_json::Value> = acc
                .reps
                .into_iter()
                .take(req.limit_per_group)
                .map(|(_, _, r)| r)
                .collect();
            let mut top_works: Vec<String> = acc.work_ids.into_iter().collect();
            top_works.sort();
            top_works.truncate(req.limit_per_group);
            out_groups.push(TermUsageGroup {
                key,
                hit_count: acc.hit_count,
                work_count: top_works.len(),
                top_works,
                representative_passages: reps,
            });
        }

        Ok(TraceTermUsageResponse {
            schema: "sinorag-term-usage-trace-v1",
            phrase: req.phrase,
            group_by: req.group_by,
            groups: out_groups,
            search_strategy: TermUsageSearchStrategy {
                used_phrase_index: phrase_index_path.is_some(),
                total_hits,
                limit_total: req.limit_total,
                limit_per_group: req.limit_per_group,
                max_candidates: candidate_limit,
            },
        })
    }

    /// Implement the query-expand-terms tool
    pub async fn query_expand_terms_impl(
        &self,
        req: crate::tools::requests::QueryExpandTermsRequest,
    ) -> Result<crate::tools::responses::QueryExpandTermsResponse> {
        use crate::templates::variants::VariantTables;
        use crate::tools::responses::{
            ExpandTermsBySource, ExpandTermsSearchStrategy, QueryExpandTermsResponse,
        };

        let tables = VariantTables::load();

        let mut variants_bucket = std::collections::BTreeSet::<String>::new();
        let mut orthographic_bucket = std::collections::BTreeSet::<String>::new();
        let mut persons_bucket = std::collections::BTreeSet::<String>::new();

        let expand_variants = matches!(req.mode.as_str(), "variants" | "all");
        let expand_orthographic = matches!(req.mode.as_str(), "orthographic" | "all");
        let expand_persons = matches!(req.mode.as_str(), "persons" | "all");

        if expand_variants {
            for v in tables.term_variants(&req.phrase) {
                if v != req.phrase {
                    variants_bucket.insert(v);
                }
            }
        }
        if expand_orthographic {
            for v in tables.orthographic_flips(&req.phrase, req.max * 2) {
                orthographic_bucket.insert(v);
            }
            // Also flip every term in the variants bucket so we cover cross-Han variants
            let cur: Vec<String> = variants_bucket.iter().cloned().collect();
            for v in cur {
                for f in tables.orthographic_flips(&v, req.max) {
                    if !variants_bucket.contains(&f) {
                        orthographic_bucket.insert(f);
                    }
                }
            }
        }
        if expand_persons {
            for a in &req.person_aliases {
                if !a.is_empty() && a != &req.phrase {
                    persons_bucket.insert(a.clone());
                }
            }
        }

        // Combined view (deduped, capped)
        let mut combined: Vec<String> = Vec::new();
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        seen.insert(req.phrase.clone());
        for v in variants_bucket
            .iter()
            .chain(orthographic_bucket.iter())
            .chain(persons_bucket.iter())
        {
            if seen.insert(v.clone()) {
                combined.push(v.clone());
                if combined.len() >= req.max {
                    break;
                }
            }
        }

        // Detect language
        let mut has_han = false;
        let mut has_latin = false;
        for ch in req.phrase.chars() {
            if (0x4E00..=0x9FFF).contains(&(ch as u32))
                || (0x3400..=0x4DBF).contains(&(ch as u32))
                || (0xF900..=0xFAFF).contains(&(ch as u32))
            {
                has_han = true;
            }
            if ch.is_ascii_alphabetic() {
                has_latin = true;
            }
        }
        let lang_guess = match (has_han, has_latin) {
            (true, false) => "zh",
            (false, true) => "en",
            (true, true) => "mixed",
            _ => "unknown",
        };

        Ok(QueryExpandTermsResponse {
            schema: "sinorag-query-expand-terms-v1",
            input: req.phrase,
            expanded: combined,
            by_source: ExpandTermsBySource {
                variants: variants_bucket.into_iter().collect::<Vec<_>>(),
                orthographic: orthographic_bucket.into_iter().collect::<Vec<_>>(),
                persons: persons_bucket.into_iter().collect::<Vec<_>>(),
            },
            search_strategy: ExpandTermsSearchStrategy {
                mode: req.mode,
                max: req.max,
                input_lang_guess: lang_guess.to_string(),
            },
        })
    }

    /// Implement the compare-usage tool
    pub async fn compare_usage_impl(
        &self,
        req: crate::tools::requests::CompareUsageRequest,
    ) -> Result<crate::tools::responses::CompareUsageResponse> {
        use crate::research_tools::stats::log_odds_distinctive_terms;
        use crate::tools::responses::{
            CompareUsageResponse, CompareUsageScope, CompareUsageSearchStrategy, CompareUsageTerm,
        };

        let catalog = self.catalog().await?;
        let doc_table = self.doc_table().await?;
        let passages = self.passages().await?;

        // Resolve doc ranges for scopes
        let range_a = self.resolve_doc_range(
            &catalog,
            req.scope_a_node_id,
            req.scope_a_work_id.as_deref(),
        )?;
        let range_b = self.resolve_doc_range(
            &catalog,
            req.scope_b_node_id,
            req.scope_b_work_id.as_deref(),
        )?;

        let (a_terms, a_term_display, a_passage_count) = self
            .collect_scope_terms(
                &passages,
                &doc_table,
                range_a,
                req.scope_a_canon.as_deref(),
                req.scope_a_period.as_deref(),
                req.limit_passages,
                req.gram_len,
            )
            .await?;

        let (b_terms, b_term_display, b_passage_count) = self
            .collect_scope_terms(
                &passages,
                &doc_table,
                range_b,
                req.scope_b_canon.as_deref(),
                req.scope_b_period.as_deref(),
                req.limit_passages,
                req.gram_len,
            )
            .await?;

        let (a_top, b_top) = log_odds_distinctive_terms(&a_terms, &b_terms, req.limit_terms);

        let distinctive_to_a: Vec<CompareUsageTerm> = a_top
            .iter()
            .map(|t| CompareUsageTerm {
                term: a_term_display
                    .get(&t.term_hash)
                    .or_else(|| b_term_display.get(&t.term_hash))
                    .cloned(),
                term_hash: t.term_hash,
                score: t.score,
                a_count: t.a_count,
                b_count: t.b_count,
            })
            .collect();

        let distinctive_to_b: Vec<CompareUsageTerm> = b_top
            .iter()
            .map(|t| CompareUsageTerm {
                term: b_term_display
                    .get(&t.term_hash)
                    .or_else(|| a_term_display.get(&t.term_hash))
                    .cloned(),
                term_hash: t.term_hash,
                score: t.score,
                a_count: t.a_count,
                b_count: t.b_count,
            })
            .collect();

        Ok(CompareUsageResponse {
            schema: "sinorag-compare-usage-v1",
            scope_a: CompareUsageScope {
                node_id: req.scope_a_node_id,
                work_id: req.scope_a_work_id,
                canon: req.scope_a_canon,
                period: req.scope_a_period,
                passage_count: a_passage_count,
            },
            scope_b: CompareUsageScope {
                node_id: req.scope_b_node_id,
                work_id: req.scope_b_work_id,
                canon: req.scope_b_canon,
                period: req.scope_b_period,
                passage_count: b_passage_count,
            },
            distinctive_to_a,
            distinctive_to_b,
            search_strategy: CompareUsageSearchStrategy {
                gram_len: req.gram_len,
                limit_passages: req.limit_passages,
                limit_terms: req.limit_terms,
            },
        })
    }

    /// Implement the collocation-search tool
    pub async fn collocation_search_impl(
        &self,
        req: crate::tools::requests::CollocationSearchRequest,
    ) -> Result<crate::tools::responses::CollocationSearchResponse> {
        use crate::research_tools::phrase::phrase_rows_with_explicit_doc_table;
        use crate::research_tools::stats::score_collocates;
        use crate::tools::responses::{
            CollocateTerm, CollocationSearchResponse, CollocationSearchStrategy,
        };

        let doc_table = self.doc_table().await?;
        let passages = self.passages().await?;
        let phrase_index_path = self.optional_phrase_path().await?;
        let catalog = self.catalog().await.ok();
        let canon = expand_optional_filter(req.scope_canon.as_deref());
        let period = expand_period_filter(req.scope_period.as_deref());
        let doc_range = if let Some(catalog) = catalog.as_deref() {
            self.resolve_doc_range(
                catalog,
                req.scope_node_id,
                req.scope_source_work_id.as_deref(),
            )?
        } else {
            None
        };
        let canon_for_index = if canon.is_empty() {
            None
        } else {
            Some(canon.as_slice())
        };
        let period_for_index = if period.len() == 1 {
            Some(period[0].as_str())
        } else {
            None
        };

        let (hits, phrase_strategy) = phrase_rows_with_explicit_doc_table(
            &passages,
            &doc_table,
            phrase_index_path.as_deref(),
            &req.phrase,
            req.limit_total,
            doc_range,
            canon_for_index,
            period_for_index,
        )
        .await?;

        let phrase_forms = crate::normalize::PhraseForms::new(&req.phrase);

        // Collect n-gram hashes from the window around each phrase occurrence
        let mut near_counts: FxHashMap<u64, u32> = FxHashMap::default();
        let mut background_counts: FxHashMap<u64, u32> = FxHashMap::default();
        let mut term_display: FxHashMap<u64, String> = FxHashMap::default();
        let mut near_total = 0u32;
        let mut bg_total = 0u32;

        for row in &hits {
            let norm = row
                .get("zh_text_normalized")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Background: all n-grams in the passage
            bg_total += count_unique_ngrams_with_terms(
                norm,
                req.gram_len,
                &mut background_counts,
                &mut term_display,
            );

            // Near: n-grams within the window around each occurrence
            let normalized_phrase = phrase_forms.form_for(norm);
            if normalized_phrase.trim().is_empty() {
                continue;
            }
            let mut search_start = 0usize;
            while let Some(rel_pos) = norm[search_start..].find(normalized_phrase) {
                let byte_pos = search_start + rel_pos;
                let char_pos = norm[..byte_pos].chars().count();
                let phrase_chars = normalized_phrase.chars().count();
                let window_start = char_pos.saturating_sub(req.window_chars);
                let window_end =
                    (char_pos + phrase_chars + req.window_chars).min(norm.chars().count());

                let window_text: String = if crate::normalize::has_word_layout(norm) {
                    // Whole words on either side of the occurrence. The
                    // phrase itself is left out (it would always be its own
                    // strongest collocate), and so is a word the window
                    // boundary cuts through.
                    let before: String = norm
                        .chars()
                        .skip(window_start)
                        .take(char_pos - window_start)
                        .collect();
                    let after: String = norm
                        .chars()
                        .skip(char_pos + phrase_chars)
                        .take(window_end.saturating_sub(char_pos + phrase_chars))
                        .collect();
                    let before = match before.find(' ') {
                        Some(cut) if window_start > 0 => &before[cut..],
                        _ => before.as_str(),
                    };
                    let after = match after.rfind(' ') {
                        Some(cut) if window_end < norm.chars().count() => &after[..cut],
                        _ => after.as_str(),
                    };
                    // "|" is not a word, so no term spans the gap.
                    format!("{before} | {after}")
                } else {
                    norm.chars()
                        .skip(window_start)
                        .take(window_end - window_start)
                        .collect()
                };
                near_total += count_unique_ngrams_with_terms(
                    &window_text,
                    req.gram_len,
                    &mut near_counts,
                    &mut term_display,
                );
                search_start = byte_pos + normalized_phrase.len();
            }
        }

        let collocates = score_collocates(&near_counts, &background_counts, req.limit_collocates);

        let collocates_vec: Vec<CollocateTerm> = collocates
            .iter()
            .map(|c| CollocateTerm {
                term: term_display.get(&c.term_hash).cloned(),
                term_hash: c.term_hash,
                score: c.score,
                near_count: c.near_count,
                background_count: c.background_count,
            })
            .collect();

        Ok(CollocationSearchResponse {
            schema: "sinorag-collocation-search-v1",
            phrase: req.phrase,
            window_chars: req.window_chars,
            gram_len: req.gram_len,
            total_passages: hits.len(),
            near_ngram_count: near_total,
            background_ngram_count: bg_total,
            collocates: collocates_vec,
            search_strategy: CollocationSearchStrategy {
                phrase: phrase_strategy,
                limit_total: req.limit_total,
                limit_collocates: req.limit_collocates,
            },
        })
    }

    /// Implement the pair-appearance tool
    pub async fn pair_appearance_impl(
        &self,
        req: crate::tools::requests::PairAppearanceRequest,
    ) -> Result<crate::tools::responses::PairAppearanceResponse> {
        use crate::tools::responses::{
            PairAppearanceHit, PairAppearanceNegativeSummary, PairAppearanceResponse,
            PairAppearanceSearchStrategy,
        };

        let unit = match req.unit.as_str() {
            "passage" | "window" | "sentence" => req.unit.clone(),
            other => {
                return Err(anyhow::anyhow!(
                    "unsupported pair-appearance unit '{other}'; supported units are passage, window, sentence"
                ));
            }
        };

        let doc_table = self.doc_table().await?;
        let passages = self.passages().await?;
        let phrase_index_path = self.optional_phrase_path().await?;
        let catalog = self.catalog().await.ok();
        let canon = expand_optional_filter(req.scope_canon.as_deref());
        let period = expand_period_filter(req.scope_period.as_deref());
        let doc_range = if let Some(catalog) = catalog.as_deref() {
            self.resolve_doc_range(
                catalog,
                req.scope_node_id,
                req.scope_source_work_id.as_deref(),
            )?
        } else {
            None
        };
        let canon_for_index = if canon.is_empty() {
            None
        } else {
            Some(canon.as_slice())
        };
        let period_for_index = if period.len() == 1 {
            Some(period[0].as_str())
        } else {
            None
        };

        let term1_variants = pair_term_variants(&req.term1, req.allow_variants);
        let term2_variants = pair_term_variants(&req.term2, req.allow_variants);
        let output_limit = req.limit.max(1);
        let candidate_limit = req.max_candidates_per_term.max(output_limit);
        let retrieval_budget =
            crate::retrieval::RetrievalBudget::new(output_limit, candidate_limit);
        let scope = pair_scope_spec(&req);

        let (term1_rows, term1_strategy) = self
            .pair_candidate_rows(
                &passages,
                &doc_table,
                phrase_index_path.as_deref(),
                &term1_variants,
                candidate_limit,
                doc_range,
                canon_for_index,
                period_for_index,
                req.scope_source_work_id.as_deref(),
            )
            .await?;
        let (term2_rows, term2_strategy) = self
            .pair_candidate_rows(
                &passages,
                &doc_table,
                phrase_index_path.as_deref(),
                &term2_variants,
                candidate_limit,
                doc_range,
                canon_for_index,
                period_for_index,
                req.scope_source_work_id.as_deref(),
            )
            .await?;

        let term2_by_pid = term2_rows
            .iter()
            .filter_map(|row| {
                row.get("passage_id")
                    .and_then(|v| v.as_str())
                    .map(|pid| (pid.to_string(), row))
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        let intersection_candidate_count = term1_rows
            .iter()
            .filter_map(|row| row.get("passage_id").and_then(|v| v.as_str()))
            .filter(|pid| term2_by_pid.contains_key(*pid))
            .count();

        let mut hits = Vec::new();
        for row in &term1_rows {
            let Some(pid) = row.get("passage_id").and_then(|v| v.as_str()) else {
                continue;
            };
            if !term2_by_pid.contains_key(pid) {
                continue;
            }
            let norm = row
                .get("zh_text_normalized")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let term1_offsets = find_offsets_for_terms(norm, &term1_variants);
            let term2_offsets = find_offsets_for_terms(norm, &term2_variants);
            if term1_offsets.is_empty() || term2_offsets.is_empty() {
                continue;
            }
            let distance = min_pair_distance(&term1_offsets, &term2_offsets, req.ordered);
            let keep = match unit.as_str() {
                "passage" => distance.is_some(),
                "window" => distance.map(|d| d <= req.window_chars).unwrap_or(false),
                "sentence" => has_sentence_pair(norm, &term1_offsets, &term2_offsets, req.ordered),
                _ => false,
            };
            if !keep {
                continue;
            }
            let quote = if req.include_snippets {
                pair_snippet(
                    norm,
                    &term1_offsets,
                    &term2_offsets,
                    req.window_chars.max(40),
                )
            } else {
                String::new()
            };
            hits.push(PairAppearanceHit {
                passage_id: pid.to_string(),
                source_work_id: opt_str(row, "source_work_id"),
                main_title: opt_str(row, "main_title"),
                heading: opt_str(row, "heading"),
                distance_chars: distance,
                term1_offsets,
                term2_offsets,
                zh_quote: quote,
            });
        }

        let pair_hit_count = hits.len();
        hits.truncate(output_limit);

        let negative_summary = if req.include_negative_summary {
            let term2_ids = term2_rows
                .iter()
                .filter_map(|row| row.get("passage_id").and_then(|v| v.as_str()))
                .collect::<std::collections::BTreeSet<_>>();
            let term1_ids = term1_rows
                .iter()
                .filter_map(|row| row.get("passage_id").and_then(|v| v.as_str()))
                .collect::<std::collections::BTreeSet<_>>();
            let term1_only = term1_ids
                .iter()
                .filter(|pid| !term2_ids.contains(**pid))
                .copied()
                .collect::<Vec<_>>();
            let term2_only = term2_ids
                .iter()
                .filter(|pid| !term1_ids.contains(**pid))
                .copied()
                .collect::<Vec<_>>();
            Some(PairAppearanceNegativeSummary {
                term1_only_count: term1_only.len(),
                term2_only_count: term2_only.len(),
                sample_term1_only_passage_ids: term1_only
                    .iter()
                    .take(10)
                    .map(|pid| (*pid).to_string())
                    .collect(),
                sample_term2_only_passage_ids: term2_only
                    .iter()
                    .take(10)
                    .map(|pid| (*pid).to_string())
                    .collect(),
            })
        } else {
            None
        };

        let mut trace = crate::retrieval::RetrievalTraceBuilder::new();
        trace.push(
            crate::retrieval::RetrievalStageReport::new(
                "term1_candidates",
                term1_rows.len(),
                term1_rows.len(),
            )
            .with_details(serde_json::json!({
                "term": req.term1.clone(),
                "variant_count": term1_variants.len(),
                "strategy": term1_strategy.clone(),
            })),
        );
        trace.push(
            crate::retrieval::RetrievalStageReport::new(
                "term2_candidates",
                term2_rows.len(),
                term2_rows.len(),
            )
            .with_details(serde_json::json!({
                "term": req.term2.clone(),
                "variant_count": term2_variants.len(),
                "strategy": term2_strategy.clone(),
            })),
        );
        trace.push(
            crate::retrieval::RetrievalStageReport::new(
                "pair_verify",
                intersection_candidate_count,
                pair_hit_count,
            )
            .with_verified_count(pair_hit_count)
            .with_details(serde_json::json!({
                "unit": unit.clone(),
                "window_chars": req.window_chars,
                "ordered": req.ordered,
            })),
        );
        trace.push(crate::retrieval::RetrievalStageReport::new(
            "return",
            pair_hit_count,
            hits.len(),
        ));

        Ok(PairAppearanceResponse {
            schema: "sinorag-pair-appearance-v1",
            candidate_budget: retrieval_budget,
            scope,
            term1: req.term1,
            term2: req.term2,
            unit: unit.clone(),
            window_chars: req.window_chars,
            ordered: req.ordered,
            total_term1_hits: term1_rows.len(),
            total_term2_hits: term2_rows.len(),
            pair_hit_count,
            hits,
            negative_summary,
            stages: trace.finish(),
            search_strategy: PairAppearanceSearchStrategy {
                term1: term1_strategy,
                term2: term2_strategy,
                unit,
                supported_units: vec!["passage", "window", "sentence"],
                used_variant_expansion: req.allow_variants,
                max_candidates_per_term: candidate_limit,
            },
        })
    }

    /// Implement the outline-search tool
    pub async fn outline_search_impl(
        &self,
        req: crate::tools::requests::OutlineSearchRequest,
    ) -> Result<crate::tools::responses::OutlineSearchResponse> {
        use crate::research_tools::phrase::phrase_rows_with_explicit_doc_table;
        use crate::research_tools::scopes::{group_hits_by_outline_node, OutlineSearchLevel};
        use crate::tools::responses::{
            OutlineSearchGroup, OutlineSearchResponse, OutlineSearchStrategy,
        };

        let passages = self.passages().await?;
        let doc_table = self.doc_table().await;
        let catalog = self.catalog().await;
        if (doc_table.is_err() || catalog.is_err())
            && req.node_id.is_none()
            && req.work_id.is_none()
        {
            let search = self
                .search_with(crate::tools::requests::SearchRequest {
                    phrase: req.phrase.clone(),
                    limit: req.limit_total,
                    mode: "hits".to_string(),
                    depth: "exact".to_string(),
                    // Unused in `hits` mode; outline-search has its own values.
                    group_by: "work".to_string(),
                    include_variants: false,
                    limit_per_group: req.limit_per_group,
                    brief: true,
                    canon: None,
                    source_work_id: None,
                    tradition: None,
                    period: None,
                    origin: None,
                    author: None,
                    title: None,
                    heading_path_prefix: None,
                }, false)
                .await?;
            let mut counts = std::collections::BTreeMap::<String, u32>::new();
            for hit in &search.hits {
                let key = match req.group_by.as_str() {
                    "division" | "passage" => hit
                        .heading
                        .as_deref()
                        .filter(|s| !s.is_empty())
                        .unwrap_or("(unknown)"),
                    _ => hit
                        .source_work_id
                        .as_deref()
                        .filter(|s| !s.is_empty())
                        .unwrap_or("(unknown)"),
                };
                *counts.entry(key.to_string()).or_insert(0) += 1;
            }
            let mut sorted_groups: Vec<(String, u32)> = counts.into_iter().collect();
            sorted_groups.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let groups = sorted_groups
                .iter()
                .take(req.limit_per_group)
                .enumerate()
                .map(|(idx, (label, count))| OutlineSearchGroup {
                    node_id: idx as u32,
                    label: label.clone(),
                    heading_path: label.clone(),
                    node_kind: "MetadataFallback".to_string(),
                    hit_count: *count,
                })
                .collect();
            return Ok(OutlineSearchResponse {
                schema: "sinorag-outline-search-v1",
                phrase: req.phrase,
                start_node_id: 0,
                start_label: "corpus".to_string(),
                group_by: req.group_by,
                total_hits: search.hits.len(),
                group_count: sorted_groups.len(),
                groups,
                search_strategy: OutlineSearchStrategy {
                    phrase: serde_json::json!({
                        "used_phrase_index": false,
                        "scope_scan": "metadata_fallback",
                        "search": search.search_strategy,
                    }),
                    limit_total: req.limit_total,
                    limit_per_group: req.limit_per_group,
                },
            });
        }

        let doc_table = doc_table?;
        let catalog = catalog?;
        let phrase_index_path = self.optional_phrase_path().await?;

        let target = match req.group_by.as_str() {
            "division" => OutlineSearchLevel::Division,
            "work" => OutlineSearchLevel::Work,
            "passage" => OutlineSearchLevel::PassageRange,
            other => {
                return Err(anyhow::anyhow!(
                    "unknown group_by `{other}`; expected division|work|passage"
                ))
            }
        };

        // Resolve starting node: explicit node_id or work_id → root_node.
        // If no scope is supplied, search corpus-wide and group all hits.
        let (start_node, start_label, doc_range) = if let Some(nid) = req.node_id {
            let node = catalog
                .get_node(nid)
                .ok_or_else(|| anyhow::anyhow!("unknown node_id: {nid}"))?;
            let range = match (node.first_doc_id, node.last_doc_id) {
                (Some(l), Some(h)) => Some((l, h)),
                _ => return Err(anyhow::anyhow!("node {nid} has no doc range")),
            };
            (nid, node.label.clone(), range)
        } else if let Some(wid) = &req.work_id {
            let work = catalog
                .get_work(wid)
                .ok_or_else(|| anyhow::anyhow!("unknown work_id: {wid}"))?;
            let root = catalog
                .get_node(work.root_node)
                .ok_or_else(|| anyhow::anyhow!("work root node missing"))?;
            let range = match (root.first_doc_id, root.last_doc_id) {
                (Some(l), Some(h)) => Some((l, h)),
                _ => return Err(anyhow::anyhow!("work {wid} has no doc range")),
            };
            (work.root_node, root.label.clone(), range)
        } else {
            (0, "corpus".to_string(), None)
        };

        let (hits, phrase_strategy) = phrase_rows_with_explicit_doc_table(
            &passages,
            &doc_table,
            phrase_index_path.as_deref(),
            &req.phrase,
            req.limit_total,
            doc_range,
            None,
            None,
        )
        .await?;

        let filtered_doc_ids: Vec<u32> = hits
            .iter()
            .filter_map(|row| {
                let pid = row.get("passage_id").and_then(|v| v.as_str())?;
                doc_table.doc_id(pid)
            })
            .collect();

        let total_hits = filtered_doc_ids.len();

        // Group by the target outline level
        let group_counts = group_hits_by_outline_node(&catalog, &filtered_doc_ids, target);

        let mut sorted_groups: Vec<(u32, u32)> = group_counts.into_iter().collect();
        sorted_groups.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let groups: Vec<OutlineSearchGroup> = sorted_groups
            .iter()
            .take(req.limit_per_group)
            .map(|(node_id, count)| {
                let node = catalog.get_node(*node_id);
                OutlineSearchGroup {
                    node_id: *node_id,
                    label: node.map(|n| n.label.clone()).unwrap_or_default(),
                    heading_path: node.map(|n| n.heading_path.clone()).unwrap_or_default(),
                    node_kind: node
                        .map(|n| format!("{:?}", &n.node_kind))
                        .unwrap_or_default(),
                    hit_count: *count,
                }
            })
            .collect();

        Ok(OutlineSearchResponse {
            schema: "sinorag-outline-search-v1",
            phrase: req.phrase,
            start_node_id: start_node,
            start_label,
            group_by: req.group_by,
            total_hits,
            group_count: sorted_groups.len(),
            groups,
            search_strategy: OutlineSearchStrategy {
                phrase: phrase_strategy,
                limit_total: req.limit_total,
                limit_per_group: req.limit_per_group,
            },
        })
    }

    /// Implement the cluster-hits tool
    pub async fn cluster_hits_impl(
        &self,
        req: crate::tools::requests::ClusterHitsRequest,
    ) -> Result<crate::tools::responses::ClusterHitsResponse> {
        use crate::research_tools::phrase::phrase_rows_with_explicit_doc_table;
        use crate::research_tools::scopes::{group_hits_by_outline_node, OutlineSearchLevel};
        use crate::tools::responses::{
            ClusterHitsCluster, ClusterHitsResponse, ClusterHitsSearchStrategy,
        };

        let doc_table = self.doc_table().await?;
        let catalog = self.catalog().await?;
        let passages = self.passages().await?;
        let phrase_index_path = self.optional_phrase_path().await?;

        let target = match req.cluster_by.as_str() {
            "work" => OutlineSearchLevel::Work,
            "division" => OutlineSearchLevel::Division,
            other => {
                return Err(anyhow::anyhow!(
                    "unknown cluster_by `{other}`; expected work|division"
                ))
            }
        };

        let candidate_limit = req.max_candidates.max(req.limit_total);
        let (hits, phrase_strategy) = phrase_rows_with_explicit_doc_table(
            &passages,
            &doc_table,
            phrase_index_path.as_deref(),
            &req.phrase,
            candidate_limit,
            None,
            None,
            None,
        )
        .await?;

        // Collect (doc_id, row) pairs
        let mut doc_rows: Vec<(u32, serde_json::Value)> = Vec::with_capacity(hits.len());
        for row in hits {
            let pid = row.get("passage_id").and_then(|v| v.as_str()).unwrap_or("");
            if let Some(did) = doc_table.doc_id(pid) {
                doc_rows.push((did, row));
            }
        }

        let doc_ids: Vec<u32> = doc_rows.iter().map(|(d, _)| *d).collect();
        let group_counts = group_hits_by_outline_node(&catalog, &doc_ids, target);

        // Sort groups by hit_count descending
        let mut sorted_groups: Vec<(u32, u32)> = group_counts.into_iter().collect();
        sorted_groups.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let clusters: Vec<ClusterHitsCluster> = sorted_groups
            .iter()
            .take(req.limit_per_cluster)
            .map(|(node_id, count)| {
                let node = catalog.get_node(*node_id);
                let node_doc_range = node.and_then(|n| n.first_doc_id.zip(n.last_doc_id));

                // Pick top representative passages within this cluster
                let mut reps: Vec<serde_json::Value> = doc_rows
                    .iter()
                    .filter(|(did, _)| {
                        if let Some((lo, hi)) = node_doc_range {
                            *did >= lo && *did <= hi
                        } else {
                            false
                        }
                    })
                    .take(3)
                    .map(|(did, row)| {
                        let mut r = row.clone();
                        if let Some(obj) = r.as_object_mut() {
                            obj.insert("doc_id".to_string(), serde_json::json!(*did));
                        }
                        r
                    })
                    .collect();
                reps.truncate(3);

                ClusterHitsCluster {
                    node_id: *node_id,
                    label: node.map(|n| n.label.clone()).unwrap_or_default(),
                    heading_path: node.map(|n| n.heading_path.clone()).unwrap_or_default(),
                    node_kind: node
                        .map(|n| format!("{:?}", &n.node_kind))
                        .unwrap_or_default(),
                    hit_count: *count,
                    representative_passages: reps,
                }
            })
            .collect();

        Ok(ClusterHitsResponse {
            schema: "sinorag-cluster-hits-v1",
            phrase: req.phrase,
            cluster_by: req.cluster_by,
            total_hits: doc_rows.len(),
            cluster_count: sorted_groups.len(),
            clusters,
            search_strategy: ClusterHitsSearchStrategy {
                phrase: phrase_strategy,
                limit_total: req.limit_total,
                limit_per_cluster: req.limit_per_cluster,
                max_candidates: candidate_limit,
            },
        })
    }

    /// Implement the absence-check tool
    pub async fn absence_check_impl(
        &self,
        req: crate::tools::requests::AbsenceCheckRequest,
    ) -> Result<crate::tools::responses::AbsenceCheckResponse> {
        use crate::research_tools::phrase::phrase_rows_with_explicit_doc_table;
        use crate::tools::responses::{
            AbsenceCheckResponse, AbsenceCheckScope, AbsenceCheckSearchStrategy,
        };

        let doc_table = self.doc_table().await?;
        let catalog = self.catalog().await?;
        let passages = self.passages().await?;
        let phrase_index_path = self.optional_phrase_path().await?;

        // Determine the doc range for the scope
        let doc_range: Option<(u32, u32)> = if let Some(nid) = req.scope_node_id {
            let node = catalog
                .get_node(nid)
                .ok_or_else(|| anyhow::anyhow!("unknown node_id: {nid}"))?;
            node.first_doc_id.zip(node.last_doc_id)
        } else if let Some(wid) = &req.scope_work_id {
            let work = catalog
                .get_work(wid)
                .ok_or_else(|| anyhow::anyhow!("unknown work_id: {wid}"))?;
            let root = catalog
                .get_node(work.root_node)
                .ok_or_else(|| anyhow::anyhow!("work root node missing"))?;
            root.first_doc_id.zip(root.last_doc_id)
        } else {
            None
        };

        let candidate_limit = req.max_candidates.max(req.limit);
        let (scoped_hits, phrase_strategy) = phrase_rows_with_explicit_doc_table(
            &passages,
            &doc_table,
            phrase_index_path.as_deref(),
            &req.phrase,
            candidate_limit,
            doc_range,
            if req.scope_canon.is_empty() {
                None
            } else {
                Some(req.scope_canon.as_slice())
            },
            req.scope_period.as_deref(),
        )
        .await?;

        let found = !scoped_hits.is_empty();
        let hit_count = scoped_hits.len();

        Ok(AbsenceCheckResponse {
            schema: "sinorag-absence-check-v1",
            phrase: req.phrase,
            scope: AbsenceCheckScope {
                work_id: req.scope_work_id,
                canon: req.scope_canon,
                period: req.scope_period,
                node_id: req.scope_node_id,
                doc_range: doc_range.map(|(l, h)| vec![l, h]),
            },
            found,
            hit_count,
            sample_hits: scoped_hits.into_iter().take(req.limit.min(5)).collect(),
            search_strategy: AbsenceCheckSearchStrategy {
                phrase: phrase_strategy,
                limit: req.limit,
                max_candidates: candidate_limit,
            },
        })
    }

    pub async fn vector_info_impl(
        &self,
        _req: crate::tools::requests::VectorInfoRequest,
    ) -> Result<crate::tools::responses::VectorInfoResponse> {
        use crate::tools::responses::VectorInfoResponse;

        let path = self.resolve_vector_path()?;
        let info = crate::vector_index::VectorIndex::header_info(&path)?;
        let doc_table = self.doc_table().await?;
        let doc_table_path = self.resolve_doc_table_path()?;
        let fp = info
            .get("doc_table_fingerprint")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let doc_table_fingerprint_match =
            crate::document_table::match_index_fingerprint(&doc_table, &doc_table_path, fp)?
                .is_some();
        let row_count = info.get("row_count").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let total_docs = doc_table.passage_ids.len() as u32;
        let coverage_ratio = if total_docs == 0 {
            0.0
        } else {
            row_count as f64 / total_docs as f64
        };
        let coverage_kind = if row_count == 0 {
            "empty"
        } else if row_count >= total_docs {
            "full"
        } else {
            "partial"
        };
        Ok(VectorInfoResponse {
            schema: "sinorag-vector-info-v1",
            index_path: path.display().to_string(),
            info,
            doc_table_fingerprint_match,
            coverage: crate::tools::responses::VectorCoverage {
                kind: coverage_kind.to_string(),
                covered_docs: row_count,
                total_docs,
                coverage_ratio,
            },
        })
    }

    pub async fn vector_neighbors_impl(
        &self,
        req: crate::tools::requests::VectorNeighborsRequest,
    ) -> Result<crate::tools::responses::VectorNeighborsResponse> {
        use crate::tools::errors::ToolError;
        use crate::tools::responses::{VectorNeighborHit, VectorNeighborsResponse};

        let modes = req.seed_passage_id.is_some() as u8
            + req.query_embedding.is_some() as u8
            + req.query_text.is_some() as u8;
        if modes != 1 {
            return Err(ToolError::InvalidArgs(
                "provide exactly one of seed_passage_id, query_embedding, or query_text"
                    .to_string(),
            )
            .into_anyhow());
        }
        if req.query_text.is_some() {
            return Err(ToolError::QueryEmbeddingProviderNotConfigured.into_anyhow());
        }

        let load_started = std::time::Instant::now();
        let vector = self.vector().await?;
        let loading_index_ms = load_started.elapsed().as_millis();
        let doc_table = self.doc_table().await?;
        let (query_mode, seed_passage_id, query) = if let Some(seed) = req.seed_passage_id.clone() {
            validate_passage_input("vector-neighbors", &seed)?;
            let doc_id = doc_table
                .doc_id(&seed)
                .ok_or_else(|| passage_not_found(&seed))?;
            let query = vector
                .vector_for_doc_id(doc_id)
                .ok_or_else(|| anyhow::anyhow!("Seed doc_id {doc_id} has no vector row"))?;
            ("seed_passage".to_string(), Some(seed), query.to_vec())
        } else {
            (
                "query_embedding".to_string(),
                None,
                req.query_embedding.clone().unwrap_or_default(),
            )
        };

        let raw_hits = vector.search_embedding(
            &query,
            req.k + seed_passage_id.is_some() as usize,
            req.ef_search,
        )?;
        let mut ids = Vec::new();
        let seed_doc_id = seed_passage_id
            .as_deref()
            .and_then(|seed| doc_table.doc_id(seed));
        let mut hits = Vec::new();
        for hit in raw_hits {
            if Some(hit.doc_id) == seed_doc_id {
                continue;
            }
            if let Some(pid) = doc_table.passage_id(hit.doc_id) {
                ids.push(pid.to_string());
                hits.push(hit);
            }
            if hits.len() >= req.k.max(1) {
                break;
            }
        }

        let rows = if req.include_text && !ids.is_empty() {
            self.passages()
                .await?
                .passages_by_ids(
                    &ids,
                    "passage_id, source_work_id, main_title, heading, period, zh_text_raw",
                )
                .await?
        } else {
            Vec::new()
        };
        let by_id: FxHashMap<String, serde_json::Value> = rows
            .into_iter()
            .filter_map(|row| {
                let id = row.get("passage_id").and_then(|v| v.as_str())?.to_string();
                Some((id, row))
            })
            .collect();

        let hit_rows = hits
            .into_iter()
            .filter_map(|hit| {
                let passage_id = doc_table.passage_id(hit.doc_id)?.to_string();
                let row = by_id.get(&passage_id);
                Some(VectorNeighborHit {
                    passage_id,
                    doc_id: hit.doc_id,
                    ann_distance: hit.ann_distance,
                    ann_score: hit.ann_score,
                    vector_score: hit.ann_score,
                    source_work_id: row.and_then(|r| opt_str(r, "source_work_id")),
                    main_title: row.and_then(|r| opt_str(r, "main_title")),
                    heading: row.and_then(|r| opt_str(r, "heading")),
                    period: row.and_then(|r| opt_str(r, "period")),
                    snippet: row
                        .and_then(|r| opt_str(r, "zh_text_raw"))
                        .map(|s| char_prefix(&s, 160)),
                    warning: "semantic-neighbor-not-exact-evidence".to_string(),
                })
            })
            .collect();

        let mut warnings = vec!["semantic-neighbor-not-exact-evidence".to_string()];
        if req.rerank {
            warnings.push("rerank_requested_but_not_implemented".to_string());
        }
        Ok(VectorNeighborsResponse {
            schema: "sinorag-vector-neighbors-v1",
            query_mode,
            seed_passage_id,
            model_id: vector.header.model_id.clone(),
            model_revision: vector.header.model_revision.clone(),
            embedding_dim: vector.header.embedding_dim,
            distance: vector.header.distance.clone(),
            normalized: vector.header.normalized,
            rerank_requested: req.rerank,
            rerank_applied: false,
            score_interpretation:
                "ann_score is monotonic inverse hnsw_rs DistL2 over normalized embeddings; it is not calibrated cosine similarity"
                    .to_string(),
            loading_index_ms: Some(loading_index_ms),
            hnsw_build_ms: Some(vector.hnsw_build_ms),
            warnings,
            hits: hit_rows,
        })
    }

    pub async fn evidence_search_impl(
        &self,
        req: crate::tools::requests::EvidenceSearchRequest,
    ) -> Result<crate::tools::responses::EvidenceSearchResponse> {
        use crate::tools::requests::Verbosity;
        use crate::tools::responses::{ComponentStatus, EvidenceSearchResponse};

        let verbosity = req.verbosity;
        let mut components = Vec::new();
        let mut warnings = Vec::new();
        let output_limit = req.limit.max(1);
        let candidate_limit = req.max_candidates.max(output_limit);
        let retrieval_budget =
            crate::retrieval::RetrievalBudget::new(output_limit, candidate_limit)
                .with_time_limits(req.max_elapsed_ms, req.max_component_ms);
        let scope = evidence_scope_spec(&req);
        let mut trace = crate::retrieval::RetrievalTraceBuilder::new();
        warnings.push(format!("workflow_quality={}", req.quality));
        warnings.push(format!(
            "candidate_budget_requested={candidate_limit}; returned_limit={output_limit}"
        ));

        let budget = WorkflowBudget::new(req.max_elapsed_ms, req.max_component_ms);

        let started = std::time::Instant::now();
        let expanded = self
            .query_expand_terms_impl(crate::tools::requests::QueryExpandTermsRequest {
                phrase: req.phrase.clone(),
                mode: "all".to_string(),
                person_aliases: Vec::new(),
                max: 20,
            })
            .await?;
        let query_expansion_elapsed_ms = started.elapsed().as_millis();
        components.push(component_ok(
            "query_expansion",
            "query-expand-terms",
            query_expansion_elapsed_ms,
            format!("{} expanded/suggested terms", expanded.expanded.len()),
        ));
        trace.push(
            crate::retrieval::RetrievalStageReport::new(
                "query_expansion",
                expanded.expanded.len(),
                expanded.expanded.len(),
            )
            .with_elapsed_ms(query_expansion_elapsed_ms)
            .with_details(serde_json::json!({
                "expanded_terms_used": req.variant_policy == "search_variants",
                "variant_policy": req.variant_policy,
            })),
        );
        let expanded_terms_used = req.variant_policy == "search_variants";
        if !expanded_terms_used && expanded.expanded.len() > 1 {
            warnings.push("expanded_terms_are_suggestions_not_evidence_inputs".to_string());
        }

        let started = std::time::Instant::now();
        let exact_req = crate::tools::requests::SearchRequest {
            phrase: req.phrase.clone(),
            limit: output_limit,
            mode: "hits".to_string(),
            depth: "exact".to_string(),
            group_by: "work".to_string(),
            include_variants: false,
            limit_per_group: 5,
            // Brief snippets unless the caller explicitly asked for `full`.
            brief: !matches!(verbosity, Verbosity::Full),
            canon: req.scope_canon.clone(),
            source_work_id: req.scope_source_work_id.clone(),
            tradition: None,
            period: req.scope_period.clone(),
            origin: None,
            author: req.author.clone(),
            title: req.title.clone(),
            heading_path_prefix: req.heading_path_prefix.clone(),
        };
        let exact = match run_budgeted_component(&budget, self.search_impl(exact_req)).await {
            ComponentOutcome::Ok(v) => v,
            ComponentOutcome::Failed(err) => return Err(err),
            ComponentOutcome::TimedOut {
                timeout_ms,
                elapsed_ms: _,
            } => {
                return Err(ToolError::Timeout {
                    component: "exact_search".to_string(),
                    timeout_ms,
                }
                .into_anyhow())
            }
            ComponentOutcome::BudgetExhausted => {
                return Err(ToolError::Timeout {
                    component: "exact_search".to_string(),
                    timeout_ms: 0,
                }
                .into_anyhow())
            }
        };
        let exact_elapsed_ms = started.elapsed().as_millis();
        components.push(component_ok(
            "exact_search",
            "search",
            exact_elapsed_ms,
            format!("{} exact hits", exact.hits.len()),
        ));
        let exact_candidate_count = exact
            .search_strategy
            .candidate_count
            .unwrap_or(exact.hits.len());
        let exact_verified_count = exact
            .search_strategy
            .verified_count
            .unwrap_or(exact.hits.len());
        let mut exact_stage = crate::retrieval::RetrievalStageReport::new(
            "exact_search",
            exact_candidate_count,
            exact.hits.len(),
        )
        .with_verified_count(exact_verified_count)
        .with_elapsed_ms(exact_elapsed_ms)
        .with_details(serde_json::json!({
            "method": exact.search_strategy.method,
            "candidate_source": exact.search_strategy.candidate_source,
            "verification_source": exact.search_strategy.verification_source,
            "used_phrase_index": exact.search_strategy.used_phrase_index,
            "filters": exact.search_strategy.filters,
        }));
        if let Some(reason) = &exact.search_strategy.fallback_reason {
            exact_stage = exact_stage.with_warning(reason.clone());
        }
        trace.push(exact_stage);

        let absence_check = if req.include_absence_check {
            let started = std::time::Instant::now();
            let component_req = crate::tools::requests::AbsenceCheckRequest {
                phrase: req.phrase.clone(),
                limit: candidate_limit,
                max_candidates: candidate_limit,
                scope_work_id: req.scope_source_work_id.clone(),
                scope_canon: req.scope_canon.clone().into_iter().collect(),
                scope_period: req.scope_period.clone(),
                scope_node_id: req.scope_node_id,
            };
            record_component_outcome(
                run_budgeted_component(&budget, self.absence_check_impl(component_req)).await,
                &mut components,
                &mut warnings,
                "absence_check",
                "absence-check",
                started,
                |v| format!("found={}", v.found),
                "absence_check_failed",
                "absence_check_timed_out",
            )
        } else {
            components.push(component_skipped(
                "absence_check",
                "absence-check",
                ComponentStatus::SkippedNotRequested,
                "include_absence_check=false",
            ));
            None
        };
        trace.push(
            crate::retrieval::RetrievalStageReport::new(
                "absence_check",
                absence_check.as_ref().map_or(0, |v| v.hit_count),
                absence_check.as_ref().map_or(0, |v| v.sample_hits.len()),
            )
            .with_details(serde_json::json!({
                "requested": req.include_absence_check,
                "found": absence_check.as_ref().map(|v| v.found),
            })),
        );

        let first_attestation = if req.include_attestation {
            let started = std::time::Instant::now();
            let component_req = crate::tools::requests::FirstAttestationRequest {
                phrase: req.phrase.clone(),
                scope_canon: req.scope_canon.clone().into_iter().collect(),
                scope_period: req.scope_period.clone().into_iter().collect(),
                scope_source_work_id: req.scope_source_work_id.clone(),
                limit: output_limit,
                max_candidates: candidate_limit,
            };
            record_component_outcome(
                run_budgeted_component(&budget, self.first_attestation_impl(component_req)).await,
                &mut components,
                &mut warnings,
                "first_attestation",
                "first-attestation",
                started,
                |v| format!("first_found={}", v.first.is_some()),
                "first_attestation_failed",
                "first_attestation_timed_out",
            )
        } else {
            components.push(component_skipped(
                "first_attestation",
                "first-attestation",
                ComponentStatus::SkippedNotRequested,
                "include_attestation=false",
            ));
            None
        };
        trace.push(
            crate::retrieval::RetrievalStageReport::new(
                "first_attestation",
                first_attestation
                    .as_ref()
                    .map_or(0, |v| v.search_strategy.candidates_verified),
                first_attestation
                    .as_ref()
                    .map_or(0, |v| usize::from(v.first.is_some()) + v.next_earlier.len()),
            )
            .with_verified_count(
                first_attestation
                    .as_ref()
                    .map_or(0, |v| v.search_strategy.after_scope_and_sort),
            )
            .with_details(serde_json::json!({
                "requested": req.include_attestation,
                "first_found": first_attestation.as_ref().and_then(|v| v.first.as_ref()).is_some(),
            })),
        );
        let phrase_history = if req.include_history {
            let started = std::time::Instant::now();
            let component_req = crate::tools::requests::PhraseHistoryRequest {
                phrase: req.phrase.clone(),
                include_variants: false,
                timeline: true,
            };
            record_component_outcome(
                run_budgeted_component(&budget, self.phrase_history_impl(component_req)).await,
                &mut components,
                &mut warnings,
                "phrase_history",
                "phrase-history",
                started,
                |_| "history returned".to_string(),
                "phrase_history_failed",
                "phrase_history_timed_out",
            )
        } else {
            components.push(component_skipped(
                "phrase_history",
                "phrase-history",
                ComponentStatus::SkippedNotRequested,
                "include_history=false",
            ));
            None
        };
        trace.push(
            crate::retrieval::RetrievalStageReport::new(
                "phrase_history",
                phrase_history.as_ref().map_or(0, |v| {
                    v.payload
                        .pointer("/analysis/returned_count")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0) as usize
                }),
                phrase_history.as_ref().map_or(0, |v| {
                    v.payload
                        .get("evidence")
                        .and_then(|value| value.as_array())
                        .map_or(0, Vec::len)
                }),
            )
            .with_details(serde_json::json!({
                "requested": req.include_history,
                "timeline_buckets": phrase_history.as_ref().and_then(|v| {
                    v.payload
                        .pointer("/analysis/timeline_buckets")
                        .and_then(|value| value.as_object())
                        .map(|object| object.len())
                }),
            })),
        );
        let usage = if req.include_usage {
            let started = std::time::Instant::now();
            let component_req = crate::tools::requests::TraceTermUsageRequest {
                phrase: req.phrase.clone(),
                group_by: "period".to_string(),
                limit_total: candidate_limit.max(200),
                limit_per_group: 5,
                max_candidates: candidate_limit.max(200),
            };
            record_component_outcome(
                run_budgeted_component(&budget, self.trace_term_usage_impl(component_req)).await,
                &mut components,
                &mut warnings,
                "term_usage",
                "trace-term-usage",
                started,
                |v| format!("{} groups", v.groups.len()),
                "trace_term_usage_failed",
                "trace_term_usage_timed_out",
            )
        } else {
            components.push(component_skipped(
                "term_usage",
                "trace-term-usage",
                ComponentStatus::SkippedNotRequested,
                "include_usage=false",
            ));
            None
        };
        trace.push(
            crate::retrieval::RetrievalStageReport::new(
                "term_usage",
                usage.as_ref().map_or(0, |v| v.search_strategy.total_hits),
                usage.as_ref().map_or(0, |v| v.groups.len()),
            )
            .with_details(serde_json::json!({
                "requested": req.include_usage,
                "group_by": "period",
            })),
        );
        let clusters = if req.include_clusters {
            let started = std::time::Instant::now();
            let component_req = crate::tools::requests::ClusterHitsRequest {
                phrase: req.phrase.clone(),
                cluster_by: "work".to_string(),
                limit_total: candidate_limit.max(200),
                limit_per_cluster: 20,
                max_candidates: candidate_limit.max(200),
            };
            record_component_outcome(
                run_budgeted_component(&budget, self.cluster_hits_impl(component_req)).await,
                &mut components,
                &mut warnings,
                "clusters",
                "cluster-hits",
                started,
                |v| format!("{} clusters", v.clusters.len()),
                "cluster_hits_failed",
                "cluster_hits_timed_out",
            )
        } else {
            components.push(component_skipped(
                "clusters",
                "cluster-hits",
                ComponentStatus::SkippedNotRequested,
                "include_clusters=false",
            ));
            None
        };
        trace.push(
            crate::retrieval::RetrievalStageReport::new(
                "clusters",
                clusters.as_ref().map_or(0, |v| v.total_hits),
                clusters.as_ref().map_or(0, |v| v.clusters.len()),
            )
            .with_details(serde_json::json!({
                "requested": req.include_clusters,
                "cluster_by": "work",
            })),
        );

        let mut indexes_used = vec!["passages.parquet".to_string()];
        if self.resolve_phrase_path().is_ok() {
            indexes_used.push("phrase.index".to_string());
        }
        if self.resolve_doc_table_path().is_ok() {
            indexes_used.push("doc_table.bin".to_string());
        }
        if self.resolve_catalog_path().is_ok() {
            indexes_used.push("catalog.index".to_string());
        }
        let mut fallbacks = Vec::new();
        if exact.search_strategy.used_phrase_index == Some(false) {
            fallbacks.push(
                exact
                    .search_strategy
                    .fallback_reason
                    .clone()
                    .unwrap_or_else(|| "phrase_index_unavailable_or_not_used".to_string()),
            );
        }
        let suggested_next_tools = vec![
            suggested_tool(
                "frontier",
                serde_json::json!({"seed": "<choose passage_id from exact.hits>", "limit": output_limit.min(10), "phrase_limit": 10}),
                "expand an exact evidence hit into distinctive phrase and TF-IDF discovery leads",
            ),
            suggested_tool(
                "source-investigate",
                serde_json::json!({"seed_passage_id": "<choose passage_id from exact.hits>", "phrases": [req.phrase.clone()]}),
                "inspect context and follow-up leads for a selected evidence hit",
            ),
        ];

        // At `summary` verbosity, keep the exact hits (the evidence) but drop the
        // optional analysis blocks — their per-stage counts remain in `stages`.
        let summary_only = matches!(verbosity, Verbosity::Summary);
        Ok(EvidenceSearchResponse {
            schema: "sinorag-evidence-search-v1",
            workflow: "exact_evidence",
            candidate_budget: retrieval_budget,
            scope,
            phrase: req.phrase,
            expanded_terms: expanded.expanded,
            expanded_terms_used,
            variant_policy: req.variant_policy,
            exact,
            absence_check: if summary_only { None } else { absence_check },
            first_attestation: if summary_only {
                None
            } else {
                first_attestation
            },
            phrase_history: if summary_only { None } else { phrase_history },
            usage: if summary_only { None } else { usage },
            clusters: if summary_only { None } else { clusters },
            stages: trace.finish(),
            components,
            suggested_next_tools,
            indexes_used,
            fallbacks,
            evidence_status: "exact_phrase_evidence".to_string(),
            warnings,
        })
    }

    pub async fn hybrid_discover_impl(
        &self,
        req: crate::tools::requests::HybridDiscoverRequest,
    ) -> Result<crate::tools::responses::HybridDiscoverResponse> {
        use crate::tools::requests::Verbosity;
        use crate::tools::responses::{
            ComponentStatus, HybridDiscoverGroups, HybridDiscoverHit, HybridDiscoverResponse,
        };

        let verbosity = req.verbosity;
        if req.seed_passage_id.is_none() && req.query_embedding.is_none() {
            return Err(ToolError::InvalidArgs(
                "provide seed_passage_id or query_embedding".to_string(),
            )
            .into_anyhow());
        }

        let mut warnings = Vec::new();
        let mut components = Vec::new();
        let mut indexes_used = Vec::new();
        let output_limit = req.limit.max(1);
        let candidate_limit = req.max_candidates.max(output_limit);
        warnings.push(format!("workflow_quality={}", req.quality));
        warnings.push(format!(
            "candidate_budget_requested={candidate_limit}; returned_limit={output_limit}"
        ));
        warnings.push(
            "prefer frontier for ordinary seed expansion; use hybrid-discover when semantic-vector candidates are specifically useful".to_string(),
        );
        let full = matches!(verbosity, Verbosity::Full);
        let summary_only = matches!(verbosity, Verbosity::Summary);

        let budget = WorkflowBudget::new(req.max_elapsed_ms, req.max_component_ms);

        let vector_started = std::time::Instant::now();
        let mut vector_stage_warning = None;
        let vector_neighbors = if self.resolve_vector_path().is_ok() {
            let vector_req = crate::tools::requests::VectorNeighborsRequest {
                seed_passage_id: req.seed_passage_id.clone(),
                query_embedding: req.query_embedding.clone(),
                query_text: None,
                k: candidate_limit,
                ef_search: 64,
                include_text: full,
                rerank: true,
            };
            match run_budgeted_component(&budget, self.vector_neighbors_impl(vector_req)).await {
                ComponentOutcome::Ok(v) => {
                    indexes_used.push("vector.index".to_string());
                    components.push(component_ok(
                        "vector_neighbors",
                        "vector-neighbors",
                        vector_started.elapsed().as_millis(),
                        format!("{} semantic candidates", v.hits.len()),
                    ));
                    Some(v)
                }
                ComponentOutcome::Failed(err) => {
                    components.push(component_failed(
                        "vector_neighbors",
                        "vector-neighbors",
                        vector_started.elapsed().as_millis(),
                        &err,
                    ));
                    warnings.push("vector_neighbors_unavailable".to_string());
                    vector_stage_warning = Some("vector_neighbors_unavailable".to_string());
                    None
                }
                ComponentOutcome::TimedOut {
                    elapsed_ms,
                    timeout_ms,
                } => {
                    components.push(component_timed_out(
                        "vector_neighbors",
                        "vector-neighbors",
                        elapsed_ms,
                        timeout_ms,
                    ));
                    warnings.push("vector_neighbors_timed_out".to_string());
                    vector_stage_warning = Some("vector_neighbors_timed_out".to_string());
                    None
                }
                ComponentOutcome::BudgetExhausted => {
                    components.push(component_skipped(
                        "vector_neighbors",
                        "vector-neighbors",
                        ComponentStatus::SkippedBudgetExhausted,
                        "budget exhausted",
                    ));
                    vector_stage_warning = Some("budget_exhausted".to_string());
                    None
                }
            }
        } else {
            components.push(component_skipped(
                "vector_neighbors",
                "vector-neighbors",
                ComponentStatus::SkippedUnavailable,
                "vector.index missing; semantic neighbors unavailable",
            ));
            warnings.push("vector_index_missing_semantic_neighbors_unavailable".to_string());
            vector_stage_warning = Some("vector_index_missing".to_string());
            None
        };
        let vector_elapsed_ms = vector_started.elapsed().as_millis();

        let mut tfidf_elapsed_ms = None;
        let tfidf_similar = if let Some(seed) = &req.seed_passage_id {
            let started = std::time::Instant::now();
            let component_req = crate::tools::requests::SimilarRequest {
                seed: seed.clone(),
                limit: candidate_limit,
                shared_ngram_limit: 12,
                shared_phrase_limit: 8,
                min_shared_phrase_len: 4,
            };
            let value = record_component_outcome(
                run_budgeted_component(&budget, self.similar_impl(component_req)).await,
                &mut components,
                &mut warnings,
                "tfidf_similar",
                "similar",
                started,
                |v| format!("{} lexical parallels", v.similar_passages.len()),
                "tfidf_similar_unavailable",
                "tfidf_similar_timed_out",
            );
            if value.is_some() {
                indexes_used.push("tfidf.index".to_string());
            }
            tfidf_elapsed_ms = Some(started.elapsed().as_millis());
            value
        } else {
            components.push(component_skipped(
                "tfidf_similar",
                "similar",
                ComponentStatus::SkippedUnavailable,
                "TF-IDF similar requires seed_passage_id",
            ));
            None
        };

        let context = if req.include_context && full {
            if let Some(seed) = &req.seed_passage_id {
                let started = std::time::Instant::now();
                let component_req = crate::tools::requests::ExpandContextAdaptiveRequest {
                    passage_id: seed.clone(),
                    max_chars: req.max_context_chars,
                };
                record_component_outcome(
                    run_budgeted_component(
                        &budget,
                        self.expand_context_adaptive_impl(component_req),
                    )
                    .await,
                    &mut components,
                    &mut warnings,
                    "context",
                    "expand-context-adaptive",
                    started,
                    |v| format!("{} context passages", v.passage_count),
                    "context_expansion_failed",
                    "context_expansion_timed_out",
                )
            } else {
                components.push(component_skipped(
                    "context",
                    "expand-context-adaptive",
                    ComponentStatus::SkippedUnavailable,
                    "context requires seed_passage_id",
                ));
                None
            }
        } else {
            let reason = if req.include_context {
                "context returned only with verbosity=full"
            } else {
                "include_context=false"
            };
            components.push(component_skipped(
                "context",
                "expand-context-adaptive",
                ComponentStatus::SkippedNotRequested,
                reason,
            ));
            None
        };

        let mut merged: FxHashMap<String, HybridDiscoverHit> = FxHashMap::default();
        if let Some(v) = &vector_neighbors {
            for (rank, hit) in v.hits.iter().enumerate() {
                let vector_rank = Some(rank + 1);
                let (semantic_score, lexical_score, final_score, candidate_sources) =
                    crate::retrieval::refresh_hybrid_scores(vector_rank, None);
                merged.insert(
                    hit.passage_id.clone(),
                    HybridDiscoverHit {
                        passage_id: hit.passage_id.clone(),
                        labels: vec!["semantic_candidate".to_string()],
                        candidate_sources: candidate_sources
                            .into_iter()
                            .map(|source| source.as_str().to_string())
                            .collect(),
                        evidence_status: "not_evidence_until_verified".to_string(),
                        vector_score: Some(hit.vector_score),
                        vector_rank,
                        tfidf_score: None,
                        tfidf_rank: None,
                        semantic_score,
                        lexical_score,
                        final_score,
                        merged_rank_reason: "semantic candidate from vector-neighbors".to_string(),
                        title: hit.main_title.clone(),
                        snippet: hit.snippet.as_ref().map(|s| char_prefix(s, 120)),
                    },
                );
            }
        }
        if let Some(s) = &tfidf_similar {
            for (rank, item) in s.similar_passages.iter().enumerate() {
                let Some(pid) = item.get("passage_id").and_then(|v| v.as_str()) else {
                    continue;
                };
                let entry = merged.entry(pid.to_string()).or_insert(HybridDiscoverHit {
                    passage_id: pid.to_string(),
                    labels: Vec::new(),
                    candidate_sources: Vec::new(),
                    evidence_status: "lexical_candidate_needs_verification".to_string(),
                    vector_score: None,
                    vector_rank: None,
                    tfidf_score: None,
                    tfidf_rank: None,
                    semantic_score: None,
                    lexical_score: None,
                    final_score: 0.0,
                    merged_rank_reason: "lexical parallel from TF-IDF similar".to_string(),
                    title: opt_str(item, "main_title"),
                    snippet: opt_str(item, "zh_text_raw").map(|s| char_prefix(&s, 120)),
                });
                if !entry.labels.iter().any(|l| l == "lexical_parallel") {
                    entry.labels.push("lexical_parallel".to_string());
                }
                entry.tfidf_score = item
                    .get("tfidf_cosine")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32);
                entry.tfidf_rank = Some(rank + 1);
                if entry.labels.iter().any(|l| l == "semantic_candidate") {
                    entry.evidence_status =
                        "overlap_candidate_needs_exact_verification".to_string();
                    entry.merged_rank_reason =
                        "appears in both vector and TF-IDF candidates".to_string();
                }
                refresh_hybrid_scores(entry);
            }
        }
        for hit in merged.values_mut() {
            refresh_hybrid_scores(hit);
        }
        let mut merged_hits: Vec<_> = merged.into_values().collect();
        let merged_candidate_count = merged_hits.len();
        merged_hits.sort_by(|a, b| {
            let a_overlap = a.vector_rank.is_some() && a.tfidf_rank.is_some();
            let b_overlap = b.vector_rank.is_some() && b.tfidf_rank.is_some();
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b_overlap.cmp(&a_overlap))
                .then_with(|| {
                    a.vector_rank
                        .unwrap_or(usize::MAX)
                        .cmp(&b.vector_rank.unwrap_or(usize::MAX))
                })
                .then_with(|| {
                    a.tfidf_rank
                        .unwrap_or(usize::MAX)
                        .cmp(&b.tfidf_rank.unwrap_or(usize::MAX))
                })
        });
        merged_hits.truncate(output_limit);
        let returned_count = merged_hits.len();

        let has_vector_candidates = vector_neighbors
            .as_ref()
            .is_some_and(|v| !v.hits.is_empty());
        let has_tfidf_candidates = tfidf_similar
            .as_ref()
            .is_some_and(|s| !s.similar_passages.is_empty());
        let (mode, mode_reason) = match (has_vector_candidates, has_tfidf_candidates) {
            (true, true) => (
                "hybrid",
                "vector and TF-IDF sources both produced candidates",
            ),
            (false, true) => (
                "lexical_only",
                "TF-IDF produced candidates; semantic vector candidates are unavailable or empty",
            ),
            (true, false) => (
                "semantic_only",
                "vector search produced candidates; TF-IDF candidates are unavailable, not applicable, or empty",
            ),
            (false, false) => (
                "unavailable",
                "no vector or TF-IDF discovery candidates were produced",
            ),
        };
        warnings.push(format!("hybrid_discover_mode={mode}"));

        let mut trace = crate::retrieval::RetrievalTraceBuilder::new();
        let mut vector_stage = crate::retrieval::RetrievalStageReport::new(
            "vector_candidates",
            vector_neighbors.as_ref().map_or(0, |v| v.hits.len()),
            vector_neighbors.as_ref().map_or(0, |v| v.hits.len()),
        )
        .with_elapsed_ms(vector_elapsed_ms);
        if let Some(warning) = vector_stage_warning {
            vector_stage = vector_stage.with_warning(warning);
        }
        trace.push(vector_stage);
        let mut tfidf_stage = crate::retrieval::RetrievalStageReport::new(
            "tfidf_candidates",
            tfidf_similar
                .as_ref()
                .map_or(0, |s| s.similar_passages.len()),
            tfidf_similar
                .as_ref()
                .map_or(0, |s| s.similar_passages.len()),
        );
        if let Some(elapsed_ms) = tfidf_elapsed_ms {
            tfidf_stage = tfidf_stage.with_elapsed_ms(elapsed_ms);
        } else if req.seed_passage_id.is_none() {
            tfidf_stage = tfidf_stage.with_warning("tfidf_requires_seed_passage_id");
        }
        trace.push(tfidf_stage);
        trace.push(crate::retrieval::RetrievalStageReport::new(
            "merge_and_rank",
            merged_candidate_count,
            merged_candidate_count,
        ));
        trace.push(crate::retrieval::RetrievalStageReport::new(
            "return",
            merged_candidate_count,
            returned_count,
        ));

        let groups = HybridDiscoverGroups {
            semantic_candidates: merged_hits
                .iter()
                .filter(|h| h.labels.iter().any(|l| l == "semantic_candidate"))
                .map(|h| h.passage_id.clone())
                .collect(),
            lexical_parallels: merged_hits
                .iter()
                .filter(|h| h.labels.iter().any(|l| l == "lexical_parallel"))
                .map(|h| h.passage_id.clone())
                .collect(),
            overlap_candidates: merged_hits
                .iter()
                .filter(|h| h.vector_rank.is_some() && h.tfidf_rank.is_some())
                .map(|h| h.passage_id.clone())
                .collect(),
        };
        let mut suggested_next_tools = Vec::new();
        if let Some(seed) = &req.seed_passage_id {
            suggested_next_tools.push(suggested_tool(
                "frontier",
                serde_json::json!({"seed": seed, "limit": output_limit.min(10), "phrase_limit": 10}),
                "use frontier as the usual next discovery step; it is lexical/distributional and avoids vector noise",
            ));
            suggested_next_tools.push(suggested_tool(
                "source-read",
                serde_json::json!({"passage_id": seed, "direction": "around", "max_chars": 4000}),
                "read around the seed passage continuously before treating candidates as evidence",
            ));
        }
        suggested_next_tools.push(suggested_tool(
            "evidence-search",
            serde_json::json!({"phrase": "<extract exact phrase from candidate>", "include_attestation": true}),
            "verify discovery candidates with exact phrase evidence before citing them",
        ));

        // `merged_hits` already carries each candidate's per-source scores and
        // ranks, so the raw `vector_neighbors`/`tfidf_similar` blocks are pure
        // duplication outside of debugging. Drop them below `full`; at `summary`
        // also drop context and the per-hit snippets, leaving counts + scored ids.
        let merged_hits = if summary_only {
            merged_hits
                .into_iter()
                .map(|mut h| {
                    h.snippet = None;
                    h
                })
                .collect()
        } else {
            merged_hits
        };

        Ok(HybridDiscoverResponse {
            schema: "sinorag-hybrid-discover-v1",
            workflow: "semantic_discovery",
            mode: mode.to_string(),
            mode_reason: mode_reason.to_string(),
            candidate_budget: crate::retrieval::RetrievalBudget::new(output_limit, candidate_limit)
                .with_time_limits(req.max_elapsed_ms, req.max_component_ms),
            merged_candidate_count,
            returned_count,
            seed_passage_id: req.seed_passage_id,
            vector_neighbors: if full { vector_neighbors } else { None },
            tfidf_similar: if full { tfidf_similar } else { None },
            context: if full { context } else { None },
            groups,
            merged_hits,
            stages: trace.finish(),
            components,
            suggested_next_tools,
            indexes_used,
            warnings,
        })
    }

    pub async fn source_investigate_impl(
        &self,
        req: crate::tools::requests::SourceInvestigateRequest,
    ) -> Result<crate::tools::responses::SourceInvestigateResponse> {
        use crate::tools::requests::Verbosity;
        use crate::tools::responses::{ComponentStatus, SourceInvestigateResponse};

        let verbosity = req.verbosity;
        let mut components = Vec::new();
        let mut risk_notes = Vec::new();
        if req.include_vector {
            risk_notes.push("vector neighbors are discovery candidates only".to_string());
        }
        if req.include_frontier || req.include_similar {
            risk_notes.push(
                "TF-IDF similar passages are lexical candidates until exact evidence is verified"
                    .to_string(),
            );
        }
        risk_notes.push(format!("workflow_quality={}", req.quality));

        let budget = WorkflowBudget::new(req.max_elapsed_ms, req.max_component_ms);
        let mut component_warnings = Vec::new();

        let started = std::time::Instant::now();
        let seed = self
            .passage_impl(crate::tools::requests::PassageRequest {
                id: req.seed_passage_id.clone(),
            })
            .await?;
        components.push(component_ok(
            "seed_passage",
            "passage",
            started.elapsed().as_millis(),
            "seed passage loaded",
        ));
        let context = if req.include_context {
            let started = std::time::Instant::now();
            let component_req = crate::tools::requests::ExpandContextAdaptiveRequest {
                passage_id: req.seed_passage_id.clone(),
                max_chars: req.max_context_chars,
            };
            record_component_outcome(
                run_budgeted_component(&budget, self.expand_context_adaptive_impl(component_req))
                    .await,
                &mut components,
                &mut component_warnings,
                "context",
                "expand-context-adaptive",
                started,
                |v| format!("{} context passages", v.passage_count),
                "context expansion failed",
                "context expansion timed out",
            )
        } else {
            components.push(component_skipped(
                "context",
                "expand-context-adaptive",
                ComponentStatus::SkippedNotRequested,
                "include_context=false",
            ));
            None
        };
        let frontier = if req.include_frontier {
            let started = std::time::Instant::now();
            let component_req = crate::tools::requests::FrontierRequest {
                seed: req.seed_passage_id.clone(),
                limit: req.limit,
                phrase_limit: 20,
                min_similarity: None,
                scope_canon: vec![],
                scope_period: vec![],
                scope_source_work_id: None,
            };
            record_component_outcome(
                run_budgeted_component(&budget, self.frontier_impl(component_req)).await,
                &mut components,
                &mut component_warnings,
                "frontier",
                "frontier",
                started,
                |_| "frontier packet returned".to_string(),
                "frontier failed or optional indexes are missing",
                "frontier timed out",
            )
        } else {
            components.push(component_skipped(
                "frontier",
                "frontier",
                ComponentStatus::SkippedNotRequested,
                "include_frontier=false",
            ));
            None
        };
        let similar = if req.include_similar {
            if let Some(frontier) = &frontier {
                let similar_passages = frontier
                    .payload
                    .get("similar_passages")
                    .and_then(|value| value.as_array())
                    .cloned()
                    .unwrap_or_default();
                components.push(component_ok(
                    "similar",
                    "frontier",
                    0,
                    format!(
                        "reused {} TF-IDF parallels already computed by frontier",
                        similar_passages.len()
                    ),
                ));
                Some(crate::tools::responses::SimilarResponse {
                    schema: "sinorag-similar-v1",
                    seed: req.seed_passage_id.clone(),
                    similar_passages,
                })
            } else {
                let started = std::time::Instant::now();
                let component_req = crate::tools::requests::SimilarRequest {
                    seed: req.seed_passage_id.clone(),
                    limit: req.limit,
                    shared_ngram_limit: 12,
                    shared_phrase_limit: 8,
                    min_shared_phrase_len: 4,
                };
                record_component_outcome(
                    run_budgeted_component(&budget, self.similar_impl(component_req)).await,
                    &mut components,
                    &mut component_warnings,
                    "similar",
                    "similar",
                    started,
                    |v| format!("{} similar passages", v.similar_passages.len()),
                    "TF-IDF similar failed or index is unavailable",
                    "TF-IDF similar timed out",
                )
            }
        } else {
            components.push(component_skipped(
                "similar",
                "similar",
                ComponentStatus::SkippedNotRequested,
                "include_similar=false",
            ));
            None
        };
        let vector_neighbors = if req.include_vector {
            let started = std::time::Instant::now();
            let component_req = crate::tools::requests::VectorNeighborsRequest {
                seed_passage_id: Some(req.seed_passage_id.clone()),
                query_embedding: None,
                query_text: None,
                k: req.limit,
                ef_search: 64,
                include_text: true,
                rerank: true,
            };
            record_component_outcome(
                run_budgeted_component(&budget, self.vector_neighbors_impl(component_req)).await,
                &mut components,
                &mut component_warnings,
                "vector_neighbors",
                "vector-neighbors",
                started,
                |v| format!("{} vector neighbors", v.hits.len()),
                "vector neighbors failed or vector coverage is missing",
                "vector neighbors timed out",
            )
        } else {
            components.push(component_skipped(
                "vector_neighbors",
                "vector-neighbors",
                ComponentStatus::SkippedNotRequested,
                "include_vector=false",
            ));
            None
        };
        let mut phrase_histories = Vec::new();
        for phrase in &req.phrases {
            let started = std::time::Instant::now();
            let component_req = crate::tools::requests::PhraseHistoryRequest {
                phrase: phrase.clone(),
                include_variants: false,
                timeline: true,
            };
            match run_budgeted_component(&budget, self.phrase_history_impl(component_req)).await {
                ComponentOutcome::Ok(history) => {
                    components.push(component_ok(
                        "phrase_history",
                        "phrase-history",
                        started.elapsed().as_millis(),
                        format!("phrase={phrase}"),
                    ));
                    phrase_histories.push(history);
                }
                ComponentOutcome::Failed(err) => {
                    components.push(component_failed(
                        "phrase_history",
                        "phrase-history",
                        started.elapsed().as_millis(),
                        &err,
                    ));
                    risk_notes.push(format!("phrase history failed for {phrase}"));
                }
                ComponentOutcome::TimedOut {
                    elapsed_ms,
                    timeout_ms,
                } => {
                    components.push(component_timed_out(
                        "phrase_history",
                        "phrase-history",
                        elapsed_ms,
                        timeout_ms,
                    ));
                    risk_notes.push(format!("phrase history timed out for {phrase}"));
                }
                ComponentOutcome::BudgetExhausted => {
                    components.push(component_skipped(
                        "phrase_history",
                        "phrase-history",
                        ComponentStatus::SkippedBudgetExhausted,
                        "budget exhausted",
                    ));
                    break;
                }
            }
        }
        risk_notes.extend(component_warnings);
        let mut suggested_next_tools: Vec<_> = req
            .phrases
            .iter()
            .map(|p| {
                suggested_tool(
                    "evidence-search",
                    serde_json::json!({"phrase": p, "include_attestation": true}),
                    "establish exact phrase evidence without repeating the history already gathered here",
                )
            })
            .collect();
        suggested_next_tools.push(suggested_tool(
            "source-read",
            serde_json::json!({"passage_id": req.seed_passage_id.clone(), "direction": "around", "max_chars": 4000}),
            "read the seed passage in a continuous source stream with citation-aware chunks",
        ));
        if !req.include_frontier {
            suggested_next_tools.push(suggested_tool(
                "frontier",
                serde_json::json!({"seed": req.seed_passage_id.clone(), "limit": req.limit}),
                "add lexical discovery candidates when broader investigation is needed",
            ));
        }
        // At `summary` verbosity, drop the raw analysis blocks (they are the bulk
        // of the payload); the next-step guidance and risk notes are enough to
        // decide where to go next. `standard`/`full` keep whatever was requested.
        let summary_only = matches!(verbosity, Verbosity::Summary);
        Ok(SourceInvestigateResponse {
            schema: "sinorag-source-investigate-v1",
            workflow: "source_investigation",
            seed_passage_id: req.seed_passage_id,
            seed,
            context: if summary_only { None } else { context },
            frontier: if summary_only { None } else { frontier },
            similar: if summary_only { None } else { similar },
            vector_neighbors: if summary_only { None } else { vector_neighbors },
            phrase_histories: if summary_only {
                Vec::new()
            } else {
                phrase_histories
            },
            components,
            suggested_next_tools,
            risk_notes,
        })
    }

    pub async fn scope_profile_impl(
        &self,
        req: crate::tools::requests::ScopeProfileRequest,
    ) -> Result<crate::tools::responses::ScopeProfileResponse> {
        use crate::tools::responses::{ComponentStatus, ScopeProfileResponse};

        let mut components = Vec::new();
        components.push(component_ok(
            "workflow_options",
            "scope-profile",
            0,
            format!("quality={}", req.quality),
        ));
        let started = std::time::Instant::now();
        let comparison = self
            .compare_usage_impl(crate::tools::requests::CompareUsageRequest {
                scope_a_node_id: req.scope_a_node_id,
                scope_a_work_id: req.scope_a_work_id,
                scope_a_canon: req.scope_a_canon,
                scope_a_period: req.scope_a_period,
                scope_b_node_id: req.scope_b_node_id,
                scope_b_work_id: req.scope_b_work_id,
                scope_b_canon: req.scope_b_canon,
                scope_b_period: req.scope_b_period,
                gram_len: req.gram_len,
                limit_passages: req.limit_passages,
                limit_terms: req.limit_terms,
            })
            .await?;
        components.push(component_ok(
            "scope_comparison",
            "compare-usage",
            started.elapsed().as_millis(),
            format!(
                "{} distinctive A terms, {} distinctive B terms",
                comparison.distinctive_to_a.len(),
                comparison.distinctive_to_b.len()
            ),
        ));
        let term_usage = if let Some(phrase) = &req.phrase {
            let started = std::time::Instant::now();
            match self
                .trace_term_usage_impl(crate::tools::requests::TraceTermUsageRequest {
                    phrase: phrase.clone(),
                    group_by: "period".to_string(),
                    limit_total: req.limit_passages,
                    limit_per_group: 5,
                    max_candidates: req.limit_passages,
                })
                .await
            {
                Ok(v) => {
                    components.push(component_ok(
                        "term_usage",
                        "trace-term-usage",
                        started.elapsed().as_millis(),
                        format!("{} groups", v.groups.len()),
                    ));
                    Some(v)
                }
                Err(err) => {
                    components.push(component_failed(
                        "term_usage",
                        "trace-term-usage",
                        started.elapsed().as_millis(),
                        &err,
                    ));
                    None
                }
            }
        } else {
            components.push(component_skipped(
                "term_usage",
                "trace-term-usage",
                ComponentStatus::SkippedNotRequested,
                "phrase not provided",
            ));
            None
        };
        let suggested_next_tools = vec![suggested_tool(
            "evidence-search",
            serde_json::json!({"phrase": "<distinctive term>", "include_usage": true}),
            "verify whether a distinctive term has exact evidence in a scope",
        )];
        Ok(ScopeProfileResponse {
            schema: "sinorag-scope-profile-v1",
            workflow: "scope_comparison",
            phrase: req.phrase,
            comparison,
            term_usage,
            components,
            suggested_next_tools,
        })
    }

    pub async fn report_from_evidence_impl(
        &self,
        req: crate::tools::requests::ReportFromEvidenceRequest,
    ) -> Result<crate::tools::responses::ReportFromEvidenceResponse> {
        use crate::tools::responses::ReportFromEvidenceResponse;

        let mut components = Vec::new();
        let mut warnings = Vec::new();
        let started = std::time::Instant::now();
        let validation = self
            .validate_adjudication_impl(crate::tools::requests::ValidateAdjudicationRequest {
                path: req.adjudication.clone(),
            })
            .await?;
        components.push(component_ok(
            "validation",
            "validate-adjudication",
            started.elapsed().as_millis(),
            format!("valid={}", validation.valid),
        ));
        if !validation.valid {
            warnings.push("adjudication_invalid_graph_and_report_skipped".to_string());
            return Ok(ReportFromEvidenceResponse {
                schema: "sinorag-report-from-evidence-v1",
                workflow: "report_from_evidence",
                validation,
                graph: None,
                report: None,
                components,
                suggested_next_tools: vec![suggested_tool(
                    "report-from-evidence",
                    serde_json::json!({"adjudication": "<corrected adjudication path>", "graph_out": "<graph.json>", "report_out": "<report.md>"}),
                    "fix the validation issues listed in `validation`, then re-run this tool — it re-validates and proceeds to graph/report build in one call",
                )],
                warnings,
            });
        }
        let started = std::time::Instant::now();
        let graph = self
            .graph_build_impl(crate::tools::requests::GraphBuildRequest {
                input: req.adjudication.clone(),
                kind: req.kind,
                name: req.name,
                out: req.graph_out.clone(),
            })
            .await?;
        components.push(component_ok(
            "graph",
            "graph-build",
            started.elapsed().as_millis(),
            "graph artifact built",
        ));
        let started = std::time::Instant::now();
        let report = self
            .report_build_impl(crate::tools::requests::ReportBuildRequest {
                inputs: vec![req.adjudication, req.graph_out],
                out: req.report_out,
                title: req.title,
                essay_max_pages: 5,
            })
            .await?;
        components.push(component_ok(
            "report",
            "report-build",
            started.elapsed().as_millis(),
            "report artifact built",
        ));
        Ok(ReportFromEvidenceResponse {
            schema: "sinorag-report-from-evidence-v1",
            workflow: "report_from_evidence",
            validation,
            graph: Some(graph),
            report: Some(report),
            components,
            suggested_next_tools: Vec::new(),
            warnings,
        })
    }

    /// Implement the batch-evidence-search tool
    pub async fn batch_evidence_search_impl(
        &self,
        req: crate::tools::requests::BatchEvidenceSearchRequest,
    ) -> Result<crate::tools::responses::BatchEvidenceSearchResponse> {
        use crate::tools::responses::{BatchEvidenceSearchResponse, BatchEvidenceSearchResult};

        use futures::{stream, StreamExt};

        self.resolve_passages_path()?;
        let limit = req.limit;
        let concurrency = req.concurrency.clamp(1, 8);
        let mut results = stream::iter(req.phrases.into_iter().enumerate().map(
            |(index, phrase)| async move {
                let result = match self
                    .search_with(crate::tools::requests::SearchRequest {
                        phrase: phrase.clone(),
                        limit,
                        mode: "hits".to_string(),
                        depth: "exact".to_string(),
                        group_by: "work".to_string(),
                        include_variants: false,
                        limit_per_group: 5,
                        brief: true,
                        canon: None,
                        source_work_id: None,
                        tradition: None,
                        period: None,
                        origin: None,
                        author: None,
                        title: None,
                        heading_path_prefix: None,
                    }, false)
                    .await
                {
                    Ok(search_result) => {
                        let sample_passage_ids: Vec<String> = search_result
                            .hits
                            .iter()
                            .take(5)
                            .map(|h| h.passage_id.clone())
                            .collect();
                        BatchEvidenceSearchResult {
                            phrase,
                            hit_count: search_result.hits.len(),
                            returned_count: search_result.hits.len(),
                            possibly_truncated: limit > 0 && search_result.hits.len() >= limit,
                            sample_passage_ids,
                            error: None,
                        }
                    }
                    Err(err) => BatchEvidenceSearchResult {
                        phrase,
                        hit_count: 0,
                        returned_count: 0,
                        possibly_truncated: false,
                        sample_passage_ids: Vec::new(),
                        error: Some(err.to_string()),
                    },
                };
                (index, result)
            },
        ))
        .buffer_unordered(concurrency)
        .collect::<Vec<_>>()
        .await;
        results.sort_unstable_by_key(|(index, _)| *index);
        let results = results.into_iter().map(|(_, result)| result).collect();

        Ok(BatchEvidenceSearchResponse {
            schema: "sinorag-batch-evidence-search-v1",
            results,
        })
    }

    async fn pair_candidate_rows(
        &self,
        passages: &DataFusionStore,
        doc_table: &DocumentTable,
        phrase_index_path: Option<&std::path::Path>,
        variants: &[String],
        limit: usize,
        doc_range: Option<(u32, u32)>,
        canon: Option<&[String]>,
        period: Option<&str>,
        source_work_id: Option<&str>,
    ) -> Result<(Vec<serde_json::Value>, serde_json::Value)> {
        use crate::research_tools::phrase::phrase_rows_for_forms;

        let mut rows = Vec::<serde_json::Value>::new();
        let mut strategies = Vec::new();
        for variant in variants {
            let variant_forms = crate::normalize::PhraseForms::from_normalized_variant(variant);
            let (variant_rows, strategy) = phrase_rows_for_forms(
                passages,
                doc_table,
                phrase_index_path,
                &variant_forms,
                limit,
                doc_range,
                canon,
                period,
            )
            .await?;
            strategies.push(serde_json::json!({
                "variant": variant,
                "strategy": strategy,
            }));
            rows.extend(variant_rows);
        }
        rows.retain(|row| {
            if let Some(work_id) = source_work_id {
                if row
                    .get("source_work_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    != work_id
                {
                    return false;
                }
            }
            true
        });
        let mut seen = std::collections::BTreeSet::<String>::new();
        rows.retain(|row| {
            let pid = row
                .get("passage_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            !pid.is_empty() && seen.insert(pid)
        });
        rows.sort_by_key(|row| {
            row.get("passage_id")
                .and_then(|v| v.as_str())
                .and_then(|pid| doc_table.doc_id(pid))
                .unwrap_or(u32::MAX)
        });
        rows.truncate(limit);

        Ok((
            rows,
            serde_json::json!({
                "variants": variants,
                "strategies": strategies,
                "deduped_count": seen.len(),
                "limit": limit,
            }),
        ))
    }

    /// Helper: resolve doc range from catalog
    fn resolve_doc_range(
        &self,
        catalog: &CorpusCatalogIndex,
        node_id: Option<u32>,
        work_id: Option<&str>,
    ) -> Result<Option<(u32, u32)>> {
        if let Some(nid) = node_id {
            let node = catalog
                .get_node(nid)
                .ok_or_else(|| anyhow::anyhow!("unknown node_id: {nid}"))?;
            return Ok(node.first_doc_id.zip(node.last_doc_id));
        }
        if let Some(wid) = work_id {
            let work = catalog
                .get_work(wid)
                .ok_or_else(|| anyhow::anyhow!("unknown work_id: {wid}"))?;
            let root = catalog
                .get_node(work.root_node)
                .ok_or_else(|| anyhow::anyhow!("work root node missing"))?;
            return Ok(root.first_doc_id.zip(root.last_doc_id));
        }
        Ok(None)
    }

    /// Helper: collect scope terms
    async fn collect_scope_terms(
        &self,
        passages: &DataFusionStore,
        doc_table: &DocumentTable,
        range: Option<(u32, u32)>,
        canon: Option<&str>,
        period: Option<&str>,
        limit_passages: usize,
        gram_len: usize,
    ) -> Result<(FxHashMap<u64, u32>, FxHashMap<u64, String>, usize)> {
        let rows = if let Some((lo, hi)) = range {
            let passage_ids: Vec<String> = (lo..=hi)
                .filter_map(|did| doc_table.passage_id(did).map(String::from))
                .take(limit_passages.max(1))
                .collect();
            passages
                .passages_by_ids(
                    &passage_ids,
                    "passage_id, zh_text_normalized, canon, period",
                )
                .await?
        } else {
            let mut where_parts = vec!["zh_text_normalized IS NOT NULL".to_string()];
            if let Some(canon) = canon {
                where_parts.push(format!(
                    "canon = {}",
                    crate::datafusion_store::sql_literal(canon)
                ));
            }
            if let Some(period) = period {
                where_parts.push(format!(
                    "period = {}",
                    crate::datafusion_store::sql_literal(period)
                ));
            }
            passages.query_json(&format!(
                "SELECT passage_id, zh_text_normalized, canon, period FROM passages WHERE {} LIMIT {}",
                where_parts.join(" AND "),
                limit_passages.max(1),
            )).await?
        };

        let mut terms: FxHashMap<u64, u32> = FxHashMap::default();
        let mut term_display: FxHashMap<u64, String> = FxHashMap::default();
        let mut passage_count = 0usize;
        for row in &rows {
            if let Some(canon_filter) = canon {
                let c = row.get("canon").and_then(|v| v.as_str()).unwrap_or("");
                if c != canon_filter {
                    continue;
                }
            }
            if let Some(period_filter) = period {
                let p = row.get("period").and_then(|v| v.as_str()).unwrap_or("");
                if p != period_filter {
                    continue;
                }
            }
            let text = row
                .get("zh_text_normalized")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if text.is_empty() {
                continue;
            }
            passage_count += 1;
            count_unique_ngrams_with_terms(text, gram_len, &mut terms, &mut term_display);
        }
        Ok((terms, term_display, passage_count))
    }

    /// Implement the pair-profile tool: grouped co-occurrence statistics.
    pub async fn pair_profile_impl(
        &self,
        req: crate::tools::requests::PairProfileRequest,
    ) -> Result<crate::tools::responses::PairProfileResponse> {
        use crate::tools::responses::{PairAppearanceHit, PairProfileGroup, PairProfileResponse};

        let unit = match req.unit.as_str() {
            "passage" | "window" | "sentence" | "section" | "work" => req.unit.clone(),
            other => anyhow::bail!(
                "unsupported pair-profile unit '{other}'; use passage, window, sentence, section, or work"
            ),
        };

        let doc_table = self.doc_table().await?;
        let passages = self.passages().await?;
        let phrase_index_path = self.optional_phrase_path().await?;
        let catalog = self.catalog().await.ok();
        let canon = expand_optional_filter(req.scope_canon.as_deref());
        let period = expand_period_filter(req.scope_period.as_deref());
        let doc_range = if let Some(cat) = catalog.as_deref() {
            self.resolve_doc_range(cat, None, req.scope_source_work_id.as_deref())?
        } else {
            None
        };
        let canon_for_index = if canon.is_empty() {
            None
        } else {
            Some(canon.as_slice())
        };
        let period_for_index = if period.len() == 1 {
            Some(period[0].as_str())
        } else {
            None
        };

        let term1_variants = pair_term_variants(&req.term1, req.allow_variants);
        let term2_variants = pair_term_variants(&req.term2, req.allow_variants);
        let candidate_limit = req.max_candidates_per_term.max(1);

        let (term1_rows, term1_strategy) = self
            .pair_candidate_rows(
                &passages,
                &doc_table,
                phrase_index_path.as_deref(),
                &term1_variants,
                candidate_limit,
                doc_range,
                canon_for_index,
                period_for_index,
                req.scope_source_work_id.as_deref(),
            )
            .await?;
        let (term2_rows, term2_strategy) = self
            .pair_candidate_rows(
                &passages,
                &doc_table,
                phrase_index_path.as_deref(),
                &term2_variants,
                candidate_limit,
                doc_range,
                canon_for_index,
                period_for_index,
                req.scope_source_work_id.as_deref(),
            )
            .await?;

        let term2_by_pid: std::collections::BTreeMap<String, &serde_json::Value> = term2_rows
            .iter()
            .filter_map(|row| {
                row.get("passage_id")
                    .and_then(|v| v.as_str())
                    .map(|pid| (pid.to_string(), row))
            })
            .collect();

        // Collect pair hits with group label
        let group_field = match req.group_by.as_str() {
            "period" => "period",
            "canon" => "canon",
            "work" => "main_title",
            "author" => "author",
            other => {
                anyhow::bail!("unknown group_by '{other}'; use period, canon, work, or author")
            }
        };

        let mut group_term1: std::collections::BTreeMap<String, usize> = Default::default();
        let mut group_term2: std::collections::BTreeMap<String, usize> = Default::default();
        let mut group_pairs: std::collections::BTreeMap<String, Vec<PairAppearanceHit>> =
            Default::default();

        for row in &term1_rows {
            let label = opt_str(row, group_field).unwrap_or_default();
            *group_term1.entry(label).or_insert(0) += 1;
        }
        for row in &term2_rows {
            let label = opt_str(row, group_field).unwrap_or_default();
            *group_term2.entry(label).or_insert(0) += 1;
        }

        let mut total_pair_hits = 0usize;
        if matches!(unit.as_str(), "section" | "work") {
            let catalog_for_unit = if unit == "section" {
                Some(catalog.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("pair-profile unit=section requires catalog.index")
                })?)
            } else {
                catalog.as_deref()
            };
            let mut term1_by_unit: std::collections::BTreeMap<String, Vec<&serde_json::Value>> =
                Default::default();
            let mut term2_by_unit: std::collections::BTreeMap<String, Vec<&serde_json::Value>> =
                Default::default();
            for row in &term1_rows {
                if let Some(key) = pair_profile_unit_key(row, &unit, &doc_table, catalog_for_unit) {
                    term1_by_unit.entry(key).or_default().push(row);
                }
            }
            for row in &term2_rows {
                if let Some(key) = pair_profile_unit_key(row, &unit, &doc_table, catalog_for_unit) {
                    term2_by_unit.entry(key).or_default().push(row);
                }
            }
            for (unit_key, term1_unit_rows) in term1_by_unit {
                let Some(term2_unit_rows) = term2_by_unit.get(&unit_key) else {
                    continue;
                };
                let Some(row) = term1_unit_rows.first().copied() else {
                    continue;
                };
                let label = opt_str(row, group_field).unwrap_or_default();
                let norm = row
                    .get("zh_text_normalized")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let t1_off = find_offsets_for_terms(norm, &term1_variants);
                let same_passage_term2 = row
                    .get("passage_id")
                    .and_then(|v| v.as_str())
                    .and_then(|pid| term2_by_pid.get(pid));
                let t2_off = same_passage_term2
                    .map(|_| find_offsets_for_terms(norm, &term2_variants))
                    .unwrap_or_default();
                let distance = if t1_off.is_empty() || t2_off.is_empty() {
                    None
                } else {
                    min_pair_distance(&t1_off, &t2_off, false)
                };
                let pid = row.get("passage_id").and_then(|v| v.as_str()).unwrap_or("");
                let quote = if distance.is_some() {
                    pair_snippet(norm, &t1_off, &t2_off, req.window_chars.max(40))
                } else {
                    let sample2 = term2_unit_rows
                        .first()
                        .and_then(|r| r.get("passage_id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    format!("{unit} co-occurrence: term1 at {pid}; term2 at {sample2}")
                };
                group_pairs
                    .entry(label)
                    .or_default()
                    .push(PairAppearanceHit {
                        passage_id: pid.to_string(),
                        source_work_id: opt_str(row, "source_work_id"),
                        main_title: opt_str(row, "main_title"),
                        heading: opt_str(row, "heading"),
                        distance_chars: distance,
                        term1_offsets: t1_off,
                        term2_offsets: t2_off,
                        zh_quote: quote,
                    });
                total_pair_hits += 1;
            }
        } else {
            for row in &term1_rows {
                let Some(pid) = row.get("passage_id").and_then(|v| v.as_str()) else {
                    continue;
                };
                if !term2_by_pid.contains_key(pid) {
                    continue;
                }
                let norm = row
                    .get("zh_text_normalized")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let t1_off = find_offsets_for_terms(norm, &term1_variants);
                let t2_off = find_offsets_for_terms(norm, &term2_variants);
                if t1_off.is_empty() || t2_off.is_empty() {
                    continue;
                }
                let distance = min_pair_distance(&t1_off, &t2_off, false);
                let keep = match unit.as_str() {
                    "passage" => distance.is_some(),
                    "window" => distance.map(|d| d <= req.window_chars).unwrap_or(false),
                    "sentence" => has_sentence_pair(norm, &t1_off, &t2_off, false),
                    _ => false,
                };
                if !keep {
                    continue;
                }
                let label = opt_str(row, group_field).unwrap_or_default();
                let hit = PairAppearanceHit {
                    passage_id: pid.to_string(),
                    source_work_id: opt_str(row, "source_work_id"),
                    main_title: opt_str(row, "main_title"),
                    heading: opt_str(row, "heading"),
                    distance_chars: distance,
                    term1_offsets: t1_off,
                    term2_offsets: t2_off,
                    zh_quote: pair_snippet(norm, &[], &[], req.window_chars.max(40)),
                };
                group_pairs.entry(label).or_default().push(hit);
                total_pair_hits += 1;
            }
        }

        // Build groups, sorted by pair_count descending
        let mut all_labels: std::collections::BTreeSet<String> = Default::default();
        all_labels.extend(group_term1.keys().cloned());
        all_labels.extend(group_term2.keys().cloned());
        all_labels.extend(group_pairs.keys().cloned());

        let mut groups: Vec<PairProfileGroup> = all_labels
            .into_iter()
            .map(|label| {
                let t1 = *group_term1.get(&label).unwrap_or(&0);
                let t2 = *group_term2.get(&label).unwrap_or(&0);
                let pairs = group_pairs.get(&label).cloned().unwrap_or_default();
                let pc = pairs.len();
                let r1 = if t1 > 0 { pc as f64 / t1 as f64 } else { 0.0 };
                let r2 = if t2 > 0 { pc as f64 / t2 as f64 } else { 0.0 };
                let sample = pairs.into_iter().take(req.sample_hits_per_group).collect();
                PairProfileGroup {
                    group_label: label,
                    term1_count: t1,
                    term2_count: t2,
                    pair_count: pc,
                    pair_rate_given_term1: r1,
                    pair_rate_given_term2: r2,
                    representative_hits: sample,
                }
            })
            .collect();
        groups.sort_by(|a, b| b.pair_count.cmp(&a.pair_count));
        groups.truncate(req.limit_groups.max(1));

        Ok(PairProfileResponse {
            schema: "sinorag-pair-profile-v1",
            term1: req.term1,
            term2: req.term2,
            unit,
            group_by: req.group_by,
            total_term1_hits: term1_rows.len(),
            total_term2_hits: term2_rows.len(),
            total_pair_hits,
            groups,
            search_strategy: serde_json::json!({
                "term1": term1_strategy,
                "term2": term2_strategy,
                "max_candidates_per_term": candidate_limit,
                "supported_units": ["passage", "window", "sentence", "section", "work"],
            }),
        })
    }

    /// Implement the person-resolve tool.
    pub async fn person_resolve_impl(
        &self,
        req: crate::tools::requests::PersonResolveRequest,
    ) -> Result<crate::tools::responses::PersonResolveResponse> {
        use crate::research::{evidence_from_row, exact_phrase_rows, SearchSpec};
        use crate::tools::responses::PersonResolveResponse;

        let passages = self.passages().await?;
        let mut forms = vec![req.name.clone()];
        for alias in &req.aliases {
            if !forms.iter().any(|v| v == alias) {
                forms.push(alias.clone());
            }
        }

        // Query DDBC authority parquet for structured data
        let authority = query_person_authority(&req.name, &req.aliases).await;

        let mut name_forms = Vec::new();
        let mut evidence = Vec::new();
        for form in &forms {
            let spec = SearchSpec::exact_phrase(form.clone(), 50);
            let rows = exact_phrase_rows(&passages, &spec).await?;
            if let Some(first) = rows.first() {
                evidence.push(evidence_from_row(first, form, "name_form_sample"));
            }
            name_forms.push(serde_json::json!({
                "form": form,
                "normalized": spec.normalized,
                "hit_count_sample": rows.len(),
                "first_hit": rows.first().cloned().unwrap_or(serde_json::Value::Null),
                "ambiguity": if form.chars().count() <= 1 { "high" } else { "unknown" }
            }));
        }

        let mut caveats = vec!["Supply all known aliases explicitly.".to_string()];
        if authority.is_none() {
            caveats.push(
                "Name not found in DDBC authority database; corpus-local search only.".to_string(),
            );
        }

        Ok(PersonResolveResponse {
            schema: "sinorag-person-resolve-v2",
            name: req.name.clone(),
            aliases: req.aliases,
            canonical_candidate: forms.first().cloned().unwrap_or_default(),
            authority,
            name_forms,
            ambiguity_notes: vec![
                "Short aliases and honorific titles may refer to more than one person.".to_string(),
                "Use person-history to inspect earliest and contextualised mentions.".to_string(),
            ],
            evidence,
            caveats,
            suggested_next: vec![
                "Run person-history with the same aliases to classify mention contexts."
                    .to_string(),
            ],
        })
    }

    /// Implement the place-resolve tool.
    pub async fn place_resolve_impl(
        &self,
        req: crate::tools::requests::PlaceResolveRequest,
    ) -> Result<crate::tools::responses::PlaceResolveResponse> {
        use crate::research::{evidence_from_row, exact_phrase_rows, SearchSpec};
        use crate::tools::responses::PlaceResolveResponse;

        let passages = self.passages().await?;
        let mut forms = vec![req.name.clone()];
        for alias in &req.aliases {
            if !forms.iter().any(|v| v == alias) {
                forms.push(alias.clone());
            }
        }

        // Query DDBC authority parquet for structured place data
        let authority = query_place_authority(&req.name, &req.aliases).await;

        let mut name_forms = Vec::new();
        let mut evidence = Vec::new();
        for form in &forms {
            let spec = SearchSpec::exact_phrase(form.clone(), 50);
            let rows = exact_phrase_rows(&passages, &spec).await?;
            if let Some(first) = rows.first() {
                evidence.push(evidence_from_row(first, form, "place_mention_sample"));
            }
            name_forms.push(serde_json::json!({
                "form": form,
                "normalized": spec.normalized,
                "hit_count_sample": rows.len(),
                "first_hit": rows.first().cloned().unwrap_or(serde_json::Value::Null),
            }));
        }

        let mut caveats = vec!["Supply all known alternate names explicitly.".to_string()];
        if authority.is_none() {
            caveats.push(
                "Place not found in DDBC authority database; corpus-local search only.".to_string(),
            );
        }

        Ok(PlaceResolveResponse {
            schema: "sinorag-place-resolve-v1",
            name: req.name.clone(),
            aliases: req.aliases,
            authority,
            name_forms,
            ambiguity_notes: vec![
                "Place names may refer to multiple locations across different periods.".to_string(),
            ],
            evidence,
            caveats,
            suggested_next: vec!["Search corpus passages for contextual mentions.".to_string()],
        })
    }

    /// Implement the person-history tool.
    pub async fn person_history_impl(
        &self,
        req: crate::tools::requests::PersonHistoryRequest,
    ) -> Result<crate::tools::responses::PersonHistoryResponse> {
        use crate::research::{evidence_from_row, exact_phrase_rows, field_str, SearchSpec};
        use crate::tools::responses::PersonHistoryResponse;
        use std::collections::btree_map::Entry;
        use std::collections::BTreeMap;

        let passages = self.passages().await?;
        let limit = req.limit.max(1);
        let mut forms = vec![req.name.clone()];
        for alias in &req.aliases {
            if !forms.iter().any(|v| v == alias) {
                forms.push(alias.clone());
            }
        }
        let canonical_candidate = forms.first().cloned().unwrap_or_default();

        let mut by_passage: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        for (idx, form) in forms.iter().enumerate() {
            let rows = exact_phrase_rows(&passages, &SearchSpec::exact_phrase(form.clone(), limit))
                .await?;
            for mut row in rows {
                let passage_id = field_str(&row, "passage_id");
                let mention_class = classify_person_mention(&row, form);
                if let Some(obj) = row.as_object_mut() {
                    obj.insert("matched_name_form".to_string(), serde_json::json!(form));
                    obj.insert("matched_name_forms".to_string(), serde_json::json!([form]));
                    obj.insert("is_primary_name".to_string(), serde_json::json!(idx == 0));
                    obj.insert(
                        "mention_class".to_string(),
                        serde_json::json!(mention_class),
                    );
                    obj.insert(
                        "ambiguity".to_string(),
                        serde_json::json!(if idx == 0 {
                            "unambiguous_candidate"
                        } else {
                            "alias_candidate"
                        }),
                    );
                }
                match by_passage.entry(passage_id) {
                    Entry::Vacant(e) => {
                        e.insert(row);
                    }
                    Entry::Occupied(mut e) => {
                        merge_person_mention(e.get_mut(), form, idx == 0);
                    }
                }
            }
        }

        let mut mentions: Vec<serde_json::Value> = by_passage.into_values().collect();
        mentions.sort_by_key(|row| {
            (
                row.get("period_rank")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(99),
                field_str(row, "source_rel_path"),
                field_str(row, "from_lb"),
                field_str(row, "xml_id"),
            )
        });
        mentions.truncate(limit);

        // Fix: earliest_unambiguous must skip hits where contains_person=false
        let earliest_unambiguous = mentions
            .iter()
            .find(|row| {
                row.get("is_primary_name").and_then(|v| v.as_bool()) == Some(true)
                    && row.get("contains_person").and_then(|v| v.as_bool()) != Some(false)
            })
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        let ambiguous_earlier_hits = mentions
            .iter()
            .filter(|row| row.get("is_primary_name").and_then(|v| v.as_bool()) != Some(true))
            .take(5)
            .cloned()
            .collect::<Vec<_>>();
        let evidence = mentions
            .iter()
            .take(12)
            .map(|row| {
                let form = row
                    .get("matched_name_form")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&req.name);
                evidence_from_row(row, form, "person_mention")
            })
            .collect::<Vec<_>>();

        // Short-name false-positive warning (≤2 chars likely a common compound)
        let name_len = req.name.chars().count();
        let false_positive_warning = if name_len <= 2 {
            Some(format!(
                "Name '{}' is {} char(s) and may be a common Chinese compound (not a personal name) in pre-Song texts. \
                 Recommend: pass compact=true first to gauge hit counts, then scope with scope_period=[\"Song\",\"Yuan\",\"Ming\",\"Qing\"] \
                 to reduce false positives.",
                req.name, name_len
            ))
        } else {
            None
        };

        // Add caveat when earliest_unambiguous is null due to contains_person filter
        let mut caveats = vec![
            "Rule-based mention classes are triage labels, not accepted historical claims."
                .to_string(),
            "Alias hits may refer to more than one person.".to_string(),
        ];
        if earliest_unambiguous.is_null()
            && mentions
                .iter()
                .any(|r| r.get("is_primary_name").and_then(|v| v.as_bool()) == Some(true))
        {
            caveats.push(
                "earliest_unambiguous is null: all primary-name hits have contains_person=false \
                 (compound word matches, not a personal name)."
                    .to_string(),
            );
        }

        // Compact mode: replace mentions with grouped summary
        let total_mentions = mentions.len();
        let (compact_summary, output_mentions) = if req.compact {
            let summary = build_person_compact_summary(&mentions);
            (Some(summary), vec![])
        } else {
            (None, mentions)
        };

        Ok(PersonHistoryResponse {
            schema: "sinorag-person-history-v1",
            name: req.name,
            aliases: req.aliases,
            aliases_searched: forms,
            canonical_candidate,
            total_mentions,
            limit,
            mentions: output_mentions,
            compact_summary,
            earliest_unambiguous,
            ambiguous_earlier_hits,
            evidence,
            caveats,
            false_positive_warning,
            suggested_next: vec![
                "Review earliest hits with source-read before asserting a first mention."
                    .to_string(),
                "Use pair-appearance to check if this person is mentioned with another figure."
                    .to_string(),
            ],
        })
    }

    /// Implement the person-profile tool.
    pub async fn person_profile_impl(
        &self,
        req: crate::tools::requests::PersonProfileRequest,
    ) -> Result<crate::tools::responses::PersonProfileResponse> {
        use crate::tools::requests::PersonHistoryRequest;
        use crate::tools::responses::PersonProfileResponse;

        // Look up DDBC authority record (already queries birth/death/teachers/students/bio)
        let authority = query_person_authority(&req.name, &req.aliases).await;

        // Assess false-positive risk
        let name_len = req.name.chars().count();
        let (false_positive_risk, false_positive_warning) = if name_len <= 2 {
            let warning = format!(
                "Name '{}' is {} char(s) and likely a common Chinese compound in pre-Song texts. \
                 Corpus hits may be overwhelmingly false positives (not this person). \
                 Scope to Song or later periods for reliable results.",
                req.name, name_len
            );
            ("high".to_string(), Some(warning))
        } else if authority.is_some() {
            ("none".to_string(), None)
        } else {
            ("low".to_string(), None)
        };

        // Determine status
        let status = if authority.is_some() {
            "fully_attested"
        } else if name_len <= 2 {
            "needs_filtered_search"
        } else {
            "partially_attested"
        }
        .to_string();

        // Extract known aliases from authority or fall back to req.aliases
        let known_aliases = if let Some(auth) = &authority {
            auth.get("alt_names")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| req.aliases.clone())
        } else {
            req.aliases.clone()
        };

        // Get compact corpus summary via person-history in compact mode
        let history_req = PersonHistoryRequest {
            name: req.name.clone(),
            aliases: req.aliases.clone(),
            limit: 500,
            compact: true,
        };
        let corpus_summary = match self.person_history_impl(history_req).await {
            Ok(hist) => hist.compact_summary,
            Err(_) => None,
        };

        let mut caveats = Vec::new();
        if authority.is_none() {
            caveats.push(
                "Name not found in DDBC authority database; authority fields will be absent."
                    .to_string(),
            );
        }
        if false_positive_risk == "high" {
            caveats.push(
                "Use person-history with compact=true and scoped periods before drawing conclusions."
                    .to_string(),
            );
        }

        Ok(PersonProfileResponse {
            schema: "sinorag-person-profile-v1",
            name: req.name,
            status,
            false_positive_risk,
            false_positive_warning,
            authority,
            known_aliases,
            corpus_summary,
            caveats,
        })
    }

    /// Implement the citation-verify tool.
    pub async fn citation_verify_impl(
        &self,
        req: crate::tools::requests::CitationVerifyRequest,
    ) -> Result<crate::tools::responses::CitationVerifyResponse> {
        use crate::research_tools::phrase::phrase_rows_with_explicit_doc_table;
        use crate::tools::responses::{CitationNearMatch, CitationVerifyResponse};

        let doc_table = self.doc_table().await?;
        let passages = self.passages().await?;
        let phrase_index_path = self.optional_phrase_path().await?;
        let catalog = self.catalog().await.ok();
        let canon = expand_optional_filter(req.scope_canon.as_deref());
        let doc_range = if let Some(cat) = catalog.as_deref() {
            self.resolve_doc_range(cat, req.scope_node_id, req.scope_source_work_id.as_deref())?
        } else {
            None
        };
        let canon_for_index = if canon.is_empty() {
            None
        } else {
            Some(canon.as_slice())
        };

        let (exact_hits, strategy) = phrase_rows_with_explicit_doc_table(
            &passages,
            &doc_table,
            phrase_index_path.as_deref(),
            &req.quote,
            req.limit,
            doc_range,
            canon_for_index,
            None,
        )
        .await?;

        let verified = !exact_hits.is_empty();
        let exact_hit_count = exact_hits.len();

        // Near-match using TF-IDF query_top_k when a v4 index is available,
        // falling back to 4-gram substring search for v3 or missing indexes.
        let mut near_matches: Vec<CitationNearMatch> = Vec::new();
        if !verified && req.include_near_matches {
            use crate::normalize::normalize_zh;
            let normalized = normalize_zh(&req.quote);
            let quote_forms = crate::normalize::PhraseForms::new(&req.quote);

            let tfidf_candidates: Vec<(u32, f32)> = if let Ok(tfidf) = self.tfidf().await {
                let k = req.near_match_limit * 5;
                let mut found = tfidf.query_top_k(&normalized, k);
                if !quote_forms.is_single() {
                    // Word-layout passages are scored on word features, which
                    // only the word form of the quote produces.
                    for (doc, score) in tfidf.query_top_k(&quote_forms.words, k) {
                        match found.iter_mut().find(|(d, _)| *d == doc) {
                            Some(entry) => entry.1 = entry.1.max(score),
                            None => found.push((doc, score)),
                        }
                    }
                    found.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    found.truncate(k.max(1));
                }
                found
            } else {
                Vec::new()
            };

            if !tfidf_candidates.is_empty() {
                // TF-IDF path: fetch passage text for each candidate, score by
                // character overlap with the normalized quote, return top results.
                let pids: Vec<String> = tfidf_candidates
                    .iter()
                    .filter_map(|(doc_id, _)| {
                        if let Some(range) = doc_range {
                            if *doc_id < range.0 || *doc_id >= range.1 {
                                return None;
                            }
                        }
                        doc_table.passage_id(*doc_id).map(|s| s.to_string())
                    })
                    .collect();
                let rows = passages
                    .passages_by_ids(
                        &pids,
                        "passage_id, source_work_id, main_title, heading, zh_text_normalized",
                    )
                    .await
                    .unwrap_or_default();
                let mut scored: Vec<(f64, String, serde_json::Value)> = rows
                    .into_iter()
                    .filter_map(|row| {
                        let pid = row
                            .get("passage_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if pid.is_empty() {
                            return None;
                        }
                        let text = row
                            .get("zh_text_normalized")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let overlap = character_overlap_score(quote_forms.form_for(text), text);
                        Some((overlap, pid, row))
                    })
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                for (overlap, pid, row) in scored.into_iter().take(req.near_match_limit) {
                    let text = row
                        .get("zh_text_normalized")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    near_matches.push(CitationNearMatch {
                        passage_id: pid,
                        source_work_id: opt_str(&row, "source_work_id"),
                        main_title: opt_str(&row, "main_title"),
                        heading: opt_str(&row, "heading"),
                        overlap_score: overlap,
                        zh_quote: text.chars().take(200).collect(),
                    });
                }
            } else {
                // Fallback: 4-gram substring search when no TF-IDF index is available.
                let chars: Vec<char> = normalized.chars().collect();
                let gram_len = 4usize;
                let mut candidates: std::collections::BTreeMap<String, Vec<serde_json::Value>> =
                    Default::default();
                for start in (0..chars.len()).step_by(gram_len).take(5) {
                    let end = (start + gram_len).min(chars.len());
                    if end - start < 2 {
                        break;
                    }
                    let sub: String = chars[start..end].iter().collect();
                    let (sub_hits, _) = phrase_rows_with_explicit_doc_table(
                        &passages,
                        &doc_table,
                        phrase_index_path.as_deref(),
                        &sub,
                        req.near_match_limit * 3,
                        doc_range,
                        canon_for_index,
                        None,
                    )
                    .await
                    .unwrap_or_default();
                    for row in sub_hits {
                        let pid = row
                            .get("passage_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if !pid.is_empty() {
                            candidates.entry(pid).or_default().push(row);
                        }
                    }
                }
                let mut scored: Vec<(f64, String, serde_json::Value)> = candidates
                    .into_iter()
                    .filter_map(|(pid, rows)| {
                        let row = rows.into_iter().next()?;
                        let text = row
                            .get("zh_text_normalized")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let overlap = character_overlap_score(quote_forms.form_for(text), text);
                        Some((overlap, pid, row))
                    })
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                for (overlap, pid, row) in scored.into_iter().take(req.near_match_limit) {
                    let text = row
                        .get("zh_text_normalized")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    near_matches.push(CitationNearMatch {
                        passage_id: pid,
                        source_work_id: opt_str(&row, "source_work_id"),
                        main_title: opt_str(&row, "main_title"),
                        heading: opt_str(&row, "heading"),
                        overlap_score: overlap,
                        zh_quote: text.chars().take(200).collect(),
                    });
                }
            }
        }

        let verdict = if verified {
            format!(
                "VERIFIED — found {} exact occurrence(s) in the corpus",
                exact_hit_count
            )
        } else if !near_matches.is_empty() {
            format!(
                "NOT VERIFIED — quote not found exactly in the scoped corpus; {} near-match(es) found",
                near_matches.len()
            )
        } else {
            "NOT VERIFIED — no exact or near matches found in the scoped corpus".to_string()
        };

        let exact_hits_json: Vec<serde_json::Value> =
            exact_hits.into_iter().take(req.limit).collect();

        Ok(CitationVerifyResponse {
            schema: "sinorag-citation-verify-v1",
            quote: req.quote,
            claimed_attribution: req.claimed_attribution,
            scope_source_work_id: req.scope_source_work_id,
            scope_canon: req.scope_canon,
            verified,
            exact_hit_count,
            exact_hits: exact_hits_json,
            near_matches,
            verdict,
            search_strategy: strategy,
        })
    }

    pub async fn run_batch_impl(
        &self,
        req: crate::tools::requests::RunBatchRequest,
    ) -> Result<crate::tools::responses::RunBatchResponse> {
        use crate::tools::batch::{
            run_one_job_ref, skipped_dependency_envelope, unresolved_dependency_envelope, BatchJob,
        };
        use crate::tools::registry::ToolCallEnvelope;
        use std::collections::HashMap;
        use std::io::Write;

        self.ensure_write_allowed("run-batch", &req.out)?;

        let jobs: Vec<BatchJob> = match (req.jobs, req.input_file) {
            (Some(jobs), None) => jobs,
            (None, Some(ref path)) => {
                let text = std::fs::read_to_string(path).map_err(|e| {
                    anyhow::anyhow!("cannot read input_file {}: {}", path.display(), e)
                })?;
                let mut out = Vec::new();
                for (i, line) in text.lines().enumerate() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let job: BatchJob = serde_json::from_str(line)
                        .map_err(|e| anyhow::anyhow!("invalid JSON at line {}: {}", i + 1, e))?;
                    out.push(job);
                }
                out
            }
            (Some(_), Some(_)) => {
                anyhow::bail!("specify `jobs` or `input_file`, not both")
            }
            (None, None) => anyhow::bail!("`jobs` or `input_file` is required"),
        };

        let jobs_total = jobs.len();
        if req.concurrency > 1 {
            tracing::debug!(
                "run-batch: concurrency={} requested; running sequentially (use \
                 `sinorag run-tools --jobs {}` for true parallel execution)",
                req.concurrency,
                req.concurrency
            );
        }
        if let Some(parent) = req.out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(&req.out)?;
        let mut writer = std::io::BufWriter::new(file);
        let started = std::time::Instant::now();
        let mut jobs_ok = 0usize;
        let mut jobs_failed = 0usize;

        let write_env = |writer: &mut std::io::BufWriter<std::fs::File>,
                         env: &ToolCallEnvelope|
         -> Result<()> {
            writeln!(writer, "{}", serde_json::to_string(env)?)?;
            writer.flush()?;
            Ok(())
        };

        if jobs.iter().any(|j| !j.depends_on.is_empty()) {
            let mut completed: HashMap<String, bool> = HashMap::new();
            let mut pending = jobs;

            loop {
                if pending.is_empty() {
                    break;
                }
                let mut made_progress = false;
                let mut next_pending = Vec::new();

                for job in pending {
                    if let Some(dep) = job
                        .depends_on
                        .iter()
                        .find(|d| completed.get(*d) == Some(&false))
                    {
                        let env = skipped_dependency_envelope(&job, dep);
                        if let Some(id) = &job.id {
                            completed.insert(id.clone(), false);
                        }
                        jobs_failed += 1;
                        made_progress = true;
                        write_env(&mut writer, &env)?;
                        if !job.continue_on_error.unwrap_or(req.continue_on_error) {
                            return Ok(crate::tools::responses::RunBatchResponse {
                                out: req.out,
                                jobs_total,
                                jobs_ok,
                                jobs_failed,
                                elapsed_ms: started.elapsed().as_millis(),
                            });
                        }
                        continue;
                    }

                    if !job.depends_on.iter().all(|d| completed.contains_key(d)) {
                        next_pending.push(job);
                        continue;
                    }

                    let env = run_one_job_ref(self, &job).await;
                    let ok = env.ok;
                    if ok {
                        jobs_ok += 1;
                    } else {
                        jobs_failed += 1;
                    }
                    if let Some(id) = &job.id {
                        completed.insert(id.clone(), ok);
                    }
                    made_progress = true;
                    write_env(&mut writer, &env)?;
                    if !ok && !job.continue_on_error.unwrap_or(req.continue_on_error) {
                        return Ok(crate::tools::responses::RunBatchResponse {
                            out: req.out,
                            jobs_total,
                            jobs_ok,
                            jobs_failed,
                            elapsed_ms: started.elapsed().as_millis(),
                        });
                    }
                }

                if !made_progress {
                    for job in next_pending {
                        let missing: Vec<String> = job
                            .depends_on
                            .iter()
                            .filter(|d| !completed.contains_key(*d))
                            .cloned()
                            .collect();
                        let env = unresolved_dependency_envelope(&job, missing);
                        jobs_failed += 1;
                        write_env(&mut writer, &env)?;
                        if !job.continue_on_error.unwrap_or(req.continue_on_error) {
                            break;
                        }
                    }
                    break;
                }

                pending = next_pending;
            }
        } else {
            for job in &jobs {
                let env = run_one_job_ref(self, job).await;
                let ok = env.ok;
                if ok {
                    jobs_ok += 1;
                } else {
                    jobs_failed += 1;
                }
                write_env(&mut writer, &env)?;
                if !ok && !job.continue_on_error.unwrap_or(req.continue_on_error) {
                    break;
                }
            }
        }

        Ok(crate::tools::responses::RunBatchResponse {
            out: req.out,
            jobs_total,
            jobs_ok,
            jobs_failed,
            elapsed_ms: started.elapsed().as_millis(),
        })
    }
}

/// Query persons.parquet for the best-matching DDBC authority record.
/// Tries exact match on primary_name, then scans alt_names_json for any alias match.
async fn query_person_authority(name: &str, aliases: &[String]) -> Option<serde_json::Value> {
    use datafusion::prelude::*;

    let path_opt = crate::dict::get_person_path();
    let dir = path_opt.as_deref()?;
    if !dir.is_dir() {
        return None;
    }

    let ctx = SessionContext::new();
    let src = dir
        .join("**/*.parquet")
        .to_string_lossy()
        .replace('\\', "/");
    ctx.register_parquet("persons", &src, ParquetReadOptions::default())
        .await
        .ok()?;

    // Try all forms: primary name + aliases
    let mut all_forms: Vec<&str> = vec![name];
    for a in aliases {
        all_forms.push(a.as_str());
    }

    for form in all_forms {
        let escaped = form.replace('\'', "''");
        let sql = format!(
            "SELECT person_id, primary_name, alt_names_json, gender, dynasty, \
             birth_year, death_year, occupation, place_of_origin, concise_bio, \
             teachers_json, students_json, wikidata_id, cbdb_id \
             FROM persons \
             WHERE primary_name = '{escaped}' OR alt_names_json LIKE '%\"{escaped}\"%' \
             LIMIT 1"
        );
        if let Ok(df) = ctx.sql(&sql).await {
            if let Ok(batches) = df.collect().await {
                for batch in &batches {
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    let row = authority_person_row(batch, 0);
                    if row.is_some() {
                        return row;
                    }
                }
            }
        }
    }
    None
}

fn authority_person_row(
    batch: &arrow::record_batch::RecordBatch,
    i: usize,
) -> Option<serde_json::Value> {
    #[allow(unused_imports)]
    use arrow::array::{Array, StringArray};
    let get = |name: &str| -> Option<String> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .and_then(|a| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            })
    };

    let person_id = get("person_id")?;
    let primary_name = get("primary_name")?;
    let mut obj = serde_json::json!({
        "person_id": person_id,
        "primary_name": primary_name,
    });
    let m = obj.as_object_mut().unwrap();
    if let Some(v) = get("gender") {
        m.insert("gender".into(), v.into());
    }
    if let Some(v) = get("dynasty") {
        m.insert("dynasty".into(), v.trim().to_string().into());
    }
    if let Some(v) = get("birth_year") {
        m.insert("birth_year".into(), v.into());
    }
    if let Some(v) = get("death_year") {
        m.insert("death_year".into(), v.into());
    }
    if let Some(v) = get("occupation") {
        m.insert("occupation".into(), v.into());
    }
    if let Some(v) = get("place_of_origin") {
        m.insert("place_of_origin".into(), v.into());
    }
    if let Some(v) = get("concise_bio") {
        m.insert("concise_bio".into(), v.into());
    }
    if let Some(v) = get("wikidata_id") {
        m.insert("wikidata_id".into(), v.into());
    }
    if let Some(v) = get("cbdb_id") {
        m.insert("cbdb_id".into(), v.into());
    }
    // Parse teacher/student JSON arrays
    if let Some(v) = get("teachers_json") {
        if let Ok(arr) = serde_json::from_str::<serde_json::Value>(&v) {
            m.insert("teachers".into(), arr);
        }
    }
    if let Some(v) = get("students_json") {
        if let Ok(arr) = serde_json::from_str::<serde_json::Value>(&v) {
            m.insert("students".into(), arr);
        }
    }
    if let Some(v) = get("alt_names_json") {
        if let Ok(arr) = serde_json::from_str::<serde_json::Value>(&v) {
            m.insert("alt_names".into(), arr);
        }
    }
    Some(obj)
}

/// Query places.parquet for the best-matching DDBC authority record.
async fn query_place_authority(name: &str, aliases: &[String]) -> Option<serde_json::Value> {
    #[allow(unused_imports)]
    use arrow::array::{Array, Float64Array, StringArray};
    use datafusion::prelude::*;

    let path_opt = crate::dict::get_place_path();
    let dir = path_opt.as_deref()?;
    if !dir.is_dir() {
        return None;
    }

    let ctx = SessionContext::new();
    let src = dir
        .join("**/*.parquet")
        .to_string_lossy()
        .replace('\\', "/");
    ctx.register_parquet("places", &src, ParquetReadOptions::default())
        .await
        .ok()?;

    let mut all_forms: Vec<&str> = vec![name];
    for a in aliases {
        all_forms.push(a.as_str());
    }

    for form in all_forms {
        let escaped = form.replace('\'', "''");
        let sql = format!(
            "SELECT place_id, primary_name, alt_names_json, latitude, longitude, \
             geo_confidence, district, category, description, parent_place_id \
             FROM places \
             WHERE primary_name = '{escaped}' OR alt_names_json LIKE '%\"{escaped}\"%' \
             LIMIT 1"
        );
        if let Ok(df) = ctx.sql(&sql).await {
            if let Ok(batches) = df.collect().await {
                for batch in &batches {
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    let get_str = |col: &str| -> Option<String> {
                        batch
                            .column_by_name(col)
                            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                            .and_then(|a| {
                                if a.is_null(0) {
                                    None
                                } else {
                                    Some(a.value(0).to_string())
                                }
                            })
                    };
                    let get_f64 = |col: &str| -> Option<f64> {
                        batch
                            .column_by_name(col)
                            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
                            .and_then(|a| if a.is_null(0) { None } else { Some(a.value(0)) })
                    };

                    let place_id = get_str("place_id")?;
                    let primary_name = get_str("primary_name")?;
                    let mut obj = serde_json::json!({
                        "place_id": place_id,
                        "primary_name": primary_name,
                    });
                    let m = obj.as_object_mut().unwrap();
                    if let Some(lat) = get_f64("latitude") {
                        m.insert("latitude".into(), lat.into());
                    }
                    if let Some(lon) = get_f64("longitude") {
                        m.insert("longitude".into(), lon.into());
                    }
                    if let Some(v) = get_str("geo_confidence") {
                        m.insert("geo_confidence".into(), v.into());
                    }
                    if let Some(v) = get_str("district") {
                        m.insert("district".into(), v.into());
                    }
                    if let Some(v) = get_str("category") {
                        m.insert("category".into(), v.trim().to_string().into());
                    }
                    if let Some(v) = get_str("description") {
                        m.insert("description".into(), v.into());
                    }
                    if let Some(v) = get_str("parent_place_id") {
                        m.insert("parent_place_id".into(), v.into());
                    }
                    if let Some(v) = get_str("alt_names_json") {
                        if let Ok(arr) = serde_json::from_str::<serde_json::Value>(&v) {
                            m.insert("alt_names".into(), arr);
                        }
                    }
                    return Some(obj);
                }
            }
        }
    }
    None
}

fn classify_person_mention(row: &serde_json::Value, form: &str) -> &'static str {
    let text = row
        .get("zh_text_raw")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if text.contains("嗣")
        || text.contains("法嗣")
        || text.contains("傳法")
        || text.contains("弟子")
    {
        "lineage_relation"
    } else if text.contains(form)
        && (text.contains("云") || text.contains("曰") || text.contains("示"))
    {
        "attributed_saying"
    } else if row.get("text_type").and_then(|v| v.as_str()) == Some("dialogue") {
        "case_appearance"
    } else if text.contains("頌") || text.contains("評") || text.contains("拈") {
        "commentarial_reference"
    } else {
        "name_mention"
    }
}

/// Build a compact grouped summary of person mentions for compact=true mode.
/// Groups by mention_class × period, returning counts and up to `samples_per_group` passage IDs.
fn build_person_compact_summary(mentions: &[serde_json::Value]) -> serde_json::Value {
    use std::collections::BTreeMap;
    // by_class_period[class][period] = vec of passage_ids
    let mut by_class_period: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
    for row in mentions {
        let class = row
            .get("mention_class")
            .and_then(|v| v.as_str())
            .unwrap_or("name_mention")
            .to_string();
        let period = row
            .get("period")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();
        let pid = row
            .get("passage_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        by_class_period
            .entry(class)
            .or_default()
            .entry(period)
            .or_default()
            .push(pid);
    }

    let mut result = serde_json::Map::new();
    for (class, periods) in &by_class_period {
        let mut period_map = serde_json::Map::new();
        for (period, ids) in periods {
            let samples: Vec<&String> = ids.iter().take(3).collect();
            period_map.insert(
                period.clone(),
                serde_json::json!({
                    "count": ids.len(),
                    "samples": samples,
                }),
            );
        }
        result.insert(class.clone(), serde_json::Value::Object(period_map));
    }

    serde_json::json!({
        "total_mentions": mentions.len(),
        "by_class_period": result,
    })
}

fn merge_person_mention(row: &mut serde_json::Value, form: &str, is_primary: bool) {
    let Some(obj) = row.as_object_mut() else {
        return;
    };
    if is_primary {
        obj.insert("is_primary_name".to_string(), serde_json::json!(true));
        obj.insert("matched_name_form".to_string(), serde_json::json!(form));
        obj.insert(
            "ambiguity".to_string(),
            serde_json::json!("unambiguous_candidate"),
        );
    }
    let entry = obj
        .entry("matched_name_forms".to_string())
        .or_insert_with(|| serde_json::json!([]));
    if let Some(forms) = entry.as_array_mut() {
        if !forms.iter().any(|v| v.as_str() == Some(form)) {
            forms.push(serde_json::json!(form));
        }
    }
}

/// How much of `query` a passage covers, 0.0 to 1.0. `query` is the quote in
/// the passage's own layout (`PhraseForms::form_for`). Word-layout text is
/// compared word by word: nearly every English passage holds every letter of
/// the alphabet, so a character comparison would score them all 1.0.
fn character_overlap_score(query: &str, text: &str) -> f64 {
    if query.is_empty() || text.is_empty() {
        return 0.0;
    }
    if crate::normalize::has_word_layout(text) {
        let words = |s: &'_ str| -> std::collections::BTreeSet<String> {
            s.split(' ')
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect()
        };
        let query_words = words(query);
        if query_words.is_empty() {
            return 0.0;
        }
        let text_words = words(text);
        let shared = query_words.intersection(&text_words).count();
        return shared as f64 / query_words.len() as f64;
    }
    let query_chars: std::collections::BTreeSet<char> = query.chars().collect();
    let text_chars: std::collections::BTreeSet<char> = text.chars().collect();
    let intersection = query_chars.intersection(&text_chars).count();
    intersection as f64 / query_chars.len() as f64
}
