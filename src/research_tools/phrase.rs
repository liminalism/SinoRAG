use crate::datafusion_store::DataFusionStore;
use crate::document_table::DocumentTable;
use crate::normalize::PhraseForms;
use crate::phrase_index::PhraseIndex;
use crate::research::{exact_phrase_rows, SearchSpec};
use anyhow::Result;
use serde_json::{json, Value};
use std::path::Path;

pub async fn phrase_rows_with_explicit_doc_table(
    store: &DataFusionStore,
    doc_table: &DocumentTable,
    phrase_index_path: Option<&Path>,
    phrase: &str,
    limit: usize,
    doc_range: Option<(u32, u32)>,
    canon: Option<&[String]>,
    period: Option<&str>,
) -> Result<(Vec<Value>, Value)> {
    let forms = PhraseForms::new(phrase);
    phrase_rows_for_forms(
        store,
        doc_table,
        phrase_index_path,
        &forms,
        limit,
        doc_range,
        canon,
        period,
    )
    .await
}

/// [`phrase_rows_with_explicit_doc_table`] for phrase forms built by the
/// caller (e.g. `PhraseForms::from_normalized_variant`).
pub async fn phrase_rows_for_forms(
    store: &DataFusionStore,
    doc_table: &DocumentTable,
    phrase_index_path: Option<&Path>,
    forms: &PhraseForms,
    limit: usize,
    doc_range: Option<(u32, u32)>,
    canon: Option<&[String]>,
    period: Option<&str>,
) -> Result<(Vec<Value>, Value)> {
    let limit = limit.max(1);

    if let Some(index_path) = phrase_index_path {
        if index_path.is_file() && !forms.is_empty() {
            let index = PhraseIndex::load(index_path)?;
            let gram_len = index.gram_len();
            if forms.indexed_len() >= gram_len {
                let result = index.candidate_ids_for_forms_streaming(forms);
                let (candidate_doc_ids, candidate_stats) = match result {
                    Some(r) => (r.doc_ids, Some(json!(r.stats))),
                    None => {
                        if let Some((lo, hi)) = doc_range {
                            let (rows, mut strategy) = doc_range_rows(
                                store, doc_table, forms, limit, lo, hi, canon, period,
                            )
                            .await?;
                            strategy["phrase_index_path"] = json!(index_path.display().to_string());
                            strategy["fallback"] = json!(true);
                            strategy["fallback_reason"] = json!("phrase_index_lookup_failed");
                            strategy["gram_size"] = json!(gram_len);
                            return Ok((rows, strategy));
                        }
                        let rows = parquet_global_rows(store, forms, limit, canon, period).await?;
                        return Ok((
                            rows,
                            json!({
                                "used_phrase_index": false,
                                "phrase_index_path": index_path.display().to_string(),
                                "fallback": true,
                                "fallback_reason": "phrase_index_lookup_failed",
                                "scope_scan": "parquet_global",
                                "gram_size": gram_len,
                                "limit": limit,
                            }),
                        ));
                    }
                };

                let scoped_doc_ids: Vec<u32> = candidate_doc_ids
                    .into_iter()
                    .filter(|did| {
                        if let Some((lo, hi)) = doc_range {
                            *did >= lo && *did <= hi
                        } else {
                            true
                        }
                    })
                    .collect();

                let passage_ids: Vec<String> = scoped_doc_ids
                    .iter()
                    .filter_map(|&did| doc_table.passage_id(did).map(String::from))
                    .collect();

                let mut rows = store.passages_by_ids(&passage_ids, PASSAGE_SELECT).await?;
                rows.retain(|row| {
                    let text = row
                        .get("zh_text_normalized")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if !forms.matches(text) {
                        return false;
                    }
                    if let Some(canon) = canon {
                        let value = row.get("canon").and_then(|v| v.as_str()).unwrap_or("");
                        if !canon.iter().any(|c| c == value) {
                            return false;
                        }
                    }
                    if let Some(period) = period {
                        if row.get("period").and_then(|v| v.as_str()).unwrap_or("") != period {
                            return false;
                        }
                    }
                    true
                });
                rows.sort_by_key(|row| {
                    let pid = row.get("passage_id").and_then(|v| v.as_str()).unwrap_or("");
                    doc_table.doc_id(pid).unwrap_or(u32::MAX)
                });
                rows.truncate(limit);

                return Ok((
                    rows,
                    json!({
                        "used_phrase_index": true,
                        "phrase_index_path": index_path.display().to_string(),
                        "fallback": false,
                        "gram_size": gram_len,
                        "candidate_stats": candidate_stats,
                        "after_doc_scope": scoped_doc_ids.len(),
                        "limit": limit,
                    }),
                ));
            }

            let fallback_reason = format!("phrase_shorter_than_index_gram_size_{gram_len}");
            if let Some((lo, hi)) = doc_range {
                let (rows, mut strategy) =
                    doc_range_rows(store, doc_table, forms, limit, lo, hi, canon, period).await?;
                strategy["phrase_index_path"] = json!(index_path.display().to_string());
                strategy["fallback"] = json!(true);
                strategy["fallback_reason"] = json!(fallback_reason);
                strategy["gram_size"] = json!(gram_len);
                return Ok((rows, strategy));
            }
            let rows = parquet_global_rows(store, forms, limit, canon, period).await?;
            return Ok((
                rows,
                json!({
                    "used_phrase_index": false,
                    "phrase_index_path": index_path.display().to_string(),
                    "fallback": true,
                    "fallback_reason": fallback_reason,
                    "scope_scan": "parquet_global",
                    "gram_size": gram_len,
                    "limit": limit,
                }),
            ));
        }
    }

    if let Some((lo, hi)) = doc_range {
        return doc_range_rows(store, doc_table, forms, limit, lo, hi, canon, period).await;
    }

    let rows = parquet_global_rows(store, forms, limit, canon, period).await?;
    Ok((
        rows,
        json!({
            "used_phrase_index": false,
            "scope_scan": "parquet_global",
            "limit": limit,
        }),
    ))
}

async fn parquet_global_rows(
    store: &DataFusionStore,
    forms: &PhraseForms,
    limit: usize,
    canon: Option<&[String]>,
    period: Option<&str>,
) -> Result<Vec<Value>> {
    let mut spec = SearchSpec::from_forms(forms.clone(), limit);
    if let Some(canon) = canon {
        spec.canon = canon.to_vec();
    }
    let mut rows = exact_phrase_rows(store, &spec).await?;
    if let Some(period) = period {
        rows.retain(|row| row.get("period").and_then(|v| v.as_str()).unwrap_or("") == period);
        rows.truncate(limit);
    }
    Ok(rows)
}

async fn doc_range_rows(
    store: &DataFusionStore,
    doc_table: &DocumentTable,
    forms: &PhraseForms,
    limit: usize,
    lo: u32,
    hi: u32,
    canon: Option<&[String]>,
    period: Option<&str>,
) -> Result<(Vec<Value>, Value)> {
    let passage_ids: Vec<String> = (lo..=hi)
        .filter_map(|did| doc_table.passage_id(did).map(String::from))
        .collect();
    let mut rows = store.passages_by_ids(&passage_ids, PASSAGE_SELECT).await?;
    rows.retain(|row| {
        let text = row
            .get("zh_text_normalized")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !forms.matches(text) {
            return false;
        }
        if let Some(canon) = canon {
            let value = row.get("canon").and_then(|v| v.as_str()).unwrap_or("");
            if !canon.iter().any(|c| c == value) {
                return false;
            }
        }
        if let Some(period) = period {
            if row.get("period").and_then(|v| v.as_str()).unwrap_or("") != period {
                return false;
            }
        }
        true
    });
    rows.sort_by_key(|row| {
        let pid = row.get("passage_id").and_then(|v| v.as_str()).unwrap_or("");
        doc_table.doc_id(pid).unwrap_or(u32::MAX)
    });
    rows.truncate(limit);
    Ok((
        rows,
        json!({
            "used_phrase_index": false,
            "scope_scan": "doc_range",
            "limit": limit.max(1),
        }),
    ))
}

const PASSAGE_SELECT: &str =
    "passage_id, source_rel_path, xml_id, div_path, heading, heading_path, \
from_lb, to_lb, zh_text_raw, zh_text_normalized, text_type, contains_person, \
contains_term, contains_foreign, canon, canon_name, traditions, period, origin, \
author, main_title, period_rank, source_corpus, source_work_id, source_section_id, \
source_locator, source_url, edition_siglum, edition_label, rights_id, rights_notes, \
retrieval_method, snapshot_id, quality_flags_json";
