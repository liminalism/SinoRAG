use crate::datafusion_store::DataFusionStore;
use crate::document_table::DocumentTable;
use crate::jsonout;
use crate::tfidf::index::{long_common_substrings, TfidfIndex};
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

pub fn info(index_path: PathBuf) -> Result<()> {
    // Header-only read — works on multi-GB indexes too.
    let payload = TfidfIndex::header_info(&index_path)?;
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}

pub async fn similar(
    parquet_path: PathBuf,
    index_path: PathBuf,
    seed: String,
    limit: usize,
    shared_ngram_limit: usize,
    shared_phrase_limit: usize,
    min_shared_phrase_len: usize,
    out: Option<PathBuf>,
) -> Result<()> {
    if !index_path.exists() {
        anyhow::bail!(
            "TF-IDF index not found at {}. Run `sinorag tfidf-build` first.",
            index_path.display()
        );
    }
    let store = DataFusionStore::open(&parquet_path).await?;

    let doc_table_path = index_path
        .parent()
        .unwrap_or(&PathBuf::from("."))
        .join("doc_table.bin");
    let doc_table = if doc_table_path.exists() {
        DocumentTable::load(&doc_table_path)?
    } else {
        anyhow::bail!(
            "DocumentTable not found at {}. Run doc-table-build first.",
            doc_table_path.display()
        );
    };

    let index = TfidfIndex::load(&index_path)?;
    let results = similar_passages_with_index(
        &store,
        &index,
        &seed,
        limit,
        shared_ngram_limit,
        shared_phrase_limit,
        min_shared_phrase_len,
        &doc_table,
    )
    .await?;
    let payload = json!({
        "seed": seed,
        "returned_count": results.len(),
        "limit": limit,
        "results": results,
    });
    jsonout::write_or_print(&payload, out)
}

pub async fn similar_batch(
    parquet_path: PathBuf,
    index_path: PathBuf,
    seeds: PathBuf,
    limit: usize,
    shared_ngram_limit: usize,
    shared_phrase_limit: usize,
    min_shared_phrase_len: usize,
    out: PathBuf,
) -> Result<()> {
    if !index_path.exists() {
        anyhow::bail!(
            "TF-IDF index not found at {}. Run `sinorag tfidf-build` first.",
            index_path.display()
        );
    }
    let seed_ids: Vec<String> = std::fs::read_to_string(&seeds)?
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToString::to_string)
        .collect();
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let store = DataFusionStore::open(&parquet_path).await?;

    let doc_table_path = index_path
        .parent()
        .unwrap_or(&PathBuf::from("."))
        .join("doc_table.bin");
    let doc_table = if doc_table_path.exists() {
        DocumentTable::load(&doc_table_path)?
    } else {
        anyhow::bail!(
            "DocumentTable not found at {}. Run doc-table-build first.",
            doc_table_path.display()
        );
    };

    let index = TfidfIndex::load(&index_path)?;
    let mut writer = BufWriter::new(File::create(&out)?);
    for seed in &seed_ids {
        let results = similar_passages_with_index(
            &store,
            &index,
            seed,
            limit,
            shared_ngram_limit,
            shared_phrase_limit,
            min_shared_phrase_len,
            &doc_table,
        )
        .await?;
        let payload = json!({
            "seed": seed,
            "returned_count": results.len(),
            "limit": limit,
            "results": results,
        });
        serde_json::to_writer(&mut writer, &payload)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    println!("wrote {}", out.display());
    println!("seeds {}", seed_ids.len());
    Ok(())
}

pub async fn similar_passages(
    store: &DataFusionStore,
    index_path: PathBuf,
    seed: &str,
    limit: usize,
    shared_ngram_limit: usize,
    shared_phrase_limit: usize,
    min_shared_phrase_len: usize,
    doc_table: &DocumentTable,
) -> Result<Vec<Value>> {
    if !index_path.exists() {
        anyhow::bail!(
            "TF-IDF index not found at {}. Run `sinorag tfidf-build` first.",
            index_path.display()
        );
    }
    let index = TfidfIndex::load(&index_path)?;
    similar_passages_with_index(
        store,
        &index,
        seed,
        limit,
        shared_ngram_limit,
        shared_phrase_limit,
        min_shared_phrase_len,
        doc_table,
    )
    .await
}

pub async fn similar_passages_with_index(
    store: &DataFusionStore,
    index: &TfidfIndex,
    seed: &str,
    limit: usize,
    shared_ngram_limit: usize,
    shared_phrase_limit: usize,
    min_shared_phrase_len: usize,
    doc_table: &DocumentTable,
) -> Result<Vec<Value>> {
    let seed_doc_id = doc_table.doc_id(seed).ok_or_else(|| {
        crate::tools::errors::ToolError::PassageNotFound {
            passage_id: seed.to_string(),
        }
        .into_anyhow()
    })?;
    let ranked = index.similar(seed_doc_id, limit)?;
    if ranked.is_empty() {
        return Ok(Vec::new());
    }

    let mut ids: Vec<String> = ranked
        .iter()
        .map(|(doc_id, _)| doc_table.passage_id(*doc_id).unwrap_or("").to_string())
        .collect();
    ids.push(seed.to_string());
    let rows = store.passages_by_ids(
        &ids,
        "passage_id, source_rel_path, xml_id, heading, from_lb, to_lb, zh_text_raw, zh_text_normalized, canon, traditions, period, origin, author, main_title",
    ).await?;
    let by_id: HashMap<String, Value> = rows
        .into_iter()
        .filter_map(|row| {
            let id = row.get("passage_id").and_then(Value::as_str)?.to_string();
            Some((id, row))
        })
        .collect();
    let seed_row = by_id
        .get(seed)
        .ok_or_else(|| anyhow!("Seed passage not found: {seed}"))?;
    let seed_norm = jsonout::value_str(seed_row, "zh_text_normalized");

    let mut results = Vec::new();
    for (doc_id, score) in ranked {
        let passage_id = doc_table.passage_id(doc_id).unwrap_or("");
        let cand = by_id.get(passage_id).cloned().unwrap_or_else(|| json!({}));
        let cand_norm = jsonout::value_str(&cand, "zh_text_normalized");
        let shared_ngrams =
            index.shared_ngrams_with_seed_text(seed_doc_id, doc_id, &seed_norm, shared_ngram_limit);
        let shared_phrases = long_common_substrings(
            &seed_norm,
            &cand_norm,
            min_shared_phrase_len,
            shared_phrase_limit,
        );

        let same_file = jsonout::value_str(seed_row, "source_rel_path")
            == jsonout::value_str(&cand, "source_rel_path");
        let same_canon =
            jsonout::value_str(seed_row, "canon") == jsonout::value_str(&cand, "canon");
        let cand_period = jsonout::value_str(&cand, "period");
        let same_period = jsonout::value_str(seed_row, "period") == cand_period;
        let cross_source = !same_file;
        let cross_period = !same_period && !cand_period.is_empty();

        let long_gram_count =
            index.long_gram_shared_count_with_seed_text(seed_doc_id, doc_id, &seed_norm, 6);
        let max_lcs_len = shared_phrases
            .iter()
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(0);
        let cand_len = doc_table
            .period_ranks
            .get(doc_id as usize)
            .copied()
            .unwrap_or(0) as usize;

        let mut graph_score = score;
        let mut reasons: Vec<&'static str> = Vec::new();

        if same_file {
            graph_score *= 0.60;
            reasons.push("same_file_penalty");
        } else {
            graph_score *= 1.10;
            reasons.push("cross_file_bonus");
        }

        if cand_len > 0 && cand_len < 15 {
            graph_score *= 0.70;
            reasons.push("short_passage_penalty");
        }

        if max_lcs_len >= 8 {
            graph_score *= 1.25;
            reasons.push("long_shared_phrase_bonus");
        } else if max_lcs_len >= 5 {
            graph_score *= 1.10;
            reasons.push("shared_phrase_bonus");
        } else if shared_phrases.is_empty() {
            graph_score *= 0.75;
            reasons.push("no_shared_phrase_penalty");
        }

        if long_gram_count >= 3 {
            graph_score *= 1.15;
            reasons.push("many_distinctive_ngrams");
        } else if long_gram_count >= 1 {
            graph_score *= 1.05;
            reasons.push("shared_long_ngram_bonus");
        }

        if cross_period {
            graph_score *= 1.05;
            reasons.push("cross_period_bonus");
        }

        let ring_hint = if same_file {
            "downrank_same_file"
        } else if cand_len > 0 && cand_len < 15 {
            "downrank_short_passage"
        } else if max_lcs_len >= 6 && long_gram_count >= 2 {
            "ring1_candidate"
        } else if long_gram_count >= 1 || max_lcs_len >= 4 {
            "ring2_motif"
        } else {
            "ring3_weak"
        };

        results.push(json!({
            "passage_id": passage_id,
            "tfidf_cosine": round_f32(score, 6),
            "graph_candidate_score": round_f32(graph_score, 6),
            "ring_hint": ring_hint,
            "reasons": reasons,
            "same_file": same_file,
            "same_canon": same_canon,
            "same_period": same_period,
            "cross_source": cross_source,
            "long_gram_count": long_gram_count,
            "max_lcs_len": max_lcs_len,
            "cand_char_len": cand_len,
            "shared_ngrams": shared_ngrams,
            "shared_phrases": shared_phrases,
            "source_rel_path": jsonout::value_str(&cand, "source_rel_path"),
            "xml_id": jsonout::value_str(&cand, "xml_id"),
            "heading": jsonout::value_str(&cand, "heading"),
            "from_lb": cand.get("from_lb").cloned().unwrap_or(Value::Null),
            "to_lb": cand.get("to_lb").cloned().unwrap_or(Value::Null),
            "zh_text_raw": jsonout::value_str(&cand, "zh_text_raw"),
            "canon": jsonout::value_str(&cand, "canon"),
            "traditions": cand.get("traditions").cloned().unwrap_or_else(|| json!([])),
            "period": cand_period,
            "origin": jsonout::value_str(&cand, "origin"),
            "author": jsonout::value_str(&cand, "author"),
            "main_title": jsonout::value_str(&cand, "main_title"),
        }));
    }
    Ok(results)
}

fn round_f32(value: f32, places: i32) -> f32 {
    let factor = 10f32.powi(places);
    (value * factor).round() / factor
}
