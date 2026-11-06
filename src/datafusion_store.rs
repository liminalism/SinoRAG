use anyhow::{Context, Result};
use datafusion::arrow::array::{
    Array, BooleanArray, Float64Array, Int32Array, Int64Array, LargeStringArray, StringArray,
    StringViewArray,
};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::research::format_citation;

pub struct DataFusionStore {
    ctx: SessionContext,
    parquet_dir: PathBuf,
}

impl DataFusionStore {
    pub async fn open(parquet_dir: impl AsRef<Path>) -> Result<Self> {
        let parquet_dir = parquet_dir.as_ref().to_path_buf();

        let ctx = SessionContext::new();

        // Use recursive glob to handle partitioned parquet (source_corpus=cbeta/, etc.)
        let source = parquet_dir
            .join("**/*.parquet")
            .to_string_lossy()
            .replace('\\', "/");

        ctx.register_parquet("passages", &source, ParquetReadOptions::default())
            .await
            .with_context(|| format!("register parquet source {source}"))?;

        Ok(Self { ctx, parquet_dir })
    }

    pub async fn query_json(&self, sql: &str) -> Result<Vec<Value>> {
        let df = self.ctx.sql(sql).await?;
        let batches = df.collect().await?;
        Ok(record_batches_to_json(&batches))
    }

    pub async fn get_passage(&self, passage_id: &str) -> Result<Value> {
        let sql = format!(
            r#"
            SELECT passage_id, source_rel_path, xml_id, div_path, heading, heading_path,
                   from_lb, to_lb, zh_text_raw, zh_text_normalized, text_type,
                   contains_person, contains_term, contains_foreign,
                   canon, canon_name, traditions, period, origin, author, main_title,
                   source_corpus, source_work_id, source_section_id, source_locator,
                   source_url, edition_siglum, edition_label, rights_id, rights_notes,
                   retrieval_method, snapshot_id, quality_flags_json
            FROM passages
            WHERE passage_id = {}
            LIMIT 1
            "#,
            sql_literal(passage_id)
        );

        let rows = self.query_json(&sql).await?;
        if let Some(mut row) = rows.into_iter().next() {
            let from_lb = row
                .get("from_lb")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let to_lb = row
                .get("to_lb")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let citation = format_citation(&row, &from_lb, &to_lb);
            if let Some(obj) = row.as_object_mut() {
                obj.insert("citation".to_string(), json!(citation));
            }
            Ok(row)
        } else {
            Err(anyhow::anyhow!("Passage not found: {passage_id}"))
        }
    }

    pub async fn passage_texts(&self, limit: Option<usize>) -> Result<Vec<(String, String)>> {
        let limit_sql = limit.map(|n| format!(" LIMIT {n}")).unwrap_or_default();
        let sql = format!(
            r#"
            SELECT passage_id, zh_text_normalized
            FROM passages
            WHERE zh_text_normalized IS NOT NULL
              AND length(zh_text_normalized) > 0
            ORDER BY passage_id
            {limit_sql}
            "#
        );
        let rows = self.query_json(&sql).await?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id = row
                .get("passage_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let text = row
                .get("zh_text_normalized")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if !id.is_empty() && !text.is_empty() {
                out.push((id, text));
            }
        }
        Ok(out)
    }

    pub async fn passages_by_ids(&self, ids: &[String], select_cols: &str) -> Result<Vec<Value>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for chunk in ids.chunks(4000) {
            let id_list = chunk
                .iter()
                .map(|id| sql_literal(id))
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!("SELECT {select_cols} FROM passages WHERE passage_id IN ({id_list})");
            out.extend(self.query_json(&sql).await?);
        }
        Ok(out)
    }

    pub fn source_fingerprint(&self) -> Value {
        match std::fs::metadata(&self.parquet_dir) {
            Ok(metadata) => {
                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or_default();
                json!({
                    "kind": "parquet_dir",
                    "path": self.parquet_dir.display().to_string(),
                    "bytes": metadata.len(),
                    "modified_unix": modified
                })
            }
            Err(_) => Value::Null,
        }
    }
}

pub fn sql_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

pub fn string_contains_sql(column: &str, value: &str) -> String {
    format!("strpos({column}, {}) > 0", sql_literal(value))
}

/// SQL form of [`crate::normalize::PhraseForms::matches`]. A pure-CJK phrase
/// yields exactly `string_contains_sql(column, zh)`, so CJK queries run the
/// same SQL as before; otherwise each stored layout is tested with its form.
pub fn phrase_match_sql(column: &str, forms: &crate::normalize::PhraseForms) -> String {
    if forms.is_single() {
        return string_contains_sql(column, &forms.zh);
    }
    let mut parts = Vec::new();
    if !forms.zh.is_empty() {
        parts.push(format!(
            "(strpos({column}, ' ') = 0 AND {})",
            string_contains_sql(column, &forms.zh)
        ));
    }
    if !forms.words.trim().is_empty() {
        parts.push(format!(
            "(strpos({column}, ' ') > 0 AND {})",
            string_contains_sql(column, &forms.words)
        ));
    }
    if parts.is_empty() {
        return "false".to_string();
    }
    format!("({})", parts.join(" OR "))
}

fn record_batches_to_json(batches: &[RecordBatch]) -> Vec<Value> {
    let mut out = Vec::new();

    for batch in batches {
        let schema = batch.schema();

        for row_idx in 0..batch.num_rows() {
            let mut obj = serde_json::Map::new();

            for (col_idx, field) in schema.fields().iter().enumerate() {
                let array = batch.column(col_idx);
                let value = array_value_to_json(array.as_ref(), row_idx, field.name());
                obj.insert(field.name().clone(), value);
            }

            out.push(Value::Object(obj));
        }
    }

    out
}

fn array_value_to_json(array: &dyn Array, row_idx: usize, column_name: &str) -> Value {
    if array.is_null(row_idx) {
        return Value::Null;
    }

    // DataFusion 40+ returns StringViewArray by default for string columns read from parquet.
    if let Some(a) = array.as_any().downcast_ref::<StringViewArray>() {
        return string_to_json(a.value(row_idx), column_name);
    }

    // Standard Utf8 — written by our storage layer; also returned by some DataFusion operations.
    if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
        return string_to_json(a.value(row_idx), column_name);
    }

    // LargeUtf8 — produced by some DataFusion aggregate / cast operations.
    if let Some(a) = array.as_any().downcast_ref::<LargeStringArray>() {
        return string_to_json(a.value(row_idx), column_name);
    }

    if let Some(a) = array.as_any().downcast_ref::<BooleanArray>() {
        return json!(a.value(row_idx));
    }

    if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
        return json!(a.value(row_idx));
    }

    if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
        return json!(a.value(row_idx));
    }

    if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
        return json!(a.value(row_idx));
    }

    Value::Null
}

/// Apply column-specific JSON parsing (traditions / *_json fields) to a raw string value.
fn string_to_json(value: &str, column_name: &str) -> Value {
    // traditions is stored as a JSON array string in parquet
    if column_name == "traditions" {
        return serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()));
    }
    if column_name.ends_with("_json") {
        return serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()));
    }
    Value::String(value.to_string())
}
