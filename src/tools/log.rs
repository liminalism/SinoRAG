use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::tools::errors::ToolErrorBody;

const DEFAULT_LOG_PATH: &str = ".sinorag/tool_calls.jsonl";
const MAX_SUMMARY_OBJECT_DEPTH: usize = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallLogRecord {
    pub schema: String,
    pub timestamp_utc: String,
    pub tool: String,
    pub ok: bool,
    pub elapsed_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    /// The error text the caller was shown. The code alone cannot tell a
    /// misspelled argument from a server fault when the log is read later.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    pub args_summary: Value,
    pub result_summary: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolLogSummary {
    pub schema: &'static str,
    pub path: String,
    pub total_calls: usize,
    pub tools: Vec<ToolLogToolSummary>,
    pub recent: Vec<ToolCallLogRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolLogToolSummary {
    pub tool: String,
    pub calls: usize,
    pub successes: usize,
    pub failures: usize,
    pub avg_elapsed_ms: f64,
    pub p95_elapsed_ms: u128,
    pub last_error_code: Option<String>,
}

pub fn default_log_path() -> PathBuf {
    std::env::var_os("SINORAG_TOOL_LOG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_LOG_PATH))
}

pub fn append_call(
    tool: &str,
    args: &Value,
    result: Option<&Value>,
    error: Option<&ToolErrorBody>,
    elapsed_ms: u128,
) {
    let path = default_log_path();
    if let Err(err) = append_call_at(&path, tool, args, result, error, elapsed_ms) {
        eprintln!(
            "[sinorag] WARNING: failed to append tool log {}: {err}",
            path.display()
        );
    }
}

pub fn append_call_at(
    path: &Path,
    tool: &str,
    args: &Value,
    result: Option<&Value>,
    error: Option<&ToolErrorBody>,
    elapsed_ms: u128,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let record = ToolCallLogRecord {
        schema: "sinorag-tool-call-log-v1".to_string(),
        timestamp_utc: Utc::now().to_rfc3339(),
        tool: tool.to_string(),
        ok: error.is_none(),
        elapsed_ms,
        error_code: error.map(|err| err.code.clone()),
        error_message: error.map(|err| err.message.chars().take(300).collect()),
        args_summary: summarize_value(args),
        result_summary: result
            .map(summarize_tool_result)
            .unwrap_or_else(|| serde_json::json!({})),
    };
    let mut line = serde_json::to_vec(&record)?;
    line.push(b'\n');
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(&line)?;
    Ok(())
}

/// Return the names of the most recent `limit` tool calls, oldest first.
///
/// Used to avoid re-suggesting a tool the agent has already pivoted to —
/// suggestions should fade once acted on rather than repeating every call.
pub fn recent_tool_names(path: &Path, limit: usize) -> Result<Vec<String>> {
    if limit == 0 || !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path)?;
    let mut names: Vec<String> = text
        .lines()
        .rev()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<ToolCallLogRecord>(line).ok())
        .map(|record| record.tool)
        .take(limit)
        .collect();
    names.reverse();
    Ok(names)
}

pub fn summarize(path: &Path, limit_recent: usize) -> Result<ToolLogSummary> {
    let mut records = Vec::new();
    if path.exists() {
        let text = std::fs::read_to_string(path)?;
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            if let Ok(record) = serde_json::from_str::<ToolCallLogRecord>(line) {
                records.push(record);
            }
        }
    }

    let mut by_tool: BTreeMap<String, Vec<&ToolCallLogRecord>> = BTreeMap::new();
    for record in &records {
        by_tool.entry(record.tool.clone()).or_default().push(record);
    }

    let mut tools = Vec::with_capacity(by_tool.len());
    for (tool, calls) in by_tool {
        let successes = calls.iter().filter(|record| record.ok).count();
        let failures = calls.len().saturating_sub(successes);
        let total_elapsed: u128 = calls.iter().map(|record| record.elapsed_ms).sum();
        let avg_elapsed_ms = if calls.is_empty() {
            0.0
        } else {
            total_elapsed as f64 / calls.len() as f64
        };
        let mut elapsed = calls
            .iter()
            .map(|record| record.elapsed_ms)
            .collect::<Vec<_>>();
        elapsed.sort_unstable();
        let p95_elapsed_ms = percentile(&elapsed, 0.95);
        let last_error_code = calls
            .iter()
            .rev()
            .find_map(|record| record.error_code.clone());
        tools.push(ToolLogToolSummary {
            tool,
            calls: calls.len(),
            successes,
            failures,
            avg_elapsed_ms,
            p95_elapsed_ms,
            last_error_code,
        });
    }
    tools.sort_by(|a, b| b.calls.cmp(&a.calls).then(a.tool.cmp(&b.tool)));

    let mut recent = records
        .iter()
        .rev()
        .take(limit_recent)
        .map(|record| {
            let mut compact = record.clone();
            // Logs from older binaries may contain deeply nested summaries.
            // Compact them again on read so one historical composite response
            // cannot overwhelm `tool-log-summary` output.
            compact.args_summary = summarize_value(&compact.args_summary);
            compact.result_summary = summarize_value(&compact.result_summary);
            compact
        })
        .collect::<Vec<_>>();
    recent.reverse();

    Ok(ToolLogSummary {
        schema: "sinorag-tool-log-summary-v1",
        path: path.display().to_string(),
        total_calls: records.len(),
        tools,
        recent,
    })
}

fn percentile(sorted: &[u128], pct: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * pct).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn summarize_tool_result(result: &Value) -> Value {
    let mut summary = summarize_value(result);
    if let Some(obj) = result.as_object() {
        let mut counts = serde_json::Map::new();
        for key in [
            "hits",
            "merged_hits",
            "groups",
            "clusters",
            "sample_hits",
            "suggested_next_tools",
            "warnings",
            "stages",
            "components",
        ] {
            if let Some(array) = obj.get(key).and_then(|value| value.as_array()) {
                counts.insert(key.to_string(), serde_json::json!(array.len()));
            }
        }
        if !counts.is_empty() {
            summary["top_level_counts"] = Value::Object(counts);
        }
        for key in [
            "schema",
            "workflow",
            "mode",
            "mode_reason",
            "returned_count",
            "hit_count",
            "pair_hit_count",
            "total_pair_hits",
        ] {
            if let Some(value) = obj.get(key) {
                summary[key] = value.clone();
            }
        }
    }
    summary
}

fn summarize_value(value: &Value) -> Value {
    summarize_value_at_depth(value, 0)
}

fn summarize_value_at_depth(value: &Value, depth: usize) -> Value {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => value.clone(),
        Value::String(s) => {
            if s.chars().count() > 80 {
                let truncated = s.chars().take(80).collect::<String>();
                serde_json::json!({"type": "string", "chars": s.chars().count(), "prefix": truncated})
            } else {
                value.clone()
            }
        }
        Value::Array(items) => serde_json::json!({
            "type": "array",
            "len": items.len(),
        }),
        Value::Object(obj) => {
            if depth >= MAX_SUMMARY_OBJECT_DEPTH {
                return serde_json::json!({
                    "type": "object",
                    "keys": obj.len(),
                });
            }
            let mut out = serde_json::Map::new();
            for (key, value) in obj {
                out.insert(key.clone(), summarize_value_at_depth(value, depth + 1));
            }
            Value::Object(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recent_tool_names_returns_last_n_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tool_calls.jsonl");
        for tool in ["search", "search", "frontier", "source-read"] {
            append_call_at(&path, tool, &serde_json::json!({}), None, None, 1).unwrap();
        }
        assert_eq!(
            recent_tool_names(&path, 2).unwrap(),
            vec!["frontier".to_string(), "source-read".to_string()]
        );
        assert_eq!(
            recent_tool_names(&path, 10).unwrap(),
            vec![
                "search".to_string(),
                "search".to_string(),
                "frontier".to_string(),
                "source-read".to_string()
            ]
        );
    }

    #[test]
    fn recent_tool_names_empty_when_log_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.jsonl");
        assert!(recent_tool_names(&path, 5).unwrap().is_empty());
    }

    #[test]
    fn summaries_bound_nested_object_detail() {
        let summary = summarize_value(&serde_json::json!({
            "result": {
                "timeline": {
                    "Song": {
                        "representative": {
                            "passage_id": "T/T01/example.xml#p1",
                            "zh_text_raw": "佛法"
                        }
                    }
                }
            }
        }));
        assert_eq!(summary["result"]["type"], "object");
        assert_eq!(summary["result"]["keys"], 1);
        assert!(serde_json::to_string(&summary).unwrap().len() < 60);
    }
}
