use serde::Serialize;
use std::path::PathBuf;

/// Classified tool error with structured information for agents
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ToolErrorBody {
    pub code: String,
    pub message: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_command: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

/// Typed tool errors
#[derive(thiserror::Error, Debug)]
pub enum ToolError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),

    #[error("missing phrase index at {path}")]
    MissingPhraseIndex { path: PathBuf },

    #[error("missing tfidf index at {path}")]
    MissingTfidfIndex { path: PathBuf },

    #[error("missing vector index at {path}")]
    MissingVectorIndex { path: PathBuf },

    #[error("missing catalog index at {path}")]
    MissingCatalogIndex { path: PathBuf },

    #[error("missing doc table at {path}")]
    MissingDocTable { path: PathBuf },

    #[error("missing passages parquet at {path}")]
    MissingPassages { path: PathBuf },

    #[error("readonly mode blocks tool: {tool}")]
    ReadonlyViolation { tool: String },

    #[error("admin tool disabled: {tool}")]
    AdminToolDisabled { tool: String },

    #[error("output path {path} is outside output root {root}")]
    OutputPathViolation { path: PathBuf, root: PathBuf },

    #[error("invalid JSON args: {0}")]
    InvalidJson(String),

    #[error("invalid args: {0}")]
    InvalidArgs(String),

    /// The arguments do not fit the tool's input schema. `problems` names
    /// each one; `accepted` and `required` are the tool's argument names.
    #[error("{tool}: {}", .problems.join("; "))]
    InvalidArguments {
        tool: String,
        problems: Vec<String>,
        accepted: Vec<String>,
        required: Vec<String>,
    },

    #[error("{tool} expects a passage_id, but got {provided}")]
    InvalidPassageId {
        tool: String,
        provided: String,
        detected_kind: String,
    },

    #[error("passage not found: {passage_id}")]
    PassageNotFound { passage_id: String },

    #[error("query embedding provider is not configured")]
    QueryEmbeddingProviderNotConfigured,

    #[error("tool component timed out: {component} after {timeout_ms} ms")]
    Timeout { component: String, timeout_ms: u64 },

    #[error("internal error: {0}")]
    Internal(String),
}

impl ToolError {
    pub fn unknown_tool(name: &str) -> Self {
        ToolError::UnknownTool(name.to_string())
    }

    pub fn into_anyhow(self) -> anyhow::Error {
        anyhow::Error::from(self)
    }
}

/// What the user has to do before there is anything to search. A model
/// cannot build the corpus from inside a session, so this is advice to pass
/// on, and it names the route this install actually has.
fn how_to_build_the_corpus() -> String {
    let from_document_intake = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("sinorag-intake.exe")))
        .is_some_and(|intake| intake.is_file());
    if from_document_intake {
        "Tell the user that no documents are ready to search yet. They need to close this \
         session, open \"Add documents to SinoRAG\" from the Start Menu, add their files and \
         let it finish, then start again. Retrying the call will not help."
            .to_string()
    } else {
        "Tell the user the corpus is not built; retrying the call will not help. It is \
         built from a terminal with `sinorag init` or `sinorag ingest`."
            .to_string()
    }
}

/// One of the derived indexes is missing while the corpus itself exists.
fn missing_index(code: &str, kind: &str, path: &std::path::Path) -> ToolErrorBody {
    ToolErrorBody {
        code: code.to_string(),
        message: format!(
            "The {kind} index has not been built: nothing at {}",
            path.display()
        ),
        suggested_command: Some(how_to_build_the_corpus()),
        details: Some(serde_json::json!({ "path": path.display().to_string() })),
    }
}

/// Classify an anyhow error into a structured ToolErrorBody
pub fn classify_tool_error(err: &anyhow::Error) -> ToolErrorBody {
    let msg = err.to_string();

    // Try to downcast to ToolError first
    if let Some(tool_err) = err.downcast_ref::<ToolError>() {
        return match tool_err {
            ToolError::UnknownTool(name) => ToolErrorBody {
                code: "unknown_tool".to_string(),
                message: format!("Unknown tool: {}", name),
                suggested_command: Some(
                    "Call tool-docs (no arguments) for the list of tools".to_string(),
                ),
                details: None,
            },
            ToolError::MissingPhraseIndex { path } => missing_index("missing_phrase_index", "phrase", path),
            ToolError::MissingTfidfIndex { path } => missing_index("missing_tfidf_index", "tfidf", path),
            ToolError::MissingVectorIndex { path } => ToolErrorBody {
                code: "missing_vector_index".to_string(),
                message: format!(
                    "Semantic (vector) search is not built for this collection: no index at {}",
                    path.display()
                ),
                suggested_command: Some(
                    "Use frontier or similar for related passages instead. (Operator: \
                     `sinorag index vector-update` builds the semantic index.)"
                        .to_string(),
                ),
                details: Some(serde_json::json!({ "path": path.display().to_string() })),
            },
            ToolError::MissingCatalogIndex { path } => missing_index("missing_catalog_index", "catalog", path),
            ToolError::MissingDocTable { path } => missing_index("missing_doc_table", "doc table", path),
            ToolError::MissingPassages { path } => ToolErrorBody {
                code: "missing_passages".to_string(),
                message: format!(
                    "Nothing has been indexed yet: there are no passages at {}",
                    path.display()
                ),
                suggested_command: Some(how_to_build_the_corpus()),
                details: Some(serde_json::json!({ "path": path.display().to_string() })),
            },
            ToolError::ReadonlyViolation { tool } => ToolErrorBody {
                code: "readonly_violation".to_string(),
                message: format!(
                    "Tool '{}' writes a file, and this SinoRAG session is read-only",
                    tool
                ),
                suggested_command: Some(
                    "Give the content to the user in your reply instead. Output tools are \
                     enabled by starting `sinorag mcp` or `sinorag agent` with --writable."
                        .to_string(),
                ),
                details: Some(serde_json::json!({ "tool": tool })),
            },
            ToolError::AdminToolDisabled { tool } => ToolErrorBody {
                code: "admin_tool_disabled".to_string(),
                message: format!("Tool '{}' requires --allow-admin-tools flag", tool),
                suggested_command: Some(
                    "Add --allow-admin-tools flag to enable admin tools".to_string(),
                ),
                details: Some(serde_json::json!({ "tool": tool })),
            },
            ToolError::OutputPathViolation { path, root } => ToolErrorBody {
                code: "output_path_denied".to_string(),
                message: format!(
                    "Output path {} is outside output root {}",
                    path.display(),
                    root.display()
                ),
                suggested_command: Some(
                    "Specify --output-root or use a path inside the allowed directory".to_string(),
                ),
                details: Some(
                    serde_json::json!({ "path": path.display().to_string(), "root": root.display().to_string() }),
                ),
            },
            ToolError::InvalidJson(s) => ToolErrorBody {
                code: "invalid_json".to_string(),
                message: format!("Invalid JSON: {}", s),
                suggested_command: None,
                details: None,
            },
            ToolError::InvalidArgs(s) => ToolErrorBody {
                code: "invalid_args".to_string(),
                message: format!("Invalid arguments: {}", s),
                suggested_command: Some(
                    "See the tool's input schema, or call tool-docs with {\"tool\": \"<name>\"}"
                        .to_string(),
                ),
                details: None,
            },
            ToolError::InvalidArguments {
                tool,
                problems,
                accepted,
                required,
            } => ToolErrorBody {
                code: "invalid_args".to_string(),
                message: format!("Invalid arguments for {}: {}", tool, problems.join("; ")),
                suggested_command: Some(if accepted.is_empty() {
                    format!("{tool} takes no arguments")
                } else if required.is_empty() {
                    format!("{tool} accepts: {}", accepted.join(", "))
                } else {
                    format!(
                        "{tool} accepts: {} (required: {})",
                        accepted.join(", "),
                        required.join(", ")
                    )
                }),
                details: Some(serde_json::json!({
                    "tool": tool,
                    "problems": problems,
                })),
            },
            ToolError::InvalidPassageId {
                tool,
                provided,
                detected_kind,
            } => {
                let (message, suggested_command) = if detected_kind == "source_work_id" {
                    (
                        format!(
                            "Tool '{tool}' expects a passage_id, but '{provided}' looks like a source_work_id"
                        ),
                        Some(format!(
                            "Use works with work_id='{provided}' for metadata, or search/source-read to select a passage in that work"
                        )),
                    )
                } else {
                    (
                        format!(
                            "Tool '{tool}' expects one passage_id, but the supplied value looks like search text"
                        ),
                        Some("Use search or evidence-search first, then pass one returned passage_id".to_string()),
                    )
                };
                ToolErrorBody {
                    code: "invalid_passage_id".to_string(),
                    message,
                    suggested_command,
                    details: Some(serde_json::json!({
                        "tool": tool,
                        "provided": provided,
                        "detected_kind": detected_kind,
                    })),
                }
            }
            ToolError::PassageNotFound { passage_id } => ToolErrorBody {
                code: "passage_not_found".to_string(),
                message: format!("No passage exists with passage_id '{passage_id}'"),
                suggested_command: Some(
                    "Use search, heading-search, or source-read to obtain an exact passage_id"
                        .to_string(),
                ),
                details: Some(serde_json::json!({ "passage_id": passage_id })),
            },
            ToolError::QueryEmbeddingProviderNotConfigured => ToolErrorBody {
                code: "query_embedding_provider_not_configured".to_string(),
                message: "Native query_text embedding is not configured; use seed_passage_id or query_embedding.".to_string(),
                suggested_command: None,
                details: None,
            },
            ToolError::Timeout {
                component,
                timeout_ms,
            } => ToolErrorBody {
                code: "timeout".to_string(),
                message: format!("{component} timed out after {timeout_ms} ms"),
                suggested_command: None,
                details: Some(serde_json::json!({
                    "component": component,
                    "timeout_ms": timeout_ms
                })),
            },
            ToolError::Internal(s) => ToolErrorBody {
                code: "internal_error".to_string(),
                message: format!("Internal error: {}", s),
                suggested_command: None,
                details: None,
            },
        };
    }

    // Fallback: string matching for common patterns
    if msg.contains("phrase") && (msg.contains("not found") || msg.contains("No such file")) {
        return ToolErrorBody {
            code: "missing_phrase_index".to_string(),
            message: msg,
            suggested_command: Some(
                "sinorag index phrase --parquet data/passages.parquet --out data/derived/phrase.index".to_string()
            ),
            details: None,
        };
    }

    if msg.contains("tfidf") && (msg.contains("not found") || msg.contains("No such file")) {
        return ToolErrorBody {
            code: "missing_tfidf_index".to_string(),
            message: msg,
            suggested_command: Some(
                "sinorag index tfidf --parquet data/passages.parquet --out data/derived/tfidf.index".to_string()
            ),
            details: None,
        };
    }

    if msg.contains("vector") && (msg.contains("not found") || msg.contains("No such file")) {
        return ToolErrorBody {
            code: "missing_vector_index".to_string(),
            message: msg,
            suggested_command: Some(
                "sinorag index vector-build --embeddings data/derived/embeddings.jsonl --model-id MODEL --out data/derived/vector.index".to_string(),
            ),
            details: None,
        };
    }

    if msg.contains("catalog") && (msg.contains("not found") || msg.contains("No such file")) {
        return ToolErrorBody {
            code: "missing_catalog_index".to_string(),
            message: msg,
            suggested_command: Some(
                "Run 'sinorag catalog-index-build' to build the catalog index".to_string(),
            ),
            details: None,
        };
    }

    if msg.contains("passages") && (msg.contains("not found") || msg.contains("No such file")) {
        return ToolErrorBody {
            code: "missing_passages".to_string(),
            message: msg,
            suggested_command: Some("Run 'sinorag ingest' to build the corpus".to_string()),
            details: None,
        };
    }

    ToolErrorBody {
        code: "internal_error".to_string(),
        message: msg,
        suggested_command: None,
        details: None,
    }
}
