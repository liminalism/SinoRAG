//! Stdio MCP server exposing the SinoRAG tool registry.
//!
//! Unlike the previous hand-rolled per-tool implementation, this is a thin
//! shim over [`crate::tools::tool_defs`]: every tool registered in the engine
//! is exposed automatically. The tool list is the single source of truth.
//!
//! Logging must go to **stderr only** — stdout carries the JSON-RPC framing.

use std::borrow::Cow;
use std::sync::Arc;

use anyhow::Result;
use rmcp::{
    handler::server::router::tool::ToolRouter,
    model::{
        CallToolRequestParams, CallToolResult, Content, Implementation, JsonObject,
        ListToolsResult, PaginatedRequestParams, ProtocolVersion, ServerCapabilities, ServerInfo,
        Tool,
    },
    service::RequestContext,
    transport::stdio,
    ErrorData as McpError, RoleServer, ServerHandler, ServiceExt,
};
use serde_json::Value;

use crate::tools::registry::{ToolCallEnvelope, ToolCallMeta};
use crate::tools::{
    call_tool_enveloped, tool_defs, EngineConfig, ToolAudience, ToolEngine, ToolErrorBody,
    ToolSafety,
};

/// Embedded doctrine shown to the model as the MCP server's `instructions`
/// string. The wrapping `agent` launcher uses the same fragments to build
/// `AGENTS.md`; here we keep a compact version so the model has the lens
/// guidance even when invoked from a non-opencode client.
const SERVER_INSTRUCTIONS: &str = include_str!("../agent/doctrine/mcp_instructions.md");
/// Appended only when the data root holds a word-layout corpus (English or
/// another alphabetic script, e.g. from document intake).
pub const WORD_CORPUS_INSTRUCTIONS: &str = include_str!("../agent/doctrine/word_corpus.md");
/// Used instead of the research doctrine when the data root holds nothing
/// but word-layout corpora: a collection of someone's documents. The
/// research doctrine is about the CBETA canon (canons, periods, lineages),
/// none of which such a collection has.
pub const DOCUMENTS_INSTRUCTIONS: &str = include_str!("../agent/doctrine/documents.md");

/// The doctrine for a data root: what the MCP server sends as its
/// `instructions`, and what `sinorag agent` writes into `AGENTS.md` in
/// place of `research` there.
pub fn doctrine_for(passages_parquet: &std::path::Path, research: &str) -> String {
    let with_words = |head: &str| {
        format!(
            "{}\n\n{}\n",
            head.trim_end(),
            WORD_CORPUS_INSTRUCTIONS.trim_end()
        )
    };
    if crate::normalize::data_root_is_word_only(passages_parquet) {
        with_words(DOCUMENTS_INSTRUCTIONS)
    } else if crate::normalize::data_root_has_word_corpus(passages_parquet) {
        with_words(research)
    } else {
        research.to_string()
    }
}

pub struct SinoragMcpServer {
    engine: Arc<ToolEngine>,
    tools: Vec<Tool>,
    instructions: String,
    /// Required by the `ServerHandler` blanket impls even though we never
    /// route through it (we override `list_tools`/`call_tool` directly).
    _router: ToolRouter<Self>,
}

impl SinoragMcpServer {
    /// The tools this server can actually run, as MCP tool definitions.
    ///
    /// A tool the session's flags rule out is not listed: a model that sees
    /// `pdf-build` in a read-only session calls it, and gets
    /// `readonly_violation` for its trouble.
    ///
    /// No `outputSchema` is advertised:
    /// - schemars writes `true` as the schema of a `serde_json::Value`
    ///   field. The official TypeScript SDK client (seen with 1.32) does not
    ///   accept that as a property schema and rejects the whole `tools/list`
    ///   response, so the server connects and shows no tools.
    /// - A tool's `output_schema` describes its `result`, but a call returns
    ///   the envelope around it (`{ok, tool, result | error, meta}`), and
    ///   MCP requires structured content to match an advertised schema.
    /// - The schemas were two thirds of the `tools/list` payload.
    ///
    /// `sinorag tools-manifest` still carries them.
    ///
    /// Nor is a tool that has nothing to work on in this data root: see
    /// [`Listing`].
    fn build_tools(config: &EngineConfig, listing: Listing) -> Vec<Tool> {
        tool_defs()
            .into_iter()
            .filter(|def| def.spec.audience != ToolAudience::InternalDebug)
            .filter(|def| tool_is_enabled(&def.spec.safety, config))
            .filter(|def| listing.offers(def.spec.name))
            .map(|def| {
                Tool::new(
                    Cow::Borrowed(def.spec.name),
                    Cow::Borrowed(def.spec.description),
                    Arc::new(mcp_input_schema(&def.spec.input_schema)),
                )
            })
            .collect()
    }

    pub fn new(engine: Arc<ToolEngine>) -> Self {
        let instructions = match engine.resolve_passages_path() {
            Ok(passages) => doctrine_for(&passages, SERVER_INSTRUCTIONS),
            Err(_) => SERVER_INSTRUCTIONS.to_string(),
        };
        let listing = Listing {
            documents_only: engine
                .resolve_passages_path()
                .is_ok_and(|p| crate::normalize::data_root_is_word_only(&p)),
            has_vector_index: engine.resolve_vector_path().is_ok(),
        };
        Self {
            tools: Self::build_tools(&engine.config, listing),
            engine,
            instructions,
            _router: ToolRouter::new(),
        }
    }
}

/// What the data root has for tools to work on. A tool left out of the list
/// can still be called by name, and answers as it always did; it is only
/// not offered, so a model does not spend calls finding out it is empty.
#[derive(Debug, Clone, Copy)]
struct Listing {
    /// Nothing but the user's own documents: no canon, period, lineage or
    /// name authority for the tools below to draw on.
    documents_only: bool,
    has_vector_index: bool,
}

/// Tools that order or group by the CBETA catalogue (period, canon,
/// tradition) or resolve names against the DDBC authority files. On a
/// documents collection they run and return one group with an empty key,
/// an "earliest" passage that is only the first one, or no match.
const CATALOGUE_TOOLS: &[&str] = &[
    "canonical-source",
    "first-attestation",
    "phrase-history",
    "trace-term-usage",
    "seed-pick",
    "person-resolve",
    "person-history",
    "person-profile",
    "place-resolve",
];

impl Listing {
    fn offers(self, tool: &str) -> bool {
        if tool == "vector-neighbors" {
            return self.has_vector_index;
        }
        !(self.documents_only && CATALOGUE_TOOLS.contains(&tool))
    }
}

/// Mirrors `registry::enforce_safety`: whether a call to a tool of this
/// safety class would be let through.
fn tool_is_enabled(safety: &ToolSafety, config: &EngineConfig) -> bool {
    match safety {
        ToolSafety::ReadOnly => true,
        ToolSafety::WritesOutput => !config.readonly,
        ToolSafety::MutatesRegistry | ToolSafety::Admin => config.allow_admin_tools,
    }
}

/// The result for a tool call whose task panicked or was cancelled.
fn crashed_envelope(tool: String, join_error: &tokio::task::JoinError) -> ToolCallEnvelope {
    tracing::error!("tool {tool} did not finish: {join_error}");
    ToolCallEnvelope {
        id: None,
        ok: false,
        error: Some(ToolErrorBody {
            code: "internal_error".to_string(),
            message: format!(
                "{tool} stopped on an internal fault ({join_error}). The server is still \
                 running and other calls are unaffected."
            ),
            suggested_command: Some(
                "Do not repeat the same call; try a narrower one or another tool, and tell \
                 the user this one failed."
                    .to_string(),
            ),
            details: None,
        }),
        tool,
        result: None,
        meta: ToolCallMeta {
            elapsed_ms: 0,
            started_utc: None,
            finished_utc: None,
        },
    }
}

fn json_value_to_object(v: &Value) -> JsonObject {
    v.as_object().cloned().unwrap_or_default()
}

/// Reduce generated JSON Schema to the conservative subset accepted reliably
/// by tool-calling providers. In JSON Schema, an optional property is already
/// optional by being absent from `required`; advertising it as both its real
/// type and `null` adds no value to MCP callers and has caused some streaming
/// clients to construct nested or truncated argument objects.
fn mcp_input_schema(schema: &Value) -> JsonObject {
    let mut schema = schema.clone();
    if let Value::Object(root) = &mut schema {
        root.remove("$schema");
        if let Some(Value::Object(properties)) = root.get_mut("properties") {
            describe_common_parameters(properties);
        }
    }
    simplify_mcp_input_schema(&mut schema);
    json_value_to_object(&schema)
}

/// What a parameter name means wherever it appears. A request field with its
/// own doc comment keeps that; this fills in the many that have none, so a
/// model is not left to guess what `scope_canon` or `seed` takes.
const COMMON_PARAMETERS: &[(&[&str], &str)] = &[
    (
        &["phrase", "term1", "term2", "quote"],
        "Exact text to find. CJK matches character for character; English matches whole \
         words, ignoring case and punctuation, with no stemming.",
    ),
    (
        &["phrases"],
        "Phrases to search, each matched exactly as `search` matches its phrase.",
    ),
    (
        &["passage_id", "seed", "seed_passage_id"],
        "A complete passage_id exactly as a search tool returned it, `#anchor` included. \
         Not a work id and not search text.",
    ),
    (
        &[
            "source_work_id",
            "scope_source_work_id",
            "scope_work_id",
            "work_id",
            "scope_a_work_id",
            "scope_b_work_id",
        ],
        "One work: a `work_id` from `works`, or the `source_work_id` of a hit.",
    ),
    (
        &["canon", "scope_canon", "scope_a_canon", "scope_b_canon"],
        "Canon code from the catalog (CBETA corpora, e.g. T or X). Leave unset for \
         collections that carry no canon, such as documents added through document intake.",
    ),
    (
        &["period", "scope_period", "scope_a_period", "scope_b_period"],
        "Period name or id from the catalog (CBETA corpora). Leave unset for collections \
         that carry no period, such as documents added through document intake.",
    ),
    (
        &["tradition"],
        "Tradition name or id from the catalog (CBETA corpora). Leave unset otherwise.",
    ),
    (
        &["origin"],
        "Origin name or id from the catalog (CBETA corpora). Leave unset otherwise.",
    ),
    (&["author"], "Part of the work's author name, any case."),
    (&["title"], "Part of the work's title, any case."),
    (
        &["node_id", "scope_node_id", "scope_a_node_id", "scope_b_node_id"],
        "A catalog outline node: the `node_id` of a cluster or outline result.",
    ),
    (&["limit", "k"], "Most results to return."),
    (&["limit_total"], "Most hits to examine in all."),
    (
        &["limit_per_group", "limit_per_cluster"],
        "Most sample passages to return for each group.",
    ),
    (
        &["max_candidates", "max_candidates_per_term"],
        "Most candidate passages to check before stopping; raise it only when a result \
         says it may be truncated.",
    ),
    (
        &["max_elapsed_ms"],
        "Time budget for the whole call, in milliseconds.",
    ),
    (
        &["max_component_ms"],
        "Time budget for each step of the call, in milliseconds.",
    ),
    (
        &["max_chars", "max_context_chars"],
        "Character budget for the text returned.",
    ),
    (
        &["brief"],
        "true returns short quotes instead of full passage text.",
    ),
    (
        &["gram_len"],
        "Length of a term: characters for CJK text, words for English text.",
    ),
    (
        &["window_chars"],
        "How far apart, in characters, the two things may be.",
    ),
    (
        &["include_variants", "allow_variants"],
        "Also match CJK variant spellings.",
    ),
    (
        &["heading_path_prefix"],
        "Only passages whose heading path starts with this text.",
    ),
];

fn describe_common_parameters(properties: &mut JsonObject) {
    for (name, schema) in properties.iter_mut() {
        let Value::Object(schema) = schema else {
            continue;
        };
        if schema.contains_key("description") {
            continue;
        }
        if let Some((_, text)) = COMMON_PARAMETERS
            .iter()
            .find(|(names, _)| names.contains(&name.as_str()))
        {
            schema.insert(
                "description".to_string(),
                Value::String((*text).to_string()),
            );
        }
    }
}

fn is_null_schema(value: &Value) -> bool {
    value
        .as_object()
        .and_then(|schema| schema.get("type"))
        .and_then(Value::as_str)
        == Some("null")
}

fn simplify_mcp_input_schema(value: &mut Value) {
    match value {
        Value::Array(values) => {
            for value in values {
                simplify_mcp_input_schema(value);
            }
        }
        Value::Object(object) => {
            if object.get("default").is_some_and(Value::is_null) {
                object.remove("default");
            }

            let scalar_type = match object.get_mut("type") {
                Some(Value::Array(types)) => {
                    types.retain(|kind| kind.as_str() != Some("null"));
                    (types.len() == 1).then(|| types[0].clone())
                }
                _ => None,
            };
            if let Some(scalar_type) = scalar_type {
                object.insert("type".to_string(), scalar_type);
            }

            for keyword in ["anyOf", "oneOf"] {
                let replacement =
                    object
                        .get(keyword)
                        .and_then(Value::as_array)
                        .and_then(|alternatives| {
                            let non_null = alternatives
                                .iter()
                                .filter(|alternative| !is_null_schema(alternative))
                                .collect::<Vec<_>>();
                            (non_null.len() == 1 && non_null.len() < alternatives.len())
                                .then(|| non_null[0].clone())
                        });
                if let Some(Value::Object(replacement)) = replacement {
                    object.remove(keyword);
                    for (key, value) in replacement {
                        object.entry(key).or_insert(value);
                    }
                }
            }

            for child in object.values_mut() {
                simplify_mcp_input_schema(child);
            }
        }
        _ => {}
    }
}

impl ServerHandler for SinoragMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            // Not `from_build_env()`: that reports the rmcp crate's own name
            // and version, which is what clients then show for this server.
            .with_server_info(
                Implementation::new("sinorag", env!("CARGO_PKG_VERSION")).with_title("SinoRAG"),
            )
            // The newest version this rmcp speaks. rmcp answers with the
            // older of this and the client's, so naming an old version here
            // would hold every newer client back to it.
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_instructions(self.instructions.clone())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult {
            tools: self.tools.clone(),
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let name = request.name.to_string();
        let args = request
            .arguments
            .map(Value::Object)
            .unwrap_or(Value::Object(Default::default()));

        // Run the tool on its own task so that a panic in one call is an
        // error result for that call. Left to unwind, it would take the
        // request with it: the client waits on a reply that never comes, or
        // sees the transport close.
        let engine = Arc::clone(&self.engine);
        let tool = name.clone();
        let envelope =
            match tokio::spawn(
                async move { call_tool_enveloped(&engine, None, tool, args).await },
            )
            .await
            {
                Ok(envelope) => envelope,
                Err(join_error) => crashed_envelope(name, &join_error),
            };

        let payload = serde_json::to_value(&envelope).map_err(|e| {
            McpError::internal_error(format!("failed to serialize tool envelope: {e}"), None)
        })?;
        // Keep the text fallback compact. Modern clients consume
        // `structured_content`; older clients still receive equivalent JSON
        // without paying the whitespace cost on large research responses.
        let text = serde_json::to_string(&payload).unwrap_or_else(|_| payload.to_string());

        let mut result = if envelope.ok {
            CallToolResult::success(vec![Content::text(text)])
        } else {
            CallToolResult::error(vec![Content::text(text)])
        };
        result.structured_content = Some(payload);
        Ok(result)
    }
}

pub async fn run(config: EngineConfig) -> Result<()> {
    let engine = Arc::new(ToolEngine::open(config).await?);

    // Emit an early diagnostic on stderr if no passages parquet is reachable.
    // The server still starts (tools that don't need a pack — `query-expand-terms`
    // and `tool-docs` — remain useful), but the operator should see
    // this before the model starts calling search tools that will all fail.
    match engine.resolve_passages_path() {
        Ok(path) if path.exists() => {
            tracing::info!("sinorag MCP: passages parquet = {}", path.display());
        }
        Ok(path) => {
            eprintln!(
                "[sinorag mcp] WARNING: no passages.parquet at {}. Most tools \
                 will fail with `missing_passages`. Run `sinorag ingest <source> \
                 <path>` or pass --pack/--passages-parquet to point at a built pack.",
                path.display()
            );
        }
        Err(e) => {
            eprintln!(
                "[sinorag mcp] WARNING: could not resolve a passages parquet path ({e}). \
                 Pass --pack <dir> or --passages-parquet <file> to enable corpus tools."
            );
        }
    }

    let server = SinoragMcpServer::new(engine);

    tracing::info!("starting sinorag MCP stdio server");
    let service = server.serve(stdio()).await.map_err(|e| {
        tracing::error!("MCP serve error: {e:?}");
        anyhow::anyhow!("MCP serve error: {e}")
    })?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::requests::{SourceReadRequest, WorkMetadataRequest, WorksRequest};
    use crate::tools::spec::schema_for;

    fn assert_no_nullable_type(value: &Value) {
        match value {
            Value::Array(values) => {
                for value in values {
                    assert_no_nullable_type(value);
                }
            }
            Value::Object(object) => {
                assert_ne!(
                    object.get("type").and_then(Value::as_str),
                    Some("null"),
                    "MCP schema retained a null alternative: {value}"
                );
                if let Some(Value::Array(types)) = object.get("type") {
                    assert!(
                        !types.iter().any(|kind| kind.as_str() == Some("null")),
                        "MCP schema retained a nullable type: {value}"
                    );
                }
                for value in object.values() {
                    assert_no_nullable_type(value);
                }
            }
            _ => {}
        }
    }

    /// A CBETA root with every index: the longest list there is.
    const EVERYTHING: Listing = Listing {
        documents_only: false,
        has_vector_index: true,
    };

    fn listed(config: &EngineConfig) -> Vec<Tool> {
        SinoragMcpServer::build_tools(config, EVERYTHING)
    }

    #[test]
    fn a_documents_collection_is_not_offered_the_catalogue_tools() {
        let names = |listing: Listing| -> Vec<String> {
            SinoragMcpServer::build_tools(&EngineConfig::default(), listing)
                .iter()
                .map(|t| t.name.to_string())
                .collect()
        };
        let everything = names(EVERYTHING);
        // Every name in the list is a real tool: a rename must not leave it
        // hiding nothing.
        for tool in CATALOGUE_TOOLS.iter().chain(&["vector-neighbors"]) {
            assert!(everything.contains(&tool.to_string()), "{tool} is not a tool");
        }
        let documents = names(Listing {
            documents_only: true,
            has_vector_index: false,
        });
        for tool in CATALOGUE_TOOLS.iter().chain(&["vector-neighbors"]) {
            assert!(!documents.contains(&tool.to_string()), "{tool} is offered");
        }
        for tool in [
            "works",
            "search",
            "batch-evidence-search",
            "passage",
            "source-read",
            "similar",
            "frontier",
            "citation-verify",
        ] {
            assert!(documents.contains(&tool.to_string()), "{tool} is missing");
        }
        assert_eq!(everything.len() - documents.len(), CATALOGUE_TOOLS.len() + 1);
    }

    /// A subschema written as a bare `true` or `false`. The official
    /// TypeScript SDK client rejects the whole tool list over one.
    fn boolean_subschemas(value: &Value, path: &str, found: &mut Vec<String>) {
        match value {
            Value::Object(object) => {
                for (key, child) in object {
                    let here = format!("{path}/{key}");
                    let holds_schemas = matches!(key.as_str(), "properties" | "$defs" | "definitions");
                    match child {
                        Value::Object(members) if holds_schemas => {
                            for (name, member) in members {
                                if member.is_boolean() {
                                    found.push(format!("{here}/{name}"));
                                } else {
                                    boolean_subschemas(member, &format!("{here}/{name}"), found);
                                }
                            }
                        }
                        Value::Bool(_) if key == "items" => found.push(here),
                        _ => boolean_subschemas(child, &here, found),
                    }
                }
            }
            Value::Array(values) => {
                for (index, child) in values.iter().enumerate() {
                    boolean_subschemas(child, &format!("{path}/{index}"), found);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn listed_tools_are_ones_every_client_can_load() {
        let tools = listed(&EngineConfig::default());
        assert!(tools.len() > 20);
        for tool in &tools {
            assert!(
                tool.output_schema.is_none(),
                "{} advertises an output schema; its result is wrapped in the envelope",
                tool.name
            );
            let mut found = Vec::new();
            boolean_subschemas(
                &Value::Object((*tool.input_schema).clone()),
                tool.name.as_ref(),
                &mut found,
            );
            assert!(found.is_empty(), "boolean subschemas: {found:?}");
            // The pattern stricter hosts apply to tool names.
            let mut chars = tool.name.chars();
            assert!(chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_'));
            assert!(chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'));
            assert!(tool.name.len() <= 64);
        }
    }

    #[test]
    fn tools_the_session_cannot_run_are_not_listed() {
        let names = |config: &EngineConfig| -> Vec<String> {
            listed(config).iter().map(|t| t.name.to_string()).collect()
        };
        let read_only = names(&EngineConfig {
            readonly: true,
            ..EngineConfig::default()
        });
        assert!(read_only.contains(&"search".to_string()));
        for writer in ["pdf-build", "run-batch", "graph-build", "report-build"] {
            assert!(!read_only.contains(&writer.to_string()), "{writer} is listed");
        }
        let writable = names(&EngineConfig {
            readonly: false,
            ..EngineConfig::default()
        });
        assert!(writable.contains(&"pdf-build".to_string()));
    }

    #[test]
    fn common_parameters_are_described() {
        let search = mcp_input_schema(&schema_for::<crate::tools::requests::SearchRequest>());
        // From the field's own doc comment.
        assert!(search["properties"]["phrase"]["description"]
            .as_str()
            .unwrap()
            .contains("no stemming"));
        assert!(search["properties"]["mode"]["description"]
            .as_str()
            .unwrap()
            .contains("clusters"));
        // From the shared table.
        assert!(search["properties"]["canon"]["description"]
            .as_str()
            .unwrap()
            .contains("Leave unset"));

        let mut described = 0usize;
        let mut total = 0usize;
        for tool in listed(&EngineConfig::default()) {
            if let Some(Value::Object(properties)) = tool.input_schema.get("properties") {
                for schema in properties.values() {
                    total += 1;
                    described += schema.get("description").is_some() as usize;
                }
            }
        }
        assert!(
            described * 10 >= total * 7,
            "only {described} of {total} tool parameters are described"
        );
    }

    #[test]
    fn a_document_collection_gets_the_documents_doctrine() {
        let root = tempfile::tempdir().unwrap();
        let parquet = root.path().join("passages.parquet");
        std::fs::create_dir_all(parquet.join("source_corpus=documents")).unwrap();

        // No profile recorded: the research doctrine, unchanged.
        assert_eq!(doctrine_for(&parquet, "RESEARCH"), "RESEARCH");

        crate::normalize::record_text_profile(&parquet, "documents", true).unwrap();
        let documents = doctrine_for(&parquet, "RESEARCH");
        assert!(documents.starts_with("# SinoRAG: the user's documents"));
        assert!(documents.contains("Searching English"));
        assert!(!documents.contains("RESEARCH"));

        // CBETA beside it: research doctrine plus the word-search section.
        std::fs::create_dir_all(parquet.join("source_corpus=cbeta")).unwrap();
        let mixed = doctrine_for(&parquet, "RESEARCH");
        assert!(mixed.starts_with("RESEARCH"));
        assert!(mixed.contains("Searching English"));
    }

    #[test]
    fn mcp_input_schemas_remove_nullable_optional_types() {
        for schema in [
            schema_for::<WorksRequest>(),
            schema_for::<SourceReadRequest>(),
        ] {
            let schema = Value::Object(mcp_input_schema(&schema));
            assert_no_nullable_type(&schema);
            assert!(schema.get("$schema").is_none());
        }

        let works = mcp_input_schema(&schema_for::<WorksRequest>());
        assert_eq!(works["properties"]["work_id"]["type"], "string");
        assert!(works["properties"]["work_id"].get("default").is_none());
        assert_eq!(works["properties"]["title"]["type"], "string");

        let metadata = mcp_input_schema(&schema_for::<WorkMetadataRequest>());
        assert_eq!(metadata["required"], serde_json::json!(["work_ids"]));
        assert_eq!(metadata["properties"]["work_ids"]["type"], "array");
        assert_eq!(
            metadata["properties"]["work_ids"]["items"]["type"],
            "string"
        );
    }
}
