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

use crate::tools::{call_tool_enveloped, tool_defs, EngineConfig, ToolAudience, ToolEngine};

/// Embedded doctrine shown to the model as the MCP server's `instructions`
/// string. The wrapping `agent` launcher uses the same fragments to build
/// `AGENTS.md`; here we keep a compact version so the model has the lens
/// guidance even when invoked from a non-opencode client.
const SERVER_INSTRUCTIONS: &str = include_str!("../agent/doctrine/mcp_instructions.md");

pub struct SinoragMcpServer {
    engine: Arc<ToolEngine>,
    tools: Vec<Tool>,
    /// Required by the `ServerHandler` blanket impls even though we never
    /// route through it (we override `list_tools`/`call_tool` directly).
    _router: ToolRouter<Self>,
}

impl SinoragMcpServer {
    fn build_tools() -> Vec<Tool> {
        tool_defs()
            .into_iter()
            .filter(|def| def.spec.audience != ToolAudience::InternalDebug)
            .map(|def| {
                let input_schema = Arc::new(mcp_input_schema(&def.spec.input_schema));
                let output_schema = json_value_to_object(&def.spec.output_schema);
                let mut tool = Tool::new(
                    Cow::Borrowed(def.spec.name),
                    Cow::Borrowed(def.spec.description),
                    input_schema,
                );
                if !output_schema.is_empty() {
                    tool = tool.with_raw_output_schema(Arc::new(output_schema));
                }
                tool
            })
            .collect()
    }

    pub fn new(engine: Arc<ToolEngine>) -> Self {
        Self {
            engine,
            tools: Self::build_tools(),
            _router: ToolRouter::new(),
        }
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
    }
    simplify_mcp_input_schema(&mut schema);
    json_value_to_object(&schema)
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
            .with_server_info(Implementation::from_build_env())
            .with_protocol_version(ProtocolVersion::V_2024_11_05)
            .with_instructions(SERVER_INSTRUCTIONS)
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

        let envelope = call_tool_enveloped(&self.engine, None, name, args).await;

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
                 will fail with `missing_artifact`. Run `sinorag ingest <source> \
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
