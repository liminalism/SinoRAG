use anyhow::Result;
use serde_json::Value;

use crate::tools::engine::ToolEngine;
use crate::tools::errors::{classify_tool_error, ToolError};
use crate::tools::requests::*;
use crate::tools::responses::*;
use crate::tools::spec::{schema_for, ToolAudience, ToolDef, ToolExample, ToolSafety, ToolSpec};

/// Standard response envelope for tool calls
#[derive(Debug, serde::Serialize)]
pub struct ToolCallEnvelope {
    pub id: Option<String>,
    pub ok: bool,
    pub tool: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<crate::tools::errors::ToolErrorBody>,

    pub meta: ToolCallMeta,
}

#[derive(Debug, serde::Serialize)]
pub struct ToolCallMeta {
    pub elapsed_ms: u128,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_utc: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_utc: Option<String>,
}

/// Enforce safety rules for a tool
fn enforce_safety(engine: &ToolEngine, spec: &ToolSpec) -> Result<()> {
    match spec.safety {
        ToolSafety::ReadOnly => Ok(()),

        ToolSafety::WritesOutput => {
            if engine.config.readonly {
                return Err(ToolError::ReadonlyViolation {
                    tool: spec.name.to_string(),
                }
                .into_anyhow());
            }
            Ok(())
        }

        ToolSafety::MutatesRegistry | ToolSafety::Admin => {
            if !engine.config.allow_admin_tools {
                return Err(ToolError::AdminToolDisabled {
                    tool: spec.name.to_string(),
                }
                .into_anyhow());
            }
            Ok(())
        }
    }
}

/// Call a tool by name with JSON arguments
pub async fn call_tool(engine: &ToolEngine, name: &str, args: Value) -> Result<Value> {
    let defs = tool_defs();

    let def = defs
        .iter()
        .find(|d| d.spec.name == name)
        .ok_or_else(|| ToolError::unknown_tool(name).into_anyhow())?;

    enforce_safety(engine, &def.spec)?;

    let mut result = (def.call)(engine, args).await?;
    crate::dict::annotate_response(&mut result).await;
    Ok(result)
}

/// Call a tool with envelope response
pub async fn call_tool_enveloped(
    engine: &ToolEngine,
    id: Option<String>,
    tool: String,
    args: Value,
) -> ToolCallEnvelope {
    let started = std::time::Instant::now();

    match call_tool(engine, &tool, args.clone()).await {
        Ok(result) => {
            let elapsed_ms = started.elapsed().as_millis();
            crate::tools::log::append_call(&tool, &args, Some(&result), None, elapsed_ms);
            ToolCallEnvelope {
                id,
                ok: true,
                tool,
                result: Some(result),
                error: None,
                meta: ToolCallMeta {
                    elapsed_ms,
                    started_utc: None,
                    finished_utc: None,
                },
            }
        }

        Err(err) => {
            let elapsed_ms = started.elapsed().as_millis();
            let error = classify_tool_error(&err);
            crate::tools::log::append_call(&tool, &args, None, Some(&error), elapsed_ms);
            ToolCallEnvelope {
                id,
                ok: false,
                tool,
                result: None,
                error: Some(error),
                meta: ToolCallMeta {
                    elapsed_ms,
                    started_utc: None,
                    finished_utc: None,
                },
            }
        }
    }
}

pub fn audience_for_tool(name: &str) -> ToolAudience {
    tool_defs()
        .into_iter()
        .find(|def| def.spec.name == name)
        .map(|def| def.spec.audience)
        .unwrap_or(ToolAudience::Specialist)
}

/// Get all tool definitions
pub fn tool_defs() -> Vec<ToolDef> {
    let mut defs = vec![
        // Status tool
        ToolDef {
            spec: ToolSpec {
                name: "status",
                audience: ToolAudience::DefaultAgent,
                description: "Show what's been ingested and which indexes are built under the data root.",
                input_schema: schema_for::<StatusRequest>(),
                output_schema: schema_for::<StatusResponse>(),
                requires: vec![],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Check system status",
                        args: serde_json::json!({}),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let _req: StatusRequest = serde_json::from_value(args)?;
                let res = engine.status_impl().await?;
                Ok(serde_json::to_value(res)?)
            }),
        },
        // Tool docs tool
        ToolDef {
            spec: ToolSpec {
                name: "tool-docs",
                audience: ToolAudience::Specialist,
                description: "Return compiled-in documentation for all tools or one named tool.",
                input_schema: schema_for::<ToolDocsRequest>(),
                output_schema: schema_for::<ToolDocsResponse>(),
                requires: vec![],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Show all tool docs",
                        args: serde_json::json!({}),
                    },
                    ToolExample {
                        title: "Show search docs",
                        args: serde_json::json!({ "tool": "search" }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: ToolDocsRequest = serde_json::from_value(args)?;
                let res = engine.tool_docs_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Tool log summary tool
        ToolDef {
            spec: ToolSpec {
                name: "tool-log-summary",
                audience: ToolAudience::Specialist,
                description: "Summarize local cross-session tool-call logs with compact performance and success/failure aggregates. Specialist: use when deciding which tools have worked reliably in this environment.",
                input_schema: schema_for::<ToolLogSummaryRequest>(),
                output_schema: schema_for::<ToolLogSummaryResponse>(),
                requires: vec![],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Summarize recent local tool calls",
                        args: serde_json::json!({ "recent": 20 }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: ToolLogSummaryRequest = serde_json::from_value(args)?;
                let res = engine.tool_log_summary_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Passage tool
        ToolDef {
            spec: ToolSpec {
                name: "passage",
                audience: ToolAudience::Specialist,
                description: "Retrieve one passage by passage_id.",
                input_schema: schema_for::<PassageRequest>(),
                output_schema: schema_for::<PassageResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Fetch seed passage",
                        args: serde_json::json!({
                            "id": "B/B13/B13n0079.xml#pB13p0047a0417"
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: PassageRequest = serde_json::from_value(args)?;
                let res = engine.passage_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Source-read tool
        ToolDef {
            spec: ToolSpec {
                name: "source-read",
                audience: ToolAudience::DefaultAgent,
                description: "Read an ordered source stream in cursor-based, citation-aware chunks. Use one compact anchor: source_work_id to start a work, passage_id (including its JSON-safe #anchor) to read around a hit, or cursor to continue.",
                input_schema: schema_for::<SourceReadRequest>(),
                output_schema: schema_for::<SourceReadResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Start reading a work",
                        args: serde_json::json!({
                            "source_work_id": "T08n0235",
                            "max_chars": 4000
                        }),
                    },
                    ToolExample {
                        title: "Read around a passage",
                        args: serde_json::json!({
                            "passage_id": "T/T08/T08n0235.xml#pT08p0750c0201",
                            "before_chars": 1200,
                            "after_chars": 1800
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: SourceReadRequest = serde_json::from_value(args)?;
                let res = engine.source_read_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Search tool
        ToolDef {
            spec: ToolSpec {
                name: "search",
                audience: ToolAudience::Specialist,
                description: "Exact phrase search across loaded passage text. Uses the phrase index when available, verifies candidates against parquet text, and falls back to a parquet scan if no index/doc table is available. Optional modes add work/division clusters or term-usage traces.",
                input_schema: schema_for::<SearchRequest>(),
                output_schema: schema_for::<SearchResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Search for exact phrase hits",
                        args: serde_json::json!({
                            "phrase": "金剛經云",
                            "limit": 5
                        }),
                    },
                    ToolExample {
                        title: "Search and cluster hits by work",
                        args: serde_json::json!({
                            "phrase": "雪峯辭洞山",
                            "mode": "clusters",
                            "group_by": "work",
                            "limit": 50,
                            "limit_per_group": 10
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: SearchRequest = serde_json::from_value(args)?;
                let res = engine.search_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },
        // Heading-search tool
        ToolDef {
            spec: ToolSpec {
                name: "heading-search",
                audience: ToolAudience::Specialist,
                description: "Search heading and section metadata by title/path, with passage-text fallback.",
                input_schema: schema_for::<HeadingSearchRequest>(),
                output_schema: schema_for::<HeadingSearchResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Find sections headed by a case title",
                        args: serde_json::json!({
                            "query": "雪峰過嶺",
                            "limit": 10,
                            "brief": true
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: HeadingSearchRequest = serde_json::from_value(args)?;
                let res = engine.heading_search_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Canonical-source tool
        ToolDef {
            spec: ToolSpec {
                name: "canonical-source",
                audience: ToolAudience::Specialist,
                description: "Find canon-side source passages for a phrase.",
                input_schema: schema_for::<CanonicalSourceRequest>(),
                output_schema: schema_for::<CanonicalSourceResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Find canonical sources for Diamond Sutra phrase",
                        args: serde_json::json!({
                            "phrase": "一切有為法如夢幻泡影",
                            "limit": 50
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: CanonicalSourceRequest = serde_json::from_value(args)?;
                let res = engine.canonical_source_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Validate-adjudication tool
        ToolDef {
            spec: ToolSpec {
                name: "validate-adjudication",
                audience: ToolAudience::Specialist,
                description: "Validate a finished adjudication JSON file (your written-up scholarly verdict) for structural correctness. A late-pipeline check, not a precondition for exploration: write your adjudication after investigating, then run this before graph-build/report-build.",
                input_schema: schema_for::<ValidateAdjudicationRequest>(),
                output_schema: schema_for::<ValidateAdjudicationResponse>(),
                requires: vec![],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Validate adjudication file",
                        args: serde_json::json!({
                            "path": "GraphDiscovery/Runs/text-reuse-discovery/adjudications/test3.json"
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: ValidateAdjudicationRequest = serde_json::from_value(args)?;
                let res = engine.validate_adjudication_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Graph-build tool
        ToolDef {
            spec: ToolSpec {
                name: "graph-build",
                audience: ToolAudience::Specialist,
                description: "Build an evidence graph from a finished adjudication JSON file. A terminal artifact-building step for after you've investigated and written up your findings — not something you need before exploring with frontier/source-investigate.",
                input_schema: schema_for::<GraphBuildRequest>(),
                output_schema: schema_for::<GraphBuildResponse>(),
                requires: vec![],
                safety: ToolSafety::WritesOutput,
                examples: vec![
                    ToolExample {
                        title: "Build evidence graph",
                        args: serde_json::json!({
                            "input": "GraphDiscovery/Runs/text-reuse-discovery/adjudications/test3.json",
                            "kind": "evidence",
                            "name": "test3",
                            "out": "GraphDiscovery/Runs/text-reuse-discovery/drafts/test3.graph-draft.json"
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: GraphBuildRequest = serde_json::from_value(args)?;
                let res = engine.graph_build_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Report-build tool
        ToolDef {
            spec: ToolSpec {
                name: "report-build",
                audience: ToolAudience::Specialist,
                description: "Assemble a markdown report from finished adjudication and graph artifact files. For writing up raw markdown directly without an adjudication, use pdf-build's input_markdown instead — this tool is for the structured-evidence pipeline's final assembly step.",
                input_schema: schema_for::<ReportBuildRequest>(),
                output_schema: schema_for::<ReportBuildResponse>(),
                requires: vec![],
                safety: ToolSafety::WritesOutput,
                examples: vec![
                    ToolExample {
                        title: "Build markdown report",
                        args: serde_json::json!({
                            "inputs": [
                                "GraphDiscovery/Runs/text-reuse-discovery/adjudications/test3.json",
                                "GraphDiscovery/Runs/text-reuse-discovery/drafts/test3.graph-draft.json"
                            ],
                            "out": "GraphDiscovery/Runs/text-reuse-discovery/dossiers/test3.report.md",
                            "title": "Canonical Dependence Test 3"
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: ReportBuildRequest = serde_json::from_value(args)?;
                let res = engine.report_build_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },
        // PDF-build tool
        ToolDef {
            spec: ToolSpec {
                name: "pdf-build",
                audience: ToolAudience::DefaultAgent,
                description: "Build a PDF with the built-in Lopdf renderer. Pass raw model-authored prose in `markdown`, a Markdown file path in `input_markdown`, or structured report/evidence JSON in `input_json`. No adjudication pipeline or external PDF tools are required.",
                input_schema: schema_for::<PdfBuildRequest>(),
                output_schema: schema_for::<PdfBuildResponse>(),
                requires: vec![],
                safety: ToolSafety::WritesOutput,
                examples: vec![
                    ToolExample {
                        title: "Build PDF from structured report JSON",
                        args: serde_json::json!({
                            "input_json": "GraphDiscovery/Runs/text-reuse-discovery/dossiers/test3.report.json",
                            "out": "GraphDiscovery/Runs/text-reuse-discovery/dossiers/test3.report.pdf",
                            "title": "Canonical Dependence Test 3"
                        }),
                    },
                    ToolExample {
                        title: "Build PDF from a Markdown file",
                        args: serde_json::json!({
                            "input_markdown": "GraphDiscovery/Runs/text-reuse-discovery/dossiers/test3.report.md",
                            "out": "GraphDiscovery/Runs/text-reuse-discovery/dossiers/test3.report.pdf",
                            "side_by_side": true
                        }),
                    },
                    ToolExample {
                        title: "Build PDF directly from model-authored prose",
                        args: serde_json::json!({
                            "markdown": "# Finding\n\nThe evidence supports a qualified conclusion.",
                            "out": "output/finding.pdf"
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: PdfBuildRequest = serde_json::from_value(args)?;
                let res = engine.pdf_build_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Work metadata tool
        ToolDef {
            spec: ToolSpec {
                name: "work-metadata",
                audience: ToolAudience::DefaultAgent,
                description: "Fetch catalog metadata for one or more exact work IDs in one call. Use required work_ids for known IDs; prefer this over parallel works calls.",
                input_schema: schema_for::<WorkMetadataRequest>(),
                output_schema: schema_for::<WorkMetadataResponse>(),
                requires: vec!["catalog.index"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Fetch metadata for two known works",
                        args: serde_json::json!({
                            "work_ids": ["T48n2002A", "T48n2002B"]
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: WorkMetadataRequest = serde_json::from_value(args)?;
                let res = engine.work_metadata_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Works tool
        ToolDef {
            spec: ToolSpec {
                name: "works",
                audience: ToolAudience::Specialist,
                description: "List or filter catalog works by tradition/period/canon or normalized title/author substring. A legacy single work_id lookup remains supported, but use work-metadata for known IDs and multi-ID retrieval.",
                input_schema: schema_for::<WorksRequest>(),
                output_schema: schema_for::<WorksResponse>(),
                requires: vec!["catalog.index"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Look up one work's metadata by ID",
                        args: serde_json::json!({
                            "work_id": "X26n0534"
                        }),
                    },
                    ToolExample {
                        title: "List all works",
                        args: serde_json::json!({
                            "limit": 50
                        }),
                    },
                    ToolExample {
                        title: "Filter by tradition",
                        args: serde_json::json!({
                            "tradition": "canon",
                            "limit": 50
                        }),
                    },
                    ToolExample {
                        title: "Find works by normalized title spelling",
                        args: serde_json::json!({
                            "title": "万松"
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: WorksRequest = serde_json::from_value(args)?;
                let res = engine.works_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Catalog-index-info tool
        ToolDef {
            spec: ToolSpec {
                name: "catalog-index-info",
                audience: ToolAudience::InternalDebug,
                description: "Show catalog index metadata.",
                input_schema: schema_for::<CatalogIndexInfoRequest>(),
                output_schema: schema_for::<CatalogIndexInfoResponse>(),
                requires: vec!["catalog.index"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Get catalog index info",
                        args: serde_json::json!({}),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let _req: CatalogIndexInfoRequest = serde_json::from_value(args)?;
                let res = engine.catalog_index_info_impl(_req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Vector-info tool
        ToolDef {
            spec: ToolSpec {
                name: "vector-info",
                audience: ToolAudience::InternalDebug,
                description: "Show vector index metadata and doc-table compatibility.",
                input_schema: schema_for::<VectorInfoRequest>(),
                output_schema: schema_for::<VectorInfoResponse>(),
                requires: vec!["vector.index", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Get vector index info",
                        args: serde_json::json!({}),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: VectorInfoRequest = serde_json::from_value(args)?;
                let res = engine.vector_info_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Vector-neighbors tool
        ToolDef {
            spec: ToolSpec {
                name: "vector-neighbors",
                audience: ToolAudience::Specialist,
                description: "Find raw semantic neighbor candidates from a seed passage or external query embedding. Results are discovery candidates, not exact evidence. Specialist: prefer frontier for ordinary seed expansion; call this directly only when raw vector neighbors are specifically needed.",
                input_schema: schema_for::<VectorNeighborsRequest>(),
                output_schema: schema_for::<VectorNeighborsResponse>(),
                requires: vec!["vector.index", "doc_table.bin", "passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Find seed passage vector neighbors",
                        args: serde_json::json!({
                            "seed_passage_id": "B/B13/B13n0079.xml#pB13p0047a0417",
                            "k": 25
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: VectorNeighborsRequest = serde_json::from_value(args)?;
                let res = engine.vector_neighbors_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Similar tool
        ToolDef {
            spec: ToolSpec {
                name: "similar",
                audience: ToolAudience::Specialist,
                description: "Find TF-IDF similar passages to a seed passage. Specialist lexical primitive: frontier wraps this with phrase-frontier discovery; call directly when you want TF-IDF parallels alone.",
                input_schema: schema_for::<SimilarRequest>(),
                output_schema: schema_for::<SimilarResponse>(),
                requires: vec!["passages.parquet", "tfidf.index", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Find similar passages",
                        args: serde_json::json!({
                            "seed": "B/B13/B13n0079.xml#pB13p0047a0417",
                            "limit": 25
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: SimilarRequest = serde_json::from_value(args)?;
                let res = engine.similar_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Frontier tool
        ToolDef {
            spec: ToolSpec {
                name: "frontier",
                audience: ToolAudience::Specialist,
                description: "Generate a discovery frontier packet for an agent session. Specialist: source-investigate wraps this — call directly only when you want the raw discovery-frontier packet. Use min_similarity (0.2–0.4) to suppress noisy tangential matches; use scope_canon/scope_period to restrict to a specific corpus section.",
                input_schema: schema_for::<FrontierRequest>(),
                output_schema: schema_for::<FrontierResponse>(),
                requires: vec!["passages.parquet", "tfidf.index", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Generate frontier packet",
                        args: serde_json::json!({
                            "seed": "B/B13/B13n0079.xml#pB13p0047a0417",
                            "limit": 25,
                            "phrase_limit": 20
                        }),
                    },
                    ToolExample {
                        title: "Filtered frontier: Tang-period Chan texts, similarity ≥ 0.3",
                        args: serde_json::json!({
                            "seed": "B/B13/B13n0079.xml#pB13p0047a0417",
                            "limit": 25,
                            "min_similarity": 0.3,
                            "scope_period": ["Tang"]
                        }),
                    },
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: FrontierRequest = serde_json::from_value(args)?;
                let res = engine.frontier_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // First-attestation tool
        ToolDef {
            spec: ToolSpec {
                name: "first-attestation",
                audience: ToolAudience::Specialist,
                description: "Find the earliest attestation of a phrase, ordered by period_rank. Specialist: evidence-search runs this as its `include_attestation` step — call directly only when the phrase is already verified and you need just the earliest attestation.",
                input_schema: schema_for::<FirstAttestationRequest>(),
                output_schema: schema_for::<FirstAttestationResponse>(),
                requires: vec!["passages.parquet", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Find earliest attestation",
                        args: serde_json::json!({
                            "phrase": "一切有為法如夢幻泡影",
                            "limit": 25
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: FirstAttestationRequest = serde_json::from_value(args)?;
                let res = engine.first_attestation_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Phrase-history tool
        ToolDef {
            spec: ToolSpec {
                name: "phrase-history",
                audience: ToolAudience::Specialist,
                description: "Analyze the historical distribution of a phrase across periods, canons, and traditions. Specialist: evidence-search (`include_history`) and source-investigate wrap this — call directly only for a standalone distribution.",
                input_schema: schema_for::<PhraseHistoryRequest>(),
                output_schema: schema_for::<PhraseHistoryResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Analyze phrase history",
                        args: serde_json::json!({
                            "phrase": "一切有為法",
                            "timeline": true
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: PhraseHistoryRequest = serde_json::from_value(args)?;
                let res = engine.phrase_history_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Phrase-index-search tool
        ToolDef {
            spec: ToolSpec {
                name: "phrase-index-search",
                audience: ToolAudience::InternalDebug,
                description: "Search for a phrase using the phrase index for fast lookup.",
                input_schema: schema_for::<PhraseIndexSearchRequest>(),
                output_schema: schema_for::<PhraseIndexSearchResponse>(),
                requires: vec!["passages.parquet", "phrase.index"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Search phrase index",
                        args: serde_json::json!({
                            "phrase": "一切有為法",
                            "limit": 25
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: PhraseIndexSearchRequest = serde_json::from_value(args)?;
                let res = engine.phrase_index_search_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Seed-pick tool
        ToolDef {
            spec: ToolSpec {
                name: "seed-pick",
                audience: ToolAudience::Specialist,
                description: "Pick unworked seed passages for research, filtered by tradition and period.",
                input_schema: schema_for::<SeedPickRequest>(),
                output_schema: schema_for::<SeedPickResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Pick seed passages",
                        args: serde_json::json!({
                            "tradition": ["canon"],
                            "period": ["Tang"],
                            "limit": 50
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: SeedPickRequest = serde_json::from_value(args)?;
                let res = engine.seed_pick_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Expand-context-adaptive tool
        ToolDef {
            spec: ToolSpec {
                name: "expand-context-adaptive",
                audience: ToolAudience::Specialist,
                description: "Expand context around a passage by climbing the catalog tree to fit a character budget. Specialist: source-read is the preferred way to read around a passage — use this when you specifically need catalog-tree context expansion to a char budget.",
                input_schema: schema_for::<ExpandContextAdaptiveRequest>(),
                output_schema: schema_for::<ExpandContextAdaptiveResponse>(),
                requires: vec!["passages.parquet", "catalog.index", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Expand context",
                        args: serde_json::json!({
                            "passage_id": "B/B13/B13n0079.xml#pB13p0047a0417",
                            "max_chars": 5000
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: ExpandContextAdaptiveRequest = serde_json::from_value(args)?;
                let res = engine.expand_context_adaptive_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Trace-term-usage tool
        ToolDef {
            spec: ToolSpec {
                name: "trace-term-usage",
                audience: ToolAudience::Specialist,
                description: "Trace term usage across periods, canons, authors, or works with hit counts and representative passages. Specialist: prefer evidence-search or scope-profile for a full workup — call directly when you only need the usage breakdown.",
                input_schema: schema_for::<TraceTermUsageRequest>(),
                output_schema: schema_for::<TraceTermUsageResponse>(),
                requires: vec!["passages.parquet", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Trace term usage by period",
                        args: serde_json::json!({
                            "phrase": "一切有為法",
                            "group_by": "period",
                            "limit_total": 200,
                            "limit_per_group": 5
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: TraceTermUsageRequest = serde_json::from_value(args)?;
                let res = engine.trace_term_usage_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Query-expand-terms tool
        ToolDef {
            spec: ToolSpec {
                name: "query-expand-terms",
                audience: ToolAudience::Specialist,
                description: "Produce variants/orthographic flips/aliases for a seed phrase using bundled lookup tables. Specialist: evidence-search runs this internally — call directly only to inspect candidate variants before searching.",
                input_schema: schema_for::<QueryExpandTermsRequest>(),
                output_schema: schema_for::<QueryExpandTermsResponse>(),
                requires: vec![],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Expand query terms",
                        args: serde_json::json!({
                            "phrase": "一切有為法",
                            "mode": "all",
                            "max": 20
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: QueryExpandTermsRequest = serde_json::from_value(args)?;
                let res = engine.query_expand_terms_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Compare-usage tool
        ToolDef {
            spec: ToolSpec {
                name: "compare-usage",
                audience: ToolAudience::Specialist,
                description: "Compare two sub-corpora and return distinctive terms using log-odds ratio scoring. Specialist: scope-profile wraps this — call directly only for a bare two-corpus distinctive-term comparison.",
                input_schema: schema_for::<CompareUsageRequest>(),
                output_schema: schema_for::<CompareUsageResponse>(),
                requires: vec!["passages.parquet", "catalog.index", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Compare usage between canons",
                        args: serde_json::json!({
                            "scope_a_canon": "canon",
                            "scope_b_canon": "zhonghua",
                            "gram_len": 1,
                            "limit_passages": 1000,
                            "limit_terms": 50
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: CompareUsageRequest = serde_json::from_value(args)?;
                let res = engine.compare_usage_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Collocation-search tool
        ToolDef {
            spec: ToolSpec {
                name: "collocation-search",
                audience: ToolAudience::Specialist,
                description: "Find terms that co-occur near a seed phrase more often than expected by chance.",
                input_schema: schema_for::<CollocationSearchRequest>(),
                output_schema: schema_for::<CollocationSearchResponse>(),
                requires: vec!["passages.parquet", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Search for collocates",
                        args: serde_json::json!({
                            "phrase": "一切有為法",
                            "window_chars": 20,
                            "gram_len": 1,
                            "limit_total": 200,
                            "limit_collocates": 30
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: CollocationSearchRequest = serde_json::from_value(args)?;
                let res = engine.collocation_search_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Pair-appearance tool
        ToolDef {
            spec: ToolSpec {
                name: "pair-appearance",
                audience: ToolAudience::DefaultAgent,
                description: "Find passages where two specified terms both appear, optionally constrained to a character window or sentence.",
                input_schema: schema_for::<PairAppearanceRequest>(),
                output_schema: schema_for::<PairAppearanceResponse>(),
                requires: vec!["passages.parquet", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Find co-mentions in the same passage",
                        args: serde_json::json!({
                            "term1": "念佛",
                            "term2": "禪",
                            "unit": "passage",
                            "limit": 10
                        }),
                    },
                    ToolExample {
                        title: "Find local co-occurrence within a window",
                        args: serde_json::json!({
                            "term1": "念佛",
                            "term2": "禪",
                            "unit": "window",
                            "window_chars": 80,
                            "limit": 10
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: PairAppearanceRequest = serde_json::from_value(args)?;
                let res = engine.pair_appearance_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Outline-search tool
        ToolDef {
            spec: ToolSpec {
                name: "outline-search",
                audience: ToolAudience::Specialist,
                description: "Search for a phrase within a catalog outline node and return hits grouped by child outline nodes.",
                input_schema: schema_for::<OutlineSearchRequest>(),
                output_schema: schema_for::<OutlineSearchResponse>(),
                requires: vec!["passages.parquet", "catalog.index", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Search within a work",
                        args: serde_json::json!({
                            "phrase": "一切有為法",
                            "work_id": "B/B13/B13n0079.xml",
                            "group_by": "division",
                            "limit_total": 200,
                            "limit_per_group": 20
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: OutlineSearchRequest = serde_json::from_value(args)?;
                let res = engine.outline_search_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Cluster-hits tool
        ToolDef {
            spec: ToolSpec {
                name: "cluster-hits",
                audience: ToolAudience::Specialist,
                description: "Cluster phrase search hits by catalog outline (work/division), returning hit counts per cluster with representative passages. Specialist: evidence-search wraps this as `include_clusters` — call directly only to cluster hits without the rest of an evidence workup.",
                input_schema: schema_for::<ClusterHitsRequest>(),
                output_schema: schema_for::<ClusterHitsResponse>(),
                requires: vec!["passages.parquet", "catalog.index", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Cluster hits by work",
                        args: serde_json::json!({
                            "phrase": "一切有為法",
                            "cluster_by": "work",
                            "limit_total": 200,
                            "limit_per_cluster": 20
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: ClusterHitsRequest = serde_json::from_value(args)?;
                let res = engine.cluster_hits_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Absence-check tool
        ToolDef {
            spec: ToolSpec {
                name: "absence-check",
                audience: ToolAudience::Specialist,
                description: "Check whether a phrase is absent from a specific catalog scope (work, canon, period). Specialist: evidence-search wraps this as `include_absence_check` — call directly only for a standalone absence test in one scope.",
                input_schema: schema_for::<AbsenceCheckRequest>(),
                output_schema: schema_for::<AbsenceCheckResponse>(),
                requires: vec!["passages.parquet", "catalog.index", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Check absence in a work",
                        args: serde_json::json!({
                            "phrase": "一切有為法",
                            "scope_work_id": "B/B13/B13n0079.xml",
                            "limit": 100
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: AbsenceCheckRequest = serde_json::from_value(args)?;
                let res = engine.absence_check_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Evidence-search wrapper
        ToolDef {
            spec: ToolSpec {
                name: "evidence-search",
                audience: ToolAudience::DefaultAgent,
                description: "Run exact phrase evidence search plus optional attestation/history/usage/cluster summaries.",
                input_schema: schema_for::<EvidenceSearchRequest>(),
                output_schema: schema_for::<EvidenceSearchResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Search phrase evidence",
                        args: serde_json::json!({
                            "phrase": "一切有為法",
                            "limit": 25,
                            "include_attestation": true,
                            "include_history": true
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: EvidenceSearchRequest = serde_json::from_value(args)?;
                let res = engine.evidence_search_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Hybrid-discover wrapper
        ToolDef {
            spec: ToolSpec {
                name: "hybrid-discover",
                audience: ToolAudience::DefaultAgent,
                description: "Compactly merge vector and TF-IDF discovery candidates from a seed passage. Prefer frontier for ordinary seed expansion; use hybrid-discover when semantic-vector candidates are explicitly useful. Context and raw sub-results are debug/full-output features.",
                input_schema: schema_for::<HybridDiscoverRequest>(),
                output_schema: schema_for::<HybridDiscoverResponse>(),
                requires: vec!["passages.parquet", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Hybrid discovery from a seed passage",
                        args: serde_json::json!({
                            "seed_passage_id": "B/B13/B13n0079.xml#pB13p0047a0417",
                            "limit": 10,
                            "verbosity": "summary"
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: HybridDiscoverRequest = serde_json::from_value(args)?;
                let res = engine.hybrid_discover_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Source-investigate wrapper
        ToolDef {
            spec: ToolSpec {
                name: "source-investigate",
                audience: ToolAudience::DefaultAgent,
                description: "Investigate one seed passage with context and a lexical discovery frontier. The fast default avoids duplicate TF-IDF output and slow semantic search; set include_similar=true for a separate reused similarity block, include_vector=true for opt-in vector neighbors, and pass phrases only when historical distributions are needed.",
                input_schema: schema_for::<SourceInvestigateRequest>(),
                output_schema: schema_for::<SourceInvestigateResponse>(),
                requires: vec!["passages.parquet", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Investigate a seed passage",
                        args: serde_json::json!({
                            "seed_passage_id": "B/B13/B13n0079.xml#pB13p0047a0417",
                            "phrases": ["一切有為法"],
                            "limit": 10
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: SourceInvestigateRequest = serde_json::from_value(args)?;
                let res = engine.source_investigate_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Scope-profile wrapper
        ToolDef {
            spec: ToolSpec {
                name: "scope-profile",
                audience: ToolAudience::DefaultAgent,
                description: "Compare two corpus scopes and optionally trace a phrase inside the same profile.",
                input_schema: schema_for::<ScopeProfileRequest>(),
                output_schema: schema_for::<ScopeProfileResponse>(),
                requires: vec!["passages.parquet", "catalog.index", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Compare usage between two canons",
                        args: serde_json::json!({
                            "scope_a_canon": "T",
                            "scope_b_canon": "X",
                            "gram_len": 1,
                            "limit_passages": 1000
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: ScopeProfileRequest = serde_json::from_value(args)?;
                let res = engine.scope_profile_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Batch-evidence-search wrapper
        ToolDef {
            spec: ToolSpec {
                name: "batch-evidence-search",
                audience: ToolAudience::Specialist,
                description: "Search independent phrases concurrently and return compact per-phrase hit counts plus sample passage IDs. Prefer this over many separate search calls when no result depends on another.",
                input_schema: schema_for::<BatchEvidenceSearchRequest>(),
                output_schema: schema_for::<BatchEvidenceSearchResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Search multiple phrases",
                        args: serde_json::json!({
                            "phrases": ["金剛經", "般若波羅蜜多"],
                            "limit": 10,
                            "concurrency": 4
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: BatchEvidenceSearchRequest = serde_json::from_value(args)?;
                let res = engine.batch_evidence_search_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // pair-profile tool
        ToolDef {
            spec: ToolSpec {
                name: "pair-profile",
                audience: ToolAudience::DefaultAgent,
                description: "Summarise how often two terms appear together versus separately, grouped by period, canon, work, or author. Supports passage/window/sentence/section/work co-occurrence units. Use for analytical questions like 'does term A appear with term B more in Song than Tang sources?'",
                input_schema: schema_for::<PairProfileRequest>(),
                output_schema: schema_for::<PairProfileResponse>(),
                requires: vec!["passages.parquet", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Compare 念佛+禪 co-occurrence rates by period",
                        args: serde_json::json!({
                            "term1": "念佛",
                            "term2": "禪",
                            "group_by": "period",
                            "unit": "passage",
                            "limit_groups": 10
                        }),
                    },
                    ToolExample {
                        title: "Compare by canon, scoped to Taishō",
                        args: serde_json::json!({
                            "term1": "如來",
                            "term2": "法身",
                            "group_by": "work",
                            "unit": "passage",
                            "scope_canon": "T",
                            "limit_groups": 15,
                            "sample_hits_per_group": 2
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: PairProfileRequest = serde_json::from_value(args)?;
                let res = engine.pair_profile_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // person-resolve tool
        ToolDef {
            spec: ToolSpec {
                name: "person-resolve",
                audience: ToolAudience::DefaultAgent,
                description: "Resolve a person's name to candidate forms and show corpus presence. Use before person-history to confirm spelling and aliases are present in the corpus.",
                input_schema: schema_for::<PersonResolveRequest>(),
                output_schema: schema_for::<PersonResolveResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Resolve with primary name and alias",
                        args: serde_json::json!({
                            "name": "雪峰義存",
                            "aliases": ["雪峰", "義存"]
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: PersonResolveRequest = serde_json::from_value(args)?;
                let res = engine.person_resolve_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // place-resolve tool
        ToolDef {
            spec: ToolSpec {
                name: "place-resolve",
                audience: ToolAudience::DefaultAgent,
                description: "Resolve a place name to its DDBC authority record (coordinates, category, alternate names) and show corpus presence. Use to disambiguate historical place names and find geographic context.",
                input_schema: schema_for::<PlaceResolveRequest>(),
                output_schema: schema_for::<PlaceResolveResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Resolve a place name",
                        args: serde_json::json!({
                            "name": "那爛陀",
                            "aliases": ["那爛陀寺"]
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: PlaceResolveRequest = serde_json::from_value(args)?;
                let res = engine.place_resolve_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // person-history tool
        ToolDef {
            spec: ToolSpec {
                name: "person-history",
                audience: ToolAudience::DefaultAgent,
                description: "Retrieve passages mentioning a person, ordered by period. Returns mention-class labels (lineage_relation, attributed_saying, case_appearance, commentarial_reference, name_mention). Run person-resolve first to confirm name forms.",
                input_schema: schema_for::<PersonHistoryRequest>(),
                output_schema: schema_for::<PersonHistoryResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Person history for 雪峰 with aliases",
                        args: serde_json::json!({
                            "name": "雪峰義存",
                            "aliases": ["雪峰", "義存"],
                            "limit": 200
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: PersonHistoryRequest = serde_json::from_value(args)?;
                let res = engine.person_history_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // person-profile tool
        ToolDef {
            spec: ToolSpec {
                name: "person-profile",
                audience: ToolAudience::DefaultAgent,
                description: "Return a structured biographical profile for a person: DDBC authority data (birth/death, dynasty, teachers, students, concise bio) plus a compact corpus mention summary. Use instead of person-history when you want a synthesized overview rather than raw passage hits. Returns false_positive_risk assessment for short names.",
                input_schema: schema_for::<PersonProfileRequest>(),
                output_schema: schema_for::<PersonProfileResponse>(),
                requires: vec!["passages.parquet"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Profile a well-attested figure",
                        args: serde_json::json!({
                            "name": "臨濟義玄",
                            "aliases": []
                        }),
                    },
                    ToolExample {
                        title: "Profile a short-name figure (high false-positive risk)",
                        args: serde_json::json!({
                            "name": "慧能",
                            "aliases": ["惠能"]
                        }),
                    },
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: PersonProfileRequest = serde_json::from_value(args)?;
                let res = engine.person_profile_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // citation-verify tool
        ToolDef {
            spec: ToolSpec {
                name: "citation-verify",
                audience: ToolAudience::DefaultAgent,
                description: "Verify whether a claimed quotation appears in the corpus, optionally scoped to a specific work, canon, or node. Returns exact hits and near-matches when the exact quote is not found. Use for provenance questions like 'is this saying really from the Diamond Sutra?'",
                input_schema: schema_for::<CitationVerifyRequest>(),
                output_schema: schema_for::<CitationVerifyResponse>(),
                requires: vec!["passages.parquet", "doc_table.bin"],
                safety: ToolSafety::ReadOnly,
                examples: vec![
                    ToolExample {
                        title: "Verify a quote in the Taishō canon",
                        args: serde_json::json!({
                            "quote": "一切有為法如夢幻泡影",
                            "scope_canon": "T",
                            "claimed_attribution": "金剛般若波羅蜜經",
                            "limit": 5,
                            "include_near_matches": true
                        }),
                    },
                    ToolExample {
                        title: "Verify scoped to a single work",
                        args: serde_json::json!({
                            "quote": "直指人心見性成佛",
                            "scope_source_work_id": "B/B13/B13n0079.xml",
                            "include_near_matches": true,
                            "near_match_limit": 5
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: CitationVerifyRequest = serde_json::from_value(args)?;
                let res = engine.citation_verify_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // run-batch tool
        ToolDef {
            spec: ToolSpec {
                name: "run-batch",
                audience: ToolAudience::DefaultAgent,
                description: "Execute a batch of tool calls from an inline job list or a JSONL \
                    file, writing all results as JSONL to an output file. Supports DAG-style \
                    `depends_on` ordering between jobs and per-job `timeout_ms`. Returns a \
                    summary (jobs_total / ok / failed / elapsed_ms) and the output file path. \
                    Use this to parallelise multi-step research workflows, persist results for \
                    later analysis, or hand off a scripted plan to the corpus engine in one call.",
                input_schema: schema_for::<RunBatchRequest>(),
                output_schema: schema_for::<RunBatchResponse>(),
                requires: vec![],
                safety: ToolSafety::WritesOutput,
                examples: vec![
                    ToolExample {
                        title: "Search two phrases in parallel, cluster results",
                        args: serde_json::json!({
                            "jobs": [
                                {"id": "s1", "tool": "search", "args": {"phrase": "金剛經", "limit": 20}},
                                {"id": "s2", "tool": "search", "args": {"phrase": "般若波羅蜜", "limit": 20}},
                                {"id": "cluster", "tool": "cluster-hits",
                                 "args": {"passage_ids": []},
                                 "depends_on": ["s1", "s2"]}
                            ],
                            "out": "runs/my-research/batch-results.jsonl",
                            "concurrency": 2
                        }),
                    },
                    ToolExample {
                        title: "Run from a pre-built JSONL plan file",
                        args: serde_json::json!({
                            "input_file": "runs/my-research/plan.jsonl",
                            "out": "runs/my-research/results.jsonl"
                        }),
                    },
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: RunBatchRequest = serde_json::from_value(args)?;
                let res = engine.run_batch_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },

        // Report-from-evidence wrapper
        ToolDef {
            spec: ToolSpec {
                name: "report-from-evidence",
                audience: ToolAudience::DefaultAgent,
                description: "One-shot pipeline that validates a finished adjudication, builds the evidence graph, and renders the markdown report. Run this after your investigation is done and you've written up an adjudication — for writing reports straight from prose without one, use pdf-build's input_markdown instead.",
                input_schema: schema_for::<ReportFromEvidenceRequest>(),
                output_schema: schema_for::<ReportFromEvidenceResponse>(),
                requires: vec![],
                safety: ToolSafety::WritesOutput,
                examples: vec![
                    ToolExample {
                        title: "Build graph and report",
                        args: serde_json::json!({
                            "adjudication": "GraphDiscovery/Runs/text-reuse-discovery/adjudications/test3.json",
                            "graph_out": "GraphDiscovery/Runs/text-reuse-discovery/drafts/test3.graph-draft.json",
                            "report_out": "GraphDiscovery/Runs/text-reuse-discovery/dossiers/test3.report.md",
                            "title": "Canonical Dependence"
                        }),
                    }
                ],
            },
            call: |engine, args| Box::pin(async move {
                let req: ReportFromEvidenceRequest = serde_json::from_value(args)?;
                let res = engine.report_from_evidence_impl(req).await?;
                Ok(serde_json::to_value(res)?)
            }),
        },
    ];

    // annotate_response injects _term_context and _entity_context into every tool
    // response. Explicitly allow additional properties in all output schemas so
    // strict JSON-Schema validators do not reject annotated responses.
    for def in &mut defs {
        if let Some(obj) = def.spec.output_schema.as_object_mut() {
            obj.insert(
                "additionalProperties".to_string(),
                serde_json::Value::Bool(true),
            );
        }
    }

    defs
}

// Implementations will be added to engine.rs

#[cfg(test)]
mod tests {
    use super::*;

    /// Guard the public surface: the default manifest should stay compact and
    /// debug-only tools should remain hidden unless explicitly requested.
    #[test]
    fn audience_labels_keep_manifest_shape() {
        let defs = tool_defs();
        let default_count = defs
            .iter()
            .filter(|def| def.spec.audience == ToolAudience::DefaultAgent)
            .count();
        let internal_count = defs
            .iter()
            .filter(|def| def.spec.audience == ToolAudience::InternalDebug)
            .count();
        assert!(
            default_count <= 18,
            "default agent manifest grew to {default_count} tools; keep it compact"
        );
        assert!(
            internal_count >= 3,
            "debug tools lost their internal labels"
        );
        assert_eq!(audience_for_tool("missing-tool"), ToolAudience::Specialist);
    }
}
