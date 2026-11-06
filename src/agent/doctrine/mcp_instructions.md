# SinoRAG MCP Server

This server exposes the SinoRAG corpus-research toolset over MCP. Each tool
returns a JSON envelope `{ ok, tool, result | error, meta }`. Read `error.code`
when `ok` is false — common codes include `unknown_tool`, `readonly_violation`,
`admin_tool_disabled`, `invalid_args`, `passage_not_found`, and the
`missing_*` codes (corpus or index not built). `error.suggested_command`
says what to do instead.

All corpus access goes through this server's tools — do not shell out to
`sinorag` directly when an MCP tool exists for the same operation.

Use `tool-docs` (or the MCP tool list) to discover available tools; their
input/output schemas are authoritative. The rest of this file is doctrine,
not a tool catalog.

---

## Tool-selection heuristics

- **Exact phrase evidence**: `search`, `evidence-search`, `phrase-index-search`,
  `canonical-source`, `first-attestation`, `phrase-history`.
- **Discovery from a seed passage (candidates, not evidence)**: prefer
  `frontier` after one or two good exact hits. It does not use the vector index;
  it combines TF-IDF similar passages with distinctive phrase frontiers and is
  usually the best bridge from known wording to unknown leads.
- **Semantic/vector discovery**: `vector-neighbors`, `hybrid-discover`. Use
  only when conceptual drift is worth the extra latency and the seed passage is
  representative. Vector hits are leads, not evidence; verify them with exact
  tools before making claims.
- **Fast lexical parallels**: `similar` for TF-IDF neighbors from a known
  passage, especially reuse/retelling candidates.
- **Scoped corpus reads**: `passage`, `source-read`, `expand-context-adaptive`,
  `heading-search`, `outline-search`.
- **Known work IDs**: `work-metadata` fetches one or many exact IDs in a single
  call. Batch IDs together instead of making parallel `works` calls; use
  `works` for catalog listing or title/author filtering.
- **Distinctive vocabulary / comparison**: `compare-usage`, `scope-profile`,
  `collocation-search`, `trace-term-usage`.
- **Absence / clustering**: `absence-check`, `cluster-hits`.
- **Variant expansion**: `query-expand-terms` (no corpus deps).
- **Several independent phrases**: `batch-evidence-search` returns compact
  per-phrase counts and sample passage IDs concurrently. Prefer it to a run of
  separate `search` calls when later inputs do not depend on earlier results.
- **Person research**: `person-resolve` then `person-history`.
- **Term-pair co-occurrence**: `pair-appearance` (individual passage evidence);
  `pair-profile` (aggregate rates by period, canon, or work).
- **Citation verification**: `citation-verify` to check whether a claimed quote
  appears in the corpus.
- **Batch execution** (gated by `--writable`): `run-batch` — submit an inline
  job array or a JSONL plan file; all results are written to a JSONL output
  file; use `depends_on` for DAG ordering and `concurrency` for parallel steps.
- **Output / write** (gated; only available when MCP is launched with
  `--writable`): `graph-build`, `report-build`, `report-from-evidence`,
  `pdf-build`, `validate-adjudication`. `pdf-build` accepts prose directly in
  `markdown`; `input_markdown` is a file path.

**Principle**: exact evidence before discovery. Start with `search`,
`evidence-search`, `works`, or `heading-search` when the user gives a phrase,
person, title, work, or doctrine. If exact work IDs are already known, use one
`work-metadata` call. Read the best hit with `passage` or
`source-read`. Then, unless the user asked only for direct lookup, run
`frontier` on the best seed passage to discover phrases and nearby source
candidates the user did not know to search for. Use `cluster-hits` or
`trace-term-usage` once you have many hits and need distribution rather than
more snippets.

Treat every discovery result (`frontier`, `similar`, `vector-neighbors`,
`hybrid-discover`, `source-investigate`) as a candidate lead. Convert it into
evidence only after exact verification and close reading.

For `source-read`, send one compact anchor per call: `source_work_id` starts at
the work's beginning, `passage_id` reads around that hit, and `cursor` continues
from a prior response. Do not repeat `source_work_id` alongside `passage_id`;
the complete passage ID already identifies its work. Its `#anchor` suffix is a
required part of the string and is valid JSON. If the MCP client itself reports
JSON parsing before execution, the call never reached SinoRAG; retry one compact
call without issuing another tool call concurrently.

---

## Output discipline

- Every accepted claim must cite a `passage_id` returned by a tool. No
  invented citations.
- A work ID such as `T48n2002B` is metadata, not a citation. Put the complete
  passage ID (including its source path and anchor) beside every quotation and
  every factual table row or bullet that depends on corpus evidence.
- Before finalizing a report, audit every claim: distinguish direct text from
  inference, describe zero hits as absence in the loaded corpus rather than
  proof of nonexistence, and do not turn the earliest loaded attestation into
  an unsupported claim about historical origin.
- Tool errors whose `code` starts with `missing_` mean the corpus or one of
  its indexes has not been built; tell the user what `suggested_command`
  says rather than guessing or retrying.
- Prefer JSON-ready output when the next consumer is another tool; prefer
  prose when the next consumer is the user.
