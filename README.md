# SinoRAG

**SinoRAG** is a local-first retrieval backend that lets an LLM agent search texts with tools instead of guessing from memory.

It turns a collection of texts into a searchable passage database, then builds compact indexes for exact phrase search, TF-IDF similarity, optional vector discovery and catalog browsing. Agents use them over MCP or a JSON CLI. Everything runs on your machine, and nothing is uploaded.

There are two ways to use it:

- **Your own documents.** Point it at any number of PDFs, web pages or text files, and ask Claude, Codex, Cursor or another AI app questions about them, with answers cited by document and page. Scanned PDFs are OCR'd. On Windows the Document Intake installer does all of this without a command line; see [Your own documents](#your-own-documents-windows).
- **Chinese Buddhist and classical Chinese research.** The original use case. It covers the full CBETA canon (TEI/XML), with term glosses, DDBC person and place authorities, and catalog, lineage and textual-reuse tools. See [Quick start](#quick-start).

> Let an LLM agent research your sources with tools instead of guessing from memory.

---

## Your own documents (Windows)

Download `SinoRAG-Document-Intake-<version>-windows-x64.zip` from [GitHub Releases](https://github.com/LegeApp/SinoRAG/releases), unzip it, and run `sinorag-intake.exe`:

1. Add PDFs, HTML or `.txt` files, or whole folders, by dragging them onto the window or with **Add files…** / **Add folder…**.
2. Choose the OCR language for scanned pages.
3. Leave **Connect AI apps** ticked to add SinoRAG as an MCP server to the AI apps it finds: Claude Desktop, Claude Code, Codex, Cursor, Windsurf, VS Code, Gemini CLI and LM Studio.
4. Press **Install and index**, then restart your AI app and ask it about your documents.

To add more files later, open **Start Menu → SinoRAG → Add documents to SinoRAG**. Files already indexed are skipped. Uninstall SinoRAG from Settings → Apps.

How it works:

- PDFs go through the Lege PDF pipeline, which uses the text layer where it can be trusted and Windows OCR elsewhere.
- Each document is split into passages of about 900 characters, which keep their page numbers.
- English and other alphabetic text is searched by whole words, case-insensitively.
- A search with no exact match still returns related passages, so the agent has somewhere to go next.
- CJK text in the same collection is searched exactly as for CBETA.

See [`gui-installer/sinorag-installer-pdf-intake/README.md`](gui-installer/sinorag-installer-pdf-intake/README.md) for the details, the install-folder layout and how to build it. On other platforms, convert your documents to CEF (see [Custom corpora](#custom-corpora-cef)) and use `sinorag ingest cef`.

---

## Quick start

For the CBETA corpus:

```bash
# 1. Download and initialize the CBETA corpus (one command)
sinorag init

# 2. Check what's built
sinorag status

# 3. Use the research tools
sinorag tools-manifest --include-examples
sinorag tool-call search --json '{"phrase":"金剛經","limit":5}'

# 4. Optional: semantic (vector) search
sinorag indexes semantic --model bge-small-zh-v1.5

# 5. Optional interactive agent session through opencode
sinorag setup opencode
sinorag agent
```

`sinorag init` downloads the pre-built CBETA corpus pack from GitHub Releases, extracts it, and builds all lexical indexes (phrase + TF-IDF) in a single run. After it completes, every tool is ready to use — no separate index build step required. The only optional step is semantic vector search, which requires a separate embedding run.

`sinorag --help` shows the full command reference. SinoRAG exposes the same tool registry through two supported interfaces: JSON CLI (`tools-manifest`, `tool-call`, `run-tools`) for scripts and reproducible batches, and MCP (`sinorag mcp`) for interactive agents. The recommended MCP path for opencode is `sinorag agent`.

---

## What `sinorag init` does

1. Downloads `cbeta-pack.7z` from GitHub Releases (~curl, progress bar shown)
2. Extracts it in-process with pure-Rust LZMA2 (no 7z binary required)
3. Produces: `passages.parquet/`, `dict.parquet/`, `persons.parquet/`, `places.parquet/`
4. Builds `doc_table.bin` and `catalog.index` (fast, a few minutes)
5. Builds `phrase.index` and `tfidf.index` (slow, up to several hours on large corpora)

After init, all tools work except `vector-neighbors` and `hybrid-discover`'s semantic path, which require a vector index (step 4 below).

---

## What it does

- Ingests your own PDF, HTML and text files (Windows Document Intake), CBETA TEI/XML (GitHub xml-p5 and ISO xml-iso layouts), CEF JSON-lines, and Terebess HTML corpora
- Searches alphabetic-script corpora by whole words (CEF `text_profile = "words"`), and CJK corpora by normalized characters
- Builds a Parquet passage store partitioned by `source_corpus`
- Annotates tool responses with Buddhist term glosses (`dict.parquet`) and DDBC person/place authority data (`persons.parquet`, `places.parquet`)
- Provides exact phrase search over normalized Chinese text (`phrase.index`)
- Builds a document table for stable `doc_id ↔ passage_id` mapping
- Builds catalog indexes for corpus/work/section navigation
- Builds TF-IDF indexes for similarity and textual reuse discovery
- Builds optional vector indexes from external embedding JSONL or local FastEmbed models for semantic discovery
- Exposes research tools through JSON schemas and MCP for LLM agents
- Produces structured JSON output for reports, graph generation, and downstream apps

---

## Supported corpora

| Source | Command | Input |
|---|---|---|
| Your own documents (Windows) | `sinorag-intake.exe` | PDF (with OCR), HTML, `.txt` files or folders |
| CBETA (automatic, from pack) | `init` | no local files needed |
| CBETA TEI/XML (GitHub) | `init --from-raw <PATH>` | CBETA root (containing `xml-p5/`) or `xml-p5/` directly |
| CBETA ISO distribution | `init --from-raw <PATH>` | CBETA ISO root (containing `xml-iso/`) |
| CEF JSON-lines | `ingest cef <FILE>` | `.jsonl` file in Corpus Exchange Format |
| Terebess HTML | `ingest terebess <DIR>` | Directory of SingleFile-saved HTML pages |

Both CBETA formats write to the same `cbeta` partition. Multiple corpora can be ingested into the same store — each lands in its own `source_corpus=<name>` Parquet partition.

---

## User workflow

### Step 1 — Initialize

```bash
sinorag init
```

Downloads the CBETA corpus pack and builds all lexical indexes. Takes anywhere from 30 minutes to a few hours depending on your machine. Uses `curl` (ships with Windows 10 1803+, macOS, and most Linux distros).

**If you already have the CBETA source files:**

```bash
sinorag init --from-raw /path/to/cbeta/xml-p5   # GitHub layout
sinorag init --from-raw /path/to/cbeta-iso       # ISO layout
```

**If you have a custom pack URL or a local file:**

```bash
sinorag init --url file:///path/to/cbeta-pack.7z
sinorag init --url https://example.com/cbeta-pack.7z
```

Use `--force` to re-initialize if the corpus is already present.

### Step 2 — Check status

```bash
sinorag status
```

Reports what's ingested, which indexes are present, and suggested next steps.

### Step 3 — Use the tools

```bash
# Scriptable / batchable JSON CLI
sinorag tools-manifest --include-examples
sinorag tool-call evidence-search --json '{"phrase":"金剛經","limit":5}'
sinorag tool-call batch-evidence-search --json '{"phrases":["金剛經","般若波羅蜜"],"concurrency":4}'
sinorag tool-call hybrid-discover --json '{"seed_passage_id":"B/B13/B13n0079.xml#pB13p0047a0417","limit":10}'
sinorag run-tools --input jobs.jsonl --output results.jsonl

# Interactive opencode session over MCP
sinorag setup opencode
sinorag agent
```

JSON CLI is the best fit for scripts, tests, repeatable batches, and audit trails. MCP is the supported interactive transport for MCP-capable agents. `sinorag agent` wraps `sinorag mcp` for opencode by regenerating `<workdir>/.opencode/opencode.json` and the sinorag-managed block in `<workdir>/AGENTS.md`, then launching opencode. If you use another MCP client, point it at `sinorag mcp`.

For several independent phrases, prefer `batch-evidence-search` to separate `search` calls; it runs up to four searches concurrently by default and returns compact samples in input order. `source-investigate` defaults to context plus the lexical frontier. Its separate similarity block and semantic vector neighbors are opt-in because frontier already contains TF-IDF parallels and vector reranking is substantially slower.

### Step 4 — Optional: semantic vector search

```bash
sinorag indexes semantic --model bge-small-zh-v1.5
```

Builds a vector index for semantic discovery. This can take many hours on large corpora and requires a binary built with `--features local-embeddings`. For TensorRT acceleration, build with `--features tensorrt`.

An external embedding flow is also available for provider-managed batches:

```bash
sinorag index vector-export --out data/derived/vector_input.jsonl
sinorag index vector-build --embeddings data/derived/embeddings.jsonl --model-id BAAI/bge-m3
```

### Rebuilding individual indexes

Phrase and TF-IDF indexes can be rebuilt separately if needed:

```bash
sinorag indexes lexical       # phrase + TF-IDF together (skips if current)
sinorag index phrase          # phrase only
sinorag index tfidf           # TF-IDF only
```

Use `--temp-dir` pointing to a fast SSD for large builds. Avoid RAM-backed `/tmp`.

---

## Data model

```text
data/
  passages.parquet/
    source_corpus=cbeta/      ← partitioned by corpus
  dict.parquet/               ← Buddhist term glossary (Soothill, Dingfubao, etc.)
  persons.parquet/            ← DDBC person authority (46k+ entries)
  places.parquet/             ← DDBC place authority (38k+ entries)
  derived/
    doc_table.bin             stable doc_id / passage_id mapping
    catalog.index             corpus / work / outline navigation
    phrase.index              exact phrase candidate index
    tfidf.index               similarity index
    vector.index              semantic discovery index      (optional)
    registry.sqlite           mutable research state        (auto-created)
```

---

## Tool dependency map

Which artifacts each tool needs:

| Tool | parquet | doc_table | catalog | phrase_index | tfidf | vector | Notes |
|---|---|---|---|---|---|---|---|
| `passage` | ✓ | — | — | — | — | — | |
| `search` / `evidence-search` | ✓ | optional | optional | optional | — | — | phrase_index speeds exact match |
| `expand-context` | ✓ | — | — | — | — | — | pure parquet window |
| `expand-context-adaptive` | ✓ | ✓ | ✓ | — | — | — | catalog for node climbing |
| `find-first-mention` | ✓ | ✓ | — | optional | — | — | phrase_index greatly speeds it up |
| `trace-term-usage` | ✓ | ✓ | — | optional | — | — | |
| `phrase-history` | ✓ | ✓ | — | optional | — | — | |
| `first-attestation` | ✓ | ✓ | — | optional | — | — | |
| `outline-search` | ✓ | ✓ | ✓ | optional | — | — | |
| `cluster-hits` | ✓ | ✓ | ✓ | optional | — | — | |
| `absence-check` | ✓ | ✓ | ✓ | optional | — | — | |
| `collocation-search` | ✓ | ✓ | — | optional | — | — | no catalog needed |
| `compare-usage` / `scope-profile` | ✓ | ✓ | ✓ | — | — | — | |
| `similar` / `similar-batch` | ✓ | ✓ | — | — | ✓ | — | |
| `vector-info` / `vector-neighbors` | optional | ✓ | — | — | — | ✓ | semantic candidates, not evidence |
| `hybrid-discover` / `source-investigate` | ✓ | ✓ | optional | optional | optional | optional | orchestrates discovery tools |
| `frontier` / `research-packet` | ✓ | ✓ | optional | optional | optional | — | |
| `query-expand-terms` | — | — | — | — | — | — | zero corpus deps |

**After `sinorag init`:** all tools in the table above are ready except those requiring `vector`.

---

## Agent tool workflow

### JSON CLI

```bash
sinorag tools-manifest --include-examples
sinorag tool-call passage --json '{"id":"B/B13/B13n0079.xml#pB13p0047a0417"}'
sinorag run-tools --input jobs.jsonl --output results.jsonl --jobs 4
```

Agent pattern:

```text
agent receives user question
  -> agent reads tool schemas from tools-manifest
  -> agent calls SinoRAG tools with JSON
  -> SinoRAG returns exact passages + metadata
  -> agent writes answer/report with citations
```

Research tools that require indexes not yet built will return a clear error rather than silently failing. The `status` command tells you what's missing before you start.

### MCP / opencode

```bash
sinorag setup opencode   # verifies opencode and prints provider setup steps
sinorag agent            # launches opencode with SinoRAG MCP pre-wired
```

MCP is the first-class interactive-agent interface, while the JSON CLI remains the first-class automation interface. For opencode, prefer `sinorag agent` instead of hand-editing MCP config. Direct `sinorag mcp` remains supported for other MCP clients and for debugging transport-level issues.

---

## Custom corpora (CEF)

```bash
# Create a skeleton CEF directory
sinorag cef-init --out my-corpus
# Edit: corpus.toml, works.jsonl, passages.jsonl

# Validate before ingesting
sinorag cef-validate --input my-corpus

# Ingest
sinorag ingest cef my-corpus/passages.jsonl
```

---

## LLM-assisted research

SinoRAG is not meant to replace a scholar or translator — it makes an agent's work inspectable.

Generated artifacts should preserve:

- exact Chinese source text and passage IDs
- source work metadata (canon, period, author, title)
- search method and confidence
- candidate graph edges and rejected evidence

---

## ReadZen integration

```text
SinoRAG generates evidence.
ReadZen organizes and displays it beside source texts.
```

SinoRAG can export research bundles (`export-readzen`, `graph-build`, `report-build`) consumable by ReadZen to attach generated reports, phrase histories, and lineage diagrams to the CBETA passages they cite.

---

## Performance notes

- Use release builds: `cargo build --release`
- Put `--temp-dir` on a fast SSD; avoid RAM-backed `/tmp` for phrase or TF-IDF builds
- Indexes are mmap-backed — O(1) RAM at query time regardless of index size
- `doc_id` integer references are preferred over repeated passage ID strings in all index structures

---

## Install

For your own documents on Windows, use the Document Intake zip from [GitHub Releases](https://github.com/LegeApp/SinoRAG/releases) (see [above](#your-own-documents-windows)). To build from source:

```bash
git clone https://github.com/LegeApp/SinoRAG
cd SinoRAG
cargo install --path .
```

---

## Status

Experimental but usable. Expect index schemas to change while the project stabilizes. The main user-facing commands (`init`, `status`, `indexes lexical`, `indexes semantic`, `tools-manifest`, `tool-call`, `run-tools`, `setup opencode`, `agent`, and `mcp`) are supported. The Windows Document Intake installer is released as a zip on GitHub Releases. It cannot remove individual documents yet.

SinoRAG supports both JSON CLI and MCP against the same tool registry: use JSON CLI for reproducible command-line workflows, and use MCP for live agent sessions. `sinorag agent` is the maintained opencode wrapper around `sinorag mcp`.

---

## License

AGPL-3.0
