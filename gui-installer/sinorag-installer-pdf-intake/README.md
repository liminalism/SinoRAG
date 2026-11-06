# SinoRAG Document Intake (Windows)

A winsafe GUI that installs SinoRAG and indexes your own **PDF, HTML and plain-text** files, so a model can answer questions from them through `sinorag mcp` (or the `sinorag agent` OpenCode shortcut).

It is a variant of `../sinorag-installer-winsafe`. It uses the same `sinorag.7z` payload and the same look. It installs into its own folder (`%LOCALAPPDATA%\Programs\SinoRAG-Documents`) with its own uninstall entry and shortcuts, so it can sit alongside the CBETA install. It does not download CBETA.

## Using it

1. Run `sinorag-intake.exe`. Add files with **Add files…** / **Add folder…**, by dropping them on the window, or by passing them on the command line.
2. Pick the OCR language for scanned pages. Windows OCR needs the matching language pack installed; the log warns up front when it is missing.
3. Press **Install and index**.
4. Later, open **Start Menu → SinoRAG → Add documents to SinoRAG** to add more. Files already indexed (matched by SHA-256) are skipped. The indexes are rebuilt from the whole collection whenever something new is added, and with nothing queued the button reads **Rebuild indexes** and does that.

Ask questions with the **SinoRAG Documents** shortcut (`sinorag.exe agent`). You can also point any MCP client at `sinorag.exe mcp` in the install folder.

Three things to know:

- **The install folder is SinoRAG's alone.** Every run rebuilds all indexes in it and the uninstaller deletes it whole, so the intake refuses a folder that already holds other files, or a CBETA install. Browsing to such a folder suggests a `SinoRAG-Documents` folder inside it.
- **Close SinoRAG before adding documents or uninstalling.** A running `sinorag.exe` (the agent window, an MCP client's server) keeps the indexes mapped, so they cannot be replaced. The intake and the uninstaller both check and say so before changing anything.
- **A run that is closed or fails part-way loses nothing.** Documents already read are kept, shortcuts and the uninstall entry are in place from the start, and the next run rebuilds the indexes (`intake/index-state.json` records which documents they cover).

## What happens to a file

| Type | Extraction |
|---|---|
| `.pdf` | Lege PDF pipeline (`lege-pdf-agent::text_pages`), plain reading order, `ocr=auto`. The native text layer is used where Lege trusts it, and Windows.Media.Ocr is used elsewhere. If a "trusted" layer still reads as symbol soup, the page is re-read with forced OCR (see `extract::text_quality`). |
| `.html` `.htm` `.xhtml` | Tags, scripts and styles are stripped, block structure is kept, entities are decoded, and `<title>` becomes the title. The charset comes from BOM, then `<meta>`, then sniffing. |
| `.txt` | The charset is sniffed: BOM, UTF-16, UTF-8, then a statistical guess (`chardetng`) among GBK, Big5, Shift_JIS, EUC-JP, EUC-KR and the Windows code pages. Form feeds become pages. |

A file with no searchable text (a scan with no OCR result, an empty page) is reported as failed and not stored.

The text is chunked into passages of about 900 characters (at most 1400), one page at a time. Wrapped lines are re-joined (CJK without spaces) and end-of-line hyphens are undone. Passage ids look like `doc-<sha12>#p0012.02` (page 12, chunk 2) and the locator is `p. 12`. HTML and text files use `#s0001`.

## Install-folder layout

```
sinorag.exe, sinorag-intake.exe, sinorag-intake-uninstaller.exe, SinoRAG.ico
intake/manifest.json            documents already stored (sha256 → work_id)
intake/index-state.json         the documents the indexes were last built from
intake/docs/<work_id>/text.txt  extracted text with "=== page N ===" markers
intake/docs/<work_id>/{work.json,passages.jsonl}   GD-CEF records
intake/cef/                     corpus assembled from all documents each run
data/passages.parquet/source_corpus=documents/
data/derived/{doc_table.bin,catalog.index,phrase.index,tfidf.index,registry.sqlite}
```

Each run rebuilds `intake/cef/` from the manifest. It then replaces the `documents` partition through `sinorag::commands::cef::ingest_blocking` and rebuilds every index through `sinorag::commands::init::build_local_indexes`. Re-ingesting shifts doc ids, so an existing `vector.index` is removed and needs to be rebuilt with `sinorag indexes semantic`.

## Building

You need `D:\projects\Lege-ecosystem` checked out next to this repo; the crate path-depends on `lege-pdf/agent` and `pdf-text`.

```sh
.\make-payload.ps1               # build sinorag.exe from this checkout, pack it as ./sinorag.7z
cargo build --release            # sinorag-intake.exe + sinorag-intake-uninstaller.exe
cargo test --lib                 # extraction / chunking / install-folder unit tests
# Headless end-to-end run, no GUI:
cargo run --example intake_cli -- <root> [--ocr eng] [--no-lexical] <files or folders>...
```

Ship the two exes side by side. The intake copies itself and the uninstaller into the install folder.

### The payload

The installer embeds `sinorag.exe` as `sinorag.7z`: `./sinorag.7z` if it exists, else the one the CBETA installers share (`../sinorag-installer/sinorag.7z`). The intake writes the indexes with the `sinorag` library it links, and the installed `sinorag.exe` reads them, so **the payload must be built from the same source as the intake**. An older `sinorag.exe` does not know the `words` text profile: English phrases find nothing and fragments of words match.

Run `make-payload.ps1` after any change to the SinoRAG sources. `build.rs` unpacks the payload and fails the build when its `sinorag.exe --version` differs from the checkout's, and warns when the payload is older than the newest file under `src/`. `SINORAG_SKIP_PAYLOAD_CHECK=1` skips both.

To query a headless `intake_cli` root, copy a current `sinorag.exe` into it: SinoRAG resolves `data/` next to its own exe.

## Search behaviour

The assembled corpus uses GD-CEF `text_profile = "words"`:

- **English and other alphabetic scripts:** phrase search is case-insensitive and matches whole words ("coolant pump" finds "Coolant-Pump"; "pump" does not find "pumps"). There is no stemming.
- **Similarity:** `similar`, `frontier` and `source-investigate` rank by shared words and word pairs. English stopwords are ignored.
- **Chinese, Japanese and Korean:** normalized exactly as for CBETA, so a mixed collection works.
- **What the model is told:** the MCP server and `sinorag agent` add a short "Searching English" section to their instructions when the data root has a word-layout corpus (`data/text_profiles.json`).
- **Upgrades:** installs from before this change rebuild their indexes once, automatically, on the next run.
- **TF-IDF settings:** min_df 2, max_df 25%, instead of CBETA's 5 / 5%.

## Known limits

- No stemming or synonyms. Search each word form you need (`batch-evidence-search` takes several at once).
- Removing a document is not supported in the GUI yet.
