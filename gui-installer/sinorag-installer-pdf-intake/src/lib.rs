//! SinoRAG document intake.
//!
//! Turns local PDF, HTML and plain-text files into a SinoRAG corpus that the
//! existing research tools (JSON CLI and `sinorag mcp`) can search:
//!
//! 1. [`extract`] reads each file to text. PDFs go through the Lege PDF
//!    pipeline: the native text layer where it passes Lege's trust checks,
//!    OCR (Windows.Media.Ocr on Windows) for scanned or garbled pages.
//! 2. [`corpus`] chunks the text into passages and keeps one directory per
//!    document under `<install>/intake/docs/`, plus a manifest keyed by the
//!    file's SHA-256 so re-running intake never adds the same file twice.
//! 3. [`pipeline`] assembles a GD-CEF corpus from every stored document,
//!    re-ingests it into `data/passages.parquet/source_corpus=documents`
//!    and rebuilds doc_table, catalog, phrase and TF-IDF indexes.
//!
//! The Windows GUI in `main.rs` wraps this with the installer steps from the
//! sibling `sinorag-installer-winsafe` (payload, shortcuts, uninstall entry).

pub mod corpus;
pub mod extract;
pub mod mcp_clients;
pub mod pipeline;

#[cfg(target_os = "windows")]
pub mod win_utils;
