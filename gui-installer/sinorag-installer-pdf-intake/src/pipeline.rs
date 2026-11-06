//! Intake run: extract new files into the store, then rebuild the SinoRAG
//! corpus partition and its indexes from everything stored.

use crate::corpus::{collect_files, sha256_file, CorpusStore, CORPUS_ID};
use crate::extract::{extract, DocKind, ExtractOptions};
use anyhow::{anyhow, Context, Result};
use sinorag::commands::init::{self, InitProgressEvent};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub enum IntakeEvent {
    /// Status line plus overall progress in 0..=1.
    Step(String, f32),
    Log(String),
    /// Per-file status for the GUI list ("indexed", "skipped: ...", "failed: ...").
    File { path: PathBuf, status: String },
}

#[derive(Debug, Clone)]
pub struct IntakeOptions {
    /// `<install>` — holds `data/` and `intake/`.
    pub root: PathBuf,
    /// Files and folders to add. Folders are scanned recursively.
    pub inputs: Vec<PathBuf>,
    pub extract: ExtractOptions,
    /// Build phrase + TF-IDF (the slow part). doc_table and catalog are always built.
    pub build_lexical: bool,
    /// Rebuild even when no new document was added (repairs a failed run).
    pub force_rebuild: bool,
    /// Overall progress band this run reports into, so the installer can
    /// put its own steps around it.
    pub progress_band: (f32, f32),
}

#[derive(Debug, Clone, Default)]
pub struct IntakeSummary {
    pub added: usize,
    pub skipped: usize,
    pub failed: usize,
    pub total_documents: usize,
    pub total_passages: usize,
    pub rebuilt: bool,
}

pub fn data_root(root: &Path) -> PathBuf {
    root.join("data")
}

pub fn intake_root(root: &Path) -> PathBuf {
    root.join("intake")
}

/// True when the corpus partition and every lexical index exist.
pub fn indexes_present(root: &Path, need_lexical: bool) -> bool {
    let data = data_root(root);
    let derived = data.join("derived");
    let mut required = vec![
        data.join("passages.parquet").join(format!("source_corpus={CORPUS_ID}")),
        derived.join("doc_table.bin"),
        derived.join("catalog.index"),
    ];
    if need_lexical {
        required.push(derived.join("phrase.index"));
        required.push(derived.join("tfidf.index"));
    }
    required.iter().all(|p| p.exists())
}

pub fn run_intake(
    opts: &IntakeOptions,
    emit: &(dyn Fn(IntakeEvent) + Send + Sync),
) -> Result<IntakeSummary> {
    let (band_lo, band_hi) = opts.progress_band;
    let band = |f: f32| band_lo + (band_hi - band_lo) * f.clamp(0.0, 1.0);
    // Within the band: extraction 0–0.55, ingest 0.55–0.62, indexes 0.62–1.0.
    let step = |label: String, f: f32| emit(IntakeEvent::Step(label, band(f)));
    let log = |line: String| emit(IntakeEvent::Log(line));

    let mut store = CorpusStore::open(intake_root(&opts.root))?;
    let files = collect_files(&opts.inputs);
    let mut summary = IntakeSummary::default();

    if files.is_empty() && !opts.inputs.is_empty() {
        log("No PDF, HTML or text files were found in the selected items.".to_string());
    }

    // ── Extraction ───────────────────────────────────────────────────────────
    let total = files.len().max(1) as f32;
    for (index, (path, kind)) in files.iter().enumerate() {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let base = index as f32 / total;
        step(format!("Reading {name} ({}/{})", index + 1, files.len()), 0.55 * base);

        let sha = match sha256_file(path) {
            Ok(sha) => sha,
            Err(e) => {
                summary.failed += 1;
                emit(IntakeEvent::File { path: path.clone(), status: format!("failed: {e:#}") });
                log(format!("{name}: {e:#}"));
                continue;
            }
        };
        if let Some(existing) = store.find_by_hash(&sha) {
            summary.skipped += 1;
            let status = format!("already indexed ({})", existing.work_id);
            log(format!("{name}: {status}"));
            emit(IntakeEvent::File { path: path.clone(), status });
            continue;
        }

        let mut on_pages = |done: u32, of: u32| {
            if of > 0 {
                let f = base + (done as f32 / of as f32) / total;
                emit(IntakeEvent::Step(
                    format!("Reading {name} ({}/{}): page {done} of {of}", index + 1, files.len()),
                    band(0.55 * f),
                ));
            }
        };
        let extracted = match extract(path, *kind, &opts.extract, &mut on_pages) {
            Ok(doc) => doc,
            Err(e) => {
                summary.failed += 1;
                emit(IntakeEvent::File { path: path.clone(), status: format!("failed: {e:#}") });
                log(format!("{name}: failed: {e:#}"));
                continue;
            }
        };
        for note in extracted.notes.iter().take(20) {
            log(format!("{name}: {note}"));
        }
        if extracted.notes.len() > 20 {
            log(format!("{name}: … {} more page notes", extracted.notes.len() - 20));
        }
        let stored = if extracted.sections.is_empty() {
            None
        } else {
            store
                .add_document(path, &sha, &extracted)
                .with_context(|| format!("storing {name}"))?
        };
        let Some(entry) = stored else {
            summary.failed += 1;
            let status = if *kind == DocKind::Pdf {
                "failed: no text found (scanned PDF without OCR text?)".to_string()
            } else {
                "failed: no text found".to_string()
            };
            log(format!("{name}: {status}"));
            emit(IntakeEvent::File { path: path.clone(), status });
            continue;
        };
        summary.added += 1;
        let mut status = format!("indexed: {} passages", entry.passages);
        if let Some(pages) = entry.pages {
            status.push_str(&format!(", {pages} pages"));
        }
        if entry.ocr_pages > 0 {
            status.push_str(&format!(", {} by OCR", entry.ocr_pages));
        }
        log(format!("{name}: {status} → {}", entry.work_id));
        emit(IntakeEvent::File { path: path.clone(), status });
    }

    summary.total_documents = store.documents().len();
    if summary.total_documents == 0 {
        return Err(if summary.failed > 0 {
            anyhow!(
                "None of the {} file(s) could be read, so there is nothing to index yet. \
                 The log says why for each file.",
                summary.failed
            )
        } else {
            anyhow!("No documents have been indexed yet. Add PDF, HTML or text files first.")
        });
    }

    let profile_changed = store.profile_changed();
    if profile_changed {
        log("Search normalization was upgraded; rebuilding the indexes for all documents.".to_string());
    }
    // Documents stored by a run that was closed or failed before its rebuild
    // finished are in the manifest but not in the indexes.
    let stale = !store.indexes_cover_manifest();
    if stale && summary.added == 0 && indexes_present(&opts.root, opts.build_lexical) {
        log("The indexes are not known to cover every stored document; rebuilding them.".to_string());
    }
    let need_rebuild = summary.added > 0
        || opts.force_rebuild
        || profile_changed
        || stale
        || !indexes_present(&opts.root, opts.build_lexical);
    if !need_rebuild {
        log("No new documents; the existing indexes are current.".to_string());
        summary.total_passages = store.documents().iter().map(|d| d.passages).sum();
        step("Indexes are current".to_string(), 1.0);
        return Ok(summary);
    }

    // ── Corpus rebuild ───────────────────────────────────────────────────────
    step("Assembling corpus".to_string(), 0.56);
    let (cef_dir, passage_count) = store.assemble_cef()?;
    summary.total_passages = passage_count;
    log(format!(
        "Corpus: {} documents, {passage_count} passages",
        summary.total_documents
    ));

    let data = data_root(&opts.root);
    let parquet = data.join("passages.parquet");
    let derived = data.join("derived");
    fs::create_dir_all(&derived)?;

    // Every index is keyed by doc_table ids, which shift when the corpus is
    // re-ingested, so all of them are rebuilt from scratch.
    store.clear_index_state()?;
    // The memory-mapped indexes go first: a running `sinorag mcp` or
    // `sinorag agent` holds them open, and it is better to stop here, with
    // the old corpus still whole, than halfway through replacing it.
    for name in ["phrase.index", "tfidf.index", "vector.index", "doc_table.bin", "catalog.index"] {
        let path = derived.join(name);
        match fs::remove_file(&path) {
            Ok(()) if name == "vector.index" => log(
                "Removed the semantic vector index; it no longer matches the corpus. \
                 Rebuild it with `sinorag indexes semantic` if you use it."
                    .to_string(),
            ),
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(anyhow!(
                    "cannot replace {} ({e}). Close anything that is using SinoRAG from this \
                     folder (the agent window, an MCP client) and run this again.",
                    path.display()
                ))
            }
        }
    }
    let partition = parquet.join(format!("source_corpus={CORPUS_ID}"));
    if partition.exists() {
        fs::remove_dir_all(&partition)
            .with_context(|| format!("removing old partition {}", partition.display()))?;
    }

    step("Writing passages".to_string(), 0.58);
    let written = sinorag::commands::cef::ingest_blocking(cef_dir.clone(), parquet.clone())
        .context("ingesting the assembled corpus")?;
    log(format!("Wrote {written} passages to {}", partition.display()));

    // build_local_indexes reports 0.48 → 1.0; map that onto 0.62 → 1.0.
    let index_progress = |event: InitProgressEvent| match event {
        InitProgressEvent::Step { label, progress } => {
            let p = progress.unwrap_or(0.48);
            let f = 0.62 + 0.38 * ((p - 0.48) / 0.52).clamp(0.0, 1.0);
            emit(IntakeEvent::Step(label, band(f)));
        }
        InitProgressEvent::Log(line) => emit(IntakeEvent::Log(line)),
        InitProgressEvent::Download { .. } | InitProgressEvent::Done(_) => {}
    };
    init::build_local_indexes_with(
        &data,
        &parquet,
        Some(&index_progress),
        opts.build_lexical,
        init::LexicalSettings::SMALL_COLLECTION,
    )
    .context("building indexes")?;
    store.record_index_state()?;
    summary.rebuilt = true;
    step("Documents indexed".to_string(), 1.0);
    Ok(summary)
}
