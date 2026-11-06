//! `sinorag init` — bootstrap the CBETA corpus from a pre-built pack.
//!
//! Default path:
//!   1. curl downloads cbeta-pack.7z from GitHub Releases
//!   2. sevenz-rust decompresses the .7z in-process (pure Rust LZMA2)
//!   3. passages.parquet/, dict.parquet/, persons.parquet/, places.parquet/
//!      are extracted into the data root
//!   4. doc_table, catalog, phrase, and TF-IDF indexes are built locally
//!      (doc_table + catalog: fast; phrase + TF-IDF: up to several hours)
//!
//! After init, all tools are ready to use except semantic vector search,
//! which requires a separate `sinorag indexes semantic` run.
//!
//! Alternative: `sinorag init --from-raw <PATH>` ingests from a local
//! CBETA corpus directory (GitHub xml-p5 or ISO xml-iso layout) and
//! produces an identical result without downloading anything.

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// Canonical pack URL for the current release.
///
/// Set to `None` until the first pack is built and uploaded to GitHub Releases.
/// When ready, replace with the direct download URL to `cbeta-pack.7z`.
///
pub const PACK_URL: Option<&str> =
    Some("https://github.com/LegeApp/SinoRAG/releases/download/corpus-release-1/cbeta-pack.7z");

const PACK_FILENAME: &str = "cbeta-pack.7z";

#[derive(Debug, Clone)]
pub enum InitProgressEvent {
    Step {
        label: String,
        progress: Option<f32>,
    },
    Download {
        received: u64,
        total: Option<u64>,
    },
    Log(String),
    Done(Result<(), String>),
}

pub trait InitProgress: Send + Sync {
    fn event(&self, event: InitProgressEvent);
}

impl<F> InitProgress for F
where
    F: Fn(InitProgressEvent) + Send + Sync,
{
    fn event(&self, event: InitProgressEvent) {
        self(event);
    }
}

fn emit(progress: Option<&dyn InitProgress>, event: InitProgressEvent) {
    if let Some(progress) = progress {
        progress.event(event);
    }
}

pub async fn run(
    url_override: Option<String>,
    force: bool,
    from_raw: Option<PathBuf>,
    out_parquet: PathBuf,
) -> Result<()> {
    let data_root = out_parquet
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("data"));

    let cbeta_partition = out_parquet.join("source_corpus=cbeta");

    if cbeta_partition.exists() && !force {
        // Corpus present. If all fast indexes already exist too, nothing to do.
        let derived = data_root.join("derived");
        let indexes_complete = [
            "doc_table.bin",
            "catalog.index",
            "phrase.index",
            "tfidf.index",
        ]
        .iter()
        .all(|name| derived.join(name).exists());
        if indexes_complete {
            eprintln!(
                "CBETA corpus and indexes are already present at {}.",
                data_root.display()
            );
            eprintln!("Run `sinorag status` to inspect, or use --force to rebuild everything.");
            eprintln!("For semantic search: sinorag index vector-update --model bge-small-zh-v1.5");
            return Ok(());
        }
        // Corpus present but indexes are incomplete — rebuild indexes only (skip download).
        eprintln!(
            "CBETA corpus found at {}. Rebuilding missing indexes...",
            data_root.display()
        );
        return tokio::task::spawn_blocking(move || {
            build_local_indexes(&data_root, &out_parquet, None, true)
        })
        .await
        .context("index rebuild task panicked")?;
    }

    // --from-raw: bypass download and ingest from local corpus files.
    if let Some(raw_path) = from_raw {
        return ingest_from_raw(raw_path, out_parquet, data_root).await;
    }

    // -- Download ----------------------------------------------------------
    let url = url_override.as_deref().or(PACK_URL).ok_or_else(|| {
        anyhow::anyhow!(
            "No official CBETA pack is published yet for this release.\n\
                 \n\
                 To ingest from your own CBETA source files:\n\
                 \n  \
                   sinorag init --from-raw /path/to/cbeta\n\
                 \n\
                 This accepts both the GitHub xml-p5 layout (one file per work)\n\
                 and the ISO xml-iso layout (one file per fascicle).\n\
                 \n\
                 To use a custom pack URL or local file path:\n\
                 \n  \
                   sinorag init --url file:///path/to/cbeta-pack.7z\n\
                   sinorag init --url https://example.com/cbeta-pack.7z"
        )
    })?;

    // download_with_curl, extract_7z, and build_local_indexes are all synchronous
    // and can block for minutes on large packs — run them on a dedicated blocking thread
    // so the Tokio runtime stays responsive.
    let url_owned = url.to_string();
    tokio::task::spawn_blocking(move || {
        run_from_pack_url_blocking(&url_owned, false, data_root, out_parquet, None, true)
    })
    .await
    .context("init blocking task panicked")?
}

pub fn run_from_pack_url_blocking(
    url: &str,
    force: bool,
    data_root: PathBuf,
    out_parquet: PathBuf,
    progress: Option<&dyn InitProgress>,
    build_phrase_tfidf: bool,
) -> Result<()> {
    let cbeta_partition = out_parquet.join("source_corpus=cbeta");
    if cbeta_partition.exists() && !force {
        let message = format!(
            "CBETA corpus is already present at {}.",
            cbeta_partition.display()
        );
        eprintln!("{message}");
        emit(progress, InitProgressEvent::Log(message));
        return Ok(());
    }

    let tmp_dir = data_root.join(".init-download");
    std::fs::create_dir_all(&tmp_dir).context("creating temporary download directory")?;
    let arc_path = tmp_dir.join(PACK_FILENAME);

    let result = (|| {
        let message = format!("Downloading pack from {url}");
        eprintln!("{message}");
        emit(
            progress,
            InitProgressEvent::Step {
                label: message,
                progress: Some(0.10),
            },
        );
        download_pack(url, &arc_path, progress)?;

        eprintln!("\nExtracting pack...");
        emit(
            progress,
            InitProgressEvent::Step {
                label: "Extracting corpus pack".to_string(),
                progress: Some(0.35),
            },
        );
        extract_7z(&arc_path, &data_root, progress)?;

        build_local_indexes(&data_root, &out_parquet, progress, build_phrase_tfidf)
    })();

    let _ = std::fs::remove_file(&arc_path);
    let _ = std::fs::remove_dir(&tmp_dir);
    emit(
        progress,
        InitProgressEvent::Done(match &result {
            Ok(()) => Ok(()),
            Err(error) => Err(error.to_string()),
        }),
    );
    result
}

// ---------------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------------

fn download_with_curl(url: &str, dest: &Path) -> Result<()> {
    let status = Command::new("curl")
        .args([
            "-L",
            "--progress-bar",
            "--fail",
            "-o",
            dest.to_str().context("non-UTF-8 destination path")?,
            url,
        ])
        .status()
        .context(
            "failed to invoke curl — ensure curl is installed and on PATH.\n\
             On Windows it ships with the OS since version 1803.",
        )?;

    if !status.success() {
        bail!("curl exited with {status}; check the URL and your network connection.");
    }
    Ok(())
}

fn download_pack(url: &str, dest: &Path, progress: Option<&dyn InitProgress>) -> Result<()> {
    if progress.is_none() {
        return download_with_curl(url, dest);
    }

    if let Some(path) = url.strip_prefix("file://") {
        let source = PathBuf::from(path);
        let total = std::fs::metadata(&source).ok().map(|m| m.len());
        copy_with_progress(&source, dest, total, progress)
    } else {
        download_with_powershell(url, dest, progress).or_else(|_| download_with_curl(url, dest))
    }
}

fn copy_with_progress(
    source: &Path,
    dest: &Path,
    total: Option<u64>,
    progress: Option<&dyn InitProgress>,
) -> Result<()> {
    let mut input =
        File::open(source).with_context(|| format!("opening pack source {}", source.display()))?;
    let mut output = File::create(dest)
        .with_context(|| format!("creating pack destination {}", dest.display()))?;
    let mut received = 0_u64;
    let mut buf = vec![0_u8; 1024 * 1024];
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        output.write_all(&buf[..n])?;
        received += n as u64;
        emit(progress, InitProgressEvent::Download { received, total });
    }
    Ok(())
}

fn download_with_powershell(
    url: &str,
    dest: &Path,
    progress: Option<&dyn InitProgress>,
) -> Result<()> {
    let script = r#"
param([string]$Url, [string]$Dest)
$ErrorActionPreference = 'Stop'
$request = [System.Net.HttpWebRequest]::Create($Url)
$request.AllowAutoRedirect = $true
$response = $request.GetResponse()
try {
  $total = $response.ContentLength
  $stream = $response.GetResponseStream()
  $file = [System.IO.File]::Create($Dest)
  try {
    $buffer = New-Object byte[] 1048576
    [Int64]$received = 0
    while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
      $file.Write($buffer, 0, $read)
      $received += $read
      Write-Output "SINORAG_DOWNLOAD $received $total"
    }
  } finally {
    $file.Dispose()
    $stream.Dispose()
  }
} finally {
  $response.Dispose()
}
"#;
    let mut cmd = Command::new("powershell");
    cmd.args([
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        script,
        url,
        dest.to_str().context("non-UTF-8 destination path")?,
    ]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let output = cmd
        .output()
        .context("failed to invoke PowerShell for download")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("SINORAG_DOWNLOAD ") {
            let mut parts = rest.split_whitespace();
            let received = parts.next().and_then(|v| v.parse::<u64>().ok());
            let total = parts.next().and_then(|v| v.parse::<i64>().ok());
            if let Some(received) = received {
                emit(
                    progress,
                    InitProgressEvent::Download {
                        received,
                        total: total.and_then(|v| (v > 0).then_some(v as u64)),
                    },
                );
            }
        }
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("PowerShell download failed: {}", stderr.trim());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/// Extract a .7z archive into `data_root` using sevenz-rust (pure Rust LZMA2).
///
/// After extraction, `passages-raw.parquet/` is renamed to `passages.parquet/`
/// if present — this handles archives built from the raw (uncompressed) source
/// directory before the canonical name was established.
fn extract_7z(
    arc_path: &Path,
    data_root: &Path,
    progress: Option<&dyn InitProgress>,
) -> Result<()> {
    std::fs::create_dir_all(data_root)
        .with_context(|| format!("creating {}", data_root.display()))?;

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} {elapsed_precise} {msg}")
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
    );
    pb.set_message("Decompressing (LZMA2)…");
    pb.enable_steady_tick(Duration::from_millis(80));

    sevenz_rust::decompress_file(arc_path, data_root)
        .map_err(|e| anyhow::anyhow!("failed to extract {}: {}", arc_path.display(), e))?;

    pb.finish_and_clear();

    // Rename passages-raw.parquet → passages.parquet if the archive used
    // the raw source directory name instead of the canonical pack name.
    let raw = data_root.join("passages-raw.parquet");
    let canonical = data_root.join("passages.parquet");
    if raw.exists() && !canonical.exists() {
        std::fs::rename(&raw, &canonical)
            .context("renaming passages-raw.parquet to passages.parquet")?;
        eprintln!("Renamed passages-raw.parquet → passages.parquet");
    }

    let file_count = walkdir::WalkDir::new(data_root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .count();
    let message = format!("Extracted {file_count} files.");
    eprintln!("{message}");
    emit(progress, InitProgressEvent::Log(message));
    Ok(())
}

// ---------------------------------------------------------------------------
// Local index build (doc_table + catalog + phrase + tfidf; vector left for user)
// ---------------------------------------------------------------------------

/// TF-IDF vocabulary settings for [`build_local_indexes_with`].
#[derive(Debug, Clone, Copy)]
pub struct LexicalSettings {
    pub min_df: u32,
    pub max_df_ratio: f32,
    pub max_features: usize,
}

impl LexicalSettings {
    /// The settings `sinorag init` has always used, tuned on CBETA.
    pub const CBETA: Self = Self {
        min_df: 5,
        max_df_ratio: 0.05,
        max_features: 200_000,
    };

    /// For small personal collections in alphabetic scripts (document
    /// intake): a term only needs to recur once to count, and common content
    /// words are kept (function words are already dropped as stopwords).
    pub const SMALL_COLLECTION: Self = Self {
        min_df: 2,
        max_df_ratio: 0.25,
        max_features: 200_000,
    };
}

/// Build doc_table, catalog and (optionally) the phrase + TF-IDF indexes for
/// whatever is in `out_parquet`, from scratch, with the CBETA settings.
pub fn build_local_indexes(
    data_root: &Path,
    out_parquet: &Path,
    progress: Option<&dyn InitProgress>,
    build_phrase_tfidf: bool,
) -> Result<()> {
    build_local_indexes_with(
        data_root,
        out_parquet,
        progress,
        build_phrase_tfidf,
        LexicalSettings::CBETA,
    )
}

/// [`build_local_indexes`] with explicit TF-IDF settings. Used by the
/// Windows document-intake GUI after it re-ingests its corpus.
pub fn build_local_indexes_with(
    data_root: &Path,
    out_parquet: &Path,
    progress: Option<&dyn InitProgress>,
    build_phrase_tfidf: bool,
    lexical: LexicalSettings,
) -> Result<()> {
    let doc_table_path = data_root.join("derived").join("doc_table.bin");
    let catalog_path = data_root.join("derived").join("catalog.index");
    let phrase_path = data_root.join("derived").join("phrase.index");
    let tfidf_path = data_root.join("derived").join("tfidf.index");

    if let Some(parent) = doc_table_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    eprintln!("\n=== Building doc_table ===");
    emit(
        progress,
        InitProgressEvent::Step {
            label: "Building document table".to_string(),
            progress: Some(0.48),
        },
    );
    crate::commands::document_table::build(
        out_parquet.to_path_buf(),
        doc_table_path.clone(),
        None,
    )?;

    eprintln!("\n=== Building catalog index ===");
    emit(
        progress,
        InitProgressEvent::Step {
            label: "Building catalog index".to_string(),
            progress: Some(0.56),
        },
    );
    crate::commands::catalog_index::build(
        out_parquet.to_path_buf(),
        catalog_path.clone(),
        None,
        Some(doc_table_path.clone()),
    )?;

    if build_phrase_tfidf {
        eprintln!("\n=== Building phrase + TF-IDF indexes ===");
        emit(
            progress,
            InitProgressEvent::Step {
                label: "Building phrase and TF-IDF indexes".to_string(),
                progress: Some(0.68),
            },
        );
        crate::commands::build_all_indexes(
            out_parquet.to_path_buf(),
            doc_table_path.clone(),
            phrase_path,
            tfidf_path,
            4,       // phrase_gram_len
            5,       // min_ngram
            8,       // max_ngram
            lexical.min_df,
            lexical.max_df_ratio,
            lexical.max_features,
            2048,    // buckets
            None,    // temp_dir
            progress,
        )?;
    } else {
        emit(
            progress,
            InitProgressEvent::Log(
                "Phrase and TF-IDF indexing skipped — run `sinorag indexes lexical` later to build them.".to_string(),
            ),
        );
    }

    crate::commands::ingest::initialize_registry_after_ingest(
        data_root,
        &doc_table_path,
        Some(&catalog_path),
        None,
        None,
    )?;

    eprintln!("\nCorpus ready. All tools are available except semantic vector search.");
    eprintln!("  Check state:               sinorag status");
    eprintln!("  Add semantic search:       sinorag index vector-update --model bge-small-zh-v1.5");
    eprintln!("  Add a CEF corpus:          sinorag ingest cef <path>");
    emit(
        progress,
        InitProgressEvent::Step {
            label: "Corpus ready".to_string(),
            progress: Some(1.0),
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// --from-raw: ingest from local CBETA source files
// ---------------------------------------------------------------------------

async fn ingest_from_raw(
    corpus_path: PathBuf,
    out_parquet: PathBuf,
    data_root: PathBuf,
) -> Result<()> {
    let out_jsonl = data_root.join("passages.jsonl");
    let catalog_path = data_root.join("derived").join("catalog.index");
    let phrase_path = data_root.join("derived").join("phrase.index");
    let tfidf_path = data_root.join("derived").join("tfidf.index");

    eprintln!(
        "Ingesting from local CBETA corpus at {}",
        corpus_path.display()
    );
    eprintln!("(phrase and TF-IDF indexes will NOT be built automatically)");

    crate::commands::ingest::run(
        corpus_path,
        out_jsonl,
        out_parquet,
        None,  // resume
        false, // append (fresh init)
        false, // build_phrase_index
        phrase_path,
        4,     // phrase_gram_len default
        false, // build_tfidf
        Some(tfidf_path),
        Some(catalog_path),
        None,  // phrase_max_memory
        false, // parallel_lexical
        crate::storage::ParquetCompression::Zstd,
    )
    .await
}
