//! The intake document store and GD-CEF corpus assembly.
//!
//! ```text
//! <install>/intake/
//!   manifest.json                 one entry per stored document, keyed by SHA-256
//!   docs/<work_id>/work.json      GD-CEF work record
//!   docs/<work_id>/passages.jsonl GD-CEF passage records
//!   docs/<work_id>/text.txt       the extracted text, with page markers, for people
//!   cef/corpus.toml               assembled corpus that `sinorag ingest cef` reads
//!   cef/works.jsonl
//!   cef/passages.jsonl
//! ```
//!
//! Each document is written to its own directory before it is added to the
//! manifest, and `cef/` is rebuilt from the manifest every run, so an
//! interrupted intake never leaves a half-written or duplicated document in
//! the corpus.

use crate::extract::{is_cjk, DocKind, ExtractedDoc};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// The `source_corpus` partition intake documents land in.
pub const CORPUS_ID: &str = "documents";
/// GD-CEF `text_profile` written into the assembled corpus.
pub const TEXT_PROFILE: &str = "words";
const MANIFEST_SCHEMA: &str = "sinorag-intake-manifest-v1";

/// Passage size targets, in characters.
pub const CHUNK_TARGET: usize = 900;
pub const CHUNK_MAX: usize = 1400;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocEntry {
    pub work_id: String,
    pub sha256: String,
    pub file_name: String,
    pub source_path: String,
    pub kind: String,
    pub title: String,
    #[serde(default)]
    pub pages: Option<u32>,
    pub passages: usize,
    #[serde(default)]
    pub ocr_pages: u32,
    pub chars: usize,
    pub added_utc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: String,
    pub documents: Vec<DocEntry>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            schema: MANIFEST_SCHEMA.to_string(),
            documents: Vec::new(),
        }
    }
}

/// `<install>/intake/index-state.json`: the documents the indexes were last
/// built from, in manifest order.
const INDEX_STATE_FILE: &str = "index-state.json";

#[derive(Debug, Serialize, Deserialize)]
struct IndexState {
    work_ids: Vec<String>,
}

pub struct CorpusStore {
    root: PathBuf,
    manifest: Manifest,
}

impl CorpusStore {
    /// Open (or create) the store at `<install>/intake`.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("docs"))
            .with_context(|| format!("creating {}", root.display()))?;
        let manifest_path = root.join("manifest.json");
        let manifest = if manifest_path.is_file() {
            let bytes = fs::read(&manifest_path)?;
            serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing {}", manifest_path.display()))?
        } else {
            Manifest::default()
        };
        Ok(Self { root, manifest })
    }

    /// Read-only peek used by the GUI to show what is already indexed.
    pub fn peek(root: &Path) -> Option<Manifest> {
        let bytes = fs::read(root.join("manifest.json")).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn documents(&self) -> &[DocEntry] {
        &self.manifest.documents
    }

    pub fn find_by_hash(&self, sha256: &str) -> Option<&DocEntry> {
        self.manifest.documents.iter().find(|d| d.sha256 == sha256)
    }

    /// Chunk and store one extracted document. Returns its manifest entry, or
    /// `None` (storing nothing) when the text yields no searchable passage.
    pub fn add_document(
        &mut self,
        source: &Path,
        sha256: &str,
        doc: &ExtractedDoc,
    ) -> Result<Option<DocEntry>> {
        let work_id = self.unique_work_id(sha256);
        let dir = self.root.join("docs").join(&work_id);

        let file_name = source
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let source_url = file_url(source);

        // Passages: chunk each section; ids carry page + chunk so they stay stable
        // and readable (doc-1a2b3c4d5e6f#p0012.02).
        let mut passages = Vec::new();
        let mut human = String::new();
        human.push_str(&format!("# {}\n# source: {}\n# sha256: {}\n\n", doc.title, source.display(), sha256));
        let mut chars = 0usize;
        let mut ordinal = 0usize;
        for section in &doc.sections {
            if let Some(page) = section.page {
                human.push_str(&format!("\n=== page {page} ===\n"));
            }
            human.push_str(section.text.trim_end());
            human.push('\n');
            for (i, chunk) in chunk_text(&section.text, CHUNK_TARGET, CHUNK_MAX)
                .into_iter()
                .enumerate()
            {
                ordinal += 1;
                chars += chunk.chars().count();
                let (passage_id, locator) = match section.page {
                    Some(page) => (
                        format!("{work_id}#p{page:04}.{:02}", i + 1),
                        format!("p. {page}"),
                    ),
                    None => (format!("{work_id}#s{ordinal:04}"), format!("part {ordinal}")),
                };
                let heading_path = match section.page {
                    Some(page) => format!("{} > page {page}", doc.title),
                    None => doc.title.clone(),
                };
                passages.push(json!({
                    "passage_id": passage_id,
                    "work_id": work_id,
                    "text": chunk,
                    "section_id": section.page.map(|p| format!("page-{p}")),
                    "section_title": doc.title,
                    "locator": locator,
                    "source_rel_path": format!("{CORPUS_ID}/{file_name}"),
                    "source_url": source_url,
                    "heading_path": heading_path,
                    "text_type": "prose",
                    "quality_flags": (doc.ocr_pages > 0).then(|| json!({"ocr_pages": doc.ocr_pages})),
                }));
            }
        }

        if passages.is_empty() {
            return Ok(None);
        }
        fs::create_dir_all(&dir)?;

        let work = json!({
            "work_id": work_id,
            "title_zh": doc.title,
            "title_en": doc.title,
            "genre": doc.kind.label(),
            "source_rel_path": format!("{CORPUS_ID}/{file_name}"),
            "source_url": source_url,
            "quality_flags": json!({
                "kind": doc.kind.label(),
                "pages": doc.page_count,
                "ocr_pages": doc.ocr_pages,
                "encoding": doc.encoding,
                "sha256": sha256,
            }),
        });

        write_atomic(&dir.join("work.json"), serde_json::to_string(&work)?.as_bytes())?;
        let mut jsonl = String::new();
        for p in &passages {
            jsonl.push_str(&serde_json::to_string(p)?);
            jsonl.push('\n');
        }
        write_atomic(&dir.join("passages.jsonl"), jsonl.as_bytes())?;
        write_atomic(&dir.join("text.txt"), human.as_bytes())?;

        let entry = DocEntry {
            work_id,
            sha256: sha256.to_string(),
            file_name,
            source_path: source.display().to_string(),
            kind: doc.kind.label().to_string(),
            title: doc.title.clone(),
            pages: doc.page_count,
            passages: passages.len(),
            ocr_pages: doc.ocr_pages,
            chars,
            added_utc: chrono::Utc::now().to_rfc3339(),
        };
        self.manifest.documents.push(entry.clone());
        self.save_manifest()?;
        Ok(Some(entry))
    }

    fn unique_work_id(&self, sha256: &str) -> String {
        // 12 hex digits is plenty; lengthen on the (astronomically rare) clash.
        let mut len = 12;
        loop {
            let id = format!("doc-{}", &sha256[..len.min(sha256.len())]);
            if !self.manifest.documents.iter().any(|d| d.work_id == id) || len >= sha256.len() {
                return id;
            }
            len += 4;
        }
    }

    fn save_manifest(&self) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(&self.manifest)?;
        write_atomic(&self.root.join("manifest.json"), &bytes)
    }

    /// True when the indexes were last built from exactly the documents now
    /// in the manifest. False after a run that stored documents but was
    /// closed, or failed, before the rebuild finished: the indexes on disk
    /// are then intact but do not cover those documents.
    pub fn indexes_cover_manifest(&self) -> bool {
        let Ok(bytes) = fs::read(self.root.join(INDEX_STATE_FILE)) else {
            return false;
        };
        let Ok(state) = serde_json::from_slice::<IndexState>(&bytes) else {
            return false;
        };
        state.work_ids.len() == self.manifest.documents.len()
            && state
                .work_ids
                .iter()
                .zip(&self.manifest.documents)
                .all(|(indexed, doc)| *indexed == doc.work_id)
    }

    /// Forget which documents the indexes cover. Call before replacing them.
    pub fn clear_index_state(&self) -> Result<()> {
        let path = self.root.join(INDEX_STATE_FILE);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e).with_context(|| format!("removing {}", path.display())),
        }
    }

    /// Record that the indexes now cover every document in the manifest.
    pub fn record_index_state(&self) -> Result<()> {
        let state = IndexState {
            work_ids: self.manifest.documents.iter().map(|d| d.work_id.clone()).collect(),
        };
        write_atomic(&self.root.join(INDEX_STATE_FILE), &serde_json::to_vec(&state)?)
    }

    /// True when the last assembled corpus used a different text profile, so
    /// the indexes must be rebuilt even if no document was added.
    pub fn profile_changed(&self) -> bool {
        let path = self.root.join("cef").join("corpus.toml");
        match fs::read_to_string(path) {
            Ok(text) => !text.contains(&format!("text_profile = \"{TEXT_PROFILE}\"")),
            Err(_) => false,
        }
    }

    /// Rebuild `cef/` from every document in the manifest. Returns the
    /// corpus directory and the passage count.
    pub fn assemble_cef(&self) -> Result<(PathBuf, usize)> {
        let cef = self.root.join("cef");
        fs::create_dir_all(&cef)?;
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let corpus_toml = format!(
            r#"schema = "gd-cef-v1"
corpus_id = "{CORPUS_ID}"
name = "Local documents"
language = "mul"
snapshot_id = "{today}"
description = "PDF, HTML and text files added through SinoRAG Document Intake."
source_type = "local-files"
# Lowercased, whole-word matching for English and other alphabetic scripts;
# Chinese/Japanese/Korean text is normalized exactly as for CBETA.
text_profile = "{TEXT_PROFILE}"
rights_id = "local"
rights_notes = "Documents supplied by the user; rights are the user's own."
default_period = ""
default_period_rank = 99

[conversion]
converter_name = "sinorag-intake"
converter_version = "{}"
conversion_date = "{today}"
notes = "PDF text via the Lege PDF pipeline (native text layer, OCR fallback)."
"#,
            env!("CARGO_PKG_VERSION")
        );
        write_atomic(&cef.join("corpus.toml"), corpus_toml.as_bytes())?;

        let mut works = BufWriter::new(fs::File::create(cef.join("works.jsonl"))?);
        let mut passages = BufWriter::new(fs::File::create(cef.join("passages.jsonl"))?);
        let mut count = 0usize;
        for doc in &self.manifest.documents {
            let dir = self.root.join("docs").join(&doc.work_id);
            let work = fs::read_to_string(dir.join("work.json"))
                .with_context(|| format!("reading stored document {}", doc.work_id))?;
            writeln!(works, "{}", work.trim())?;
            let body = fs::read_to_string(dir.join("passages.jsonl"))
                .with_context(|| format!("reading stored passages {}", doc.work_id))?;
            for line in body.lines().filter(|l| !l.trim().is_empty()) {
                writeln!(passages, "{line}")?;
                count += 1;
            }
        }
        works.flush()?;
        passages.flush()?;
        Ok((cef, count))
    }
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn file_url(path: &Path) -> String {
    let abs = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    path_to_file_url(&abs.display().to_string())
}

fn path_to_file_url(s: &str) -> String {
    // canonicalize returns the verbatim form: \\?\C:\dir\file or \\?\UNC\server\share\file.
    let (prefix, rest) = if let Some(unc) = s.strip_prefix(r"\\?\UNC\") {
        ("file://", unc)
    } else if let Some(unc) = s.strip_prefix(r"\\").filter(|r| !r.starts_with('?')) {
        ("file://", unc)
    } else {
        let local = s.strip_prefix(r"\\?\").unwrap_or(s);
        (if local.starts_with('/') { "file://" } else { "file:///" }, local)
    };
    // Escape what would end or corrupt the path part of a URL. Non-ASCII
    // letters stay readable.
    let mut url = String::from(prefix);
    for c in rest.chars() {
        match c {
            '\\' => url.push('/'),
            ' ' | '#' | '?' | '%' | '"' | '<' | '>' | '[' | ']' | '^' | '`' | '{' | '|' | '}' => {
                url.push_str(&format!("%{:02X}", c as u32))
            }
            c if c.is_control() => url.push_str(&format!("%{:02X}", c as u32)),
            c => url.push(c),
        }
    }
    url
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp-write");
    fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

// ── Chunking ────────────────────────────────────────────────────────────────

/// Split a page (or a whole HTML/text body) into passages of roughly
/// `target` characters, never more than `max`, breaking at paragraph, then
/// sentence, then character boundaries. Wrapped lines inside a paragraph are
/// re-joined (with a space for alphabetic scripts, without one for CJK, and
/// with end-of-line hyphenation undone).
pub fn chunk_text(text: &str, target: usize, max: usize) -> Vec<String> {
    let paragraphs = paragraphs(text);
    let mut pieces: Vec<String> = Vec::new();
    for para in paragraphs {
        if para.chars().count() <= max {
            pieces.push(para);
        } else {
            pieces.extend(split_long(&para, target, max));
        }
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_len = 0usize;
    for piece in pieces {
        let len = piece.chars().count();
        if current_len > 0 && current_len + 1 + len > target {
            chunks.push(std::mem::take(&mut current));
            current_len = 0;
        }
        if current_len > 0 {
            current.push('\n');
            current_len += 1;
        }
        current.push_str(&piece);
        current_len += len;
    }
    if current_len > 0 {
        chunks.push(current);
    }

    // Fold a short tail into its predecessor when that stays under `max`.
    if chunks.len() >= 2 {
        let last_len = chunks[chunks.len() - 1].chars().count();
        let prev_len = chunks[chunks.len() - 2].chars().count();
        if last_len < target / 4 && prev_len + 1 + last_len <= max {
            let last = chunks.pop().unwrap_or_default();
            if let Some(prev) = chunks.last_mut() {
                prev.push('\n');
                prev.push_str(&last);
            }
        }
    }

    chunks
        .into_iter()
        .filter(|c| c.chars().filter(|ch| ch.is_alphanumeric()).count() >= 2)
        .collect()
}

fn paragraphs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            continue;
        }
        if current.is_empty() {
            current.push_str(line);
            continue;
        }
        let prev = current.chars().last().unwrap_or(' ');
        let next = line.chars().next().unwrap_or(' ');
        let prev2 = current.chars().rev().nth(1).unwrap_or(' ');
        if prev == '-' && prev2.is_alphabetic() && next.is_lowercase() {
            current.pop(); // "exam-\nple" → "example"
        } else if !(is_cjk(prev) || is_cjk(next) || is_cjk_punct(prev)) {
            current.push(' ');
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn is_cjk_punct(c: char) -> bool {
    matches!(c, '\u{3000}'..='\u{303f}' | '\u{ff00}'..='\u{ffef}')
}

fn split_long(para: &str, target: usize, max: usize) -> Vec<String> {
    // Sentences: end after 。！？；!? or a '.' followed by whitespace.
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = para.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        current.push(c);
        let next = chars.get(i + 1).copied();
        let end = matches!(c, '。' | '！' | '？' | '；' | '!' | '?')
            || (c == '.' && next.is_none_or(char::is_whitespace));
        if end {
            sentences.push(std::mem::take(&mut current).trim().to_string());
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }

    let mut out = Vec::new();
    let mut buf = String::new();
    let mut buf_len = 0usize;
    for sentence in sentences {
        let len = sentence.chars().count();
        if len > max {
            if buf_len > 0 {
                out.push(std::mem::take(&mut buf));
                buf_len = 0;
            }
            out.extend(hard_split(&sentence, target));
            continue;
        }
        if buf_len > 0 && buf_len + 1 + len > target {
            out.push(std::mem::take(&mut buf));
            buf_len = 0;
        }
        if buf_len > 0 {
            let joiner = if sentence.chars().next().is_some_and(is_cjk) { "" } else { " " };
            buf.push_str(joiner);
            buf_len += joiner.len();
        }
        buf.push_str(&sentence);
        buf_len += len;
    }
    if buf_len > 0 {
        out.push(buf);
    }
    out
}

/// Cut text that has no sentence break into pieces of at most `target`
/// characters. A piece ends at a space when there is one in its second half,
/// so a word is cut in two (and lost to whole-word search) only when the
/// text really is unbroken, as CJK is.
fn hard_split(text: &str, target: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let target = target.max(1);
    let mut out = Vec::new();
    let mut start = 0;
    while chars.len() - start > target {
        let window = &chars[start..start + target];
        let cut = window
            .iter()
            .rposition(|c| c.is_whitespace())
            .filter(|at| *at >= target / 2)
            .map_or(target, |at| at + 1);
        out.push(window[..cut].iter().collect::<String>().trim_end().to_string());
        start += cut;
    }
    let tail: String = chars[start..].iter().collect();
    if !tail.trim().is_empty() {
        out.push(tail.trim().to_string());
    }
    out
}

/// Kinds a folder scan picks up.
pub fn collect_files(paths: &[PathBuf]) -> Vec<(PathBuf, DocKind)> {
    let mut out = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut found: Vec<(PathBuf, DocKind)> = walk(path)
                .into_iter()
                .filter_map(|p| DocKind::from_path(&p).map(|k| (p, k)))
                .collect();
            found.sort_by(|a, b| a.0.cmp(&b.0));
            out.extend(found);
        } else if let Some(kind) = DocKind::from_path(path) {
            out.push((path.clone(), kind));
        }
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|(p, _)| seen.insert(p.clone()));
    out
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = fs::read_dir(&d) else { continue };
        for entry in rd.flatten() {
            let p = entry.path();
            match entry.file_type() {
                Ok(t) if t.is_dir() => stack.push(p),
                Ok(t) if t.is_file() => out.push(p),
                _ => {}
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejoins_wrapped_latin_lines_and_hyphens() {
        let chunks = chunk_text("The quick brown\nfox jumps over the exam-\nple dog.\n\nNew para.", 900, 1400);
        assert_eq!(chunks, vec!["The quick brown fox jumps over the example dog.\nNew para."]);
    }

    #[test]
    fn rejoins_cjk_without_spaces() {
        let chunks = chunk_text("如是我聞。一時佛在\n舍衛國祇樹給孤獨園", 900, 1400);
        assert_eq!(chunks, vec!["如是我聞。一時佛在舍衛國祇樹給孤獨園"]);
    }

    #[test]
    fn long_text_respects_max() {
        let sentence = "This is a sentence of moderate length. ";
        let text = sentence.repeat(200);
        let chunks = chunk_text(&text, 900, 1400);
        assert!(chunks.len() > 5);
        assert!(chunks.iter().all(|c| c.chars().count() <= 1400));
    }

    #[test]
    fn unbroken_cjk_is_hard_split() {
        let text = "佛".repeat(5000);
        let chunks = chunk_text(&text, 900, 1400);
        assert!(chunks.iter().all(|c| c.chars().count() <= 1400));
        assert_eq!(chunks.iter().map(|c| c.chars().count()).sum::<usize>(), 5000);
    }

    #[test]
    fn text_without_sentence_breaks_is_split_between_words() {
        // A table or index: thousands of characters and no full stop.
        let words: Vec<String> = (0..600).map(|i| format!("entry{i}")).collect();
        let chunks = chunk_text(&words.join(" "), 900, 1400);
        assert!(chunks.len() > 3);
        assert!(chunks.iter().all(|c| c.chars().count() <= 1400));
        let rejoined: Vec<&str> = chunks.iter().flat_map(|c| c.split_whitespace()).collect();
        assert_eq!(rejoined, words.iter().map(String::as_str).collect::<Vec<_>>());
    }

    #[test]
    fn file_urls_escape_reserved_characters() {
        assert_eq!(
            path_to_file_url(r"\\?\C:\My Docs\R&D #3 100%.pdf"),
            "file:///C:/My%20Docs/R&D%20%233%20100%25.pdf"
        );
        assert_eq!(
            path_to_file_url(r"\\?\UNC\server\share\經 典.txt"),
            "file://server/share/經%20典.txt"
        );
    }

    fn sample_doc(text: &str) -> ExtractedDoc {
        ExtractedDoc {
            kind: DocKind::Text,
            title: "Sample".to_string(),
            sections: vec![crate::extract::Section { page: None, text: text.to_string() }],
            page_count: None,
            ocr_pages: 0,
            notes: Vec::new(),
            encoding: None,
        }
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sinorag-intake-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn index_state_tracks_the_manifest() {
        let root = scratch_dir("index-state");
        let mut store = CorpusStore::open(&root).unwrap();
        assert!(!store.indexes_cover_manifest());

        let a = "a".repeat(64);
        store.add_document(Path::new("a.txt"), &a, &sample_doc("The coolant pump.")).unwrap().unwrap();
        assert!(!store.indexes_cover_manifest());
        store.record_index_state().unwrap();
        assert!(store.indexes_cover_manifest());

        // A document stored after the last rebuild makes the indexes stale,
        // and that survives reopening the store (a closed or crashed run).
        let b = "b".repeat(64);
        store.add_document(Path::new("b.txt"), &b, &sample_doc("The heron colony.")).unwrap().unwrap();
        assert!(!store.indexes_cover_manifest());
        let store = CorpusStore::open(&root).unwrap();
        assert!(!store.indexes_cover_manifest());
        store.record_index_state().unwrap();
        assert!(store.indexes_cover_manifest());
        store.clear_index_state().unwrap();
        assert!(!store.indexes_cover_manifest());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn text_with_nothing_searchable_is_not_stored() {
        let root = scratch_dir("empty-doc");
        let mut store = CorpusStore::open(&root).unwrap();
        let sha = "c".repeat(64);
        assert!(store.add_document(Path::new("c.txt"), &sha, &sample_doc("— 3 —\n\n* * *")).unwrap().is_none());
        assert!(store.documents().is_empty());
        assert!(!root.join("docs").join("doc-cccccccccccc").exists());
        let _ = fs::remove_dir_all(&root);
    }
}
