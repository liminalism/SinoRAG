// TF-IDF index over CJK character n-grams (8-bit log-quantized).
//
// Builder structure is partially inspired by the local `fasttfidf` crate:
// stream already-normalized text, split parquet work at row-group granularity,
// and aggregate worker-local document frequencies before writing buckets. We
// intentionally keep SinoRAG's CJK n-gram hashing, Rayon execution, stable
// hash-sorted term IDs, and mmap-oriented index format.
//
// Pipeline:
//   Phase 1 — DF buckets:    write pre-aggregated (gram_hash, df_count)
//                            records per bucket.
//   Phase 2 — Build vocab:   merge buckets, sum DF, filter by min/max DF +
//                            max_features, assign term_ids.
//   Phase 3 — Rows + posting buckets: re-scan parquet, compute TF, multiply
//             by IDF, L2-normalise, write row entries inline + write
//             (term_id, doc_id, weight) records into posting buckets.
//             Writer thread streams temporary f32 row records and tracks
//             per-term `w_max`.
//   Phase 4 — Posting merge: for each posting bucket, sort by (term_id, doc_id)
//             and emit a contiguous posting list per term. Quantize on emit
//             using `w_max[tid]`.
//   Requantize rows — second pass over the temporary f32 row records,
//             quantizing each weight against `w_max` and rewriting as 5-byte
//             entries.
//   Save  —   write the file directly from the build's local Vecs.
//
// On-disk format: header (512 B) + vocab table
// (32 B/entry, adds w_max:f32) + IDF array + per-doc row offsets + per-doc
// row lengths + row blob (5 B/entry: term_id u32 + q u8) + postings blob
// (5 B/entry: doc_id u32 + q u8). See header constant ranges below.
//
// Query side: `TfidfIndex::open` mmaps the file. Every accessor is a slice
// into the mmap; weights are dequantized on the fly via per-term `w_max`.

use crate::arrow_helpers::extract_passage_columns;
use crate::document_table::DocumentTable;
use crate::text_analyzer::{analyze_normalized, AnalyzeOptions, AnalyzeScratch, FilterMode};
use crate::tfidf::ngram::{char_ngram_hashes, char_ngrams};
use crate::tfidf::quantize::{dequantize_log_u8, quantize_log_u8};
use anyhow::Result;
use memmap2::Mmap;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use xxhash_rust::xxh3::xxh3_64;

pub type DocId = u32;
pub type TermId = u32;

// ---------------------------------------------------------------------------
// File format constants
// ---------------------------------------------------------------------------

const MAGIC_TFIDF: &[u8; 8] = b"SRTF3VAA";
const HEADER_SIZE: usize = 512;
const VOCAB_ENTRY_SIZE: usize = 32; // u64 + u32 + u32 + u64 + u32 + f32
const ROW_ENTRY_SIZE: usize = 5; // term_id u32 + qweight u8
const POSTING_ENTRY_SIZE: usize = 5; // doc_id u32 + qweight u8
const POSTING_BUCKET_RECORD_SIZE: usize = 12; // term_id u32 + doc_id u32 + weight f32

const HDR_VOCAB_COUNT: std::ops::Range<usize> = 10..14;
const HDR_DOC_COUNT: std::ops::Range<usize> = 14..18;
const HDR_MIN_NGRAM: std::ops::Range<usize> = 18..20;
const HDR_MAX_NGRAM: std::ops::Range<usize> = 20..22;
const HDR_MIN_DF: std::ops::Range<usize> = 22..26;
const HDR_MAX_FEATURES: std::ops::Range<usize> = 26..30;
const HDR_MAX_DF_RATIO: std::ops::Range<usize> = 30..34;
const HDR_FP: std::ops::Range<usize> = 34..98;
const HDR_VOCAB_OFF: std::ops::Range<usize> = 98..106;
const HDR_IDF_OFF: std::ops::Range<usize> = 106..114;
const HDR_ROW_OFFSETS: std::ops::Range<usize> = 114..122;
const HDR_ROW_LENGTHS: std::ops::Range<usize> = 122..130;
const HDR_ROW_BLOB_OFF: std::ops::Range<usize> = 130..138;
const HDR_ROW_BLOB_LEN: std::ops::Range<usize> = 138..146;
const HDR_POST_BLOB_OFF: std::ops::Range<usize> = 146..154;
const HDR_POST_BLOB_LEN: std::ops::Range<usize> = 154..162;
const HDR_QUERY_OFF: std::ops::Range<usize> = 162..170;
const HDR_QUERY_COUNT: std::ops::Range<usize> = 170..178;

const QUERY_ENTRY_SIZE: usize = 12; // term_hash u64 + term_id u32

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfidfParams {
    pub min_ngram: usize,
    pub max_ngram: usize,
    pub min_df: u32,
    pub max_df_ratio: f32,
    pub max_features: usize,
    pub dtype: String,
    pub analyzer: String,
}

impl TfidfParams {
    pub fn default() -> Self {
        Self {
            min_ngram: 5,
            max_ngram: 8,
            min_df: 5,
            max_df_ratio: 0.05,
            max_features: 200_000,
            dtype: "float32".to_string(),
            analyzer: "char".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VocabEntry {
    pub term_hash: u64,
    pub term_id: TermId,
    pub df: u32,
    pub postings_offset: u64,
    pub postings_count: u32,
    pub w_max: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SharedNgram {
    pub ngram: String,
    pub contribution: f32,
    pub seed_weight: f32,
    pub candidate_weight: f32,
}

/// Mmap-backed TF-IDF index. All sections are accessed by slicing into the
/// mmap; quantized weights are dequantized on-the-fly per term.
pub struct TfidfIndex {
    params: TfidfParams,
    doc_table_fingerprint: String,
    version: u16,
    vocab_count: usize,
    doc_count: usize,
    vocab_off: usize,
    idf_off: usize,
    row_offsets_off: usize,
    row_lengths_off: usize,
    row_blob_off: usize,
    row_blob_len: usize,
    postings_blob_off: usize,
    postings_blob_len: usize,
    query_table_off: usize,
    query_table_count: usize,
    mmap: Arc<Mmap>,
}

// ---------------------------------------------------------------------------
// Open + accessors
// ---------------------------------------------------------------------------

impl TfidfIndex {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    pub fn load(path: &Path) -> Result<Self> {
        Self::open(path)
    }

    fn from_mmap(mmap: Mmap) -> Result<Self> {
        if mmap.len() < HEADER_SIZE {
            anyhow::bail!("TF-IDF index too small");
        }
        if &mmap[0..8] != MAGIC_TFIDF {
            anyhow::bail!("invalid TF-IDF magic; rebuild required");
        }
        let version = u16::from_le_bytes([mmap[8], mmap[9]]);
        if version != 3 && version != 4 {
            anyhow::bail!(
                "unsupported TF-IDF format version {}; rebuild required",
                version
            );
        }

        let vocab_count = u32::from_le_bytes(mmap[HDR_VOCAB_COUNT].try_into()?) as usize;
        let doc_count = u32::from_le_bytes(mmap[HDR_DOC_COUNT].try_into()?) as usize;
        let min_ngram = u16::from_le_bytes(mmap[HDR_MIN_NGRAM].try_into()?) as usize;
        let max_ngram = u16::from_le_bytes(mmap[HDR_MAX_NGRAM].try_into()?) as usize;
        let min_df = u32::from_le_bytes(mmap[HDR_MIN_DF].try_into()?);
        let max_features = u32::from_le_bytes(mmap[HDR_MAX_FEATURES].try_into()?) as usize;
        let max_df_ratio = f32::from_le_bytes(mmap[HDR_MAX_DF_RATIO].try_into()?);
        let fp_end = mmap[HDR_FP]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(HDR_FP.len());
        let doc_table_fingerprint =
            String::from_utf8_lossy(&mmap[HDR_FP.start..HDR_FP.start + fp_end]).to_string();

        let vocab_off = u64::from_le_bytes(mmap[HDR_VOCAB_OFF].try_into()?) as usize;
        let idf_off = u64::from_le_bytes(mmap[HDR_IDF_OFF].try_into()?) as usize;
        let row_offsets_off = u64::from_le_bytes(mmap[HDR_ROW_OFFSETS].try_into()?) as usize;
        let row_lengths_off = u64::from_le_bytes(mmap[HDR_ROW_LENGTHS].try_into()?) as usize;
        let row_blob_off = u64::from_le_bytes(mmap[HDR_ROW_BLOB_OFF].try_into()?) as usize;
        let row_blob_len = u64::from_le_bytes(mmap[HDR_ROW_BLOB_LEN].try_into()?) as usize;
        let postings_blob_off = u64::from_le_bytes(mmap[HDR_POST_BLOB_OFF].try_into()?) as usize;
        let postings_blob_len = u64::from_le_bytes(mmap[HDR_POST_BLOB_LEN].try_into()?) as usize;

        if postings_blob_off + postings_blob_len > mmap.len() {
            anyhow::bail!("TF-IDF sections exceed file length");
        }

        let (query_table_off, query_table_count) = if version >= 4 {
            let off = u64::from_le_bytes(mmap[HDR_QUERY_OFF].try_into()?) as usize;
            let count = u64::from_le_bytes(mmap[HDR_QUERY_COUNT].try_into()?) as usize;
            if off + count * QUERY_ENTRY_SIZE > mmap.len() {
                anyhow::bail!("TF-IDF query table exceeds file length");
            }
            (off, count)
        } else {
            (0, 0)
        };

        let params = TfidfParams {
            min_ngram,
            max_ngram,
            min_df,
            max_df_ratio,
            max_features,
            dtype: "u8-log-quantized".to_string(),
            analyzer: "char".to_string(),
        };

        Ok(Self {
            params,
            doc_table_fingerprint,
            version,
            vocab_count,
            doc_count,
            vocab_off,
            idf_off,
            row_offsets_off,
            row_lengths_off,
            row_blob_off,
            row_blob_len,
            postings_blob_off,
            postings_blob_len,
            query_table_off,
            query_table_count,
            mmap: Arc::new(mmap),
        })
    }

    pub fn params(&self) -> &TfidfParams {
        &self.params
    }
    pub fn doc_count(&self) -> usize {
        self.doc_count
    }
    pub fn vocab_count(&self) -> usize {
        self.vocab_count
    }
    pub fn doc_table_fingerprint(&self) -> &str {
        &self.doc_table_fingerprint
    }

    pub fn vocab_entry(&self, term_id: TermId) -> VocabEntry {
        let off = self.vocab_off + (term_id as usize) * VOCAB_ENTRY_SIZE;
        let s = &self.mmap[off..off + VOCAB_ENTRY_SIZE];
        VocabEntry {
            term_hash: u64::from_le_bytes(s[0..8].try_into().unwrap()),
            term_id: u32::from_le_bytes(s[8..12].try_into().unwrap()),
            df: u32::from_le_bytes(s[12..16].try_into().unwrap()),
            postings_offset: u64::from_le_bytes(s[16..24].try_into().unwrap()),
            postings_count: u32::from_le_bytes(s[24..28].try_into().unwrap()),
            w_max: f32::from_le_bytes(s[28..32].try_into().unwrap()),
        }
    }

    fn idf_at(&self, term_id: TermId) -> f32 {
        let off = self.idf_off + (term_id as usize) * 4;
        f32::from_le_bytes(self.mmap[off..off + 4].try_into().unwrap())
    }

    fn row_offset_at(&self, doc_id: DocId) -> u64 {
        let off = self.row_offsets_off + (doc_id as usize) * 8;
        u64::from_le_bytes(self.mmap[off..off + 8].try_into().unwrap())
    }

    fn row_length_at(&self, doc_id: DocId) -> u32 {
        let off = self.row_lengths_off + (doc_id as usize) * 4;
        u32::from_le_bytes(self.mmap[off..off + 4].try_into().unwrap())
    }

    /// Decode a single document's row into (term_id, dequantized weight) pairs.
    pub fn row_entries(&self, doc_id: DocId) -> Vec<(TermId, f32)> {
        let idx = doc_id as usize;
        if idx >= self.doc_count {
            return Vec::new();
        }
        let off = self.row_offset_at(doc_id);
        if off == u64::MAX {
            return Vec::new();
        }
        let len = self.row_length_at(doc_id) as usize;
        let base = self.row_blob_off + off as usize;
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let e = base + i * ROW_ENTRY_SIZE;
            let tid = u32::from_le_bytes(self.mmap[e..e + 4].try_into().unwrap());
            let q = self.mmap[e + 4];
            let w_max = self.vocab_entry(tid).w_max;
            out.push((tid, dequantize_log_u8(q, w_max)));
        }
        out
    }

    /// Cosine similarity via posting lists. Row vectors were L2-normalised
    /// pre-quantization, so dot product approximates cosine similarity.
    pub fn similar(&self, doc_id: DocId, limit: usize) -> Result<Vec<(DocId, f32)>> {
        let idx = doc_id as usize;
        if idx >= self.doc_count {
            anyhow::bail!(
                "doc_id {} out of range (doc_count={})",
                doc_id,
                self.doc_count
            );
        }
        let row = self.row_entries(doc_id);
        if row.is_empty() {
            return Ok(Vec::new());
        }

        let mut scores: FxHashMap<DocId, f32> = FxHashMap::default();
        for (tid, seed_w) in row {
            let ve = self.vocab_entry(tid);
            let p_base = self.postings_blob_off + ve.postings_offset as usize;
            let p_count = ve.postings_count as usize;
            let w_max = ve.w_max;
            for j in 0..p_count {
                let p = p_base + j * POSTING_ENTRY_SIZE;
                let cand_doc = u32::from_le_bytes(self.mmap[p..p + 4].try_into().unwrap());
                if cand_doc == doc_id {
                    continue;
                }
                let q = self.mmap[p + 4];
                let cand_w = dequantize_log_u8(q, w_max);
                *scores.entry(cand_doc).or_insert(0.0) += seed_w * cand_w;
            }
        }

        let mut ranked: Vec<(DocId, f32)> = scores.into_iter().filter(|(_, s)| *s > 0.0).collect();
        ranked.sort_by_key(|&(_, s)| Reverse(OrderedFloat(s)));
        ranked.truncate(limit.max(1));
        Ok(ranked)
    }

    pub fn shared_ngrams_with_seed_text(
        &self,
        seed_doc: DocId,
        cand_doc: DocId,
        seed_text: &str,
        limit: usize,
    ) -> Vec<SharedNgram> {
        let seed_row = self.row_entries(seed_doc);
        let cand_row = self.row_entries(cand_doc);
        if seed_row.is_empty() || cand_row.is_empty() {
            return Vec::new();
        }

        let mut hash_to_str: FxHashMap<u64, String> = FxHashMap::default();
        for gram in char_ngrams(seed_text, self.params.min_ngram, self.params.max_ngram) {
            let h = xxh3_64(gram.as_bytes());
            hash_to_str.entry(h).or_insert(gram);
        }
        crate::tfidf::ngram::for_each_word_feature(seed_text, |word, h| {
            hash_to_str.entry(h).or_insert_with(|| word.to_string());
        });

        let mut shared: Vec<SharedNgram> = Vec::new();
        let (mut i, mut j) = (0usize, 0usize);
        while i < seed_row.len() && j < cand_row.len() {
            match seed_row[i].0.cmp(&cand_row[j].0) {
                std::cmp::Ordering::Equal => {
                    let tid = seed_row[i].0;
                    let term_hash = self.vocab_entry(tid).term_hash;
                    if let Some(ngram) = hash_to_str.get(&term_hash) {
                        let sw = seed_row[i].1;
                        let cw = cand_row[j].1;
                        shared.push(SharedNgram {
                            ngram: ngram.clone(),
                            contribution: round_f32(sw * cw, 8),
                            seed_weight: round_f32(sw, 8),
                            candidate_weight: round_f32(cw, 8),
                        });
                    }
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        shared.sort_by_key(|item| {
            Reverse((OrderedFloat(item.contribution), item.ngram.chars().count()))
        });
        shared.truncate(limit);
        shared
    }

    pub fn long_gram_shared_count_with_seed_text(
        &self,
        seed_doc: DocId,
        cand_doc: DocId,
        seed_text: &str,
        min_gram_len: usize,
    ) -> usize {
        let seed_row = self.row_entries(seed_doc);
        let cand_row = self.row_entries(cand_doc);
        if seed_row.is_empty() || cand_row.is_empty() {
            return 0;
        }

        let lo = min_gram_len.max(self.params.min_ngram);
        let hi = self.params.max_ngram;
        if lo > hi {
            return 0;
        }
        let mut long_hashes: FxHashSet<u64> = FxHashSet::default();
        for n in lo..=hi {
            for h in char_ngram_hashes(seed_text, n, n) {
                long_hashes.insert(h);
            }
        }

        let mut count = 0usize;
        let (mut i, mut j) = (0usize, 0usize);
        while i < seed_row.len() && j < cand_row.len() {
            match seed_row[i].0.cmp(&cand_row[j].0) {
                std::cmp::Ordering::Equal => {
                    let tid = seed_row[i].0;
                    if long_hashes.contains(&self.vocab_entry(tid).term_hash) {
                        count += 1;
                    }
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        count
    }

    /// Binary search in the query table (v4+ only). Returns `None` for v3 indexes.
    fn query_lookup(&self, hash: u64) -> Option<TermId> {
        if self.query_table_count == 0 {
            return None;
        }
        let mut lo = 0usize;
        let mut hi = self.query_table_count;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let off = self.query_table_off + mid * QUERY_ENTRY_SIZE;
            let h = u64::from_le_bytes(self.mmap[off..off + 8].try_into().unwrap());
            match h.cmp(&hash) {
                std::cmp::Ordering::Equal => {
                    let tid = u32::from_le_bytes(self.mmap[off + 8..off + 12].try_into().unwrap());
                    return Some(tid);
                }
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }

    /// Score documents against a raw text query using TF-IDF dot product.
    /// Requires a v4 index (query table); returns empty for v3.
    pub fn query_top_k(&self, text: &str, k: usize) -> Vec<(DocId, f32)> {
        if self.query_table_count == 0 || k == 0 {
            return Vec::new();
        }
        let mut term_counts: FxHashMap<TermId, u32> = FxHashMap::default();
        let mut total = 0u32;
        for n in self.params.min_ngram..=self.params.max_ngram {
            for h in char_ngram_hashes(text, n, n) {
                total += 1;
                if let Some(tid) = self.query_lookup(h) {
                    *term_counts.entry(tid).or_insert(0) += 1;
                }
            }
        }
        // Word-layout query text (see `crate::normalize::normalize_words`).
        crate::tfidf::ngram::for_each_word_feature(text, |_, h| {
            total += 1;
            if let Some(tid) = self.query_lookup(h) {
                *term_counts.entry(tid).or_insert(0) += 1;
            }
        });
        if term_counts.is_empty() || total == 0 {
            return Vec::new();
        }
        let tf_total = total as f32;
        let mut query_vec: Vec<(TermId, f32)> = term_counts
            .iter()
            .map(|(&tid, &count)| (tid, (count as f32 / tf_total) * self.idf_at(tid)))
            .collect();
        let norm = query_vec.iter().map(|(_, w)| w * w).sum::<f32>().sqrt();
        if norm == 0.0 {
            return Vec::new();
        }
        for (_, w) in query_vec.iter_mut() {
            *w /= norm;
        }
        let mut scores: FxHashMap<DocId, f32> = FxHashMap::default();
        for (tid, qw) in query_vec {
            let ve = self.vocab_entry(tid);
            let p_base = self.postings_blob_off + ve.postings_offset as usize;
            let p_count = ve.postings_count as usize;
            let w_max = ve.w_max;
            for j in 0..p_count {
                let p = p_base + j * POSTING_ENTRY_SIZE;
                let doc_id = u32::from_le_bytes(self.mmap[p..p + 4].try_into().unwrap());
                let q = self.mmap[p + 4];
                let dw = dequantize_log_u8(q, w_max);
                *scores.entry(doc_id).or_insert(0.0) += qw * dw;
            }
        }
        let mut ranked: Vec<(DocId, f32)> = scores.into_iter().filter(|(_, s)| *s > 0.0).collect();
        ranked.sort_by_key(|&(_, s)| Reverse(OrderedFloat(s)));
        ranked.truncate(k.max(1));
        ranked
    }

    pub fn info_payload(&self) -> serde_json::Value {
        let postings_nnz = self.postings_blob_len / POSTING_ENTRY_SIZE;
        let row_nnz = self.row_blob_len / ROW_ENTRY_SIZE;
        let docs = self.doc_count;
        let cols = self.vocab_count;
        let density = if docs == 0 || cols == 0 {
            0.0
        } else {
            (row_nnz as f64) / ((docs * cols) as f64)
        };
        serde_json::json!({
            "schema": "sinorag-tfidf",
            "version": self.version,
            "quantization": "u8-log",
            "documents": docs,
            "matrix_shape": [docs, cols],
            "matrix_nnz": row_nnz,
            "postings_nnz": postings_nnz,
            "density": round_f64(density, 8),
            "features": cols,
            "row_blob_bytes": self.row_blob_len,
            "postings_blob_bytes": self.postings_blob_len,
            "params": {
                "analyzer": self.params.analyzer,
                "ngram_range": [self.params.min_ngram, self.params.max_ngram],
                "min_df": self.params.min_df,
                "max_df": self.params.max_df_ratio,
                "max_features": self.params.max_features,
                "dtype": self.params.dtype,
            },
            "doc_table_fingerprint": self.doc_table_fingerprint,
        })
    }

    /// Header-only read for `info` on huge files.
    pub fn header_info(path: &Path) -> Result<serde_json::Value> {
        let mut file = File::open(path)?;
        let mut hdr = [0u8; HEADER_SIZE];
        file.read_exact(&mut hdr)?;
        if &hdr[0..8] != MAGIC_TFIDF {
            anyhow::bail!("invalid TF-IDF magic; rebuild required");
        }
        let version = u16::from_le_bytes([hdr[8], hdr[9]]);
        if version != 3 && version != 4 {
            anyhow::bail!(
                "unsupported TF-IDF format version {}; rebuild required",
                version
            );
        }
        let vocab_count = u32::from_le_bytes(hdr[HDR_VOCAB_COUNT].try_into()?) as usize;
        let doc_count = u32::from_le_bytes(hdr[HDR_DOC_COUNT].try_into()?) as usize;
        let min_ngram = u16::from_le_bytes(hdr[HDR_MIN_NGRAM].try_into()?) as usize;
        let max_ngram = u16::from_le_bytes(hdr[HDR_MAX_NGRAM].try_into()?) as usize;
        let min_df = u32::from_le_bytes(hdr[HDR_MIN_DF].try_into()?);
        let max_features = u32::from_le_bytes(hdr[HDR_MAX_FEATURES].try_into()?) as usize;
        let max_df_ratio = f32::from_le_bytes(hdr[HDR_MAX_DF_RATIO].try_into()?);
        let fp_end = hdr[HDR_FP]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(HDR_FP.len());
        let doc_table_fingerprint =
            String::from_utf8_lossy(&hdr[HDR_FP.start..HDR_FP.start + fp_end]).to_string();
        let row_blob_len = u64::from_le_bytes(hdr[HDR_ROW_BLOB_LEN].try_into()?);
        let post_blob_len = u64::from_le_bytes(hdr[HDR_POST_BLOB_LEN].try_into()?);
        let file_bytes = std::fs::metadata(path)?.len();
        Ok(serde_json::json!({
            "schema": "sinorag-tfidf",
            "version": version,
            "quantization": "u8-log",
            "documents": doc_count,
            "features": vocab_count,
            "row_blob_bytes": row_blob_len,
            "postings_blob_bytes": post_blob_len,
            "file_bytes": file_bytes,
            "params": {
                "ngram_range": [min_ngram, max_ngram],
                "min_df": min_df,
                "max_df": max_df_ratio,
                "max_features": max_features,
            },
            "doc_table_fingerprint": doc_table_fingerprint,
        }))
    }
}

// ---------------------------------------------------------------------------
// Free helpers — operate on raw text, no index needed
// ---------------------------------------------------------------------------

/// Word-layout version of [`long_common_substrings`]: shared runs of whole
/// words (longest first, at least `min_len` characters), so a reader sees
/// "coolant pump bearing" rather than ragged character slices of it.
fn long_common_word_runs(a: &str, b: &str, min_len: usize, limit: usize) -> Vec<String> {
    let (short, long) = if a.len() > b.len() { (b, a) } else { (a, b) };
    let words: Vec<&str> = short.split(' ').filter(|w| !w.is_empty()).collect();
    let mut out: Vec<String> = Vec::new();
    for n in (1..=words.len().min(12)).rev() {
        for window in words.windows(n) {
            let run = window.join(" ");
            if run.chars().count() < min_len || out.iter().any(|o| o.contains(&run)) {
                continue;
            }
            if long.contains(&format!(" {run} ")) {
                out.push(run);
                if out.len() >= limit {
                    return out;
                }
            }
        }
    }
    out
}

pub fn long_common_substrings(a: &str, b: &str, min_len: usize, limit: usize) -> Vec<String> {
    if crate::normalize::has_word_layout(a) && crate::normalize::has_word_layout(b) {
        return long_common_word_runs(a, b, min_len, limit);
    }
    let mut seen: FxHashSet<String> = FxHashSet::default();
    let mut matches = Vec::new();
    let (short, long) = if a.chars().count() > b.chars().count() {
        (b, a)
    } else {
        (a, b)
    };
    let chars: Vec<char> = short.chars().collect();
    let max_len = 24usize.min(chars.len());

    for length in (min_len..=max_len).rev() {
        if chars.len() < length {
            continue;
        }
        for index in 0..=(chars.len() - length) {
            let frag: String = chars[index..index + length].iter().collect();
            if seen.insert(frag.clone()) && long.contains(&frag) {
                matches.push(frag);
                if matches.len() >= limit {
                    return matches;
                }
            }
        }
    }
    matches
}

// ---------------------------------------------------------------------------
// External-sort builder helpers
// ---------------------------------------------------------------------------

fn progress_key(p: &Path) -> String {
    let comps: Vec<String> = p
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    match comps.len() {
        0 => String::new(),
        1 => comps[0].clone(),
        n => format!("{}/{}", comps[n - 2], comps[n - 1]),
    }
}

#[derive(Clone)]
struct ParquetWorkUnit {
    path: PathBuf,
    row_group: Option<usize>,
}

fn parquet_work_units(files: &[PathBuf]) -> Result<Vec<ParquetWorkUnit>> {
    let mut units = Vec::new();
    for path in files {
        let builder = crate::parquet_metadata::global_cache().get_or_load(path)?;
        let row_groups = builder.metadata().num_row_groups();
        if row_groups == 0 {
            units.push(ParquetWorkUnit {
                path: path.clone(),
                row_group: None,
            });
        } else {
            for row_group in 0..row_groups {
                units.push(ParquetWorkUnit {
                    path: path.clone(),
                    row_group: Some(row_group),
                });
            }
        }
    }
    Ok(units)
}

fn progress_key_for_unit(unit: &ParquetWorkUnit) -> String {
    match unit.row_group {
        Some(row_group) => format!("{}#rg{}", progress_key(&unit.path), row_group),
        None => progress_key(&unit.path),
    }
}

struct BucketWriterCache {
    temp_dir: PathBuf,
    max_open: usize,
    writers: HashMap<usize, BufWriter<File>>,
    lru: std::collections::VecDeque<usize>,
}

impl BucketWriterCache {
    fn new(temp_dir: PathBuf, max_open: usize) -> Self {
        Self {
            temp_dir,
            max_open: max_open.max(1),
            writers: HashMap::new(),
            lru: std::collections::VecDeque::new(),
        }
    }

    fn bucket_path(&self, bucket: usize) -> PathBuf {
        self.temp_dir.join(format!("bucket-{:04}.bin", bucket))
    }

    fn touch(&mut self, bucket: usize) {
        if let Some(pos) = self.lru.iter().position(|&b| b == bucket) {
            self.lru.remove(pos);
        }
        self.lru.push_back(bucket);
    }

    fn evict_if_needed(&mut self) -> Result<()> {
        while self.writers.len() >= self.max_open {
            let Some(victim) = self.lru.pop_front() else {
                break;
            };
            if let Some(mut w) = self.writers.remove(&victim) {
                w.flush()?;
            }
        }
        Ok(())
    }

    fn ensure(&mut self, bucket: usize) -> Result<()> {
        if !self.writers.contains_key(&bucket) {
            self.evict_if_needed()?;
            let path = self.bucket_path(bucket);
            let file = OpenOptions::new().create(true).append(true).open(path)?;
            self.writers.insert(bucket, BufWriter::new(file));
        }
        self.touch(bucket);
        Ok(())
    }

    fn write_bytes(&mut self, bucket: usize, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        self.ensure(bucket)?;
        let w = self.writers.get_mut(&bucket).unwrap();
        w.write_all(bytes)?;
        Ok(())
    }

    fn flush_all(mut self) -> Result<()> {
        for (_, mut w) in self.writers.drain() {
            w.flush()?;
        }
        Ok(())
    }
}

fn append_df_count_record(buf: &mut Vec<u8>, term_hash: u64, df_count: u32) {
    buf.extend_from_slice(&term_hash.to_le_bytes());
    buf.extend_from_slice(&df_count.to_le_bytes());
}

fn append_posting_record(buf: &mut Vec<u8>, term_id: TermId, doc_id: DocId, weight: f32) {
    buf.extend_from_slice(&term_id.to_le_bytes());
    buf.extend_from_slice(&doc_id.to_le_bytes());
    buf.extend_from_slice(&weight.to_le_bytes());
}

const BUCKET_BUFFER_FLUSH_BYTES: usize = 64 * 1024 * 1024;
const LOCAL_DF_FLUSH_TERMS: usize = 500_000;

fn flush_bucket_buffers(writers: &mut BucketWriterCache, buffers: &mut [Vec<u8>]) -> Result<()> {
    for (bucket, buf) in buffers.iter_mut().enumerate() {
        if !buf.is_empty() {
            writers.write_bytes(bucket, buf)?;
            buf.clear();
        }
    }
    Ok(())
}

fn flush_df_counts(
    writers: &mut BucketWriterCache,
    buffers: &mut [Vec<u8>],
    counts: &mut FxHashMap<u64, u32>,
    bucket_count: usize,
    buffered_bytes: &mut usize,
) -> Result<()> {
    for (&hash, &count) in counts.iter() {
        let bucket = (hash as usize) % bucket_count;
        append_df_count_record(&mut buffers[bucket], hash, count);
        *buffered_bytes += 12;
        if *buffered_bytes >= BUCKET_BUFFER_FLUSH_BYTES {
            flush_bucket_buffers(writers, buffers)?;
            *buffered_bytes = 0;
        }
    }
    counts.clear();
    Ok(())
}

// ---------------------------------------------------------------------------
// Build entry point
// ---------------------------------------------------------------------------

pub fn build(
    parquet_path: PathBuf,
    doc_table_path: PathBuf,
    out_path: PathBuf,
    params: TfidfParams,
    bucket_count: usize,
    temp_dir: Option<PathBuf>,
) -> Result<()> {
    let doc_table = DocumentTable::load(&doc_table_path)?;
    build_from_table(
        parquet_path,
        doc_table,
        out_path,
        params,
        bucket_count,
        temp_dir,
    )
}

pub(crate) fn build_from_table(
    parquet_path: PathBuf,
    doc_table: DocumentTable,
    out_path: PathBuf,
    params: TfidfParams,
    bucket_count: usize,
    temp_dir: Option<PathBuf>,
) -> Result<()> {
    build_from_table_with_progress(
        parquet_path,
        doc_table,
        out_path,
        params,
        bucket_count,
        temp_dir,
        None,
    )
}

pub(crate) fn build_from_table_with_progress(
    parquet_path: PathBuf,
    doc_table: DocumentTable,
    out_path: PathBuf,
    params: TfidfParams,
    bucket_count: usize,
    temp_dir: Option<PathBuf>,
    progress: Option<&crate::phrase_index::BuildProgress<'_>>,
) -> Result<()> {
    let temp_dir = temp_dir.unwrap_or_else(|| {
        let mut p = out_path.as_os_str().to_os_string();
        p.push(".work");
        PathBuf::from(p)
    });

    eprintln!("=== TF-IDF builder (u8-log quantized) ===");
    eprintln!("Parquet : {}", parquet_path.display());
    eprintln!("Output  : {}", out_path.display());
    eprintln!(
        "Params  : min_n={} max_n={} min_df={} max_df={} max_features={}",
        params.min_ngram, params.max_ngram, params.min_df, params.max_df_ratio, params.max_features
    );
    eprintln!("Buckets : {}", bucket_count);
    eprintln!("Temp dir: {}", temp_dir.display());

    let phase1_done = temp_dir.join("phase1.done");
    let phase1_prog = temp_dir.join("phase1_progress.txt");
    let phase2_vocab = temp_dir.join("phase2.vocab.bin");
    let phase2_idf_f = temp_dir.join("phase2.idf.bin");
    let phase2_termid = temp_dir.join("phase2.termid.bin");

    let resuming = phase1_done.exists() || phase1_prog.exists() || phase2_vocab.exists();

    if resuming {
        eprintln!(
            "Resume mode: detected partial run in {}",
            temp_dir.display()
        );
    } else if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)?;
    }

    fs::create_dir_all(&temp_dir)?;
    let df_dir = temp_dir.join("df_buckets");
    let post_dir = temp_dir.join("post_buckets");
    if !phase1_done.exists() {
        fs::create_dir_all(&df_dir)?;
    }
    fs::create_dir_all(&post_dir)?;

    let doc_table_fingerprint = doc_table.source_fingerprint.clone();
    let doc_count = doc_table.passage_ids.len();
    eprintln!("\n[Phase 0] doc table: {} passages", doc_count);

    let files = crate::phrase_index::parquet_files(&parquet_path)?;
    let work_units = parquet_work_units(&files)?;
    eprintln!(
        "  {} parquet file(s), {} row-group work unit(s)",
        files.len(),
        work_units.len()
    );

    // -----------------------------------------------------------------------
    // Phase 1 — DF bucket writes (parallel, resumable)
    // -----------------------------------------------------------------------
    if phase1_done.exists() {
        eprintln!("\n[Phase 1] Skipping (complete)");
    } else {
        eprintln!("\n[Phase 1] DF buckets...");

        let mut completed: FxHashSet<String> = FxHashSet::default();
        if phase1_prog.exists() {
            for line in fs::read_to_string(&phase1_prog)?.lines() {
                if !line.is_empty() {
                    completed.insert(line.to_string());
                }
            }
            eprintln!(
                "  Resume: {}/{} work units already processed",
                completed.len(),
                work_units.len()
            );
        }
        let pending: Vec<&ParquetWorkUnit> = work_units
            .iter()
            .filter(|unit| {
                !completed.contains(&progress_key_for_unit(unit))
                    && !completed.contains(&progress_key(&unit.path))
            })
            .collect();

        let nthreads = rayon::current_num_threads();
        let handles_per_thread = (700usize / nthreads.max(1)).max(4).min(64);
        eprintln!(
            "  {} pending work units on {} threads ({} handles/thread)",
            pending.len(),
            nthreads,
            handles_per_thread
        );
        for t in 0..nthreads {
            fs::create_dir_all(df_dir.join(format!("t{}", t)))?;
        }

        let counter = AtomicUsize::new(completed.len());
        let total = work_units.len();
        emit_build_progress(
            progress,
            "TF-IDF phase 1: computing document frequencies",
            completed.len(),
            total,
            0.25 * fraction(completed.len(), total),
        );
        let prog_f = Mutex::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&phase1_prog)?,
        );
        let min_n = params.min_ngram;
        let max_n = params.max_ngram;
        let analyze_opts = AnalyzeOptions {
            min_n,
            max_n,
            filter: FilterMode::CjkOnly,
            apply_low_value_filter: true,
            dedup: true,
            count_tf: false,
        };

        pending.par_iter().try_for_each(|unit| -> Result<()> {
            let t = rayon::current_thread_index().unwrap_or(0);
            let mut w = BucketWriterCache::new(df_dir.join(format!("t{}", t)), handles_per_thread);
            let mut bucket_buffers: Vec<Vec<u8>> = (0..bucket_count).map(|_| Vec::new()).collect();
            let mut buffered_bytes = 0usize;
            let mut local_df: FxHashMap<u64, u32> = FxHashMap::default();
            let mut scratch = AnalyzeScratch::new();

            let mut builder = crate::parquet_metadata::global_cache().get_or_load(&unit.path)?;
            if let Some(row_group) = unit.row_group {
                builder = builder.with_row_groups(vec![row_group]);
            }
            let reader = builder.build()?;
            for batch in reader {
                let batch = batch?;
                let (pids, texts) = extract_passage_columns(&batch)?;
                for i in 0..batch.num_rows() {
                    if doc_table.doc_id(pids.value(i)).is_some() {
                        analyze_normalized(texts.value(i), &analyze_opts, &mut scratch);
                        for &hash in &scratch.unique {
                            *local_df.entry(hash).or_insert(0) += 1;
                        }
                        if local_df.len() >= LOCAL_DF_FLUSH_TERMS {
                            flush_df_counts(
                                &mut w,
                                &mut bucket_buffers,
                                &mut local_df,
                                bucket_count,
                                &mut buffered_bytes,
                            )?;
                        }
                    }
                }
            }
            flush_df_counts(
                &mut w,
                &mut bucket_buffers,
                &mut local_df,
                bucket_count,
                &mut buffered_bytes,
            )?;
            flush_bucket_buffers(&mut w, &mut bucket_buffers)?;
            w.flush_all()?;

            let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 100 == 0 || n == total {
                eprintln!("  {}/{}", n, total);
                emit_build_progress(
                    progress,
                    "TF-IDF phase 1: computing document frequencies",
                    n,
                    total,
                    0.25 * fraction(n, total),
                );
            }
            writeln!(prog_f.lock().unwrap(), "{}", progress_key_for_unit(unit))?;
            Ok(())
        })?;

        eprintln!(
            "  Merging {} threads × {} buckets...",
            nthreads, bucket_count
        );
        for bucket_idx in 0..bucket_count {
            let main = df_dir.join(format!("bucket-{:04}.bin", bucket_idx));
            let mut out = OpenOptions::new().create(true).append(true).open(&main)?;
            for t in 0..nthreads {
                let src = df_dir.join(format!("t{}/bucket-{:04}.bin", t, bucket_idx));
                match File::open(&src) {
                    Ok(mut f) => {
                        std::io::copy(&mut f, &mut out)?;
                        drop(f);
                        fs::remove_file(&src)?;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e.into()),
                }
            }
        }
        for t in 0..nthreads {
            let _ = fs::remove_dir(df_dir.join(format!("t{}", t)));
        }
        fs::write(&phase1_done, b"")?;
        emit_build_progress(
            progress,
            "TF-IDF phase 1: computing document frequencies",
            total,
            total,
            0.25,
        );
        eprintln!("  Phase 1 complete.");
    }

    // -----------------------------------------------------------------------
    // Phase 2 — build vocabulary (parallel, resumable)
    // -----------------------------------------------------------------------
    let (mut vocab_entries, idf, term_to_id) = if phase2_vocab.exists() {
        eprintln!("\n[Phase 2] Loading saved vocabulary...");
        let ve: Vec<VocabEntry> = bincode::deserialize(&fs::read(&phase2_vocab)?)?;
        let iv: Vec<f32> = bincode::deserialize(&fs::read(&phase2_idf_f)?)?;
        let ti: HashMap<u64, TermId> = bincode::deserialize(&fs::read(&phase2_termid)?)?;
        eprintln!("  {} terms loaded", ve.len());
        (ve, iv, ti)
    } else {
        // Guard: Phase 1 done but df_dir missing and no phase2 checkpoint means
        // the previous run crashed mid-Phase-2 after df_dir was already deleted.
        // Silently proceeding would build an empty vocabulary — bail instead.
        if !df_dir.exists() {
            anyhow::bail!(
                "TF-IDF build is in a corrupted resume state: Phase 1 completed and \
                 df_buckets/ was already removed, but the Phase 2 vocabulary checkpoint \
                 (phase2.vocab.bin) was never written.\n\
                 \n\
                 Remove the work directory and start fresh:\n  \
                   rm -rf {}\n  sinorag init",
                temp_dir.display()
            );
        }

        eprintln!("\n[Phase 2] Building vocabulary...");
        emit_build_progress(
            progress,
            "TF-IDF phase 2: building vocabulary",
            0,
            bucket_count,
            0.25,
        );

        // Gram hashes are partitioned across buckets by `hash % bucket_count`,
        // so every DF record for a term lands in exactly one bucket and its
        // global document frequency is fully known within that bucket. Apply the
        // min_df/max_df cutoff inside each bucket so only vocabulary candidates
        // (a few million terms at most) are ever held in memory — rather than
        // materialising every distinct n-gram (hundreds of millions for a large
        // corpus) into `bucket_dfs` plus a second full copy in a `term_df` map.
        // That former double-buffer was a multi-GB spike that grew with the
        // corpus and, with little/no swap, would silently OOM-kill the build.
        let max_df_count = if params.max_df_ratio <= 0.0 {
            doc_count as u32
        } else {
            ((doc_count as f32) * params.max_df_ratio).floor().max(1.0) as u32
        };

        let vocab_counter = AtomicUsize::new(0);
        let bucket_vocabs: Vec<Vec<(u64, u32)>> = (0..bucket_count)
            .into_par_iter()
            .map(|bucket_idx| -> Result<Vec<(u64, u32)>> {
                let path = df_dir.join(format!("bucket-{:04}.bin", bucket_idx));
                if !path.exists() {
                    let done = vocab_counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 256 == 0 || done == bucket_count {
                        emit_build_progress(
                            progress,
                            "TF-IDF phase 2: building vocabulary",
                            done,
                            bucket_count,
                            0.25 + 0.20 * fraction(done, bucket_count),
                        );
                    }
                    return Ok(Vec::new());
                }
                let raw = fs::read(&path)?;
                let n = raw.len() / 12;
                let mut records: Vec<(u64, u32)> = Vec::with_capacity(n);
                for k in 0..n {
                    let base = k * 12;
                    let h = u64::from_le_bytes(raw[base..base + 8].try_into()?);
                    let df_count = u32::from_le_bytes(raw[base + 8..base + 12].try_into()?);
                    records.push((h, df_count));
                }
                drop(raw);
                records.sort_unstable_by_key(|(h, _)| *h);
                let mut local: Vec<(u64, u32)> = Vec::new();
                let mut i = 0;
                while i < records.len() {
                    let h = records[i].0;
                    let mut j = i;
                    let mut df_sum: u64 = 0;
                    while j < records.len() && records[j].0 == h {
                        df_sum += records[j].1 as u64;
                        j += 1;
                    }
                    let df = df_sum.min(u32::MAX as u64) as u32;
                    // Filter here, while this term's full DF is known, so rare
                    // and ubiquitous grams never accumulate in memory.
                    if df >= params.min_df && df <= max_df_count {
                        local.push((h, df));
                    }
                    i = j;
                }
                let done = vocab_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 256 == 0 || done == bucket_count {
                    emit_build_progress(
                        progress,
                        "TF-IDF phase 2: building vocabulary",
                        done,
                        bucket_count,
                        0.25 + 0.20 * fraction(done, bucket_count),
                    );
                }
                Ok(local)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut vocab: Vec<(u64, u32)> = bucket_vocabs.into_iter().flatten().collect();
        eprintln!("  {} terms pass filters", vocab.len());
        // Sort by descending DF, breaking ties by hash so that the max_features
        // truncation below is fully deterministic from one run to the next.
        vocab.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if params.max_features > 0 && vocab.len() > params.max_features {
            vocab.truncate(params.max_features);
        }
        vocab.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        eprintln!("  Final vocab: {}", vocab.len());

        let idf: Vec<f32> = vocab
            .iter()
            .map(|(_, df)| {
                let df = *df as f32;
                ((doc_count as f32 - df + 0.5) / (df + 0.5) + 1.0).ln()
            })
            .collect();

        let term_to_id: HashMap<u64, TermId> = vocab
            .iter()
            .enumerate()
            .map(|(i, (h, _))| (*h, i as TermId))
            .collect();

        let vocab_entries: Vec<VocabEntry> = vocab
            .iter()
            .enumerate()
            .map(|(i, (h, df))| VocabEntry {
                term_hash: *h,
                term_id: i as TermId,
                df: *df,
                postings_offset: 0,
                postings_count: 0,
                w_max: 0.0,
            })
            .collect();

        fs::write(&phase2_vocab, bincode::serialize(&vocab_entries)?)?;
        fs::write(&phase2_idf_f, bincode::serialize(&idf)?)?;
        fs::write(&phase2_termid, bincode::serialize(&term_to_id)?)?;
        // df_dir is no longer needed now that the checkpoint is safely written.
        let _ = fs::remove_dir_all(&df_dir);
        emit_build_progress(
            progress,
            "TF-IDF phase 2: building vocabulary",
            bucket_count,
            bucket_count,
            0.45,
        );

        (vocab_entries, idf, term_to_id)
    };

    let vocab_count = vocab_entries.len();

    // -----------------------------------------------------------------------
    // Phase 3 — rows + posting bucket writes (parallel)
    // The writer thread streams *intermediate* f32 row records to disk and
    // tracks per-term w_max (largest weight ever observed).
    // -----------------------------------------------------------------------
    eprintln!("\n[Phase 3] Rows + posting buckets...");
    let nthreads = rayon::current_num_threads();
    let handles_per_thread = (700usize / nthreads.max(1)).max(4).min(64);
    for t in 0..nthreads {
        fs::create_dir_all(post_dir.join(format!("t{}", t)))?;
    }

    // Intermediate row bytes are (term_id u32, weight f32) = 8 B per entry.
    const ROW_F32_ENTRY: usize = 8;
    const ROW_F32_RECORD_HEADER: usize = 8;
    let row_f32_path = temp_dir.join("rows_f32.tmp");

    let (row_tx, row_rx) = std::sync::mpsc::sync_channel::<(DocId, Vec<u8>)>(8192);
    let writer_handle = {
        let row_f32_path = row_f32_path.clone();
        let mut w_max: Vec<f32> = vec![0.0; vocab_count];
        std::thread::spawn(move || -> Result<(u64, Vec<f32>)> {
            let mut row_file = BufWriter::new(File::create(&row_f32_path)?);
            let mut bytes_written = 0u64;
            for (doc_id, bytes) in row_rx {
                // Track per-term w_max while consuming the row.
                for chunk in bytes.chunks_exact(ROW_F32_ENTRY) {
                    let tid = u32::from_le_bytes(chunk[0..4].try_into().unwrap()) as usize;
                    let w = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
                    if w > w_max[tid] {
                        w_max[tid] = w;
                    }
                }
                let byte_len = bytes.len() as u32;
                row_file.write_all(&doc_id.to_le_bytes())?;
                row_file.write_all(&byte_len.to_le_bytes())?;
                row_file.write_all(&bytes)?;
                bytes_written += (ROW_F32_RECORD_HEADER + bytes.len()) as u64;
            }
            row_file.flush()?;
            Ok((bytes_written, w_max))
        })
    };

    let counter3 = AtomicUsize::new(0usize);
    let total3 = work_units.len();
    emit_build_progress(
        progress,
        "TF-IDF phase 3: writing rows and postings",
        0,
        total3,
        0.45,
    );
    let min_n = params.min_ngram;
    let max_n = params.max_ngram;
    let analyze_opts = AnalyzeOptions {
        min_n,
        max_n,
        filter: FilterMode::CjkOnly,
        apply_low_value_filter: true,
        dedup: false,
        count_tf: true,
    };

    work_units
        .par_iter()
        .try_for_each_with(row_tx, |tx, unit| -> Result<()> {
            let t = rayon::current_thread_index().unwrap_or(0);
            let mut post_w =
                BucketWriterCache::new(post_dir.join(format!("t{}", t)), handles_per_thread);
            let mut posting_buffers: Vec<Vec<u8>> = (0..bucket_count).map(|_| Vec::new()).collect();
            let mut buffered_bytes = 0usize;
            let mut scratch = AnalyzeScratch::new();

            let mut builder = crate::parquet_metadata::global_cache().get_or_load(&unit.path)?;
            if let Some(row_group) = unit.row_group {
                builder = builder.with_row_groups(vec![row_group]);
            }
            let reader = builder.build()?;
            for batch in reader {
                let batch = batch?;
                let (pids, texts) = extract_passage_columns(&batch)?;
                for i in 0..batch.num_rows() {
                    let pid = pids.value(i);
                    let Some(doc_id) = doc_table.doc_id(pid) else {
                        continue;
                    };

                    let mut term_counts: FxHashMap<TermId, u32> = FxHashMap::default();
                    analyze_normalized(texts.value(i), &analyze_opts, &mut scratch);
                    for &(h, count) in &scratch.counts {
                        if let Some(&tid) = term_to_id.get(&h) {
                            *term_counts.entry(tid).or_insert(0) += count;
                        }
                    }
                    if term_counts.is_empty() {
                        continue;
                    }

                    let total_tf: u32 = term_counts.values().sum();
                    let mut row: Vec<(TermId, f32)> = term_counts
                        .iter()
                        .map(|(&tid, &c)| (tid, (c as f32 / total_tf as f32) * idf[tid as usize]))
                        .collect();

                    let norm = row.iter().map(|(_, w)| w * w).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for (_, w) in row.iter_mut() {
                            *w /= norm;
                        }
                    }
                    row.sort_unstable_by_key(|(tid, _)| *tid);

                    let mut row_bytes = Vec::with_capacity(row.len() * ROW_F32_ENTRY);
                    for (tid, w) in &row {
                        row_bytes.extend_from_slice(&tid.to_le_bytes());
                        row_bytes.extend_from_slice(&w.to_le_bytes());
                    }
                    tx.send((doc_id, row_bytes))
                        .map_err(|e| anyhow::anyhow!("row channel closed: {}", e))?;

                    for (tid, w) in &row {
                        let bucket = (*tid as usize) % bucket_count;
                        append_posting_record(&mut posting_buffers[bucket], *tid, doc_id, *w);
                        buffered_bytes += POSTING_BUCKET_RECORD_SIZE;
                        if buffered_bytes >= BUCKET_BUFFER_FLUSH_BYTES {
                            flush_bucket_buffers(&mut post_w, &mut posting_buffers)?;
                            buffered_bytes = 0;
                        }
                    }
                }
            }
            flush_bucket_buffers(&mut post_w, &mut posting_buffers)?;
            post_w.flush_all()?;

            let n = counter3.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 100 == 0 || n == total3 {
                eprintln!("  {}/{}", n, total3);
                emit_build_progress(
                    progress,
                    "TF-IDF phase 3: writing rows and postings",
                    n,
                    total3,
                    0.45 + 0.25 * fraction(n, total3),
                );
            }
            Ok(())
        })?;

    let (row_f32_bytes, w_max) = writer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("row writer thread panicked"))??;

    eprintln!(
        "  Intermediate f32 row records: {} bytes ({:.1} MB)",
        row_f32_bytes,
        row_f32_bytes as f64 / 1e6
    );

    // Persist w_max into vocab entries now (used both by Phase 4 quantization
    // and by the final saved index).
    for (tid, entry) in vocab_entries.iter_mut().enumerate() {
        entry.w_max = w_max[tid];
    }

    // Merge per-thread posting bucket files.
    eprintln!("  Merging posting thread buckets...");
    for bucket_idx in 0..bucket_count {
        let main = post_dir.join(format!("bucket-{:04}.bin", bucket_idx));
        let mut out = OpenOptions::new().create(true).append(true).open(&main)?;
        for t in 0..nthreads {
            let src = post_dir.join(format!("t{}/bucket-{:04}.bin", t, bucket_idx));
            match File::open(&src) {
                Ok(mut f) => {
                    std::io::copy(&mut f, &mut out)?;
                    drop(f);
                    fs::remove_file(&src)?;
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
    for t in 0..nthreads {
        let _ = fs::remove_dir(post_dir.join(format!("t{}", t)));
    }
    emit_build_progress(
        progress,
        "TF-IDF phase 3: writing rows and postings",
        total3,
        total3,
        0.70,
    );

    // -----------------------------------------------------------------------
    // Phase 4 — merge posting buckets, quantize on emit (parallel)
    // -----------------------------------------------------------------------
    eprintln!("\n[Phase 4] Merging posting buckets + quantizing...");

    let w_max_arc: Arc<Vec<f32>> = Arc::new(w_max);

    // Process posting buckets one at a time, appending each term's posting list
    // directly into the final blob. Buckets are disjoint by `term_id % bucket
    // _count`, so a term's full posting list is contained in a single bucket and
    // can be emitted in order. Reading one bucket at a time — rather than
    // collecting every bucket's quantized output into `bucket_results` and then
    // copying it all again into `postings_blob` — keeps peak memory near the
    // final blob size plus one bucket, instead of roughly twice the blob. The
    // per-bucket sort is parallelised so cores stay busy on the dominant cost.
    let mut postings_blob: Vec<u8> = Vec::new();
    emit_build_progress(
        progress,
        "TF-IDF phase 4: quantizing posting buckets",
        0,
        bucket_count,
        0.70,
    );
    for bucket_idx in 0..bucket_count {
        if bucket_idx % 256 == 0 {
            eprintln!("  bucket {}/{}", bucket_idx, bucket_count);
            emit_build_progress(
                progress,
                "TF-IDF phase 4: quantizing posting buckets",
                bucket_idx,
                bucket_count,
                0.70 + 0.15 * fraction(bucket_idx, bucket_count),
            );
        }
        let path = post_dir.join(format!("bucket-{:04}.bin", bucket_idx));
        if !path.exists() {
            continue;
        }
        let raw = fs::read(&path)?;
        let n = raw.len() / POSTING_BUCKET_RECORD_SIZE;
        let mut records: Vec<(TermId, DocId, f32)> = Vec::with_capacity(n);
        for k in 0..n {
            let base = k * POSTING_BUCKET_RECORD_SIZE;
            let tid = u32::from_le_bytes(raw[base..base + 4].try_into()?);
            let did = u32::from_le_bytes(raw[base + 4..base + 8].try_into()?);
            let w = f32::from_le_bytes(raw[base + 8..base + 12].try_into()?);
            records.push((tid, did, w));
        }
        drop(raw);
        records.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut i = 0;
        while i < records.len() {
            let tid = records[i].0;
            let wm = w_max_arc[tid as usize];
            let mut j = i;
            while j < records.len() && records[j].0 == tid {
                j += 1;
            }
            vocab_entries[tid as usize].postings_offset = postings_blob.len() as u64;
            vocab_entries[tid as usize].postings_count = (j - i) as u32;
            for k in i..j {
                let (_, did, w) = records[k];
                postings_blob.extend_from_slice(&did.to_le_bytes());
                postings_blob.push(quantize_log_u8(w, wm));
            }
            i = j;
        }
    }
    let _ = fs::remove_dir_all(&post_dir);
    emit_build_progress(
        progress,
        "TF-IDF phase 4: quantizing posting buckets",
        bucket_count,
        bucket_count,
        0.85,
    );
    eprintln!(
        "  Quantized postings blob: {} bytes ({:.1} MB)",
        postings_blob.len(),
        postings_blob.len() as f64 / 1e6
    );

    // -----------------------------------------------------------------------
    // Quantize row blob — second pass converting the intermediate f32 row
    // records into a packed 5-byte/entry quantized blob.
    // -----------------------------------------------------------------------
    eprintln!("\n[Quantize] Re-emitting row blob with u8 weights...");
    emit_build_progress(
        progress,
        "TF-IDF phase 5: quantizing row vectors",
        0,
        1,
        0.85,
    );
    let mut row_blob_q: Vec<u8> = Vec::with_capacity((row_f32_bytes as usize) * 5 / 8);
    let mut row_offsets_q: Vec<u64> = vec![u64::MAX; doc_count];
    let mut row_lengths: Vec<u32> = vec![0u32; doc_count];
    let mut row_file = File::open(&row_f32_path)?;
    let mut header = [0u8; ROW_F32_RECORD_HEADER];
    let mut row_buf = Vec::new();
    loop {
        match row_file.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let doc_id = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        let byte_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        if byte_len % ROW_F32_ENTRY != 0 {
            return Err(anyhow::anyhow!(
                "corrupt temporary row record for doc {}: {} bytes",
                doc_id,
                byte_len
            ));
        }
        row_buf.resize(byte_len, 0);
        row_file.read_exact(&mut row_buf)?;
        row_offsets_q[doc_id] = row_blob_q.len() as u64;
        row_lengths[doc_id] = (byte_len / ROW_F32_ENTRY) as u32;
        for chunk in row_buf.chunks_exact(ROW_F32_ENTRY) {
            let tid = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let w = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
            let wm = vocab_entries[tid as usize].w_max;
            row_blob_q.extend_from_slice(&tid.to_le_bytes());
            row_blob_q.push(quantize_log_u8(w, wm));
        }
    }
    drop(row_file);
    let _ = fs::remove_file(&row_f32_path);
    eprintln!(
        "  Quantized row blob: {} bytes ({:.1} MB)",
        row_blob_q.len(),
        row_blob_q.len() as f64 / 1e6
    );
    emit_build_progress(
        progress,
        "TF-IDF phase 5: quantizing row vectors",
        1,
        1,
        0.95,
    );

    // -----------------------------------------------------------------------
    // Save
    // -----------------------------------------------------------------------
    eprintln!("\n[Save] Writing index...");
    emit_build_progress(progress, "TF-IDF phase 6: writing final index", 0, 1, 0.95);
    write_index_file(
        &out_path,
        &params,
        &doc_table_fingerprint,
        &vocab_entries,
        &idf,
        doc_count,
        &row_offsets_q,
        &row_lengths,
        &row_blob_q,
        &postings_blob,
    )?;

    let _ = fs::remove_dir_all(&temp_dir);
    eprintln!("\n=== Complete ===");
    eprintln!("Output: {}", out_path.display());
    emit_build_progress(progress, "TF-IDF index complete", 1, 1, 1.0);
    Ok(())
}

fn fraction(done: usize, total: usize) -> f32 {
    if total == 0 {
        1.0
    } else {
        (done as f32 / total as f32).clamp(0.0, 1.0)
    }
}

fn emit_build_progress(
    progress: Option<&crate::phrase_index::BuildProgress<'_>>,
    label: &str,
    done: usize,
    total: usize,
    fraction: f32,
) {
    if let Some(progress) = progress {
        progress(label, done.min(total), total, fraction.clamp(0.0, 1.0));
    }
}

// ---------------------------------------------------------------------------
// Write the on-disk index from the builder's in-memory state.
// ---------------------------------------------------------------------------

fn write_index_file(
    path: &Path,
    params: &TfidfParams,
    doc_table_fingerprint: &str,
    vocab: &[VocabEntry],
    idf: &[f32],
    doc_count: usize,
    row_offsets: &[u64],
    row_lengths: &[u32],
    row_blob: &[u8],
    postings_blob: &[u8],
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("index.tmp");
    let mut f = BufWriter::new(File::create(&tmp)?);

    let vocab_count = vocab.len() as u32;
    let doc_count_u32 = doc_count as u32;
    let row_blob_len = row_blob.len() as u64;
    let postings_blob_len = postings_blob.len() as u64;

    let vocab_off = HEADER_SIZE as u64;
    let idf_off = vocab_off + vocab_count as u64 * VOCAB_ENTRY_SIZE as u64;
    let row_off_off = idf_off + vocab_count as u64 * 4;
    let row_len_off = row_off_off + doc_count as u64 * 8;
    let row_blob_off = row_len_off + doc_count as u64 * 4;
    let post_blob_off = row_blob_off + row_blob_len;
    let query_table_off = post_blob_off + postings_blob_len;
    let query_table_count = vocab_count as u64;

    let mut hdr = vec![0u8; HEADER_SIZE];
    hdr[0..8].copy_from_slice(MAGIC_TFIDF);
    hdr[8..10].copy_from_slice(&4u16.to_le_bytes());
    hdr[HDR_VOCAB_COUNT].copy_from_slice(&vocab_count.to_le_bytes());
    hdr[HDR_DOC_COUNT].copy_from_slice(&doc_count_u32.to_le_bytes());
    hdr[HDR_MIN_NGRAM].copy_from_slice(&(params.min_ngram as u16).to_le_bytes());
    hdr[HDR_MAX_NGRAM].copy_from_slice(&(params.max_ngram as u16).to_le_bytes());
    hdr[HDR_MIN_DF].copy_from_slice(&params.min_df.to_le_bytes());
    hdr[HDR_MAX_FEATURES].copy_from_slice(&(params.max_features as u32).to_le_bytes());
    hdr[HDR_MAX_DF_RATIO].copy_from_slice(&params.max_df_ratio.to_le_bytes());

    let fp = doc_table_fingerprint.as_bytes();
    let n = fp.len().min(HDR_FP.len());
    hdr[HDR_FP.start..HDR_FP.start + n].copy_from_slice(&fp[..n]);

    hdr[HDR_VOCAB_OFF].copy_from_slice(&vocab_off.to_le_bytes());
    hdr[HDR_IDF_OFF].copy_from_slice(&idf_off.to_le_bytes());
    hdr[HDR_ROW_OFFSETS].copy_from_slice(&row_off_off.to_le_bytes());
    hdr[HDR_ROW_LENGTHS].copy_from_slice(&row_len_off.to_le_bytes());
    hdr[HDR_ROW_BLOB_OFF].copy_from_slice(&row_blob_off.to_le_bytes());
    hdr[HDR_ROW_BLOB_LEN].copy_from_slice(&row_blob_len.to_le_bytes());
    hdr[HDR_POST_BLOB_OFF].copy_from_slice(&post_blob_off.to_le_bytes());
    hdr[HDR_POST_BLOB_LEN].copy_from_slice(&postings_blob_len.to_le_bytes());
    hdr[HDR_QUERY_OFF].copy_from_slice(&query_table_off.to_le_bytes());
    hdr[HDR_QUERY_COUNT].copy_from_slice(&query_table_count.to_le_bytes());
    f.write_all(&hdr)?;

    for e in vocab {
        f.write_all(&e.term_hash.to_le_bytes())?;
        f.write_all(&e.term_id.to_le_bytes())?;
        f.write_all(&e.df.to_le_bytes())?;
        f.write_all(&e.postings_offset.to_le_bytes())?;
        f.write_all(&e.postings_count.to_le_bytes())?;
        f.write_all(&e.w_max.to_le_bytes())?;
    }
    for &v in idf {
        f.write_all(&v.to_le_bytes())?;
    }
    for &o in row_offsets {
        f.write_all(&o.to_le_bytes())?;
    }
    for &l in row_lengths {
        f.write_all(&l.to_le_bytes())?;
    }
    f.write_all(row_blob)?;
    f.write_all(postings_blob)?;
    // Query table: sorted (term_hash u64, term_id u32) pairs.
    // Vocab is already sorted by term_hash from phase 2, so we write in order.
    for e in vocab {
        f.write_all(&e.term_hash.to_le_bytes())?;
        f.write_all(&e.term_id.to_le_bytes())?;
    }
    f.flush()?;
    drop(f);
    fs::rename(&tmp, path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn round_f32(value: f32, places: i32) -> f32 {
    let factor = 10f32.powi(places);
    (value * factor).round() / factor
}

fn round_f64(value: f64, places: i32) -> f64 {
    let factor = 10f64.powi(places);
    (value * factor).round() / factor
}

// idf_at is only used internally for diagnostics; keep it from being dead-stripped.
#[allow(dead_code)]
fn _idf_keep_alive(t: &TfidfIndex, tid: TermId) -> f32 {
    t.idf_at(tid)
}

#[cfg(test)]
mod word_run_tests {
    use super::long_common_substrings;
    use crate::normalize::{normalize_words, normalize_zh};

    #[test]
    fn word_layout_shares_whole_word_runs() {
        let a = normalize_words("The coolant pump bearing on unit four showed vibration.");
        let b = normalize_words("A coolant pump bearing seized; unit four was shut down.");
        assert_eq!(
            long_common_substrings(&a, &b, 6, 8),
            vec!["coolant pump bearing".to_string(), "unit four".to_string()]
        );
    }

    #[test]
    fn cjk_layout_keeps_character_fragments() {
        let a = normalize_zh("如是我聞一時佛在舍衛國");
        let b = normalize_zh("爾時如是我聞一時佛在");
        let got = long_common_substrings(&a, &b, 4, 3);
        assert_eq!(got[0], normalize_zh("如是我聞一時佛在"));
    }
}
