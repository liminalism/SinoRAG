// Streaming, memory-bounded CJK phrase index (hybrid roaring/varint).
//
// Pipeline (single canonical version):
//   Phase 1 — scan parquet, normalise text, write (gram_hash, doc_id) records
//             into bucket files (bucketed by ordered hash range).
//   Phase 2 — sort each bucket in fixed-size in-memory chunks, write sorted runs.
//   Phase 3 — k-way merge each bucket's sorted runs, dedup (gram, doc_id),
//             choose encoding per gram (delta-varint for sparse, Roaring for
//             dense grams), stream to a single postings file.
//   Phase 4 — write the final index file: header + gram entries + postings blob.
//
// On-disk format:
//   header (256 B)
//     [0..8]     magic = "SRPH3VAA"
//     [8..10]    format version u16
//     [10..12]   gram_len u16
//     [12..16]   num_grams u32
//     [16..80]   doc_table_fingerprint (null-padded utf8, max 64 B)
//     [80..144]  schema string (null-padded utf8, max 64 B)
//     [144..152] gram_table_offset u64 (= 256)
//     [152..160] postings_blob_size u64
//     [160..164] dense_gram_count u32 (informational)
//     [164..168] varint_gram_count u32 (informational)
//     [168..172] num_docs_at_build u32 (informational)
//     [172..256] reserved
//   gram entries (24 B each, sorted by gram_hash for binary search)
//     [ 0.. 8]  gram_hash u64
//     [ 8..16]  postings_offset u64 (within postings blob)
//     [16..20]  postings_len u32 (bytes)
//     [20..21]  encoding u8 (0 = delta-varint, 1 = roaring)
//     [21..24]  reserved
//   postings blob — concatenation of per-gram encoded postings.

use crate::arrow_helpers::extract_passage_columns;
use crate::datafusion_store::{sql_literal, DataFusionStore};
use crate::document_table::DocumentTable;
use crate::normalize::normalize_zh;
use crate::parquet_metadata::global_cache;
use anyhow::Result;
use memmap2::Mmap;
use roaring::RoaringBitmap;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use walkdir::WalkDir;
use xxhash_rust::xxh3::xxh3_64;

pub type BuildProgress<'a> = dyn Fn(&str, usize, usize, f32) + Send + Sync + 'a;

pub type DocId = u32;

const MAGIC: &[u8; 8] = b"SRPH3VAA";
const HEADER_SIZE: usize = 256;
const GRAM_ENTRY_SIZE: usize = 24;
const SCHEMA_LABEL: &[u8] = b"sinorag-phrase-index-range-buckets";

const ENC_VARINT: u8 = 0;
const ENC_ROARING: u8 = 1;

const HDR_GRAM_LEN: std::ops::Range<usize> = 10..12;
const HDR_NUM_GRAMS: std::ops::Range<usize> = 12..16;
const HDR_FP: std::ops::Range<usize> = 16..80;
const HDR_SCHEMA: std::ops::Range<usize> = 80..144;
const HDR_GRAM_TABLE_OFF: std::ops::Range<usize> = 144..152;
const HDR_POSTINGS_SIZE: std::ops::Range<usize> = 152..160;
const HDR_DENSE_GRAMS: std::ops::Range<usize> = 160..164;
const HDR_VARINT_GRAMS: std::ops::Range<usize> = 164..168;
const HDR_NUM_DOCS: std::ops::Range<usize> = 168..172;

/// Mmap-backed phrase index. Each gram's postings carry an encoding flag so
/// dense grams use Roaring bitmaps (fast bitwise AND) while sparse ones stay
/// on delta-varint (compact).
#[derive(Debug, Clone)]
pub struct PhraseIndex {
    schema: String,
    gram_len: usize,
    doc_table_fingerprint: String,
    num_grams: usize,
    dense_gram_count: usize,
    varint_gram_count: usize,
    num_docs_at_build: usize,
    gram_table_offset: usize,
    postings_offset: usize,
    postings_len: usize,
    mmap: Arc<Mmap>,
}

impl PhraseIndex {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            anyhow::bail!("PhraseIndex file too small");
        }
        if &mmap[0..8] != MAGIC {
            anyhow::bail!("PhraseIndex magic mismatch; rebuild required");
        }
        let version = u16::from_le_bytes([mmap[8], mmap[9]]);
        if version != 3 {
            anyhow::bail!("Unsupported PhraseIndex format; rebuild required");
        }

        let gram_len = u16::from_le_bytes(mmap[HDR_GRAM_LEN].try_into()?) as usize;
        let num_grams = u32::from_le_bytes(mmap[HDR_NUM_GRAMS].try_into()?) as usize;
        let doc_table_fingerprint = read_padded_string(&mmap[HDR_FP]);
        let schema = read_padded_string(&mmap[HDR_SCHEMA]);
        ensure_supported_schema(&schema)?;
        let gram_table_offset = u64::from_le_bytes(mmap[HDR_GRAM_TABLE_OFF].try_into()?) as usize;
        let postings_len = u64::from_le_bytes(mmap[HDR_POSTINGS_SIZE].try_into()?) as usize;
        let dense_gram_count = u32::from_le_bytes(mmap[HDR_DENSE_GRAMS].try_into()?) as usize;
        let varint_gram_count = u32::from_le_bytes(mmap[HDR_VARINT_GRAMS].try_into()?) as usize;
        let num_docs_at_build = u32::from_le_bytes(mmap[HDR_NUM_DOCS].try_into()?) as usize;

        let gram_table_size = num_grams * GRAM_ENTRY_SIZE;
        let postings_offset = gram_table_offset + gram_table_size;

        if postings_offset + postings_len > mmap.len() {
            anyhow::bail!(
                "PhraseIndex sections exceed file length \
                 (postings_off={} postings_len={} file_len={})",
                postings_offset,
                postings_len,
                mmap.len()
            );
        }

        Ok(Self {
            schema,
            gram_len,
            doc_table_fingerprint,
            num_grams,
            dense_gram_count,
            varint_gram_count,
            num_docs_at_build,
            gram_table_offset,
            postings_offset,
            postings_len,
            mmap: Arc::new(mmap),
        })
    }

    pub fn load(path: &Path) -> Result<Self> {
        Self::open(path)
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }
    pub fn gram_len(&self) -> usize {
        self.gram_len
    }
    pub fn doc_table_fingerprint(&self) -> &str {
        &self.doc_table_fingerprint
    }
    pub fn num_grams(&self) -> usize {
        self.num_grams
    }

    /// Read a single gram entry from the mmap.
    fn gram_entry(&self, idx: usize) -> (u64, u64, u32, u8) {
        let off = self.gram_table_offset + idx * GRAM_ENTRY_SIZE;
        let s = &self.mmap[off..off + GRAM_ENTRY_SIZE];
        let hash = u64::from_le_bytes(s[0..8].try_into().unwrap());
        let p_off = u64::from_le_bytes(s[8..16].try_into().unwrap());
        let p_len = u32::from_le_bytes(s[16..20].try_into().unwrap());
        let enc = s[20];
        (hash, p_off, p_len, enc)
    }

    /// Binary search the sorted gram-entry table for `hash`.
    fn find_gram(&self, hash: u64) -> Option<(u64, u32, u8)> {
        let (mut lo, mut hi) = (0usize, self.num_grams);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (h, p_off, p_len, enc) = self.gram_entry(mid);
            match h.cmp(&hash) {
                std::cmp::Ordering::Equal => return Some((p_off, p_len, enc)),
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }

    pub fn candidate_ids(&self, phrase: &str, limit: usize) -> Vec<DocId> {
        let normalized = normalize_zh(phrase);
        self.candidate_ids_for_normalized(&normalized, limit)
    }

    pub fn candidate_ids_for_normalized(&self, normalized: &str, limit: usize) -> Vec<DocId> {
        self.candidate_ids_for_normalized_streaming(normalized)
            .map(|r| {
                let mut ids = r.doc_ids;
                ids.truncate(limit.max(1));
                ids
            })
            .unwrap_or_default()
    }

    /// Candidates for a phrase in both stored layouts (see
    /// [`crate::normalize::PhraseForms`]). Grams ignore spaces, so the two
    /// forms differ only by case; a pure-CJK phrase does one lookup exactly
    /// as before. Results are sorted, deduplicated doc ids; candidates still
    /// need verifying with `PhraseForms::matches`.
    pub fn candidate_ids_for_forms_streaming(
        &self,
        forms: &crate::normalize::PhraseForms,
    ) -> Option<PhraseCandidateResult> {
        let first = self.candidate_ids_for_normalized_streaming(&forms.zh)?;
        let words_compact: String = forms.words.chars().filter(|c| *c != ' ').collect();
        if forms.is_single() || words_compact == forms.zh {
            return Some(first);
        }
        let second = self.candidate_ids_for_normalized_streaming(&words_compact)?;
        let (a, b) = (&first.doc_ids, &second.doc_ids);
        let mut merged = Vec::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < a.len() || j < b.len() {
            let next = match (a.get(i), b.get(j)) {
                (Some(&x), Some(&y)) if x == y => {
                    i += 1;
                    j += 1;
                    x
                }
                (Some(&x), Some(&y)) if x < y => {
                    i += 1;
                    x
                }
                (Some(_), Some(&y)) => {
                    j += 1;
                    y
                }
                (Some(&x), None) => {
                    i += 1;
                    x
                }
                (None, Some(&y)) => {
                    j += 1;
                    y
                }
                (None, None) => break,
            };
            merged.push(next);
        }
        let mut stats = first.stats;
        stats.final_candidates = merged.len();
        Some(PhraseCandidateResult {
            doc_ids: merged,
            stats,
        })
    }

    /// Hybrid intersection: bitwise-AND across all Roaring postings, then
    /// stream-intersect each varint posting against the result.
    pub fn candidate_ids_for_normalized_streaming(
        &self,
        normalized: &str,
    ) -> Option<PhraseCandidateResult> {
        let grams = phrase_index_grams(normalized, self.gram_len);
        if grams.is_empty() {
            return Some(PhraseCandidateResult {
                doc_ids: Vec::new(),
                stats: PhraseCandidateStats::empty(0),
            });
        }

        let gram_count = grams.len();
        let mut postings: Vec<PostingSlice<'_>> = Vec::with_capacity(gram_count);
        let mut missing = 0usize;

        for gram in &grams {
            let hash = xxh3_64(gram.as_bytes());
            let Some((p_off, p_len, enc)) = self.find_gram(hash) else {
                missing += 1;
                continue;
            };
            let abs = self.postings_offset + p_off as usize;
            let slice = &self.mmap[abs..abs + p_len as usize];
            postings.push(PostingSlice {
                bytes: slice,
                encoded_len: p_len as usize,
                encoding: enc,
            });
        }

        if missing > 0 {
            return Some(PhraseCandidateResult {
                doc_ids: Vec::new(),
                stats: PhraseCandidateStats {
                    gram_count,
                    missing_gram_count: missing,
                    postings_encoded_bytes: postings.iter().map(|p| p.encoded_len).collect(),
                    smallest_postings_bytes: 0,
                    initial_candidates: 0,
                    final_candidates: 0,
                    decoded_full_lists: 0,
                    streamed_lists: 0,
                    roaring_lists: postings
                        .iter()
                        .filter(|p| p.encoding == ENC_ROARING)
                        .count(),
                    varint_lists: postings.iter().filter(|p| p.encoding == ENC_VARINT).count(),
                },
            });
        }

        let postings_encoded_bytes: Vec<usize> = postings.iter().map(|p| p.encoded_len).collect();
        let roaring_lists = postings
            .iter()
            .filter(|p| p.encoding == ENC_ROARING)
            .count();
        let varint_lists_total = postings.len() - roaring_lists;

        // Compute Roaring AND across all dense postings.
        let mut roaring_acc: Option<RoaringBitmap> = None;
        let mut varint_slices: Vec<&[u8]> = Vec::new();
        for p in &postings {
            match p.encoding {
                ENC_ROARING => {
                    let bm = match RoaringBitmap::deserialize_from(p.bytes) {
                        Ok(b) => b,
                        Err(_) => return None,
                    };
                    roaring_acc = Some(match roaring_acc.take() {
                        None => bm,
                        Some(mut acc) => {
                            acc &= &bm;
                            acc
                        }
                    });
                    if let Some(ref acc) = roaring_acc {
                        if acc.is_empty() {
                            break;
                        }
                    }
                }
                _ => varint_slices.push(p.bytes),
            }
        }

        // Seed the candidate set: prefer the Roaring AND result (often the
        // smaller selectivity); otherwise decode the smallest varint list.
        let (mut candidates, decoded_full, mut streamed) = if let Some(rb) = roaring_acc {
            let v: Vec<DocId> = rb.iter().collect();
            (v, 1usize, 0usize)
        } else {
            varint_slices.sort_by_key(|b| b.len());
            if varint_slices.is_empty() {
                return Some(PhraseCandidateResult {
                    doc_ids: Vec::new(),
                    stats: PhraseCandidateStats::empty(gram_count),
                });
            }
            let first = varint_slices.remove(0);
            let v: Vec<DocId> = DeltaDocIdIter::new(first).collect();
            (v, 1usize, 0usize)
        };
        let initial = candidates.len();
        let smallest_bytes = postings_encoded_bytes.iter().copied().min().unwrap_or(0);

        for vbytes in &varint_slices {
            if candidates.is_empty() {
                break;
            }
            intersect_candidates_with_stream(&mut candidates, DeltaDocIdIter::new(vbytes));
            streamed += 1;
        }

        let final_count = candidates.len();

        Some(PhraseCandidateResult {
            doc_ids: candidates,
            stats: PhraseCandidateStats {
                gram_count,
                missing_gram_count: 0,
                postings_encoded_bytes,
                smallest_postings_bytes: smallest_bytes,
                initial_candidates: initial,
                final_candidates: final_count,
                decoded_full_lists: decoded_full,
                streamed_lists: streamed,
                roaring_lists,
                varint_lists: varint_lists_total,
            },
        })
    }

    pub fn info_payload(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": self.schema,
            "gram_len": self.gram_len,
            "doc_table_fingerprint": self.doc_table_fingerprint,
            "num_grams": self.num_grams,
            "dense_gram_count": self.dense_gram_count,
            "varint_gram_count": self.varint_gram_count,
            "num_docs_at_build": self.num_docs_at_build,
            "postings_bytes": self.postings_len,
        })
    }

    /// Read just the 256-byte header.
    pub fn header_info(path: &Path) -> Result<serde_json::Value> {
        let mut file = File::open(path)?;
        let mut hdr = [0u8; HEADER_SIZE];
        file.read_exact(&mut hdr)?;
        if &hdr[0..8] != MAGIC {
            anyhow::bail!("PhraseIndex magic mismatch; rebuild required");
        }
        let version = u16::from_le_bytes([hdr[8], hdr[9]]);
        if version != 3 {
            anyhow::bail!("Unsupported PhraseIndex format; rebuild required");
        }
        let gram_len = u16::from_le_bytes(hdr[HDR_GRAM_LEN].try_into()?) as usize;
        let num_grams = u32::from_le_bytes(hdr[HDR_NUM_GRAMS].try_into()?) as usize;
        let doc_table_fingerprint = read_padded_string(&hdr[HDR_FP]);
        let schema = read_padded_string(&hdr[HDR_SCHEMA]);
        ensure_supported_schema(&schema)?;
        let postings_bytes = u64::from_le_bytes(hdr[HDR_POSTINGS_SIZE].try_into()?);
        let dense_gram_count = u32::from_le_bytes(hdr[HDR_DENSE_GRAMS].try_into()?);
        let varint_gram_count = u32::from_le_bytes(hdr[HDR_VARINT_GRAMS].try_into()?);
        let num_docs_at_build = u32::from_le_bytes(hdr[HDR_NUM_DOCS].try_into()?);
        let file_bytes = std::fs::metadata(path)?.len();
        Ok(serde_json::json!({
            "schema": schema,
            "gram_len": gram_len,
            "doc_table_fingerprint": doc_table_fingerprint,
            "num_grams": num_grams,
            "dense_gram_count": dense_gram_count,
            "varint_gram_count": varint_gram_count,
            "num_docs_at_build": num_docs_at_build,
            "postings_bytes": postings_bytes,
            "file_bytes": file_bytes,
            "version": version,
        }))
    }
}

// ---------------------------------------------------------------------------
// Tokenisation
// ---------------------------------------------------------------------------

fn phrase_index_grams(text: &str, gram_len: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.len() < gram_len {
        return Vec::new();
    }
    chars
        .windows(gram_len)
        .map(|w| w.iter().collect())
        .collect()
}

#[cfg(test)]
fn intersect_sorted(a: &[DocId], b: &[DocId]) -> Vec<DocId> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    out
}

/// Filter `candidates` in-place, retaining only doc_ids that also appear in
/// the streaming `postings` iterator. Both must be sorted ascending.
fn intersect_candidates_with_stream(candidates: &mut Vec<DocId>, postings: DeltaDocIdIter<'_>) {
    if candidates.is_empty() {
        return;
    }
    let mut out = 0usize;
    let mut stream = postings.peekable();
    let n = candidates.len();
    for i in 0..n {
        let candidate = candidates[i];
        while let Some(&doc_id) = stream.peek() {
            if doc_id < candidate {
                stream.next();
            } else {
                break;
            }
        }
        if matches!(stream.peek(), Some(&doc_id) if doc_id == candidate) {
            candidates[out] = candidate;
            out += 1;
        }
    }
    candidates.truncate(out);
}

#[derive(Debug, Clone)]
struct PostingSlice<'a> {
    bytes: &'a [u8],
    encoded_len: usize,
    encoding: u8,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PhraseCandidateStats {
    pub gram_count: usize,
    pub missing_gram_count: usize,
    pub postings_encoded_bytes: Vec<usize>,
    pub smallest_postings_bytes: usize,
    pub initial_candidates: usize,
    pub final_candidates: usize,
    pub decoded_full_lists: usize,
    pub streamed_lists: usize,
    pub roaring_lists: usize,
    pub varint_lists: usize,
}

impl PhraseCandidateStats {
    fn empty(gram_count: usize) -> Self {
        Self {
            gram_count,
            missing_gram_count: 0,
            postings_encoded_bytes: Vec::new(),
            smallest_postings_bytes: 0,
            initial_candidates: 0,
            final_candidates: 0,
            decoded_full_lists: 0,
            streamed_lists: 0,
            roaring_lists: 0,
            varint_lists: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhraseCandidateResult {
    pub doc_ids: Vec<DocId>,
    pub stats: PhraseCandidateStats,
}

// ---------------------------------------------------------------------------
// Varint codec — zig-zag delta encoding of sorted DocId lists
// ---------------------------------------------------------------------------

fn encode_sorted_docids_delta_varint(doc_ids: &[DocId], output: &mut Vec<u8>) {
    if doc_ids.is_empty() {
        return;
    }
    let mut prev = doc_ids[0] as i64;
    encode_varint(prev, output);
    for &doc_id in doc_ids.iter().skip(1) {
        let delta = doc_id as i64 - prev;
        encode_varint(delta, output);
        prev = doc_id as i64;
    }
}

fn encode_varint(mut value: i64, output: &mut Vec<u8>) {
    let mut zz: u64 = if value < 0 {
        ((!value as u64) << 1) | 1
    } else {
        (value as u64) << 1
    };
    let _ = &mut value;
    while zz >= 0x80 {
        output.push((zz as u8 & 0x7F) | 0x80);
        zz >>= 7;
    }
    output.push(zz as u8);
}

#[cfg(test)]
fn decode_delta_varint_docids(data: &[u8]) -> Result<Vec<DocId>> {
    let mut doc_ids = Vec::new();
    let mut iter = DeltaDocIdIter::new(data);
    while let Some(doc_id) = iter.next() {
        doc_ids.push(doc_id);
    }
    Ok(doc_ids)
}

#[derive(Debug, Clone)]
pub struct DeltaDocIdIter<'a> {
    bytes: &'a [u8],
    pos: usize,
    current: i64,
    done: bool,
}

impl<'a> DeltaDocIdIter<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            current: 0,
            done: false,
        }
    }

    fn next_internal(&mut self) -> Option<DocId> {
        if self.done || self.pos >= self.bytes.len() {
            return None;
        }
        let (zz, consumed) = decode_varint(&self.bytes[self.pos..]).ok()?;
        self.pos += consumed;
        let value = if zz & 1 == 0 {
            (zz >> 1) as i64
        } else {
            -((zz >> 1) as i64) - 1
        };
        self.current += value;
        Some(self.current as DocId)
    }
}

impl<'a> Iterator for DeltaDocIdIter<'a> {
    type Item = DocId;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_internal()
    }
}

fn decode_varint(data: &[u8]) -> Result<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    let mut pos = 0usize;
    for byte in data.iter() {
        pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
        if shift >= 64 {
            anyhow::bail!("varint overflow");
        }
    }
    anyhow::bail!("truncated varint")
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

struct BucketWriterCache {
    temp_dir: PathBuf,
    max_open: usize,
    writers: HashMap<usize, BufWriter<File>>,
    lru: VecDeque<usize>,
}

impl BucketWriterCache {
    fn new(temp_dir: PathBuf, max_open: usize) -> Self {
        Self {
            temp_dir,
            max_open: max_open.max(1),
            writers: HashMap::new(),
            lru: VecDeque::new(),
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

    fn write_bytes(&mut self, bucket: usize, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        if !self.writers.contains_key(&bucket) {
            self.evict_if_needed()?;
            let path = self.bucket_path(bucket);
            let file = OpenOptions::new().create(true).append(true).open(path)?;
            self.writers.insert(bucket, BufWriter::new(file));
        }
        self.touch(bucket);
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

fn append_bucket_record(buf: &mut Vec<u8>, gram_hash: u64, doc_id: u32) {
    buf.extend_from_slice(&gram_hash.to_le_bytes());
    buf.extend_from_slice(&doc_id.to_le_bytes());
}

fn hash_bucket(hash: u64, bucket_count: usize) -> usize {
    ((hash as u128 * bucket_count as u128) >> 64) as usize
}

fn flush_bucket_buffers(writers: &mut BucketWriterCache, buffers: &mut [Vec<u8>]) -> Result<()> {
    for (bucket, buf) in buffers.iter_mut().enumerate() {
        if !buf.is_empty() {
            writers.write_bytes(bucket, buf)?;
            buf.clear();
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct BucketRecord {
    gram_hash: u64,
    doc_id: DocId,
}

struct SortedChunkReader {
    file: File,
    buf: [u8; 12],
    exhausted: bool,
}

impl SortedChunkReader {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            file: File::open(path)?,
            buf: [0u8; 12],
            exhausted: false,
        })
    }

    fn next(&mut self) -> Result<Option<BucketRecord>> {
        if self.exhausted {
            return Ok(None);
        }
        match self.file.read_exact(&mut self.buf) {
            Ok(()) => {
                let gram_hash = u64::from_le_bytes(self.buf[0..8].try_into().unwrap());
                let doc_id = u32::from_le_bytes(self.buf[8..12].try_into().unwrap());
                Ok(Some(BucketRecord { gram_hash, doc_id }))
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.exhausted = true;
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
struct HeapItem {
    record: BucketRecord,
    reader_idx: usize,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .record
            .gram_hash
            .cmp(&self.record.gram_hash)
            .then_with(|| other.record.doc_id.cmp(&self.record.doc_id))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

const SORT_CHUNK_RECORDS: usize = 1_000_000;

pub fn build(
    parquet_path: PathBuf,
    doc_table_path: PathBuf,
    out_path: PathBuf,
    gram_len: usize,
    bucket_count: usize,
    temp_dir: Option<PathBuf>,
) -> Result<()> {
    let doc_table = DocumentTable::load(&doc_table_path)?;
    build_from_table(
        parquet_path,
        doc_table,
        out_path,
        gram_len,
        bucket_count,
        temp_dir,
    )
}

pub(crate) fn build_from_table(
    parquet_path: PathBuf,
    doc_table: DocumentTable,
    out_path: PathBuf,
    gram_len: usize,
    bucket_count: usize,
    temp_dir: Option<PathBuf>,
) -> Result<()> {
    build_from_table_with_progress(
        parquet_path,
        doc_table,
        out_path,
        gram_len,
        bucket_count,
        temp_dir,
        None,
    )
}

pub(crate) fn build_from_table_with_progress(
    parquet_path: PathBuf,
    doc_table: DocumentTable,
    out_path: PathBuf,
    gram_len: usize,
    bucket_count: usize,
    temp_dir: Option<PathBuf>,
    progress: Option<&BuildProgress<'_>>,
) -> Result<()> {
    if bucket_count == 0 {
        anyhow::bail!("bucket_count must be greater than zero");
    }
    let temp_dir = temp_dir.unwrap_or_else(|| {
        let mut p = out_path.as_os_str().to_os_string();
        p.push(".work");
        PathBuf::from(p)
    });

    eprintln!("=== PhraseIndex builder (hybrid roaring/varint) ===");
    eprintln!("Parquet : {}", parquet_path.display());
    eprintln!("Output  : {}", out_path.display());
    eprintln!("Gram len: {}", gram_len);
    eprintln!("Buckets : {}", bucket_count);
    eprintln!("Temp dir: {}", temp_dir.display());

    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)?;
    }
    fs::create_dir_all(&temp_dir)?;

    let doc_table_fingerprint = doc_table.source_fingerprint.clone();
    let num_docs = doc_table.passage_ids.len();
    eprintln!("Loaded {} passages from doc_table", num_docs);

    // Density threshold: above which a gram's postings switch to Roaring.
    let density_threshold = (num_docs as u64 / 64).max(1) as u32;
    eprintln!(
        "Density threshold: df > {} → Roaring encoding",
        density_threshold
    );

    let files = parquet_files(&parquet_path)?;
    eprintln!("Found {} parquet file(s)", files.len());

    // -----------------------------------------------------------------------
    // Phase 1 — bucket writes.
    // -----------------------------------------------------------------------
    const REPORT_INTERVAL: Duration = Duration::from_secs(20);
    eprintln!("\n[Phase 1] Bucket writes...");
    {
        const MAX_OPEN: usize = 64;
        let mut writers = BucketWriterCache::new(temp_dir.clone(), MAX_OPEN);
        let mut bucket_buffers: Vec<Vec<u8>> = (0..bucket_count).map(|_| Vec::new()).collect();
        let mut total_records: u64 = 0;
        let mut processed = 0usize;
        let phase1_start = Instant::now();
        let mut last_report = Instant::now();
        let total_files = files.len();
        emit_build_progress(
            progress,
            "Phrase index phase 1: scanning parquet files",
            0,
            total_files,
            0.0,
        );

        let analyze_opts = crate::text_analyzer::AnalyzeOptions {
            min_n: gram_len,
            max_n: gram_len,
            filter: crate::text_analyzer::FilterMode::WhitespaceOnly,
            apply_low_value_filter: false,
            dedup: true,
            count_tf: false,
        };
        let mut scratch = crate::text_analyzer::AnalyzeScratch::new();

        for file_path in &files {
            processed += 1;
            let builder = global_cache().get_or_load(file_path)?;
            let reader = builder.build()?;
            for batch in reader {
                let batch = batch?;
                let (passage_ids, text_arr) = extract_passage_columns(&batch)?;
                for i in 0..batch.num_rows() {
                    let pid = passage_ids.value(i);
                    let text = text_arr.value(i);
                    let Some(doc_id) = doc_table.doc_id(pid) else {
                        continue;
                    };

                    crate::text_analyzer::analyze_normalized(text, &analyze_opts, &mut scratch);
                    for &hash in &scratch.unique {
                        let bucket = hash_bucket(hash, bucket_count);
                        append_bucket_record(&mut bucket_buffers[bucket], hash, doc_id);
                        total_records += 1;
                    }
                }
            }
            flush_bucket_buffers(&mut writers, &mut bucket_buffers)?;
            if last_report.elapsed() >= REPORT_INTERVAL {
                let elapsed = phase1_start.elapsed().as_secs();
                let eta = if processed > 0 {
                    elapsed * (total_files - processed) as u64 / processed as u64
                } else {
                    0
                };
                eprintln!(
                    "  [Phase 1] {}/{} files  |  {} records  |  {}s elapsed  |  ~{}s remaining",
                    processed, total_files, total_records, elapsed, eta
                );
                emit_build_progress(
                    progress,
                    "Phrase index phase 1: scanning parquet files",
                    processed,
                    total_files,
                    0.35 * fraction(processed, total_files),
                );
                last_report = Instant::now();
            }
        }
        writers.flush_all()?;
        emit_build_progress(
            progress,
            "Phrase index phase 1: scanning parquet files",
            total_files,
            total_files,
            0.35,
        );
        eprintln!(
            "  [Phase 1] done — {} files, {} records, {}s",
            total_files,
            total_records,
            phase1_start.elapsed().as_secs()
        );
    }

    // -----------------------------------------------------------------------
    // Phase 2 — sort each bucket in 1M-record chunks.
    // -----------------------------------------------------------------------
    eprintln!("\n[Phase 2] Sorting chunks per bucket...");
    let mut all_sorted_chunks: Vec<Vec<PathBuf>> = Vec::with_capacity(bucket_count);
    {
        let phase2_start = Instant::now();
        let mut last_report = Instant::now();
        emit_build_progress(
            progress,
            "Phrase index phase 2: sorting buckets",
            0,
            bucket_count,
            0.35,
        );
        for bucket_idx in 0..bucket_count {
            let bucket_path = temp_dir.join(format!("bucket-{:04}.bin", bucket_idx));
            if !bucket_path.exists() {
                all_sorted_chunks.push(Vec::new());
                continue;
            }
            if last_report.elapsed() >= REPORT_INTERVAL {
                let elapsed = phase2_start.elapsed().as_secs();
                let eta = if bucket_idx > 0 {
                    elapsed * (bucket_count - bucket_idx) as u64 / bucket_idx as u64
                } else {
                    0
                };
                eprintln!(
                    "  [Phase 2] {}/{} buckets  |  {}s elapsed  |  ~{}s remaining",
                    bucket_idx, bucket_count, elapsed, eta
                );
                emit_build_progress(
                    progress,
                    "Phrase index phase 2: sorting buckets",
                    bucket_idx,
                    bucket_count,
                    0.35 + 0.30 * fraction(bucket_idx, bucket_count),
                );
                last_report = Instant::now();
            }

            let file_size = fs::metadata(&bucket_path)?.len() as usize;
            let record_count = file_size / 12;
            let mut sorted_chunks = Vec::new();
            let mut bucket_file = File::open(&bucket_path)?;
            let mut chunk_idx = 0usize;
            let mut remaining = record_count;

            while remaining > 0 {
                let to_read = SORT_CHUNK_RECORDS.min(remaining);
                let mut buf = vec![0u8; to_read * 12];
                bucket_file.read_exact(&mut buf)?;

                let mut records: Vec<BucketRecord> = Vec::with_capacity(to_read);
                for chunk in buf.chunks_exact(12) {
                    let gram_hash = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
                    let doc_id = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
                    records.push(BucketRecord { gram_hash, doc_id });
                }
                records.sort_unstable_by_key(|r| (r.gram_hash, r.doc_id));

                let sorted_path =
                    temp_dir.join(format!("sorted-{:04}-{:04}.bin", bucket_idx, chunk_idx));
                let mut sorted_file = BufWriter::new(File::create(&sorted_path)?);
                for r in &records {
                    sorted_file.write_all(&r.gram_hash.to_le_bytes())?;
                    sorted_file.write_all(&r.doc_id.to_le_bytes())?;
                }
                sorted_file.flush()?;
                drop(sorted_file);

                sorted_chunks.push(sorted_path);
                chunk_idx += 1;
                remaining -= to_read;
            }
            drop(bucket_file);
            fs::remove_file(&bucket_path)?;
            all_sorted_chunks.push(sorted_chunks);
        } // end bucket loop
        emit_build_progress(
            progress,
            "Phrase index phase 2: sorting buckets",
            bucket_count,
            bucket_count,
            0.65,
        );
        eprintln!("  [Phase 2] done — {}s", phase2_start.elapsed().as_secs());
    } // end Phase 2 block

    // -----------------------------------------------------------------------
    // Phase 3 — k-way merge each bucket; pick encoding per gram.
    // -----------------------------------------------------------------------
    eprintln!("\n[Phase 3] K-way merge + encoding selection...");
    let entries_path = temp_dir.join("gram_entries.tmp");
    let postings_path = temp_dir.join("postings.tmp");
    let mut entries_writer = BufWriter::new(File::create(&entries_path)?);
    let mut postings_writer = BufWriter::new(File::create(&postings_path)?);
    let mut postings_offset: u64 = 0;
    let mut gram_count: u64 = 0;
    let mut dense_gram_count: u64 = 0;
    let mut varint_gram_count: u64 = 0;

    let phase3_start = Instant::now();
    let mut last_report = Instant::now();
    emit_build_progress(
        progress,
        "Phrase index phase 3: merging buckets",
        0,
        bucket_count,
        0.65,
    );
    for (bucket_idx, sorted_chunks) in all_sorted_chunks.iter().enumerate() {
        if sorted_chunks.is_empty() {
            continue;
        }
        if last_report.elapsed() >= REPORT_INTERVAL {
            let elapsed = phase3_start.elapsed().as_secs();
            let eta = if bucket_idx > 0 {
                elapsed * (bucket_count - bucket_idx) as u64 / bucket_idx as u64
            } else {
                0
            };
            eprintln!(
                "  [Phase 3] {}/{} buckets  |  {} grams  |  {}s elapsed  |  ~{}s remaining",
                bucket_idx, bucket_count, gram_count, elapsed, eta
            );
            emit_build_progress(
                progress,
                "Phrase index phase 3: merging buckets",
                bucket_idx,
                bucket_count,
                0.65 + 0.30 * fraction(bucket_idx, bucket_count),
            );
            last_report = Instant::now();
        }

        let mut readers: Vec<SortedChunkReader> = sorted_chunks
            .iter()
            .filter_map(|p| SortedChunkReader::open(p).ok())
            .collect();
        if readers.is_empty() {
            continue;
        }

        let mut heap = BinaryHeap::new();
        for (idx, reader) in readers.iter_mut().enumerate() {
            if let Some(record) = reader.next()? {
                heap.push(HeapItem {
                    record,
                    reader_idx: idx,
                });
            }
        }

        let mut current_hash: Option<u64> = None;
        let mut current_docs: Vec<DocId> = Vec::new();
        let mut last_doc: Option<DocId> = None;

        while let Some(item) = heap.pop() {
            let r = item.record;
            if current_hash != Some(r.gram_hash) {
                emit_gram(
                    current_hash,
                    &mut current_docs,
                    &mut entries_writer,
                    &mut postings_writer,
                    &mut postings_offset,
                    &mut gram_count,
                    &mut dense_gram_count,
                    &mut varint_gram_count,
                    density_threshold,
                )?;
                current_hash = Some(r.gram_hash);
                current_docs.clear();
                last_doc = None;
            }
            if last_doc != Some(r.doc_id) {
                current_docs.push(r.doc_id);
                last_doc = Some(r.doc_id);
            }
            if let Some(next) = readers[item.reader_idx].next()? {
                heap.push(HeapItem {
                    record: next,
                    reader_idx: item.reader_idx,
                });
            }
        }
        emit_gram(
            current_hash,
            &mut current_docs,
            &mut entries_writer,
            &mut postings_writer,
            &mut postings_offset,
            &mut gram_count,
            &mut dense_gram_count,
            &mut varint_gram_count,
            density_threshold,
        )?;

        for p in sorted_chunks {
            let _ = fs::remove_file(p);
        }
    }
    entries_writer.flush()?;
    postings_writer.flush()?;
    drop(entries_writer);
    drop(postings_writer);
    emit_build_progress(
        progress,
        "Phrase index phase 3: merging buckets",
        bucket_count,
        bucket_count,
        0.95,
    );
    eprintln!(
        "  [Phase 3] done — {} grams (roaring: {}, varint: {}), {}s",
        gram_count,
        dense_gram_count,
        varint_gram_count,
        phase3_start.elapsed().as_secs()
    );

    // -----------------------------------------------------------------------
    // Phase 4 — write final index file (header + entries + postings).
    // -----------------------------------------------------------------------
    eprintln!("\n[Phase 4] Writing final index...");
    emit_build_progress(
        progress,
        "Phrase index phase 4: writing final index",
        0,
        1,
        0.95,
    );
    write_final_index(
        &out_path,
        &entries_path,
        &postings_path,
        gram_len,
        &doc_table_fingerprint,
        gram_count,
        dense_gram_count,
        varint_gram_count,
        num_docs as u64,
    )?;

    let _ = fs::remove_file(&entries_path);
    let _ = fs::remove_file(&postings_path);
    let _ = fs::remove_dir(&temp_dir);

    eprintln!("\n=== Complete ===");
    eprintln!("Output: {}", out_path.display());
    emit_build_progress(progress, "Phrase index complete", 1, 1, 1.0);
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
    progress: Option<&BuildProgress<'_>>,
    label: &str,
    done: usize,
    total: usize,
    fraction: f32,
) {
    if let Some(progress) = progress {
        progress(label, done.min(total), total, fraction.clamp(0.0, 1.0));
    }
}

fn emit_gram(
    current_hash: Option<u64>,
    current_docs: &mut Vec<DocId>,
    entries_writer: &mut BufWriter<File>,
    postings_writer: &mut BufWriter<File>,
    postings_offset: &mut u64,
    gram_count: &mut u64,
    dense_gram_count: &mut u64,
    varint_gram_count: &mut u64,
    density_threshold: u32,
) -> Result<()> {
    let Some(hash) = current_hash else {
        return Ok(());
    };
    if current_docs.is_empty() {
        return Ok(());
    }
    current_docs.sort_unstable();
    current_docs.dedup();

    let df = current_docs.len() as u32;
    let (encoded, encoding): (Vec<u8>, u8) = if df > density_threshold {
        let mut bm = RoaringBitmap::new();
        for &d in current_docs.iter() {
            bm.insert(d);
        }
        let mut buf: Vec<u8> = Vec::with_capacity(bm.serialized_size());
        bm.serialize_into(&mut buf)?;
        *dense_gram_count += 1;
        (buf, ENC_ROARING)
    } else {
        let mut buf = Vec::new();
        encode_sorted_docids_delta_varint(current_docs, &mut buf);
        *varint_gram_count += 1;
        (buf, ENC_VARINT)
    };

    let before = *postings_offset;
    postings_writer.write_all(&encoded)?;
    *postings_offset += encoded.len() as u64;

    entries_writer.write_all(&hash.to_le_bytes())?;
    entries_writer.write_all(&before.to_le_bytes())?;
    entries_writer.write_all(&(encoded.len() as u32).to_le_bytes())?;
    entries_writer.write_all(&[encoding, 0u8, 0u8, 0u8])?; // encoding + 3-byte pad
    *gram_count += 1;
    Ok(())
}

fn write_final_index(
    out_path: &Path,
    entries_path: &Path,
    postings_path: &Path,
    gram_len: usize,
    doc_table_fingerprint: &str,
    gram_count: u64,
    dense_gram_count: u64,
    varint_gram_count: u64,
    num_docs: u64,
) -> Result<()> {
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let postings_size = fs::metadata(postings_path)?.len();
    let tmp = out_path.with_extension("index.tmp");
    let mut out = BufWriter::new(File::create(&tmp)?);

    let mut hdr = vec![0u8; HEADER_SIZE];
    hdr[0..8].copy_from_slice(MAGIC);
    hdr[8..10].copy_from_slice(&3u16.to_le_bytes());
    hdr[HDR_GRAM_LEN].copy_from_slice(&(gram_len as u16).to_le_bytes());
    hdr[HDR_NUM_GRAMS].copy_from_slice(&(gram_count as u32).to_le_bytes());
    write_padded_string(&mut hdr[HDR_FP], doc_table_fingerprint);
    write_padded_string(&mut hdr[HDR_SCHEMA], schema_label());
    hdr[HDR_GRAM_TABLE_OFF].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    hdr[HDR_POSTINGS_SIZE].copy_from_slice(&postings_size.to_le_bytes());
    hdr[HDR_DENSE_GRAMS].copy_from_slice(&(dense_gram_count as u32).to_le_bytes());
    hdr[HDR_VARINT_GRAMS].copy_from_slice(&(varint_gram_count as u32).to_le_bytes());
    hdr[HDR_NUM_DOCS].copy_from_slice(&(num_docs as u32).to_le_bytes());

    out.write_all(&hdr)?;
    let mut entries = File::open(entries_path)?;
    std::io::copy(&mut entries, &mut out)?;
    let mut postings = File::open(postings_path)?;
    std::io::copy(&mut postings, &mut out)?;
    out.flush()?;
    drop(out);
    fs::rename(&tmp, out_path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Public utility helpers
// ---------------------------------------------------------------------------

pub fn parquet_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !path.is_dir() {
        anyhow::bail!("Parquet path not found: {}", path.display());
    }
    let mut files = Vec::new();
    for entry in WalkDir::new(path).into_iter().filter_map(Result::ok) {
        if entry.file_type().is_file()
            && entry.path().extension().and_then(|v| v.to_str()) == Some("parquet")
        {
            files.push(entry.into_path());
        }
    }
    files.sort();
    Ok(files)
}

pub fn ids_to_sql_list(ids: &[String]) -> String {
    ids.iter()
        .map(|id| sql_literal(id))
        .collect::<Vec<_>>()
        .join(", ")
}

pub async fn load_passage_texts_from_store(
    store: &DataFusionStore,
    limit: Option<usize>,
) -> Result<Vec<(String, String)>> {
    store.passage_texts(limit).await
}

fn write_padded_string(field: &mut [u8], value: &str) {
    let bytes = value.as_bytes();
    let n = bytes.len().min(field.len());
    field[..n].copy_from_slice(&bytes[..n]);
    for b in &mut field[n..] {
        *b = 0;
    }
}

fn read_padded_string(field: &[u8]) -> String {
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    String::from_utf8_lossy(&field[..end]).to_string()
}

fn schema_label() -> &'static str {
    std::str::from_utf8(SCHEMA_LABEL).unwrap_or("sinorag-phrase-index-range-buckets")
}

fn ensure_supported_schema(schema: &str) -> Result<()> {
    let legacy_range_bucket_schema =
        schema.starts_with("sinorag-phrase-index-") && schema.ends_with("-range-buckets");
    if schema != schema_label() && !legacy_range_bucket_schema {
        anyhow::bail!(
            "unsupported PhraseIndex schema `{}`; rebuild required (expected `{}`)",
            schema,
            schema_label()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_test_list(doc_ids: &[DocId]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_sorted_docids_delta_varint(doc_ids, &mut out);
        out
    }

    #[test]
    fn delta_doc_id_iter_matches_vec_decoder() {
        let doc_ids = vec![1, 3, 10, 10_000, 10_005];
        let encoded = encode_test_list(&doc_ids);
        let decoded_vec = decode_delta_varint_docids(&encoded).unwrap();
        let decoded_iter: Vec<DocId> = DeltaDocIdIter::new(&encoded).collect();
        assert_eq!(decoded_vec, doc_ids);
        assert_eq!(decoded_iter, doc_ids);
    }

    #[test]
    fn delta_doc_id_iter_empty() {
        let encoded = encode_test_list(&[]);
        let decoded: Vec<DocId> = DeltaDocIdIter::new(&encoded).collect();
        assert!(decoded.is_empty());
    }

    #[test]
    fn delta_doc_id_iter_single() {
        let encoded = encode_test_list(&[42]);
        let decoded: Vec<DocId> = DeltaDocIdIter::new(&encoded).collect();
        assert_eq!(decoded, vec![42]);
    }

    #[test]
    fn streaming_intersection_matches_old_vec_intersection() {
        let lists: Vec<Vec<DocId>> = vec![
            vec![1, 2, 3, 10, 20, 30],
            vec![2, 3, 4, 10, 99],
            vec![3, 10, 11, 12],
        ];
        let encoded: Vec<Vec<u8>> = lists.iter().map(|xs| encode_test_list(xs)).collect();

        let decoded: Vec<Vec<DocId>> = encoded
            .iter()
            .map(|e| decode_delta_varint_docids(e).unwrap())
            .collect();
        let mut old = decoded[0].clone();
        for d in &decoded[1..] {
            old = intersect_sorted(&old, d);
        }

        let mut postings: Vec<PostingSlice<'_>> = encoded
            .iter()
            .map(|e| PostingSlice {
                bytes: e.as_slice(),
                encoded_len: e.len(),
                encoding: ENC_VARINT,
            })
            .collect();
        postings.sort_by_key(|p| p.encoded_len);
        let first = postings.remove(0);
        let mut new: Vec<DocId> = DeltaDocIdIter::new(first.bytes).collect();
        for p in postings {
            intersect_candidates_with_stream(&mut new, DeltaDocIdIter::new(p.bytes));
        }

        assert_eq!(old, new);
        assert_eq!(new, vec![3, 10]);
    }

    #[test]
    fn roaring_round_trip() {
        let docs: Vec<DocId> = (0u32..1000).filter(|d| d % 3 == 0).collect();
        let mut bm = RoaringBitmap::new();
        for &d in &docs {
            bm.insert(d);
        }
        let mut buf: Vec<u8> = Vec::new();
        bm.serialize_into(&mut buf).unwrap();
        let bm2 = RoaringBitmap::deserialize_from(buf.as_slice()).unwrap();
        let out: Vec<DocId> = bm2.iter().collect();
        assert_eq!(docs, out);
    }

    #[test]
    fn duplicate_query_grams_deduped_by_hash_lookup() {
        let grams = phrase_index_grams("如如如如", 2);
        assert_eq!(grams.len(), 3);
        let h0 = xxh3_64(grams[0].as_bytes());
        assert!(grams.iter().all(|g| xxh3_64(g.as_bytes()) == h0));
    }

    #[test]
    fn range_buckets_concatenate_in_global_hash_order() {
        let bucket_count = 8;
        let mut buckets: Vec<Vec<u64>> = (0..bucket_count).map(|_| Vec::new()).collect();
        let hashes = [
            0,
            1,
            u64::MAX,
            u64::MAX - 1,
            1_u64 << 63,
            (1_u64 << 63) - 1,
            0x1111_2222_3333_4444,
            0x9999_aaaa_bbbb_cccc,
            0x7777_8888_9999_aaaa,
            0x2222_3333_4444_5555,
        ];
        for hash in hashes {
            buckets[hash_bucket(hash, bucket_count)].push(hash);
        }
        for bucket in &mut buckets {
            bucket.sort_unstable();
        }
        let concatenated: Vec<u64> = buckets.into_iter().flatten().collect();
        let mut expected = hashes.to_vec();
        expected.sort_unstable();
        assert_eq!(concatenated, expected);
    }
}
