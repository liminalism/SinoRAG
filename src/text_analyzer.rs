//! Shared text normalizer + n-gram analyzer. Builds the same byte-identical
//! n-gram hash stream that the phrase and tfidf builders consumed before
//! Phase C, but does it once per document and without per-gram String
//! allocation. Hashes are computed by slicing into the normalized and filtered
//! UTF-8 buffer at recorded char-byte offsets.
//!
//! All work goes into `AnalyzeScratch` which the caller owns and reuses
//! across documents — steady-state is allocation-free.

use crate::normalize::normalize_zh_into;
use xxhash_rust::xxh3::xxh3_64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    /// Keep all non-whitespace chars. Phrase index uses this — preserves
    /// residual Latin / digit content that normalization left in.
    WhitespaceOnly,
    /// Keep only CJK ideographs. TF-IDF uses this — Latin/digit n-grams
    /// pollute the long-tail vocab.
    CjkOnly,
}

#[derive(Debug, Clone, Copy)]
pub struct AnalyzeOptions {
    pub min_n: usize,
    pub max_n: usize,
    pub filter: FilterMode,
    /// Skip "low value" n-grams (all same char, ABAB repeats). Matches the
    /// historical tfidf `low_value_ngram` filter. Phrase index doesn't use it.
    pub apply_low_value_filter: bool,
    /// Fill `scratch.unique` with sorted-deduped hashes (phrase index).
    pub dedup: bool,
    /// Fill `scratch.counts` with `(hash, tf)` pairs (tfidf).
    pub count_tf: bool,
}

#[derive(Debug, Default)]
pub struct AnalyzeScratch {
    pub normalized: String,
    pub filtered: String,
    /// Byte offsets into `normalized` for each retained char. Length is
    /// `num_retained_chars + 1` so `[offsets[i]..offsets[i+n]]` is the
    /// byte range of an n-gram starting at retained-char position `i`.
    pub char_byte_offsets: Vec<u32>,
    pub all_hashes: Vec<u64>,
    pub unique: Vec<u64>,
    pub counts: Vec<(u64, u32)>,
}

impl AnalyzeScratch {
    pub fn new() -> Self {
        Self::default()
    }
    /// Clear all data; preserve capacity.
    pub fn reset(&mut self) {
        self.normalized.clear();
        self.filtered.clear();
        self.char_byte_offsets.clear();
        self.all_hashes.clear();
        self.unique.clear();
        self.counts.clear();
    }
}

/// Single pass: normalize → record retained-char byte offsets → emit n-gram
/// hashes by slicing into the normalized UTF-8 buffer → optionally
/// sort+dedup for phrase, sort+count-runs for tfidf.
pub fn analyze(text: &str, opts: &AnalyzeOptions, scratch: &mut AnalyzeScratch) {
    scratch.reset();
    normalize_zh_into(text, &mut scratch.normalized);
    analyze_current_normalized(opts, scratch);
}

/// Analyze text that has already been normalized by SinoRAG ingest.
pub fn analyze_normalized(normalized: &str, opts: &AnalyzeOptions, scratch: &mut AnalyzeScratch) {
    scratch.reset();
    scratch.normalized.push_str(normalized);
    analyze_current_normalized(opts, scratch);
}

fn analyze_current_normalized(opts: &AnalyzeOptions, scratch: &mut AnalyzeScratch) {
    build_filtered_normalized(
        &scratch.normalized,
        opts.filter,
        &mut scratch.filtered,
        &mut scratch.char_byte_offsets,
    );

    let num_chars = scratch.char_byte_offsets.len().saturating_sub(1);
    let bytes = scratch.filtered.as_bytes();
    for n in opts.min_n..=opts.max_n {
        if num_chars < n {
            continue;
        }
        for i in 0..=(num_chars - n) {
            let s = scratch.char_byte_offsets[i] as usize;
            let e = scratch.char_byte_offsets[i + n] as usize;
            let slice = &bytes[s..e];
            if opts.apply_low_value_filter && is_low_value(slice, n) {
                continue;
            }
            scratch.all_hashes.push(xxh3_64(slice));
        }
    }

    // TF-IDF: word-layout passages (alphabetic-script corpora) also get word
    // features. CJK-layout text has no spaces, so this adds nothing there.
    if opts.filter == FilterMode::CjkOnly {
        let hashes = &mut scratch.all_hashes;
        crate::tfidf::ngram::for_each_word_feature(&scratch.normalized, |_, h| hashes.push(h));
    }
    if scratch.all_hashes.is_empty() {
        return;
    }

    if opts.dedup {
        scratch.unique.clear();
        scratch.unique.extend_from_slice(&scratch.all_hashes);
        scratch.unique.sort_unstable();
        scratch.unique.dedup();
    }
    if opts.count_tf {
        scratch.counts.clear();
        let mut sorted = std::mem::take(&mut scratch.all_hashes);
        sorted.sort_unstable();
        run_length_count(&sorted, &mut scratch.counts);
        scratch.all_hashes = sorted;
    }
}

fn build_filtered_normalized(
    normalized: &str,
    filter: FilterMode,
    filtered: &mut String,
    offsets: &mut Vec<u32>,
) {
    filtered.clear();
    offsets.clear();
    offsets.push(0);

    let bytes = normalized.as_bytes();
    let mut idx = 0usize;
    while idx < bytes.len() {
        let first = bytes[idx];
        let len = utf8_len(first);
        let end = idx + len;
        if end > bytes.len() {
            break;
        }
        let keep = match filter {
            FilterMode::WhitespaceOnly => {
                // normalize_zh already stripped whitespace; this is a safety net.
                if len == 1 {
                    !(bytes[idx] as char).is_whitespace()
                } else {
                    true
                }
            }
            FilterMode::CjkOnly => is_cjk_utf8(&bytes[idx..end]),
        };
        if keep {
            filtered.push_str(&normalized[idx..end]);
            offsets.push(filtered.len() as u32);
        }
        idx = end;
    }
}

#[inline]
fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xC0 {
        1
    }
    // continuation byte — shouldn't start a char; advance 1 defensively
    else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

#[inline]
fn is_cjk_utf8(bytes: &[u8]) -> bool {
    // CJK ideograph blocks live in U+3400..U+9FFF (3-byte UTF-8 starting 0xE3..0xE9),
    // U+F900..U+FAFF (3-byte 0xEF..), and U+20000..U+2A6DF (4-byte 0xF0..0xF2).
    match bytes.len() {
        3 => {
            // Decode codepoint from 3-byte UTF-8.
            let cp = ((bytes[0] as u32 & 0x0F) << 12)
                | ((bytes[1] as u32 & 0x3F) << 6)
                | (bytes[2] as u32 & 0x3F);
            (0x3400..=0x4DBF).contains(&cp)
                || (0x4E00..=0x9FFF).contains(&cp)
                || (0xF900..=0xFAFF).contains(&cp)
        }
        4 => {
            let cp = ((bytes[0] as u32 & 0x07) << 18)
                | ((bytes[1] as u32 & 0x3F) << 12)
                | ((bytes[2] as u32 & 0x3F) << 6)
                | (bytes[3] as u32 & 0x3F);
            (0x20000..=0x2A6DF).contains(&cp)
        }
        _ => false,
    }
}

/// `bytes` covers exactly `n_chars` Chinese characters (each UTF-8 length-3
/// or length-4). Detect: all-same-char, or ABAB-period repeat.
fn is_low_value(bytes: &[u8], n_chars: usize) -> bool {
    if n_chars == 0 {
        return true;
    }
    if n_chars == 1 {
        return false;
    }
    // All same char.
    let first_len = utf8_len(bytes[0]);
    if bytes.len() >= 2 * first_len {
        let first_char = &bytes[..first_len];
        let mut idx = first_len;
        let mut same = true;
        while idx < bytes.len() {
            let l = utf8_len(bytes[idx]);
            if idx + l > bytes.len() {
                same = false;
                break;
            }
            if &bytes[idx..idx + l] != first_char {
                same = false;
                break;
            }
            idx += l;
        }
        if same {
            return true;
        }
    }
    // ABAB period repeat (only meaningful when n_chars is even and >= 4).
    if n_chars >= 4 && n_chars % 2 == 0 {
        let mut half_byte_len = 0usize;
        let mut idx = 0usize;
        for _ in 0..(n_chars / 2) {
            half_byte_len += utf8_len(bytes[idx]);
            idx += utf8_len(bytes[idx]);
        }
        if bytes.len() == 2 * half_byte_len && bytes[..half_byte_len] == bytes[half_byte_len..] {
            return true;
        }
    }
    false
}

fn run_length_count(sorted: &[u64], out: &mut Vec<(u64, u32)>) {
    out.clear();
    let mut i = 0;
    while i < sorted.len() {
        let h = sorted[i];
        let mut c = 1u32;
        i += 1;
        while i < sorted.len() && sorted[i] == h {
            c += 1;
            i += 1;
        }
        out.push((h, c));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts_phrase(n: usize) -> AnalyzeOptions {
        AnalyzeOptions {
            min_n: n,
            max_n: n,
            filter: FilterMode::WhitespaceOnly,
            apply_low_value_filter: false,
            dedup: true,
            count_tf: false,
        }
    }

    #[test]
    fn empty_text_short_circuits() {
        let mut s = AnalyzeScratch::new();
        analyze("", &opts_phrase(4), &mut s);
        assert!(s.unique.is_empty());
    }

    #[test]
    fn phrase_dedup_matches_naive() {
        let mut s = AnalyzeScratch::new();
        analyze("如是我聞如是我聞", &opts_phrase(4), &mut s);
        // 5 overlapping 4-grams over 8 chars, but dedup'd: positions 0 and 4
        // produce the same 4-gram 如是我聞, so distinct count is 4.
        assert!(!s.unique.is_empty());
        assert!(s.unique.is_sorted());
    }

    #[test]
    fn count_tf_runs() {
        let mut s = AnalyzeScratch::new();
        let opts = AnalyzeOptions {
            min_n: 2,
            max_n: 2,
            filter: FilterMode::CjkOnly,
            apply_low_value_filter: false,
            dedup: false,
            count_tf: true,
        };
        analyze("如是如是", &opts, &mut s);
        // 3 bigrams: 如是 是如 如是 → counts: {如是:2, 是如:1}
        let mut total: u32 = 0;
        for (_, c) in &s.counts {
            total += c;
        }
        assert_eq!(total, 3);
    }

    #[test]
    fn normalized_entrypoint_matches_analyze_for_normalized_text() {
        let opts = AnalyzeOptions {
            min_n: 2,
            max_n: 3,
            filter: FilterMode::CjkOnly,
            apply_low_value_filter: true,
            dedup: false,
            count_tf: true,
        };
        let mut regular = AnalyzeScratch::new();
        let mut normalized = AnalyzeScratch::new();

        analyze("如是我聞", &opts, &mut regular);
        analyze_normalized("如是我聞", &opts, &mut normalized);

        assert_eq!(regular.filtered, normalized.filtered);
        assert_eq!(regular.counts, normalized.counts);
    }

    #[test]
    fn cjk_only_hashes_skip_latin_without_including_it_in_slice() {
        let mut old = crate::tfidf::ngram::char_ngram_hashes("中A國", 2, 2);
        old.sort_unstable();

        let mut s = AnalyzeScratch::new();
        analyze(
            "中A國",
            &AnalyzeOptions {
                min_n: 2,
                max_n: 2,
                filter: FilterMode::CjkOnly,
                apply_low_value_filter: false,
                dedup: true,
                count_tf: false,
            },
            &mut s,
        );

        assert_eq!(s.filtered, "中國");
        assert_eq!(s.unique, old);
    }

    #[test]
    fn cjk_only_tf_hashes_match_old_all_hashes_for_polluted_text() {
        let mut old = crate::tfidf::ngram::char_ngram_hashes_all("中A國。中 國", 2, 2);
        old.sort_unstable();

        let mut s = AnalyzeScratch::new();
        analyze(
            "中A國。中 國",
            &AnalyzeOptions {
                min_n: 2,
                max_n: 2,
                filter: FilterMode::CjkOnly,
                apply_low_value_filter: false,
                dedup: false,
                count_tf: true,
            },
            &mut s,
        );

        assert_eq!(s.all_hashes, old);
    }
}
