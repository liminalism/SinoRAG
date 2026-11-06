use std::collections::HashMap;
use std::sync::OnceLock;
use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_normalization::UnicodeNormalization;
use wide::u8x16;

// Character variant tables from https://github.com/zhaowenping/cbeta (cc/ directory).
// See THIRD_PARTY_NOTICES.md for attribution.
const ST_CHARS: &str = include_str!("../assets/cbeta/STCharacters.txt");
const KX_VARIANTS: &str = include_str!("../assets/cbeta/KXVariants.txt");
const JP_VARIANTS: &str = include_str!("../assets/cbeta/JPVariants.txt");

static VARIANT_MAP: OnceLock<HashMap<char, char>> = OnceLock::new();

/// Maps variant CJK characters to a single canonical form (modern Traditional Chinese).
///
/// Sources:
///   STCharacters   — Simplified → Traditional (first target when ambiguous)
///   KXVariants     — Kangxi archaic → modern standard (reversed from file)
///   JPVariants     — Japanese variant → standard
fn variant_map() -> &'static HashMap<char, char> {
    VARIANT_MAP.get_or_init(|| {
        let mut map = HashMap::with_capacity(4500);

        // STCharacters: tab-delimited, Simplified → Traditional.
        // Some lines have space-separated alternatives; take only the first.
        for line in ST_CHARS.lines() {
            let line = line.trim();
            if let Some((src, targets)) = line.split_once('\t') {
                let src = src.trim();
                let target = targets.split_whitespace().next().unwrap_or("");
                if let (Some(s), Some(t)) = (src.chars().next(), target.chars().next()) {
                    if s != t {
                        map.insert(s, t);
                    }
                }
            }
        }

        // KXVariants: space-delimited, `modern kangxi_form`.
        // Buddhist texts use Kangxi forms, so normalize modern → kangxi.
        // This aligns with STCharacters (Simplified also maps to the
        // traditional/kangxi form), so all variants converge.
        for line in KX_VARIANTS.lines() {
            let line = line.trim();
            let mut chars = line.split_whitespace();
            if let (Some(modern), Some(kangxi)) = (chars.next(), chars.next()) {
                if let (Some(m), Some(k)) = (modern.chars().next(), kangxi.chars().next()) {
                    if m != k {
                        map.entry(m).or_insert(k);
                    }
                }
            }
        }

        // JPVariants: tab-delimited, `jp_variant standard`.
        // Only insert if neither the source nor target is already mapped by
        // ST or KX — avoids Japanese Shinjitai simplification (國→国) from
        // overriding the correct Traditional Chinese form.
        for line in JP_VARIANTS.lines() {
            let line = line.trim();
            if let Some((src, target)) = line.split_once('\t') {
                if let (Some(s), Some(t)) =
                    (src.trim().chars().next(), target.trim().chars().next())
                {
                    if s != t && !map.contains_key(&s) && !map.contains_key(&t) {
                        map.insert(s, t);
                    }
                }
            }
        }

        map
    })
}

pub fn contains_cjk(text: &str) -> bool {
    text.chars().any(|ch| {
        ('\u{3400}'..='\u{4dbf}').contains(&ch)
            || ('\u{4e00}'..='\u{9fff}').contains(&ch)
            || ('\u{f900}'..='\u{faff}').contains(&ch)
    })
}

pub fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn normalize_zh(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    normalize_zh_into(text, &mut out);
    out
}

/// Allocation-conscious variant of `normalize_zh`. Reuses `out`'s capacity
/// across calls. Phase C hot path uses this from `text_analyzer::analyze`.
///
/// Includes an ASCII fast-path: 16-byte input blocks that are pure ASCII
/// (high bit clear, no whitespace, no punctuation, no symbol) get accepted
/// wholesale via one `wide::u8x16` mask check. Mixed and non-ASCII blocks
/// fall back to the scalar NFKC + general-category filter.
pub fn normalize_zh_into(text: &str, out: &mut String) {
    out.clear();
    out.reserve(text.len());

    let bytes = text.as_bytes();
    let mut i = 0usize;
    let mut scalar_start = 0usize;
    while i + 16 <= bytes.len() {
        if all_ascii_keep_block_16(&bytes[i..i + 16]) {
            // Safe: every byte is < 0x80 (verified by the mask) so the slice
            // is valid UTF-8 trivially.
            // SAFETY: ASCII subset confirmed above.
            unsafe {
                out.push_str(std::str::from_utf8_unchecked(&bytes[i..i + 16]));
            }
            i += 16;
            scalar_start = i;
        } else {
            // Mixed or non-ASCII — drop into scalar fallback for the rest
            // of the string (single contiguous tail is faster than chunked
            // round-trips between SIMD and scalar).
            break;
        }
    }
    let _ = scalar_start;

    // Scalar fallback for the remainder. Use NFKC + GeneralCategory exactly
    // as the original implementation did.
    if i < bytes.len() {
        // SAFETY: `i` only advances on confirmed ASCII codepoint boundaries
        // (every retained byte was < 0x80, which is a 1-byte UTF-8 sequence).
        let tail = unsafe { std::str::from_utf8_unchecked(&bytes[i..]) };
        let vmap = variant_map();
        for ch in tail.nfkc() {
            if ch.is_whitespace() {
                continue;
            }
            if is_strippable_category(ch) {
                continue;
            }
            let ch = vmap.get(&ch).copied().unwrap_or(ch);
            out.push(ch);
        }
    }
}

// ---------------------------------------------------------------------------
// Word-aware normalization (alphabetic-script corpora)
// ---------------------------------------------------------------------------
//
// `normalize_zh` deletes every space and keeps case, which is right for
// classical Chinese but leaves English with no word boundaries
// ("cat" matches "concatenate") and case-sensitive matching.
//
// `normalize_words` is the alternative stored form for corpora ingested with
// `text_profile = "words"` (GD-CEF corpus.toml). The rules are:
//   * CJK ideographs, kana and hangul ("dense" characters) go through exactly
//     the `normalize_zh` pipeline (NFKC, punctuation dropped, variant map),
//     and adjacent dense characters are still joined without spaces;
//   * every other letter/digit run is a word: lowercased, and wrapped in
//     single spaces, so " coolant pump " — a stored text and a query agree
//     on whole-word boundaries.
//
// Two properties keep CJK behaviour unchanged, and the tests pin them:
//   1. For text without word characters, `normalize_words == normalize_zh`.
//   2. `normalize_zh` output never contains a space, while word-layout text
//      with any word character always does. So whether a stored passage
//      uses the word layout can be told from the text itself
//      (`has_word_layout`), and queries are matched against each layout with
//      the matching query form (`PhraseForms`).

/// Characters that keep the dense (CJK) treatment under `normalize_words`.
pub fn is_dense_char(ch: char) -> bool {
    matches!(ch,
        '\u{3040}'..='\u{30ff}'     // hiragana, katakana
        | '\u{3100}'..='\u{312f}'   // bopomofo
        | '\u{3400}'..='\u{4dbf}'
        | '\u{4e00}'..='\u{9fff}'
        | '\u{ac00}'..='\u{d7af}'   // hangul syllables
        | '\u{f900}'..='\u{faff}'
        | '\u{20000}'..='\u{3134f}')
}

/// Word-aware normalization; see the section comment above.
pub fn normalize_words(text: &str) -> String {
    let vmap = variant_map();
    let mut out = String::with_capacity(text.len() + 8);
    let mut in_word = false;
    let mut broke = false; // a dropped separator since the last kept char
    for ch in text.nfkc() {
        if ch.is_whitespace() || is_strippable_category(ch) {
            broke = true;
            continue;
        }
        let ch = vmap.get(&ch).copied().unwrap_or(ch);
        if is_dense_char(ch) {
            if in_word {
                out.push(' ');
                in_word = false;
            }
            out.push(ch);
        } else {
            if !in_word {
                if !out.ends_with(' ') {
                    out.push(' ');
                }
            } else if broke {
                out.push(' ');
            }
            out.extend(ch.to_lowercase());
            in_word = true;
        }
        broke = false;
    }
    if in_word {
        out.push(' ');
    }
    out
}

/// True when a stored normalized text uses the word layout. `normalize_zh`
/// never emits whitespace, so CBETA and other CJK-normalized passages are
/// always `false`.
#[inline]
pub fn has_word_layout(normalized: &str) -> bool {
    normalized.as_bytes().contains(&b' ')
}

/// A query phrase in both stored layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhraseForms {
    /// `normalize_zh(phrase)`: matched against CJK-layout passages.
    pub zh: String,
    /// `normalize_words(phrase)`: matched against word-layout passages.
    pub words: String,
}

impl PhraseForms {
    pub fn new(phrase: &str) -> Self {
        let zh = normalize_zh(phrase);
        let words = normalize_words(phrase);
        Self { zh, words }
    }

    /// Forms for an already-normalized variant string (as produced by
    /// `normalize_zh` or `normalize_words`), matched only against the layout
    /// it belongs to. Pure-CJK variants apply to both layouts.
    pub fn from_normalized_variant(variant: &str) -> Self {
        if variant.contains(' ') {
            return Self { zh: String::new(), words: variant.to_string() };
        }
        // Re-normalizing keeps the CJK path identical to callers that used
        // to pass these variants through `normalize_zh` a second time.
        let zh = normalize_zh(variant);
        if variant.chars().all(is_dense_char) {
            Self { words: zh.clone(), zh }
        } else {
            Self { zh, words: String::new() }
        }
    }

    /// Both layouts agree (pure CJK phrase): one form does for everything,
    /// and callers keep their original single-form code path.
    pub fn is_single(&self) -> bool {
        self.zh == self.words
    }

    pub fn is_empty(&self) -> bool {
        self.zh.is_empty() && self.words.trim().is_empty()
    }

    /// Distinct forms, CJK form first.
    pub fn distinct(&self) -> Vec<&str> {
        if self.is_single() {
            vec![self.zh.as_str()]
        } else {
            vec![self.zh.as_str(), self.words.as_str()]
        }
    }

    /// Exact-phrase test against one stored normalized text.
    pub fn matches(&self, stored: &str) -> bool {
        if has_word_layout(stored) {
            !self.words.trim().is_empty() && stored.contains(self.words.as_str())
        } else {
            !self.zh.is_empty() && stored.contains(self.zh.as_str())
        }
    }

    /// The form that applies to one stored text (for locating occurrences).
    pub fn form_for(&self, stored: &str) -> &str {
        if has_word_layout(stored) {
            &self.words
        } else {
            &self.zh
        }
    }

    /// Character count used against the phrase-index gram size (spaces are
    /// not indexed). The shorter form decides.
    pub fn indexed_len(&self) -> usize {
        let words = self.words.chars().filter(|c| *c != ' ').count();
        let zh = self.zh.chars().count();
        match (zh, words) {
            (0, w) => w,
            (z, 0) => z,
            (z, w) => z.min(w),
        }
    }
}

/// File next to `passages.parquet/` recording which corpora were ingested
/// with the word layout: `{"<corpus_id>": "words", ...}`. Absent for
/// CJK-only data roots.
pub const TEXT_PROFILES_FILE: &str = "text_profiles.json";

/// Record (or clear) a corpus's text profile for the data root holding
/// `passages_parquet`.
pub fn record_text_profile(
    passages_parquet: &std::path::Path,
    corpus_id: &str,
    words: bool,
) -> std::io::Result<()> {
    let Some(root) = passages_parquet.parent() else {
        return Ok(());
    };
    let path = root.join(TEXT_PROFILES_FILE);
    let mut map: std::collections::BTreeMap<String, String> = std::fs::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let changed = if words {
        map.insert(corpus_id.to_string(), "words".to_string()).as_deref() != Some("words")
    } else {
        map.remove(corpus_id).is_some()
    };
    if !changed {
        return Ok(());
    }
    if map.is_empty() {
        let _ = std::fs::remove_file(&path);
        return Ok(());
    }
    std::fs::write(&path, serde_json::to_vec_pretty(&map).unwrap_or_default())
}

/// True when any corpus under this data root uses the word layout.
pub fn data_root_has_word_corpus(passages_parquet: &std::path::Path) -> bool {
    passages_parquet
        .parent()
        .map(|root| root.join(TEXT_PROFILES_FILE))
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<std::collections::BTreeMap<String, String>>(&b).ok())
        .is_some_and(|m| m.values().any(|v| v == "words"))
}

/// True when the data root holds word-layout corpora and nothing else: a
/// collection of someone's documents, with no CBETA beside it.
pub fn data_root_is_word_only(passages_parquet: &std::path::Path) -> bool {
    let profiles: std::collections::BTreeMap<String, String> = passages_parquet
        .parent()
        .map(|root| root.join(TEXT_PROFILES_FILE))
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let Ok(partitions) = std::fs::read_dir(passages_parquet) else {
        return false;
    };
    let mut corpora = 0usize;
    for entry in partitions.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(corpus) = name.strip_prefix("source_corpus=") else {
            continue;
        };
        if profiles.get(corpus).map(String::as_str) != Some("words") {
            return false;
        }
        corpora += 1;
    }
    corpora > 0
}

/// `true` iff every byte in `block` is ASCII non-whitespace, non-punctuation,
/// non-symbol (the "keep" set under the existing `normalize_zh` policy).
///
/// Implements the policy as a set of mask checks on a u8x16 SIMD register:
/// - reject any byte >= 0x80 (would be a UTF-8 continuation / leading byte).
/// - reject 0x00..=0x20 (control chars + space).
/// - reject 0x7F (DEL).
/// - reject ASCII punctuation ranges (`!`-`/`, `:`-`@`, `[`-``` ` ```, `{`-`~`).
///
/// Calling this on a block that doesn't fully satisfy the predicate returns
/// false; the caller then falls back to the scalar path.
#[inline]
fn all_ascii_keep_block_16(block: &[u8]) -> bool {
    debug_assert_eq!(block.len(), 16);
    let arr: [u8; 16] = block.try_into().expect("len 16 enforced");
    let v = u8x16::new(arr);

    // wide-0.7's u8x16 has cmp_eq, max, min, move_mask — but no unsigned
    // cmp_lt. We synthesize range checks via:
    //   (v >= low)  ⟺  v.max(low) == v
    //   (v <= high) ⟺  v.min(high) == v
    // Each comparison returns 0xFF in passing lanes, 0x00 in failing lanes.
    // The keep-set is ASCII alnum: 0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A.

    let in_range = |lo: u8, hi: u8| -> u8x16 {
        let ge = v.max(u8x16::splat(lo)).cmp_eq(v);
        let le = v.min(u8x16::splat(hi)).cmp_eq(v);
        ge & le
    };
    let allowed = in_range(0x30, 0x39) | in_range(0x41, 0x5A) | in_range(0x61, 0x7A);
    // move_mask gathers the high bit of each lane; passing lanes are 0xFF
    // (high bit set), failing are 0x00. All-pass ⇔ low 16 bits all set.
    allowed.move_mask() == 0xFFFF
}

#[inline]
fn is_strippable_category(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
            | GeneralCategory::SpaceSeparator
            | GeneralCategory::LineSeparator
            | GeneralCategory::ParagraphSeparator
            | GeneralCategory::MathSymbol
            | GeneralCategory::CurrencySymbol
            | GeneralCategory::ModifierSymbol
            | GeneralCategory::OtherSymbol
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_data_root_is_word_only_when_every_corpus_is() {
        let root = tempfile::tempdir().unwrap();
        let parquet = root.path().join("passages.parquet");
        assert!(!data_root_is_word_only(&parquet), "no corpus at all");

        std::fs::create_dir_all(parquet.join("source_corpus=documents")).unwrap();
        assert!(!data_root_is_word_only(&parquet), "profile not recorded");
        record_text_profile(&parquet, "documents", true).unwrap();
        assert!(data_root_is_word_only(&parquet));

        std::fs::create_dir_all(parquet.join("source_corpus=cbeta")).unwrap();
        assert!(!data_root_is_word_only(&parquet), "a CJK corpus beside it");
        assert!(data_root_has_word_corpus(&parquet));
    }

    #[test]
    fn normalize_strips_punct_and_ws() {
        assert_eq!(normalize_zh("a b!c"), "abc");
    }

    #[test]
    fn cjk_preserved() {
        assert_eq!(normalize_zh("如是我聞"), "如是我聞");
    }

    #[test]
    fn cjk_punct_stripped() {
        assert_eq!(normalize_zh("如是， 我聞。"), "如是我聞");
    }

    #[test]
    fn variant_map_loads() {
        let m = variant_map();
        assert!(m.len() > 3000, "expected >3000 entries, got {}", m.len());
    }

    #[test]
    fn simplified_normalizes_to_traditional() {
        // 禅 (Simplified) → 禪 (Traditional)
        assert_eq!(normalize_zh("禅"), "禪");
    }

    #[test]
    fn modern_normalizes_to_kangxi() {
        // 眾 (modern) → 衆 (Kangxi standard used in Buddhist texts)
        assert_eq!(normalize_zh("眾"), "衆");
    }

    #[test]
    fn mixed_variants_in_phrase() {
        // 钵 (Simplified) and 鉢 (Traditional/Kangxi) should converge.
        // 缽 (modern) also maps to 鉢 (Kangxi) via KXVariants.
        let a = normalize_zh("持鉢入城");
        let b = normalize_zh("持钵入城");
        let c = normalize_zh("持缽入城");
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn words_layout_lowercases_and_bounds_words() {
        assert_eq!(normalize_words("The Coolant-Pump, replaced."), " the coolant pump replaced ");
        assert_eq!(normalize_words("Hello, 世界 world"), " hello 世界 world ");
        assert_eq!(normalize_words("卷3"), format!("{} 3 ", normalize_zh("卷")));
    }

    #[test]
    fn words_layout_equals_zh_for_pure_cjk() {
        for s in ["如是我聞", "如是， 我聞。", "持钵入城", "禅眾", "「一時佛在」舍衛國"] {
            assert_eq!(normalize_words(s), normalize_zh(s), "{s}");
            assert!(PhraseForms::new(s).is_single());
        }
    }

    #[test]
    fn zh_layout_never_has_spaces() {
        for s in ["a b c", "Kumārajīva 譯", "T08n0235 金剛經", "x\u{3000}y"] {
            assert!(!has_word_layout(&normalize_zh(s)), "{s}");
        }
    }

    #[test]
    fn phrase_forms_respect_word_boundaries() {
        let stored = normalize_words("We concatenate strings; the cat sat.");
        assert!(PhraseForms::new("cat").matches(&stored));
        assert!(PhraseForms::new("The CAT sat").matches(&stored));
        assert!(!PhraseForms::new("cate").matches(&stored));
        assert!(!PhraseForms::new("cat sat on").matches(&stored));
    }

    #[test]
    fn phrase_forms_on_cjk_layout_match_like_before() {
        let stored = normalize_zh("如是我聞。一時佛在舍衛國，T08n0235");
        for q in ["我聞", "一時佛在", "T08n0235", "佛在舍衛國T08"] {
            assert_eq!(
                PhraseForms::new(q).matches(&stored),
                stored.contains(&normalize_zh(q)),
                "{q}"
            );
        }
    }

    #[test]
    fn variant_forms_stay_in_their_layout() {
        let cjk_stored = normalize_zh("Kumarajiva譯 kumarajiva");
        let word_stored = normalize_words("Kumarajiva translated");
        let zh_variant = PhraseForms::from_normalized_variant(&normalize_zh("Kumarajiva"));
        let word_variant = PhraseForms::from_normalized_variant(&normalize_words("Kumarajiva"));
        assert!(zh_variant.matches(&cjk_stored));
        assert!(!zh_variant.matches(&word_stored));
        assert!(word_variant.matches(&word_stored));
        // The lowercased word form must never reach CJK-layout text.
        assert!(!word_variant.matches(&normalize_zh("kumarajiva")));
        let dense = PhraseForms::from_normalized_variant("如是我聞");
        assert!(dense.matches(&normalize_words("如是我聞 thus")));
        assert_eq!(word_variant.indexed_len(), 10);
    }

    #[test]
    fn ascii_fast_path_matches_scalar() {
        // 32 ASCII chars; first 16 are pure alnum, then a punct triggers scalar tail.
        let a = normalize_zh("0123456789abcdefABCDEF, hello world!");
        // Scalar reference: punctuation + whitespace stripped.
        assert_eq!(a, "0123456789abcdefABCDEFhelloworld");
    }
}
