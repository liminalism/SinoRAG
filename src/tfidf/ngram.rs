/// Returns deduplicated n-gram hashes for a document.
/// Use this for document-frequency (DF) counting — each hash appears at most once.
pub fn char_ngram_hashes(text: &str, min_n: usize, max_n: usize) -> Vec<u64> {
    use xxhash_rust::xxh3::xxh3_64;

    let chars: Vec<char> = text.chars().filter(|c| is_cjk_ideograph(*c)).collect();
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();

    for n in min_n..=max_n {
        if chars.len() < n {
            continue;
        }

        for i in 0..=chars.len() - n {
            let mut gram = String::new();
            for ch in &chars[i..i + n] {
                gram.push(*ch);
            }
            if low_value_ngram(&gram) {
                continue;
            }

            let hash = xxh3_64(gram.as_bytes());
            if seen.insert(hash) {
                result.push(hash);
            }
        }
    }

    result
}

/// Returns all n-gram hashes including duplicates, for term-frequency (TF) counting.
/// Unlike `char_ngram_hashes`, the same hash can appear multiple times if the n-gram
/// recurs within the document.
pub fn char_ngram_hashes_all(text: &str, min_n: usize, max_n: usize) -> Vec<u64> {
    use xxhash_rust::xxh3::xxh3_64;

    let chars: Vec<char> = text.chars().filter(|c| is_cjk_ideograph(*c)).collect();
    let mut result = Vec::new();

    for n in min_n..=max_n {
        if chars.len() < n {
            continue;
        }
        for i in 0..=chars.len() - n {
            let mut gram = String::with_capacity(n * 4);
            for ch in &chars[i..i + n] {
                gram.push(*ch);
            }
            if low_value_ngram(&gram) {
                continue;
            }
            result.push(xxh3_64(gram.as_bytes()));
        }
    }

    result
}

pub fn char_ngrams(text: &str, min_n: usize, max_n: usize) -> impl Iterator<Item = String> + '_ {
    // Strip to CJK ideographs only — handles any non-CJK chars that survive upstream normalization
    // and prevents punctuation/digit artifacts from becoming accidental features.
    let chars: Vec<char> = text.chars().filter(|c| is_cjk_ideograph(*c)).collect();
    let mut grams = Vec::new();

    for n in min_n..=max_n {
        if chars.len() < n {
            continue;
        }
        for i in 0..=(chars.len() - n) {
            let gram: String = chars[i..i + n].iter().collect();
            if !low_value_ngram(&gram) {
                grams.push(gram);
            }
        }
    }

    grams.into_iter()
}

fn is_cjk_ideograph(c: char) -> bool {
    ('\u{3400}'..='\u{4DBF}').contains(&c)    // CJK Extension A
        || ('\u{4E00}'..='\u{9FFF}').contains(&c) // CJK Unified Ideographs
        || ('\u{F900}'..='\u{FAFF}').contains(&c) // CJK Compatibility Ideographs
        || ('\u{20000}'..='\u{2A6DF}').contains(&c) // CJK Extension B
}

fn low_value_ngram(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    low_value_ngram_chars(&chars)
}

fn low_value_ngram_chars(chars: &[char]) -> bool {
    if chars.is_empty() {
        return true;
    }
    if chars.iter().all(|c| *c == chars[0]) {
        return true;
    }
    // ABAB repeating pattern — e.g. 如是如是, 云何云何
    let len = chars.len();
    if len >= 4 && len % 2 == 0 {
        let half = len / 2;
        if chars[..half] == chars[half..] {
            return true;
        }
    }
    false
}

/// Writes deduplicated n-gram hashes into `out` (clears first). Reuses `out`
/// across calls to avoid repeated allocation.
pub fn char_ngram_hashes_into(text: &str, min_n: usize, max_n: usize, out: &mut Vec<u64>) {
    use std::collections::HashSet;
    use xxhash_rust::xxh3::xxh3_64;
    out.clear();
    let chars: Vec<char> = text.chars().filter(|c| is_cjk_ideograph(*c)).collect();
    let mut seen: HashSet<u64> = HashSet::new();
    let mut buf = Vec::<u8>::with_capacity(max_n * 4);
    for n in min_n..=max_n {
        if chars.len() < n {
            continue;
        }
        for i in 0..=(chars.len() - n) {
            let window = &chars[i..i + n];
            if low_value_ngram_chars(window) {
                continue;
            }
            buf.clear();
            for ch in window {
                let mut tmp = [0u8; 4];
                buf.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
            }
            let hash = xxh3_64(&buf);
            if seen.insert(hash) {
                out.push(hash);
            }
        }
    }
}

/// Writes all n-gram hashes (including duplicates) into `out` (clears first).
/// Reuses `out` across calls to avoid repeated allocation.
pub fn char_ngram_hashes_all_into(text: &str, min_n: usize, max_n: usize, out: &mut Vec<u64>) {
    use xxhash_rust::xxh3::xxh3_64;
    out.clear();
    let chars: Vec<char> = text.chars().filter(|c| is_cjk_ideograph(*c)).collect();
    let mut buf = Vec::<u8>::with_capacity(max_n * 4);
    for n in min_n..=max_n {
        if chars.len() < n {
            continue;
        }
        for i in 0..=(chars.len() - n) {
            let window = &chars[i..i + n];
            if low_value_ngram_chars(window) {
                continue;
            }
            buf.clear();
            for ch in window {
                let mut tmp = [0u8; 4];
                buf.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
            }
            out.push(xxh3_64(&buf));
        }
    }
}

// ---------------------------------------------------------------------------
// Word features for word-layout passages
// ---------------------------------------------------------------------------
//
// CJK n-grams (above) are the only TF-IDF features of CJK-layout text, as
// before. Passages stored with `normalize_words` (spaces around lowercased
// words; see `crate::normalize`) additionally contribute word unigrams and
// adjacent-word bigrams, so similarity works for English and other
// alphabetic scripts. CJK-layout text never contains a space, so it never
// produces a word feature.
//
// Feature hashes are taken over "\u{1}" + feature. CJK n-gram strings hold
// only ideographs, so the two hash spaces cannot collide by construction.

const WORD_FEATURE_PREFIX: &str = "\u{1}";

/// Common English function words; dropped as unigrams and bigram parts.
const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "even",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "however", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "may", "me", "might", "more", "most", "much", "must", "my", "myself",
    "no", "nor", "not", "now", "of", "off", "on", "once", "one", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shall", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "therefore", "these", "they", "this", "those", "through", "thus", "to", "too", "under",
    "until", "up", "upon", "us", "very", "was", "we", "were", "what", "when", "where", "whether",
    "which", "while", "who", "whom", "whose", "why", "will", "with", "within", "without", "would",
    "yet", "you", "your", "yours", "yourself", "yourselves",
];

fn is_stopword(word: &str) -> bool {
    STOPWORDS.binary_search(&word).is_ok()
}

/// Whether a word-layout token is a content word: not a stopword, a stray
/// letter or a short number.
pub(crate) fn keep_word(token: &str) -> bool {
    if token.chars().any(crate::normalize::is_dense_char) {
        return false;
    }
    let len = token.chars().count();
    if token.chars().all(|c| c.is_ascii_digit()) {
        return len >= 4; // years, part numbers
    }
    len >= 3 && !is_stopword(token)
}

/// Call `f(feature, hash)` for every word feature of a stored normalized
/// text, with repeats (for term frequency). No-op for CJK-layout text.
pub fn for_each_word_feature(normalized: &str, mut f: impl FnMut(&str, u64)) {
    use xxhash_rust::xxh3::xxh3_64;
    if !crate::normalize::has_word_layout(normalized) {
        return;
    }
    let mut key = String::with_capacity(64);
    let mut prev: Option<&str> = None;
    for token in normalized.split(' ').filter(|t| !t.is_empty()) {
        if !keep_word(token) {
            prev = None;
            continue;
        }
        key.clear();
        key.push_str(WORD_FEATURE_PREFIX);
        key.push_str(token);
        f(token, xxh3_64(key.as_bytes()));
        if let Some(p) = prev {
            key.clear();
            key.push_str(WORD_FEATURE_PREFIX);
            key.push_str(p);
            key.push(' ');
            key.push_str(token);
            f(&key[WORD_FEATURE_PREFIX.len()..], xxh3_64(key.as_bytes()));
        }
        prev = Some(token);
    }
}

#[cfg(test)]
mod word_feature_tests {
    use super::*;

    #[test]
    fn stopwords_are_sorted_for_binary_search() {
        let mut sorted = STOPWORDS.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, STOPWORDS);
    }

    #[test]
    fn word_features_unigrams_and_bigrams() {
        let text = crate::normalize::normalize_words("The coolant pump on unit seven was replaced in 2002.");
        let mut feats = Vec::new();
        for_each_word_feature(&text, |s, _| feats.push(s.to_string()));
        assert_eq!(
            feats,
            vec!["coolant", "pump", "coolant pump", "unit", "seven", "unit seven", "replaced", "2002"]
        );
    }

    #[test]
    fn cjk_layout_has_no_word_features() {
        let text = crate::normalize::normalize_zh("如是我聞 Kumarajiva translated this");
        let mut n = 0;
        for_each_word_feature(&text, |_, _| n += 1);
        assert_eq!(n, 0);
    }
}
