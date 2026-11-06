use serde::{Deserialize, Serialize};

/// GraphDiscovery Corpus Exchange Format v1
/// This module defines the data structures for the GD-CEF format.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusToml {
    pub schema: String,
    pub corpus_id: String,
    pub name: String,
    pub language: String,
    pub snapshot_id: String,

    #[serde(default)]
    pub script: Option<String>,

    #[serde(default)]
    pub description: Option<String>,

    #[serde(default)]
    pub source_url: Option<String>,

    #[serde(default)]
    pub source_type: Option<String>,

    pub rights_id: String,

    #[serde(default)]
    pub rights_notes: Option<String>,

    #[serde(default)]
    pub default_period: Option<String>,

    #[serde(default)]
    pub default_period_rank: Option<i32>,

    #[serde(default)]
    pub default_origin: Option<String>,

    #[serde(default)]
    pub default_traditions: Vec<String>,

    #[serde(default)]
    pub conversion: Option<ConversionInfo>,

    /// How passage text is normalized for search: `"cjk"` (default — the
    /// CBETA normalization: spaces and punctuation removed, case kept) or
    /// `"words"` (for English and other alphabetic scripts: lowercased, with
    /// whole-word boundaries kept; CJK characters are treated as in `"cjk"`).
    /// Ignored for passages that supply `text_normalized` themselves.
    #[serde(default)]
    pub text_profile: Option<String>,
}

impl CorpusToml {
    /// True for `text_profile = "words"`.
    pub fn uses_word_profile(&self) -> bool {
        matches!(self.text_profile.as_deref(), Some("words"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionInfo {
    #[serde(default)]
    pub converter_name: Option<String>,

    #[serde(default)]
    pub converter_version: Option<String>,

    #[serde(default)]
    pub conversion_date: Option<String>,

    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkRecord {
    pub work_id: String,
    pub title_zh: String,

    #[serde(default)]
    pub title_en: Option<String>,

    #[serde(default)]
    pub author: Option<String>,

    #[serde(default)]
    pub dynasty: Option<String>,

    #[serde(default)]
    pub period: Option<String>,

    #[serde(default)]
    pub period_rank: Option<i32>,

    #[serde(default)]
    pub date_start: Option<i32>,

    #[serde(default)]
    pub date_end: Option<i32>,

    #[serde(default)]
    pub date_certainty: Option<String>,

    #[serde(default)]
    pub traditions: Vec<String>,

    #[serde(default)]
    pub genre: Option<String>,

    #[serde(default)]
    pub source_rel_path: Option<String>,

    #[serde(default)]
    pub source_url: Option<String>,

    #[serde(default)]
    pub rights_id: Option<String>,

    #[serde(default)]
    pub quality_flags: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassageRecord {
    pub passage_id: String,
    pub work_id: String,
    pub text: String,

    #[serde(default)]
    pub rights_id: Option<String>,

    #[serde(default)]
    pub section_id: Option<String>,

    #[serde(default)]
    pub section_title: Option<String>,

    #[serde(default)]
    pub locator: Option<String>,

    #[serde(default)]
    pub source_rel_path: Option<String>,

    #[serde(default)]
    pub source_url: Option<String>,

    #[serde(default)]
    pub text_normalized: Option<String>,

    #[serde(default)]
    pub text_type: Option<String>,

    #[serde(default)]
    pub heading_path: Option<String>,

    #[serde(default)]
    pub line_start: Option<String>,

    #[serde(default)]
    pub line_end: Option<String>,

    #[serde(default)]
    pub contains_person: Option<bool>,

    #[serde(default)]
    pub contains_term: Option<bool>,

    #[serde(default)]
    pub quality_flags: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub schema: String,
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub stats: ValidationStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub file: String,
    pub line: Option<usize>,
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStats {
    pub works: usize,
    pub passages: usize,
    pub cjk_chars: usize,
}
