use crate::cef::{
    CorpusToml, PassageRecord, ValidationError, ValidationReport, ValidationStats,
    ValidationWarning, WorkRecord,
};
use crate::models::PassageRecord as ModelPassageRecord;
use crate::normalize::normalize_zh;
use crate::storage;
use anyhow::{anyhow, Context, Result};
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

pub fn validate(input: PathBuf) -> Result<ValidationReport> {
    let corpus_dir = if input.is_file() {
        input
            .parent()
            .ok_or_else(|| anyhow!("Cannot determine corpus directory from file path"))?
    } else {
        &input
    };

    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Check corpus.toml
    let corpus_toml_path = corpus_dir.join("corpus.toml");
    if !corpus_toml_path.exists() {
        errors.push(ValidationError {
            file: "corpus.toml".to_string(),
            line: None,
            code: "missing_file".to_string(),
            message: "corpus.toml is required".to_string(),
        });
    } else {
        let corpus_toml_content = fs::read_to_string(&corpus_toml_path)
            .with_context(|| format!("read corpus.toml from {}", corpus_toml_path.display()))?;

        let corpus: CorpusToml = toml::from_str(&corpus_toml_content)
            .with_context(|| format!("parse corpus.toml from {}", corpus_toml_path.display()))?;

        if corpus.schema != "gd-cef-v1" {
            errors.push(ValidationError {
                file: "corpus.toml".to_string(),
                line: None,
                code: "invalid_schema".to_string(),
                message: format!("schema must be 'gd-cef-v1', got '{}'", corpus.schema),
            });
        }

        if corpus.corpus_id.is_empty() {
            errors.push(ValidationError {
                file: "corpus.toml".to_string(),
                line: None,
                code: "missing_field".to_string(),
                message: "corpus_id is required".to_string(),
            });
        }

        if corpus.name.is_empty() {
            errors.push(ValidationError {
                file: "corpus.toml".to_string(),
                line: None,
                code: "missing_field".to_string(),
                message: "name is required".to_string(),
            });
        }

        if corpus.language.is_empty() {
            errors.push(ValidationError {
                file: "corpus.toml".to_string(),
                line: None,
                code: "missing_field".to_string(),
                message: "language is required".to_string(),
            });
        }

        if let Some(profile) = corpus.text_profile.as_deref() {
            if !matches!(profile, "cjk" | "words") {
                errors.push(ValidationError {
                    file: "corpus.toml".to_string(),
                    line: None,
                    code: "invalid_text_profile".to_string(),
                    message: format!("text_profile must be 'cjk' or 'words', got '{profile}'"),
                });
            }
        }

        if corpus.snapshot_id.is_empty() {
            errors.push(ValidationError {
                file: "corpus.toml".to_string(),
                line: None,
                code: "missing_field".to_string(),
                message: "snapshot_id is required".to_string(),
            });
        }

        if corpus.rights_id.is_empty() {
            errors.push(ValidationError {
                file: "corpus.toml".to_string(),
                line: None,
                code: "missing_field".to_string(),
                message: "rights_id is required".to_string(),
            });
        }
    }

    // Check works.jsonl
    let works_path = corpus_dir.join("works.jsonl");
    let mut works: Vec<WorkRecord> = Vec::new();
    let mut work_ids = HashSet::new();

    if !works_path.exists() {
        errors.push(ValidationError {
            file: "works.jsonl".to_string(),
            line: None,
            code: "missing_file".to_string(),
            message: "works.jsonl is required".to_string(),
        });
    } else {
        let works_content = fs::read_to_string(&works_path)
            .with_context(|| format!("read works.jsonl from {}", works_path.display()))?;

        for (line_num, line) in works_content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<WorkRecord>(line) {
                Ok(work) => {
                    if work.work_id.is_empty() {
                        errors.push(ValidationError {
                            file: "works.jsonl".to_string(),
                            line: Some(line_num + 1),
                            code: "missing_field".to_string(),
                            message: "work_id is required".to_string(),
                        });
                    } else {
                        if work_ids.contains(&work.work_id) {
                            errors.push(ValidationError {
                                file: "works.jsonl".to_string(),
                                line: Some(line_num + 1),
                                code: "duplicate_work_id".to_string(),
                                message: format!("duplicate work_id: {}", work.work_id),
                            });
                        } else {
                            work_ids.insert(work.work_id.clone());
                        }
                    }

                    if work.title_zh.is_empty() {
                        errors.push(ValidationError {
                            file: "works.jsonl".to_string(),
                            line: Some(line_num + 1),
                            code: "missing_field".to_string(),
                            message: "title_zh is required".to_string(),
                        });
                    }

                    if let (Some(start), Some(end)) = (work.date_start, work.date_end) {
                        if start > end {
                            errors.push(ValidationError {
                                file: "works.jsonl".to_string(),
                                line: Some(line_num + 1),
                                code: "invalid_date_range".to_string(),
                                message: format!("date_start ({}) > date_end ({})", start, end),
                            });
                        }
                    }

                    works.push(work);
                }
                Err(e) => {
                    errors.push(ValidationError {
                        file: "works.jsonl".to_string(),
                        line: Some(line_num + 1),
                        code: "invalid_json".to_string(),
                        message: format!("failed to parse JSON: {}", e),
                    });
                }
            }
        }
    }

    // Check passages.jsonl
    let passages_path = corpus_dir.join("passages.jsonl");
    let mut passages: Vec<PassageRecord> = Vec::new();
    let mut passage_ids = HashSet::new();
    let mut cjk_char_count = 0usize;
    let missing_dates_count;

    if !passages_path.exists() {
        errors.push(ValidationError {
            file: "passages.jsonl".to_string(),
            line: None,
            code: "missing_file".to_string(),
            message: "passages.jsonl is required".to_string(),
        });
    } else {
        let passages_content = fs::read_to_string(&passages_path)
            .with_context(|| format!("read passages.jsonl from {}", passages_path.display()))?;

        for (line_num, line) in passages_content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<PassageRecord>(line) {
                Ok(passage) => {
                    if passage.passage_id.is_empty() {
                        errors.push(ValidationError {
                            file: "passages.jsonl".to_string(),
                            line: Some(line_num + 1),
                            code: "missing_field".to_string(),
                            message: "passage_id is required".to_string(),
                        });
                    } else {
                        if passage_ids.contains(&passage.passage_id) {
                            errors.push(ValidationError {
                                file: "passages.jsonl".to_string(),
                                line: Some(line_num + 1),
                                code: "duplicate_passage_id".to_string(),
                                message: format!("duplicate passage_id: {}", passage.passage_id),
                            });
                        } else {
                            passage_ids.insert(passage.passage_id.clone());
                        }
                    }

                    if passage.work_id.is_empty() {
                        errors.push(ValidationError {
                            file: "passages.jsonl".to_string(),
                            line: Some(line_num + 1),
                            code: "missing_field".to_string(),
                            message: "work_id is required".to_string(),
                        });
                    } else if !work_ids.contains(&passage.work_id) {
                        errors.push(ValidationError {
                            file: "passages.jsonl".to_string(),
                            line: Some(line_num + 1),
                            code: "invalid_work_id".to_string(),
                            message: format!(
                                "work_id '{}' not found in works.jsonl",
                                passage.work_id
                            ),
                        });
                    }

                    if passage.text.is_empty() {
                        errors.push(ValidationError {
                            file: "passages.jsonl".to_string(),
                            line: Some(line_num + 1),
                            code: "missing_field".to_string(),
                            message: "text is required".to_string(),
                        });
                    } else {
                        // Count CJK characters
                        cjk_char_count += passage.text.chars().filter(|c| is_cjk(*c)).count();

                        // Warn if text is too short or too long
                        let cjk_count = passage.text.chars().filter(|c| is_cjk(*c)).count();
                        if cjk_count < 8 {
                            warnings.push(ValidationWarning {
                                code: "short_passage".to_string(),
                                message: format!("passage_id '{}' has only {} CJK characters (recommended minimum: 8)", passage.passage_id, cjk_count),
                            });
                        } else if cjk_count > 3000 {
                            warnings.push(ValidationWarning {
                                code: "long_passage".to_string(),
                                message: format!("passage_id '{}' has {} CJK characters (recommended maximum: 3000)", passage.passage_id, cjk_count),
                            });
                        }
                    }

                    passages.push(passage);
                }
                Err(e) => {
                    errors.push(ValidationError {
                        file: "passages.jsonl".to_string(),
                        line: Some(line_num + 1),
                        code: "invalid_json".to_string(),
                        message: format!("failed to parse JSON: {}", e),
                    });
                }
            }
        }
    }

    // Check for missing dates
    if !works.is_empty() {
        missing_dates_count = works.iter().filter(|w| w.date_start.is_none()).count();
        let missing_pct = (missing_dates_count * 100) / works.len();
        if missing_pct > 50 {
            warnings.push(ValidationWarning {
                code: "missing_dates".to_string(),
                message: format!("{}% of works have no date_start/date_end; first-attestation reports will use period_rank only.", missing_pct),
            });
        }
    }

    let stats = ValidationStats {
        works: works.len(),
        passages: passages.len(),
        cjk_chars: cjk_char_count,
    };

    Ok(ValidationReport {
        schema: "gd-cef-validation-v1".to_string(),
        valid: errors.is_empty(),
        errors,
        warnings,
        stats,
    })
}

fn is_cjk(c: char) -> bool {
    // Basic CJK range detection
    match c {
        '\u{4E00}'..='\u{9FFF}' | // CJK Unified Ideographs
        '\u{3400}'..='\u{4DBF}' | // CJK Unified Ideographs Extension A
        '\u{20000}'..='\u{2A6DF}' | // CJK Unified Ideographs Extension B
        '\u{2A700}'..='\u{2B73F}' | // CJK Unified Ideographs Extension C
        '\u{2B740}'..='\u{2B81F}' | // CJK Unified Ideographs Extension D
        '\u{2B820}'..='\u{2CEAF}' | // CJK Unified Ideographs Extension E
        '\u{F900}'..='\u{FAFF}' | // CJK Compatibility Ideographs
        '\u{2F800}'..='\u{2FA1F}' => // CJK Compatibility Ideographs Supplement
            true,
        _ => false,
    }
}

pub fn init(out: PathBuf) -> Result<()> {
    fs::create_dir_all(&out)?;

    // Create corpus.toml
    let corpus_toml = r#"schema = "gd-cef-v1"
corpus_id = "my-corpus"
name = "My Corpus"
language = "zh-Hant"
script = "traditional"
snapshot_id = "2026-05-09"

description = "Description of your corpus."
source_url = ""
source_type = ""

rights_id = "unknown"
rights_notes = "Add rights information here."

default_period = "Unknown"
default_period_rank = 99
default_origin = ""
default_traditions = []

[conversion]
converter_name = "manual"
converter_version = "1.0.0"
conversion_date = "2026-05-09"
notes = "Notes about the conversion process."
"#;
    fs::write(out.join("corpus.toml"), corpus_toml)?;

    // Create works.jsonl with example
    let works_jsonl = r#"{"work_id":"work-0001","title_zh":"示例作品","title_en":"Example Work","author":"作者","period":"Tang","period_rank":1}
{"work_id":"work-0002","title_zh":"另一部作品","title_en":"Another Work"}
"#;
    fs::write(out.join("works.jsonl"), works_jsonl)?;

    // Create passages.jsonl with example
    let passages_jsonl = r#"{"passage_id":"work-0001#p000001","work_id":"work-0001","text":"子曰學而時習之不亦說乎","text_normalized":"子曰學而時習之不亦說乎"}
{"passage_id":"work-0001#p000002","work_id":"work-0001","text":"有朋自遠方來不亦樂乎"}
{"passage_id":"work-0002#p000001","work_id":"work-0002","text":"這是另一部作品的示例文本"}
"#;
    fs::write(out.join("passages.jsonl"), passages_jsonl)?;

    // Create README.md
    let readme = r#"# GraphDiscovery Corpus Exchange Format (GD-CEF)

This directory contains a corpus in the GD-CEF v1 format.

## Files

- `corpus.toml` - Corpus metadata and configuration
- `works.jsonl` - One row per work/text/book
- `passages.jsonl` - One row per searchable passage

## Usage

1. Validate the corpus:
   ```bash
   sinorag cef-validate --input /path/to/corpus
   ```

2. Ingest into GraphDiscovery:
   ```bash
   sinorag ingest-cef --input /path/to/corpus --out-parquet /path/to/passages.parquet
   ```

3. View statistics:
   ```bash
   sinorag cef-stats --input /path/to/corpus
   ```

## Format Documentation

See GD-CEF v1 specification for details on required and optional fields.
"#;
    fs::write(out.join("README.md"), readme)?;

    eprintln!("Created GD-CEF template in {}", out.display());
    eprintln!("Edit the files to add your corpus data, then run validation.");
    Ok(())
}

pub fn stats(input: PathBuf) -> Result<()> {
    let report = validate(input)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

pub async fn ingest(input: PathBuf, out_parquet: PathBuf) -> Result<()> {
    ingest_blocking(input, out_parquet).map(|_| ())
}

/// Synchronous body of [`ingest`], for callers without a Tokio runtime (the
/// Windows document-intake GUI). Returns the number of passages written.
pub fn ingest_blocking(input: PathBuf, out_parquet: PathBuf) -> Result<usize> {
    let corpus_dir = if input.is_file() {
        input
            .parent()
            .ok_or_else(|| anyhow!("Cannot determine corpus directory from file path"))?
    } else {
        &input
    };

    // Validate first
    let report = validate(input.clone())?;
    if !report.valid {
        anyhow::bail!(
            "Validation failed with {} errors. Run cef-validate for details.",
            report.errors.len()
        );
    }

    eprintln!(
        "Validation passed. Ingesting {} passages...",
        report.stats.passages
    );

    // Read corpus.toml
    let corpus_toml_path = corpus_dir.join("corpus.toml");
    let corpus_toml_content = fs::read_to_string(&corpus_toml_path)?;
    let corpus: CorpusToml = toml::from_str(&corpus_toml_content)?;

    // Read works.jsonl into a map
    let works_path = corpus_dir.join("works.jsonl");
    let works_content = fs::read_to_string(&works_path)?;
    let mut work_map: std::collections::HashMap<String, WorkRecord> =
        std::collections::HashMap::new();
    for (line_no, line) in works_content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let work = serde_json::from_str::<WorkRecord>(line)
            .with_context(|| format!("parse {} line {}", works_path.display(), line_no + 1))?;
        work_map.insert(work.work_id.clone(), work);
    }

    // Read passages.jsonl and convert to PassageRecord
    let passages_path = corpus_dir.join("passages.jsonl");
    let passages_content = fs::read_to_string(&passages_path)?;
    let mut passages: Vec<ModelPassageRecord> = Vec::new();
    let mut passage_ord: u32 = 0;

    for (line_no, line) in passages_content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let passage = serde_json::from_str::<PassageRecord>(line)
            .with_context(|| format!("parse {} line {}", passages_path.display(), line_no + 1))?;
        let work = work_map.get(&passage.work_id);

        let text_normalized = passage.text_normalized.unwrap_or_else(|| {
            if corpus.uses_word_profile() {
                crate::normalize::normalize_words(&passage.text)
            } else {
                normalize_zh(&passage.text)
            }
        });

        passages.push(
            ModelPassageRecord {
                source_corpus: corpus.corpus_id.clone(),
                source_work_id: passage.work_id.clone(),
                source_section_id: passage.section_id.clone().unwrap_or_default(),
                source_locator: passage.locator.clone().unwrap_or_default(),
                source_url: passage
                    .source_url
                    .clone()
                    .unwrap_or_else(|| corpus.source_url.clone().unwrap_or_default()),
                edition_siglum: String::new(),
                edition_label: String::new(),
                rights_id: passage
                    .rights_id
                    .clone()
                    .unwrap_or_else(|| corpus.rights_id.clone()),
                rights_notes: corpus.rights_notes.clone().unwrap_or_default(),
                retrieval_method: "gd-cef".to_string(),
                snapshot_id: corpus.snapshot_id.clone(),
                quality_flags_json: passage
                    .quality_flags
                    .clone()
                    .map(|v| serde_json::to_string(&v).unwrap())
                    .unwrap_or_default(),
                passage_id: passage.passage_id.clone(),
                source_rel_path: passage
                    .source_rel_path
                    .clone()
                    .unwrap_or_else(|| format!("{}/{}", corpus.corpus_id, passage.work_id)),
                xml_id: passage
                    .passage_id
                    .split('#')
                    .nth(1)
                    .unwrap_or("unknown")
                    .to_string(),
                div_path: String::new(),
                heading: passage.section_title.clone().unwrap_or_else(|| {
                    work.and_then(|w| Some(w.title_zh.clone()))
                        .unwrap_or_default()
                }),
                heading_path: passage.heading_path.clone().unwrap_or_default(),
                from_lb: passage.line_start.clone(),
                to_lb: passage.line_end.clone(),
                passage_ord_in_file: passage_ord,
                zh_text_raw: passage.text.clone(),
                zh_text_normalized: text_normalized,
                text_type: passage
                    .text_type
                    .clone()
                    .unwrap_or_else(|| "prose".to_string()),
                contains_person: passage.contains_person.unwrap_or(false),
                contains_term: passage.contains_term.unwrap_or(false),
                contains_foreign: false,
                canon: String::new(),
                canon_name: String::new(),
                traditions: work
                    .and_then(|w| {
                        if w.traditions.is_empty() {
                            None
                        } else {
                            Some(w.traditions.clone())
                        }
                    })
                    .unwrap_or_else(|| corpus.default_traditions.clone()),
                period: work
                    .and_then(|w| w.period.clone())
                    .unwrap_or_else(|| corpus.default_period.clone().unwrap_or_default()),
                origin: corpus.default_origin.clone().unwrap_or_default(),
                author: work.and_then(|w| w.author.clone()).unwrap_or_default(),
                main_title: work
                    .and_then(|w| Some(w.title_zh.clone()))
                    .unwrap_or_default(),
                period_rank: work
                    .and_then(|w| w.period_rank)
                    .unwrap_or_else(|| corpus.default_period_rank.unwrap_or(99)),
                zh: String::new(),
                normalized_zh: String::new(),
            }
            .finalize_aliases(),
        );

        passage_ord += 1;
    }

    // Write to Parquet
    eprintln!(
        "Writing {} passages to {}",
        passages.len(),
        out_parquet.display()
    );

    let mut batch = storage::PassageBatch::default();
    for passage in &passages {
        batch.push(passage)?;
    }
    storage::write_parquet_part_partitioned(
        &batch,
        &out_parquet,
        &corpus.corpus_id,
        0,
        crate::storage::ParquetCompression::Zstd,
    )?;

    crate::normalize::record_text_profile(
        &out_parquet,
        &corpus.corpus_id,
        corpus.uses_word_profile(),
    )
    .context("recording the corpus text profile")?;

    eprintln!(
        "Ingest complete. Wrote {} passages to {}",
        passages.len(),
        out_parquet.display()
    );
    Ok(passages.len())
}
