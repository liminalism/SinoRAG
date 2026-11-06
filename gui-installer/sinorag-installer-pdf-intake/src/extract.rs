//! File → text. PDF through the Lege pipeline, HTML through a small
//! tag-stripper, plain text through charset detection.

use anyhow::{anyhow, Context, Result};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocKind {
    Pdf,
    Html,
    Text,
}

impl DocKind {
    pub fn label(self) -> &'static str {
        match self {
            DocKind::Pdf => "pdf",
            DocKind::Html => "html",
            DocKind::Text => "text",
        }
    }

    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "pdf" => Some(DocKind::Pdf),
            "html" | "htm" | "xhtml" => Some(DocKind::Html),
            "txt" | "text" => Some(DocKind::Text),
            _ => None,
        }
    }
}

/// One addressable stretch of a document: a PDF page, or the whole body of
/// an HTML/text file (text files are split on form feeds when they have them).
#[derive(Debug, Clone)]
pub struct Section {
    /// 1-based page number for PDFs and form-feed-paged text files.
    pub page: Option<u32>,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct ExtractedDoc {
    pub kind: DocKind,
    pub title: String,
    pub sections: Vec<Section>,
    pub page_count: Option<u32>,
    /// Pages whose text came from OCR rather than the PDF's text layer.
    pub ocr_pages: u32,
    /// Per-page problems worth surfacing in the log (failed pages, empty scans).
    pub notes: Vec<String>,
    /// Charset the file was decoded with (HTML/text only).
    pub encoding: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ExtractOptions {
    /// Lege / Tesseract-style OCR language code: eng, chi_tra, chi_sim, jpn, ...
    pub ocr_language: String,
    /// Pages per Lege text request; also the progress granularity.
    pub pdf_batch_pages: u32,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            ocr_language: "eng".to_string(),
            pdf_batch_pages: 8,
        }
    }
}

/// Extract `path`. `progress(done, total)` is called while a PDF's pages are read.
pub fn extract(
    path: &Path,
    kind: DocKind,
    opts: &ExtractOptions,
    progress: &mut dyn FnMut(u32, u32),
) -> Result<ExtractedDoc> {
    match kind {
        DocKind::Pdf => extract_pdf(path, opts, progress),
        DocKind::Html => {
            let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            refuse_binary(&bytes)?;
            Ok(extract_html_bytes(&bytes, &file_stem(path)))
        }
        DocKind::Text => {
            let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            refuse_binary(&bytes)?;
            Ok(extract_text_bytes(&bytes, &file_stem(path)))
        }
    }
}

/// A file named `.txt` or `.html` that is something else. Decoded as text it
/// would still produce "passages": thousands of them, all noise, which then
/// rank in every search. So say what the file is and index nothing.
fn refuse_binary(bytes: &[u8]) -> Result<()> {
    match binary_kind(bytes) {
        None => Ok(()),
        Some("a PDF") => Err(anyhow!(
            "this is a PDF with a text file's name. Rename it to end in .pdf and add it again."
        )),
        Some(what) => Err(anyhow!(
            "this is {what}, not text. Save it as PDF or plain text from the program that made it, \
             and add that file instead."
        )),
    }
}

/// What `bytes` are when they are not text: a known file signature, or
/// content too full of control characters to be writing in any encoding.
pub fn binary_kind(bytes: &[u8]) -> Option<&'static str> {
    const SIGNATURES: &[(&[u8], &str)] = &[
        (b"%PDF-", "a PDF"),
        (b"PK\x03\x04", "a Word, Excel, EPUB or ZIP file"),
        (b"\xD0\xCF\x11\xE0", "an older Word or Excel file"),
        (b"{\\rtf", "a Rich Text (RTF) document"),
        (b"\x89PNG", "an image"),
        (b"\xFF\xD8\xFF", "an image"),
        (b"GIF8", "an image"),
        (b"7z\xBC\xAF", "a compressed archive"),
        (b"\x1F\x8B", "a compressed archive"),
        (b"Rar!", "a compressed archive"),
    ];
    if let Some((_, what)) = SIGNATURES.iter().find(|(magic, _)| bytes.starts_with(magic)) {
        return Some(what);
    }
    // Judge the text as it will be indexed, so UTF-16 (half NUL bytes) and
    // ISO-2022-JP (escape sequences) are seen as the text they are.
    let sample = &bytes[..bytes.len().min(64 * 1024)];
    let (text, _) = decode_text(sample, None);
    let total = text.chars().count();
    let control = text
        .chars()
        .filter(|c| c.is_control() && !matches!(c, '\t' | '\n' | '\r' | '\u{000C}'))
        .count();
    // Text has none, bar a stray one; compressed or compiled data has about
    // one in ten.
    (control >= 8 && control * 50 > total).then_some("a binary file")
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Untitled".to_string())
}

// ── PDF ──────────────────────────────────────────────────────────────────────

fn extract_pdf(
    path: &Path,
    opts: &ExtractOptions,
    progress: &mut dyn FnMut(u32, u32),
) -> Result<ExtractedDoc> {
    use lege_pdf_agent::commands::text::{text_pages, OcrMode, TextLayout, TextRequest};

    let (_identity, snapshot) = lege_pdf_agent::open_document(path, None)
        .map_err(|e| anyhow!("{e:#}"))
        .with_context(|| format!("opening PDF {}", path.display()))?;
    let page_count = snapshot.page_count() as u32;

    let request = TextRequest {
        layout: TextLayout::Plain,
        bbox: None,
        options: pdf_text::TextPageOptions::default(),
        ocr: OcrMode::Auto,
        ocr_language: &opts.ocr_language,
        // No page, byte or time limits: intake wants every page in full.
        bounds: lege_pdf_agent::Bounds {
            max_pages: 0,
            max_bytes: 0,
            timeout_secs: 0,
            ..Default::default()
        },
        max_chars: 0,
        start_char: 0,
    };

    let mut sections = Vec::with_capacity(page_count as usize);
    let mut notes = Vec::new();
    let mut ocr_pages = 0u32;
    let batch = opts.pdf_batch_pages.max(1);
    let mut first = 1u32;
    progress(0, page_count);
    while first <= page_count {
        let last = (first + batch - 1).min(page_count);
        let range = format!("{first}-{last}");
        let result = text_pages(&snapshot, Some(&range), &request)
            .map_err(|e| anyhow!("{e:#}"))
            .with_context(|| format!("reading pages {range}"))?;
        for warning in result["warnings"].as_array().into_iter().flatten() {
            if let Some(w) = warning.as_str() {
                notes.push(format!("pages {range}: {w}"));
            }
        }
        for entry in result["pages"].as_array().into_iter().flatten() {
            let page = entry["page"].as_u64().unwrap_or(0) as u32;
            if let Some(error) = entry["error"].as_str() {
                notes.push(format!("page {page}: {error}"));
                continue;
            }
            let data = &entry["data"];
            let mut text = data["text"].as_str().unwrap_or_default().replace("\r\n", "\n");
            let mut by_ocr = data["provenance"]["source"] == "ocr";
            // Lege's `auto` trust check accepts some garbled hidden text layers
            // (e.g. scans whose invisible OCR layer uses fonts without a usable
            // ToUnicode map). When the native text reads as symbol soup, OCR
            // the page outright and keep whichever result reads better.
            if !by_ocr && !text.trim().is_empty() && text_quality(&text) < GARBLED_THRESHOLD {
                let forced = TextRequest {
                    ocr: OcrMode::Always,
                    ..request
                };
                let page_range = page.to_string();
                match text_pages(&snapshot, Some(&page_range), &forced) {
                    Ok(r) => {
                        let ocr_text = r["pages"][0]["data"]["text"]
                            .as_str()
                            .unwrap_or_default()
                            .replace("\r\n", "\n");
                        if text_quality(&ocr_text) > text_quality(&text) {
                            notes.push(format!("page {page}: text layer looked garbled; used OCR instead"));
                            text = ocr_text;
                            by_ocr = true;
                        }
                    }
                    Err(e) => notes.push(format!("page {page}: OCR retry failed: {e:#}")),
                }
            }
            if by_ocr {
                ocr_pages += 1;
            }
            if text.trim().is_empty() {
                if data["provenance"]["page_content_kind"] == "scanned-image" {
                    notes.push(format!(
                        "page {page}: scanned image with no OCR text (is the OCR language pack installed?)"
                    ));
                }
                continue;
            }
            sections.push(Section {
                page: Some(page),
                text,
            });
        }
        progress(last, page_count);
        first = last + 1;
    }

    Ok(ExtractedDoc {
        kind: DocKind::Pdf,
        title: file_stem(path),
        sections,
        page_count: Some(page_count),
        ocr_pages,
        notes,
        encoding: None,
    })
}

/// Below this, a page's native text is treated as garbled and re-read by OCR.
const GARBLED_THRESHOLD: f32 = 0.6;

/// Rough readability score in 0..=1: the share of non-space characters that
/// are letters, digits or CJK (as opposed to symbol soup like `#$&=%`),
/// scaled down when alphabetic "words" are mostly vowel-less fragments.
pub fn text_quality(text: &str) -> f32 {
    let mut total = 0usize;
    let mut good = 0usize;
    for c in text.chars().filter(|c| !c.is_whitespace()) {
        total += 1;
        if c.is_alphanumeric() || is_cjk(c) || matches!(c, '.' | ',' | ';' | ':' | '\'' | '"' | '(' | ')' | '-' | '’' | '“' | '”' | '，' | '。' | '、') {
            good += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    let char_score = good as f32 / total as f32;

    // Latin-script check: real words of 3+ letters nearly always carry a vowel.
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| w.chars().count() >= 3 && w.chars().all(|c| c.is_ascii_alphabetic()))
        .collect();
    if words.len() >= 10 {
        let voweled = words
            .iter()
            .filter(|w| w.chars().any(|c| "aeiouyAEIOUY".contains(c)))
            .count();
        let word_score = voweled as f32 / words.len() as f32;
        return char_score.min(word_score + 0.1);
    }
    char_score
}

// ── Plain text ───────────────────────────────────────────────────────────────

pub fn extract_text_bytes(bytes: &[u8], fallback_title: &str) -> ExtractedDoc {
    let (text, encoding) = decode_text(bytes, None);
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let pages: Vec<&str> = text.split('\u{000C}').collect();
    let paged = pages.len() > 1;
    let sections = pages
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.trim().is_empty())
        .map(|(i, t)| Section {
            page: paged.then_some(i as u32 + 1),
            text: (*t).to_string(),
        })
        .collect();
    ExtractedDoc {
        kind: DocKind::Text,
        title: fallback_title.to_string(),
        sections,
        page_count: paged.then_some(pages.len() as u32),
        ocr_pages: 0,
        notes: Vec::new(),
        encoding: Some(encoding),
    }
}

/// Decode bytes of unknown charset. Order: BOM, declared charset, UTF-16
/// without BOM (NUL-heavy), UTF-8, then a statistical guess among the legacy
/// encodings (GBK, Big5, Shift_JIS, EUC-JP, EUC-KR, the Windows code pages).
pub fn decode_text(bytes: &[u8], declared: Option<&str>) -> (String, String) {
    use encoding_rs::{Encoding, UTF_16BE, UTF_16LE, UTF_8};

    if let Some((enc, bom_len)) = Encoding::for_bom(bytes) {
        let (text, _) = enc.decode_without_bom_handling(&bytes[bom_len..]);
        return (text.into_owned(), enc.name().to_string());
    }
    if let Some(enc) = declared.and_then(|l| Encoding::for_label(l.trim().as_bytes())) {
        // A page declaring UTF-16 in a meta tag is lying (the tag was readable as ASCII).
        if enc != UTF_16LE && enc != UTF_16BE {
            let (text, _) = enc.decode_without_bom_handling(bytes);
            return (text.into_owned(), enc.name().to_string());
        }
    }
    let sample = &bytes[..bytes.len().min(4096)];
    if sample.len() >= 4 {
        let even_nuls = sample.iter().step_by(2).filter(|b| **b == 0).count();
        let odd_nuls = sample.iter().skip(1).step_by(2).filter(|b| **b == 0).count();
        let half = sample.len() / 2;
        if odd_nuls * 3 > half {
            return (UTF_16LE.decode_without_bom_handling(bytes).0.into_owned(), "UTF-16LE".into());
        }
        if even_nuls * 3 > half {
            return (UTF_16BE.decode_without_bom_handling(bytes).0.into_owned(), "UTF-16BE".into());
        }
    }
    if let Ok(text) = std::str::from_utf8(bytes) {
        return (text.to_string(), UTF_8.name().to_string());
    }
    // Not UTF-8. Validity alone cannot tell the legacy CJK encodings apart
    // (nearly every Big5 file is also a valid GBK byte sequence, and the
    // other way round), so let the detector weigh character frequencies.
    // ISO-2022-JP is allowed: these are local files, not pages that run scripts.
    let mut detector = chardetng::EncodingDetector::new(chardetng::Iso2022JpDetection::Allow);
    detector.feed(bytes, true);
    let enc = detector.guess(None, chardetng::Utf8Detection::Deny);
    (
        enc.decode_without_bom_handling(bytes).0.into_owned(),
        enc.name().to_string(),
    )
}

pub fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{3040}'..='\u{30ff}'   // kana
        | '\u{3400}'..='\u{4dbf}'
        | '\u{4e00}'..='\u{9fff}'
        | '\u{f900}'..='\u{faff}'
        | '\u{20000}'..='\u{2ffff}'
        | '\u{ac00}'..='\u{d7af}' // hangul
    )
}

// ── HTML ─────────────────────────────────────────────────────────────────────

pub fn extract_html_bytes(bytes: &[u8], fallback_title: &str) -> ExtractedDoc {
    let declared = sniff_html_charset(bytes);
    let (html, encoding) = decode_text(bytes, declared.as_deref());
    let (title, body) = html_to_text(&html);
    let title = title
        .filter(|t| !t.trim().is_empty())
        .unwrap_or_else(|| fallback_title.to_string());
    let sections = if body.trim().is_empty() {
        Vec::new()
    } else {
        vec![Section { page: None, text: body }]
    };
    ExtractedDoc {
        kind: DocKind::Html,
        title,
        sections,
        page_count: None,
        ocr_pages: 0,
        notes: Vec::new(),
        encoding: Some(encoding),
    }
}

/// `<meta charset="x">` or `<meta http-equiv=... content="...; charset=x">`
/// in the first 4 KB, read as ASCII.
fn sniff_html_charset(bytes: &[u8]) -> Option<String> {
    let head = &bytes[..bytes.len().min(4096)];
    let lower: String = head
        .iter()
        .map(|b| if b.is_ascii() { (*b as char).to_ascii_lowercase() } else { ' ' })
        .collect();
    let at = lower.find("charset=")?;
    let rest = lower[at + 8..].trim_start_matches(['"', '\'', ' ']);
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_'))
        .unwrap_or(rest.len());
    let label = &rest[..end];
    (!label.is_empty()).then(|| label.to_string())
}

/// Strip markup to readable text. Returns (title, body).
pub fn html_to_text(html: &str) -> (Option<String>, String) {
    // Elements dropped with everything inside them. `head` is not listed:
    // its closing tag is optional, and the only text it can hold is in
    // `title`, `script` and `style`, which are.
    const SKIP: &[&str] = &[
        "title", "script", "style", "noscript", "template", "svg", "iframe", "object",
    ];
    const BLOCK: &[&str] = &[
        "p", "div", "section", "article", "main", "header", "footer", "aside", "nav", "blockquote",
        "pre", "ul", "ol", "li", "dl", "dt", "dd", "table", "tr", "h1", "h2", "h3", "h4", "h5",
        "h6", "hr", "figure", "figcaption", "address", "form", "fieldset", "details", "summary",
        "caption", "thead", "tbody", "tfoot",
    ];

    let title = find_ci(html, "<title").and_then(|a| {
        let open_end = a + html[a..].find('>')? + 1;
        let close = open_end + find_ci(&html[open_end..], "</title")?;
        let t = collapse_ws(&decode_entities(&html[open_end..close]));
        (!t.is_empty()).then_some(t)
    });

    let mut out = String::with_capacity(html.len() / 2);
    let mut rest = html;

    while !rest.is_empty() {
        let lt = rest.find('<').unwrap_or(rest.len());
        out.push_str(&rest[..lt]);
        rest = &rest[lt..];
        if rest.is_empty() {
            break;
        }
        // Comments, doctype, CDATA, processing instructions.
        if rest.starts_with("<!--") {
            rest = rest.find("-->").map_or("", |e| &rest[e + 3..]);
            continue;
        }
        if rest.starts_with("<!") || rest.starts_with("<?") {
            rest = rest.find('>').map_or("", |e| &rest[e + 1..]);
            continue;
        }
        // A tag name starts with a letter. Anything else ("a < b", "1<2")
        // is text: keep the '<'.
        let after = rest[1..].strip_prefix('/').unwrap_or(&rest[1..]);
        if !after.starts_with(|c: char| c.is_ascii_alphabetic()) {
            out.push('<');
            rest = &rest[1..];
            continue;
        }
        let Some(gt) = find_tag_end(rest) else {
            break;
        };
        let tag = &rest[1..gt];
        let closing = tag.starts_with('/');
        let name = after
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        rest = &rest[gt + 1..];

        if !closing && SKIP.contains(&name.as_str()) && !tag.trim_end().ends_with('/') {
            // Jump straight to the closing tag. What is in between is not
            // parsed: a script or style body is not markup, and a `<` or a
            // quote in it ("i<n", "don't") would otherwise be read as a tag
            // that swallows the closing tag and the page after it.
            rest = after_closing_tag(rest, &name);
            continue;
        }
        match name.as_str() {
            "br" => out.push('\n'),
            "td" | "th" if !closing => out.push('\t'),
            n if BLOCK.contains(&n) => out.push_str("\n\n"),
            _ => {}
        }
    }

    (title, tidy_lines(&decode_entities(&out)))
}

/// The input after the `</name>` that closes a skipped element, or nothing
/// when the element is never closed.
fn after_closing_tag<'a>(mut rest: &'a str, name: &str) -> &'a str {
    let needle = format!("</{name}");
    while let Some(at) = find_ci(rest, &needle) {
        rest = &rest[at + needle.len()..];
        // `</scripts` or `</style-guide` is some other tag.
        if !rest.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '-') {
            return rest.find('>').map_or("", |gt| &rest[gt + 1..]);
        }
    }
    ""
}

/// End of a tag, skipping '>' inside quoted attribute values.
fn find_tag_end(s: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    for (i, c) in s.char_indices().skip(1) {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '"') | (None, '\'') => quote = Some(c),
            (None, '>') => return Some(i),
            _ => {}
        }
    }
    None
}

fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.len() > h.len() {
        return None;
    }
    (0..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Soft hyphen, zero-width space / non-joiner / joiner, and the direction marks.
const INVISIBLE: [char; 6] = ['\u{ad}', '\u{200b}', '\u{200c}', '\u{200d}', '\u{200e}', '\u{200f}'];

/// Collapse runs of horizontal whitespace, trim lines, keep at most one blank line.
fn tidy_lines(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blank_run = 0;
    for line in s.lines() {
        // Soft hyphens and zero-width marks are invisible in a browser and
        // would split a word for search. The other Unicode spaces fold to one
        // (the ideographic space is left for the CJK normalizer).
        let line = line.replace(INVISIBLE, "");
        let line = line
            .split(|c: char| c.is_whitespace() && c != '\u{3000}')
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if line.is_empty() {
            blank_run += 1;
            if blank_run == 1 && !out.is_empty() {
                out.push('\n');
            }
            continue;
        }
        blank_run = 0;
        out.push_str(&line);
        out.push('\n');
    }
    out.trim().to_string()
}

pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let end = rest[1..].find(';').map(|e| e + 1).filter(|e| *e <= 12);
        let Some(end) = end else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let entity = &rest[1..end];
        let decoded = if let Some(num) = entity.strip_prefix('#') {
            let code = if let Some(hex) = num.strip_prefix(['x', 'X']) {
                u32::from_str_radix(hex, 16).ok()
            } else {
                num.parse::<u32>().ok()
            };
            code.and_then(numeric_reference)
        } else {
            named_entity(entity)
        };
        match decoded {
            Some(d) => {
                out.push(d);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// A numeric character reference. 128–159 are C1 controls in Unicode, but
/// pages that use them mean the Windows-1252 characters (`&#150;` is an en
/// dash), which is also how browsers read them.
fn numeric_reference(code: u32) -> Option<char> {
    if (0x80..=0x9f).contains(&code) {
        let byte = [code as u8];
        let (text, _) = encoding_rs::WINDOWS_1252.decode_without_bom_handling(&byte);
        return text.chars().next();
    }
    char::from_u32(code).filter(|c| !c.is_control())
}

/// HTML 4 Latin-1 entities, in code point order from U+00A0.
const LATIN1_ENTITIES: [&str; 96] = [
    "nbsp", "iexcl", "cent", "pound", "curren", "yen", "brvbar", "sect", "uml", "copy", "ordf",
    "laquo", "not", "shy", "reg", "macr", "deg", "plusmn", "sup2", "sup3", "acute", "micro",
    "para", "middot", "cedil", "sup1", "ordm", "raquo", "frac14", "frac12", "frac34", "iquest",
    "Agrave", "Aacute", "Acirc", "Atilde", "Auml", "Aring", "AElig", "Ccedil", "Egrave",
    "Eacute", "Ecirc", "Euml", "Igrave", "Iacute", "Icirc", "Iuml", "ETH", "Ntilde", "Ograve",
    "Oacute", "Ocirc", "Otilde", "Ouml", "times", "Oslash", "Ugrave", "Uacute", "Ucirc", "Uuml",
    "Yacute", "THORN", "szlig", "agrave", "aacute", "acirc", "atilde", "auml", "aring", "aelig",
    "ccedil", "egrave", "eacute", "ecirc", "euml", "igrave", "iacute", "icirc", "iuml", "eth",
    "ntilde", "ograve", "oacute", "ocirc", "otilde", "ouml", "divide", "oslash", "ugrave",
    "uacute", "ucirc", "uuml", "yacute", "thorn", "yuml",
];

/// Greek letter entities, in code point order from U+03B1; the capitals
/// ("Alpha") run in the same order from U+0391. "sigmaf" has no capital.
const GREEK_ENTITIES: [&str; 25] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigmaf", "sigma", "tau", "upsilon",
    "phi", "chi", "psi", "omega",
];

fn named_entity(name: &str) -> Option<char> {
    let special = match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "OElig" => 'Œ',
        "oelig" => 'œ',
        "Scaron" => 'Š',
        "scaron" => 'š',
        "Yuml" => 'Ÿ',
        "fnof" => 'ƒ',
        "circ" => 'ˆ',
        "tilde" => '˜',
        "ensp" => '\u{2002}',
        "emsp" => '\u{2003}',
        "thinsp" => '\u{2009}',
        "zwnj" => '\u{200c}',
        "zwj" => '\u{200d}',
        "lrm" => '\u{200e}',
        "rlm" => '\u{200f}',
        "ndash" => '–',
        "mdash" => '—',
        "lsquo" => '‘',
        "rsquo" => '’',
        "sbquo" => '‚',
        "ldquo" => '“',
        "rdquo" => '”',
        "bdquo" => '„',
        "dagger" => '†',
        "Dagger" => '‡',
        "bull" => '•',
        "hellip" => '…',
        "permil" => '‰',
        "prime" => '′',
        "Prime" => '″',
        "lsaquo" => '‹',
        "rsaquo" => '›',
        "euro" => '€',
        "trade" => '™',
        "larr" => '←',
        "uarr" => '↑',
        "rarr" => '→',
        "darr" => '↓',
        "harr" => '↔',
        "minus" => '−',
        "infin" => '∞',
        "asymp" => '≈',
        "ne" => '≠',
        "le" => '≤',
        "ge" => '≥',
        _ => '\0',
    };
    if special != '\0' {
        return Some(special);
    }
    if let Some(i) = LATIN1_ENTITIES.iter().position(|n| *n == name) {
        return char::from_u32(0xa0 + i as u32);
    }
    let first = name.chars().next()?;
    let capital = first.is_ascii_uppercase();
    let lower = format!("{}{}", first.to_ascii_lowercase(), &name[first.len_utf8()..]);
    let i = GREEK_ENTITIES.iter().position(|n| *n == lower)?;
    if capital && lower == "sigmaf" {
        return None;
    }
    char::from_u32(if capital { 0x391 } else { 0x3b1 } + i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_strips_markup_and_reads_title() {
        let html = r#"<!DOCTYPE html><html><head><meta charset="utf-8"><title>My &amp; Page</title>
            <style>p{color:red}</style><script>var x = "<p>";</script></head>
            <body><h1>Heading</h1><p>First&nbsp;para with <b>bold</b> text.</p>
            <!-- hidden --><p>Second<br>line &#x4E2D;&#25991;</p></body></html>"#;
        let (title, body) = html_to_text(html);
        assert_eq!(title.as_deref(), Some("My & Page"));
        assert!(body.contains("Heading"));
        assert!(body.contains("First para with bold text."));
        assert!(body.contains("Second\nline 中文"));
        assert!(!body.contains("color"));
        assert!(!body.contains("var x"));
        assert!(!body.contains("hidden"));
    }

    #[test]
    fn binary_files_with_a_text_name_are_refused() {
        assert_eq!(binary_kind(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n1 0 obj"), Some("a PDF"));
        assert_eq!(binary_kind(b"PK\x03\x04\x14\x00\x06\x00"), Some("a Word, Excel, EPUB or ZIP file"));
        assert_eq!(binary_kind(b"{\\rtf1\\ansi Hello"), Some("a Rich Text (RTF) document"));
        // No signature, but nothing a person wrote either.
        let noise: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        assert_eq!(binary_kind(&noise), Some("a binary file"));

        // Text in every encoding intake reads is let through.
        assert_eq!(binary_kind(b""), None);
        assert_eq!(binary_kind("Plain text.\r\n\tIndented, with a form feed\u{000C}and 中文.".as_bytes()), None);
        let utf16: Vec<u8> = "Sixteen-bit text without a byte-order mark.\n"
            .encode_utf16()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(binary_kind(&utf16), None);
        let (big5, _, _) = encoding_rs::BIG5.encode("如是我聞。一時佛在舍衛國祇樹給孤獨園。\n");
        assert_eq!(binary_kind(&big5), None);
        let (jis, _, _) = encoding_rs::ISO_2022_JP.encode("日本語のテキストです。\n");
        assert_eq!(binary_kind(&jis), None);
        // One stray control character does not condemn a document.
        assert_eq!(binary_kind(b"A page\x01 with a stray byte, and plenty of ordinary text after it."), None);

        let dir = std::env::temp_dir().join(format!("sinorag-intake-binary-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.txt");
        std::fs::write(&path, b"%PDF-1.7\n").unwrap();
        let err = extract(&path, DocKind::Text, &ExtractOptions::default(), &mut |_, _| {}).unwrap_err();
        assert!(format!("{err:#}").contains("Rename it to end in .pdf"), "{err:#}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quality_separates_prose_from_symbol_soup() {
        let prose = "I write in response to your request that LizardTech amend certain language found in the source code.";
        let cjk = "如是我聞。一時佛在舍衛國祇樹給孤獨園，與大比丘眾千二百五十人俱。";
        let soup = r#"=U$H&)%A$   2* !’ 55 ## *##    2 9     $" C 66   :  (= H HH# )  2JF ; # $$5# <I"#;
        assert!(text_quality(prose) > GARBLED_THRESHOLD);
        assert!(text_quality(cjk) > GARBLED_THRESHOLD);
        assert!(text_quality(soup) < GARBLED_THRESHOLD);
    }

    const SUTRA_TRAD: &str = "如是我聞。一時佛在舍衛國祇樹給孤獨園，與大比丘眾千二百五十人俱。\
        爾時世尊食時，著衣持缽，入舍衛大城乞食。於其城中次第乞已，還至本處。";
    const SUTRA_SIMP: &str = "如是我闻。一时佛在舍卫国祇树给孤独园，与大比丘众千二百五十人俱。\
        尔时世尊食时，着衣持钵，入舍卫大城乞食。于其城中次第乞已，还至本处。";

    #[test]
    fn decodes_gbk_text() {
        let (bytes, _, bad) = encoding_rs::GB18030.encode(SUTRA_SIMP);
        assert!(!bad);
        assert_eq!(decode_text(&bytes, None).0, SUTRA_SIMP);
        // Traditional characters in a GBK file, as in older mainland editions.
        let (bytes, _, _) = encoding_rs::GB18030.encode(SUTRA_TRAD);
        assert_eq!(decode_text(&bytes, None).0, SUTRA_TRAD);
    }

    #[test]
    fn decodes_big5_text() {
        let (bytes, _, bad) = encoding_rs::BIG5.encode(SUTRA_TRAD);
        assert!(!bad);
        let (text, enc) = decode_text(&bytes, None);
        assert_eq!(text, SUTRA_TRAD);
        assert_eq!(enc, "Big5");
    }

    #[test]
    fn decodes_shift_jis_and_windows_1252_text() {
        let ja = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。";
        let (bytes, _, _) = encoding_rs::SHIFT_JIS.encode(ja);
        assert_eq!(decode_text(&bytes, None).0, ja);
        let de = "Die Kühlmittelpumpe läuft während des Betriebs ständig; größere Störungen sind selten.";
        let (bytes, _, _) = encoding_rs::WINDOWS_1252.encode(de);
        assert_eq!(decode_text(&bytes, None).0, de);
    }

    #[test]
    fn html_survives_script_and_style_bodies() {
        // "i<n" and the apostrophe used to be read as a tag that swallowed
        // </script> and everything after it.
        let html = "<html><head><title>T</title><script>for(var i=0;i<n;i++){s=\"don't\";}</script>\
            <style>a>b{color:red}</style><body><p>Before</p>\
            <script>if (a<b) { x = 1; }</script><p>After the script.</p></body></html>";
        let (title, body) = html_to_text(html);
        assert_eq!(title.as_deref(), Some("T"));
        assert_eq!(body, "Before\n\nAfter the script.");
    }

    #[test]
    fn html_head_without_closing_tag_keeps_the_body() {
        let (title, body) = html_to_text("<head><title>T</title><meta name=x><body><p>Text</p>");
        assert_eq!(title.as_deref(), Some("T"));
        assert_eq!(body, "Text");
    }

    #[test]
    fn html_keeps_a_bare_less_than_sign() {
        let (_, body) = html_to_text("<p>if 1<2 and 3>2 then x < y</p>");
        assert_eq!(body, "if 1<2 and 3>2 then x < y");
    }

    #[test]
    fn html_decodes_accented_and_numeric_entities() {
        let (_, body) = html_to_text(
            "<p>caf&eacute; M&uuml;nchen &Alpha;&omega; 5&nbsp;&euro; a&#150;b &#x4E2D; \
             soft&shy;hyphen &bogus; AT&T</p>",
        );
        assert_eq!(body, "café München Αω 5 € a–b 中 softhyphen &bogus; AT&T");
    }

    #[test]
    fn decodes_utf16le_bom() {
        let mut bytes = vec![0xFF, 0xFE];
        for u in "hello".encode_utf16() {
            bytes.extend_from_slice(&u.to_le_bytes());
        }
        assert_eq!(decode_text(&bytes, None).0, "hello");
    }

    #[test]
    fn text_splits_on_form_feed() {
        let doc = extract_text_bytes(b"page one\x0cpage two", "t");
        assert_eq!(doc.sections.len(), 2);
        assert_eq!(doc.sections[1].page, Some(2));
    }
}
