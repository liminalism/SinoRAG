---

## Searching English and other alphabetic-script text

This data root includes documents indexed with word-aware search (for
example PDFs, web pages and text files added through SinoRAG Document
Intake). For those passages:

- Phrase tools (`search`, `evidence-search`, `batch-evidence-search`,
  `citation-verify`, `pair-appearance`) match **whole words, ignoring case and
  punctuation**: `coolant pump` finds "Coolant-Pump", but `pump` does not
  find "pumps" and `cat` does not find "concatenate". There is no stemming —
  search singular and plural, or other word forms, as separate phrases
  (`batch-evidence-search` does this in one call).
- Keep phrases short and distinctive (2–4 words). A long sentence only
  matches if the document has it word for word.
- `similar`, `frontier` and `source-investigate` rank by shared words and
  word pairs, so they work as "more like this passage" for English text.
- `zh_text_raw` / `zh_quote` fields hold the passage text whatever its
  language; the `zh_` prefix is historical.
- Passage IDs look like `doc-<hash>#p0012.02` (page 12, second chunk of a
  PDF) or `doc-<hash>#s0003` (HTML/text). `source_locator` gives the page;
  `source_url` points at the original file. Cite the passage ID and page.
- Chinese/Japanese/Korean text in the same collection is searched exactly as
  in the CBETA corpus (character-based, no word boundaries).
