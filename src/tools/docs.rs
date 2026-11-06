use serde_json::{json, Value};

pub fn docs_payload(tool: Option<&str>) -> Value {
    if let Some(name) = tool {
        if let Some(doc) = DOCS.iter().find(|doc| doc.name == name) {
            return doc.to_value();
        }
        return json!({
            "error": "unknown_tool",
            "tool": name,
            "available": DOCS.iter().map(|doc| doc.name).collect::<Vec<_>>(),
        });
    }

    json!({
        "overview": "SinoRAG tools are JSON commands intended for agents and scripts. Start with status, then use evidence-search for exact evidence, pair-appearance for specified term-pair evidence, frontier for seed-passage discovery, source-read for continuous close reading, and passage/expand-context-adaptive for targeted context.",
        "workflow": [
            "status: verify which indexes are available in the pack.",
            "evidence-search: exact phrase evidence with optional attestation/history/usage summaries.",
            "pair-appearance: direct specified term1+term2 co-appearance evidence by passage, window, or sentence.",
            "frontier: expand a known seed passage into distinctive phrase leads and TF-IDF lexical parallels.",
            "hybrid-discover: compactly merge vector semantic neighbors and TF-IDF lexical parallels when vector candidates are specifically useful.",
            "search: quick exact phrase lookup across loaded passage text.",
            "heading-search: find section or heading names before searching inside a scope.",
            "cluster-hits: answer where the phrase occurs by work or division.",
            "trace-term-usage: answer how usage is distributed by period, canon, author, or work.",
            "source-read: read a source sequentially in cursor-based chunks after search/discovery identifies a work or passage.",
            "passage or expand-context-adaptive: retrieve compact or adaptive context for selected hits.",
            "batch-evidence-search: search independent phrase lists concurrently instead of issuing repeated search calls.",
            "report-build/report-from-evidence then pdf-build: create structured dossiers, or pass raw prose directly to pdf-build with its markdown field."
        ],
        "important_distinctions": {
            "trace-term-usage_vs_cluster-hits": "trace-term-usage is for analytical distribution by metadata buckets such as period, canon, author, or work. cluster-hits is for navigational clustering by catalog outline nodes such as work or division, with representative passages.",
            "search_vs_outline-search": "search is corpus-wide by default and returns direct passage hits. outline-search searches within a catalog node when scoped, or groups corpus-wide hits when no scope is supplied.",
            "heading-search_vs_search": "heading-search matches heading and heading_path metadata, plus normalized passage text as a fallback. search matches phrase text in passages.",
            "pair-appearance_vs_collocation-search": "pair-appearance verifies a specified pair of terms in the same passage/window/sentence. collocation-search discovers likely nearby terms from one seed phrase and does not prove a requested pair.",
            "pair-appearance_vs_pair-profile": "pair-appearance returns individual passages where two terms co-appear (evidence retrieval). pair-profile aggregates co-occurrence rates across periods, canons, or works (statistical analysis).",
            "person-resolve_vs_person-history": "person-resolve confirms name forms are in the corpus and shows a sample. person-history returns all passages ordered by period with mention-class labels.",
            "citation-verify_use": "citation-verify is for provenance verification: does this exact text appear in the scoped corpus? When the exact quote is not found it returns near-matches by character overlap."
            ,"frontier_vs_hybrid-discover": "frontier is the default second-stage discovery tool after exact hits; it does not use vector.index. hybrid-discover is for explicit semantic-vector candidate expansion and returns compact output unless verbosity=full."
            ,"vector_vs_evidence": "vector-neighbors and hybrid-discover produce semantic discovery candidates. Use evidence-search, phrase tools, and close reading before treating a candidate as evidence."
        },
        "task_routing": {
            "exact phrase occurrence": "evidence-search",
            "first attestation": "evidence-search",
            "absence or negative evidence": "evidence-search",
            "term pair co-occurrence (passages)": "pair-appearance",
            "term pair co-occurrence rates by period/canon": "pair-profile",
            "person mentions": "person-resolve then person-history",
            "verify a citation or quote": "citation-verify",
            "related passages": "frontier",
            "continuous source reading": "source-read",
            "read around a passage": "source-read",
            "one passage investigation": "source-investigate",
            "scope comparison": "scope-profile",
            "artifact generation": "report-from-evidence",
            "pdf generation": "pdf-build"
        },
        "tools": DOCS.iter().map(ToolDoc::to_value).collect::<Vec<_>>(),
    })
}

pub fn doc_for_tool(name: &str) -> Option<Value> {
    DOCS.iter()
        .find(|doc| doc.name == name)
        .map(ToolDoc::to_value)
}

struct ToolDoc {
    name: &'static str,
    purpose: &'static str,
    use_when: &'static str,
    notes: &'static str,
}

impl ToolDoc {
    fn to_value(&self) -> Value {
        json!({
            "name": self.name,
            "purpose": self.purpose,
            "use_when": self.use_when,
            "notes": self.notes,
        })
    }
}

const DOCS: &[ToolDoc] = &[
    ToolDoc { name: "status", purpose: "Report available corpus resources and indexes.", use_when: "Run first to see whether passages, phrase index, catalog, doc table, TF-IDF, and registry exist.", notes: "Read-only and cheap." },
    ToolDoc { name: "tool-docs", purpose: "Return this built-in documentation.", use_when: "Use when choosing a tool or explaining command differences.", notes: "Pass {\"tool\":\"search\"} for one tool or {} for the full guide." },
    ToolDoc { name: "search", purpose: "Quick corpus-wide exact phrase lookup.", use_when: "Use for ordinary text search across every loaded passage. Add mode=clusters, trace, or all for grouped summaries.", notes: "Layered: phrase index when available, parquet verification, parquet scan fallback. brief=true suppresses verbose representative metadata." },
    ToolDoc { name: "heading-search", purpose: "Find headings, section names, and heading paths.", use_when: "Use when the query is a title, case heading, section label, or when you need a work/section scope before text search.", notes: "Works with passages.parquet alone; catalog indexes are not required." },
    ToolDoc { name: "passage", purpose: "Retrieve one passage by passage_id.", use_when: "Use after search or cluster tools identify an exact passage.", notes: "Returns compact passage text and basic work metadata." },
    ToolDoc { name: "source-read", purpose: "Read an ordered source stream in stable, cursor-based chunks.", use_when: "Use when search/discovery has identified a work or passage and the agent needs to inspect the source continuously instead of piecemeal snippets.", notes: "Pass one compact anchor: source_work_id starts a work, passage_id reads around that hit, and cursor continues forward. Passage IDs retain their #anchor, which is safe JSON. Returns citeable main text plus overlap/preview segments and location metadata." },
    ToolDoc { name: "canonical-source", purpose: "Find canon-side source passages for a phrase.", use_when: "Use for source verification, citation dependence, and sutra-side candidates.", notes: "Filters toward rows with canon metadata." },
    ToolDoc { name: "validate-adjudication", purpose: "Validate a finished adjudication JSON file's structure.", use_when: "A late-pipeline check: write your adjudication after investigating, then validate it before graph-build/report-build. Not a precondition for exploration.", notes: "Checks structure, not scholarly correctness." },
    ToolDoc { name: "graph-build", purpose: "Build an evidence graph from a finished adjudication JSON file.", use_when: "Use once your adjudication is validated and you need graph artifacts — a terminal artifact-building step, not a starting point.", notes: "Writes output files." },
    ToolDoc { name: "report-build", purpose: "Assemble a markdown report from finished adjudication and graph files.", use_when: "Use when producing a dossier/report artifact from completed, written-up evidence. To render raw prose directly without an adjudication, use pdf-build's markdown field instead.", notes: "Writes output files." },
    ToolDoc { name: "pdf-build", purpose: "Build a PDF with the built-in Lopdf renderer from Markdown or structured report/evidence JSON.", use_when: "Use when an agent needs a PDF output and should not rely on external tools such as pandoc, wkhtmltopdf, or LaTeX.", notes: "Pass raw prose in markdown, a file path in input_markdown, or structured report JSON in input_json." },
    ToolDoc { name: "work-metadata", purpose: "Fetch catalog metadata for one or more known work IDs in one call.", use_when: "Use when exact source_work_id values are already known and title, author, period, canon, traditions, or passage counts are needed.", notes: "work_ids is required. Batch known IDs together instead of issuing parallel works calls; missing IDs are reported explicitly." },
    ToolDoc { name: "works", purpose: "List and filter works from the catalog.", use_when: "Use to identify work IDs or filter corpus areas by metadata, including normalized title/author substrings. For already-known IDs, use work-metadata.", notes: "Requires catalog.index and reuses the engine's cached catalog. Legacy single work_id lookup remains supported." },
    ToolDoc { name: "catalog-index-info", purpose: "Show catalog index metadata.", use_when: "Use to inspect catalog coverage and availability.", notes: "Requires catalog.index." },
    ToolDoc { name: "vector-info", purpose: "Show vector index metadata and compatibility.", use_when: "Use to confirm the vector index model, dimension, row count, and doc-table fingerprint.", notes: "Requires vector.index and doc_table.bin." },
    ToolDoc { name: "vector-neighbors", purpose: "Find semantic neighbor candidates from a seed passage or external query embedding.", use_when: "Use for conceptual discovery, paraphrase candidates, or candidate expansion.", notes: "Vector hits are not citation-grade evidence. query_text is intentionally unsupported until an embedding provider is configured." },
    ToolDoc { name: "similar", purpose: "Find TF-IDF similar passages to a seed passage.", use_when: "Use for text reuse or thematic similarity starting from a known passage.", notes: "Requires TF-IDF index and doc table." },
    ToolDoc { name: "frontier", purpose: "Generate a discovery frontier packet for an agent session.", use_when: "Use after one or two exact hits to expand a seed passage into promising leads.", notes: "Does not use vector.index. Combines TF-IDF similar passages and phrase extraction, and is the preferred ordinary discovery bridge." },
    ToolDoc { name: "first-attestation", purpose: "Find earliest loaded-corpus occurrence of a phrase.", use_when: "Use for historical ordering claims inside the loaded corpus.", notes: "Earliest means earliest by corpus period_rank, not absolute historical origin." },
    ToolDoc { name: "phrase-history", purpose: "Analyze phrase distribution across periods/canons/traditions.", use_when: "Use for historical spread and timeline-style summaries.", notes: "Can include variants if requested." },
    ToolDoc { name: "phrase-index-search", purpose: "Force phrase-index lookup for exact phrase hits.", use_when: "Use when you specifically want to validate the phrase index path.", notes: "Unlike search, this errors when the phrase index is missing." },
    ToolDoc { name: "seed-pick", purpose: "Pick unworked seed passages for research.", use_when: "Use to start an exploratory research run from candidate passages.", notes: "Can filter by tradition and period." },
    ToolDoc { name: "expand-context-adaptive", purpose: "Expand passage context by climbing the catalog tree.", use_when: "Use after selecting a hit and needing surrounding context under a character budget.", notes: "Requires catalog and doc table." },
    ToolDoc { name: "trace-term-usage", purpose: "Group phrase hits by metadata buckets.", use_when: "Use to answer how a phrase is distributed by period, canon, author, or work.", notes: "This is analytical distribution. For navigational work/division clusters, use cluster-hits." },
    ToolDoc { name: "query-expand-terms", purpose: "Generate variants, orthographic flips, and aliases.", use_when: "Use before searching when exact wording may vary.", notes: "Does not search by itself." },
    ToolDoc { name: "compare-usage", purpose: "Compare two sub-corpora and score distinctive terms.", use_when: "Use for differential vocabulary between scopes.", notes: "Requires catalog and doc table." },
    ToolDoc { name: "collocation-search", purpose: "Discover likely collocates near one seed phrase.", use_when: "Use to identify local semantic or formulaic companions when the second term is unknown.", notes: "Scores terms near occurrences against background terms. For a specified term1+term2 evidence question, use pair-appearance." },
    ToolDoc { name: "pair-appearance", purpose: "Find direct evidence where two specified terms both appear.", use_when: "Use when the question is 'show passages where term1 appears with term2' or when verifying a proposed association.", notes: "Supports unit=passage, window, or sentence. Uses phrase-index candidates when available, verifies normalized text, returns offsets and snippets." },
    ToolDoc { name: "pair-profile", purpose: "Summarise co-occurrence rates for two terms grouped by period, canon, work, or author.", use_when: "Use for analytical questions like 'does念佛 appear with 禪 more in Song than Tang sources?' where aggregate distribution matters, not individual passages.", notes: "Returns pair_rate_given_term1 and pair_rate_given_term2 per group. For individual passage evidence, use pair-appearance." },
    ToolDoc { name: "person-resolve", purpose: "Resolve a person's name to candidate forms and corpus presence.", use_when: "Use before person-history to confirm that name forms and aliases appear in the corpus and to understand ambiguity.", notes: "Corpus-local resolver; no external authority file is consulted. Supply all aliases explicitly." },
    ToolDoc { name: "person-history", purpose: "Retrieve passages mentioning a person ordered by period with mention-class labels.", use_when: "Use to trace a person's presence across the corpus, classify mentions, and find earliest unambiguous references.", notes: "Mention classes (lineage_relation, attributed_saying, case_appearance, commentarial_reference, name_mention) are rule-based triage labels. Run person-resolve first." },
    ToolDoc { name: "citation-verify", purpose: "Verify whether a claimed quotation appears in the corpus.", use_when: "Use when validating a claimed attribution — e.g. 'is this saying really from the Diamond Sutra?' Scope to the claimed work when known.", notes: "Returns exact hits. When none found, returns near-matches scored by character overlap. Near-match is heuristic, not authoritative." },
    ToolDoc { name: "outline-search", purpose: "Search within or across catalog outline nodes and group hits.", use_when: "Use for scoped work/node searches, or corpus-wide grouped outline search if no scope is supplied.", notes: "Falls back to metadata grouping when no catalog/doc table exists and no scope is requested." },
    ToolDoc { name: "cluster-hits", purpose: "Cluster phrase hits by work or division.", use_when: "Use to answer where hits are concentrated in the corpus outline.", notes: "This is navigational clustering. For period/canon/author distributions, use trace-term-usage." },
    ToolDoc { name: "absence-check", purpose: "Check whether a phrase appears within a specific scope.", use_when: "Use for negative evidence in a work, canon, period, or node.", notes: "Absence is only meaningful for the loaded corpus and selected scope." },
    ToolDoc { name: "evidence-search", purpose: "Run exact phrase evidence search plus optional summaries.", use_when: "Use as the default agent tool for phrase evidence, attestation, history, term usage, and clusters.", notes: "Wraps simpler exact-evidence tools and reports index/fallback details." },
    ToolDoc { name: "batch-evidence-search", purpose: "Run compact exact searches for multiple independent phrases concurrently.", use_when: "Use for quick triage over a list of known phrases before selecting individual phrases for full evidence-search.", notes: "Preserves input order. Counts are returned counts and possibly_truncated is true when the requested limit was reached." },
    ToolDoc { name: "hybrid-discover", purpose: "Compactly merge semantic and lexical discovery candidates.", use_when: "Use when vector-semantic candidates are specifically useful; prefer frontier for ordinary expansion from a seed passage.", notes: "Defaults are intentionally small. Context and raw sub-results are returned only with verbosity=full. Vector hits are semantic candidates, not evidence." },
    ToolDoc { name: "source-investigate", purpose: "Gather context and a lexical discovery frontier for one seed, with optional history or semantic expansion.", use_when: "Use when beginning a source-dependence or passage-level investigation.", notes: "The fast default omits a duplicate similarity block and vector search. include_similar reuses frontier output; include_vector opts into slower semantic neighbors." },
    ToolDoc { name: "scope-profile", purpose: "Compare two corpus scopes and optionally trace a phrase.", use_when: "Use for period/canon/work vocabulary comparison and scoped term usage.", notes: "Wraps compare-usage and trace-term-usage." },
    ToolDoc { name: "report-from-evidence", purpose: "Validate adjudication, build graph, and build report in one workflow.", use_when: "Run this after your investigation is done and you've written up an adjudication — for reports straight from prose without one, use pdf-build's markdown field instead.", notes: "Writes output files and respects readonly/output-root safety." },
];
