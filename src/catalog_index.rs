use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

pub type NodeId = u32;
pub type DocId = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusCatalogIndex {
    pub schema: String,
    pub source_fingerprint: Option<String>,
    pub doc_table_fingerprint: Option<String>,

    pub works: Vec<WorkRecord>,
    pub nodes: Vec<OutlineNode>,

    pub work_id_map: FxHashMap<String, usize>,
    pub doc_parent: FxHashMap<DocId, NodeId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkRecord {
    pub work_id: String,
    pub source_corpus: String,
    pub canon: String,
    pub canon_name: String,
    pub main_title: String,
    pub author: String,
    pub period: String,
    pub period_rank: i32,
    pub origin: String,
    pub traditions: Vec<String>,
    pub source_rel_paths: Vec<String>,
    pub root_node: NodeId,
    pub passage_count: u32,
    pub cjk_char_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlineNode {
    pub node_id: NodeId,
    pub parent_id: Option<NodeId>,
    pub children: Vec<NodeId>,

    pub source_corpus: String,
    pub work_id: String,
    pub source_rel_path: String,

    pub node_kind: OutlineNodeKind,
    pub label: String,
    pub heading_path: String,
    pub div_path: String,

    pub first_doc_id: Option<DocId>,
    pub last_doc_id: Option<DocId>,
    pub passage_count: u32,
    pub cjk_char_count: u32,

    pub from_lb: Option<String>,
    pub to_lb: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlineNodeKind {
    Corpus,
    Canon,
    Work,
    Volume,
    Fascicle,
    Chapter,
    Section,
    Division,
    PassageRange,
}

impl CorpusCatalogIndex {
    pub fn new() -> Self {
        Self {
            schema: "readzen-corpus-catalog".to_string(),
            source_fingerprint: None,
            doc_table_fingerprint: None,
            works: Vec::new(),
            nodes: Vec::new(),
            work_id_map: FxHashMap::default(),
            doc_parent: FxHashMap::default(),
        }
    }

    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bincode::serialize(self)?)?;
        Ok(())
    }

    pub fn save_atomic(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let bytes = bincode::serialize(self)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension(format!(
            "{}.tmp",
            path.extension().and_then(|s| s.to_str()).unwrap_or("bin")
        ));
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let mut catalog: Self = bincode::deserialize(&std::fs::read(path)?)?;
        // Older catalog builders split the JSON-encoded parquet value on
        // commas, leaving entries such as `[\"Chan/Zen\"` and
        // `\"Historical\"]`. Repair those records on load so existing packs do
        // not need a full index rebuild just to return clean metadata.
        for work in &mut catalog.works {
            work.traditions = normalize_tradition_values(&work.traditions);
        }
        Ok(catalog)
    }

    pub fn info_payload(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": self.schema,
            "works": self.works.len(),
            "nodes": self.nodes.len(),
            "source_fingerprint": self.source_fingerprint,
            "doc_table_fingerprint": self.doc_table_fingerprint,
            "doc_parent_entries": self.doc_parent.len(),
        })
    }

    pub fn get_work(&self, work_id: &str) -> Option<&WorkRecord> {
        self.work_id_map
            .get(work_id)
            .and_then(|idx| self.works.get(*idx))
    }

    pub fn get_node(&self, node_id: NodeId) -> Option<&OutlineNode> {
        self.nodes.get(node_id as usize)
    }

    pub fn get_children(&self, node_id: NodeId) -> Option<Vec<&OutlineNode>> {
        let node = self.get_node(node_id)?;
        Some(
            node.children
                .iter()
                .filter_map(|&child_id| self.get_node(child_id))
                .collect(),
        )
    }

    /// Push a node with the right node_id stamped on it. Returns the assigned id.
    pub fn push_node(&mut self, mut node: OutlineNode) -> NodeId {
        let id = self.nodes.len() as NodeId;
        node.node_id = id;
        self.nodes.push(node);
        id
    }

    /// Append `child` to `parent`'s children list.
    pub fn add_child(&mut self, parent: NodeId, child: NodeId) {
        if let Some(p) = self.nodes.get_mut(parent as usize) {
            p.children.push(child);
        }
    }
}

fn normalize_tradition_values(values: &[String]) -> Vec<String> {
    let joined = values.join(",");
    if let Ok(parsed) = serde_json::from_str::<Vec<String>>(&joined) {
        return parsed;
    }
    values
        .iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::normalize_tradition_values;

    #[test]
    fn repairs_legacy_json_fragments_without_changing_clean_values() {
        assert_eq!(
            normalize_tradition_values(&["[\"Historical\"]".to_string()]),
            vec!["Historical"]
        );
        assert_eq!(
            normalize_tradition_values(&[
                "[\"Chan/Zen\"".to_string(),
                "\"Historical\"]".to_string()
            ]),
            vec!["Chan/Zen", "Historical"]
        );
        assert_eq!(
            normalize_tradition_values(&["Chan/Zen".to_string(), "Historical".to_string()]),
            vec!["Chan/Zen", "Historical"]
        );
    }
}

impl OutlineNode {
    /// Minimal node constructor for the builder. node_id is stamped in by
    /// `CorpusCatalogIndex::push_node`.
    pub fn leaf(
        kind: OutlineNodeKind,
        parent_id: Option<NodeId>,
        source_corpus: String,
        work_id: String,
        source_rel_path: String,
        label: String,
    ) -> Self {
        Self {
            node_id: 0,
            parent_id,
            children: Vec::new(),
            source_corpus,
            work_id,
            source_rel_path,
            node_kind: kind,
            label,
            heading_path: String::new(),
            div_path: String::new(),
            first_doc_id: None,
            last_doc_id: None,
            passage_count: 0,
            cjk_char_count: 0,
            from_lb: None,
            to_lb: None,
        }
    }
}
