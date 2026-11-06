//! Register the installed `sinorag.exe mcp` with the agent clients found on
//! this machine, and remove it again on uninstall.
//!
//! Each client keeps its MCP servers in a config file of its own; the entry
//! added is named [`SERVER_NAME`] and starts `<install>\sinorag.exe mcp`.
//! Only that entry is touched. A file that does not parse as plain JSON
//! (VS Code allows comments) or TOML is left alone and reported, so the
//! user can add the entry by hand. Before a file is first changed, a copy is
//! kept beside it as `<name>.sinorag-backup`.
//!
//! OpenCode is not listed: `sinorag agent` (the SinoRAG Documents shortcut)
//! writes its own project config in the install folder.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Map, Value};
use std::fs;
use std::path::{Path, PathBuf};

/// Distinct from "sinorag", which a CBETA install may use, so both can be
/// connected at once.
pub const SERVER_NAME: &str = "sinorag-documents";

/// How a client stores its servers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// `{ "mcpServers": { name: { command, args, env } } }` (Claude Desktop,
    /// Claude Code, Cursor, Windsurf, Gemini CLI, LM Studio).
    McpServers,
    /// Same, with `"type": "stdio"` in the entry (Claude Code's user scope).
    McpServersTyped,
    /// `{ "servers": { name: { type: "stdio", command, args, env } } }` (VS Code).
    VsCode,
    /// `[mcp_servers.<name>]` in TOML (Codex).
    CodexToml,
}

#[derive(Clone, Debug)]
pub struct Client {
    pub name: String,
    pub config: PathBuf,
    pub format: Format,
}

/// Where the per-user folders are. Tests point these at a temp dir.
#[derive(Clone, Debug)]
pub struct Locations {
    pub home: PathBuf,
    pub appdata: PathBuf,
    pub local_appdata: PathBuf,
    pub codex_home: Option<PathBuf>,
}

impl Locations {
    pub fn from_env() -> Option<Self> {
        let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        Some(Self {
            home: var("USERPROFILE").or_else(|| var("HOME"))?,
            appdata: var("APPDATA")?,
            local_appdata: var("LOCALAPPDATA")?,
            codex_home: var("CODEX_HOME"),
        })
    }
}

/// The clients that look installed for this user: their config folder (or,
/// for Claude Code, its config file) exists.
pub fn detect_clients(loc: &Locations) -> Vec<Client> {
    let mut out = Vec::new();
    let mut add = |name: &str, present: bool, config: PathBuf, format: Format| {
        if present {
            out.push(Client { name: name.to_string(), config, format });
        }
    };

    // Claude Desktop: the classic installer uses %APPDATA%\Claude; the MSIX
    // (Store / winget) build reads the same path through its package's
    // virtualised AppData, so that is where the file has to go.
    let classic = loc.appdata.join("Claude");
    add(
        "Claude Desktop",
        classic.is_dir(),
        classic.join("claude_desktop_config.json"),
        Format::McpServers,
    );
    if let Ok(entries) = fs::read_dir(loc.local_appdata.join("Packages")) {
        let mut packages: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.is_dir()
                    && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("Claude_"))
            })
            .collect();
        packages.sort();
        for package in packages {
            add(
                "Claude Desktop (Store)",
                true,
                package.join(r"LocalCache\Roaming\Claude\claude_desktop_config.json"),
                Format::McpServers,
            );
        }
    }

    let claude_code = loc.home.join(".claude.json");
    add("Claude Code", claude_code.is_file(), claude_code, Format::McpServersTyped);

    let codex = loc.codex_home.clone().unwrap_or_else(|| loc.home.join(".codex"));
    add("Codex", codex.is_dir(), codex.join("config.toml"), Format::CodexToml);

    let cursor = loc.home.join(".cursor");
    add("Cursor", cursor.is_dir(), cursor.join("mcp.json"), Format::McpServers);

    let windsurf = loc.home.join(r".codeium\windsurf");
    add("Windsurf", windsurf.is_dir(), windsurf.join("mcp_config.json"), Format::McpServers);

    for (name, folder) in [("VS Code", "Code"), ("VS Code Insiders", "Code - Insiders")] {
        let user = loc.appdata.join(folder).join("User");
        add(name, user.is_dir(), user.join("mcp.json"), Format::VsCode);
    }

    let gemini = loc.home.join(".gemini");
    add("Gemini CLI", gemini.is_dir(), gemini.join("settings.json"), Format::McpServers);

    let lmstudio = loc.home.join(".lmstudio");
    add("LM Studio", lmstudio.is_dir(), lmstudio.join("mcp.json"), Format::McpServers);

    out
}

/// What happened to one client.
#[derive(Debug)]
pub struct Outcome {
    pub client: String,
    pub config: PathBuf,
    pub result: Result<Change, String>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Change {
    Added,
    Updated,
    Unchanged,
    Removed,
    Absent,
}

/// The server as clients start it. The tool log goes to the install folder:
/// without this it lands in whatever folder the client starts servers in,
/// which for an editor is the user's open project.
fn server_spec(exe: &Path) -> (String, Vec<String>, Vec<(String, String)>) {
    let log = exe
        .parent()
        .map(|d| d.join(r".sinorag\tool_calls.jsonl"))
        .unwrap_or_else(|| PathBuf::from(r".sinorag\tool_calls.jsonl"));
    (
        exe.display().to_string(),
        vec!["mcp".to_string()],
        vec![("SINORAG_TOOL_LOG".to_string(), log.display().to_string())],
    )
}

/// Add or update the entry in every client given.
pub fn register(clients: &[Client], exe: &Path) -> Vec<Outcome> {
    clients
        .iter()
        .map(|c| Outcome {
            client: c.name.clone(),
            config: c.config.clone(),
            result: register_one(c, exe).map_err(|e| format!("{e:#}")),
        })
        .collect()
}

/// Remove the entry wherever it still starts this `exe`. An entry of that
/// name that the user pointed somewhere else is kept.
pub fn unregister(clients: &[Client], exe: &Path) -> Vec<Outcome> {
    clients
        .iter()
        .map(|c| Outcome {
            client: c.name.clone(),
            config: c.config.clone(),
            result: unregister_one(c, exe).map_err(|e| format!("{e:#}")),
        })
        .collect()
}

/// Whether any client already has the entry (so a re-run keeps it current).
pub fn registered_anywhere(clients: &[Client]) -> bool {
    clients.iter().any(|c| match c.format {
        Format::CodexToml => read_toml(&c.config)
            .ok()
            .and_then(|d| d.get("mcp_servers").and_then(|t| t.get(SERVER_NAME)).map(|_| ()))
            .is_some(),
        format => read_json(&c.config)
            .ok()
            .and_then(|v| v.get(json_key(format)).and_then(|s| s.get(SERVER_NAME)).map(|_| ()))
            .is_some(),
    })
}

fn json_key(format: Format) -> &'static str {
    if format == Format::VsCode {
        "servers"
    } else {
        "mcpServers"
    }
}

fn register_one(client: &Client, exe: &Path) -> Result<Change> {
    let (command, args, env) = server_spec(exe);
    match client.format {
        Format::CodexToml => {
            let mut doc = read_toml(&client.config)?;
            let servers = doc
                .entry("mcp_servers")
                .or_insert_with(|| {
                    let mut t = toml_edit::Table::new();
                    t.set_implicit(true);
                    toml_edit::Item::Table(t)
                })
                .as_table_mut()
                .ok_or_else(|| anyhow!("`mcp_servers` is not a table"))?;
            let mut entry = toml_edit::Table::new();
            entry.insert("command", toml_edit::value(command.as_str()));
            let mut arr = toml_edit::Array::new();
            for a in &args {
                arr.push(a.as_str());
            }
            entry.insert("args", toml_edit::value(arr));
            let mut env_table = toml_edit::InlineTable::new();
            for (k, v) in &env {
                env_table.insert(k, v.as_str().into());
            }
            entry.insert("env", toml_edit::value(env_table));
            let change = match servers.get(SERVER_NAME) {
                None => Change::Added,
                Some(old) if toml_spec(old) == Some((command.clone(), args.clone(), env.clone())) => {
                    return Ok(Change::Unchanged)
                }
                Some(_) => Change::Updated,
            };
            servers.insert(SERVER_NAME, toml_edit::Item::Table(entry));
            write_atomic(&client.config, doc.to_string().as_bytes())?;
            Ok(change)
        }
        format => {
            let mut root = read_json(&client.config)?;
            let obj = root
                .as_object_mut()
                .ok_or_else(|| anyhow!("the file is not a JSON object"))?;
            let servers = obj
                .entry(json_key(format))
                .or_insert_with(|| Value::Object(Map::new()))
                .as_object_mut()
                .ok_or_else(|| anyhow!("`{}` is not a JSON object", json_key(format)))?;
            let mut entry = Map::new();
            if format != Format::McpServers {
                entry.insert("type".into(), json!("stdio"));
            }
            entry.insert("command".into(), json!(command));
            entry.insert("args".into(), json!(args));
            let env: Map<String, Value> = env.into_iter().map(|(k, v)| (k, json!(v))).collect();
            entry.insert("env".into(), Value::Object(env));
            let entry = Value::Object(entry);
            let change = match servers.get(SERVER_NAME) {
                None => Change::Added,
                Some(old) if *old == entry => return Ok(Change::Unchanged),
                Some(_) => Change::Updated,
            };
            servers.insert(SERVER_NAME.to_string(), entry);
            write_json(&client.config, &root)?;
            Ok(change)
        }
    }
}

type Spec = (String, Vec<String>, Vec<(String, String)>);

/// An existing Codex entry read back as (command, args, env).
fn toml_spec(item: &toml_edit::Item) -> Option<Spec> {
    let command = item.get("command")?.as_str()?.to_string();
    let args = item
        .get("args")?
        .as_array()?
        .iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect::<Option<Vec<_>>>()?;
    let mut env = item
        .get("env")?
        .as_table_like()?
        .iter()
        .map(|(k, v)| Some((k.to_string(), v.as_str()?.to_string())))
        .collect::<Option<Vec<_>>>()?;
    env.sort();
    Some((command, args, env))
}

fn same_exe(command: &str, exe: &Path) -> bool {
    command.eq_ignore_ascii_case(&exe.display().to_string())
}

fn unregister_one(client: &Client, exe: &Path) -> Result<Change> {
    if !client.config.is_file() {
        return Ok(Change::Absent);
    }
    match client.format {
        Format::CodexToml => {
            let mut doc = read_toml(&client.config)?;
            let Some(servers) = doc.get_mut("mcp_servers").and_then(|t| t.as_table_like_mut()) else {
                return Ok(Change::Absent);
            };
            let ours = servers
                .get(SERVER_NAME)
                .and_then(|e| e.get("command"))
                .and_then(|c| c.as_str())
                .is_some_and(|c| same_exe(c, exe));
            if !ours {
                return Ok(Change::Absent);
            }
            servers.remove(SERVER_NAME);
            write_atomic(&client.config, doc.to_string().as_bytes())?;
            Ok(Change::Removed)
        }
        format => {
            let mut root = read_json(&client.config)?;
            let Some(servers) = root.get_mut(json_key(format)).and_then(Value::as_object_mut) else {
                return Ok(Change::Absent);
            };
            let ours = servers
                .get(SERVER_NAME)
                .and_then(|e| e.get("command"))
                .and_then(Value::as_str)
                .is_some_and(|c| same_exe(c, exe));
            if !ours {
                return Ok(Change::Absent);
            }
            servers.remove(SERVER_NAME);
            write_json(&client.config, &root)?;
            Ok(Change::Removed)
        }
    }
}

fn read_text(path: &Path) -> Result<Option<String>> {
    match fs::read(path) {
        Ok(bytes) => {
            let text = String::from_utf8(bytes).map_err(|_| anyhow!("{} is not UTF-8", path.display()))?;
            Ok(Some(text.trim_start_matches('\u{feff}').to_string()))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

fn read_json(path: &Path) -> Result<Value> {
    match read_text(path)? {
        Some(text) if !text.trim().is_empty() => serde_json::from_str(&text).map_err(|e| {
            anyhow!(
                "{} is not plain JSON ({e}); it was left as it is, add the server by hand",
                path.display()
            )
        }),
        _ => Ok(Value::Object(Map::new())),
    }
}

fn read_toml(path: &Path) -> Result<toml_edit::DocumentMut> {
    let text = read_text(path)?.unwrap_or_default();
    text.parse::<toml_edit::DocumentMut>().map_err(|e| {
        anyhow!(
            "{} is not valid TOML ({e}); it was left as it is, add the server by hand",
            path.display()
        )
    })
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    write_atomic(path, text.as_bytes())
}

/// Keep the first original as a backup, then replace the file in one step so
/// a client reading it never sees half a file.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    if path.is_file() {
        let backup = sibling(path, ".sinorag-backup");
        if !backup.exists() {
            fs::copy(path, &backup).with_context(|| format!("backing up {}", path.display()))?;
        }
    }
    let tmp = sibling(path, ".sinorag-tmp");
    fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sinorag-mcp-clients-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn locations(root: &Path) -> Locations {
        Locations {
            home: root.join("home"),
            appdata: root.join("roaming"),
            local_appdata: root.join("local"),
            codex_home: None,
        }
    }

    const EXE: &str = r"C:\Users\you\AppData\Local\Programs\SinoRAG-Documents\sinorag.exe";

    #[test]
    fn detects_only_present_clients() {
        let root = temp_root("detect");
        let loc = locations(&root);
        fs::create_dir_all(loc.home.join(".cursor")).unwrap();
        fs::create_dir_all(loc.local_appdata.join(r"Packages\Claude_pzs8sxrjxfjjc")).unwrap();
        fs::create_dir_all(loc.home.join(".codex")).unwrap();
        let names: Vec<String> = detect_clients(&loc).into_iter().map(|c| c.name).collect();
        assert_eq!(names, ["Claude Desktop (Store)", "Codex", "Cursor"]);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn json_register_keeps_other_settings_and_unregisters() {
        let root = temp_root("json");
        let config = root.join("settings.json");
        fs::write(&config, "\u{feff}{\"theme\":\"dark\",\"mcpServers\":{\"other\":{\"command\":\"x\"}}}").unwrap();
        let client = Client { name: "Gemini CLI".into(), config: config.clone(), format: Format::McpServers };
        let exe = Path::new(EXE);

        let out = register(std::slice::from_ref(&client), exe);
        assert_eq!(out[0].result, Ok(Change::Added));
        let v: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(v["theme"], "dark");
        assert_eq!(v["mcpServers"]["other"]["command"], "x");
        assert_eq!(v["mcpServers"][SERVER_NAME]["command"], EXE);
        assert_eq!(v["mcpServers"][SERVER_NAME]["args"], json!(["mcp"]));
        assert!(v["mcpServers"][SERVER_NAME]["env"]["SINORAG_TOOL_LOG"]
            .as_str()
            .unwrap()
            .ends_with(r"SinoRAG-Documents\.sinorag\tool_calls.jsonl"));
        assert!(v["mcpServers"][SERVER_NAME].get("type").is_none());
        assert!(sibling(&config, ".sinorag-backup").is_file());
        assert!(registered_anywhere(std::slice::from_ref(&client)));

        assert_eq!(register(std::slice::from_ref(&client), exe)[0].result, Ok(Change::Unchanged));
        // Another install's exe is not ours to remove.
        let elsewhere = Path::new(r"D:\Other\sinorag.exe");
        assert_eq!(unregister(std::slice::from_ref(&client), elsewhere)[0].result, Ok(Change::Absent));
        assert_eq!(unregister(std::slice::from_ref(&client), exe)[0].result, Ok(Change::Removed));
        let v: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert!(v["mcpServers"].get(SERVER_NAME).is_none());
        assert_eq!(v["mcpServers"]["other"]["command"], "x");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn vscode_uses_servers_with_type() {
        let root = temp_root("vscode");
        let config = root.join(r"Code\User\mcp.json");
        let client = Client { name: "VS Code".into(), config: config.clone(), format: Format::VsCode };
        assert_eq!(register(std::slice::from_ref(&client), Path::new(EXE))[0].result, Ok(Change::Added));
        let v: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(v["servers"][SERVER_NAME]["type"], "stdio");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn jsonc_file_is_left_alone() {
        let root = temp_root("jsonc");
        let config = root.join("mcp.json");
        let original = "// my servers\n{ \"servers\": {} }\n";
        fs::write(&config, original).unwrap();
        let client = Client { name: "VS Code".into(), config: config.clone(), format: Format::VsCode };
        let out = register(std::slice::from_ref(&client), Path::new(EXE));
        assert!(out[0].result.as_ref().unwrap_err().contains("by hand"));
        assert_eq!(fs::read_to_string(&config).unwrap(), original);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn codex_toml_keeps_comments() {
        let root = temp_root("codex");
        let config = root.join("config.toml");
        fs::write(&config, "# my settings\nmodel = \"o3\"\n\n[mcp_servers.other]\ncommand = \"x\"\n").unwrap();
        let client = Client { name: "Codex".into(), config: config.clone(), format: Format::CodexToml };
        let exe = Path::new(EXE);
        assert_eq!(register(std::slice::from_ref(&client), exe)[0].result, Ok(Change::Added));
        let text = fs::read_to_string(&config).unwrap();
        assert!(text.starts_with("# my settings\nmodel = \"o3\""), "{text}");
        let doc: toml_edit::DocumentMut = text.parse().unwrap();
        assert_eq!(doc["mcp_servers"][SERVER_NAME]["command"].as_str(), Some(EXE));
        assert_eq!(doc["mcp_servers"]["other"]["command"].as_str(), Some("x"));
        assert_eq!(register(std::slice::from_ref(&client), exe)[0].result, Ok(Change::Unchanged));
        assert_eq!(unregister(std::slice::from_ref(&client), exe)[0].result, Ok(Change::Removed));
        let doc: toml_edit::DocumentMut = fs::read_to_string(&config).unwrap().parse().unwrap();
        assert!(doc["mcp_servers"].get(SERVER_NAME).is_none());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn codex_new_file() {
        let root = temp_root("codex-new");
        let config = root.join(r".codex\config.toml");
        let client = Client { name: "Codex".into(), config: config.clone(), format: Format::CodexToml };
        assert_eq!(register(std::slice::from_ref(&client), Path::new(EXE))[0].result, Ok(Change::Added));
        let text = fs::read_to_string(&config).unwrap();
        assert!(text.contains("[mcp_servers.sinorag-documents]"), "{text}");
        let _ = fs::remove_dir_all(root);
    }
}
