//! Drives the built `sinorag mcp` binary over stdio, the way a client does.
//!
//! The unit tests check the pieces; this checks what a client actually
//! receives, which is where the defects were: a server name of "rmcp", a
//! protocol version pinned a generation back, a tool list some clients
//! refused to load, and errors a model could not act on. Nothing here needs
//! a corpus: every assertion holds whether or not a data root sits beside
//! the binary.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{json, Value};

struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl Server {
    fn start(extra_args: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_sinorag"))
            .arg("mcp")
            .args(extra_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("start sinorag mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut server = Self {
            child,
            stdin,
            stdout,
            next_id: 0,
        };
        let init = server.request(
            "initialize",
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": { "name": "mcp_stdio test", "version": "0" }
            }),
        );
        assert_eq!(init["result"]["serverInfo"]["name"], "sinorag");
        assert_eq!(
            init["result"]["serverInfo"]["version"],
            env!("CARGO_PKG_VERSION")
        );
        // The version the client asked for, not an older one of the server's.
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert!(init["result"]["instructions"]
            .as_str()
            .is_some_and(|text| text.contains("SinoRAG")));
        server.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }));
        server
    }

    fn send(&mut self, message: &Value) {
        let mut line = serde_json::to_vec(message).unwrap();
        line.push(b'\n');
        self.stdin.write_all(&line).unwrap();
        self.stdin.flush().unwrap();
    }

    /// Send a request and return its response. Every line the server writes
    /// to stdout on the way must be JSON: that stream is the protocol.
    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }));
        loop {
            let mut line = String::new();
            let read = self.stdout.read_line(&mut line).unwrap();
            assert!(read > 0, "server closed stdout while answering {method}");
            let message: Value = serde_json::from_str(line.trim_end())
                .unwrap_or_else(|err| panic!("non-JSON on stdout ({err}): {line}"));
            if message["id"] == json!(id) {
                return message;
            }
        }
    }

    /// Call a tool and return the envelope it answered with.
    fn call(&mut self, tool: &str, arguments: Value) -> (bool, Value) {
        let reply = self.request("tools/call", json!({ "name": tool, "arguments": arguments }));
        assert!(
            reply.get("error").is_none(),
            "a tool failure must be a result with isError, not a JSON-RPC error: {reply}"
        );
        let result = &reply["result"];
        let is_error = result["isError"].as_bool().unwrap_or(false);
        let structured = result["structuredContent"].clone();
        // Both halves carry the same envelope: hosts differ in which one
        // they show the model.
        let text: Value =
            serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text, structured);
        assert_eq!(structured["ok"], json!(!is_error));
        assert_eq!(structured["tool"], tool);
        (is_error, structured)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn tool_names(server: &mut Server) -> Vec<String> {
    let reply = server.request("tools/list", json!({}));
    let tools = reply["result"]["tools"].as_array().expect("tool list");
    for tool in tools {
        assert!(
            tool.get("outputSchema").is_none(),
            "{} advertises an output schema",
            tool["name"]
        );
        assert_eq!(tool["inputSchema"]["type"], "object");
        assert!(tool["description"].as_str().is_some_and(|d| !d.is_empty()));
    }
    tools
        .iter()
        .map(|tool| tool["name"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn a_read_only_session_lists_only_what_it_can_run() {
    let mut server = Server::start(&[]);
    let names = tool_names(&mut server);
    for expected in ["status", "search", "passage", "source-read", "works"] {
        assert!(names.iter().any(|n| n == expected), "{expected} is missing");
    }
    for writer in ["pdf-build", "run-batch", "report-build"] {
        assert!(!names.iter().any(|n| n == writer), "{writer} is listed");
    }

    let mut writable = Server::start(&["--writable"]);
    let names = tool_names(&mut writable);
    assert!(names.iter().any(|n| n == "pdf-build"));
}

#[test]
fn a_bad_call_is_answered_with_what_to_send_instead() {
    let mut server = Server::start(&[]);

    let (is_error, envelope) = server.call("status", json!({}));
    assert!(!is_error, "{envelope}");
    assert_eq!(envelope["result"]["schema"], "sinorag-status-v1");

    // The misspelling the tool log shows a model making.
    let (is_error, envelope) = server.call("search", json!({ "pattern": "coolant pump" }));
    assert!(is_error);
    assert_eq!(envelope["error"]["code"], "invalid_args");
    let message = envelope["error"]["message"].as_str().unwrap();
    assert!(message.contains("missing required argument `phrase`"), "{message}");
    assert!(message.contains("unknown argument `pattern`"), "{message}");
    assert!(envelope["error"]["suggested_command"]
        .as_str()
        .unwrap()
        .contains("(required: phrase)"));

    let (is_error, envelope) = server.call("search", json!({ "phrase": "x", "limit": "three" }));
    assert!(is_error);
    assert_eq!(envelope["error"]["code"], "invalid_args");

    let (is_error, envelope) = server.call("search", json!({ "phrase": "x", "mode": "cluster" }));
    assert!(is_error);
    assert_eq!(envelope["error"]["code"], "invalid_args");
    assert!(envelope["error"]["message"]
        .as_str()
        .unwrap()
        .contains("hits, clusters, trace, all"));

    let (is_error, envelope) = server.call("no-such-tool", json!({}));
    assert!(is_error);
    assert_eq!(envelope["error"]["code"], "unknown_tool");

    // A write tool is refused by name even though it is not listed.
    let (is_error, envelope) = server.call("pdf-build", json!({ "out": "x.pdf", "markdown": "# x" }));
    assert!(is_error);
    assert_eq!(envelope["error"]["code"], "readonly_violation");

    // The session is still usable after all of that.
    let (is_error, _) = server.call("status", json!({}));
    assert!(!is_error);
}
