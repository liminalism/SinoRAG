use anyhow::Result;
use clap::Parser;
use sinorag::{cli, commands};

// Note: the Windows linker stack reserve is bumped to 32 MiB via
// `.cargo/config.toml`, because debug builds generate very large stack
// frames (proc-macro-generated MCP tool futures, clap parser) that
// overflow the default 1 MiB Windows main-thread stack at startup.
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // Logs go to stderr. The default writer is stdout, which `sinorag mcp`
    // reserves for JSON-RPC: one warning there and the client drops the
    // connection as a protocol error.
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_writer(std::io::stderr)
        .init();

    // Resolve all relative paths (e.g. "data/...") against the exe's directory
    // so the binary can be invoked from any working directory.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let _ = std::env::set_current_dir(exe_dir);
        }
    }

    let cli = cli::Cli::parse();
    commands::run(cli).await
}
