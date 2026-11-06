//! Headless intake, for testing the pipeline without the GUI or payload:
//!
//!   cargo run --example intake_cli -- <install-root> [--ocr eng] [--no-lexical] <files or folders>...
//!
//! Then query it with a current sinorag.exe copied into <install-root>
//! (SinoRAG resolves `data/` next to its own exe, not the working directory):
//!   <install-root>\sinorag tool-call search --json '{"phrase":"...","limit":5}'

use sinorag_intake::extract::ExtractOptions;
use sinorag_intake::pipeline::{run_intake, IntakeEvent, IntakeOptions};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("usage: intake_cli <root> [--ocr LANG] [--no-lexical] <inputs>..."))?);
    let mut ocr = "eng".to_string();
    let mut lexical = true;
    let mut inputs = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ocr" => ocr = args.next().unwrap_or(ocr),
            "--no-lexical" => lexical = false,
            _ => inputs.push(PathBuf::from(a)),
        }
    }
    let opts = IntakeOptions {
        root,
        inputs,
        extract: ExtractOptions { ocr_language: ocr, ..Default::default() },
        build_lexical: lexical,
        force_rebuild: false,
        progress_band: (0.0, 1.0),
    };
    let summary = run_intake(&opts, &|event| match event {
        IntakeEvent::Step(s, p) => eprintln!("[{:>3.0}%] {s}", p * 100.0),
        IntakeEvent::Log(l) => eprintln!("      {l}"),
        IntakeEvent::File { path, status } => eprintln!("      {} — {status}", path.display()),
    })?;
    println!("{summary:#?}");
    Ok(())
}
