use image::{ImageBuffer, Rgba};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("out dir"));

    let workspace_cargo = manifest_dir.join("../../Cargo.toml");
    let version = read_package_version(&workspace_cargo);
    println!("cargo:rustc-env=EXPECTED_SINORAG_VERSION={version}");
    println!("cargo:rerun-if-changed={}", workspace_cargo.display());

    // The payload is sinorag.exe in a .7z. `make-payload.ps1` writes one
    // here; otherwise fall back to the one the CBETA installers share.
    let local_path = manifest_dir.join("sinorag.7z");
    let sibling_path = manifest_dir.join("../sinorag-installer/sinorag.7z");
    // Watch both, so that adding a local payload later re-runs this script.
    println!("cargo:rerun-if-changed={}", local_path.display());
    println!("cargo:rerun-if-changed={}", sibling_path.display());
    let payload_path = if local_path.exists() {
        local_path.clone()
    } else {
        sibling_path.clone()
    };
    if !payload_path.exists() {
        panic!(
            "missing compressed SinoRAG payload at {} or {}; run make-payload.ps1",
            local_path.display(),
            sibling_path.display()
        );
    }
    println!("cargo:rustc-env=SINORAG_PAYLOAD_PATH={}", payload_path.display());
    check_payload(&payload_path, &version, &manifest_dir, &out_dir);

    let png_path = out_dir.join("sinorag-icon.png");
    let ico_path = out_dir.join("SinoRAG.ico");
    write_icon_png(&png_path);
    write_icon_ico(&png_path, &ico_path);
    println!("cargo:rustc-env=SINORAG_ICON_PNG={}", png_path.display());
    println!("cargo:rustc-env=SINORAG_ICON_ICO={}", ico_path.display());
}

/// The payload is built and packed separately from this crate, so nothing
/// else stops an old sinorag.exe from being embedded. That matters more here
/// than for the CBETA installers: the intake writes the indexes with the
/// `sinorag` library it links, and the installed sinorag.exe has to read
/// them, so the two must come from the same source.
///
/// A payload of another version fails the build. A payload packed before the
/// newest source change only warns, because file times are not proof.
fn check_payload(payload: &Path, expected: &str, manifest_dir: &Path, out_dir: &Path) {
    println!("cargo:rerun-if-env-changed=SINORAG_SKIP_PAYLOAD_CHECK");
    let sources = manifest_dir.join("../../src");
    println!("cargo:rerun-if-changed={}", sources.display());
    if env::var_os("SINORAG_SKIP_PAYLOAD_CHECK").is_some() {
        return;
    }
    let remedy = format!(
        "Rebuild it with {} (or set SINORAG_SKIP_PAYLOAD_CHECK=1 to build anyway).",
        manifest_dir.join("make-payload.ps1").display()
    );

    match payload_version(payload, out_dir) {
        Ok(found) if found == expected => {}
        Ok(found) => panic!(
            "the payload {} holds sinorag.exe {found}, but this checkout is SinoRAG {expected}. {remedy}",
            payload.display()
        ),
        Err(why) => println!(
            "cargo:warning=could not check the version of the sinorag.exe in {}: {why}",
            payload.display()
        ),
    }

    let packed = fs::metadata(payload).and_then(|m| m.modified()).ok();
    if let (Some(packed), Some((changed, file))) = (packed, newest_source(&sources)) {
        if changed > packed {
            println!(
                "cargo:warning={} was packed before the latest SinoRAG source change ({}), so the \
                 installed sinorag.exe will not include it. {remedy}",
                payload.display(),
                file.display()
            );
        }
    }
}

/// Unpack the payload with the bsdtar that ships with Windows and ask the
/// sinorag.exe inside for its version.
fn payload_version(payload: &Path, out_dir: &Path) -> Result<String, String> {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") || !cfg!(windows) {
        return Err("not building on Windows for Windows".to_string());
    }
    let tar = PathBuf::from(env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into()))
        .join("System32")
        .join("tar.exe");
    let dir = out_dir.join("payload-check");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    let version = (|| {
        let unpacked = Command::new(&tar)
            .arg("-xf")
            .arg(payload)
            .arg("-C")
            .arg(&dir)
            .output()
            .map_err(|e| format!("running {}: {e}", tar.display()))?;
        if !unpacked.status.success() {
            return Err(format!("tar: {}", String::from_utf8_lossy(&unpacked.stderr).trim()));
        }
        let exe = dir.join("sinorag.exe");
        let out = Command::new(&exe)
            .arg("--version")
            .output()
            .map_err(|e| format!("running {}: {e}", exe.display()))?;
        String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .last()
            .map(str::to_string)
            .ok_or_else(|| "sinorag.exe --version printed nothing".to_string())
    })();
    // The unpacked exe is well over 100 MB; do not leave it in OUT_DIR.
    let _ = fs::remove_dir_all(&dir);
    version
}

/// The most recently modified `.rs` file under `dir`.
fn newest_source(dir: &Path) -> Option<(SystemTime, PathBuf)> {
    let mut newest: Option<(SystemTime, PathBuf)> = None;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d).ok()?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                if let Ok(time) = entry.metadata().and_then(|m| m.modified()) {
                    if newest.as_ref().is_none_or(|(t, _)| time > *t) {
                        newest = Some((time, path));
                    }
                }
            }
        }
    }
    newest
}

fn read_package_version(path: &Path) -> String {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let mut in_package = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_package = trimmed == "[package]";
            continue;
        }
        if in_package {
            if let Some(rest) = trimmed.strip_prefix("version") {
                if let Some((_, value)) = rest.split_once('=') {
                    return value.trim().trim_matches('"').to_string();
                }
            }
        }
    }
    panic!("no [package] version found in {}", path.display());
}

fn write_icon_png(path: &Path) {
    let mut img = ImageBuffer::from_pixel(128, 128, Rgba([18, 50, 53, 255]));
    for y in 0..128 {
        for x in 0..128 {
            let tint = ((x + y) / 14) as u8;
            img.put_pixel(
                x,
                y,
                Rgba([18 + tint.min(16), 50 + tint.min(22), 53 + tint.min(20), 255]),
            );
        }
    }
    for y in 103..112 {
        for x in 13..115 {
            img.put_pixel(x, y, Rgba([154, 73, 55, 255]));
        }
    }
    for y in 113..118 {
        for x in 13..83 {
            img.put_pixel(x, y, Rgba([197, 154, 61, 255]));
        }
    }
    draw_text(&mut img, 12, 32, "SINO", 4, Rgba([247, 244, 234, 255]));
    draw_text(&mut img, 22, 73, "RAG", 4, Rgba([247, 244, 234, 255]));
    img.save(path).expect("write png icon");
}

fn write_icon_ico(png_path: &Path, ico_path: &Path) {
    let image = ico::IconImage::read_png(fs::File::open(png_path).expect("open png"))
        .expect("read png for ico");
    let dir_entry = ico::IconDirEntry::encode(&image).expect("encode ico entry");
    let mut dir = ico::IconDir::new(ico::ResourceType::Icon);
    dir.add_entry(dir_entry);
    dir.write(fs::File::create(ico_path).expect("create ico"))
        .expect("write ico");
}

fn draw_text(
    img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    x: u32,
    y: u32,
    text: &str,
    scale: u32,
    color: Rgba<u8>,
) {
    let mut cursor = x;
    for ch in text.chars() {
        if ch == ' ' {
            cursor += 4 * scale;
            continue;
        }
        draw_char(img, cursor, y, ch, scale, color);
        cursor += 6 * scale;
    }
}

fn draw_char(
    img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    x: u32,
    y: u32,
    ch: char,
    scale: u32,
    color: Rgba<u8>,
) {
    let rows = match ch {
        'A' => [
            "01110", "10001", "10001", "11111", "10001", "10001", "10001",
        ],
        'G' => [
            "01111", "10000", "10000", "10111", "10001", "10001", "01111",
        ],
        'I' => [
            "11111", "00100", "00100", "00100", "00100", "00100", "11111",
        ],
        'N' => [
            "10001", "11001", "10101", "10011", "10001", "10001", "10001",
        ],
        'O' => [
            "01110", "10001", "10001", "10001", "10001", "10001", "01110",
        ],
        'R' => [
            "11110", "10001", "10001", "11110", "10100", "10010", "10001",
        ],
        'S' => [
            "01111", "10000", "10000", "01110", "00001", "00001", "11110",
        ],
        _ => [
            "00000", "00000", "00000", "00000", "00000", "00000", "00000",
        ],
    };

    for (row_idx, row) in rows.iter().enumerate() {
        for (col_idx, bit) in row.as_bytes().iter().enumerate() {
            if *bit == b'1' {
                for dy in 0..scale {
                    for dx in 0..scale {
                        let px = x + col_idx as u32 * scale + dx;
                        let py = y + row_idx as u32 * scale + dy;
                        if px < img.width() && py < img.height() {
                            img.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }
    }
}
