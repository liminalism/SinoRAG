//! Windows helpers shared by the intake GUI and its uninstaller. Adapted from
//! `sinorag-installer-winsafe`; every name (install dir, uninstall key,
//! shortcuts, mutex) is distinct so this variant installs side by side with
//! the CBETA installer instead of overwriting it.

use std::path::{Path, PathBuf};
use windows::Win32::Foundation::{ERROR_ALREADY_EXISTS, ERROR_SUCCESS};
use windows::Win32::System::Com::CoTaskMemFree;
use windows::Win32::System::Registry::{
    RegCloseKey, RegCreateKeyExW, RegDeleteKeyW, RegOpenKeyExW, RegQueryValueExW,
    RegSetValueExW, HKEY, HKEY_CURRENT_USER, KEY_READ, KEY_WRITE,
    REG_DWORD, REG_OPTION_NON_VOLATILE, REG_SZ, REG_VALUE_TYPE,
};
use windows::Win32::UI::Shell::{
    FOLDERID_Desktop, FOLDERID_LocalAppData, FOLDERID_Programs, KNOWN_FOLDER_FLAG,
    SHGetKnownFolderPath,
};
use windows::core::{GUID, PCWSTR};

pub const UNINSTALL_SUBKEY: &str =
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall\SinoRAG-Documents";

pub fn to_wide(path: &std::path::Path) -> Vec<u16> {
    path.to_string_lossy()
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect()
}

pub fn to_wide_str(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

pub fn known_folder_path(id: &GUID) -> Option<PathBuf> {
    unsafe {
        let pwstr = SHGetKnownFolderPath(id, KNOWN_FOLDER_FLAG(0), None).ok()?;
        let result = pwstr.to_string().ok().map(PathBuf::from);
        CoTaskMemFree(Some(pwstr.as_ptr() as *const std::ffi::c_void));
        result
    }
}

pub fn desktop_shortcut_path() -> Option<PathBuf> {
    known_folder_path(&FOLDERID_Desktop).map(|p| p.join("SinoRAG Documents.lnk"))
}

pub fn start_menu_shortcut_path() -> Option<PathBuf> {
    known_folder_path(&FOLDERID_Programs).map(|p| p.join("SinoRAG").join("SinoRAG Documents.lnk"))
}

/// Start Menu entry that re-opens the intake window to add more documents.
pub fn start_menu_intake_shortcut_path() -> Option<PathBuf> {
    known_folder_path(&FOLDERID_Programs)
        .map(|p| p.join("SinoRAG").join("Add documents to SinoRAG.lnk"))
}

pub fn default_install_path() -> PathBuf {
    known_folder_path(&FOLDERID_LocalAppData)
        .map(|p| p.join("Programs").join("SinoRAG-Documents"))
        .unwrap_or_else(|| PathBuf::from(r"C:\SinoRAG-Documents"))
}

pub fn write_uninstall_entry(install_path: &std::path::Path, version: &str) {
    let _ = try_write_uninstall_entry(install_path, version);
}

fn try_write_uninstall_entry(
    install_path: &std::path::Path,
    version: &str,
) -> windows::core::Result<()> {
    let subkey = to_wide_str(UNINSTALL_SUBKEY);
    let mut hkey = HKEY::default();
    unsafe {
        RegCreateKeyExW(
            HKEY_CURRENT_USER,
            PCWSTR::from_raw(subkey.as_ptr()),
            None,
            PCWSTR::null(),
            REG_OPTION_NON_VOLATILE,
            KEY_WRITE,
            None,
            &mut hkey,
            None,
        )
        .ok()?;
    }

    let display_name = format!("SinoRAG Documents {version}");
    let icon_path = install_path.join("SinoRAG.ico").display().to_string();
    // UninstallString is a command line, so a path with spaces (a user name
    // with a space puts one in %LOCALAPPDATA%) has to be quoted.
    let uninstall_str = format!(
        "\"{}\"",
        install_path.join("sinorag-intake-uninstaller.exe").display()
    );
    let install_loc = install_path.display().to_string();

    set_reg_sz(hkey, "DisplayName", &display_name);
    set_reg_sz(hkey, "DisplayVersion", version);
    set_reg_sz(hkey, "DisplayIcon", &icon_path);
    set_reg_sz(hkey, "Publisher", "SinoRAG");
    set_reg_sz(hkey, "InstallLocation", &install_loc);
    set_reg_sz(hkey, "UninstallString", &uninstall_str);
    set_reg_dword(hkey, "NoModify", 1);
    set_reg_dword(hkey, "NoRepair", 1);

    unsafe { RegCloseKey(hkey).ok() }
}

fn set_reg_sz(hkey: HKEY, name: &str, value: &str) {
    let name_w = to_wide_str(name);
    let value_w: Vec<u16> = value.encode_utf16().chain(std::iter::once(0)).collect();
    let bytes =
        unsafe { std::slice::from_raw_parts(value_w.as_ptr() as *const u8, value_w.len() * 2) };
    unsafe {
        let _ = RegSetValueExW(
            hkey,
            PCWSTR::from_raw(name_w.as_ptr()),
            None,
            REG_SZ,
            Some(bytes),
        );
    }
}

fn set_reg_dword(hkey: HKEY, name: &str, value: u32) {
    let name_w = to_wide_str(name);
    let bytes = value.to_le_bytes();
    unsafe {
        let _ = RegSetValueExW(
            hkey,
            PCWSTR::from_raw(name_w.as_ptr()),
            None,
            REG_DWORD,
            Some(&bytes),
        );
    }
}

pub fn delete_uninstall_entry() {
    let subkey = to_wide_str(UNINSTALL_SUBKEY);
    unsafe {
        let _ = RegDeleteKeyW(
            HKEY_CURRENT_USER,
            PCWSTR::from_raw(subkey.as_ptr()),
        );
    }
}

pub fn read_install_location() -> Option<PathBuf> {
    let subkey = to_wide_str(UNINSTALL_SUBKEY);
    let mut hkey = HKEY::default();
    let opened = unsafe {
        RegOpenKeyExW(
            HKEY_CURRENT_USER,
            PCWSTR::from_raw(subkey.as_ptr()),
            None,
            KEY_READ,
            &mut hkey,
        )
    };
    if opened != ERROR_SUCCESS {
        return None;
    }

    let value_name = to_wide_str("InstallLocation");
    let mut buf = vec![0u16; 520];
    let mut data_size = (buf.len() * 2) as u32;
    let mut kind = REG_VALUE_TYPE::default();
    let queried = unsafe {
        RegQueryValueExW(
            hkey,
            PCWSTR::from_raw(value_name.as_ptr()),
            None,
            Some(&mut kind),
            Some(buf.as_mut_ptr() as *mut u8),
            Some(&mut data_size),
        )
    };
    unsafe { let _ = RegCloseKey(hkey); }

    if queried != ERROR_SUCCESS {
        return None;
    }

    // The stored string normally ends in a NUL, but nothing guarantees it.
    let char_count = (data_size as usize / 2).min(buf.len());
    let s = String::from_utf16_lossy(&buf[..char_count]);
    let s = s.trim_end_matches('\0');
    if s.is_empty() { None } else { Some(PathBuf::from(s)) }
}

/// What a folder holds, as far as installing into it (or removing it) goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallFolder {
    /// Missing or empty.
    Fresh,
    /// A SinoRAG Documents install, complete or interrupted.
    Installed,
    /// A SinoRAG install with another corpus in it (the name is given).
    OtherCorpus(String),
    /// Something else is in there (one entry is given as an example).
    Occupied(String),
}

/// Everything the intake puts in its install folder.
const OWN_ENTRIES: &[&str] = &[
    "sinorag.exe",
    "sinorag-intake.exe",
    "sinorag-intake-uninstaller.exe",
    "sinorag.ico",
    ".sinorag-payload.7z",
    "intake",
    "data",
];

/// What SinoRAG itself adds to the folder once it is used: the OpenCode
/// project files `sinorag agent` writes, and the tool-call log.
const RUNTIME_ENTRIES: &[&str] = &[".opencode", "agents.md", ".sinorag"];

/// Entries in the install folder that neither the intake nor SinoRAG put
/// there: notes or reports a chat session saved, say. The uninstaller
/// deletes the folder whole, so it names these first.
pub fn foreign_entries(dir: &Path) -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|name| {
                    let lower = name.to_ascii_lowercase();
                    !OWN_ENTRIES.contains(&lower.as_str()) && !RUNTIME_ENTRIES.contains(&lower.as_str())
                })
                .collect()
        })
        .unwrap_or_default();
    found.sort();
    found
}

/// The install folder is the intake's alone: every run rebuilds all the
/// indexes in it, and the uninstaller deletes it whole. So it must not be a
/// folder with someone's files in it, nor a CBETA install.
pub fn classify_install_folder(dir: &Path) -> InstallFolder {
    let entries: Vec<String> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect(),
        Err(_) => return InstallFolder::Fresh,
    };
    if entries.is_empty() {
        return InstallFolder::Fresh;
    }
    if let Ok(partitions) = std::fs::read_dir(dir.join("data").join("passages.parquet")) {
        for name in partitions.flatten().map(|e| e.file_name().to_string_lossy().into_owned()) {
            if let Some(corpus) = name.strip_prefix("source_corpus=") {
                if corpus != crate::corpus::CORPUS_ID {
                    return InstallFolder::OtherCorpus(corpus.to_string());
                }
            }
        }
    }
    let has = |name: &str| entries.iter().any(|e| e.eq_ignore_ascii_case(name));
    if has("intake") || has("sinorag-intake.exe") {
        return InstallFolder::Installed;
    }
    // An install that was interrupted before the intake copied itself in.
    match entries
        .iter()
        .find(|e| !OWN_ENTRIES.contains(&e.to_ascii_lowercase().as_str()))
    {
        None => InstallFolder::Installed,
        Some(other) => InstallFolder::Occupied(other.clone()),
    }
}

/// True when a process is running `<install>\sinorag.exe` (the agent window,
/// or an MCP client's server). It keeps the indexes mapped, so they cannot
/// be replaced or deleted until it exits. Windows refuses write access to
/// the image of a running program, which is what this probes.
pub fn sinorag_is_running(install_dir: &Path) -> bool {
    const ERROR_SHARING_VIOLATION: i32 = 32;
    match std::fs::OpenOptions::new()
        .write(true)
        .open(install_dir.join("sinorag.exe"))
    {
        Ok(_) => false,
        Err(e) => e.raw_os_error() == Some(ERROR_SHARING_VIOLATION),
    }
}

/// Whether Windows can OCR `code` (a Lege language code such as `eng` or
/// `chi_tra`), or `None` when that cannot be determined. Mirrors how
/// lege-ocr creates its engine, so the answer is the one extraction will get.
pub fn ocr_language_installed(code: &str) -> Option<bool> {
    use windows::Globalization::Language;
    use windows::Media::Ocr::OcrEngine;

    let lower = code.to_ascii_lowercase();
    let tag = match lower.as_str() {
        "eng" => "en",
        "deu" | "ger" => "de",
        "fra" | "fre" => "fr",
        "spa" => "es",
        "ita" => "it",
        "por" => "pt",
        "nld" | "dut" => "nl",
        "chi_sim" => "zh-Hans",
        "chi_tra" => "zh-Hant",
        _ => code,
    };
    let language = Language::CreateLanguage(&windows::core::HSTRING::from(tag)).ok()?;
    Some(OcrEngine::TryCreateFromLanguage(&language).is_ok())
}

/// Bring the window of an already running intake to the front.
pub fn focus_window_titled(title: &str) {
    use windows::Win32::UI::WindowsAndMessaging::{
        FindWindowW, IsIconic, SetForegroundWindow, ShowWindow, SW_RESTORE,
    };
    let title = to_wide_str(title);
    unsafe {
        if let Ok(hwnd) = FindWindowW(PCWSTR::null(), PCWSTR::from_raw(title.as_ptr())) {
            if IsIconic(hwnd).as_bool() {
                let _ = ShowWindow(hwnd, SW_RESTORE);
            }
            let _ = SetForegroundWindow(hwnd);
        }
    }
}

pub fn try_single_instance() -> Option<SingleInstanceGuard> {
    use windows::Win32::System::Threading::CreateMutexW;

    let name = to_wide_str("Local\\SinoRAGDocumentIntake");
    let handle = unsafe {
        CreateMutexW(
            None,
            true,
            PCWSTR::from_raw(name.as_ptr()),
        )
    }
    .ok()?;

    let last_err = unsafe { windows::Win32::Foundation::GetLastError() };
    if last_err == ERROR_ALREADY_EXISTS {
        return None;
    }
    Some(SingleInstanceGuard(handle))
}

pub struct SingleInstanceGuard(windows::Win32::Foundation::HANDLE);

impl Drop for SingleInstanceGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sinorag-intake-folder-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn install_folder_must_be_the_intakes_own() {
        let dir = scratch("classify");
        assert_eq!(classify_install_folder(&dir.join("missing")), InstallFolder::Fresh);
        assert_eq!(classify_install_folder(&dir), InstallFolder::Fresh);

        // Somebody's folder: not to be installed into, never to be deleted.
        fs::write(dir.join("thesis.docx"), b"x").unwrap();
        assert_eq!(classify_install_folder(&dir), InstallFolder::Occupied("thesis.docx".to_string()));
        // ... even with a sinorag.exe lying in it.
        fs::write(dir.join("sinorag.exe"), b"x").unwrap();
        assert!(matches!(classify_install_folder(&dir), InstallFolder::Occupied(_)));
        fs::remove_file(dir.join("thesis.docx")).unwrap();

        // An install interrupted right after the payload was unpacked.
        fs::write(dir.join(".sinorag-payload.7z"), b"x").unwrap();
        assert_eq!(classify_install_folder(&dir), InstallFolder::Installed);

        // A finished one, with whatever SinoRAG wrote next to it since.
        fs::create_dir_all(dir.join("intake")).unwrap();
        fs::create_dir_all(dir.join(".opencode")).unwrap();
        fs::create_dir_all(dir.join(".sinorag")).unwrap();
        fs::write(dir.join("AGENTS.md"), b"x").unwrap();
        fs::create_dir_all(dir.join("data/passages.parquet/source_corpus=documents")).unwrap();
        assert_eq!(classify_install_folder(&dir), InstallFolder::Installed);
        assert!(foreign_entries(&dir).is_empty(), "{:?}", foreign_entries(&dir));

        // What a chat session saved there is the user's, and is named as such.
        fs::write(dir.join("summary.md"), b"x").unwrap();
        fs::create_dir_all(dir.join("Reports")).unwrap();
        assert_eq!(foreign_entries(&dir), vec!["Reports".to_string(), "summary.md".to_string()]);
        assert_eq!(classify_install_folder(&dir), InstallFolder::Installed);
        fs::remove_file(dir.join("summary.md")).unwrap();
        fs::remove_dir(dir.join("Reports")).unwrap();

        // A CBETA install is not the intake's to rebuild or remove.
        fs::create_dir_all(dir.join("data/passages.parquet/source_corpus=cbeta")).unwrap();
        assert_eq!(classify_install_folder(&dir), InstallFolder::OtherCorpus("cbeta".to_string()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ocr_probe_answers_without_failing() {
        // Which languages are installed depends on the machine; a language
        // Windows has never heard of is unavailable everywhere.
        println!("eng: {:?}, chi_tra: {:?}", ocr_language_installed("eng"), ocr_language_installed("chi_tra"));
        assert_ne!(ocr_language_installed("tlh"), Some(true));
    }

    #[test]
    fn a_free_exe_is_not_reported_as_running() {
        let dir = scratch("running");
        assert!(!sinorag_is_running(&dir));
        fs::write(dir.join("sinorag.exe"), b"x").unwrap();
        assert!(!sinorag_is_running(&dir));
        let _ = fs::remove_dir_all(&dir);
    }
}
