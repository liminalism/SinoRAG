#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("sinorag-intake-uninstaller is Windows-only.");
}

#[cfg(target_os = "windows")]
fn main() {
    use sinorag_intake::win_utils::{
        classify_install_folder, delete_uninstall_entry, desktop_shortcut_path, foreign_entries,
        read_install_location, sinorag_is_running, start_menu_intake_shortcut_path,
        start_menu_shortcut_path, try_single_instance, InstallFolder,
    };
    use std::io::{self, Write};
    use std::path::PathBuf;

    /// Stop without removing anything. This usually runs in its own console
    /// window (from Settings, or by double-click), which closes on exit, so
    /// wait for a key or the reason is never seen.
    fn refuse(reason: &str) -> ! {
        eprintln!();
        eprintln!("{reason}");
        eprintln!("Nothing was removed.");
        print!("Press Enter to close. ");
        let _ = io::stdout().flush();
        let _ = io::stdin().read_line(&mut String::new());
        std::process::exit(1);
    }

    let registered = read_install_location();
    let install_dir = registered.clone().unwrap_or_else(|| {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(r"C:\SinoRAG-Documents"))
    });

    println!("SinoRAG Documents Uninstaller");
    println!("-----------------------------");
    println!("Install location: {}", install_dir.display());

    // The folder is deleted whole, so be sure it is the install. Without a
    // registry entry the location is only a guess (the folder this program
    // is in, which could be Downloads), and has to look like one.
    let remove_folder = match classify_install_folder(&install_dir) {
        InstallFolder::Installed
            if registered.is_some() || install_dir.join("sinorag.exe").is_file() =>
        {
            true
        }
        InstallFolder::Fresh if registered.is_some() => false,
        InstallFolder::Installed | InstallFolder::Fresh => {
            refuse("No SinoRAG Documents install was found: there is no uninstall entry, and that folder is not one.")
        }
        InstallFolder::OtherCorpus(corpus) => refuse(&format!(
            "That folder also holds the \"{corpus}\" corpus of another SinoRAG install, \
             so it will not be deleted. Uninstall that one instead, or delete the folder by hand."
        )),
        InstallFolder::Occupied(entry) => refuse(&format!(
            "That folder does not look like a SinoRAG Documents install (it contains \"{entry}\"), \
             so it will not be deleted."
        )),
    };

    // Both keep files in the folder open; removing it would half-succeed.
    let _instance = try_single_instance().unwrap_or_else(|| {
        refuse("The SinoRAG Document Intake window is open. Close it and run the uninstaller again.")
    });
    if sinorag_is_running(&install_dir) {
        refuse(
            "SinoRAG is running from that folder (the agent window, or an MCP client that started \
             `sinorag mcp`). Close it and run the uninstaller again.",
        );
    }

    println!();
    if remove_folder {
        println!("This removes SinoRAG, the indexed document text and all indexes.");
        println!("EVERYTHING in that folder is deleted.");
        println!("Your original PDF, HTML and text files are not touched.");
        // Anything else there was saved by the user or by a chat session.
        let others = foreign_entries(&install_dir);
        if !others.is_empty() {
            println!();
            println!("The folder also holds these, which SinoRAG did not put there. They are deleted too;");
            println!("move them out first if you want to keep them:");
            for name in others.iter().take(15) {
                println!("  {name}");
            }
            if others.len() > 15 {
                println!("  ... and {} more", others.len() - 15);
            }
            println!();
        }
    } else {
        println!("That folder is already gone. This removes the shortcuts and the uninstall entry.");
    }
    print!("Uninstall SinoRAG Documents? [y/N] ");
    let _ = io::stdout().flush();

    let mut answer = String::new();
    if io::stdin().read_line(&mut answer).is_err() {
        eprintln!("Failed to read input. Aborting.");
        std::process::exit(1);
    }

    // A piped answer can arrive with a byte-order mark in front of it.
    let answer = answer.trim_start_matches('\u{feff}').trim().to_lowercase();
    if answer != "y" && answer != "yes" {
        println!("Aborted.");
        return;
    }

    for path in [desktop_shortcut_path(), start_menu_intake_shortcut_path(), start_menu_shortcut_path()]
        .into_iter()
        .flatten()
    {
        if path.exists() {
            match std::fs::remove_file(&path) {
                Ok(()) => println!("Removed: {}", path.display()),
                Err(e) => eprintln!("Warning: could not remove {}: {e}", path.display()),
            }
        }
    }
    // The Start Menu folder is shared with the CBETA install; this only
    // removes it when it is empty.
    if let Some(folder) = start_menu_shortcut_path().and_then(|p| p.parent().map(PathBuf::from)) {
        let _ = std::fs::remove_dir(folder);
    }

    delete_uninstall_entry();
    println!("Removed uninstall registry entry.");

    if !remove_folder {
        println!("SinoRAG Documents has been uninstalled.");
        return;
    }

    // This program is inside the folder, so the folder can only go once it
    // has exited: hand the removal to a detached cmd that waits a moment.
    use std::os::windows::process::CommandExt;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    println!("Scheduling install directory removal...");
    let command_line = format!(
        "/c \"ping 127.0.0.1 -n 3 >nul & rmdir /s /q \"{}\"\"",
        install_dir.display()
    );
    match std::process::Command::new("cmd")
        // Verbatim: cmd does not understand the \" escaping that `.args`
        // would apply to the quoted path, and rmdir then fails on it.
        .raw_arg(&command_line)
        // Not the install folder: a process's current directory cannot be removed.
        .current_dir(std::env::temp_dir())
        .creation_flags(DETACHED_PROCESS)
        .spawn()
    {
        Ok(_) => println!("SinoRAG Documents has been uninstalled."),
        Err(e) => {
            eprintln!(
                "Could not schedule directory removal: {e}\n\
                 Please delete {} manually.",
                install_dir.display()
            );
        }
    }
}
