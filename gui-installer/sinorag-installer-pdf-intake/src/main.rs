#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

//! SinoRAG Document Intake — Windows GUI.
//!
//! First run: installs SinoRAG (same payload as the CBETA installer) into its
//! own folder, extracts the chosen PDFs / HTML / text files, builds the
//! indexes and creates shortcuts. Later runs (Start Menu → "Add documents to
//! SinoRAG") add more files and rebuild the indexes.

#[cfg(target_os = "windows")]
mod app {
    use anyhow::{anyhow, Context, Result};
    use sinorag_intake::corpus::{installed_payload_id, payload_id, record_payload_id, CorpusStore};
    use sinorag_intake::extract::{DocKind, ExtractOptions};
    use sinorag_intake::mcp_clients::{
        detect_clients, register as register_mcp, registered_anywhere, Change, Client as McpClient, Locations,
        SERVER_NAME,
    };
    use sinorag_intake::pipeline::{intake_root, run_intake, IntakeEvent, IntakeOptions, IntakeSummary};
    use sinorag_intake::win_utils::{
        classify_install_folder, default_install_path, desktop_shortcut_path,
        focus_window_titled, ocr_language_installed, read_install_location, sinorag_is_running,
        start_menu_intake_shortcut_path, start_menu_shortcut_path, to_wide, to_wide_str,
        try_single_instance, write_uninstall_entry, InstallFolder,
    };
    use std::cell::{Cell, RefCell};
    use std::fs;
    use std::os::windows::process::CommandExt;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::rc::Rc;
    use std::thread;
    use windows::core::{Interface, PCWSTR};
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CoUninitialize, IPersistFile, CLSCTX_INPROC_SERVER,
        COINIT_APARTMENTTHREADED,
    };
    use windows::Win32::UI::Shell::{IShellLinkW, ShellLink};
    use winsafe::{self as w, co, gui, prelude::*};

    const PAYLOAD_7Z: &[u8] = include_bytes!(env!("SINORAG_PAYLOAD_PATH"));
    const SHORTCUT_ICON: &[u8] = include_bytes!(env!("SINORAG_ICON_ICO"));
    const EXPECTED_SINORAG_VERSION: &str = env!("EXPECTED_SINORAG_VERSION");
    const INTAKE_EXE: &str = "sinorag-intake.exe";
    const UNINSTALLER_EXE: &str = "sinorag-intake-uninstaller.exe";

    const WINDOW_TITLE: &str = "SinoRAG Document Intake";
    /// Folder name suggested inside a folder that already has things in it.
    const INSTALL_FOLDER_NAME: &str = "SinoRAG-Documents";

    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    // Client size at 100% scaling. The window is not resizable, so with its
    // caption it has to fit the work area of a 1366x768 screen, and of a
    // 1920x1080 one at 150% (720 units high, less the taskbar).
    const W: i32 = 560;
    const H: i32 = 632;

    // Palette shared with the CBETA winsafe installer.
    const CINNABAR: (u8, u8, u8) = (154, 73, 55);
    const CINNABAR_HOVER: (u8, u8, u8) = (175, 91, 69);
    const PAPER: (u8, u8, u8) = (247, 244, 234);
    const PANEL_BG: (u8, u8, u8) = (236, 231, 216);
    const TEXT: (u8, u8, u8) = (23, 36, 38);
    const MUTED: (u8, u8, u8) = (93, 104, 102);
    const BORDER: (u8, u8, u8) = (138, 124, 99);
    const TEXT_INVERSE: (u8, u8, u8) = (255, 249, 238);

    /// (label, Lege OCR code). Windows OCR needs the matching language pack
    /// (Settings → Time & language → Language & region).
    const OCR_LANGUAGES: &[(&str, &str)] = &[
        ("English", "eng"),
        ("Chinese (Traditional)", "chi_tra"),
        ("Chinese (Simplified)", "chi_sim"),
        ("Japanese", "ja"),
        ("Korean", "ko"),
        ("German", "deu"),
        ("French", "fra"),
        ("Spanish", "spa"),
        ("Italian", "ita"),
        ("Portuguese", "por"),
        ("Dutch", "nld"),
    ];

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Phase {
        Ready,
        Running,
        Complete,
        Failed,
    }

    #[derive(Clone, Debug)]
    struct RunRequest {
        install_path: PathBuf,
        files: Vec<PathBuf>,
        ocr_language: String,
        desktop_shortcut: bool,
        start_menu_shortcut: bool,
        build_lexical: bool,
        install_opencode: bool,
        /// AI apps to add the MCP server to (empty: none).
        mcp_clients: Vec<McpClient>,
    }

    #[derive(Debug)]
    enum UiEvent {
        Intake(IntakeEvent),
        OpenCodeMissing,
        Done(Result<IntakeSummary, String>),
    }

    struct ComGuard {
        initialized: bool,
    }

    impl ComGuard {
        fn init_sta() -> Self {
            let initialized = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED).is_ok() };
            Self { initialized }
        }
    }

    impl Drop for ComGuard {
        fn drop(&mut self) {
            if self.initialized {
                unsafe { CoUninitialize() };
            }
        }
    }

    #[derive(Clone)]
    pub struct IntakeWindow {
        wnd: gui::WindowMain,
        subtitle: gui::Label,
        labels: Vec<gui::Label>,
        indexed_label: gui::Label,
        path_edit: gui::Edit,
        browse_btn: gui::Button,
        file_list: gui::ListView<PathBuf>,
        add_files_btn: gui::Button,
        add_folder_btn: gui::Button,
        remove_btn: gui::Button,
        clear_btn: gui::Button,
        ocr_combo: gui::ComboBox,
        desktop_cb: gui::CheckBox,
        start_menu_cb: gui::CheckBox,
        lexical_cb: gui::CheckBox,
        opencode_cb: gui::CheckBox,
        mcp_cb: gui::CheckBox,
        mcp_clients: Rc<Vec<McpClient>>,
        step_label: gui::Label,
        progress_bar: gui::Label,
        log_edit: gui::Edit,
        action_btn: gui::Button,

        phase: Rc<Cell<Phase>>,
        rx: Rc<RefCell<Option<flume::Receiver<UiEvent>>>>,
        progress_pct: Rc<Cell<u32>>,
        opencode_missing: Rc<Cell<bool>>,
    }

    fn label(wnd: &gui::WindowMain, text: &str, x: i32, y: i32, w: i32, h: i32) -> gui::Label {
        gui::Label::new(
            wnd,
            gui::LabelOpts {
                text,
                position: gui::dpi(x, y),
                size: gui::dpi(w, h),
                ..Default::default()
            },
        )
    }

    fn button(wnd: &gui::WindowMain, text: &str, x: i32, y: i32, w: i32, h: i32) -> gui::Button {
        let (bw, bh) = gui::dpi(w, h);
        gui::Button::new(
            wnd,
            gui::ButtonOpts {
                text,
                position: gui::dpi(x, y),
                width: bw,
                height: bh,
                ..Default::default()
            },
        )
    }

    fn checkbox(wnd: &gui::WindowMain, text: &str, x: i32, y: i32, checked: bool) -> gui::CheckBox {
        gui::CheckBox::new(
            wnd,
            gui::CheckBoxOpts {
                text,
                position: gui::dpi(x, y),
                size: gui::dpi(250, 24),
                check_state: if checked { co::BST::CHECKED } else { co::BST::UNCHECKED },
                ..Default::default()
            },
        )
    }

    impl IntakeWindow {
        pub fn create_and_run(initial_files: Vec<PathBuf>) -> w::AnyResult<i32> {
            let _mutex = match try_single_instance() {
                Some(guard) => guard,
                None => {
                    // Already open: show that window instead of doing nothing.
                    focus_window_titled(WINDOW_TITLE);
                    return Ok(0);
                }
            };

            // Re-runs open on the existing install; first runs on the default folder.
            let existing = read_install_location().filter(|p| p.join("sinorag.exe").is_file());
            // On a re-run, offer the shortcuts that are there, so that adding
            // documents does not bring back one the user deleted.
            let reinstall = existing.is_some();
            let has_desktop = desktop_shortcut_path().is_some_and(|p| p.exists());
            let has_start_menu = start_menu_shortcut_path().is_some_and(|p| p.exists())
                || start_menu_intake_shortcut_path().is_some_and(|p| p.exists());
            let install_path = existing.unwrap_or_else(default_install_path);

            let br_paper = w::HBRUSH::CreateSolidBrush(w::COLORREF::from_rgb(PAPER.0, PAPER.1, PAPER.2))?.leak();
            let br_panel =
                w::HBRUSH::CreateSolidBrush(w::COLORREF::from_rgb(PANEL_BG.0, PANEL_BG.1, PANEL_BG.2))?.leak();
            let hfont_subtitle = {
                let mut lf = w::LOGFONT::new_face(-19, "Segoe UI");
                lf.lfWeight = co::FW::SEMIBOLD;
                w::HFONT::CreateFontIndirect(&lf)?.leak()
            };
            let hfont_ui = w::HFONT::CreateFontIndirect(&w::LOGFONT::new_face(-13, "Segoe UI"))?.leak();

            let wnd = gui::WindowMain::new(gui::WindowMainOpts {
                title: WINDOW_TITLE,
                // The icon build.rs embeds as resource 1.
                class_icon: gui::Icon::Id(1),
                size: gui::dpi(W, H),
                style: co::WS::CAPTION | co::WS::SYSMENU | co::WS::MINIMIZEBOX,
                ex_style: co::WS_EX::LEFT | co::WS_EX::ACCEPTFILES,
                ..Default::default()
            });

            let subtitle = gui::Label::new(
                &wnd,
                gui::LabelOpts {
                    text: "Index PDFs, web pages and text files for SinoRAG",
                    position: gui::dpi(20, 14),
                    size: gui::dpi(W - 40, 26),
                    control_style: co::SS::CENTER,
                    ..Default::default()
                },
            );

            let mut labels = Vec::new();
            labels.push(label(&wnd, "Install folder", 20, 46, 300, 20));
            let (edit_w, edit_h) = gui::dpi(W - 92, 26);
            let path_edit = gui::Edit::new(
                &wnd,
                gui::EditOpts {
                    text: &install_path.display().to_string(),
                    position: gui::dpi(20, 66),
                    width: edit_w,
                    height: edit_h,
                    ..Default::default()
                },
            );
            let browse_btn = button(&wnd, "...", W - 64, 65, 44, 28);

            labels.push(label(&wnd, "Documents to add (or drop files and folders here)", 20, 102, 310, 20));
            let indexed_label = gui::Label::new(
                &wnd,
                gui::LabelOpts {
                    text: "",
                    position: gui::dpi(340, 102),
                    size: gui::dpi(W - 360, 20),
                    control_style: co::SS::RIGHT,
                    ..Default::default()
                },
            );

            let columns = [
                ("File", gui::dpi_x(270)),
                ("Type", gui::dpi_x(50)),
                ("Status", gui::dpi_x(180)),
            ];
            let file_list = gui::ListView::<PathBuf>::new(
                &wnd,
                gui::ListViewOpts {
                    position: gui::dpi(20, 124),
                    size: gui::dpi(W - 40, 136),
                    columns: &columns,
                    control_style: co::LVS::REPORT | co::LVS::SHOWSELALWAYS | co::LVS::NOSORTHEADER,
                    control_ex_style: co::LVS_EX::FULLROWSELECT | co::LVS_EX::GRIDLINES,
                    ..Default::default()
                },
            );

            let add_files_btn = button(&wnd, "Add files...", 20, 268, 124, 28);
            let add_folder_btn = button(&wnd, "Add folder...", 152, 268, 124, 28);
            let remove_btn = button(&wnd, "Remove selected", 284, 268, 124, 28);
            let clear_btn = button(&wnd, "Clear list", 416, 268, 124, 28);

            labels.push(label(&wnd, "OCR language (scanned pages)", 20, 312, 200, 20));
            let ocr_names: Vec<&str> = OCR_LANGUAGES.iter().map(|(n, _)| *n).collect();
            let ocr_combo = gui::ComboBox::new(
                &wnd,
                gui::ComboBoxOpts {
                    position: gui::dpi(224, 308),
                    width: gui::dpi_x(220),
                    items: &ocr_names,
                    selected_item: Some(0),
                    ..Default::default()
                },
            );

            let desktop_cb = checkbox(&wnd, "Desktop shortcut", 20, 340, !reinstall || has_desktop);
            let start_menu_cb = checkbox(&wnd, "Start Menu shortcuts", 290, 340, !reinstall || has_start_menu);
            let lexical_cb = checkbox(&wnd, "Phrase and TF-IDF indexes", 20, 364, true);
            // Adding documents to an existing install should not start installing other software.
            let opencode_cb = checkbox(&wnd, "OpenCode integration", 290, 364, !reinstall);
            // Same rule for the AI apps: a re-run keeps their entries current,
            // but does not bring one back that the user took out.
            let mcp_clients = Locations::from_env().map(|loc| detect_clients(&loc)).unwrap_or_default();
            let mcp_label = if mcp_clients.is_empty() {
                "Connect AI apps (none found: Claude, Codex, Cursor, VS Code, Gemini...)".to_string()
            } else {
                let names: Vec<&str> = mcp_clients.iter().map(|c| c.name.as_str()).collect();
                format!("Connect AI apps: {}", names.join(", "))
            };
            let mcp_cb = gui::CheckBox::new(
                &wnd,
                gui::CheckBoxOpts {
                    text: &mcp_label,
                    position: gui::dpi(20, 388),
                    size: gui::dpi(W - 40, 24),
                    check_state: if !mcp_clients.is_empty() && (!reinstall || registered_anywhere(&mcp_clients)) {
                        co::BST::CHECKED
                    } else {
                        co::BST::UNCHECKED
                    },
                    ..Default::default()
                },
            );

            // One line: a status longer than the window ends in "…" instead of
            // wrapping out of sight, and an "&" in a file name stays an "&".
            let step_label = gui::Label::new(
                &wnd,
                gui::LabelOpts {
                    text: "Add documents, then press the button below.",
                    position: gui::dpi(20, 420),
                    size: gui::dpi(W - 40, 20),
                    control_style: co::SS::LEFT | co::SS::NOTIFY | co::SS::NOPREFIX | co::SS::ENDELLIPSIS,
                    ..Default::default()
                },
            );
            let progress_bar = gui::Label::new(
                &wnd,
                gui::LabelOpts {
                    text: "",
                    position: gui::dpi(20, 444),
                    size: gui::dpi(W - 40, 18),
                    control_style: co::SS::OWNERDRAW,
                    ..Default::default()
                },
            );
            let (log_w, log_h) = gui::dpi(W - 40, 100);
            let log_edit = gui::Edit::new(
                &wnd,
                gui::EditOpts {
                    text: "",
                    position: gui::dpi(20, 470),
                    width: log_w,
                    height: log_h,
                    control_style: co::ES::MULTILINE | co::ES::AUTOVSCROLL | co::ES::READONLY | co::ES::WANTRETURN,
                    window_style: co::WS::CHILD | co::WS::VISIBLE | co::WS::BORDER | co::WS::VSCROLL | co::WS::TABSTOP,
                    ..Default::default()
                },
            );
            let (action_w, action_h) = gui::dpi(W - 40, 38);
            let action_btn = gui::Button::new(
                &wnd,
                gui::ButtonOpts {
                    text: "Install and index",
                    position: gui::dpi(20, 580),
                    width: action_w,
                    height: action_h,
                    control_style: co::BS::OWNERDRAW,
                    ..Default::default()
                },
            );

            let me = Self {
                wnd,
                subtitle,
                labels,
                indexed_label,
                path_edit,
                browse_btn,
                file_list,
                add_files_btn,
                add_folder_btn,
                remove_btn,
                clear_btn,
                ocr_combo,
                desktop_cb,
                start_menu_cb,
                lexical_cb,
                opencode_cb,
                mcp_cb,
                mcp_clients: Rc::new(mcp_clients),
                step_label,
                progress_bar,
                log_edit,
                action_btn,
                phase: Rc::new(Cell::new(Phase::Ready)),
                rx: Rc::new(RefCell::new(None)),
                progress_pct: Rc::new(Cell::new(0)),
                opencode_missing: Rc::new(Cell::new(false)),
            };
            me.events(br_paper, br_panel, hfont_subtitle, hfont_ui, initial_files);
            me.wnd.run_main(None)
        }

        fn events(
            &self,
            br_paper: w::HBRUSH,
            br_panel: w::HBRUSH,
            hfont_subtitle: w::HFONT,
            hfont_ui: w::HFONT,
            initial_files: Vec<PathBuf>,
        ) {
            let br_paper_static = unsafe { br_paper.raw_copy() };
            let hfont_btn = unsafe { hfont_ui.raw_copy() };

            let me = self.clone();
            self.wnd.on().wm_create(move |_| {
                me.wnd.hwnd().SetTimer(1, 75, None)?;
                unsafe {
                    let _ = me.subtitle.hwnd().SendMessage(w::msg::wm::SetFont {
                        hfont: hfont_subtitle.raw_copy(),
                        redraw: true,
                    });
                    let mut hwnds: Vec<&w::HWND> = me.labels.iter().map(|l| l.hwnd()).collect();
                    hwnds.extend([
                        me.indexed_label.hwnd(),
                        me.path_edit.hwnd(),
                        me.browse_btn.hwnd(),
                        me.file_list.hwnd(),
                        me.add_files_btn.hwnd(),
                        me.add_folder_btn.hwnd(),
                        me.remove_btn.hwnd(),
                        me.clear_btn.hwnd(),
                        me.ocr_combo.hwnd(),
                        me.desktop_cb.hwnd(),
                        me.start_menu_cb.hwnd(),
                        me.lexical_cb.hwnd(),
                        me.opencode_cb.hwnd(),
                        me.mcp_cb.hwnd(),
                        me.step_label.hwnd(),
                        me.log_edit.hwnd(),
                    ]);
                    for hwnd in hwnds {
                        let _ = hwnd.SendMessage(w::msg::wm::SetFont {
                            hfont: hfont_ui.raw_copy(),
                            redraw: true,
                        });
                    }
                }
                me.add_paths(&initial_files)?;
                me.refresh_install_state()?;
                Ok(0)
            });

            let wnd = self.wnd.clone();
            self.wnd.on().wm_erase_bkgnd(move |p| {
                let rc = wnd.hwnd().GetClientRect()?;
                p.hdc.FillRect(rc, &br_paper)?;
                Ok(1)
            });

            let subtitle = self.subtitle.clone();
            let indexed = self.indexed_label.clone();
            self.wnd.on().wm_ctl_color_static(move |p| {
                let muted = p.hwnd == *subtitle.hwnd() || p.hwnd == *indexed.hwnd();
                let c = if muted { MUTED } else { TEXT };
                p.hdc.SetTextColor(w::COLORREF::from_rgb(c.0, c.1, c.2))?;
                p.hdc.SetBkColor(w::COLORREF::from_rgb(PAPER.0, PAPER.1, PAPER.2))?;
                Ok(unsafe { br_paper_static.raw_copy() })
            });

            self.wnd.on().wm_ctl_color_edit(move |p| {
                p.hdc.SetTextColor(w::COLORREF::from_rgb(TEXT.0, TEXT.1, TEXT.2))?;
                p.hdc.SetBkColor(w::COLORREF::from_rgb(PANEL_BG.0, PANEL_BG.1, PANEL_BG.2))?;
                Ok(unsafe { br_panel.raw_copy() })
            });

            // Owner-drawn progress bar and primary button (same look as the CBETA installer).
            let action = self.action_btn.clone();
            let progress = self.progress_bar.clone();
            let pct = self.progress_pct.clone();
            self.wnd.on().wm(co::WM::DRAWITEM, move |p: w::msg::WndMsg| {
                let dis = unsafe { &*(p.lparam as *const w::DRAWITEMSTRUCT) };
                if dis.hwndItem == *progress.hwnd() {
                    let full = dis.rcItem;
                    let border = w::HBRUSH::CreateSolidBrush(w::COLORREF::from_rgb(BORDER.0, BORDER.1, BORDER.2))?;
                    let trough =
                        w::HBRUSH::CreateSolidBrush(w::COLORREF::from_rgb(PANEL_BG.0, PANEL_BG.1, PANEL_BG.2))?;
                    dis.hDC.FillRect(full, &border)?;
                    let inner = w::RECT {
                        left: full.left + 1,
                        top: full.top + 1,
                        right: full.right - 1,
                        bottom: full.bottom - 1,
                    };
                    dis.hDC.FillRect(inner, &trough)?;
                    let fill_w = (inner.right - inner.left) * pct.get().min(100) as i32 / 100;
                    if fill_w > 0 {
                        let fill =
                            w::HBRUSH::CreateSolidBrush(w::COLORREF::from_rgb(CINNABAR.0, CINNABAR.1, CINNABAR.2))?;
                        dis.hDC.FillRect(
                            w::RECT { right: inner.left + fill_w, ..inner },
                            &fill,
                        )?;
                    }
                    return Ok(1);
                }
                if dis.hwndItem != *action.hwnd() {
                    return Ok(0);
                }
                let fill = if dis.itemState.has(co::ODS::DISABLED) {
                    w::COLORREF::from_rgb(150, 110, 100)
                } else if dis.itemState.has(co::ODS::SELECTED) {
                    w::COLORREF::from_rgb(CINNABAR_HOVER.0, CINNABAR_HOVER.1, CINNABAR_HOVER.2)
                } else {
                    w::COLORREF::from_rgb(CINNABAR.0, CINNABAR.1, CINNABAR.2)
                };
                let brush = w::HBRUSH::CreateSolidBrush(fill)?;
                dis.hDC.FillRect(dis.rcItem, &brush)?;
                let font = unsafe { hfont_btn.raw_copy() };
                let _prev = dis.hDC.SelectObject(&font);
                dis.hDC.SetBkMode(co::BKMODE::TRANSPARENT)?;
                dis.hDC.SetTextColor(w::COLORREF::from_rgb(TEXT_INVERSE.0, TEXT_INVERSE.1, TEXT_INVERSE.2))?;
                let text = action.hwnd().GetWindowText()?;
                dis.hDC.DrawText(&text, dis.rcItem, co::DT::CENTER | co::DT::VCENTER | co::DT::SINGLELINE)?;
                Ok(1)
            });

            let me = self.clone();
            self.browse_btn.on().bn_clicked(move || {
                if let Some(dir) = pick_folder(me.wnd.hwnd(), "Choose the install folder")? {
                    // Picking a folder that has things in it (D:\Apps) means
                    // "install in there", not "this folder is the install":
                    // SinoRAG gets a folder of its own inside it.
                    let dir = match classify_install_folder(&dir) {
                        InstallFolder::Fresh | InstallFolder::Installed => dir,
                        InstallFolder::OtherCorpus(_) | InstallFolder::Occupied(_) => dir.join(INSTALL_FOLDER_NAME),
                    };
                    me.path_edit.hwnd().SetWindowText(&dir.display().to_string())?;
                    me.refresh_install_state()?;
                }
                Ok(())
            });

            let me = self.clone();
            self.path_edit.on().en_kill_focus(move || {
                me.refresh_install_state()?;
                Ok(())
            });

            let me = self.clone();
            self.add_files_btn.on().bn_clicked(move || {
                let files = pick_files(me.wnd.hwnd())?;
                me.add_paths(&files)?;
                Ok(())
            });

            let me = self.clone();
            self.add_folder_btn.on().bn_clicked(move || {
                if let Some(dir) = pick_folder(me.wnd.hwnd(), "Add every PDF, HTML and text file in a folder")? {
                    me.add_paths(&[dir])?;
                }
                Ok(())
            });

            let me = self.clone();
            self.remove_btn.on().bn_clicked(move || {
                if me.phase.get() != Phase::Running {
                    me.file_list.items().delete_selected()?;
                    me.list_was_edited()?;
                }
                Ok(())
            });

            let me = self.clone();
            self.clear_btn.on().bn_clicked(move || {
                if me.phase.get() != Phase::Running {
                    me.file_list.items().delete_all()?;
                    me.list_was_edited()?;
                }
                Ok(())
            });

            let me = self.clone();
            self.wnd.on().wm_drop_files(move |p| {
                if me.phase.get() == Phase::Running {
                    return Ok(());
                }
                let dropped = p
                    .hdrop
                    .DragQueryFile()?
                    .map(|r| r.map(PathBuf::from))
                    .collect::<w::SysResult<Vec<_>>>()?;
                me.add_paths(&dropped)?;
                Ok(())
            });

            let me = self.clone();
            self.action_btn.on().bn_clicked(move || {
                match me.phase.get() {
                    Phase::Running => {}
                    Phase::Complete if me.file_list.items().count() == 0 || me.all_done() => {
                        let path = me.path_edit.hwnd().GetWindowText()?;
                        let _ = Command::new("explorer").arg(path.trim()).spawn();
                    }
                    _ => me.start_run()?,
                }
                Ok(())
            });

            let me = self.clone();
            self.wnd.on().wm_timer(1, move || {
                me.drain_events()?;
                Ok(())
            });
        }

        /// Add files (and folders, expanded) to the list, skipping duplicates
        /// and unsupported types.
        fn add_paths(&self, paths: &[PathBuf]) -> w::AnyResult<()> {
            let mut ignored = 0usize;
            for path in paths {
                let expanded = if path.is_dir() {
                    sinorag_intake::corpus::collect_files(std::slice::from_ref(path))
                } else {
                    match DocKind::from_path(path) {
                        Some(kind) => vec![(path.clone(), kind)],
                        None => {
                            ignored += 1;
                            continue;
                        }
                    }
                };
                for (file, kind) in expanded {
                    let exists = self.file_list.items().iter().any(|item| *item.data().borrow() == file);
                    if exists {
                        continue;
                    }
                    let name = file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    self.file_list.items().add(&[name.as_str(), kind.label(), "queued"], None, file)?;
                }
            }
            if ignored > 0 {
                self.append_log(&format!(
                    "Ignored {ignored} item(s): only .pdf, .html/.htm and .txt files are supported."
                ))?;
            }
            if self.phase.get() != Phase::Running {
                self.phase.set(Phase::Ready);
            }
            self.refresh_install_state()?;
            Ok(())
        }

        /// After "Remove selected" / "Clear list". Emptying the list ends the
        /// finished state, so the button offers a run again ("Rebuild
        /// indexes") instead of staying on "Done - open the install folder".
        fn list_was_edited(&self) -> w::AnyResult<()> {
            if self.file_list.items().count() == 0 {
                self.phase.set(Phase::Ready);
            }
            self.refresh_install_state()
        }

        fn queued_files(&self) -> Vec<PathBuf> {
            self.file_list.items().iter().map(|item| item.data().borrow().clone()).collect()
        }

        fn all_done(&self) -> bool {
            self.file_list.items().iter().all(|item| item.text(2) != "queued")
        }

        /// Update the "already indexed" note and the action label for the current folder.
        fn refresh_install_state(&self) -> w::AnyResult<()> {
            let path = PathBuf::from(self.path_edit.hwnd().GetWindowText()?.trim());
            let installed = path.join("sinorag.exe").is_file();
            let docs = CorpusStore::peek(&intake_root(&path)).map(|m| m.documents.len()).unwrap_or(0);
            let note = match (installed, docs) {
                (false, _) => "Not installed here yet".to_string(),
                (true, 0) => "Installed; no documents yet".to_string(),
                (true, 1) => "1 document indexed".to_string(),
                (true, n) => format!("{n} documents indexed"),
            };
            self.indexed_label.hwnd().SetWindowText(&note)?;
            if self.phase.get() == Phase::Running {
                return Ok(());
            }
            let queued = self.file_list.items().count();
            let label = match (self.phase.get(), installed, queued) {
                (Phase::Complete, _, _) if self.all_done() => "Done - open the install folder".to_string(),
                (_, false, 0) => "Install".to_string(),
                (_, false, _) => format!("Install and index {queued} file(s)"),
                (_, true, 0) => "Rebuild indexes".to_string(),
                (_, true, _) => format!("Index {queued} file(s) and rebuild indexes"),
            };
            self.action_btn.hwnd().SetWindowText(&label)?;
            self.action_btn.hwnd().InvalidateRect(None, false)?;
            Ok(())
        }

        fn start_run(&self) -> w::AnyResult<()> {
            let path_text = self.path_edit.hwnd().GetWindowText()?;
            let install_path = match validate_install_path(path_text.trim()) {
                Ok(p) => p,
                Err(e) => {
                    self.set_step("The install folder is not usable.")?;
                    self.append_log(&format!("Error: {e}"))?;
                    return Ok(());
                }
            };
            let ocr_language = self
                .ocr_combo
                .items()
                .selected_index()
                .and_then(|i| OCR_LANGUAGES.get(i as usize))
                .map(|(_, code)| code.to_string())
                .unwrap_or_else(|| "eng".to_string());
            let request = RunRequest {
                install_path: install_path.clone(),
                files: self.queued_files(),
                ocr_language,
                desktop_shortcut: self.desktop_cb.is_checked(),
                start_menu_shortcut: self.start_menu_cb.is_checked(),
                build_lexical: self.lexical_cb.is_checked(),
                install_opencode: self.opencode_cb.is_checked(),
                mcp_clients: if self.mcp_cb.is_checked() { (*self.mcp_clients).clone() } else { Vec::new() },
            };

            for item in self.file_list.items().iter() {
                item.set_text(2, "queued")?;
            }
            self.phase.set(Phase::Running);
            self.opencode_missing.set(false);
            self.progress_pct.set(0);
            self.set_controls_enabled(false)?;
            self.action_btn.hwnd().SetWindowText("Working...")?;
            self.set_step("Starting")?;
            self.repaint_progress()?;
            self.log_edit.hwnd().SetWindowText(&format!("Install folder: {}\r\n", install_path.display()))?;

            let (tx, rx) = flume::unbounded::<UiEvent>();
            *self.rx.borrow_mut() = Some(rx);
            thread::spawn(move || {
                let result = run_impl(request, &tx).map_err(|e| format!("{e:#}"));
                let _ = tx.send(UiEvent::Done(result));
            });
            Ok(())
        }

        fn drain_events(&self) -> w::AnyResult<()> {
            let Some(rx) = self.rx.borrow().as_ref().cloned() else {
                return Ok(());
            };
            for event in rx.try_iter().take(100) {
                let done = matches!(event, UiEvent::Done(_));
                self.apply_event(event)?;
                if done {
                    *self.rx.borrow_mut() = None;
                    break;
                }
            }
            Ok(())
        }

        fn apply_event(&self, event: UiEvent) -> w::AnyResult<()> {
            match event {
                UiEvent::Intake(IntakeEvent::Step(text, progress)) => {
                    let pct = ((progress.clamp(0.0, 1.0) * 100.0).round() as u32).max(self.progress_pct.get());
                    self.progress_pct.set(pct);
                    self.set_step(&text)?;
                    self.repaint_progress()?;
                }
                UiEvent::Intake(IntakeEvent::Log(line)) => self.append_log(&line)?,
                UiEvent::Intake(IntakeEvent::File { path, status }) => {
                    if let Some(item) = self.file_list.items().iter().find(|i| *i.data().borrow() == path) {
                        item.set_text(2, &status)?;
                        let _ = item.ensure_visible();
                    }
                }
                UiEvent::OpenCodeMissing => self.opencode_missing.set(true),
                UiEvent::Done(Ok(summary)) => {
                    self.phase.set(Phase::Complete);
                    self.progress_pct.set(100);
                    self.repaint_progress()?;
                    // The status line is one short line; the log has the detail.
                    let mut status = if summary.total_documents == 0 {
                        "Installed. Add documents to index them.".to_string()
                    } else {
                        format!(
                            "Done: {} added, {} already indexed, {} failed.",
                            summary.added, summary.skipped, summary.failed
                        )
                    };
                    self.append_log(&format!(
                        "{status} Corpus: {} documents, {} passages.",
                        summary.total_documents, summary.total_passages
                    ))?;
                    if self.opencode_missing.get() {
                        status.push_str(" OpenCode not found (see log).");
                    }
                    self.set_step(&status)?;
                    if summary.rebuilt {
                        let exe = PathBuf::from(self.path_edit.hwnd().GetWindowText()?.trim()).join("sinorag.exe");
                        // Name a way in that this run actually created.
                        let opener = if self.start_menu_cb.is_checked() {
                            "open \"SinoRAG Documents\" from the Start Menu".to_string()
                        } else if self.desktop_cb.is_checked() {
                            "open the \"SinoRAG Documents\" shortcut on the desktop".to_string()
                        } else {
                            format!("run \"{}\" agent", exe.display())
                        };
                        self.append_log(&format!(
                            "To ask questions: {opener}. It starts OpenCode with these documents connected."
                        ))?;
                        self.append_log(&format!(
                            "To use another MCP client (Claude Desktop, Claude Code, Cursor, ...): add a \
                             local (stdio) server whose command is \"{}\" with the single argument: mcp",
                            exe.display()
                        ))?;
                        self.append_log(
                            "Close those before adding more documents here: a running session keeps the \
                             indexes open.",
                        )?;
                    }
                    self.set_controls_enabled(true)?;
                    self.refresh_install_state()?;
                }
                UiEvent::Done(Err(error)) => {
                    self.phase.set(Phase::Failed);
                    self.set_step("Failed. See the log; fix the problem and press the button to retry.")?;
                    self.append_log(&format!("Failed: {error}"))?;
                    self.set_controls_enabled(true)?;
                    self.refresh_install_state()?;
                }
            }
            Ok(())
        }

        fn set_controls_enabled(&self, enabled: bool) -> w::AnyResult<()> {
            for hwnd in [
                self.path_edit.hwnd(),
                self.browse_btn.hwnd(),
                self.add_files_btn.hwnd(),
                self.add_folder_btn.hwnd(),
                self.remove_btn.hwnd(),
                self.clear_btn.hwnd(),
                self.ocr_combo.hwnd(),
                self.desktop_cb.hwnd(),
                self.start_menu_cb.hwnd(),
                self.lexical_cb.hwnd(),
                self.opencode_cb.hwnd(),
                self.mcp_cb.hwnd(),
            ] {
                hwnd.EnableWindow(enabled);
            }
            Ok(())
        }

        fn set_step(&self, text: &str) -> w::SysResult<()> {
            self.step_label.hwnd().SetWindowText(text)
        }

        fn repaint_progress(&self) -> w::SysResult<()> {
            self.progress_bar.hwnd().InvalidateRect(None, false)
        }

        fn append_log(&self, line: &str) -> w::SysResult<()> {
            let hwnd = self.log_edit.hwnd();
            let mut text = hwnd.GetWindowText()?;
            // Keep the control responsive on very long runs.
            if text.len() > 200_000 {
                let cut = text.len() - 150_000;
                let cut = text[cut..].find("\r\n").map_or(cut, |i| cut + i + 2);
                text = text[cut..].to_string();
            }
            if !text.is_empty() && !text.ends_with("\r\n") {
                text.push_str("\r\n");
            }
            text.push_str(&line.replace('\n', "\r\n"));
            text.push_str("\r\n");
            hwnd.SetWindowText(&text)?;
            unsafe {
                hwnd.SendMessage(w::msg::em::SetSel { start: -1, end: -1 });
                hwnd.SendMessage(w::msg::em::ScrollCaret {});
            }
            Ok(())
        }
    }

    fn pick_files(parent: &w::HWND) -> w::AnyResult<Vec<PathBuf>> {
        let dialog = w::CoCreateInstance::<w::IFileOpenDialog>(
            &co::CLSID::FileOpenDialog,
            None::<&w::IUnknown>,
            co::CLSCTX::INPROC_SERVER,
        )?;
        dialog.SetOptions(
            dialog.GetOptions()? | co::FOS::FORCEFILESYSTEM | co::FOS::FILEMUSTEXIST | co::FOS::ALLOWMULTISELECT,
        )?;
        dialog.SetFileTypes(&[
            ("Documents (PDF, HTML, text)", "*.pdf;*.html;*.htm;*.xhtml;*.txt"),
            ("PDF documents", "*.pdf"),
            ("Web pages", "*.html;*.htm;*.xhtml"),
            ("Text files", "*.txt"),
        ])?;
        dialog.SetFileTypeIndex(1)?;
        dialog.SetTitle("Add documents")?;
        if !dialog.Show(parent)? {
            return Ok(Vec::new());
        }
        let paths = dialog
            .GetResults()?
            .iter()?
            .map(|item| item.and_then(|i| i.GetDisplayName(co::SIGDN::FILESYSPATH)).map(PathBuf::from))
            .collect::<w::HrResult<Vec<_>>>()?;
        Ok(paths)
    }

    fn pick_folder(parent: &w::HWND, title: &str) -> w::AnyResult<Option<PathBuf>> {
        let dialog = w::CoCreateInstance::<w::IFileOpenDialog>(
            &co::CLSID::FileOpenDialog,
            None::<&w::IUnknown>,
            co::CLSCTX::INPROC_SERVER,
        )?;
        dialog.SetOptions(dialog.GetOptions()? | co::FOS::FORCEFILESYSTEM | co::FOS::PICKFOLDERS)?;
        dialog.SetTitle(title)?;
        if !dialog.Show(parent)? {
            return Ok(None);
        }
        let path = dialog.GetResult()?.GetDisplayName(co::SIGDN::FILESYSPATH)?;
        Ok(Some(PathBuf::from(path)))
    }

    pub fn run() {
        // Files passed on the command line (e.g. dropped on the exe) are queued.
        let files: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
        if let Err(e) = IntakeWindow::create_and_run(files) {
            let _ = w::HWND::NULL.MessageBox(
                &format!("SinoRAG Document Intake could not start:\n{e}"),
                "SinoRAG Document Intake",
                co::MB::ICONERROR,
            );
        }
    }

    // ── Worker ───────────────────────────────────────────────────────────────

    fn run_impl(request: RunRequest, tx: &flume::Sender<UiEvent>) -> Result<IntakeSummary> {
        let step = |text: &str, p: f32| {
            let _ = tx.send(UiEvent::Intake(IntakeEvent::Step(text.to_string(), p)));
        };
        let log = |text: String| {
            let _ = tx.send(UiEvent::Intake(IntakeEvent::Log(text)));
        };

        fs::create_dir_all(&request.install_path)
            .with_context(|| format!("creating {}", request.install_path.display()))?;
        let exe_path = request.install_path.join("sinorag.exe");
        let icon_path = request.install_path.join("SinoRAG.ico");

        // A running sinorag.exe cannot be upgraded, and it keeps the indexes
        // mapped so they cannot be replaced. Say so before changing anything.
        if sinorag_is_running(&request.install_path) {
            return Err(anyhow!(
                "SinoRAG is running from this folder (the agent window, or an MCP client that started \
                 `sinorag mcp`). Close it, then press the button again."
            ));
        }

        // ── Install / upgrade SinoRAG itself ────────────────────────────────
        step("Checking SinoRAG", 0.01);
        // The same version is not the same program: compare the payload this
        // installer carries with the one the folder was last given.
        let payload = payload_id(PAYLOAD_7Z);
        let same_payload = installed_payload_id(&request.install_path).as_deref() == Some(payload.as_str());
        match installed_sinorag_version(&exe_path) {
            Some(v) if v == EXPECTED_SINORAG_VERSION && same_payload => {
                log(format!("SinoRAG {v} is already installed."))
            }
            other => {
                match other {
                    Some(v) if v == EXPECTED_SINORAG_VERSION => {
                        log(format!("Replacing SinoRAG {v} with this installer's build of it."))
                    }
                    Some(v) => log(format!("Upgrading SinoRAG {v} → {EXPECTED_SINORAG_VERSION}.")),
                    None => {}
                }
                step("Installing SinoRAG", 0.02);
                extract_payload(&request.install_path)?;
                record_payload_id(&request.install_path, &payload)?;
                match installed_sinorag_version(&exe_path) {
                    Some(v) if v == EXPECTED_SINORAG_VERSION => log(format!("Installed {}", exe_path.display())),
                    // build.rs refuses such a payload; this is for a build that skipped the check.
                    Some(v) => log(format!(
                        "Warning: this installer carries SinoRAG {v}, not {EXPECTED_SINORAG_VERSION}. \
                         Searches may not match the indexes built here."
                    )),
                    None => {
                        return Err(anyhow!(
                            "{} was installed but does not start (`sinorag.exe --version` failed)",
                            exe_path.display()
                        ))
                    }
                }
            }
        }
        fs::write(&icon_path, SHORTCUT_ICON).with_context(|| format!("writing {}", icon_path.display()))?;
        deploy_self(&request.install_path, &log)?;

        // Shortcuts and the uninstall entry come before the long part, so an
        // install whose indexing fails or is closed can still be re-opened
        // from the Start Menu and removed from Settings.
        step("Creating shortcuts", 0.04);
        create_shortcuts(&request, &exe_path, &icon_path, &log)?;
        step("Registering uninstaller", 0.05);
        write_uninstall_entry(&request.install_path, EXPECTED_SINORAG_VERSION);
        if !request.mcp_clients.is_empty() {
            step("Connecting AI apps", 0.055);
            connect_ai_apps(&request.mcp_clients, &exe_path, &log);
        }

        // ── Intake + index rebuild ──────────────────────────────────────────
        let queued_pdf = request.files.iter().any(|f| DocKind::from_path(f) == Some(DocKind::Pdf));
        if queued_pdf && ocr_language_installed(&request.ocr_language) == Some(false) {
            let name = OCR_LANGUAGES
                .iter()
                .find(|(_, code)| *code == request.ocr_language)
                .map_or(request.ocr_language.as_str(), |(name, _)| *name);
            log(format!(
                "Warning: Windows has no OCR support for {name} installed, so scanned PDF pages will \
                 come out empty. Add the language in Settings > Time & language > Language & region, \
                 then add the affected files again."
            ));
        }
        let tx_events = tx.clone();
        let emit = move |event: IntakeEvent| {
            let _ = tx_events.send(UiEvent::Intake(event));
        };
        let opts = IntakeOptions {
            root: request.install_path.clone(),
            inputs: request.files.clone(),
            extract: ExtractOptions {
                ocr_language: request.ocr_language.clone(),
                ..Default::default()
            },
            build_lexical: request.build_lexical,
            // With nothing queued the button reads "Rebuild indexes": do that.
            force_rebuild: request.files.is_empty(),
            progress_band: (0.06, 0.94),
        };
        let has_docs = CorpusStore::peek(&intake_root(&request.install_path))
            .is_some_and(|m| !m.documents.is_empty());
        let summary = if request.files.is_empty() && !has_docs {
            log("No documents queued; installing only. Run this again to add documents.".to_string());
            IntakeSummary::default()
        } else {
            run_intake(&opts, &emit)?
        };

        // ── OpenCode ────────────────────────────────────────────────────────
        if request.install_opencode {
            install_or_verify_opencode(tx);
        }
        step("Finished", 1.0);
        Ok(summary)
    }

    /// Add `sinorag.exe mcp` to each AI app's MCP config. A failure is logged
    /// and does not stop the run: the README says how to add it by hand.
    fn connect_ai_apps(clients: &[McpClient], exe_path: &Path, log: &dyn Fn(String)) {
        let mut connected = 0usize;
        for outcome in register_mcp(clients, exe_path) {
            match outcome.result {
                Ok(change) => {
                    connected += 1;
                    let what = match change {
                        Change::Added => "added",
                        Change::Updated => "updated",
                        _ => "already set up",
                    };
                    log(format!(
                        "{}: MCP server \"{SERVER_NAME}\" {what} ({}).",
                        outcome.client,
                        outcome.config.display()
                    ));
                }
                Err(e) => log(format!("Warning: {} was not connected: {e}", outcome.client)),
            }
        }
        if connected > 0 {
            log("Restart those apps to pick up the server. While one of them is running it keeps \
                 SinoRAG open, so quit it before adding more documents."
                .to_string());
        }
    }

    fn extract_payload(install_path: &Path) -> Result<()> {
        let archive = install_path.join(".sinorag-payload.7z");
        fs::write(&archive, PAYLOAD_7Z).with_context(|| format!("writing {}", archive.display()))?;
        let result = sevenz_rust::decompress_file(&archive, install_path)
            .map_err(|e| anyhow!("failed to extract payload archive: {e}"));
        let _ = fs::remove_file(&archive);
        result
    }

    /// Copy this intake exe and its uninstaller into the install folder so the
    /// Start Menu entry can re-open intake later.
    fn deploy_self(install_path: &Path, log: &dyn Fn(String)) -> Result<()> {
        let current = std::env::current_exe().context("locating the intake exe")?;
        let here = current.parent().map(Path::to_path_buf).unwrap_or_default();
        let same_dir = fs::canonicalize(&here).ok() == fs::canonicalize(install_path).ok();
        if same_dir {
            return Ok(());
        }
        let target = install_path.join(INTAKE_EXE);
        fs::copy(&current, &target).with_context(|| format!("copying intake to {}", target.display()))?;
        let uninstaller = here.join(UNINSTALLER_EXE);
        if uninstaller.is_file() {
            fs::copy(&uninstaller, install_path.join(UNINSTALLER_EXE))
                .with_context(|| format!("copying {UNINSTALLER_EXE}"))?;
        } else {
            log(format!("Warning: {UNINSTALLER_EXE} was not found next to the intake exe."));
        }
        Ok(())
    }

    fn create_shortcuts(
        request: &RunRequest,
        exe_path: &Path,
        icon_path: &Path,
        log: &dyn Fn(String),
    ) -> Result<()> {
        let _com = ComGuard::init_sta();
        let dir = &request.install_path;
        if request.desktop_shortcut {
            if let Some(path) = desktop_shortcut_path() {
                create_windows_shortcut(exe_path, "agent", &path, dir, icon_path)?;
                log(format!("Desktop shortcut: {}", path.display()));
            }
        }
        if request.start_menu_shortcut {
            if let Some(path) = start_menu_shortcut_path() {
                if let Some(folder) = path.parent() {
                    fs::create_dir_all(folder)?;
                }
                create_windows_shortcut(exe_path, "agent", &path, dir, icon_path)?;
                log(format!("Start Menu shortcut: {}", path.display()));
            }
            if let Some(path) = start_menu_intake_shortcut_path() {
                let intake = dir.join(INTAKE_EXE);
                if intake.is_file() {
                    create_windows_shortcut(&intake, "", &path, dir, icon_path)?;
                    log(format!("Start Menu shortcut: {}", path.display()));
                }
            }
        }
        Ok(())
    }

    fn create_windows_shortcut(
        target: &Path,
        arguments: &str,
        shortcut_path: &Path,
        working_dir: &Path,
        icon_path: &Path,
    ) -> Result<()> {
        unsafe {
            let link: IShellLinkW = CoCreateInstance(&ShellLink, None, CLSCTX_INPROC_SERVER)
                .map_err(|e| anyhow!("creating shell link: {e:?}"))?;
            let t = to_wide(target);
            link.SetPath(PCWSTR::from_raw(t.as_ptr())).map_err(|e| anyhow!("shortcut target: {e:?}"))?;
            let a = to_wide_str(arguments);
            link.SetArguments(PCWSTR::from_raw(a.as_ptr())).map_err(|e| anyhow!("shortcut arguments: {e:?}"))?;
            let wd = to_wide(working_dir);
            link.SetWorkingDirectory(PCWSTR::from_raw(wd.as_ptr()))
                .map_err(|e| anyhow!("shortcut working dir: {e:?}"))?;
            let ic = to_wide(icon_path);
            link.SetIconLocation(PCWSTR::from_raw(ic.as_ptr()), 0).map_err(|e| anyhow!("shortcut icon: {e:?}"))?;
            let file: IPersistFile = link.cast().map_err(|e| anyhow!("IPersistFile: {e:?}"))?;
            let sp = to_wide(shortcut_path);
            file.Save(PCWSTR::from_raw(sp.as_ptr()), true)
                .map_err(|e| anyhow!("saving shortcut {}: {e:?}", shortcut_path.display()))?;
        }
        Ok(())
    }

    /// Make sure `sinorag agent` will find OpenCode, installing it if it can.
    /// Nothing here fails the run: the documents are indexed by now, and
    /// OpenCode can be installed afterwards.
    fn install_or_verify_opencode(tx: &flume::Sender<UiEvent>) {
        use sinorag::which::locate_on_path;

        let step = |text: &str, p: f32| {
            let _ = tx.send(UiEvent::Intake(IntakeEvent::Step(text.to_string(), p)));
        };
        let log = |text: String| {
            let _ = tx.send(UiEvent::Intake(IntakeEvent::Log(text)));
        };
        // The lookup `sinorag agent` itself does, so "installed" means the
        // shortcut will work. It honours PATHEXT, which matters twice: npm
        // installs `opencode.cmd`, and npm is `npm.cmd`; neither is found by
        // spawning the bare name.
        let opencode_works = || locate_on_path("opencode").is_some_and(|path| command_ok(&path, &["--version"]));

        step("Checking OpenCode", 0.95);
        if opencode_works() {
            log("OpenCode is installed.".to_string());
            return;
        }
        if locate_on_path("bash").is_some() {
            step("Installing OpenCode", 0.955);
            let installer = Command::new("powershell")
                .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "curl.exe -fsSL https://opencode.ai/install | bash"])
                .creation_flags(CREATE_NO_WINDOW)
                .status();
            match installer {
                Ok(status) if status.success() && opencode_works() => {
                    log("OpenCode installed.".to_string());
                    return;
                }
                Ok(status) => log(format!("The OpenCode installer finished ({status}) without putting opencode on PATH.")),
                Err(e) => log(format!("Could not run the OpenCode installer: {e}")),
            }
        }
        if let Some(npm) = locate_on_path("npm") {
            step("Installing OpenCode with npm", 0.96);
            let installer = Command::new(&npm)
                .args(["install", "-g", "opencode-ai"])
                .creation_flags(CREATE_NO_WINDOW)
                .status();
            match installer {
                Ok(status) if status.success() && opencode_works() => {
                    log("OpenCode installed.".to_string());
                    return;
                }
                Ok(status) => log(format!("npm install -g opencode-ai finished ({status}) without putting opencode on PATH.")),
                Err(e) => log(format!("Could not run {}: {e}", npm.display())),
            }
        }
        log("Warning: OpenCode was not found. Install it (https://opencode.ai), run `opencode auth login`, \
             then use the SinoRAG Documents shortcut."
            .to_string());
        let _ = tx.send(UiEvent::OpenCodeMissing);
    }

    fn command_ok(program: &Path, args: &[&str]) -> bool {
        Command::new(program)
            .args(args)
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn installed_sinorag_version(exe: &Path) -> Option<String> {
        if !exe.is_file() {
            return None;
        }
        let out = Command::new(exe).arg("--version").creation_flags(CREATE_NO_WINDOW).output().ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout).split_whitespace().last().map(str::to_string)
    }

    fn validate_install_path(value: &str) -> Result<PathBuf, String> {
        if value.is_empty() {
            return Err("install folder is empty".to_string());
        }
        let path = PathBuf::from(value);
        if !path.is_absolute() {
            return Err("install folder must be an absolute path".to_string());
        }
        let parent = path.parent().ok_or("install folder cannot be a whole drive; choose a folder on it")?;
        // The folder becomes SinoRAG's alone: every run rebuilds all indexes
        // in it and the uninstaller deletes it whole.
        match classify_install_folder(&path) {
            InstallFolder::Fresh | InstallFolder::Installed => {}
            InstallFolder::OtherCorpus(corpus) => {
                return Err(format!(
                    "{} is a SinoRAG install that holds the \"{corpus}\" corpus. Document Intake rebuilds \
                     every index in its folder, so it needs a folder of its own.",
                    path.display()
                ))
            }
            InstallFolder::Occupied(entry) => {
                return Err(format!(
                    "{} already has other files in it (\"{entry}\"). Choose an empty or new folder, such \
                     as {}: uninstalling deletes the whole install folder.",
                    path.display(),
                    path.join(INSTALL_FOLDER_NAME).display()
                ))
            }
        }
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        // Probe where the files will go: an existing install folder can be
        // writable when its parent is not, and the other way round.
        let probe_in = if path.is_dir() { path.as_path() } else { parent };
        let probe = probe_in.join(format!(".sinorag-write-test-{}", std::process::id()));
        fs::create_dir(&probe).map_err(|e| format!("{} is not writable: {e}", probe_in.display()))?;
        let _ = fs::remove_dir(&probe);
        Ok(path)
    }
}

#[cfg(target_os = "windows")]
fn main() {
    app::run();
}

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("SinoRAG Document Intake is Windows-only. Use `sinorag ingest cef` elsewhere.");
}
