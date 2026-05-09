"""
launcher.py — AI EA v20  Full Control Centre
=============================================
Equivalent to running everything from VSCode:
  - Start / Stop / Restart  ai_ea.py
  - Run trainer.py, run_backtest.py, Optimizer.py, or any .py script
  - Each script gets its own live log tab with colour coding
  - Kill individual processes at any time
  - Full .env editor with tabbed sections + raw editor, Save and Apply
  - Open project folder / log folder / .env in system editor
"""

import os
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

# ── Path resolution (frozen EXE vs raw .py) ──────────────────────────────────
if getattr(sys, "frozen", False):
    BASE = sys._MEIPASS                        # extracted bundle (read-only)
    WORK = os.path.dirname(sys.executable)     # folder the .exe lives in
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
    WORK = BASE

ENV_PATH  = os.path.join(WORK, ".env")
LOG_DIR   = os.path.join(WORK, "logs")
LOG_PATH  = os.path.join(LOG_DIR, "ai_ea.log")

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#0d1117"
BG2       = "#161b22"
BG3       = "#21262d"
BG4       = "#0a0e14"
ACCENT    = "#238636"
ACCENT_H  = "#2ea043"
DANGER    = "#da3633"
DANGER_H  = "#f85149"
WARN_C    = "#d29922"
GOLD      = "#e3b341"
BLUE      = "#1f6feb"
BLUE_H    = "#388bfd"
PURPLE    = "#8957e5"
TEXT      = "#c9d1d9"
TEXT_DIM  = "#8b949e"
BORDER    = "#30363d"
MONO      = ("Consolas", 9)
SANS      = ("Segoe UI", 9)
SANS_B    = ("Segoe UI", 9, "bold")
HEAD      = ("Segoe UI", 11, "bold")
SMALL     = ("Segoe UI", 8)

# ── .env helpers ──────────────────────────────────────────────────────────────
def load_env(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return f.readlines()

def save_env(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

def parse_env(lines):
    out = {}
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s:
            k, _, v = s.partition("=")
            v = v.split("#")[0].strip()
            out[k.strip()] = v
    return out

def update_env_key(lines, key, value):
    new_lines, found = [], False
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, _, _ = s.partition("=")
            if k.strip() == key:
                new_lines.append(f"{key}={value}\n")
                found = True
                continue
        new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}\n")
    return new_lines

# ── .env section definitions ──────────────────────────────────────────────────
ENV_SECTIONS = {
    "Broker": [
        ("BROKER_TYPE",         "Broker type",              "mt5",      "mt5 | alpaca | ibkr | ctrader"),
        ("MT5_LOGIN",           "MT5 account number",       "",         "Integer login"),
        ("MT5_PASSWORD",        "MT5 password",             "",         "Will be masked"),
        ("MT5_SERVER",          "MT5 server",               "",         "e.g. GTioMarketsPty-Demo"),
        ("ALPACA_API_KEY",      "Alpaca API key",           "",         ""),
        ("ALPACA_SECRET_KEY",   "Alpaca secret",            "",         ""),
        ("ALPACA_PAPER",        "Alpaca paper mode",        "true",     "true | false"),
        ("ALPACA_DATA_FEED",    "Alpaca data feed",         "iex",      "iex (free) | sip (paid)"),
        ("ALPACA_OFFLINE",      "Alpaca offline mode",      "false",    "true = simulation only"),
        ("IBKR_HOST",           "IBKR host",                "127.0.0.1",""),
        ("IBKR_PORT",           "IBKR port",                "7497",     "7497=paper / 7496=live"),
        ("IBKR_CLIENT_ID",      "IBKR client ID",           "1",        ""),
        ("IBKR_PAPER",          "IBKR paper mode",          "true",     "true | false"),
        ("CTRADER_CLIENT_ID",   "cTrader client ID",        "",         ""),
        ("CTRADER_CLIENT_SECRET","cTrader client secret",   "",         ""),
        ("CTRADER_ACCESS_TOKEN","cTrader access token",     "",         ""),
        ("CTRADER_ACCOUNT_ID",  "cTrader account ID",       "0",        ""),
        ("CTRADER_DEMO",        "cTrader demo mode",        "true",     "true | false"),
    ],
    "Symbols": [
        ("SYMBOLS",             "Symbols (comma-sep)",
         "XAUUSD..,BTCUSD..,EURUSD..",
         "Exact broker symbol names"),
        ("MAX_POSITIONS_SYMBOL","Max positions / symbol",   "2",        ""),
    ],
    "Risk": [
        ("RISK_PER_TRADE",      "Risk per trade (%)",       "0",        "0 = auto-tier"),
        ("MAX_DAILY_LOSS",      "Max daily loss (%)",       "0",        "0 = auto-tier"),
        ("MAX_DRAWDOWN",        "Max drawdown (%)",         "0",        "0 = auto-tier"),
        ("MAX_TRADES_DAY",      "Max trades / day",         "0",        "0 = auto-tier"),
        ("MAX_CONCURRENT",      "Max concurrent pos.",      "0",        "0 = auto-tier"),
        ("MAX_GROUP_RISK_PCT",  "Max group risk (%)",       "0.06",     "Correlated pair limit"),
        ("ATR_MULTIPLIER",      "ATR multiplier",           "1.5",      "SL/TP ATR scaling"),
        ("LOT_SIZE",            "Default lot size",         "0.01",     "Minimum lot"),
    ],
    "Signal": [
        ("MIN_SIGNAL_PROB",     "Min signal prob",          "0.38",     "0.0 to 1.0"),
        ("PROP_MODE",           "Prop firm mode",           "false",    "true | false"),
        ("SLEEP_INTERVAL",      "Cycle interval (s)",       "300",      "Seconds between cycles"),
        ("BARS",                "ML training bars",         "8160",     "H1 bars (~1 yr)"),
    ],
    "Logging": [
        ("LOG_LEVEL",           "Log level",                "INFO",     "DEBUG | INFO | WARNING"),
    ],
}

# ── Runnable scripts ──────────────────────────────────────────────────────────
# (display_name, script_filename_or_None, default_args, description, accent_colour)
SCRIPTS = [
    ("AI EA (Main)",    "ai_ea.py",       "",
     "Main trading engine — connects to broker, trades live",  ACCENT),
    ("Trainer",         "trainer.py",     "--all-symbols --period all",
     "Train / retrain ML models for all symbols",              BLUE),
    ("Backtest",        "run_backtest.py","--bars 8000",
     "Walk-forward backtest with Monte Carlo on all symbols",  PURPLE),
    ("Optimizer",       "Optimizer.py",   "",
     "Grid-search optimal SL/TP parameters",                   WARN_C),
    ("Custom Script",   None,             "",
     "Browse for any .py file and run it with custom args",    TEXT_DIM),
]

# ── Process record ────────────────────────────────────────────────────────────
class ProcRecord:
    def __init__(self, label, script, args_str):
        self.label    = label
        self.script   = script
        self.args_str = args_str
        self.proc     = None
        self.running  = False
        self.log_widget = None
        self.tab_frame  = None

# =============================================================================
# Main Application
# =============================================================================
class AIEAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI EA v20 — Control Centre")
        self.geometry("1200x800")
        self.minsize(980, 660)
        self.configure(bg=BG)
        self.resizable(True, True)

        ico = os.path.join(BASE, "icon.ico")
        if os.path.exists(ico):
            try:
                self.iconbitmap(ico)
            except Exception:
                pass

        self._env_lines  = load_env(ENV_PATH)
        self._env_vars   = parse_env(self._env_lines)
        self._field_vars = {}       # section -> {key -> StringVar}
        self._procs      = {}       # label   -> ProcRecord
        self._ea_proc    = None     # shortcut to the EA's ProcRecord

        self._style_notebook()
        self._build_header()
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        self._build_notebook()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_procs()

    # =========================================================================
    # Styles
    # =========================================================================
    def _style_notebook(self):
        s = ttk.Style(self)
        s.theme_use("default")
        s.configure("Dark.TNotebook",
                    background=BG, borderwidth=0, tabmargins=[0, 0, 0, 0])
        s.configure("Dark.TNotebook.Tab",
                    background=BG3, foreground=TEXT_DIM,
                    padding=[14, 7], font=SANS, borderwidth=0)
        s.map("Dark.TNotebook.Tab",
              background=[("selected", BG2)],
              foreground=[("selected", TEXT)])

    # =========================================================================
    # Header
    # =========================================================================
    def _build_header(self):
        hdr = tk.Frame(self, bg=BG2, height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="  AI EA", font=("Segoe UI", 14, "bold"),
                 bg=BG2, fg=GOLD).pack(side="left", padx=(14, 0))
        tk.Label(hdr, text="v20", font=("Segoe UI", 10),
                 bg=BG2, fg=TEXT_DIM).pack(side="left", padx=(4, 16))

        self._lbl_status = tk.Label(hdr, text="STOPPED", font=SANS_B,
                                    bg=BG2, fg=DANGER)
        self._lbl_status.pack(side="left", padx=4)
        self._lbl_equity = tk.Label(hdr, text="", font=SANS, bg=BG2, fg=TEXT_DIM)
        self._lbl_equity.pack(side="left", padx=10)

        # Utility buttons (right-aligned, packed right-to-left)
        self._btn_restart = self._hbtn(hdr, "Restart EA", BG3, BORDER, self._restart_ea, "disabled")
        self._btn_restart.pack(side="right", padx=(2, 14))
        self._btn_stop = self._hbtn(hdr, "Stop EA", DANGER, DANGER_H, self._stop_ea, "disabled")
        self._btn_stop.pack(side="right", padx=2)
        self._btn_start = self._hbtn(hdr, "Start EA", ACCENT, ACCENT_H, self._start_ea)
        self._btn_start.pack(side="right", padx=2)

        self._hbtn(hdr, "Open Folder", BG3, BORDER,
                   lambda: os.startfile(WORK)).pack(side="right", padx=(0, 2))
        self._hbtn(hdr, "Open Logs", BG3, BORDER,
                   lambda: (os.makedirs(LOG_DIR, exist_ok=True), os.startfile(LOG_DIR))
                   ).pack(side="right", padx=2)
        self._hbtn(hdr, "Edit .env", BG3, BORDER,
                   lambda: os.startfile(ENV_PATH) if os.path.exists(ENV_PATH)
                   else messagebox.showinfo("Info", "No .env found next to the EXE.")
                   ).pack(side="right", padx=2)

    def _hbtn(self, parent, text, bg, bg_h, cmd, state="normal"):
        b = tk.Button(parent, text=text, font=SANS_B, bg=bg, fg=TEXT,
                      activebackground=bg_h, activeforeground=TEXT,
                      relief="flat", bd=0, padx=12, pady=6,
                      cursor="hand2", state=state, command=cmd)
        b.bind("<Enter>", lambda e: b.config(bg=bg_h) if b["state"] == "normal" else None)
        b.bind("<Leave>", lambda e: b.config(bg=bg)   if b["state"] == "normal" else None)
        return b

    # =========================================================================
    # Main Notebook
    # =========================================================================
    def _build_notebook(self):
        self._nb = ttk.Notebook(self, style="Dark.TNotebook")
        self._nb.pack(fill="both", expand=True)

        # Tab 0 — EA Live Log
        f0 = tk.Frame(self._nb, bg=BG)
        self._nb.add(f0, text="  EA Log  ")
        self._ea_logbox = self._make_log_panel(f0, "AI EA  Live Log")

        # Tab 1 — Script Runner
        f1 = tk.Frame(self._nb, bg=BG)
        self._nb.add(f1, text="  Run Scripts  ")
        self._build_runner_tab(f1)

        # Tab 2 — Process Manager
        f2 = tk.Frame(self._nb, bg=BG)
        self._nb.add(f2, text="  Processes  ")
        self._build_pm_tab(f2)

        # Tabs — .env sections
        for section, fields in ENV_SECTIONS.items():
            frm = tk.Frame(self._nb, bg=BG)
            self._nb.add(frm, text=f"  {section}  ")
            self._build_env_tab(frm, section, fields)

        # Tab — Raw .env
        raw = tk.Frame(self._nb, bg=BG)
        self._nb.add(raw, text="  Raw .env  ")
        self._build_raw_tab(raw)

    # =========================================================================
    # Log panel factory
    # =========================================================================
    def _make_log_panel(self, parent, title="Log"):
        ctrl = tk.Frame(parent, bg=BG, pady=6)
        ctrl.pack(fill="x", padx=14)
        tk.Label(ctrl, text=title, font=SANS_B, bg=BG, fg=TEXT).pack(side="left")

        autoscroll = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Auto-scroll", variable=autoscroll,
                       font=SANS, bg=BG, fg=TEXT_DIM, activebackground=BG,
                       selectcolor=BG3, relief="flat").pack(side="right", padx=6)

        lb = scrolledtext.ScrolledText(
            parent, bg=BG4, fg="#e6edf3", font=MONO,
            insertbackground=TEXT, relief="flat", bd=0,
            wrap="none", state="disabled"
        )
        lb.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        # Wire clear button after lb is created
        clr = tk.Button(ctrl, text="Clear", font=SANS, bg=BG3, fg=TEXT_DIM,
                        activebackground=BORDER, activeforeground=TEXT,
                        relief="flat", bd=0, padx=10, pady=3, cursor="hand2",
                        command=lambda: self._clear_log(lb))
        clr.pack(side="right")

        for tag, col, bold in [
            ("INFO",    "#8b949e", False),
            ("OK",      "#3fb950", False),
            ("WARNING", "#d29922", False),
            ("ERROR",   "#f85149", False),
            ("CRITICAL","#f85149", True),
            ("ORDER",   GOLD,      False),
            ("HEADER",  BLUE,      True),
            ("TRAINER", "#a5d6ff", False),
        ]:
            fnt = (*MONO[:2], "bold") if bold else MONO
            lb.tag_config(tag, foreground=col, font=fnt)

        lb._autoscroll = autoscroll
        lb._lines      = []
        return lb

    def _clear_log(self, lb):
        lb._lines.clear()
        lb.config(state="normal")
        lb.delete("1.0", "end")
        lb.config(state="disabled")

    def _append(self, lb, line, force_tag=""):
        if lb is None or not lb.winfo_exists():
            return
        tag = force_tag
        if not tag:
            u = line.upper()
            if "| ERROR" in u or "ERROR |" in u:    tag = "ERROR"
            elif "| WARNING" in u or "WARN" in u:   tag = "WARNING"
            elif "| CRITICAL" in u:                  tag = "CRITICAL"
            elif "ORDER" in line and "▶" in line:    tag = "ORDER"
            elif "TRAINER" in line or "EPOCH" in u:  tag = "TRAINER"
            elif "| INFO" in u:                      tag = "INFO"

        lb.config(state="normal")
        lb.insert("end", line, tag)
        lb._lines.append(line)
        if len(lb._lines) > 3000:
            lb._lines = lb._lines[-3000:]
            lb.delete("1.0", "200.0")
        if lb._autoscroll.get():
            lb.see("end")
        lb.config(state="disabled")

        m = re.search(r"equity=\$?([\d,]+\.?\d*)", line)
        if m:
            self._lbl_equity.config(text=f"Equity: ${m.group(1)}")

    # =========================================================================
    # Script Runner Tab
    # =========================================================================
    def _build_runner_tab(self, parent):
        tk.Label(parent, text="Run Scripts", font=HEAD, bg=BG, fg=GOLD,
                 anchor="w").pack(fill="x", padx=24, pady=(18, 2))
        tk.Label(parent,
                 text="Launch any component. Each process gets a live log in the Processes tab.",
                 font=SMALL, bg=BG, fg=TEXT_DIM, anchor="w").pack(fill="x", padx=24, pady=(0, 10))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=24, pady=(0, 12))

        for display, script, default_args, desc, colour in SCRIPTS:
            self._script_card(parent, display, script, default_args, desc, colour)

    def _script_card(self, parent, display, script, default_args, desc, colour):
        card = tk.Frame(parent, bg=BG2)
        card.pack(fill="x", padx=24, pady=4)

        left = tk.Frame(card, bg=BG2)
        left.pack(side="left", fill="both", expand=True, padx=16, pady=10)

        tk.Label(left, text=display, font=SANS_B, bg=BG2,
                 fg=colour, anchor="w").pack(fill="x")
        tk.Label(left, text=desc, font=SMALL, bg=BG2,
                 fg=TEXT_DIM, anchor="w").pack(fill="x", pady=(2, 6))

        arg_row = tk.Frame(left, bg=BG2)
        arg_row.pack(fill="x")
        tk.Label(arg_row, text="Args:", font=SMALL, bg=BG2,
                 fg=TEXT_DIM).pack(side="left")
        args_var = tk.StringVar(value=default_args)
        tk.Entry(arg_row, textvariable=args_var, font=MONO,
                 bg=BG3, fg=TEXT, insertbackground=TEXT,
                 relief="flat", bd=0, highlightthickness=1,
                 highlightbackground=BORDER, highlightcolor=colour,
                 width=52).pack(side="left", padx=(6, 0), ipady=4, ipadx=4)

        right = tk.Frame(card, bg=BG2)
        right.pack(side="right", padx=16, pady=10)

        def _run(s=script, av=args_var, dn=display):
            chosen = s
            if chosen is None:
                chosen = filedialog.askopenfilename(
                    title="Select Python script",
                    initialdir=WORK,
                    filetypes=[("Python files", "*.py"), ("All", "*.*")]
                )
                if not chosen:
                    return
            self._launch(chosen, av.get().strip(), dn)

        tk.Button(right, text="Run", font=SANS_B,
                  bg=colour if colour != TEXT_DIM else BG3,
                  fg=BG if colour not in (TEXT_DIM,) else TEXT,
                  activebackground=ACCENT_H, activeforeground=BG,
                  relief="flat", bd=0, padx=18, pady=7,
                  cursor="hand2", command=_run).pack()

    # =========================================================================
    # Process Manager Tab
    # =========================================================================
    def _build_pm_tab(self, parent):
        top = tk.Frame(parent, bg=BG)
        top.pack(fill="both", expand=True)

        tk.Label(top, text="Active Processes", font=HEAD, bg=BG, fg=GOLD,
                 anchor="w").pack(fill="x", padx=24, pady=(14, 2))

        # Process list area
        self._pm_list = tk.Frame(top, bg=BG)
        self._pm_list.pack(fill="x", padx=24, pady=(0, 8))

        tk.Frame(top, bg=BORDER, height=1).pack(fill="x", padx=24)

        tk.Label(top, text="Process Log", font=SANS_B, bg=BG, fg=TEXT_DIM,
                 anchor="w").pack(fill="x", padx=24, pady=(8, 2))

        # Log notebook for per-process tabs
        self._pm_nb = ttk.Notebook(top, style="Dark.TNotebook")
        self._pm_nb.pack(fill="both", expand=True, padx=24, pady=(0, 14))

    def _pm_refresh(self):
        for w in self._pm_list.winfo_children():
            w.destroy()

        if not self._procs:
            tk.Label(self._pm_list, text="No processes running.",
                     font=SANS, bg=BG, fg=TEXT_DIM
                     ).pack(anchor="w", padx=4, pady=8)
            return

        # Header row
        hdr = tk.Frame(self._pm_list, bg=BG3)
        hdr.pack(fill="x", pady=(0, 1))
        for col, w in [("Script", 20), ("Status", 10), ("PID", 8), ("Args", 30), ("", 14)]:
            tk.Label(hdr, text=col, font=SANS_B, bg=BG3, fg=TEXT_DIM,
                     width=w, anchor="w").pack(side="left", padx=(8, 0), pady=4)

        for label, rec in list(self._procs.items()):
            is_running = rec.running
            s_text = "RUNNING" if is_running else "STOPPED"
            s_col  = "#3fb950" if is_running else DANGER
            pid    = str(rec.proc.pid) if rec.proc and rec.proc.poll() is None else "-"
            args   = rec.args_str[:36] + "..." if len(rec.args_str) > 36 else rec.args_str

            row = tk.Frame(self._pm_list, bg=BG2 if is_running else BG3)
            row.pack(fill="x", pady=1)

            tk.Label(row, text=label[:22], font=SANS_B, bg=row["bg"],
                     fg=TEXT, width=20, anchor="w").pack(side="left", padx=8, pady=5)
            tk.Label(row, text=s_text, font=SANS, bg=row["bg"],
                     fg=s_col, width=10, anchor="w").pack(side="left")
            tk.Label(row, text=pid, font=MONO, bg=row["bg"],
                     fg=TEXT_DIM, width=8, anchor="w").pack(side="left")
            tk.Label(row, text=args, font=MONO, bg=row["bg"],
                     fg=TEXT_DIM, width=30, anchor="w").pack(side="left")

            if is_running:
                tk.Button(row, text="Kill", font=SMALL,
                          bg=DANGER, fg="white",
                          activebackground=DANGER_H, activeforeground="white",
                          relief="flat", bd=0, padx=10, pady=3, cursor="hand2",
                          command=lambda r=rec: self._kill(r)
                          ).pack(side="right", padx=8)
            else:
                tk.Button(row, text="Remove", font=SMALL,
                          bg=BG3, fg=TEXT_DIM,
                          activebackground=BORDER, activeforeground=TEXT,
                          relief="flat", bd=0, padx=10, pady=3, cursor="hand2",
                          command=lambda lb=label: self._remove_proc(lb)
                          ).pack(side="right", padx=8)

    def _kill(self, rec):
        rec.running = False
        if rec.proc and rec.proc.poll() is None:
            rec.proc.terminate()
            try:
                rec.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rec.proc.kill()
        self._pm_refresh()
        if rec is self._ea_proc:
            self._ea_proc = None
        self._update_ea_btns()

    def _remove_proc(self, label):
        rec = self._procs.pop(label, None)
        if rec and rec.tab_frame and rec.tab_frame.winfo_exists():
            self._pm_nb.forget(rec.tab_frame)
        self._pm_refresh()

    # =========================================================================
    # .env Section Tabs
    # =========================================================================
    def _build_env_tab(self, parent, section, fields):
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG)
        wid = canvas.create_window((0, 0), window=inner, anchor="nw")

        canvas.bind("<Configure>", lambda e: canvas.itemconfig(wid, width=e.width))
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        self._field_vars[section] = {}

        tk.Label(inner, text=section, font=HEAD, bg=BG, fg=GOLD,
                 anchor="w").grid(row=0, column=0, columnspan=3,
                                  sticky="w", padx=24, pady=(20, 4))
        tk.Frame(inner, bg=BORDER, height=1).grid(
            row=1, column=0, columnspan=3, sticky="ew", padx=24, pady=(0, 14))

        for i, (key, label, default, hint) in enumerate(fields, start=2):
            svar = tk.StringVar(value=self._env_vars.get(key, default))
            self._field_vars[section][key] = svar

            tk.Label(inner, text=label, font=SANS_B, bg=BG, fg=TEXT,
                     anchor="w", width=26).grid(row=i, column=0,
                                                sticky="w", padx=(24, 8), pady=5)

            show = "*" if any(x in key for x in ("PASSWORD", "SECRET", "TOKEN")) else ""
            tk.Entry(inner, textvariable=svar, font=MONO,
                     bg=BG3, fg=TEXT, insertbackground=TEXT,
                     relief="flat", bd=0, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=ACCENT,
                     show=show, width=44).grid(row=i, column=1, sticky="w",
                                               padx=(0, 10), pady=5,
                                               ipady=5, ipadx=6)

            if hint:
                tk.Label(inner, text=hint, font=SMALL, bg=BG,
                         fg=TEXT_DIM, anchor="w").grid(row=i, column=2,
                                                        sticky="w", padx=(0, 24))

        last = len(fields) + 2
        save_row = tk.Frame(inner, bg=BG)
        save_row.grid(row=last, column=0, columnspan=3,
                      sticky="w", padx=24, pady=(20, 32))

        tk.Button(save_row, text="  Save and Apply  ", font=SANS_B,
                  bg=ACCENT, fg="white", activebackground=ACCENT_H,
                  activeforeground="white", relief="flat", bd=0,
                  padx=14, pady=7, cursor="hand2",
                  command=lambda s=section: self._save_section(s)
                  ).pack(side="left")
        tk.Label(save_row,
                 text="  Saves .env immediately. If EA is running it will ask to restart.",
                 font=SMALL, bg=BG, fg=TEXT_DIM).pack(side="left", padx=10)

    # =========================================================================
    # Raw .env Tab
    # =========================================================================
    def _build_raw_tab(self, parent):
        tk.Label(parent,
                 text="Raw .env — the complete file as stored on disk. "
                      "All comments and blank lines preserved.",
                 font=SANS, bg=BG, fg=TEXT_DIM,
                 anchor="w").pack(fill="x", padx=14, pady=(10, 4))

        self._raw_text = scrolledtext.ScrolledText(
            parent, bg=BG3, fg=TEXT, font=MONO,
            insertbackground=TEXT, relief="flat", bd=0,
            highlightthickness=1, highlightbackground=BORDER,
            wrap="none"
        )
        self._raw_text.pack(fill="both", expand=True, padx=14, pady=(0, 6))
        self._raw_text.insert("1.0", "".join(self._env_lines))

        btn_row = tk.Frame(parent, bg=BG)
        btn_row.pack(fill="x", padx=14, pady=(0, 14))

        tk.Button(btn_row, text="  Save Raw  ", font=SANS_B,
                  bg=ACCENT, fg="white", activebackground=ACCENT_H,
                  activeforeground="white", relief="flat", bd=0,
                  padx=14, pady=7, cursor="hand2",
                  command=self._save_raw).pack(side="left")
        tk.Button(btn_row, text="  Reload from disk  ", font=SANS,
                  bg=BG3, fg=TEXT_DIM, activebackground=BORDER,
                  activeforeground=TEXT, relief="flat", bd=0,
                  padx=12, pady=6, cursor="hand2",
                  command=self._reload_raw).pack(side="left", padx=8)
        tk.Button(btn_row, text="  Open in Notepad  ", font=SANS,
                  bg=BG3, fg=TEXT_DIM, activebackground=BORDER,
                  activeforeground=TEXT, relief="flat", bd=0,
                  padx=12, pady=6, cursor="hand2",
                  command=lambda: os.startfile(ENV_PATH)
                  if os.path.exists(ENV_PATH)
                  else messagebox.showinfo("Not found", ".env not found.")
                  ).pack(side="left", padx=4)

    # =========================================================================
    # EA start / stop
    # =========================================================================
    def _start_ea(self):
        script = os.path.join(BASE, "ai_ea.py")
        rec = self._launch(script, "", "AI EA", is_ea=True)
        if rec:
            self._ea_proc = rec

    def _stop_ea(self):
        if self._ea_proc:
            self._kill(self._ea_proc)
            self._ea_proc = None
        self._update_ea_btns()

    def _restart_ea(self):
        self._stop_ea()
        self.after(500, self._start_ea)

    # =========================================================================
    # Generic launcher
    # =========================================================================
    def _launch(self, script_path, args_str, label, is_ea=False):
        if not os.path.isabs(script_path):
            script_path = os.path.join(BASE, script_path)
        if not os.path.exists(script_path):
            messagebox.showerror("Not found", f"Script not found:\n{script_path}")
            return None

        # Deduplicate label
        base = label
        n = 1
        while label in self._procs and self._procs[label].running:
            n += 1
            label = f"{base} [{n}]"

        rec = ProcRecord(label, script_path, args_str)

        cmd = [sys.executable, script_path]
        if args_str.strip():
            cmd += args_str.split()

        try:
            cflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            proc = subprocess.Popen(
                cmd, cwd=WORK,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=cflags,
            )
        except Exception as e:
            messagebox.showerror("Launch Error", str(e))
            return None

        rec.proc    = proc
        rec.running = True

        # Choose log target
        if is_ea:
            lb = self._ea_logbox
        else:
            frm = tk.Frame(self._pm_nb, bg=BG)
            self._pm_nb.add(frm, text=f"  {label}  ")
            lb = self._make_log_panel(frm, label)
            rec.log_widget = lb
            rec.tab_frame  = frm

        self._procs[label] = rec
        self._append(lb, f"── Launched: {os.path.basename(script_path)}"
                        f" {args_str} (PID {proc.pid}) ──\n", "HEADER")

        threading.Thread(target=self._read_proc,
                         args=(rec, lb), daemon=True).start()

        if is_ea:
            threading.Thread(target=self._tail_file,
                             args=(lb,), daemon=True).start()
            self._update_ea_btns()

        self._pm_refresh()
        return rec

    def _read_proc(self, rec, lb):
        while rec.proc:
            line = rec.proc.stdout.readline()
            if not line:
                if rec.proc.poll() is not None:
                    self.after(0, self._proc_ended, rec, lb)
                    break
                continue
            self.after(0, self._append, lb, line)

    def _tail_file(self, lb):
        """Tail the on-disk log file for the EA."""
        if not os.path.exists(LOG_PATH):
            return
        try:
            with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)
                while self._ea_proc and self._ea_proc.running:
                    line = f.readline()
                    if line:
                        self.after(0, self._append, lb, line)
                    else:
                        time.sleep(0.2)
        except Exception:
            pass

    def _proc_ended(self, rec, lb):
        rec.running = False
        self._append(lb, f"── Process exited: {os.path.basename(rec.script)} ──\n", "WARNING")
        if rec is self._ea_proc:
            self._ea_proc = None
            self._update_ea_btns()
        self._pm_refresh()

    # =========================================================================
    # Status refresh
    # =========================================================================
    def _update_ea_btns(self):
        ea_on = self._ea_proc is not None and self._ea_proc.running
        if ea_on:
            self._lbl_status.config(text="RUNNING", fg="#3fb950")
            self._btn_start.config(state="disabled", bg="#1a4023")
            self._btn_stop.config(state="normal",   bg=DANGER)
            self._btn_restart.config(state="normal", bg=BG3)
        else:
            self._lbl_status.config(text="STOPPED", fg=DANGER)
            self._btn_start.config(state="normal",   bg=ACCENT)
            self._btn_stop.config(state="disabled",  bg="#4a1515")
            self._btn_restart.config(state="disabled", bg="#1e1e1e")
            self._lbl_equity.config(text="")

    def _poll_procs(self):
        self._pm_refresh()
        self.after(2500, self._poll_procs)

    # =========================================================================
    # .env save helpers
    # =========================================================================
    def _save_section(self, section):
        for key, _, _, _ in ENV_SECTIONS[section]:
            svar = self._field_vars.get(section, {}).get(key)
            if svar:
                self._env_lines = update_env_key(self._env_lines, key, svar.get())
        save_env(ENV_PATH, self._env_lines)
        self._env_vars = parse_env(self._env_lines)
        self._raw_text.delete("1.0", "end")
        self._raw_text.insert("1.0", "".join(self._env_lines))
        if self._ea_proc and self._ea_proc.running:
            if messagebox.askyesno("Restart EA?",
                    "Settings saved.\n\nRestart the EA now to apply them?"):
                self._restart_ea()
        else:
            messagebox.showinfo("Saved", f"[{section}] saved to .env")

    def _save_raw(self):
        content = self._raw_text.get("1.0", "end")
        self._env_lines = [l + "\n" for l in content.splitlines()]
        save_env(ENV_PATH, self._env_lines)
        self._env_vars = parse_env(self._env_lines)
        for section, fields in ENV_SECTIONS.items():
            for key, _, _, _ in fields:
                svar = self._field_vars.get(section, {}).get(key)
                if svar:
                    svar.set(self._env_vars.get(key, ""))
        if self._ea_proc and self._ea_proc.running:
            if messagebox.askyesno("Restart EA?",
                    ".env saved.\n\nRestart EA to apply?"):
                self._restart_ea()
        else:
            messagebox.showinfo("Saved", ".env written to disk.")

    def _reload_raw(self):
        self._env_lines = load_env(ENV_PATH)
        self._raw_text.delete("1.0", "end")
        self._raw_text.insert("1.0", "".join(self._env_lines))

    # =========================================================================
    # Close
    # =========================================================================
    def _on_close(self):
        running = [r for r in self._procs.values() if r.running]
        if running:
            names = ", ".join(r.label for r in running)
            if not messagebox.askyesno("Quit",
                    f"Running: {names}\n\nStop all and exit?"):
                return
            for r in running:
                r.running = False
                if r.proc and r.proc.poll() is None:
                    r.proc.terminate()
        self.destroy()


if __name__ == "__main__":
    app = AIEAApp()
    app.mainloop()
