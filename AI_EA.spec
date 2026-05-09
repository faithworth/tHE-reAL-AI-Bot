# -*- mode: python ; coding: utf-8 -*-
import os, glob

ROOT     = r"C:\Users\Ian Sunino\Desktop\AI_EA_v4_FINAL\AI_EA-V3.0.7-main"
LAUNCHER = r"C:\Users\Ian Sunino\Desktop\AI_EA_v4_FINAL\AI_EA-V3.0.7-main\launcher.py"

# Bundle every .py source file so they can be launched as subprocesses
py_sources = [(f, ".") for f in glob.glob(os.path.join(ROOT, "*.py"))
              if os.path.basename(f) != "launcher.py"]

datas = py_sources + [
    (os.path.join(ROOT, ".env.example"), "."),
    (os.path.join(ROOT, "requirements.txt"), "."),
]
if os.path.exists(os.path.join(ROOT, "icon.ico")):
    datas.append((os.path.join(ROOT, "icon.ico"), "."))
if os.path.exists(os.path.join(ROOT, "GUIDE.md")):
    datas.append((os.path.join(ROOT, "GUIDE.md"), "."))

block_cipher = None

a = Analysis(
    [LAUNCHER],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "sklearn", "sklearn.ensemble", "sklearn.tree",
        "sklearn.preprocessing", "sklearn.linear_model",
        "sklearn.model_selection", "sklearn.metrics",
        "xgboost", "lightgbm",
        "scipy", "scipy.stats",
        "pandas", "numpy",
        "MetaTrader5",
        "alpaca", "alpaca.trading", "alpaca.data",
        "ib_insync",
        "matplotlib", "matplotlib.backends.backend_tkagg",
        "plotly",
        "requests", "cryptography", "pytz",
        "tkinter", "tkinter.scrolledtext", "tkinter.ttk",
        "tkinter.filedialog", "tkinter.messagebox", "tkinter.font",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "wx"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ONE-FILE EXE — all dependencies merged into a single .exe
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="AI_EA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon=r"C:\Users\Ian Sunino\Desktop\AI_EA_v4_FINAL\AI_EA-V3.0.7-main\icon.ico" if os.path.exists(r"C:\Users\Ian Sunino\Desktop\AI_EA_v4_FINAL\AI_EA-V3.0.7-main\icon.ico") else None,
)
