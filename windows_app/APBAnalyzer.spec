# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for APB タイミング解析
ビルド: pyinstaller windows_app/APBAnalyzer.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_all

block_cipher = None

# ── Streamlit の静的ファイル・データをすべて収集 ──────────────────
st_datas, st_binaries, st_hiddenimports = collect_all("streamlit")
plotly_datas, _, _ = collect_all("plotly")

# timing_analyzer ディレクトリを丸ごと同梱
app_datas = [
    (os.path.join("..", "timing_analyzer"), "timing_analyzer"),
]

all_datas    = app_datas + st_datas + plotly_datas
all_binaries = st_binaries

hidden_imports = st_hiddenimports + [
    # analyzer / app 内で使う主要パッケージ
    "pandas",
    "numpy",
    "plotly",
    "plotly.graph_objects",
    "plotly.express",
    # Streamlit 内部
    "streamlit.runtime.scriptrunner.magic_funcs",
    "streamlit.web.cli",
    # pywebview バックエンド（Windows は mshtml/edgechromium）
    "webview",
    "webview.platforms.winforms",
    # その他
    "altair",
    "pyarrow",
    "tzdata",
]

a = Analysis(
    [os.path.join("..", "windows_app", "launcher.py")],
    pathex=[os.path.join("..", )],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "scipy", "sklearn", "tensorflow", "torch",
        "IPython", "jupyter", "notebook",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # onedir モード（起動が速い）
    name="APBAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                  # コンソールウィンドウを非表示
    icon=None,                      # アイコンを設定する場合: icon="icon.ico"
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="APBAnalyzer",
)
