"""
APB タイミング解析 - Windows アプリランチャー

起動フロー:
  1. 空きポートを取得
  2. Streamlit サーバーをデーモンスレッドで起動
  3. サーバーが応答するまで待機（スプラッシュ表示）
  4. pywebview でネイティブウィンドウを開く
  5. ウィンドウを閉じるとプロセス全体を終了
"""

# ============================================================
# CRITICAL: freeze_support を最初に呼ぶ（multiprocessing spawn ループ防止）
# PyInstaller が子プロセスを起動するとき、このモジュールを import するが
# freeze_support() がここで処理を止める
# ============================================================
import multiprocessing
multiprocessing.freeze_support()

import sys
import os
import socket
import time
import threading

import webview

# PyInstaller が streamlit を確実にバンドルするようトップレベルで import
import streamlit                       # noqa: F401
import streamlit.web.cli               # noqa: F401
import streamlit.web.bootstrap         # noqa: F401
import streamlit.runtime.scriptrunner  # noqa: F401


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def resource_path(*parts: str) -> str:
    """PyInstaller の _MEIPASS に対応したパス解決"""
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.join(os.path.dirname(__file__), "..")
    return os.path.normpath(os.path.join(base, *parts))


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


# --------------------------------------------------------------------------- #
# Streamlit をスレッドで起動（bootstrap.run() を直接呼び出す）
# cli.main() → Click 経由だと PyInstaller 環境で問題が起きるため直接呼ぶ
# --------------------------------------------------------------------------- #
def _run_streamlit(app_py: str, port: int) -> None:
    """デーモンスレッドで Streamlit bootstrap を実行"""
    # bootstrap.run() がシグナルハンドラーを設定しようとするが、
    # スレッド内では signal.signal() が使えないのでパッチする
    import signal
    import threading as _threading
    _orig_signal = signal.signal

    def _thread_safe_signal(sig, handler):
        if _threading.current_thread() is _threading.main_thread():
            return _orig_signal(sig, handler)
        # サブスレッドではシグナル設定を無視（例外を出さない）

    signal.signal = _thread_safe_signal

    # config.set_option() で直接設定（env var / flag_options より確実）
    from streamlit import config as st_config
    st_config.set_option("server.port",              port)
    st_config.set_option("server.address",           "127.0.0.1")
    st_config.set_option("server.headless",          True)
    st_config.set_option("browser.gatherUsageStats", False)
    st_config.set_option("server.fileWatcherType",   "none")
    # Node dev server を無効化（デフォルト True になる場合があるため明示的に False）
    st_config.set_option("global.developmentMode",   False)

    from streamlit.web.bootstrap import run as st_bootstrap_run
    try:
        st_bootstrap_run(
            main_script_path=app_py,
            is_hello=False,
            args=[],
            flag_options={},
        )
    except SystemExit:
        pass
    finally:
        signal.signal = _orig_signal


# --------------------------------------------------------------------------- #
# Splash HTML
# --------------------------------------------------------------------------- #
SPLASH_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {
    margin: 0;
    background: #0e1117;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    font-family: 'Segoe UI', sans-serif;
    color: #fafafa;
  }
  .title { font-size: 28px; font-weight: bold; margin-bottom: 12px; }
  .sub   { font-size: 14px; color: #888; margin-bottom: 40px; }
  .spinner {
    width: 48px; height: 48px;
    border: 5px solid #333;
    border-top-color: #ff4b4b;
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <div class="title">&#127981; APB タイミング解析</div>
  <div class="sub">アプリを起動しています...</div>
  <div class="spinner"></div>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    port = find_free_port()
    app_py = resource_path("timing_analyzer", "app.py")

    # Streamlit をデーモンスレッドで起動
    st_thread = threading.Thread(
        target=_run_streamlit, args=(app_py, port), daemon=True
    )
    st_thread.start()

    # --- スプラッシュウィンドウ ---
    splash = webview.create_window(
        "APB タイミング解析 - 起動中",
        html=SPLASH_HTML,
        width=480,
        height=320,
        resizable=False,
        frameless=False,
    )

    _main_window_created = threading.Event()

    def on_splash_loaded():
        if _main_window_created.is_set():
            return  # 複数回 fired された場合は無視
        ready = wait_for_server(port, timeout=60)
        if not ready:
            splash.load_html(
                "<body style='background:#0e1117;color:#ff4b4b;"
                "font-family:sans-serif;display:flex;align-items:center;"
                "justify-content:center;height:100vh;font-size:18px;'>"
                "起動タイムアウト。アプリを再起動してください。</body>"
            )
            return

        _main_window_created.set()
        # メインウィンドウを先に作ってから splash を閉じる
        # （splash を先に destroy すると webview.start() が即終了するため）
        webview.create_window(
            "APB タイミング解析",
            url=f"http://127.0.0.1:{port}",
            width=1440,
            height=900,
            min_size=(900, 600),
        )
        time.sleep(0.3)   # メインウィンドウが登録されるまで少し待つ
        splash.destroy()

    splash.events.loaded += lambda: threading.Thread(
        target=on_splash_loaded, daemon=True
    ).start()

    webview.start(gui="edgechromium", debug=False)


if __name__ == "__main__":
    main()
