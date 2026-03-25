"""
APB タイミング解析 - Windows アプリランチャー

起動フロー:
  1. 空きポートを取得
  2. Streamlit サーバーをバックグラウンドで起動
  3. サーバーが応答するまで待機（スプラッシュ表示）
  4. pywebview でネイティブウィンドウを開く
  5. ウィンドウを閉じると Streamlit プロセスも終了
"""

import sys
import os
import socket
import time
import threading
import subprocess
import webview


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
  <div class="title">🏭 APB タイミング解析</div>
  <div class="sub">アプリを起動しています...</div>
  <div class="spinner"></div>
</body>
</html>
"""


def main() -> None:
    port = find_free_port()
    app_py = resource_path("timing_analyzer", "app.py")

    # --- Streamlit サーバー起動 ---
    env = os.environ.copy()
    # Streamlit のブラウザ自動起動を無効化
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m", "streamlit", "run", app_py,
            "--server.port", str(port),
            "--server.address", "127.0.0.1",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )

    # --- スプラッシュウィンドウ ---
    splash = webview.create_window(
        "APB タイミング解析 - 起動中",
        html=SPLASH_HTML,
        width=480,
        height=320,
        resizable=False,
        frameless=False,
    )

    def on_splash_loaded():
        # サーバー待機（別スレッド）
        ready = wait_for_server(port, timeout=60)
        if not ready:
            splash.load_html(
                "<body style='background:#0e1117;color:#ff4b4b;"
                "font-family:sans-serif;display:flex;align-items:center;"
                "justify-content:center;height:100vh;font-size:18px;'>"
                "起動タイムアウト。アプリを再起動してください。</body>"
            )
            return

        # メインウィンドウに切り替え
        splash.destroy()
        main_win = webview.create_window(
            "APB タイミング解析",
            url=f"http://127.0.0.1:{port}",
            width=1440,
            height=900,
            min_size=(900, 600),
        )

        def on_main_closed():
            proc.terminate()

        main_win.events.closed += on_main_closed

    splash.events.loaded += lambda: threading.Thread(
        target=on_splash_loaded, daemon=True
    ).start()

    webview.start(debug=False)

    # ウィンドウが全て閉じられた後のクリーンアップ
    if proc.poll() is None:
        proc.terminate()


if __name__ == "__main__":
    main()
