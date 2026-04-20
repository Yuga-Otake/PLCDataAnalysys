"""
APB Timing Analyzer — Streamlit ランチャー
通常実行・PyInstaller EXE 両対応
"""
import sys
import os
import threading
import webbrowser
import time

PORT = 8501


def _open_browser():
    """起動後にブラウザを自動で開く（5 秒待機）"""
    time.sleep(5)
    webbrowser.open(f"http://localhost:{PORT}")


def _find_app_py() -> str:
    """app.py の絶対パスを解決する"""
    if getattr(sys, "frozen", False):
        # PyInstaller --onedir: データは sys._MEIPASS 以下
        candidates = [
            os.path.join(sys._MEIPASS, "timing_analyzer", "app.py"),
            os.path.join(os.path.dirname(sys.executable), "timing_analyzer", "app.py"),
        ]
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base, "timing_analyzer", "app.py"),
        ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    # 見つからない場合は先頭候補を返す（Streamlit がエラーを出す）
    return candidates[0]


def main():
    app_path = _find_app_py()

    # ブラウザをバックグラウンドで遅延起動
    threading.Thread(target=_open_browser, daemon=True).start()

    # Streamlit を Python API 経由で起動
    sys.argv = [
        "streamlit", "run", app_path,
        f"--server.port={PORT}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    from streamlit.web import cli as stcli
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
