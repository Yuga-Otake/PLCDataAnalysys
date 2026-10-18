"""
app.py — APB タイミング解析 エントリーポイント（画面遷移の定義）

利用者の仕事を「準備」（立ち上げ時に 1 回）と「点検」（定期）に分けてメニューを構成する。
"""
import os
import sys
import time

import streamlit as st


def _reload_updated_modules():
    """更新されたアプリのモジュールを読み直す。

    Streamlit Community Cloud はリポジトリの更新を取り込んでも、プロセス内に読み込み済みの自作モジュール
    （core/*.py など）を古いまま使い続けることがあり、新しく追加した関数を import できずに止まる
    （ページのスクリプトだけは毎回読み直される）。読み込んだ後にファイルが更新されていたら、
    このアプリのモジュールをまとめて sys.modules から外し、次の import で読み直させる。
    """
    base = os.path.dirname(os.path.abspath(__file__)) + os.sep
    loaded_at = getattr(sys, "_apb_modules_loaded_at", None)
    if loaded_at is None:
        try:   # 初回はプロセスの開始時刻（それより後に更新されたファイルは古いまま読まれている）
            loaded_at = os.stat(f"/proc/{os.getpid()}").st_ctime
        except OSError:
            loaded_at = time.time()
    mine = [n for n, m in list(sys.modules.items())
            if n != "__main__" and (getattr(m, "__file__", None) or "").startswith(base)]
    if any(os.path.getmtime(sys.modules[n].__file__) > loaded_at
           for n in mine if os.path.exists(sys.modules[n].__file__)):
        for n in mine:
            sys.modules.pop(n, None)
    sys._apb_modules_loaded_at = time.time()


_reload_updated_modules()

from core.state import enable_session_persistence, ensure_session  # noqa: E402
from core.ui import apply_theme  # noqa: E402
from core.wave_inspect import _wi_restore_config  # noqa: E402

st.set_page_config(page_title="APB タイミング解析", page_icon=":material/timeline:", layout="wide")
enable_session_persistence()
ensure_session()
# 波形検査の検出条件（前回の自動保存）を最初に復元する。推移画面からも使うため波形検査条件の画面を待たない
if "wi_config_loaded" not in st.session_state:
    _wi_restore_config(st.session_state)
    st.session_state["wi_config_loaded"] = True

nav = st.navigation({
    "": [
        st.Page("views/home.py", title="ホーム", icon=":material/home:", default=True),
    ],
    "準備": [
        st.Page("views/data.py", title="データ", icon=":material/folder_open:"),
        st.Page("views/process.py", title="工程とステップ", icon=":material/account_tree:"),
        st.Page("views/baseline.py", title="基準値", icon=":material/straighten:"),
        st.Page("views/wave.py", title="波形検査条件", icon=":material/show_chart:"),
    ],
    "点検": [
        st.Page("views/evaluate.py", title="最新データの評価", icon=":material/fact_check:"),
        st.Page("views/trend.py", title="推移", icon=":material/trending_up:"),
        st.Page("views/capability.py", title="工程能力", icon=":material/insights:"),
        # 評価・推移から開く掘り下げ専用ページ（メニューには出さない）
        st.Page("views/step.py", title="ステップ詳細", url_path="step", visibility="hidden"),
    ],
    "ツール": [
        st.Page("pages/4_ロガー設定ナビ.py", title="ロガー設定ナビ", icon=":material/tune:",
                url_path="logger"),
        st.Page("pages/6_AIチャット.py", title="AI アシスタント", icon=":material/chat:",
                url_path="assistant"),
        st.Page("pages/0_このツールについて.py", title="使い方", icon=":material/help:",
                url_path="about"),
    ],
})
apply_theme()
nav.run()
