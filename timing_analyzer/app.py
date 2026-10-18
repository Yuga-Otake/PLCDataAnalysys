"""
app.py — APB タイミング解析 エントリーポイント（画面遷移の定義）

利用者の仕事を「準備」（立ち上げ時に 1 回）と「点検」（定期）に分けてメニューを構成する。
旧画面（legacy_app.py）は新画面への置き換えが終わるまで「移行中」に残す。
"""
import streamlit as st

from core.state import enable_session_persistence, ensure_session
from core.ui import apply_theme

st.set_page_config(page_title="APB タイミング解析", page_icon=":material/timeline:", layout="wide")
enable_session_persistence()
ensure_session()

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
    "移行中": [
        st.Page("legacy_app.py", title="旧画面", icon=":material/history:", url_path="legacy"),
        # 旧画面のガントをダブルクリックすると別タブで開く（/histogram?proc=…）
        st.Page("pages/5_histogram.py", title="ヒストグラム詳細", url_path="histogram",
                visibility="hidden"),
    ],
})
apply_theme()
nav.run()
