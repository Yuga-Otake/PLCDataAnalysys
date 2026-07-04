"""
6_AIチャット.py — Google Gemini を使った AI チャット画面

機能:
  - 解析データについての質問に日本語で回答
  - 「品質分析モードにして」「Step2のヒストグラムを見せて」等の操作をAIが実行
  - Function Calling でアプリの表示モード・CSV・工程展開・ヒストグラム遷移を操作
"""

import sys
import os

# timing_analyzer/ ディレクトリを import パスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ai_chat import (
    get_api_key,
    get_or_init_history,
    append_to_history,
    history_for_display,
    build_analysis_context,
    call_gemini,
    execute_tool,
)

st.set_page_config(
    page_title="AI チャット | APB解析",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    '<style>[data-testid="stSidebarNav"] { display: none !important; }</style>',
    unsafe_allow_html=True,
)

# ── ページ先頭でナビゲーション処理 ──────────────────────────────────────────
# st.switch_page() はウィジェット外で呼ぶ必要があるため、
# execute_tool() が _ai_nav_target にセットした値をここで処理する
if nav_target := st.session_state.pop("_ai_nav_target", None):
    st.switch_page(nav_target)

# ── APIキー取得 ──────────────────────────────────────────────────────────────
api_key = get_api_key()

# ── サイドバー ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.page_link("app.py", label="← メイン画面に戻る")
    st.divider()

    with st.expander("🔑 Google API キー設定", expanded=not bool(api_key)):
        st.text_input(
            "API Key",
            type="password",
            key="ai_api_key_input",
            help="GOOGLE_API_KEY 環境変数または .streamlit/secrets.toml でも設定できます",
        )
    api_key = get_api_key()

    if st.button("🗑️ 会話履歴をクリア", use_container_width=True):
        st.session_state["ai_chat_history"] = []
        st.rerun()

    st.divider()
    st.markdown("**📊 現在の解析データ**")
    processes = st.session_state.get("processes", {})
    if processes:
        for pname in processes:
            hist_ctx = st.session_state.get(f"_hist_ctx_{pname}", {})
            has_stats = bool(hist_ctx.get("step_stats"))
            icon = "✅" if has_stats else "⏳"
            st.caption(f"{icon} {pname}")
    else:
        st.caption("工程未設定（メイン画面で設定してください）")

    st.divider()
    st.markdown(
        "**操作例**\n"
        "- 品質分析モードにして\n"
        "- ProcessAのStep2のヒストグラムを見せて\n"
        "- Step2の遅延が大きい原因は？\n"
        "- NGデータのCSVに切り替えて"
    )

# ── メインエリア ─────────────────────────────────────────────────────────────
st.title("🤖 AI チャット — APB 解析アシスタント")
st.caption("質問と操作どちらもできます")

if not api_key:
    st.warning("サイドバーで Google API キーを入力してください。\n\n"
               "または環境変数 `GOOGLE_API_KEY` か `.streamlit/secrets.toml` に設定してください。")
    st.stop()

# 会話履歴を描画
for msg in history_for_display(st.session_state):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── チャット入力 ─────────────────────────────────────────────────────────────
if user_input := st.chat_input("質問または操作を入力してください"):
    # ユーザーメッセージを即時表示
    with st.chat_message("user"):
        st.markdown(user_input)

    history = get_or_init_history(st.session_state)
    context = build_analysis_context(st.session_state)

    # Gemini API 呼び出し
    with st.spinner("考え中..."):
        result = call_gemini(user_input, history, context, api_key)

    if result["type"] == "tool_use":
        # ツール実行
        tool_msg, next_page = execute_tool(
            result["tool_name"], result["tool_input"], st.session_state
        )
        with st.chat_message("assistant"):
            st.markdown(tool_msg)

        append_to_history(st.session_state, "user", user_input)
        append_to_history(st.session_state, "model", tool_msg)

        if next_page:
            st.session_state["_ai_nav_target"] = next_page

        st.rerun()

    else:
        # テキスト応答
        with st.chat_message("assistant"):
            st.markdown(result["content"])

        append_to_history(st.session_state, "user", user_input)
        append_to_history(st.session_state, "model", result["content"])
        st.rerun()
