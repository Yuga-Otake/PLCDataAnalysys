"""
ai_chat.py — Google Gemini を使った AI チャット機能のバックエンド

APIキー設定方法（優先度順）:
  1. .streamlit/secrets.toml に GOOGLE_API_KEY = "AIza..."
  2. 環境変数 GOOGLE_API_KEY
  3. チャット画面のサイドバーで入力
"""

import os
import streamlit as st
import google.generativeai as genai
from google.generativeai import types as gtypes

SYSTEM_PROMPT = """あなたはPLC設備タイミング解析ツール「APBタイミング解析」のAIアシスタントです。
ユーザーは製造現場の保全担当者や生産技術者です。

## 役割
- 現在の解析データについて日本語で答えてください
- 「〜を見せて」「〜に切り替えて」「〜を表示して」などの操作指示にはツールを呼び出してください
- 「〜はなぜ？」「〜の原因は？」「〜について教えて」にはテキストで回答してください
- 具体的な数値を引用して根拠ある回答をしてください（200〜400文字程度を目安に）
- データにない情報は「現在の解析データには含まれていません」と回答してください

## 画面の説明
- 設定モード: 工程・ステップの設定とガントチャート表示
- 監視モード: リアルタイム傾向監視
- 品質分析モード: Cpk等の品質指標確認
"""

APP_TOOLS = gtypes.Tool(function_declarations=[
    gtypes.FunctionDeclaration(
        name="switch_view_mode",
        description="アプリの表示モードを切り替える。設定・監視・品質分析の3種類。",
        parameters=gtypes.Schema(
            type=gtypes.Type.OBJECT,
            properties={
                "mode": gtypes.Schema(
                    type=gtypes.Type.STRING,
                    enum=["設定", "監視", "品質分析"],
                    description="切り替える表示モード",
                ),
            },
            required=["mode"],
        ),
    ),
    gtypes.FunctionDeclaration(
        name="expand_process",
        description="メイン画面で指定した工程のセクションを展開して表示する。",
        parameters=gtypes.Schema(
            type=gtypes.Type.OBJECT,
            properties={
                "process_name": gtypes.Schema(
                    type=gtypes.Type.STRING,
                    description="展開する工程名",
                ),
            },
            required=["process_name"],
        ),
    ),
    gtypes.FunctionDeclaration(
        name="navigate_to_histogram",
        description="指定した工程・ステップのヒストグラム詳細ページに移動する。",
        parameters=gtypes.Schema(
            type=gtypes.Type.OBJECT,
            properties={
                "process_name": gtypes.Schema(type=gtypes.Type.STRING, description="工程名"),
                "step_name": gtypes.Schema(type=gtypes.Type.STRING, description="ステップ名"),
            },
            required=["process_name", "step_name"],
        ),
    ),
    gtypes.FunctionDeclaration(
        name="switch_active_csv",
        description="解析対象のCSVをラベル名で切り替える。",
        parameters=gtypes.Schema(
            type=gtypes.Type.OBJECT,
            properties={
                "csv_label": gtypes.Schema(
                    type=gtypes.Type.STRING,
                    description="切り替えるCSVのラベル名",
                ),
            },
            required=["csv_label"],
        ),
    ),
])

# モード名マッピング（絵文字付きの実際のsession_state値へ）
_MODE_MAP = {
    "設定": "⚙️ 設定",
    "監視": "👁️ 監視",
    "品質分析": "📊 品質分析",
}


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("GOOGLE_API_KEY")
        if key:
            return key
    except Exception:
        pass
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    return st.session_state.get("ai_api_key_input") or None


def _pk(pname: str, suffix: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in pname)
    return f"P_{safe}__{suffix}"


def build_analysis_context(ss: dict) -> str:
    """session_state から現在の解析状態を日本語テキストで返す。例外は出さない。"""
    lines = ["=== APB タイミング解析 現在の状態 ==="]

    lines.append(f"表示モード: {ss.get('view_mode', '不明')}")

    csv_store = ss.get("csv_store", {})
    active_key = ss.get("active_csv", "")
    if active_key and active_key in csv_store:
        entry = csv_store[active_key]
        label = entry.get("label", active_key)
        df = entry.get("df")
        n_rows = f"{len(df):,}" if df is not None else "不明"
        lines.append(f"アクティブCSV: {label} ({n_rows}行)")
    else:
        lines.append("CSVデータ: 未読み込み")

    if len(csv_store) > 1:
        all_labels = [e.get("label", k) for k, e in csv_store.items()]
        lines.append(f"読み込み済みCSV一覧: {', '.join(all_labels)}")

    processes = ss.get("processes", {})
    if not processes:
        lines.append("\n工程: 未設定（メイン画面で工程を追加してください）")
        return "\n".join(lines)

    for pname in processes:
        trigger = ss.get(_pk(pname, "trigger"), "不明")
        edge = ss.get(_pk(pname, "edge"), "RISE")
        takt = ss.get(_pk(pname, "takt"), 0)
        steps_list = ss.get(_pk(pname, "steps_list"), [])

        lines.append(f"\n工程: {pname}")
        lines.append(f"  トリガー: {trigger} ({'RISE=立上り' if edge == 'RISE' else 'FALL=立下り'})")
        lines.append(f"  タクト目標: {takt} ms" if takt > 0 else "  タクト目標: 未設定")
        lines.append(f"  設定ステップ数: {len(steps_list)}")

        hist_ctx = ss.get(f"_hist_ctx_{pname}", {})
        step_stats = hist_ctx.get("step_stats", [])
        result_df = hist_ctx.get("result_df")

        if result_df is not None and len(result_df) > 0:
            lines.append(f"  解析サイクル数: {len(result_df)}")

        if step_stats:
            lines.append("  ステップ統計:")
            total_t = sum(s.get("mean", 0) for s in step_stats) or 1.0
            for s in step_stats:
                mean_v = s.get("mean", 0)
                std_v = s.get("abs_std", s.get("std", 0))
                min_v = s.get("abs_min", s.get("min", 0))
                max_v = s.get("abs_max", s.get("max", 0))
                pct = mean_v / total_t * 100
                cpk_str = "—"
                if takt > 0 and std_v > 0:
                    cpk = min((takt - mean_v) / (3 * std_v), mean_v / (3 * std_v))
                    flag = "✅" if cpk >= 1.33 else ("⚠️" if cpk >= 1.0 else "❌")
                    cpk_str = f"{cpk:.2f} {flag}"
                lines.append(
                    f"    - {s.get('name', '?')}: "
                    f"平均={mean_v:.1f}ms, σ={std_v:.1f}ms, "
                    f"min={min_v:.1f}, max={max_v:.1f}, "
                    f"タクト比={pct:.1f}%, Cpk={cpk_str}"
                )
        elif steps_list:
            names = [step["name"] for step in steps_list if "name" in step]
            lines.append(f"  ステップ: {', '.join(names)}（解析未実行 — メイン画面で表示してください）")
        else:
            lines.append("  ステップ: 未設定")

    return "\n".join(lines)


def get_or_init_history(ss: dict) -> list:
    if "ai_chat_history" not in ss:
        ss["ai_chat_history"] = []
    return ss["ai_chat_history"]


def append_to_history(ss: dict, role: str, content: str, max_turns: int = 10) -> None:
    """role は "user" または "model"（Gemini形式）"""
    ss.setdefault("ai_chat_history", [])
    ss["ai_chat_history"].append({"role": role, "parts": [{"text": content}]})
    limit = max_turns * 2
    if len(ss["ai_chat_history"]) > limit:
        ss["ai_chat_history"] = ss["ai_chat_history"][-limit:]


def history_for_display(ss: dict) -> list:
    """st.chat_message 表示用に role を "user"/"assistant" に変換"""
    result = []
    for msg in ss.get("ai_chat_history", []):
        role = "assistant" if msg.get("role") == "model" else "user"
        text = (msg.get("parts") or [{}])[0].get("text", "")
        result.append({"role": role, "content": text})
    return result


def call_gemini(
    user_message: str,
    history: list,
    context: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash",
) -> dict:
    """
    Gemini API を呼び出す。
    戻り値:
      {"type": "text", "content": str}
      {"type": "tool_use", "tool_name": str, "tool_input": dict}
      エラー時: {"type": "text", "content": "エラー: ..."}
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            tools=[APP_TOOLS],
            system_instruction=SYSTEM_PROMPT + "\n\n" + context,
        )
        chat = model.start_chat(history=history)
        response = chat.send_message(user_message)
    except Exception as e:
        return {"type": "text", "content": f"エラーが発生しました: {e}"}

    for part in response.parts:
        if hasattr(part, "function_call") and part.function_call.name:
            fc = part.function_call
            return {
                "type": "tool_use",
                "tool_name": fc.name,
                "tool_input": dict(fc.args),
            }

    try:
        text = response.text
    except Exception:
        text = "応答を取得できませんでした。"
    return {"type": "text", "content": text}


def execute_tool(tool_name: str, tool_input: dict, ss: dict) -> tuple[str, str | None]:
    """
    session_state を更新してツールを実行する。
    戻り値: (ユーザー向けメッセージ, 遷移先ページパス or None)
    """
    if tool_name == "switch_view_mode":
        raw_mode = tool_input.get("mode", "設定")
        mode = _MODE_MAP.get(raw_mode, "⚙️ 設定")
        ss["view_mode"] = mode
        return f"表示モードを「{mode}」に切り替えます。", "app.py"

    if tool_name == "expand_process":
        pname = tool_input.get("process_name", "")
        ss["_expand_new"] = pname
        return f"工程「{pname}」を展開してメイン画面に移動します。", "app.py"

    if tool_name == "navigate_to_histogram":
        pname = tool_input.get("process_name", "")
        sname = tool_input.get("step_name", "")
        if f"_hist_ctx_{pname}" in ss:
            ss["_hist_nav_proc"] = pname
            ss["_hist_nav_step"] = sname
            return f"「{pname}」の「{sname}」ヒストグラムに移動します。", "pages/5_histogram.py"
        return (
            f"「{pname}」はまだ解析されていません。"
            "メイン画面でガントチャートを表示してからお試しください。",
            None,
        )

    if tool_name == "switch_active_csv":
        label = tool_input.get("csv_label", "")
        csv_store = ss.get("csv_store", {})
        for key, entry in csv_store.items():
            if entry.get("label") == label:
                ss["active_csv"] = key
                ss["compare_mode"] = False
                return f"CSVを「{label}」に切り替えます。", "app.py"
        labels = [e.get("label", k) for k, e in csv_store.items()]
        return f"「{label}」が見つかりません。読み込み済みCSV: {labels}", None

    return "不明なツールです。", None
