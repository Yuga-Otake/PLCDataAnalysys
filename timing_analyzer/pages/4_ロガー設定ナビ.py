"""
4_ロガー設定ナビ.py — 保全担当者向けデータ取得設計支援ページ

機能:
  - タイミング解析済み CSV をもとに「どの変数をロガーに登録するか」
    「サンプリング周期は何 ms か」「何秒分記録すれば良いか」を自動提示
  - session_state["df"] があれば自動引き継ぎ、なければ新規アップロード
"""

import streamlit as st
import pandas as pd
import numpy as np

from analyzer import (
    load_csv,
    detect_bool_columns,
    calc_variable_periods,
    normalize_bool_series,
)

st.set_page_config(page_title="ロガー設定ナビ", page_icon="📋", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── APB 標準サンプリング周期 [ms] ────────────────────────────────
APB_STANDARD_MS = [1, 2, 5, 10, 20, 50, 100]


# ═══════════════════════════════════════════════════════════════
# ユーティリティ関数
# ═══════════════════════════════════════════════════════════════

def recommend_sampling_ms(min_period_ms: float) -> int:
    """最速変化変数の周期 / 4 → APB 標準値に切り下げ（ナイキスト 4倍則）"""
    target = min_period_ms / 4.0
    candidates = [v for v in APB_STANDARD_MS if v <= target]
    return candidates[-1] if candidates else APB_STANDARD_MS[0]


def get_program_blocks(col_names: list) -> list:
    """'ProgramBlock.VarName' 形式から ProgramBlock を抽出しユニーク化"""
    blocks = set()
    for c in col_names:
        if "." in c:
            blocks.add(c.split(".")[0])
    return sorted(blocks)


def classify_variable(change_count: int, period_ms, period_cv) -> str:
    """
    change_count : RISE 回数（Bool）または値変化回数（数値）
    period_ms    : RISE 間隔平均（None = 変化なし or 計算不能）
    period_cv    : 周期変動係数 = std/mean（安定性指標、None 可）
    """
    if change_count == 0:
        return "❌ 除外推奨"
    if period_ms is not None and period_cv is not None and period_cv < 0.05:
        return "🎯 トリガー候補"
    if change_count >= 2:
        return "✅ 記録対象"
    return "⚠️ 確認推奨"


def calc_period_cv(df: pd.DataFrame, col: str, col_type: str) -> tuple:
    """(change_count, period_ms, period_cv) を返す"""
    try:
        if col_type == "bool":
            series = normalize_bool_series(df[col]).values
            prev   = np.concatenate([[series[0]], series[:-1]])
            rises  = np.where((prev == 0) & (series == 1))[0]
            change_count = int(len(rises))
            if len(rises) >= 2:
                times    = df["Timestamp"].iloc[rises]
                diffs_ms = times.diff().dropna().dt.total_seconds() * 1000
                mean_p   = float(diffs_ms.mean())
                std_p    = float(diffs_ms.std(ddof=0)) if len(diffs_ms) > 1 else 0.0
                cv       = std_p / mean_p if mean_p > 0 else None
                return change_count, mean_p, cv
            return change_count, None, None
        else:
            # 数値変数: 差分が 0 でない行をカウント
            vals   = df[col].dropna().values
            diffs  = np.diff(vals.astype(float))
            change_count = int(np.sum(diffs != 0))
            return change_count, None, None
    except Exception:
        return 0, None, None


# ═══════════════════════════════════════════════════════════════
# ページタイトル
# ═══════════════════════════════════════════════════════════════

st.title("📋 ロガー設定ナビ")
st.caption("取得済みのタイミングデータから、APB Variable Logger の設定値を自動提案します。")

# ═══════════════════════════════════════════════════════════════
# データソース
# ═══════════════════════════════════════════════════════════════

df = st.session_state.get("df", None)
source_label = ""

if df is not None:
    source_label = st.session_state.get("uploaded_filename", "（タイミング解析から引き継ぎ）")
    st.info(f"✅ タイミング解析のデータを引き継いでいます： **{source_label}**　（{len(df):,} 行 × {len(df.columns)} 列）")
    with st.expander("別のCSVをアップロードする場合はこちら"):
        up = st.file_uploader("CSV ファイルを選択", type=["csv"], key="nav_upload")
        if up is not None:
            df = load_csv(up)
            source_label = up.name
            st.success(f"新しいファイルをロードしました: {up.name}")
else:
    st.warning("タイミング解析でCSVをロードしていません。こちらからアップロードしてください。")
    up = st.file_uploader("CSV ファイルを選択", type=["csv"], key="nav_upload_main")
    if up is not None:
        df = load_csv(up)
        source_label = up.name
        st.success(f"ロード完了: {up.name}　({len(df):,} 行)")

if df is None:
    st.stop()

# ── Timestamp 列の確認 ──────────────────────────────────────────
if "Timestamp" not in df.columns:
    st.error("Timestamp 列が見つかりません。")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# 変数分析
# ═══════════════════════════════════════════════════════════════

col_types  = detect_bool_columns(df)
bool_cols  = [c for c, t in col_types.items() if t == "bool"]
num_cols   = [c for c, t in col_types.items() if t == "numeric"]
all_signal = bool_cols + num_cols

# 周期計算（Bool のみ）
periods = calc_variable_periods(df, bool_cols)

# 各変数のスクリーニング
rows = []
for col in all_signal:
    ctype    = col_types[col]
    period   = periods.get(col) if ctype == "bool" else None
    change_count, period_ms, cv = calc_period_cv(df, col, ctype)

    # period_ms は calc_variable_periods の値を優先（精度が高い）
    if ctype == "bool" and period is not None:
        period_ms = period
        # CV を再計算（calc_variable_periods は mean のみ返すため）
        try:
            series = normalize_bool_series(df[col]).values
            prev   = np.concatenate([[series[0]], series[:-1]])
            rises  = np.where((prev == 0) & (series == 1))[0]
            if len(rises) >= 2:
                times    = df["Timestamp"].iloc[rises]
                diffs_ms = times.diff().dropna().dt.total_seconds() * 1000
                std_p    = float(diffs_ms.std(ddof=0)) if len(diffs_ms) > 1 else 0.0
                cv       = std_p / period_ms if period_ms > 0 else None
        except Exception:
            pass

    classification = classify_variable(change_count, period_ms, cv)
    block = col.split(".")[0] if "." in col else "（その他）"

    rows.append({
        "変数名":       col,
        "種別":         "Bool" if ctype == "bool" else "数値",
        "ブロック":     block,
        "変化回数":     change_count,
        "推定周期[ms]": round(period_ms, 1) if period_ms is not None else None,
        "推奨分類":     classification,
        "登録":         classification not in ("❌ 除外推奨",),
    })

screen_df = pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════
# セクション 1: 推奨設定値カード
# ═══════════════════════════════════════════════════════════════

st.divider()
st.subheader("📊 ロガー設定値アドバイザー")

trigger_candidates = screen_df[screen_df["推奨分類"] == "🎯 トリガー候補"]
valid_periods      = screen_df["推定周期[ms]"].dropna()

# 推奨サンプリング周期
if len(valid_periods) > 0:
    min_period    = float(valid_periods.min())
    rec_sampling  = recommend_sampling_ms(min_period)
else:
    min_period    = None
    rec_sampling  = 1  # フォールバック

# 推奨ログウィンドウ（トリガー候補の平均周期 × 1.5、なければ全変数最大周期 × 1.5）
if len(trigger_candidates) > 0 and trigger_candidates["推定周期[ms]"].notna().any():
    trigger_period_ms = float(trigger_candidates["推定周期[ms]"].dropna().mean())
    rec_window_ms     = round(trigger_period_ms * 1.5)
elif len(valid_periods) > 0:
    rec_window_ms = round(float(valid_periods.max()) * 1.5)
else:
    rec_window_ms = 1000

# 推奨トリガー変数（CV 最小 = 最も安定した周期変数）
rec_trigger = "—"
if len(trigger_candidates) > 0:
    rec_trigger = trigger_candidates.iloc[0]["変数名"]

# 変数数サマリー
n_trigger  = len(trigger_candidates)
n_record   = len(screen_df[screen_df["推奨分類"] == "✅ 記録対象"])
n_check    = len(screen_df[screen_df["推奨分類"] == "⚠️ 確認推奨"])
n_exclude  = len(screen_df[screen_df["推奨分類"] == "❌ 除外推奨"])
n_bool_tot = len(bool_cols)
n_num_tot  = len(num_cols)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🕐 推奨サンプリング周期", f"{rec_sampling} ms",
              help=f"最速変化変数の推定周期 {min_period:.0f} ms の 1/4（ナイキスト 4倍則）をもとにAPB標準値に切り下げ" if min_period else "")
with c2:
    st.metric("⏱ 推奨ログウィンドウ長", f"{rec_window_ms:,} ms",
              delta=f"≈ {rec_window_ms/1000:.1f} 秒",
              help="トリガー候補の平均周期 × 1.5")
with c3:
    st.metric("🎯 推奨トリガー変数", rec_trigger,
              help="周期変動係数（CV）が最も小さい変数（最も安定した周期信号）")
with c4:
    st.metric("📦 変数数サマリー",
              f"Bool {n_bool_tot} / 数値 {n_num_tot}",
              delta=f"🎯{n_trigger} ✅{n_record} ⚠️{n_check} ❌{n_exclude}",
              help="🎯トリガー候補 / ✅記録対象 / ⚠️確認推奨 / ❌除外推奨")

# ═══════════════════════════════════════════════════════════════
# セクション 2: 変数スクリーニングテーブル
# ═══════════════════════════════════════════════════════════════

st.divider()
st.subheader("🔍 変数スクリーニングテーブル")
st.caption("チェックボックスで登録変数を選択できます。フィルタで絞り込んでください。")

# フィルタバー
fb1, fb2, fb3 = st.columns([2, 3, 3])
with fb1:
    blocks_available = ["全て"] + get_program_blocks(all_signal)
    sel_block = st.selectbox("プログラムブロック", blocks_available, key="nav_block")
with fb2:
    all_cls = ["🎯 トリガー候補", "✅ 記録対象", "⚠️ 確認推奨", "❌ 除外推奨"]
    sel_cls = st.multiselect("推奨分類", all_cls, default=all_cls, key="nav_cls")
with fb3:
    kw = st.text_input("キーワード検索", placeholder="変数名の一部を入力", key="nav_kw")

# フィルタ適用
filtered = screen_df.copy()
if sel_block != "全て":
    filtered = filtered[filtered["ブロック"] == sel_block]
if sel_cls:
    filtered = filtered[filtered["推奨分類"].isin(sel_cls)]
if kw:
    filtered = filtered[filtered["変数名"].str.contains(kw, case=False, na=False)]

# 並び替え（推奨分類 → 推定周期 昇順）
cls_order = {"🎯 トリガー候補": 0, "✅ 記録対象": 1, "⚠️ 確認推奨": 2, "❌ 除外推奨": 3}
filtered = filtered.copy()
filtered["_sort"] = filtered["推奨分類"].map(cls_order)
filtered = filtered.sort_values(["_sort", "推定周期[ms]"], na_position="last").drop(columns=["_sort"])

st.write(f"表示中: **{len(filtered)}** 変数（全 {len(screen_df)} 変数）")

# data_editor でチェックボックス付きテーブル表示
edited = st.data_editor(
    filtered.reset_index(drop=True),
    column_config={
        "登録":         st.column_config.CheckboxColumn("✅ ロガー登録", width="small"),
        "変数名":       st.column_config.TextColumn("変数名", width="large"),
        "種別":         st.column_config.TextColumn("種別", width="small"),
        "ブロック":     st.column_config.TextColumn("ブロック", width="medium"),
        "変化回数":     st.column_config.NumberColumn("変化回数", format="%d"),
        "推定周期[ms]": st.column_config.NumberColumn("推定周期[ms]", format="%.1f"),
        "推奨分類":     st.column_config.TextColumn("推奨分類", width="medium"),
    },
    use_container_width=True,
    hide_index=True,
    key="nav_editor",
    height=420,
)

# ═══════════════════════════════════════════════════════════════
# セクション 3: エクスポート
# ═══════════════════════════════════════════════════════════════

st.divider()
st.subheader("💾 エクスポート")

selected_vars = edited[edited["登録"] == True]["変数名"].tolist()

st.write(f"チェック済み変数: **{len(selected_vars)}** 件")

if selected_vars:
    # APB ロガー設定画面への貼り付け用テキスト（1行1変数）
    txt_content = "\n".join(selected_vars)
    st.download_button(
        label="📥 変数リストをダウンロード (.txt)",
        data=txt_content,
        file_name="apb_logger_variables.txt",
        mime="text/plain",
    )

    with st.expander("変数リストのプレビュー"):
        st.code(txt_content, language=None)
else:
    st.info("テーブルで変数にチェックを入れると、ダウンロードボタンが有効になります。")
