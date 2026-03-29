"""
0_このツールについて.py — ヘルプ・概要説明ページ（日英対応）
"""

import streamlit as st

st.set_page_config(
    page_title="このツールについて | About This Tool",
    page_icon="ℹ️",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── 言語切り替え ─────────────────────────────────────────────────
lang = st.radio(
    "言語 / Language",
    ["🇯🇵 日本語", "🇬🇧 English"],
    horizontal=True,
    key="help_lang",
    label_visibility="collapsed",
)
JP = (lang == "🇯🇵 日本語")

# ═══════════════════════════════════════════════════════════════
# コンテンツ定義（日本語 / English）
# ═══════════════════════════════════════════════════════════════

if JP:
    st.title("ℹ️ このツールについて")
    st.caption("APB タイミング解析ツール — APBの補助ツール｜立ち上げタクト短縮・予兆保全を支援")
else:
    st.title("ℹ️ About This Tool")
    st.caption("APB Timing Analyzer — APB Companion Tool | Supporting Takt Reduction & Predictive Maintenance")

st.divider()

# ── セクション 1: APB とは ────────────────────────────────────
if JP:
    st.subheader("🔁 APB（Automation PlayBack）とは")
    st.markdown("""
APB（**Automation PlayBack**）は、PLCが制御する生産設備の **全変数データを一定期間にわたって記録・再生できる機能**です。
もともとは**設備の不具合発生時に「何が起きていたか」を事後分析するための原因究明ツール**として開発されました。

- 設備が動作している間、数万に及ぶ内部変数（センサ値・フラグ・タイマー値など）を **ms 単位でログ取得**
- 取得したCSVデータをもとに、設備の動作を時系列で振り返ることができる
- 不具合発生時の「何が起きていたか」を事後に詳細解析できる
""")
    st.info("""
**📌 このツールはAPBの補助ツールです**

APBが持つ「**設備の全データを網羅的に取得する**」という能力に注目し、原因分析という本来用途を超えた新たな価値を引き出します。

→ **立ち上げ時のタクトタイムを基準として記録し、現状との差分を可視化**することで、問題の予兆を早期に発見。
→ ベテランの経験や勘に頼らず、**データドリブンな予兆保全・メンテナンス判断**を実現します。
""")
else:
    st.subheader("🔁 What is APB (Automation PlayBack)?")
    st.markdown("""
APB (**Automation PlayBack**) is a function that **records and replays all PLC variable data over a defined time period** for production equipment under PLC control.
It was originally developed as a **fault cause analysis tool** — to investigate "what was happening" when equipment failures occurred.

- While equipment is running, tens of thousands of internal variables (sensor values, flags, timer values, etc.) are **logged at millisecond resolution**
- The collected CSV data enables time-series review of equipment operation
- Post-incident deep analysis of "what was happening" during a fault becomes possible
""")
    st.info("""
**📌 This tool is an APB companion tool**

It leverages APB's core capability — **comprehensive capture of all equipment variables** — and re-builds new value beyond its original fault-analysis purpose.

→ **Records takt time at initial equipment startup as a baseline, then visualizes deviations over time** to detect early signs of degradation.
→ Enables **data-driven predictive maintenance decisions** without relying on veteran expertise or intuition.
""")

st.divider()

# ── セクション 2: 背景・課題 ──────────────────────────────────
if JP:
    st.subheader("⚠️ このツールが生まれた背景")
    st.markdown("""
APBはすでに現場に導入されており、設備の**全変数データを取得し続けている**。
しかしそのデータは「不具合が起きたときだけ見る」という使われ方に留まっていた。

> 「せっかく全データが揃っているのに、不具合対応にしか使っていないのはもったいない」

この発想から、APBデータを**立ち上げタクト短縮・予兆保全**という新しい目的に再活用するツールが生まれました。
""")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**このツールを使う前の現場では...**

- 🔴 **不具合が起きてから初めて気づく**
  → 生産停止・品質問題が表面化してから対処
- 🔴 **じわじわ進む性能劣化に気づかない**
  → タクトタイムのわずかな悪化、動作遅延の蓄積
- 🔴 **原因究明に時間と経験が必要**
  → ベテラン技術者でないとデータを読めない
- 🔴 **設備立ち上げ時との比較ができない**
  → 「最初はどうだったか」という基準がない
""")
    with col_b:
        st.markdown("""
**このツール（APBの補助）を使うと...**

- 🟢 **立ち上げ時との差分を定量的に可視化**
  → 「いつから」「どのステップが」遅くなったかわかる
- 🟢 **問題が顕在化する前に予兆を検知**
  → NG 判定・σ逸脱で異常を早期発見
- 🟢 **データを見るだけで根拠ある判断が可能**
  → 経験に頼らず誰でも分析できる
- 🟢 **メンテナンスのタイミングを根拠をもって決定**
  → 「そろそろ調整が必要」を数値で説明できる
""")
else:
    st.subheader("⚠️ Background: Why This Tool Was Created")
    st.markdown("""
APB is already deployed on the shop floor, continuously capturing **all equipment variable data**.
Yet that data was only ever used reactively — looked at only after a fault occurred.

> "We have all this data available, but we're only using it when something breaks — that's a waste."

From that insight, this tool was created to **re-purpose APB data** for a new goal: reducing startup takt time and enabling predictive maintenance.
""")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Before this tool, the shop floor faced...**

- 🔴 **Problems only noticed after they occur**
  → Production stops and quality issues before any action
- 🔴 **Gradual performance degradation goes undetected**
  → Slow accumulation of takt time increases and motion delays
- 🔴 **Root cause analysis requires time and expertise**
  → Only experienced engineers can interpret the data
- 🔴 **No baseline for comparison with initial setup**
  → "How was it when we first started?" has no answer
""")
    with col_b:
        st.markdown("""
**With this tool (APB companion)...**

- 🟢 **Quantitative visualization of deviation from baseline**
  → See "since when" and "which step" slowed down
- 🟢 **Detect early signs before problems become critical**
  → NG judgment and σ-deviation catch anomalies early
- 🟢 **Data-driven decisions without relying on experience**
  → Anyone can analyze — no expert required
- 🟢 **Maintenance timing backed by evidence**
  → "Adjustment needed soon" explained with numbers
""")

st.divider()

# ── セクション 3: 対象ユーザー・適用現場 ────────────────────────
if JP:
    st.subheader("🏭 対象ユーザーと適用現場")

    u1, u2, u3 = st.columns(3)
    with u1:
        st.markdown("""
**👷 保全担当者**
- 設備の定期点検・予防保全に活用
- 「どこを調整すべきか」をデータで判断
- ロガー設定ナビで変数取得設計も支援
""")
    with u2:
        st.markdown("""
**🔧 生産技術者**
- 設備立ち上げ時の基準値を登録・管理
- 工程ごとのタクトタイム分析・改善
- 傾向解析で経時変化を追跡
""")
    with u3:
        st.markdown("""
**🏭 適用現場の例**
- 自動車部品の組立ライン
- 食品・飲料の充填・包装ライン
- 電子部品の実装・検査ライン
- その他 PLC 制御の自動化設備全般
""")
else:
    st.subheader("🏭 Target Users & Applications")

    u1, u2, u3 = st.columns(3)
    with u1:
        st.markdown("""
**👷 Maintenance Staff**
- Apply to scheduled inspections and preventive maintenance
- Data-driven decisions on "what needs adjustment"
- Logger Setup Navigator assists with variable selection
""")
    with u2:
        st.markdown("""
**🔧 Production Engineers**
- Register and manage baseline values at equipment setup
- Takt time analysis and improvement per process
- Track long-term trends with trend analysis
""")
    with u3:
        st.markdown("""
**🏭 Example Applications**
- Automotive parts assembly lines
- Food & beverage filling/packaging lines
- Electronics assembly and inspection lines
- Any PLC-controlled automated equipment
""")

st.divider()

# ── セクション 4: 機能説明 ────────────────────────────────────
if JP:
    st.subheader("📐 主な機能")
else:
    st.subheader("📐 Key Features")

feat_cols = st.columns(2)

if JP:
    features = [
        ("⚙️ 画面設定 — 工程・ステップ設定",
         "解析する工程（ProA, ProB, ProC...）ごとに、どの変数を「ステップ」として計測するかを登録。"
         "ガントチャートでサイクルごとのタイミングを可視化します。"),
        ("📐 新データ評価 — 現状データの異常検知",
         "立ち上げ時に登録した**基準値**と、現在取得した新しいCSVを比較。"
         "各ステップの遅れ・ばらつきがσ判定でNG表示されます。"),
        ("📈 傾向解析 — 複数時期の経時変化追跡",
         "複数の時期のCSVをまとめてアップロードし、ステップ別の時系列トレンドをグラフ表示。"
         "「いつから悪化し始めたか」を一目で確認できます。"),
        ("📋 ロガー設定ナビ — データ取得設計支援",
         "既存のAPBデータから「どの変数をロガーに登録すべきか」「サンプリング周期は何ms か」を自動提案。"
         "保全担当者がデータ取得設計を経験なしに行えます。"),
        ("💾 設定ファイル保存・読み込み",
         "工程設定・ステップ・基準値をJSONとして保存。別の端末や次回起動時に即座に復元できます。"),
        ("🎯 基準値登録",
         "立ち上げ時の平均値・最小値・特定サイクル値を基準として登録。以降の評価・傾向解析の比較軸になります。"),
    ]
else:
    features = [
        ("⚙️ Configuration — Process & Step Setup",
         "Register which variables to measure as 'steps' for each process (ProA, ProB, ProC...). "
         "Visualize per-cycle timing as Gantt charts."),
        ("📐 New Data Evaluation — Anomaly Detection",
         "Compare **baseline values** registered at equipment startup against newly acquired CSV data. "
         "Each step's delay and variation is flagged as NG using σ judgment."),
        ("📈 Trend Analysis — Long-term Change Tracking",
         "Upload multiple CSVs from different time periods to display step-by-step time-series trends. "
         "See at a glance 'when did it start degrading'."),
        ("📋 Logger Setup Navigator — Data Acquisition Design",
         "Automatically recommends 'which variables to register' and 'what sampling interval to use' "
         "based on existing APB data. Enables maintenance staff to design data acquisition without expertise."),
        ("💾 Settings Save & Load",
         "Save process configurations, steps, and baseline values as JSON. "
         "Instantly restore on a different device or next session."),
        ("🎯 Baseline Registration",
         "Register average, minimum, or specific cycle values at startup as baselines. "
         "These become the reference for all subsequent evaluation and trend analysis."),
    ]

for i, (title, desc) in enumerate(features):
    with feat_cols[i % 2]:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)

st.divider()

# ── セクション 5: 使い方フロー ─────────────────────────────────
if JP:
    st.subheader("🗺️ 推奨ワークフロー")
    st.markdown("""
```
① CSVをアップロード（またはサンプルデータを使用）
        ↓
② 工程を追加（+ 工程を追加）→ トリガー変数を選択
        ↓
③ ステップを追加（各工程に計測したい変数を登録）
        ↓
④ ガントチャートで現状のタイミングを確認
        ↓
⑤ 基準値を登録（設備立ち上げ時のデータで行うのが理想）
        ↓
⑥ 新データ評価 → 定期的に新しいCSVをアップロードして比較
        ↓
⑦ 傾向解析 → 複数時期のCSVで経時劣化をトラッキング
        ↓
⑧ 異常・傾向変化を検知したらメンテナンスへ
```
""")
    st.info("💡 設定をJSONで保存しておくと、次回から⑥の評価だけで運用できます。")
else:
    st.subheader("🗺️ Recommended Workflow")
    st.markdown("""
```
① Upload CSV (or use sample data)
        ↓
② Add a process (+ Add Process) → Select trigger variable
        ↓
③ Add steps (register variables to measure for each process)
        ↓
④ Review current timing via Gantt chart
        ↓
⑤ Register baseline (ideally using data from initial equipment setup)
        ↓
⑥ New Data Evaluation → Periodically upload new CSV and compare
        ↓
⑦ Trend Analysis → Track gradual degradation with multi-period CSVs
        ↓
⑧ When anomaly or trend change detected → trigger maintenance
```
""")
    st.info("💡 Save settings as JSON so that future use only requires step ⑥ onward.")

st.divider()

# ── フッター ──────────────────────────────────────────────────
if JP:
    st.caption("© APB タイミング解析ツール　— 設備の健康管理をデータドリブンへ")
else:
    st.caption("© APB Timing Analyzer — Bringing data-driven health management to production equipment")
