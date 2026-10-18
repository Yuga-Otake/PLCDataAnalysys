"""
0_このツールについて.py — 使い方（ツールの目的・画面の構成・進め方・判定ルール・用語、日英対応）
"""
import pandas as pd
import streamlit as st

from core.evaluate import rule_text
from core.ui import page_header

lang = st.segmented_control("言語 / Language", ["日本語", "English"], default="日本語",
                            key="help_lang", label_visibility="collapsed")
JP = lang != "English"

if JP:
    page_header(
        "使い方",
        "APB（Automation PlayBack）が記録した PLC の CSV から、設備の各動作がサイクル開始から何 ms 後に"
        "起きたかを測り、立ち上げ時（正常）と今を比べて、遅れやばらつきを見つけるツールです。",
        eyebrow="ツール",
    )
else:
    page_header(
        "How to Use",
        "This tool measures, from APB (Automation PlayBack) CSV logs, how many milliseconds after each cycle start "
        "every machine action happens, and compares the current state with the startup (normal) state to find "
        "delays and growing variation.",
        eyebrow="Tools",
    )

# ── 2 つの仕事 ───────────────────────────────────────────────
st.subheader("準備と点検" if JP else "Setup and Inspection", anchor=False)
c1, c2 = st.columns(2)
with c1.container(border=True):
    if JP:
        st.markdown("**準備**（設備の立ち上げ時に 1 回）\n\n"
                    "1. データ — 正常に動いていたときの CSV を基準データとして読み込む\n"
                    "2. 工程とステップ — サイクル開始信号を選び、時間を測りたい動作をステップとして選ぶ\n"
                    "3. 基準値 — 正常時の各ステップのタイミングを基準値として登録する\n"
                    "4. 設定の保存 — 工程・ステップ・基準値を設定ファイル（JSON）に保存する")
    else:
        st.markdown("**Setup** (once, at machine startup)\n\n"
                    "1. Data — load a CSV recorded while the machine ran normally as the reference data\n"
                    "2. Processes & Steps — choose the cycle-start signal and the actions to time as steps\n"
                    "3. Baseline — register the normal timing of each step\n"
                    "4. Save settings — save processes, steps and baselines to a settings file (JSON)")
with c2.container(border=True):
    if JP:
        st.markdown("**点検**（定期的に）\n\n"
                    "1. データ — 設定ファイルを読み込み、点検したい時期の CSV を比較データとして追加する\n"
                    "2. 最新データの評価 — 基準値から外れた工程・ステップを確認する\n"
                    "3. ステップ詳細 — 気になるステップの分布・サイクルごとの値・波形を見る\n"
                    "4. 推移・工程能力 — いつから悪化したか、規格に対する余裕はどれだけかを見る")
    else:
        st.markdown("**Inspection** (periodically)\n\n"
                    "1. Data — load the settings file and add CSVs from the period to inspect\n"
                    "2. Evaluate latest data — see which processes and steps left the baseline\n"
                    "3. Step detail — check the distribution, per-cycle values and waveforms\n"
                    "4. Trend / Capability — see since when it degraded and how much margin remains")
st.caption("ホームの「準備の進み具合」に、次にやることが表示されます。" if JP
           else "The Home page shows the next thing to do under “Setup progress”.")

# ── 画面の構成 ───────────────────────────────────────────────
st.subheader("画面の構成" if JP else "Screens", anchor=False)
if JP:
    rows = [
        ("", "ホーム", "準備の進み具合、次にやること、工程の一覧、最新データの判定"),
        ("準備", "データ", "基準データ・比較データの読み込みと事前診断（記録の抜け・一定の変数・比較データとの整合）、設定ファイル"),
        ("準備", "工程とステップ", "工程の提案、動作シーケンスの仮説から登録、外れサイクルの検出、検索して追加、タイミング図"),
        ("準備", "基準値", "外れサイクルを除いて基準値を算出・登録、登録値の修正"),
        ("準備", "波形検査条件", "波形の形から出した検出点の候補（ピーク・立ち上がりなど）の採用、検出点・良品範囲の調整（詳細設定）"),
        ("点検", "最新データの評価", "工程ごとの状態、ステップの判定表、基準と今回のタイミング比較"),
        ("点検", "推移", "全ステップ×時期のヒートマップ、ステップの推移と管理図、波形検査の検出点の推移"),
        ("点検", "工程能力", "許容幅に対する Cpk、タクト目標に対する工程能力、良品条件の一覧"),
        ("ツール", "ロガー設定ナビ", "どの変数を何 ms 周期で記録すればよいかの提案"),
        ("ツール", "AI アシスタント", "解析結果についての質問、画面の切り替え"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["メニュー", "画面", "できること"]), hide_index=True, width="stretch")
else:
    rows = [
        ("", "Home", "Setup progress, next step, process list, verdict for the latest data"),
        ("Setup", "Data", "Load CSVs with a pre-check (recording gaps, constant variables, consistency), settings files"),
        ("Setup", "Processes & Steps", "Process proposals, action-sequence hypothesis, abnormal cycles, search, timing chart"),
        ("Setup", "Baseline", "Compute baselines excluding abnormal cycles, edit registered values"),
        ("Setup", "Waveform inspection", "Adopt detection points proposed from the waveform shape (peak, rise, …), tune points and good ranges (advanced)"),
        ("Inspection", "Evaluate latest data", "Status per process, step verdict table, baseline vs. current timing"),
        ("Inspection", "Trend", "Step × period heatmap, step trend and control chart, detection-point trend"),
        ("Inspection", "Capability", "Cpk against the tolerance and against the takt target, baseline list"),
        ("Tools", "Logger setup navigator", "Suggests which variables to log and at what interval"),
        ("Tools", "AI assistant", "Questions about the results, switching screens"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Menu", "Screen", "What you can do"]), hide_index=True, width="stretch")

# ── 判定ルール ───────────────────────────────────────────────
st.subheader("判定ルール" if JP else "Verdict rule", anchor=False)
if JP:
    st.markdown(rule_text())
    st.caption("各点検画面の「判定ルール」ボタンで、σ の倍数・最小許容幅・異常とする NG率を変更できます。")
else:
    st.markdown("Tolerance = baseline ± max(N·σ, minimum width) (default N = 3, minimum 2 ms). "
                "Cycles outside the tolerance are NG; an NG rate of 10% or more is *Abnormal*, at least one NG "
                "cycle is *Caution*. Event steps are judged by their time, period steps by their length. "
                "With few baseline cycles, σ is estimated generously (95% upper confidence bound).")
    st.caption("The thresholds can be changed with the “判定ルール” button on each inspection screen.")

# ── 用語 ─────────────────────────────────────────────────────
st.subheader("用語" if JP else "Terms", anchor=False)
if JP:
    terms = [
        ("工程", "1 サイクルのまとまり。サイクルの始まりを示す Bool 変数（サイクル開始信号）で区切ります。"),
        ("ステップ", "工程の中で時間を測る動作。発生時刻を測る「時刻」と、長さを測る「ON期間」「開始〜終了」「数値条件」があります。"),
        ("基準データ", "設備が正常に動いていたとき（立ち上げ時など）の CSV。2〜10 サイクルで工程・ステップを定義できますが、基準値には 10 サイクル以上を勧めます。"),
        ("比較データ", "点検したい時期の CSV。時期の違うものを複数入れると推移を確認できます。"),
        ("基準値", "正常時の各ステップのタイミング（平均）とばらつき（σ）。点検の比較の基準です。"),
        ("許容幅", "基準値からどこまで離れてよいかの幅。外に出たサイクルを NG とします。"),
        ("Cpk", "許容幅に対して、ばらつきと偏りを考えた余裕の大きさ。1.33 以上で十分、1.00 未満は不足です。"),
    ]
else:
    terms = [
        ("Process", "One cycle unit, delimited by a Bool cycle-start signal."),
        ("Step", "An action timed within a process: an event time, or a period (ON period, start–end, numeric condition)."),
        ("Reference data", "A CSV recorded while the machine ran normally. 2–10 cycles suffice to define steps; 10+ are recommended for baselines."),
        ("Comparison data", "CSVs from periods to inspect. Several periods give a trend."),
        ("Baseline", "Normal timing (mean) and variation (σ) of each step — the reference for inspection."),
        ("Tolerance", "How far a value may deviate from the baseline. Cycles outside are NG."),
        ("Cpk", "Margin against the tolerance considering spread and offset. ≥1.33 is sufficient, <1.00 insufficient."),
    ]
for t, d in terms:
    st.markdown(f"**{t}** — {d}")

# ── 背景 ─────────────────────────────────────────────────────
with st.expander("このツールが生まれた背景" if JP else "Background"):
    if JP:
        st.markdown(
            "APB は、PLC が制御する設備の全変数を ms 単位で記録・再生できる機能です。もともとは不具合が起きたときに"
            "「何が起きていたか」を調べる原因究明ツールとして使われてきました。\n\n"
            "せっかく全データが揃っているのに、不具合のときにしか見ないのはもったいない。そこで、立ち上げ時のタイミングを"
            "基準として記録し、今との差を見える化することで、じわじわ進む性能劣化を問題になる前に見つけ、"
            "経験や勘に頼らずにメンテナンスの時期を判断できるようにしたのがこのツールです。")
    else:
        st.markdown(
            "APB records and replays every PLC variable at millisecond resolution. It has mainly been used after a "
            "failure, to find out what happened.\n\n"
            "Since all the data is already there, this tool reuses it: it records the startup timing as a baseline "
            "and visualizes the difference from today, so gradual degradation is found before it becomes a problem "
            "and maintenance timing can be decided from data rather than experience.")
