"""
core/tutorial_data.py — チュートリアル用の練習データ（基準 1 本＋時期の違う比較 4 本）

2 工程（供給工程・圧入工程）の設備を 1 ms 周期で 30 サイクル記録した APB 形式の CSV を作る。
時期が進むにつれて、次の 2 つの変化が少しずつ出るように仕込んである（チュートリアルで見つけてもらう）。
  - 供給工程「部品検出」が入るタイミングのばらつきが大きくなる（ステップの長さがばらつく → 判定・工程能力に出る）
  - 圧入工程「プレス力」のピークが少しずつ上がる（波形検査点の推移に出る）
乱数の種は固定なので、毎回同じデータになる。
"""
import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

CYCLE_MS = 600
N_CYCLES = 30

# (ファイル名, 開始日時, 部品検出のばらつき σ[ms], プレス力ピーク[kN])
PERIODS = [
    ("基準_立ち上げ時.csv", "2026-04-01 08:00:00", 2.0, 25.0),
    ("比較_1か月後.csv", "2026-05-01 08:00:00", 3.0, 25.4),
    ("比較_3か月後.csv", "2026-07-01 08:00:00", 6.0, 26.3),
    ("比較_6か月後.csv", "2026-10-01 08:00:00", 10.0, 27.6),
    ("比較_9か月後.csv", "2027-01-05 08:00:00", 15.0, 29.2),
]
REF_FILE = PERIODS[0][0]
CMP_FILES = [p[0] for p in PERIODS[1:]]

_README = """APB タイミング解析 チュートリアル用の練習データ

ZIP を展開すると、次の 5 つの CSV が出てきます。チュートリアルの案内に従って、画面から読み込んでください。

  基準_立ち上げ時.csv   設備を立ち上げたとき（正常）のデータ。基準データとして使います
  比較_1か月後.csv      1 か月後のデータ
  比較_3か月後.csv      3 か月後のデータ
  比較_6か月後.csv      6 か月後のデータ
  比較_9か月後.csv      9 か月後のデータ

どれも 2 工程（供給工程・圧入工程）の設備を 1 ms 周期で 30 サイクル記録したものです。
時期が進むにつれて、ある変化が少しずつ起きています。チュートリアルの中で探してみてください。
"""


def _ramp(t, t0, t1, a, b):
    return a + (b - a) * np.clip((t - t0) / (t1 - t0), 0.0, 1.0)


def _period_df(start: str, jitter_sd: float, peak: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = CYCLE_MS * N_CYCLES
    t_all = np.arange(n, dtype=float)
    cyc = (t_all // CYCLE_MS).astype(int)
    t = t_all - cyc * CYCLE_MS                       # サイクル開始からの時間 [ms]
    j = lambda s: rng.normal(0, s, N_CYCLES)[cyc]    # サイクルごとのゆらぎ（サイクル内は一定）
    on = lambda a, b: ((t >= a) & (t < b)).astype(int)

    d = {"Date Time": (pd.Timestamp(start) + pd.to_timedelta(t_all, unit="ms")).strftime("%Y-%m-%d %H:%M:%S.%f")}
    # ── 供給工程 ──
    detect = np.clip(320 + j(jitter_sd), 312, 405)  # 部品検出が入る時刻（時期が進むとばらつく）
    feed_req = 40 + j(1.0)
    slide = 180 + j(1.5)
    d["供給工程.サイクル開始"] = on(0, 10)
    d["供給工程.投入要求"] = on(feed_req, slide)
    d["供給工程.スライダ前進"] = on(slide, slide + 130)
    d["供給工程.部品検出"] = on(detect, 415)
    d["供給工程.クランプON"] = on(415, 470 + j(1.0))
    d["供給工程.実行中"] = on(0, 520)
    d["供給工程.スライダ変位[mm]"] = np.where(t < 470, _ramp(t, slide, slide + 130, 0, 120), 0) \
        + rng.normal(0, 0.05, n)
    d["供給工程.クランプ圧力[MPa]"] = np.where(t < 520, _ramp(t, 418, 440, 0, 8), _ramp(t, 520, 545, 8, 0)) \
        + rng.normal(0, 0.04, n)
    # ── 圧入工程 ──
    pk = peak + j(0.15)
    contact = 200 + j(1.5)
    d["圧入工程.サイクル開始"] = on(0, 10)
    d["圧入工程.プレス下降"] = on(60 + j(1.0), 330)
    d["圧入工程.プレス接触"] = on(contact, 400)
    d["圧入工程.プレス上昇"] = on(400, 480 + j(1.0))
    d["圧入工程.実行中"] = on(0, 500)
    d["圧入工程.プレス力[kN]"] = np.where(t < 398, _ramp(t, contact, 398, 0, 1) * pk, _ramp(t, 398, 440, 1, 0) * pk) \
        + rng.normal(0, 0.08, n)
    d["圧入工程.プレス変位[mm]"] = np.where(t < 200, _ramp(t, 60, 200, 0, 8),
                                    np.where(t < 398, _ramp(t, 200, 398, 8, 40), _ramp(t, 398, 480, 40, 0))) \
        + rng.normal(0, 0.03, n)
    df = pd.DataFrame(d)
    for c in df.columns[1:]:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    return df


@st.cache_data(show_spinner=False)
def tutorial_frames() -> dict:
    """{ファイル名: CSV の中身（bytes）}"""
    out = {}
    for i, (name, start, sd, peak) in enumerate(PERIODS):
        out[name] = _period_df(start, sd, peak, seed=100 + i).to_csv(index=False).encode("utf-8")
    return out


@st.cache_data(show_spinner=False)
def tutorial_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, raw in tutorial_frames().items():
            z.writestr(f"APB_チュートリアル/{name}", raw)
        z.writestr("APB_チュートリアル/README.txt", _README)
    return buf.getvalue()
