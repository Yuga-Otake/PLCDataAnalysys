"""
gen_sample.py  — サンプルCSV生成スクリプト
実機 APB Variable Logger の出力フォーマットに準拠:
  - 列名: "Date Time"
  - タイムスタンプ: "2026-02-06 13:26:00.088848"（マイクロ秒）
  - 変数名: "ProgramBlock.VariableName" 形式
  - Bool値: 0 / 1（整数）

【並行動作3工程ライン】 ← タクトごとに全工程が同時に別ワークを処理
  Assembly.ProA: 供給工程  〜450ms（ピッキング・クランプ）
  Assembly.ProB: 圧入工程  〜530ms（プレスフィッティング）← ボトルネック
  Assembly.ProC: 検査排出  〜310ms（カメラ検査・排出）
  タクト時間: 600ms（ProBに合わせた最小タクト）

数値変数（Case文シミュレーション）:
  Assembly.ProB_Step  : ProB内部ステップ（0=待機, 1〜4=各フェーズ）
  Assembly.LineStep   : 残稼働工程数ベースの工程状態
                        1=全3工程実行中, 2=2工程実行中, 3=ProBのみ
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate(filename, n_cycles, seed,
             anomaly_pa_slow, anomaly_pb_slow, anomaly_pc_slow,
             anomaly_pb_extreme=None):

    rng = np.random.default_rng(seed)
    if anomaly_pb_extreme is None:
        anomaly_pb_extreme = set()

    CYCLE_MS   = 600    # タクト時間（ProBがボトルネック） [ms]
    PULSE_MS   = 50     # 信号パルス幅 [ms]
    total_rows = n_cycles * CYCLE_MS

    # 開始時刻（実機ログに合わせた書式）
    t0 = datetime(2026, 2, 6, 8, 0, 0)

    # ── Bool列（実機の "ProgramBlock.VariableName" 形式）─────────
    bool_cols = [
        'Assembly.ProA_CycleStart',
        'Assembly.ProA_LoadReq',
        'Assembly.ProA_SliderFwd',
        'Assembly.ProA_PartDetect',
        'Assembly.ProA_ClampON',
        'Assembly.ProB_CycleStart',
        'Assembly.ProB_PressDown',
        'Assembly.ProB_PressContact',
        'Assembly.ProB_PressComplete',
        'Assembly.ProB_PressUp',
        'Assembly.ProC_CycleStart',
        'Assembly.ProC_CamTrigger',
        'Assembly.ProC_InspectDone',
        'Assembly.ProC_GateOpen',
        'Assembly.ProC_EjectDone',
    ]
    data = {c: np.zeros(total_rows, dtype=np.int8) for c in bool_cols}
    # 各工程「実行中」持続信号（range モード用）
    data['Assembly.ProA_Running'] = np.zeros(total_rows, dtype=np.int8)
    data['Assembly.ProB_Running'] = np.zeros(total_rows, dtype=np.int8)
    data['Assembly.ProC_Running'] = np.zeros(total_rows, dtype=np.int8)
    # 数値変数（Case文シミュレーション / numeric 条件モード用）
    data['Assembly.ProB_Step'] = np.zeros(total_rows, dtype=np.int8)
    data['Assembly.LineStep']  = np.zeros(total_rows, dtype=np.int8)
    # アナログ信号
    data['Assembly.ProA_ClampPressure'] = np.zeros(total_rows)
    data['Assembly.ProB_PressForce']    = np.zeros(total_rows)
    data['Assembly.ProC_InspectScore']  = np.zeros(total_rows)

    def pulse(col, start, dur=PULSE_MS):
        s = int(np.clip(start, 0, total_rows - 1))
        e = min(s + dur, total_rows)
        data[col][s:e] = 1

    def fill_range(col, start, end):
        """start から end まで持続 ON（前の信号の終わりが次の信号の始まり）"""
        s = int(np.clip(start, 0, total_rows - 1))
        e = int(np.clip(end,   0, total_rows))
        if e > s:
            data[col][s:e] = 1

    for c in range(n_cycles):
        base = c * CYCLE_MS
        cyc  = c + 1

        # ── Assembly.ProA (供給工程) — 0〜450ms ──────────────────
        # ProA, ProB, ProC は全て base (t=0) から同時にスタート（並行動作）
        pa_base = base
        pulse('Assembly.ProA_CycleStart', pa_base)

        pa_extra  = 80 if cyc in anomaly_pa_slow else 0
        pa_load   = pa_base + int(rng.normal(50,  5))
        pa_slider = pa_base + int(rng.normal(180, 15))
        pa_detect = pa_base + int(rng.normal(320, 18)) + pa_extra
        pa_clamp  = pa_base + int(rng.normal(400, 18)) + pa_extra

        pa_load   = max(pa_base + 10,   pa_load)
        pa_slider = max(pa_load  + 20,  pa_slider)
        pa_detect = max(pa_slider + 20, pa_detect)
        pa_clamp  = max(pa_detect + 20, pa_clamp)

        # 各サブステップは「前が終わったら次が始まる」持続ON範囲
        pa_end = min(pa_clamp + PULSE_MS, base + CYCLE_MS)  # ProA 完了時刻
        fill_range('Assembly.ProA_LoadReq',    pa_load,   pa_slider)
        fill_range('Assembly.ProA_SliderFwd',  pa_slider, pa_detect)
        fill_range('Assembly.ProA_PartDetect', pa_detect, pa_clamp)
        fill_range('Assembly.ProA_ClampON',    pa_clamp,  pa_end)

        # ClampPressure: 0 → 8MPa → 0（台形）
        for t in range(pa_clamp, min(pa_clamp + 150, base + CYCLE_MS)):
            if t >= total_rows: break
            p = (t - pa_clamp) / 150.0
            if   p < 0.25: v = 8.0 * p / 0.25
            elif p < 0.75: v = 8.0
            else:          v = 8.0 * (1.0 - (p - 0.75) / 0.25)
            data['Assembly.ProA_ClampPressure'][t] = max(0.0, v + rng.normal(0, 0.12))

        # ── Assembly.ProB (圧入工程) — 0〜530ms ── ← ボトルネック
        pb_base = base   # ProA と同時スタート（別ワークを処理）
        pulse('Assembly.ProB_CycleStart', pb_base)

        pb_extra   = 60  if cyc in anomaly_pb_slow    else 0
        pb_extreme = 150 if cyc in anomaly_pb_extreme else 0

        pb_down     = pb_base + int(rng.normal(60,  8))
        pb_contact  = pb_base + int(rng.normal(200, 20))
        pb_complete = pb_base + int(rng.normal(400, 25)) + pb_extra + pb_extreme
        pb_up       = pb_base + int(rng.normal(480, 15)) + pb_extra + pb_extreme

        pb_down     = max(pb_base + 10,     pb_down)
        pb_contact  = max(pb_down  + 20,    pb_contact)
        pb_complete = max(pb_contact + 20,  pb_complete)
        pb_up       = max(pb_complete + 20, pb_up)

        pb_end = min(pb_up + PULSE_MS, base + CYCLE_MS)  # ProB 完了時刻
        fill_range('Assembly.ProB_PressDown',     pb_down,     pb_contact)
        fill_range('Assembly.ProB_PressContact',  pb_contact,  pb_complete)
        fill_range('Assembly.ProB_PressComplete', pb_complete, pb_up)
        fill_range('Assembly.ProB_PressUp',       pb_up,       pb_end)

        peak_f = 25.0 + (1.5 if (cyc in anomaly_pb_slow or cyc in anomaly_pb_extreme) else 0.0)
        span   = max(1, pb_complete - pb_contact)
        for t in range(pb_contact, min(pb_up + 30, base + CYCLE_MS)):
            if t >= total_rows: break
            p = (t - pb_contact) / span
            if   p < 0.20: v = peak_f * p / 0.20
            elif p < 0.85: v = peak_f
            else:          v = peak_f * max(0.0, 1.0 - (p - 0.85) / 0.15) * 0.5
            data['Assembly.ProB_PressForce'][t] = max(0.0, v + rng.normal(0, 0.25))

        # ── Assembly.ProC (検査排出工程) — 0〜310ms ─────────────
        pc_base = base   # ProA・ProB と同時スタート（別ワークを処理）
        pulse('Assembly.ProC_CycleStart', pc_base)

        pc_extra = 50 if cyc in anomaly_pc_slow else 0
        is_ng    = cyc in anomaly_pc_slow

        pc_cam     = pc_base + int(rng.normal(30,  3))
        pc_inspect = pc_base + int(rng.normal(120, 12)) + pc_extra
        pc_gate    = pc_base + int(rng.normal(175,  8)) + pc_extra
        pc_eject   = pc_base + int(rng.normal(260, 12)) + pc_extra

        pc_cam     = max(pc_base + 10,    pc_cam)
        pc_inspect = max(pc_cam   + 20,   pc_inspect)
        pc_gate    = max(pc_inspect + 10, pc_gate)
        pc_eject   = max(pc_gate   + 20,  pc_eject)

        pc_end = min(pc_eject + PULSE_MS, base + CYCLE_MS)  # ProC 完了時刻
        fill_range('Assembly.ProC_CamTrigger',  pc_cam,     pc_inspect)
        fill_range('Assembly.ProC_InspectDone', pc_inspect, pc_gate)
        fill_range('Assembly.ProC_GateOpen',    pc_gate,    pc_eject)
        fill_range('Assembly.ProC_EjectDone',   pc_eject,   pc_end)

        score_base = 58.0 if is_ng else 94.0
        score_std  =  4.0 if is_ng else  2.0
        for t in range(pc_cam, min(pc_inspect + 30, base + CYCLE_MS)):
            if t >= total_rows: break
            data['Assembly.ProC_InspectScore'][t] = float(
                np.clip(rng.normal(score_base, score_std), 0, 100))

        # ── Running 信号 & Step カウンタ（全工程タイミング確定後）─────
        def _c(v): return int(np.clip(v, 0, total_rows))

        # 各工程「実行中」持続信号（range モードのデモ用）
        data['Assembly.ProA_Running'][_c(base):_c(pa_end)] = 1
        data['Assembly.ProB_Running'][_c(base):_c(pb_end)] = 1
        data['Assembly.ProC_Running'][_c(base):_c(pc_end)] = 1

        # Assembly.ProB_Step: ProB 圧入工程の Case 文ステップ番号
        # 0=待機, 1=下降中, 2=接触待ち, 3=圧入中, 4=後退中
        data['Assembly.ProB_Step'][_c(base)       : _c(pb_down)    ] = 1
        data['Assembly.ProB_Step'][_c(pb_down)    : _c(pb_contact) ] = 2
        data['Assembly.ProB_Step'][_c(pb_contact) : _c(pb_complete)] = 3
        data['Assembly.ProB_Step'][_c(pb_complete): _c(pb_end)     ] = 4

        # Assembly.LineStep: 何工程が動いているかを表すライン状態
        # 工程完了時刻を昇順に並べ、段階的に減少
        ends = sorted([pa_end, pb_end, pc_end])
        # 全3工程が動いている区間（0 〜 最初の完了まで）
        data['Assembly.LineStep'][_c(base)    : _c(ends[0])] = 3
        # 2工程が動いている区間
        data['Assembly.LineStep'][_c(ends[0]) : _c(ends[1])] = 2
        # 1工程のみ（ボトルネックが動いている区間）
        data['Assembly.LineStep'][_c(ends[1]) : _c(ends[2])] = 1

    # ─── DataFrame構築 ─────────────────────────────────────────
    cur = t0
    ts_list = []
    for _ in range(total_rows):
        ts_list.append(cur.strftime('%Y-%m-%d %H:%M:%S.%f'))
        cur += timedelta(milliseconds=1)

    df = pd.DataFrame({'Date Time': ts_list})
    # 数値変数（先頭に配置）
    df['Assembly.ProB_Step'] = data['Assembly.ProB_Step'].astype(int)
    df['Assembly.LineStep']  = data['Assembly.LineStep'].astype(int)
    # Bool パルス信号
    for c in bool_cols:
        df[c] = data[c].astype(int)
    # 工程「実行中」持続信号
    df['Assembly.ProA_Running'] = data['Assembly.ProA_Running'].astype(int)
    df['Assembly.ProB_Running'] = data['Assembly.ProB_Running'].astype(int)
    df['Assembly.ProC_Running'] = data['Assembly.ProC_Running'].astype(int)
    # アナログ信号
    df['Assembly.ProA_ClampPressure'] = np.round(data['Assembly.ProA_ClampPressure'], 3)
    df['Assembly.ProB_PressForce']    = np.round(data['Assembly.ProB_PressForce'],    3)
    df['Assembly.ProC_InspectScore']  = np.round(data['Assembly.ProC_InspectScore'],  2)

    df.to_csv(filename, index=False)
    print(f"OK {filename}  ({len(df):,} rows, {n_cycles} cycles, takt={CYCLE_MS}ms)")


# ─── 通常ファイル ──────────────────────────────────────────────
generate(
    'sample_playback.csv',
    n_cycles=30, seed=42,
    anomaly_pa_slow={10, 23},
    anomaly_pb_slow={7, 14, 21},
    anomaly_pc_slow={18},
)

# ─── 異常ファイル（機械劣化シナリオ） ─────────────────────────
generate(
    'sample_playback_ng.csv',
    n_cycles=30, seed=123,
    anomaly_pa_slow={5, 12, 17, 20, 25},
    anomaly_pb_slow={3, 7, 10, 14, 17, 20, 23, 26, 29},
    anomaly_pb_extreme={25},
    anomaly_pc_slow={6, 11, 15, 19, 22, 28},
)

# ─── 検証 ──────────────────────────────────────────────────────
CYCLE_MS = 600
for fname in ['sample_playback.csv', 'sample_playback_ng.csv']:
    df = pd.read_csv(fname)
    print(f"\n--- {fname} ---")
    print("columns:", df.columns.tolist()[:6], "...")
    print("shape  :", df.shape)
    print("cyc | ProA_PartDetect | ProB_PressComplete | ProC_InspectDone | ProB_Step3(圧入)")
    for c in range(min(5, 30)):
        base = c * CYCLE_MS
        def rise(col, b=base):
            s = df[col].iloc[b:b+CYCLE_MS]
            rs = s[s.diff().fillna(0) > 0].index
            return (int(rs[0]) - b) if len(rs) else -1
        def step3_dur(b=base):
            s = df['Assembly.ProB_Step'].iloc[b:b+CYCLE_MS]
            on = s[s == 3]
            return len(on) if len(on) else 0
        print(f"  {c+1:2d} | {rise('Assembly.ProA_PartDetect'):4d}ms"
              f" | {rise('Assembly.ProB_PressComplete'):4d}ms"
              f" | {rise('Assembly.ProC_InspectDone'):4d}ms"
              f" | {step3_dur():4d}ms")
