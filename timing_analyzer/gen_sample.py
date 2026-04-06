"""
gen_sample.py  — サンプルCSV生成スクリプト
実機 APB Variable Logger の出力フォーマットに準拠:
  - 列名: "Date Time"
  - タイムスタンプ: "2026-02-06 13:26:00.088848"（マイクロ秒）
  - 変数名: "工程名.変数名" 形式（日本語）
  - Bool値: 0 / 1（整数）

【並行動作3工程ライン】 ← タクトごとに全工程が同時に別ワークを処理
  供給工程  : ピッキング・クランプ  〜450ms
  圧入工程  : プレスフィッティング  〜530ms  ← ボトルネック
  検査排出工程: カメラ検査・排出     〜310ms
  タクト時間: 600ms（圧入工程に合わせた最小タクト）

数値変数（Case文シミュレーション）:
  圧入工程.ステップ番号 : 圧入工程内部ステップ（0=待機, 1〜4=各フェーズ）
  ライン.稼働工程数     : 動いている工程の数（3→2→1 と減少）
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

    CYCLE_MS   = 600    # タクト時間（圧入工程がボトルネック） [ms]
    PULSE_MS   = 50     # 信号パルス幅 [ms]
    total_rows = n_cycles * CYCLE_MS

    # 開始時刻（実機ログに合わせた書式）
    t0 = datetime(2026, 2, 6, 8, 0, 0)

    # ── Bool列 ───────────────────────────────────────────────────
    bool_cols = [
        '供給工程.サイクル開始',
        '供給工程.投入要求',
        '供給工程.スライダ前進',
        '供給工程.部品検出',
        '供給工程.クランプON',
        '圧入工程.サイクル開始',
        '圧入工程.プレス下降',
        '圧入工程.プレス接触',
        '圧入工程.圧入完了',
        '圧入工程.プレス上昇',
        '検査排出工程.サイクル開始',
        '検査排出工程.カメラトリガ',
        '検査排出工程.検査完了',
        '検査排出工程.ゲート開',
        '検査排出工程.排出完了',
    ]
    data = {c: np.zeros(total_rows, dtype=np.int8) for c in bool_cols}
    # 各工程「実行中」持続信号（range モード用）
    data['供給工程.実行中']     = np.zeros(total_rows, dtype=np.int8)
    data['圧入工程.実行中']     = np.zeros(total_rows, dtype=np.int8)
    data['検査排出工程.実行中'] = np.zeros(total_rows, dtype=np.int8)
    # 数値変数（Case文シミュレーション / numeric 条件モード用）
    data['圧入工程.ステップ番号'] = np.zeros(total_rows, dtype=np.int8)
    data['ライン.稼働工程数']     = np.zeros(total_rows, dtype=np.int8)
    # アナログ信号（既存）
    data['供給工程.クランプ圧力[MPa]'] = np.zeros(total_rows)
    data['圧入工程.プレス力[kN]']      = np.zeros(total_rows)
    data['検査排出工程.検査スコア']    = np.zeros(total_rows)
    # アナログ信号（新規：波形監視用）
    data['供給工程.スライダ変位[mm]']    = np.zeros(total_rows)
    data['圧入工程.プレス変位[mm]']      = np.zeros(total_rows)
    data['圧入工程.モータトルク[N·m]']  = np.zeros(total_rows)
    data['検査排出工程.照度センサ[lx]']  = np.zeros(total_rows)

    def pulse(col, start, dur=PULSE_MS):
        s = int(np.clip(start, 0, total_rows - 1))
        e = min(s + dur, total_rows)
        data[col][s:e] = 1

    def fill_range(col, start, end):
        s = int(np.clip(start, 0, total_rows - 1))
        e = int(np.clip(end,   0, total_rows))
        if e > s:
            data[col][s:e] = 1

    for c in range(n_cycles):
        base = c * CYCLE_MS
        cyc  = c + 1

        # ── 供給工程 — 0〜450ms ──────────────────────────────────
        pa_base = base
        pulse('供給工程.サイクル開始', pa_base)

        pa_extra  = 80 if cyc in anomaly_pa_slow else 0
        pa_load   = pa_base + int(rng.normal(50,  5))
        pa_slider = pa_base + int(rng.normal(180, 15))
        pa_detect = pa_base + int(rng.normal(320, 18)) + pa_extra
        pa_clamp  = pa_base + int(rng.normal(400, 18)) + pa_extra

        pa_load   = max(pa_base + 10,   pa_load)
        pa_slider = max(pa_load  + 20,  pa_slider)
        pa_detect = max(pa_slider + 20, pa_detect)
        pa_clamp  = max(pa_detect + 20, pa_clamp)

        pa_end = min(pa_clamp + PULSE_MS, base + CYCLE_MS)
        fill_range('供給工程.投入要求',   pa_load,   pa_slider)
        fill_range('供給工程.スライダ前進', pa_slider, pa_detect)
        fill_range('供給工程.部品検出',   pa_detect, pa_clamp)
        fill_range('供給工程.クランプON', pa_clamp,  pa_end)

        # クランプ圧力: 0 → 8MPa → 0（台形）
        for t in range(pa_clamp, min(pa_clamp + 150, base + CYCLE_MS)):
            if t >= total_rows: break
            p = (t - pa_clamp) / 150.0
            if   p < 0.25: v = 8.0 * p / 0.25
            elif p < 0.75: v = 8.0
            else:          v = 8.0 * (1.0 - (p - 0.75) / 0.25)
            data['供給工程.クランプ圧力[MPa]'][t] = max(0.0, v + rng.normal(0, 0.12))

        # スライダ変位: 0 → 120mm（前進中に直線増加、前進後は120mm保持）
        slider_span = max(1, pa_detect - pa_slider)
        for t in range(pa_slider, min(pa_end, base + CYCLE_MS)):
            if t >= total_rows: break
            progress = min(1.0, (t - pa_slider) / slider_span)
            data['供給工程.スライダ変位[mm]'][t] = 120.0 * progress + rng.normal(0, 0.3)

        # ── 圧入工程 — 0〜530ms ← ボトルネック ──────────────────
        pb_base = base
        pulse('圧入工程.サイクル開始', pb_base)

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

        pb_end = min(pb_up + PULSE_MS, base + CYCLE_MS)
        fill_range('圧入工程.プレス下降', pb_down,     pb_contact)
        fill_range('圧入工程.プレス接触', pb_contact,  pb_complete)
        fill_range('圧入工程.圧入完了',  pb_complete, pb_up)
        fill_range('圧入工程.プレス上昇', pb_up,       pb_end)

        # ── 圧入 物理モデル ──────────────────────────────────────────
        # 変位モデル:
        #   アプローチ区間 (pb_down → pb_contact): 0 → CONTACT_DISP [mm]（空走）
        #   圧入区間      (pb_contact → pb_complete): CONTACT_DISP → MAX_DISP [mm]
        #   上昇区間      (pb_complete → pb_up+30): MAX_DISP → 0 mm（急速後退）
        # 力モデル（変位から導出）:
        #   penetration = disp - CONTACT_DISP（≥0 のみ有効）
        #   F = k × penetration^FORCE_EXP  ← 干渉嵌め特有の超線形特性
        CONTACT_DISP = 8.0   # 接触時変位 [mm]（アプローチ空走量）
        PRESS_DEPTH  = 32.0  # 圧入深さ（接触後の追加変位） [mm]
        FORCE_EXP    = 1.3   # 力-変位の非線形指数
        TORQUE_RATIO = 0.50  # 力→トルク換算係数 [N·m/kN]

        is_anom = cyc in anomaly_pb_slow or cyc in anomaly_pb_extreme
        is_ext  = cyc in anomaly_pb_extreme
        peak_f  = 27.5 if is_anom else 25.0
        disp_peak = CONTACT_DISP + PRESS_DEPTH + (3.0 if is_ext else 0.0)
        actual_depth = disp_peak - CONTACT_DISP
        k_cyc = peak_f / (actual_depth ** FORCE_EXP)

        approach_span = max(1, pb_contact - pb_down)
        press_span    = max(1, pb_complete - pb_contact)
        retract_span  = max(1, pb_up - pb_complete)

        for t in range(pb_down, min(pb_up + 30, base + CYCLE_MS)):
            if t >= total_rows: break
            if t < pb_contact:
                # アプローチ: 変位 0 → CONTACT_DISP、力≈0（接触前は無負荷）
                p     = (t - pb_down) / approach_span
                disp  = CONTACT_DISP * p
                force = 0.0
            elif t < pb_complete:
                # 圧入: 変位 CONTACT_DISP → disp_peak、力は penetration^FORCE_EXP
                p     = (t - pb_contact) / press_span
                disp  = CONTACT_DISP + actual_depth * p
                force = k_cyc * (max(0.0, disp - CONTACT_DISP) ** FORCE_EXP)
            else:
                # 上昇: 変位 disp_peak → 0（力は急速に抜ける）
                p     = (t - pb_complete) / retract_span
                disp  = disp_peak * max(0.0, 1.0 - p)
                pen   = max(0.0, disp - CONTACT_DISP)
                force = k_cyc * (pen ** FORCE_EXP) * max(0.0, 1.0 - p / 0.3)

            data['圧入工程.プレス変位[mm]'][t] = max(0.0, disp + rng.normal(0, 0.15))
            data['圧入工程.プレス力[kN]'][t]   = max(0.0, force + rng.normal(0, 0.25))
            # モータトルク: 力と正比例（+ 小ノイズ）
            data['圧入工程.モータトルク[N·m]'][t] = max(
                0.0, force * TORQUE_RATIO + rng.normal(0, 0.18))

        # ── 検査排出工程 — 0〜310ms ──────────────────────────────
        pc_base = base
        pulse('検査排出工程.サイクル開始', pc_base)

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

        pc_end = min(pc_eject + PULSE_MS, base + CYCLE_MS)
        fill_range('検査排出工程.カメラトリガ', pc_cam,     pc_inspect)
        fill_range('検査排出工程.検査完了',     pc_inspect, pc_gate)
        fill_range('検査排出工程.ゲート開',     pc_gate,    pc_eject)
        fill_range('検査排出工程.排出完了',     pc_eject,   pc_end)

        score_base = 58.0 if is_ng else 94.0
        score_std  =  4.0 if is_ng else  2.0
        for t in range(pc_cam, min(pc_inspect + 30, base + CYCLE_MS)):
            if t >= total_rows: break
            data['検査排出工程.検査スコア'][t] = float(
                np.clip(rng.normal(score_base, score_std), 0, 100))

        # 照度センサ: カメラトリガ〜検査完了の間に一定照度（スパイク立ち上がり）
        illum_peak = 850.0 if not is_ng else 820.0
        illum_span = max(1, pc_inspect - pc_cam)
        for t in range(pc_cam, min(pc_inspect + 10, base + CYCLE_MS)):
            if t >= total_rows: break
            p = (t - pc_cam) / illum_span
            if   p < 0.05: v = illum_peak * p / 0.05
            elif p < 0.95: v = illum_peak
            else:          v = illum_peak * (1.0 - (p - 0.95) / 0.05)
            data['検査排出工程.照度センサ[lx]'][t] = max(0.0, v + rng.normal(0, 5.0))

        # ── 実行中信号 & ステップカウンタ ────────────────────────
        def _c(v): return int(np.clip(v, 0, total_rows))

        data['供給工程.実行中'][_c(base):_c(pa_end)] = 1
        data['圧入工程.実行中'][_c(base):_c(pb_end)] = 1
        data['検査排出工程.実行中'][_c(base):_c(pc_end)] = 1

        # 圧入工程.ステップ番号: 0=待機, 1=下降中, 2=接触待ち, 3=圧入中, 4=後退中
        data['圧入工程.ステップ番号'][_c(base)       : _c(pb_down)    ] = 1
        data['圧入工程.ステップ番号'][_c(pb_down)    : _c(pb_contact) ] = 2
        data['圧入工程.ステップ番号'][_c(pb_contact) : _c(pb_complete)] = 3
        data['圧入工程.ステップ番号'][_c(pb_complete): _c(pb_end)     ] = 4

        # ライン.稼働工程数: 工程完了時刻を昇順に並べ段階的に減少
        ends = sorted([pa_end, pb_end, pc_end])
        data['ライン.稼働工程数'][_c(base)    : _c(ends[0])] = 3
        data['ライン.稼働工程数'][_c(ends[0]) : _c(ends[1])] = 2
        data['ライン.稼働工程数'][_c(ends[1]) : _c(ends[2])] = 1

    # ── DataFrame構築 ─────────────────────────────────────────
    cur = t0
    ts_list = []
    for _ in range(total_rows):
        ts_list.append(cur.strftime('%Y-%m-%d %H:%M:%S.%f'))
        cur += timedelta(milliseconds=1)

    df = pd.DataFrame({'Date Time': ts_list})
    # 数値変数（先頭に配置）
    df['圧入工程.ステップ番号'] = data['圧入工程.ステップ番号'].astype(int)
    df['ライン.稼働工程数']     = data['ライン.稼働工程数'].astype(int)
    # Boolパルス信号
    for c in bool_cols:
        df[c] = data[c].astype(int)
    # 工程「実行中」持続信号
    df['供給工程.実行中']     = data['供給工程.実行中'].astype(int)
    df['圧入工程.実行中']     = data['圧入工程.実行中'].astype(int)
    df['検査排出工程.実行中'] = data['検査排出工程.実行中'].astype(int)
    # アナログ信号
    df['供給工程.クランプ圧力[MPa]']   = np.round(data['供給工程.クランプ圧力[MPa]'],   3)
    df['供給工程.スライダ変位[mm]']    = np.round(data['供給工程.スライダ変位[mm]'],    2)
    df['圧入工程.プレス力[kN]']        = np.round(data['圧入工程.プレス力[kN]'],        3)
    df['圧入工程.プレス変位[mm]']      = np.round(data['圧入工程.プレス変位[mm]'],      2)
    df['圧入工程.モータトルク[N·m]']  = np.round(data['圧入工程.モータトルク[N·m]'],   3)
    df['検査排出工程.検査スコア']      = np.round(data['検査排出工程.検査スコア'],       2)
    df['検査排出工程.照度センサ[lx]']  = np.round(data['検査排出工程.照度センサ[lx]'],  1)

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
    print("cyc | 部品検出 | 圧入完了 | 検査完了 | ステップ3(圧入)")
    for c in range(min(5, 30)):
        base = c * CYCLE_MS
        def rise(col, b=base):
            s = df[col].iloc[b:b+CYCLE_MS]
            rs = s[s.diff().fillna(0) > 0].index
            return (int(rs[0]) - b) if len(rs) else -1
        def step3_dur(b=base):
            s = df['圧入工程.ステップ番号'].iloc[b:b+CYCLE_MS]
            on = s[s == 3]
            return len(on) if len(on) else 0
        print(f"  {c+1:2d} | {rise('供給工程.部品検出'):4d}ms"
              f" | {rise('圧入工程.圧入完了'):4d}ms"
              f" | {rise('検査排出工程.検査完了'):4d}ms"
              f" | {step3_dur():4d}ms")
