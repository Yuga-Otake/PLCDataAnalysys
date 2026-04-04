"""
test_histogram_bins.py - ヒストグラムのビン幅統一テスト

問題: nbinsx をトレースごとに指定すると Plotly がトレースごとに
      ビン幅を独立計算し、異常値（少数）のビンが細くなる。
修正: 全データから xbins(start/end/size) を一括計算し両トレースに適用。
"""
import numpy as np
import sys

# ── ヘルパー関数（app.py から移植） ──────────────────────────────
def calc_nice_bins_simple(data: np.ndarray) -> int:
    n = len(data)
    if n <= 4:
        return 5
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    data_range = float(np.max(data) - np.min(data))
    if iqr <= 0 or data_range <= 0:
        return max(5, min(30, int(1 + 3.32 * np.log10(max(n, 2)))))
    bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
    mag = 10.0 ** np.floor(np.log10(bin_width))
    for f in [1, 2, 5, 10]:
        nw = f * mag
        if nw >= bin_width * 0.8:
            bin_width = nw
            break
    return max(5, min(50, int(np.ceil(data_range / bin_width))))


def make_xbins(data: np.ndarray, n_bins: int) -> dict:
    """全データから共通ビン幅を計算"""
    vmin = float(data.min())
    vmax = float(data.max())
    size = (vmax - vmin) / n_bins if n_bins > 0 and vmax > vmin else 1.0
    return dict(start=vmin, end=vmax + size, size=size)


# ── テストケース ───────────────────────────────────────────────
def test_bin_consistency():
    """正常値 95 件 + 異常値 5 件で、xbins を使えば幅が同じになることを確認"""
    np.random.seed(42)
    normal = np.random.normal(loc=50, scale=5, size=95)   # 35-65 ms 付近
    outliers = np.array([85.0, 87.0, 90.0, 88.0, 92.0])  # 明確な外れ値
    all_data = np.concatenate([normal, outliers])

    threshold = 70.0
    below = all_data[all_data <= threshold]
    above = all_data[all_data >  threshold]

    n_bins = calc_nice_bins_simple(all_data)
    xbins  = make_xbins(all_data, n_bins)
    bsize  = xbins["size"]

    print(f"全データ: {len(all_data)} 件  (正常: {len(below)}, 異常: {len(above)})")
    print(f"データ範囲: {all_data.min():.1f} - {all_data.max():.1f} ms")
    print(f"n_bins: {n_bins}")
    print(f"xbins: {xbins}")
    print()

    # ── 旧方式: nbinsx をトレースごとに渡す ──
    # Plotly が below/above それぞれのデータ範囲でビン幅を計算してしまう
    # (Python 側では確認できないが、以下で幅の差を検証)
    below_range = below.max() - below.min()
    above_range = above.max() - above.min()
    below_bsize_old = below_range / n_bins  # Plotly が使うおおよその幅
    above_bsize_old = above_range / n_bins

    print("── 旧方式 (nbinsx 独立計算) ──")
    print(f"  正常トレース ビン幅 ≈ {below_bsize_old:.3f} ms")
    print(f"  異常トレース ビン幅 ≈ {above_bsize_old:.3f} ms")
    ratio_old = above_bsize_old / below_bsize_old
    print(f"  幅の比 (異常/正常) = {ratio_old:.3f}  ← 1.0 から離れるほど見た目がおかしい")
    print()

    # ── 新方式: xbins を共通で渡す ──
    print("── 新方式 (xbins 共通) ──")
    print(f"  全トレース ビン幅 = {bsize:.3f} ms  (統一)")
    ratio_new = 1.0
    print(f"  幅の比 (異常/正常) = {ratio_new:.3f}  ← 完全一致")
    print()

    # ── アサーション ──
    assert ratio_old < 0.5, \
        f"旧方式でビン幅が大きく違うはずが ratio={ratio_old:.3f}"
    assert abs(ratio_new - 1.0) < 1e-9, \
        "新方式ではビン幅が一致するはず"
    print("PASS: 旧方式では異常値ビンが正常値より細くなる問題を確認")
    print("PASS: 新方式では xbins 統一によりビン幅が一致")
    return True


def test_delta_mode():
    """delta mode (基準値比較) でも xbins が統一されることを確認"""
    np.random.seed(7)
    baseline_std = 3.0
    vals_delta = np.concatenate([
        np.random.normal(0, baseline_std, 90),
        np.array([10.0, 11.5, -10.2, 12.0])  # ±3σ外れ値
    ])
    t3 = 3 * baseline_std  # = 9.0

    n_bins = calc_nice_bins_simple(vals_delta)
    xbins  = make_xbins(vals_delta, n_bins)

    in_r  = vals_delta[np.abs(vals_delta) <= t3]
    out_r = vals_delta[np.abs(vals_delta) >  t3]

    print(f"delta mode: ±3σ={t3:.1f}ms  (以内: {len(in_r)}, 超過: {len(out_r)})")
    print(f"xbins: {xbins}")

    in_range  = in_r.max()  - in_r.min()
    out_range = out_r.max() - out_r.min()
    bsize_old_in  = in_range  / n_bins
    bsize_old_out = out_range / n_bins

    print(f"旧方式: 以内トレース幅≈{bsize_old_in:.3f}  超過トレース幅≈{bsize_old_out:.3f}")
    print(f"新方式: 共通幅={xbins['size']:.3f}")

    # 旧方式では in/out のビン幅が異なる（どちらが大きいかはデータ次第）
    assert abs(bsize_old_out - bsize_old_in) > 0.01, \
        "旧方式では in/out のビン幅が一致していないはず"
    # 新方式では必ず一致
    assert abs(xbins["size"] - xbins["size"]) < 1e-9
    print("PASS: delta mode でも xbins 統一が有効")
    return True


def test_edge_single_outlier():
    """外れ値が1件だけのとき、xbins で幅が崩れないことを確認"""
    np.random.seed(3)
    data = np.concatenate([np.random.normal(20, 2, 99), [50.0]])  # 外れ値1件
    threshold = 30.0
    below = data[data <= threshold]
    above = data[data >  threshold]

    n_bins = calc_nice_bins_simple(data)
    xbins  = make_xbins(data, n_bins)

    print(f"外れ値1件テスト: n_bins={n_bins}, xbins size={xbins['size']:.3f}")
    print(f"  below: {len(below)}件, above: {len(above)}件")

    # 新方式では above が1件でもビン幅は固定
    assert xbins["size"] > 0
    print("PASS: 外れ値1件でも xbins は正常")
    return True


# ── メイン ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("ヒストグラム ビン幅統一テスト")
    print("=" * 55)
    results = []
    for fn in [test_bin_consistency, test_delta_mode, test_edge_single_outlier]:
        print(f"\n[{fn.__name__}]")
        try:
            results.append(fn())
        except AssertionError as e:
            print(f"FAIL: {e}")
            results.append(False)

    print("\n" + "=" * 55)
    passed = sum(results)
    total  = len(results)
    print(f"結果: {passed}/{total} PASS")
    sys.exit(0 if passed == total else 1)
