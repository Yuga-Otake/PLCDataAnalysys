"""
views/wave.py — 波形検査条件（段階 2・3 で実装予定の仮ページ）
"""
from core.ui import planned_page

planned_page(
    "波形検査条件",
    "アナログ値の波形から検出点や良品範囲を設定します。結果は評価と推移に表示されます。",
    will_have=['時間軸・XY グラフでの検出点の設定', '良品範囲の登録', '推移への送出の設定'],
    legacy_where="波形検査",
    legacy_kwargs=dict(tab='🔍 波形検査'),
    eyebrow="準備（詳細設定）",
)
