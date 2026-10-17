"""
views/trend.py — 推移（段階 2・3 で実装予定の仮ページ）
"""
from core.ui import planned_page

planned_page(
    "推移",
    "複数時期の比較データを並べ、いつから・どのステップが悪化したかを確認します。",
    will_have=['ステップ別の推移グラフ', 'Xbar-R 管理図', '波形検査の検出点の推移'],
    legacy_where="傾向解析",
    legacy_kwargs=dict(tab='📈 傾向解析'),
    eyebrow="点検",
)
