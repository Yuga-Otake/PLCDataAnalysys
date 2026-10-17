"""
views/capability.py — 工程能力（段階 2・3 で実装予定の仮ページ）
"""
from core.ui import planned_page

planned_page(
    "工程能力",
    "工程ごとの Cpk と、登録済みの良品条件を一覧します。",
    will_have=['工程横断の Cpk 一覧', '良品条件（基準値）の一覧'],
    legacy_where="工程タイミング ＞ 品質分析",
    legacy_kwargs=dict(view_mode='📊 品質分析', tab='🏭 工程タイミング'),
    eyebrow="点検",
)
