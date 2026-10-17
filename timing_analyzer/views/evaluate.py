"""
views/evaluate.py — 最新データの評価（段階 2・3 で実装予定の仮ページ）
"""
from core.ui import planned_page

planned_page(
    "最新データの評価",
    "比較データを基準値と比べ、遅れやばらつきが出ているステップを外れの大きい順に示します。",
    will_have=['工程ごとの状態（正常・注意・異常）', '基準値から外れたステップの一覧と、ステップ詳細への掘り下げ'],
    legacy_where="工程タイミング ＞ 監視",
    legacy_kwargs=dict(view_mode='👁️ 監視', tab='🏭 工程タイミング'),
    eyebrow="点検",
)
