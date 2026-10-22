"""
core/tutorial.py — チュートリアルモード（画面を暗くして操作する場所を照らし、カーソルと吹き出しで案内する）

流れ：練習データ（基準 1 本・比較 4 本）をダウンロード → 自分で読み込み → 準備（工程・ステップ・基準値・波形検査・保存）
→ 点検（比較データ・評価・ステップ詳細・推移・工程能力）。各ステップは、利用者が実際に操作したことを
session_state から判定して自動で次へ進む（説明だけのステップは「次へ」）。迷ったときは「代わりにやる」。

表示のしくみ：
  - 吹き出しは Streamlit のコンテナ（key=tutbubble）。中のボタンは普通の st.button。CSS で画面に固定し、
    下の JavaScript が照らす場所の近くへ動かす
  - 暗転・スポットライト・カーソルは、components.html の中の JavaScript が親の画面に要素を足して描く
    （クリックは下の画面にそのまま届く。操作は本物の画面で行う）
"""
import io
import json

import streamlit as st
import streamlit.components.v1 as components

from core.state import pk
from core.tutorial_data import CMP_FILES, REF_FILE, tutorial_frames

REF_LABEL = REF_FILE[:-4]
CMP_LABELS = [f[:-4] for f in CMP_FILES]
FEED, PRESS = "供給工程", "圧入工程"

# 画面（url_path）→ メニューの名前
PAGES = {"": "ホーム", "data": "データ", "process": "工程とステップ", "baseline": "基準値", "wave": "波形検査条件",
         "evaluate": "最新データの評価", "trend": "推移", "capability": "工程能力", "step": "ステップ詳細",
         "tutorial": "チュートリアル"}
_PAGE_FILES = {"": "views/home.py", "data": "views/data.py", "process": "views/process.py",
               "baseline": "views/baseline.py", "wave": "views/wave.py", "evaluate": "views/evaluate.py",
               "trend": "views/trend.py", "capability": "views/capability.py", "step": "views/step.py",
               "tutorial": "views/tutorial.py"}


# ── 状態の判定 ───────────────────────────────────────────────

def _ss():
    return st.session_state


def _store() -> dict:
    return _ss().get("csv_store", {})


def _ref_loaded() -> bool:
    return any(e.get("is_ref") and e.get("label") == REF_LABEL for e in _store().values())


def _n_cmp() -> int:
    return sum(1 for e in _store().values() if not e.get("is_ref") and e.get("label") in CMP_LABELS)


def _steps(p: str) -> int:
    return len(_ss().get(pk(p, "steps_list"), []) or [])


def _has_baseline(p: str) -> bool:
    return bool(_ss().get(pk(p, "baseline")))


def _n_wave_points() -> int:
    return sum(len(v or []) for k, v in _ss().items()
               if isinstance(k, str) and k.startswith("wvol___global_") and k.endswith("_t_det_list"))


def _sel(p: str) -> bool:
    return _ss().get("sel_process") == p


# ── 「代わりにやる」 ─────────────────────────────────────────

def _load(name: str, is_ref: bool):
    from analyzer import load_csv
    from core.data import add_csv, detect_bool_columns
    df = load_csv(io.BytesIO(tutorial_frames()[name]))
    add_csv(name, df, detect_bool_columns(df), name[:-4], is_ref, "tutorial")


def _auto_ref():
    _load(REF_FILE, True)


def _auto_cmp():
    for f in CMP_FILES:
        if not any(e.get("label") == f[:-4] for e in _store().values()):
            _load(f, False)


def _auto_processes():
    from core.profile import active_bool_vars, group_hypotheses
    from core.state import add_process, ref_entry
    _, ref = ref_entry()
    if not ref:
        return
    for g in group_hypotheses(ref["df"], tuple(active_bool_vars(ref))):
        if g["trigger"] and g["name"] not in _ss().get("processes", {}):
            add_process(g["name"], g["trigger"])
    _ss()["sel_process"] = FEED


def _select(p: str):
    def f():
        _ss()["sel_process"] = p
    return f


def _select_without_baseline():
    other = next((p for p in _ss().get("processes", {}) if not _has_baseline(p)), None)
    if other:
        _ss()["sel_process"] = other


def _open_detail():
    from core.evaluate import compare_keys_by_time
    keys = compare_keys_by_time()
    if keys:
        _ss()["sel_process"] = FEED
        _ss()["detail"] = {"proc": FEED, "step": "部品検出", "data": keys[-1]}
        _ss()["_tut_goto"] = "step"


# ── ステップ ─────────────────────────────────────────────────
# page: そのステップを行う画面（違う画面にいるときは、その画面へ戻るよう案内する）
# target: 照らす場所 {"sel": CSS, "text": 含む文字, "exact": 文字が完全一致, "label": 入力欄の名前}
# done: 終わったかの判定（None は「次へ」で進む説明ステップ）
# auto: 「代わりにやる」で行うこと（"goto:<url_path>" は画面の移動）

def _link(name: str) -> dict:
    return {"sel": '[data-testid="stSidebarNavLink"]', "text": name, "exact": True, "sidebar": True}


def _btn(text: str) -> dict:
    return {"sel": "button", "text": text}


def _md(text: str) -> dict:
    return {"sel": '[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"]', "text": text}


def _select_box(label: str) -> dict:
    return {"sel": '[data-testid="stSelectbox"]', "label": label}


def _goto(page: str, title: str, body: str, ch: str) -> dict:
    return {"ch": ch, "page": None, "title": title, "body": body, "target": _link(PAGES[page]),
            "done": lambda ctx, p=page: ctx["page"] == p, "auto": f"goto:{page}"}


STEPS = [
    # ── はじめに ──
    {"ch": "はじめに", "page": "tutorial", "title": "チュートリアルへようこそ",
     "body": "練習用のデータを使って、準備（基準を作る）から点検（変化を見つける）までを実際に操作します。"
             "暗くなった画面の中で、明るく照らされた場所を操作してください。操作すると自動で次へ進みます。"
             "迷ったときは「代わりにやる」、やめたいときは「終了」を押してください。",
     "target": None, "done": None},
    {"ch": "はじめに", "page": "tutorial", "title": "練習データをダウンロード",
     "body": "まず練習データ（ZIP）をダウンロードします。中には、設備を立ち上げたときの<b>基準データ</b>と、"
             "1・3・6・9 か月後の<b>比較データ</b>の CSV が入っています。",
     "target": _btn("練習データをダウンロード"), "done": lambda ctx: _ss().get("_tut_downloaded"),
     "next_label": "ダウンロード済み"},
    {"ch": "はじめに", "page": "tutorial", "title": "ZIP を展開しておきます",
     "body": "ダウンロードした ZIP を展開（右クリック →「すべて展開」など）してください。"
             "「APB_チュートリアル」フォルダに CSV が 5 つできます。この後、画面から 1 つずつ読み込みます。",
     "target": None, "done": None},
    # ── 準備 1 データ ──
    _goto("data", "「データ」画面を開く", "準備は「データ」から始めます。左のメニューの「データ」を押してください。", "準備 1 データ"),
    {"ch": "準備 1 データ", "page": "data", "title": "基準データを読み込む",
     "body": "ここに <b>基準_立ち上げ時.csv</b> をドラッグするか、「Upload」から選んでください。"
             "基準データは、設備が正常に動いていたときの記録です。",
     "target": {"sel": '[data-testid="stFileUploader"]', "text": "基準データを読み込む"},
     "done": lambda ctx: _ref_loaded(), "auto": _auto_ref},
    {"ch": "準備 1 データ", "page": "data", "title": "記録の質を自動でチェック",
     "body": "読み込むと、記録の抜けや、動いていない変数がないかを自動で調べます。"
             "「記録 良好」なら、そのまま工程の登録に進めます。",
     "target": _md("動いている変数"), "done": None},
    # ── 準備 2 工程とステップ ──
    _goto("process", "「工程とステップ」を開く", "次に、設備の工程と、時間を測る動作（ステップ）を決めます。", "準備 2 工程とステップ"),
    {"ch": "準備 2 工程とステップ", "page": "process", "title": "工程の提案を採用する",
     "body": "変数名と、周期的に立ち上がる信号から、<b>供給工程</b>と<b>圧入工程</b>が提案されています。"
             "「すべて作成」を押してください。",
     "target": _btn("すべて作成"), "done": lambda ctx: len(_ss().get("processes", {})) >= 2,
     "auto": _auto_processes},
    {"ch": "準備 2 工程とステップ", "page": "process", "title": "供給工程のステップを登録する",
     "body": "1 サイクルの中で変数が変化する順番を、ツールが仮説として並べています。"
             "違うものがあれば表で直せますが、今回はそのまま「ステップとして登録」を押してください。",
     "target": _btn("ステップとして登録"), "done": lambda ctx: _steps(FEED) > 0, "pre": _select(FEED)},
    {"ch": "準備 2 工程とステップ", "page": "process", "title": "圧入工程に切り替える",
     "body": "「工程」で<b>圧入工程</b>を選んでください。",
     "target": _select_box("工程"), "done": lambda ctx: _sel(PRESS) or _steps(PRESS) > 0, "auto": _select(PRESS)},
    {"ch": "準備 2 工程とステップ", "page": "process", "title": "圧入工程のステップを登録する",
     "body": "同じように「ステップとして登録」を押してください。",
     "target": _btn("ステップとして登録"), "done": lambda ctx: _steps(PRESS) > 0},
    {"ch": "準備 2 工程とステップ", "page": "process", "title": "タイミング図",
     "body": "各ステップがサイクル開始から何 ms 後に起きているかの図です。棒が動作中の期間、薄い部分がサイクルごとの幅です。"
             "これが、この設備の「正常なとき」の動きです。",
     "target": {"sel": '[data-testid="stPlotlyChart"]'}, "done": None},
    # ── 準備 3 基準値 ──
    _goto("baseline", "「基準値」を開く", "正常なときのタイミングを、点検の基準として記録します。", "準備 3 基準値"),
    {"ch": "準備 3 基準値", "page": "baseline", "title": "基準値を登録する",
     "body": "基準データから各ステップの平均とばらつきを計算しました。「この内容で登録」を押してください。",
     "target": _btn("この内容で登録"), "done": lambda ctx: any(_has_baseline(p) for p in (FEED, PRESS))},
    {"ch": "準備 3 基準値", "page": "baseline", "title": "もう一方の工程も登録する",
     "body": "「工程」でもう一方の工程を選び、同じように「この内容で登録」を押してください。",
     "target": _select_box("工程"), "done": lambda ctx: _has_baseline(FEED) and _has_baseline(PRESS),
     "auto": _select_without_baseline, "auto_label": "工程を切り替える"},
    # ── 準備 4 波形検査 ──
    {"ch": "準備 4 波形検査", "page": "baseline", "title": "アナログ波形も点検できます",
     "body": "圧力や力などのアナログ波形にも、ピークや立ち上がりの「検査点」を置けます。"
             "「波形検査の候補を見る」を押してください。",
     "target": _btn("波形検査の候補を見る"), "done": lambda ctx: ctx["page"] == "wave", "auto": "goto:wave"},
    {"ch": "準備 4 波形検査", "page": "wave", "title": "検査点の候補を採用する",
     "body": "波形の形から、検査点のタイプ（ピーク・立ち上がりなど）と設定値を提案しています。"
             "「検出点として登録」を押してください。",
     "target": _btn("検出点として登録"), "done": lambda ctx: _n_wave_points() > 0},
    {"ch": "準備 4 波形検査", "page": "wave", "title": "工程 → 検査波形 → 検査点",
     "body": "登録した検査点は、工程ごと・波形ごとに整理されます。図の ◇ が検査点の位置、下の帯が工程のステップです。"
             "⚙️ で波形を見ながら調整、🗑 で削除できます。",
     "target": _md("工程 → 検査波形 → 検査点"), "done": None},
    # ── 準備 5 保存 ──
    _goto("", "ホームに戻る", "ホームでは、準備の進み具合と次にやることが分かります。", "準備 5 保存"),
    {"ch": "準備 5 保存", "page": "", "title": "基準データと設定を保存する",
     "body": "準備の最後に、基準データと設定を 1 つのファイル（ZIP）に保存します。"
             "次からはこのファイルを読み込むだけで点検を始められます。",
     "target": _btn("基準データ＋設定を保存"), "done": lambda ctx: _ss().get("_project_saved"),
     "next_label": "保存した"},
    # ── 点検 1 比較データ ──
    _goto("data", "「データ」画面を開く", "ここからは点検です。点検したい時期のデータを読み込みます。", "点検 1 比較データ"),
    {"ch": "点検 1 比較データ", "page": "data", "title": "比較データを読み込む",
     "body": "<b>比較_1か月後</b>〜<b>比較_9か月後</b> の 4 つの CSV を、まとめてここにドラッグしてください"
             "（「Upload」から複数選ぶこともできます）。",
     "target": {"sel": '[data-testid="stFileUploader"]', "text": "比較データを追加"},
     "done": lambda ctx: _n_cmp() >= len(CMP_FILES), "auto": _auto_cmp},
    # ── 点検 2 評価 ──
    _goto("evaluate", "「最新データの評価」を開く", "いちばん新しい比較データ（9 か月後）を、基準値と比べます。", "点検 2 評価"),
    {"ch": "点検 2 評価", "page": "evaluate", "title": "工程ごとの状態",
     "body": "<b>供給工程</b>が「異常」になっています。どのステップがおかしいのか、下の表で見ていきます。",
     "target": _md("工程ごとの状態"), "done": None},
    {"ch": "点検 2 評価", "page": "evaluate", "title": "供給工程を選ぶ",
     "body": "「工程」で<b>供給工程</b>を選んでください。",
     "target": _select_box("工程"), "done": lambda ctx: _sel(FEED), "auto": _select(FEED)},
    {"ch": "点検 2 評価", "page": "evaluate", "title": "おかしいステップを開く",
     "body": "判定の表で、「異常」になっている<b>部品検出</b>の行をクリックしてください。ステップの詳細が開きます。",
     "target": {"sel": '[data-testid="stDataFrame"]'}, "done": lambda ctx: ctx["page"] == "step",
     "auto": _open_detail},
    {"ch": "点検 2 評価", "page": "step", "title": "ステップの詳細",
     "body": "基準（立ち上げ時）に比べて、部品検出の長さが<b>サイクルごとに大きくばらついて</b>います。"
             "ばらつきが増えると、平均が少し変わっただけでも許容幅の外に出るサイクルが増えます（赤が NG）。",
     "target": {"sel": '[data-testid="stPlotlyChart"]'}, "done": None},
    # ── 点検 3 推移 ──
    _goto("trend", "「推移」を開く", "いつから変わり始めたのかを、時期を並べて見ます。", "点検 3 推移"),
    {"ch": "点検 3 推移", "page": "trend", "title": "ばらつきの推移",
     "body": "「ステップの推移と管理図」では、部品検出の箱（ばらつき）が時期とともに大きくなり、"
             "右の管理図の S（標準偏差）も上がっていくのが分かります。じわじわ進む劣化の典型です。",
     "target": _md("箱はサイクルのばらつき"), "done": None},
    {"ch": "点検 3 推移", "page": "trend", "title": "圧入工程に切り替える",
     "body": "「工程」で<b>圧入工程</b>を選んでください。波形検査点の推移を見ます。",
     "target": _select_box("工程"), "done": lambda ctx: _sel(PRESS), "auto": _select(PRESS)},
    {"ch": "点検 3 推移", "page": "trend", "title": "波形のピークの変化",
     "body": "下の「波形検査点の推移」で、<b>プレス力のピーク</b>の値が時期とともに上がっています。"
             "タイミングの表には出ない、押し込む力の変化を見つけられました。",
     "target": _md("の検出点の時期ごとの平均"), "done": None},
    # ── 点検 4 工程能力 ──
    _goto("capability", "「工程能力」を開く", "規格（許容幅）に対して、どれだけ余裕があるかを見ます。", "点検 4 工程能力"),
    {"ch": "点検 4 工程能力", "page": "capability", "title": "余裕（Cpk）が減っている",
     "body": "表は Cpk（許容幅に対する余裕の大きさ）の小さい順です。ばらつきが増えた供給工程の<b>部品検出</b>は、"
             "1.00 も下回る「不足」です（1.33 以上で十分）。"
             "このまま進むと不良や停止につながるので、センサーやワークの位置決めを点検する目安になります。",
     "target": {"sel": '[data-testid="stDataFrame"]'}, "done": None},
    {"ch": "完了", "page": None, "title": "チュートリアル完了",
     "body": "お疲れさまでした。準備（基準データ → 工程とステップ → 基準値 → 波形検査 → 保存）と、"
             "点検（比較データ → 評価 → 推移 → 工程能力）を一通り体験しました。"
             "実際の設備では、自分の APB の CSV で同じ手順を進めてください。",
     "target": None, "done": None, "last": True},
]
CHAPTERS = list(dict.fromkeys(s["ch"] for s in STEPS))


# ── 開始・終了 ───────────────────────────────────────────────

def active() -> bool:
    return bool(_ss().get("tut_on"))


def start():
    """今のデータ・工程・波形検査の設定を片付けてから始める"""
    from core.data import set_use_sample
    ss = _ss()
    set_use_sample(False)
    ss["csv_store"] = {}
    ss["ref_csv_key"] = None
    for k in [k for k in list(ss.keys()) if isinstance(k, str) and (
            k.startswith("P_") or k.startswith("wvol_") or k.startswith("_wave_cands_")
            or k in ("sel_process", "detail", "active_csv", "_project_saved", "wi_sel___global",
                     "_data_ref_loaded", "_data_cmp_seen", "_data_cfg_loaded"))]:
        del ss[k]
    ss["processes"] = {}
    ss["tut_on"], ss["tut_i"] = True, 0


def stop():
    _ss()["tut_on"] = False


def _next():
    _ss()["tut_i"] = min(_ss().get("tut_i", 0) + 1, len(STEPS) - 1)


def _back():
    _ss()["tut_i"] = max(_ss().get("tut_i", 0) - 1, 0)


def _run_auto(step: dict):
    a = step.get("auto")
    if isinstance(a, str) and a.startswith("goto:"):
        _ss()["_tut_goto"] = a[5:]
    elif callable(a):
        a()


def progress() -> tuple:
    return _ss().get("tut_i", 0), len(STEPS)


# ── 表示 ─────────────────────────────────────────────────────

_CSS = """
<style>
[class*="st-key-tutbubble"] {
  position: fixed !important; z-index: 1000004; width: min(400px, calc(100vw - 24px));
  right: 24px; bottom: 24px; background: var(--apb-card, #fff); color: var(--apb-ink, #1f2328);
  border: 1px solid var(--apb-accent, #1f6feb); border-radius: 10px; padding: 14px 16px 10px;
  box-shadow: 0 12px 32px rgba(0,0,0,.35); transition: left .35s ease, top .35s ease;
}
[class*="st-key-tutbubble"] .tut-ch { font-size: 11px; color: var(--apb-label, #6e7781); letter-spacing: .04em; }
[class*="st-key-tutbubble"] .tut-title { font-size: 16px; font-weight: 700; margin: 2px 0 4px; }
[class*="st-key-tutbubble"] .tut-body { font-size: 13.5px; line-height: 1.7; }
[class*="st-key-tutbubble"] .tut-bar { height: 3px; background: var(--apb-hair, #d0d7de); border-radius: 2px; margin-top: 8px; }
[class*="st-key-tutbubble"] .tut-bar > i { display: block; height: 3px; background: var(--apb-accent, #1f6feb); border-radius: 2px; }
[class*="st-key-tutbubble"] button p { font-size: 12.5px; }
[class*="st-key-tutbubble"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 6px !important; }
[class*="st-key-tutbubble"] [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; }
[class*="st-key-tutbubble"] button { padding-left: 6px; padding-right: 6px; min-height: 34px; }
</style>
"""

_JS = r"""
<script>
(function () {
  const P = window.parent, D = P.document, cfg = __CFG__;
  const old = P.__apbTut; if (old && old.timer) P.clearInterval(old.timer);
  const ids = ["apbTutShade", "apbTutHole", "apbTutCursor", "apbTutRing"];
  if (!cfg.on) { ids.forEach(i => { const e = D.getElementById(i); if (e) e.remove(); }); P.__apbTut = null; return; }
  const st = P.__apbTut = Object.assign(P.__apbTut || {}, { step: cfg.step });
  function el(id, css, html) {
    let e = D.getElementById(id);
    if (!e) { e = D.createElement("div"); e.id = id; D.body.appendChild(e); }
    e.style.cssText = css; if (html !== undefined) e.innerHTML = html; return e;
  }
  const shade = el("apbTutShade", "position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:1000000;pointer-events:none;display:none;");
  const hole = el("apbTutHole", "position:fixed;z-index:1000001;pointer-events:none;border-radius:8px;"
    + "box-shadow:0 0 0 9999px rgba(0,0,0,.55);outline:3px solid #58a6ff;outline-offset:2px;"
    + "transition:left .35s ease,top .35s ease,width .35s ease,height .35s ease;display:none;");
  const cur = el("apbTutCursor", "position:fixed;z-index:1000003;pointer-events:none;width:28px;height:28px;"
    + "transition:left .9s cubic-bezier(.4,0,.2,1),top .9s cubic-bezier(.4,0,.2,1);display:none;",
    '<svg viewBox="0 0 24 24" width="28" height="28"><path d="M4 2l15 9-6.5 1.5L16 20l-3 1.5-3.5-7.5L4 18z" '
    + 'fill="#fff" stroke="#1f2328" stroke-width="1.4" stroke-linejoin="round"/></svg>');
  const ring = el("apbTutRing", "position:fixed;z-index:1000002;pointer-events:none;width:36px;height:36px;"
    + "margin:-18px 0 0 -18px;border-radius:50%;border:3px solid #58a6ff;opacity:0;display:none;");
  if (!D.getElementById("apbTutStyle")) {
    const s = D.createElement("style"); s.id = "apbTutStyle";
    s.textContent = "@keyframes apbTutPulse{0%{transform:scale(.4);opacity:.9}100%{transform:scale(1.8);opacity:0}}";
    D.head.appendChild(s);
  }
  // 画面の外（閉じたサイドバーの中など）にあるものは見えていないとみなす
  function visible(e) { const r = e.getBoundingClientRect();
    return r.width > 2 && r.height > 2 && e.offsetParent !== null && r.right > 0 && r.left < P.innerWidth; }
  function find() {
    const t = cfg.target; if (!t) return null;
    let els = Array.from(D.querySelectorAll(t.sel));
    if (t.label) els = els.filter(e => { const l = e.querySelector('[data-testid="stWidgetLabel"]'); return l && l.innerText.trim() === t.label; });
    // 完全一致は行単位で比べる（メニューのリンクはアイコン名も文字として持っているため）
    if (t.text) els = els.filter(e => t.exact ? e.innerText.split("\n").some(x => x.trim() === t.text)
                                              : e.innerText.includes(t.text));
    // 文字で探すときは、いちばん内側（小さい）要素を選ぶ
    els = els.filter(visible);
    if (t.text && t.sel.indexOf("button") < 0) els.sort((a, b) => a.innerText.length - b.innerText.length);
    return els[0] || null;
  }
  function bubble() { return D.querySelector('[class*="st-key-tutbubble"]'); }
  let lastTarget = null, arrived = false;
  function place() {
    const b = bubble(); let t = find();
    if (!t && cfg.target && cfg.target.sidebar) {
      const ex = D.querySelector('[data-testid="stExpandSidebarButton"]');
      if (ex && visible(ex) && st.opened !== cfg.step) { st.opened = cfg.step; ex.click(); }
    }
    if (!t) {
      hole.style.display = "none"; cur.style.display = "none"; ring.style.display = "none";
      shade.style.display = cfg.target ? "none" : "block";
      if (b) { b.style.left = ""; b.style.top = ""; b.style.right = ""; b.style.bottom = "";
               if (!cfg.target) { b.style.left = "50%"; b.style.top = "50%"; b.style.transform = "translate(-50%,-50%)"; } }
      return;
    }
    if (st.scrolled !== cfg.step) { st.scrolled = cfg.step; t.scrollIntoView({ block: "center", behavior: "smooth" }); }
    shade.style.display = "none";
    const r = t.getBoundingClientRect(), pad = 6;
    Object.assign(hole.style, { display: "block", left: (r.left - pad) + "px", top: (r.top - pad) + "px",
      width: (r.width + 2 * pad) + "px", height: (r.height + 2 * pad) + "px" });
    // カーソル：前の位置から、照らした場所の中心へ動かす
    const cx = r.left + Math.min(r.width / 2, 120), cy = r.top + Math.min(r.height / 2, 40);
    if (lastTarget !== t) {
      lastTarget = t; arrived = false;
      const from = st.cursor || { x: P.innerWidth / 2, y: P.innerHeight / 2 };
      cur.style.transition = "none"; cur.style.left = from.x + "px"; cur.style.top = from.y + "px"; cur.style.display = "block";
      void cur.offsetWidth;
      cur.style.transition = "left .9s cubic-bezier(.4,0,.2,1),top .9s cubic-bezier(.4,0,.2,1)";
      cur.style.left = cx + "px"; cur.style.top = cy + "px";
      P.setTimeout(() => { arrived = true; ring.style.display = "block"; ring.style.left = cx + "px"; ring.style.top = cy + "px";
        ring.style.animation = "none"; void ring.offsetWidth; ring.style.animation = "apbTutPulse 1.4s ease-out infinite"; }, 950);
    } else if (arrived) {
      cur.style.left = cx + "px"; cur.style.top = cy + "px"; ring.style.left = cx + "px"; ring.style.top = cy + "px";
    }
    st.cursor = { x: cx, y: cy };
    // 吹き出し：照らした場所の下（入らなければ上、それも無理なら右下の定位置）
    if (b) {
      b.style.transform = ""; b.style.right = "auto"; b.style.bottom = "auto";
      const bw = b.offsetWidth, bh = b.offsetHeight, vw = P.innerWidth, vh = P.innerHeight, m = 12;
      let top = r.bottom + 16, left = Math.min(Math.max(m, r.left), vw - bw - m);
      if (top + bh > vh - m) top = r.top - bh - 16;
      if (top < m) { top = vh - bh - m; left = vw - bw - m; }
      if (r.right + bw + 24 < vw && (r.height > vh * 0.5)) { left = r.right + 16; top = Math.min(Math.max(m, r.top), vh - bh - m); }
      b.style.left = left + "px"; b.style.top = top + "px";
    }
  }
  place();
  st.timer = P.setInterval(place, 300);
})();
</script>
"""


def _bubble(step: dict, i: int, n: int, wrong_page: str | None):
    with st.container(key="tutbubble"):
        title, body = step["title"], step["body"]
        if wrong_page is not None:
            title = f"「{PAGES.get(wrong_page, wrong_page)}」画面に戻りましょう"
            body = f"次は「{PAGES.get(wrong_page, wrong_page)}」画面で「{step['title']}」を行います。左のメニューから開いてください。"
        ch_no = CHAPTERS.index(step["ch"]) + 1
        st.markdown(f"<div class='tut-ch'>{step['ch']}（{ch_no}/{len(CHAPTERS)}）・ステップ {i + 1}/{n}</div>"
                    f"<div class='tut-title'>{title}</div><div class='tut-body'>{body}</div>"
                    f"<div class='tut-bar'><i style='width:{(i + 1) / n * 100:.0f}%'></i></div>",
                    unsafe_allow_html=True)
        cols = st.columns([1.1, 1.45, 0.72, 0.72], gap="small")
        if step.get("last"):
            cols[0].button("終了する", type="primary", key="tut_finish", on_click=stop, width="stretch")
        elif wrong_page is not None:
            if cols[0].button("移動する", type="primary", key="tut_move", width="stretch"):
                _ss()["_tut_goto"] = wrong_page
                st.rerun()
        else:
            if step["done"] is None or step.get("next_label"):
                cols[0].button(step.get("next_label", "次へ"), type="primary" if step["done"] is None else "secondary",
                               key="tut_next", on_click=_next, width="stretch")
            if step.get("auto"):
                cols[1].button(step.get("auto_label", "代わりにやる"), key="tut_auto", on_click=_run_auto,
                               args=(step,), width="stretch")
        if i > 0:
            cols[2].button("戻る", key="tut_back", on_click=_back, type="tertiary", width="stretch")
        cols[3].button("終了", key="tut_stop", on_click=stop, type="tertiary", width="stretch")


def render(page: str):
    """app.py から毎回呼ぶ（ページの描画の前）。チュートリアル中なら、進み具合を判定して案内を出す"""
    ss = _ss()
    goto = ss.pop("_tut_goto", None)
    if goto is not None and goto != page and goto in _PAGE_FILES:
        st.switch_page(_PAGE_FILES[goto])
    if not active():
        if ss.pop("_tut_drawn", False):     # 終了した直後だけ、画面に足した要素を片付ける
            components.html(_JS.replace("__CFG__", json.dumps({"on": False})), height=0)
        return
    ctx = {"page": page}
    i = ss.get("tut_i", 0)
    # 操作が終わっているステップは飛ばす（「次へ」で進む説明ステップは止まる）
    while i < len(STEPS) - 1 and STEPS[i]["done"] is not None and STEPS[i]["done"](ctx):
        i += 1
    ss["tut_i"] = i
    step = STEPS[i]
    if step.get("pre") and not ss.get(f"_tut_pre_{i}"):
        ss[f"_tut_pre_{i}"] = True
        step["pre"]()
    need = step.get("page")
    wrong = need if (need is not None and need != page) else None
    target = _link(PAGES[need]) if wrong is not None else step.get("target")
    st.markdown(_CSS, unsafe_allow_html=True)
    _bubble(step, i, len(STEPS), wrong)
    components.html(_JS.replace("__CFG__", json.dumps(
        {"on": True, "step": f"{i}:{wrong or ''}", "target": target}, ensure_ascii=False)), height=0)
    ss["_tut_drawn"] = True
