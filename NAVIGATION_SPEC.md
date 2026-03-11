# 画面遷移設計（マルチページ構成）

## 全体構造

Streamlitのマルチページ構成（`pages/` ディレクトリ構成）で実装する。

```
timing_analyzer/
├── app.py                  # エントリーポイント・ナビゲーション定義
├── pages/
│   ├── 1_工程登録.py
│   ├── 2_ガントチャート.py
│   └── 3_ヒストグラム詳細.py
├── analyzer.py
├── comparator.py
├── requirements.txt
├── sample_playback.csv
└── sample_playback_ng.csv
```

ページ間の状態共有は `st.session_state` で行う。主要なキー：

```python
st.session_state["csv_df"]            # 読み込んだDataFrame
st.session_state["processes"]         # 登録済み工程リスト
st.session_state["selected_process"]  # Page2で選択中の工程名
st.session_state["selected_step"]     # Page3で表示するステップ名
```

---

## Page 1：工程登録・一覧

### CSVアップロード

アップロード後：
- 先頭10行をプレビュー表示
- Bool変数を自動スキャンして「工程基準変数の候補」をチェックボックスで一覧表示
  - 判定条件：列の値が 0/1/TRUE/FALSE のみで構成される

### 工程の登録UI

候補変数をチェックすると工程名入力欄とエッジ選択が展開される。
「工程を登録」ボタンで `st.session_state["processes"]` に追加：

```python
st.session_state["processes"][process_name] = {
    "trigger_var": var,
    "edge": "RISE",  # or "FALL"
    "step_order": [],    # Page2で定義（初期は空）
    "takt_target_ms": 0, # Page2で設定
}
```

### 登録済み工程一覧（カード形式）

登録済み工程をカードで並べて表示する（`st.columns` で横並び）：

```
┌─────────────────────────────────┐
│ 工程A                            │
│ 基準変数: Process_A_Start        │
│ エッジ: 立ち上がり                │
│ 検出サイクル数: 18                │
│ ステップ定義: 未設定              │
│                    [分析する →]  │
└─────────────────────────────────┘
```

- ステップ定義が済んでいれば「ステップ定義: 4ステップ」と表示
- 「分析する →」ボタンで `st.session_state["selected_process"]` に工程名をセットして
  `st.switch_page("pages/2_ガントチャート.py")` で遷移
- 削除ボタンも各カードに配置

---

## Page 2：ガントチャート

### ヘッダ

```
← 工程一覧に戻る    工程A のタクト分析
```

### 左カラム（設定パネル、幅の比率 = 1）

**ステップ定義：**

この工程に含まれるBool変数を `st.multiselect` で選択し、
`st.data_editor` で順序・ステップ名・色を編集できる：

| 順序 | 変数名 | ステップ名 | 色 |
|------|--------|-----------|-----|
| 1 | PLC_Ready | PLC準備完了 | #4472C4 |
| 2 | Valve_A | バルブA開 | #ED7D31 |
| 3 | Sensor_1 | センサ1検知 | #70AD47 |

編集内容は `st.session_state["processes"][selected_process]["step_order"]` に即時反映する。

**タクト目標設定：**

```python
takt_target = st.number_input("タクト目標[ms]", min_value=0, value=0)
```

**JSON保存・復元：**

```python
# 保存
st.download_button("設定を保存(JSON)", data=json.dumps(config), file_name="steps.json")
# 復元
uploaded_json = st.file_uploader("設定を読み込む(.json)", type="json")
```

### 右カラム（ガントチャート本体、幅の比率 = 3）

ステップが1つ以上定義されると描画する。

**ガントチャート（Plotly 横棒）：**

- 全サイクル平均値ベースの横棒チャート
- 各棒：平均区間長（solid color）
- 薄い帯：min〜max のばらつき範囲（opacity=0.2で重ねる）
- 赤破線：タクト目標ライン（設定時のみ）
- ホバー表示：ステップ名・平均・min・max・タクト比率[%]
- Y軸は順序通りに上から下に並べる（`autorange="reversed"`）

**棒クリックでPage 3へ遷移：**

```python
event = st.plotly_chart(fig, on_select="rerun", key="gantt_chart")
if event and event.selection.points:
    clicked_step = event.selection.points[0]["y"]
    st.session_state["selected_step"] = clicked_step
    st.switch_page("pages/3_ヒストグラム詳細.py")
```

**ステップ別統計テーブル（ガントの下に表示）：**

| ステップ名 | 平均[ms] | 標準偏差[ms] | min[ms] | max[ms] | タクト比率[%] |
|-----------|---------|------------|---------|---------|-------------|
| PLC準備完了 | 5.2 | 0.3 | 4.8 | 5.9 | 17% |
| バルブA開   | 13.4 | 1.8 | 10.1 | 17.2 | 44% |
| センサ1検知 | 11.8 | 2.1 | 9.0 | 15.8 | 39% |

---

## Page 3：ヒストグラム詳細

### ヘッダ

```
← ガントチャートに戻る    工程A > バルブA開（ステップ詳細）
```

```python
if st.button("← ガントチャートに戻る"):
    st.switch_page("pages/2_ガントチャート.py")
```

### 左カラム：ヒストグラム

- Plotlyヒストグラム（スタージェスの公式でビン数自動計算）
- 閾値ライン（垂直破線）＋閾値以内を青・超過を赤で色分け
- 統計情報テーブル：サンプル数・平均・標準偏差・3σ上限・達成率

### 右カラム：波形重ね表示

このステップの全サイクルを薄い色で重ね表示し、平均波形を太線で重ねる。
X軸：ステップ開始からの経過時間[ms]。

### 下段：閾値設定

```python
threshold = st.number_input("閾値[ms]を設定", min_value=0.0, value=0.0, step=0.5)
st.caption(f"推奨閾値（平均 + 3σ）: {mean + 3*std:.1f} ms")
```

閾値を変更するとヒストグラムがリアルタイムで更新される（`st.session_state` で保持）。

---

## 画面遷移まとめ

```
Page 1：工程登録・一覧
  │
  │  「分析する →」クリック
  ↓
Page 2：ガントチャート
  │
  │  ガントの棒をクリック
  ↓
Page 3：ヒストグラム詳細
  │
  │  「← ガントチャートに戻る」
  ↑
Page 2（戻る）
  │
  │  「← 工程一覧に戻る」
  ↑
Page 1（戻る）
```

---

## ファイル構成（最終版）

```
timing_analyzer/
├── app.py                  # エントリーポイント・ナビゲーション定義
├── pages/
│   ├── 1_工程登録.py        # Page1
│   ├── 2_ガントチャート.py   # Page2
│   └── 3_ヒストグラム詳細.py # Page3
├── analyzer.py             # サイクル検出・遅れ時間計算
├── comparator.py           # 正常vs異常比較ロジック
├── requirements.txt
├── sample_playback.csv     # テスト用サンプルCSV（正常）
└── sample_playback_ng.csv  # テスト用サンプルCSV（異常）
```

※ 既存の仕様（単体解析タブ1〜4・比較分析タブ5〜8）は
Page2・Page3の中にタブとして組み込む形で統合する。
タブ9・10（ガントチャート・ドリルダウン）がPage2・Page3の主コンテンツとなる。
