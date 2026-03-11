# Claude Codeへの依頼：Automation Playback タイミング解析ツール

## 概要

オムロン「オートメーションプレイバック」機能で取得したCSVデータを読み込み、
PLCの動作サイクルごとのBool信号タイミングを可視化・分析するStreamlitアプリを作成してほしい。

目的はオートメーションプレイバック機能の次期開発に向けたPoC（プロトタイプ）であり、
「このような分析機能を製品に組み込むと有用か」を検証するためのもの。

---

## 入力CSVの仕様

- 1行目：変数名ヘッダ
- 2行目以降：計測値（サンプリング周期 1ms）
- 1列目：タイムスタンプ（`yyyy/mm/dd HH:MM:SS.fff` 形式）
- 2列目以降：PLCの変数値（Bool変数は 0/1 または TRUE/FALSE、数値変数も混在）

例：
```
Timestamp,CycleStart,Valve_A,Sensor_1,PLC_Ready,Pressure,Temperature
2024/01/15 10:00:00.000,0,0,0,0,10.5,25.0
2024/01/15 10:00:00.001,1,0,0,1,10.5,25.0
...
```

サンプルCSVファイル：`sample_playback.csv`（同ディレクトリに配置）

---

## アプリの構成（Streamlit シングルページ）

### ① CSV読み込みエリア
- `st.file_uploader` でCSVをアップロード
- アップロード後、データのプレビュー（先頭10行）を表示
- 変数の自動判定：各列の値を見てBool変数か数値変数かを判定して表示
  - Bool判定条件：列の値が 0/1/TRUE/FALSE のみで構成されている

### ② 解析設定パネル（サイドバー）

CSVアップロード後に以下の設定UIを表示する：

**サイクル基準設定**
- `st.selectbox`：サイクル基準変数を選択（全変数から選択）
- `st.radio`：トリガーエッジ選択（「立ち上がり（OFF→ON）」/ 「立ち下がり（ON→OFF）」）

**観測変数設定**
- `st.multiselect`：観測するBool変数を複数選択（Bool変数のみリストアップ）

**閾値設定**
- `st.number_input`：閾値（ms）を入力（0=設定なし）
- 閾値を設定した場合、ヒストグラム上に閾値ラインを描画し、達成率を表示

**解析実行**
- `st.button`：「解析実行」ボタン

### ③ 解析結果エリア

解析実行後、以下をタブで表示：

**タブ1：サイクル一覧テーブル**
- 列：サイクル#、開始時刻、サイクル長[ms]、各観測変数の遅れ時間[ms]、閾値判定（OK/NG）
- 閾値NGの行はハイライト表示（`st.dataframe` の `style` 活用）
- CSVダウンロードボタン

**タブ2：ヒストグラム**
- 観測変数ごとにヒストグラムを表示（Plotly Express使用）
- ビン数：スタージェスの公式で自動計算（`int(1 + 3.32 * log10(n))`）
- 閾値が設定されている場合：垂直ラインを追加、閾値以内をblue/超過をredで色分け
- X軸：遅れ時間[ms]、Y軸：頻度
- 統計情報を各グラフの下に表示：サンプル数、最小、最大、平均、標準偏差、3σ上限、閾値達成率

**タブ3：時系列波形**
- 選択した変数の時系列をプロット（Plotly使用）
- 縦軸：変数値、横軸：時刻
- サイクル開始タイミングに垂直線（破線）を重ねる
- Bool変数は塗りつぶし波形（`fill='tozeroy'`）
- 数値変数も同じグラフに重ねて表示可能（第2Y軸）

**タブ4：サイクル比較**
- 各サイクルを時間ゼロ基準に揃えて重ねてプロット
- 観測変数ごとに、全サイクルの波形を薄い色で重ね、平均波形を太線で表示
- サイクル間のばらつきを視覚的に確認できる

---

## 正常vs異常 比較分析機能

### 概要

通常解析（単一CSVファイル）に加えて、正常時と異常時のCSVを1ファイルずつアップロードして
比較分析できるモードを追加する。

### UI構成

サイドバー上部に **モード切替** を設ける：
- `st.radio`：「単体解析」／「正常vs異常 比較」

**比較モード選択時のCSVアップロードUI：**
```
st.file_uploader("正常時CSVをアップロード", key="normal_csv")
st.file_uploader("異常時CSVをアップロード", key="abnormal_csv")
```

両ファイルがアップロードされたら、サイクル基準・観測変数の設定は共通で使い回す
（両ファイルの列名が一致していることを前提とする。不一致時は `st.warning` で案内）。

### 比較結果タブ（既存タブの後ろに追加）

**タブ5：重ねヒストグラム（正常 vs 異常）**

観測変数ごとに1グラフ。正常と異常の遅れ時間分布を半透明で重ねて表示：

```python
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=normal_delays, name="正常", opacity=0.6,
    marker_color="royalblue", nbinsx=bin_count
))
fig.add_trace(go.Histogram(
    x=abnormal_delays, name="異常", opacity=0.6,
    marker_color="crimson", nbinsx=bin_count
))
fig.update_layout(barmode='overlay')
```

各グラフの下に比較統計テーブルを表示：

| 指標 | 正常 | 異常 | 差分（異常-正常） |
|------|------|------|-----------------|
| サンプル数 | N | N | - |
| 平均[ms] | X | X | ΔX |
| 標準偏差[ms] | X | X | ΔX |
| 3σ上限[ms] | X | X | ΔX |

---

**タブ6：ずれランキング**

全観測変数の「平均遅れ時間の差分（異常平均 - 正常平均）」を計算し、
差分の大きい順にバーチャートで表示：

```python
# 差分計算
diff_data = []
for col in target_cols:
    delta = abnormal_means[col] - normal_means[col]
    z_score = delta / normal_stds[col] if normal_stds[col] > 0 else 0
    diff_data.append({
        "変数名": col,
        "遅れ差分[ms]": delta,
        "正規化スコア(Zスコア)": z_score  # 正常のσで正規化
    })
df_diff = pd.DataFrame(diff_data).sort_values("遅れ差分[ms]", ascending=False)
```

表示内容：
- バーチャート（横軸：変数名、縦軸：遅れ差分[ms]、色：差分が大きいほど赤）
- 上位の変数に「⚠️ 要注意」バッジ（Zスコア > 2.0 を自動ピックアップ）
- テーブルでも同じデータを表示（ダウンロード可能）

---

**タブ7：自動ピックアップ（異常原因候補）**

「正常の分布から大きく外れた変数」を自動で絞り込んで提示する。

判定ロジック：
```python
def is_anomalous(col, normal_delays, abnormal_delays):
    normal_mean = np.mean(normal_delays)
    normal_std = np.std(normal_delays)
    threshold_3sigma = normal_mean + 3 * normal_std

    # 異常サイクルの何%が正常の3σを超えているか
    exceed_rate = np.mean(np.array(abnormal_delays) > threshold_3sigma)
    return exceed_rate, exceed_rate > 0.3  # 30%以上が3σ超えなら「異常候補」
```

表示：
- 異常候補の変数リストを `st.metric` で目立たせる（超過率をデルタ表示）
- 「この信号が異常時に遅延している可能性が高い」という文言とともに提示
- 候補変数について、正常vs異常の重ねヒストグラムをその場で再表示

---

**タブ8：波形重ね比較**

サイクルを時間ゼロ基準に揃えて、正常グループ（青系）と異常グループ（赤系）を重ねてプロット。

```python
# 正常サイクルを薄い青で重ねる
for cycle in normal_cycles:
    fig.add_trace(go.Scatter(
        x=cycle['time_offset_ms'], y=cycle['value'],
        line=dict(color='rgba(65,105,225,0.2)'), showlegend=False
    ))

# 異常サイクルを薄い赤で重ねる
for cycle in abnormal_cycles:
    fig.add_trace(go.Scatter(
        x=cycle['time_offset_ms'], y=cycle['value'],
        line=dict(color='rgba(220,20,60,0.2)'), showlegend=False
    ))

# 各グループの平均波形を太線で重ねる
fig.add_trace(go.Scatter(
    x=time_axis, y=normal_mean_wave,
    line=dict(color='royalblue', width=3), name='正常 平均'
))
fig.add_trace(go.Scatter(
    x=time_axis, y=abnormal_mean_wave,
    line=dict(color='crimson', width=3), name='異常 平均'
))
```

変数は `st.selectbox` で切り替えて表示。

---

### 比較分析のアルゴリズム補足

両ファイルを独立してサイクル検出・遅れ時間計算し、それぞれの結果DataFrameを比較する。
サイクル数が異なっても動作すること（単純に各グループの統計値を比較するだけでよい）。

```python
# analyzer.py に追加する関数
def compare_normal_abnormal(normal_df, abnormal_df, trigger_col, edge, target_cols):
    normal_result = analyze_cycles(normal_df, trigger_col, edge, target_cols)
    abnormal_result = analyze_cycles(abnormal_df, trigger_col, edge, target_cols)
    return normal_result, abnormal_result
```

---

## アルゴリズム仕様

### サイクル検出
```python
# タイムスタンプをdatetimeに変換（pandas）
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M:%S.%f')

# 基準変数の立ち上がりエッジ検出
df['_prev'] = df[trigger_col].shift(1)
if edge == 'RISE':
    cycle_starts = df[(df['_prev'] == 0) & (df[trigger_col] == 1)].index
elif edge == 'FALL':
    cycle_starts = df[(df['_prev'] == 1) & (df[trigger_col] == 0)].index
```

### 遅れ時間計算
```python
# 各サイクルで観測変数が最初にONになる行を検索
for i, start_idx in enumerate(cycle_starts[:-1]):
    end_idx = cycle_starts[i+1]
    cycle_df = df.loc[start_idx:end_idx-1]
    start_time = df.loc[start_idx, 'Timestamp']
    
    for target_col in target_cols:
        on_rows = cycle_df[cycle_df[target_col] == 1]
        if len(on_rows) > 0:
            delay_ms = (on_rows.iloc[0]['Timestamp'] - start_time).total_seconds() * 1000
```

---

## 技術スタック

```
streamlit>=1.32.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.24.0
```

`requirements.txt` を生成すること。

---

## ファイル構成

```
timing_analyzer/
├── app.py                    # Streamlitメインアプリ（モード切替・UI）
├── analyzer.py               # 解析ロジック（CSVパース・サイクル検出・遅れ計算）
├── comparator.py             # 比較解析ロジック（正常vs異常の差分・ランキング・自動ピックアップ）
├── requirements.txt
├── sample_playback.csv       # テスト用サンプルCSV：正常データ（添付）
└── sample_playback_ng.csv    # テスト用サンプルCSV：異常データ（生成すること）
```

---

## 実装上の注意

- `analyzer.py` に解析ロジックを分離し、`app.py` はUI描画のみにする
- 大きいCSV（最大200,000行）でも動作するよう、pandas のベクトル演算を活用
- エラーハンドリング：CSVのタイムスタンプ形式が異なる場合に `st.error` で案内
- `st.session_state` を使い、設定変更時に解析結果が消えないようにする
- 日本語表示（UI文字列はすべて日本語）

---

## 起動方法（READMEに記載）

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ガントチャート機能（タクト分析・ドリルダウン）

### 概要

サイクル内の各信号ONタイミングをガントチャートで可視化し、
タクト全体のどのステップにどれだけ時間がかかっているかを把握する。
棒をクリックすると該当区間の詳細波形にドリルダウンできる。

---

### UI構成

#### ガントチャート設定パネル（サイドバー内、専用セクション）

**ステップ定義：手動定義のみ**

ユーザーが「どの変数のONタイミングを、どの順序でステップとして扱うか」を自分で定義する。

**操作UI：**

```
st.multiselect で変数を選択
↓
選んだ変数を st.data_editor（編集可能テーブル）で順序を並び替え
　列：順序番号 / 変数名 / ステップ名（自由記入） / 色（任意）
↓
「この設定をJSONで保存」ボタン → ダウンロード
「JSONを読み込む」ボタン → アップロードで前回設定を復元
```

例：
| 順序 | 変数名 | ステップ名 | 色 |
|------|--------|-----------|-----|
| 1 | CycleStart | サイクル開始 | blue |
| 2 | PLC_Ready | PLC準備完了 | green |
| 3 | Valve_A | バルブA開 | orange |
| 4 | Sensor_1 | センサ1検知 | red |

**タクト目標設定：**
- `st.number_input`：タクト目標時間[ms]（0=設定なし）

---

### ガントチャート表示（タブ9：タクト分析）

#### 上段：ガントチャート本体

Plotlyの横棒チャート（`go.Bar` horizontal）で実装：

```python
# 各ステップの区間を計算
# 区間 = 「このステップのONタイミング」から「次のステップのONタイミング」まで
# 起点は常にサイクル開始（t=0）

steps = [
    {"name": "PLC準備完了",  "start": 0,    "mean": 5,  "min": 4,  "max": 7},
    {"name": "バルブA開",    "start": 5,    "mean": 13, "min": 10, "max": 17},
    {"name": "センサ1検知",  "start": 18,   "mean": 12, "min": 9,  "max": 16},
]

fig = go.Figure()
for step in steps:
    # 平均バー
    fig.add_trace(go.Bar(
        name=step["name"],
        y=[step["name"]],
        x=[step["mean"]],
        base=[step["start"]],
        orientation='h',
        width=0.4,
        marker_color=step["color"],
        customdata=[[step["start"], step["mean"], step["min"], step["max"]]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "開始: %{customdata[0]:.1f} ms<br>"
            "区間長（平均）: %{customdata[1]:.1f} ms<br>"
            "min: %{customdata[2]:.1f} ms / max: %{customdata[3]:.1f} ms"
        )
    ))
    # バラツキ帯（min〜maxを薄い色で）
    fig.add_trace(go.Bar(
        y=[step["name"]],
        x=[step["max"] - step["min"]],
        base=[step["start"] + step["min"]],
        orientation='h',
        width=0.4,
        marker_color=step["color"],
        opacity=0.2,
        showlegend=False,
        hoverinfo='skip'
    ))

# タクト目標ライン
if takt_target > 0:
    fig.add_vline(x=takt_target, line_dash="dash", line_color="red",
                  annotation_text=f"タクト目標 {takt_target}ms")

fig.update_layout(
    barmode='overlay',
    title="タクト分析ガントチャート（サイクル内シーケンス）",
    xaxis_title="サイクル開始からの経過時間 [ms]",
    yaxis=dict(autorange="reversed"),  # 上から順に表示
    height=400
)
```

ホバー時に表示する情報：ステップ名、開始時刻、区間長の平均/min/max

#### 中段：表示ベース

**全サイクルの平均値**をベースに表示する（固定）。サイクル選択スライダーは不要。

#### 下段：ステップ別統計テーブル

| ステップ名 | 平均[ms] | 標準偏差[ms] | min[ms] | max[ms] | タクト比率[%] |
|-----------|---------|------------|---------|---------|-------------|
| PLC準備完了 | 5.2 | 0.3 | 4.8 | 5.9 | 17% |
| バルブA開   | 13.4 | 1.8 | 10.1 | 17.2 | 44% |
| センサ1検知 | 11.8 | 2.1 | 9.0 | 15.8 | 39% |

「タクト比率」= そのステップの平均区間長 / 全ステップ合計 × 100

---

### ドリルダウン（タブ10：区間詳細波形）

ガントチャートのステップ名を `st.selectbox` で選択すると、その区間を拡大表示する。

```python
selected_step = st.selectbox("詳細を見るステップを選択", step_names)
# 選択されたステップの開始〜終了時刻の範囲でデータをスライス
step_start_ms = step_def[selected_step]["mean_start"]
step_end_ms   = step_def[selected_step]["mean_start"] + step_def[selected_step]["mean_duration"] * 1.5

# その時間範囲の全変数の波形を表示
```

表示内容：
- 選択区間の全変数をPlotlyで波形表示（Bool変数は塗りつぶし、数値変数は折れ線）
- 全サイクルを薄い色で重ね表示 ＋ 平均波形を太線で重ねる
- 区間の開始・終了タイミングに垂直破線を表示
- 「前のステップ」「次のステップ」ボタンでシーケンスを順番に閲覧できる

```python
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    if st.button("◀ 前のステップ"):
        st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
with col3:
    if st.button("次のステップ ▶"):
        st.session_state.step_idx = min(len(steps)-1, st.session_state.step_idx + 1)
```

---

### ガントチャート設定のJSON保存・復元仕様

```json
{
  "version": "1.0",
  "takt_target_ms": 50,
  "steps": [
    {"order": 1, "variable": "CycleStart", "label": "サイクル開始", "color": "#4472C4"},
    {"order": 2, "variable": "PLC_Ready",  "label": "PLC準備完了",  "color": "#70AD47"},
    {"order": 3, "variable": "Valve_A",    "label": "バルブA開",    "color": "#ED7D31"},
    {"order": 4, "variable": "Sensor_1",   "label": "センサ1検知",  "color": "#FF0000"}
  ]
}
```

保存：`st.download_button` でJSONをダウンロード
復元：`st.file_uploader` でJSONを読み込み、設定を自動反映

