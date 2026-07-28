# strategy-lab

![CI](https://github.com/oimosanAI/strategy-lab/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/oimosanAI/strategy-lab/graph/badge.svg)](https://codecov.io/gh/oimosanAI/strategy-lab)
![tests](https://img.shields.io/badge/tests-283%20passed-brightgreen)
<!-- tests badge is static -- update the count manually (`poetry run pytest -q`) whenever new tests are added -->

## 1. プロジェクト概要

クオンツ・リスク管理職種への応募を意図した技術ポートフォリオ。単一戦略の実装ではなく、**複数戦略を統一的なバックテスト・評価基盤の上で比較検証できるフレームワーク**として設計した。ペアトレード（統計的裁定）とマルチファクター（モメンタム＋低ボラティリティ）の2戦略を、look-ahead biasのないバックテストエンジン・統計的有意性検定・取引コストを織り込んだ評価指標の上に実装し、S&P 500の実データでEnd-to-Endの動作検証を行った。

詳細な要件定義は[REQUIREMENTS.md](REQUIREMENTS.md)を参照。

## 2. 設計思想の核

- **Look-ahead biasの二重防御**：時点tの意思決定はt以前の情報のみに依存するという制約を、(a) `EXECUTION_LAG`をエンジンの1箇所だけに適用する構造的な仕組みと、(b) 未来の価格を摂動させても過去の意思決定が変化しないことを実際に検証する性質ベースの検証（`assert_causal`）の二重で担保する。
- **検証ツール自体の正しさも検証する**：ガードや統計検定は「意図通りに通ること」だけでなく「実際に不正な実装を捕まえられること」を、意図的に壊した負の対照実装を用いて確認する。ガードが通ることだけを確認しても、そのガードが機能している証拠にはならないという前提に立つ。
- **実データ検証で見つかった問題は隠さず記録する**：ユニットテストでは原理的に検出できない、実データ特有の問題（統計的検定の前提、ポジションサイジングのスケール等）が見つかった場合、それを解決した上で経緯・原因・教訓をコードと同じ場所に記録し、都合の良い結果だけを残さない。

## 3. 主要な発見

Phase 3の実データE2E検証では、いずれもユニットテストの粒度では検出できない、実運用に直結する問題を発見した。

### 3-1. 多重検定問題（ペアトレード）

候補ペアのスクリーニングにおいて、検定対象を2セクター（54銘柄）に限定した場合と、全11セクター（503銘柄）に拡張した場合とで、統計的結論が反転する現象を実データで確認した。

| 検定範囲 | 検定数（n_tests） | 生存ペア数（Bonferroni補正後） |
|---|---|---|
| 2セクター（54銘柄） | 308 | 1（AEP-FE） |
| 全宇宙（495銘柄） | 5,551 | **0** |

同一の生データ・同一の生p値（7.666e-05）が、検定族の定義次第で「有意」（adjusted p=0.024）にも「有意でない」（adjusted p=0.43）にもなる。「候補プールを広げれば良いペアが見つかりやすくなる」という直感は、多重検定補正の厳格化と正面から相殺するという教訓を、実データで定量的に示した。詳細は[strategies/pairs_trading/README.md](strategies/pairs_trading/README.md)。

### 3-2. ポジションサイジングのスケール見落とし（マルチファクター）

約100銘柄のロング・約100銘柄のショートを同時に保有するクロスセクショナル戦略を初めて実データで実行したところ、ポートフォリオ全体のgross exposureが最大56倍に達し、日次リターンが-126%、累積エクイティカーブがほぼゼロまで低下するという、数学的に不可能な数値が出力された。原因はlook-ahead biasではなく、既存のポジションサイジング機構（`VolTargetSizer`）が銘柄単位のレバレッジ上限のみを持ち、ポートフォリオ全体のエクスポージャーには何の制約もなかったこと——2銘柄構成のペアトレードでは顕在化しなかった設計ギャップだった。

| | 修正前 | 修正後 |
|---|---|---|
| Gross exposure（中央値/最大） | 39.9x / 56.3x | 1.20x / 1.48x |
| max_drawdown | -100.02%（数学的に無意味） | -12.52%（妥当） |

「非現実的に良すぎる数値」だけでなく「数学的に不可能なほど悪い数値」も、look-ahead bias以外の設計層（ポジションサイジング・ポートフォリオ構築）を疑う調査対象であるべき、という一般化可能な教訓を得た。詳細は[strategies/factor_momentum/README.md](strategies/factor_momentum/README.md)。

### 3-3. セクター中立化：設計目的の達成とバックテスト成績のトレードオフ

マルチファクター戦略にセクター中立化（セクターごとにquintileを構築し、セクター間で均等な予算配分を行う設計）を追加実装し、globalモード（全宇宙横断のクロスセクショナル）と実データで比較した。

| | Global | Sector-Neutral |
|---|---|---|
| セクター別net exposureのstd | 0.0757 | **0.0206**（約1/3.7） |
| OOS Sharpe | 0.44 | 0.03 |
| IS→OOS劣化 | 0.24 | **1.44** |

セクター集中抑制という設計目的自体は明確に達成された（セクター別net exposureのばらつきが約1/3.7に縮小）。しかし副作用として、IS→OOS劣化が大幅に悪化した。原因の一つとして、globalモードの見かけの好成績自体が、この評価期間にたまたま好調だったセクター（Utilities・Financials）への偏りに起因していた可能性がある——セクター中立化はまさにこの種の偶然のセクター寄与を排除するために存在する機能であり、排除後にOOSエッジが消えたという事実は、その解釈を支持する。

ここから得た教訓：**「新機能が設計目的（セクター中立性）を達成したこと」と「その機能がバックテスト結果を改善すること」は別問題である。** 理論的に妥当な機能を導入しても、バックテストの見かけの成績が向上するとは限らない——むしろ、それまで見えていた「良い成績」の一部が取り除かれるべきノイズやバイアスだったことが明らかになる場合がある。詳細は[strategies/factor_momentum/README.md](strategies/factor_momentum/README.md)。

### 3-4. VRPシグナルは単純に持ち続けるより明確に悪い結果を生んだ（Vol Arbitrage）

Phase 3拡張として、VIXと実現ボラティリティの乖離（Volatility Risk Premium, VRP）を取引する戦略を`SVXY`（ショートVIX先物ETF）で検証した。当初の想定（コンタンゴ減衰による右肩下がり）に反し、`SVXY`は対象期間中に+93%上昇していた。

「シグナルの効果」と「SVXY自体の値動き」を切り分けるため、Buy&Hold・サイジングのみ（VolTargetSizer適用、常にロング）・シグナル込みの3段階に分解したところ：

| | OOS累積リターン |
|---|---|
| Buy & Hold | +13.50% |
| サイジングのみ（タイミングなし） | +2.40% |
| VRPシグナル込み（実際の戦略） | **-0.41%** |

VRPタイミングシグナルは、単純に「常に持っておく」場合と比べて**明確に負の価値を追加していた**。これは他の2発見（統計的に中立な「エッジなし」）とは質的に異なる、より強いネガティブな結果である（統計的には有意でない）。原因（trailing RVという設計上の制約、`vrp_threshold`の根拠の弱さ、`SVXY`固有の値動き、のどれが主要因か）は今回の検証だけでは切り分けられていない。続けてwalk-forward分析を実施したところ、OOS Sharpeはウィンドウごとに-1.38〜+1.72まで振動し、上記の単一分割の結果が広い分布からの一標本に過ぎなかったことが、pairs_trading・factor_momentumに続き3戦略目でも確認された。詳細は[strategies/vol_arbitrage/README.md](strategies/vol_arbitrage/README.md)。

### 3-5. 生成済みレポート

上記の検証で実際に`core.evaluation.report`から生成したMarkdownレポートを[`reports/`](reports/)に格納している。

- [`reports/pairs_trading_aep_fe.md`](reports/pairs_trading_aep_fe.md) — ペアトレード（AEP-FE）単体レポート
- [`reports/factor_momentum_global.md`](reports/factor_momentum_global.md) — マルチファクター（globalモード）単体レポート
- [`reports/factor_momentum_sector_neutral.md`](reports/factor_momentum_sector_neutral.md) — マルチファクター（sector_neutralモード）単体レポート
- [`reports/strategy_comparison.md`](reports/strategy_comparison.md) — `render_comparison_report`によるペアトレード vs マルチファクターの横並び比較
- [`reports/vol_arbitrage_vrp.md`](reports/vol_arbitrage_vrp.md) — Vol Arbitrage（VRP戦略）単体レポート

## 4. 技術スタック・アーキテクチャ概要

```
core/
├── data/         DataLoader・キャッシュ・S&P500ユニバース取得
├── backtest/     バックテストエンジン（look-ahead防止、取引コスト、ポジションサイジング、IS/OOS分割）
└── evaluation/   評価指標（Sharpe/Sortino/Calmar等）・統計的有意性検定・レポート生成
strategies/
├── pairs_trading/     統計的裁定（Engle-Granger/Johansen共和分検定、Bonferroni補正、静的/Kalmanヘッジ）
├── factor_momentum/   マルチファクター（12-1モメンタム＋60日低ボラティリティ、月次リバランス）
└── vol_arbitrage/     Volatility Risk Premium（VIX vs SPY実現ボラティリティ、SVXYロングオンリー、Phase 3拡張）
```

主要ライブラリ：`pandas` / `numpy` / `statsmodels` / `scipy`（数値計算・統計検定）、`yfinance`（データ取得）、`pytest`（テスト、283件・カバレッジ98%）、`ruff` / `mypy`（静的検査）、`poetry`（依存管理）。

## 5. 再現手順

```bash
git clone https://github.com/oimosanAI/strategy-lab.git
cd strategy-lab
poetry install
poetry run pytest --cov=core --cov=strategies
```

## 6. 既知の限界

- **サバイバーシップバイアス（Level A）**：現在のS&P 500構成銘柄のみを使用しており、上場廃止・合併銘柄は候補から除外されている。
- **Walk-forwardは実施済みだが、成績の振れ幅・選定の不安定性が残る**：anchored 4ウィンドウでのwalk-forward検証（両戦略）を実施済み。pairs_tradingは「勝者」のペアがウィンドウごとに入れ替わる選定の不安定性、factor_momentumはOOS Sharpeが-0.92〜+1.87まで振動する結果の不安定性が判明しており、単一分割の限界は解消したが新たな限界として記録している。
- **セクター中立化は実装済みだが副作用がある**：セクター集中抑制という設計目的（§3-3）は実データで確認済みだが、IS→OOS劣化が悪化する副作用が観測されており、現時点でglobalモードに対する明確な優位性は確認できていない。
- **`check_exposure_limits()`は事後モニターであり強制ではない**：将来の設定変更で同種のスケール問題が再発しても、バックテスト自体は警告なく実行される。
- **Vol Arbitrageもwalk-forward・パラメータ感度分析済み**：anchored 4ウィンドウでOOS Sharpeが-1.38〜+1.72まで振動することを確認し、`vrp_threshold`感度分析（5点グリッド、負の閾値含む）ではフィルタを弱めても改善しないことを確認した。原因の切り分け（trailing RVの設計上の制約か、SVXY固有の値動きか）は依然として未解決。

各限界の詳細は[strategies/pairs_trading/README.md](strategies/pairs_trading/README.md)・[strategies/factor_momentum/README.md](strategies/factor_momentum/README.md)・[REQUIREMENTS.md](REQUIREMENTS.md)を参照。
