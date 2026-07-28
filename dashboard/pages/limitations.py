"""既知の限界ページ: README.md Sec 6相当。静的コンテンツのみ、
再計算は一切行わない。"""

from __future__ import annotations

import streamlit as st

st.title("既知の限界")

st.markdown(
    """
- **サバイバーシップバイアス（Level A）**：現在のS&P 500構成銘柄のみを使用しており、上場廃止・合併銘柄は候補から除外されている。
- **Walk-forwardは実施済みだが、成績の振れ幅・選定の不安定性が残る**：anchored 4ウィンドウでのwalk-forward検証（3戦略）を実施済み。OOS Sharpeがウィンドウごとに大きく振動する結果の不安定性が判明している。
- **セクター中立化は実装済みだが副作用がある**：セクター集中抑制という設計目的は実データで確認済みだが、IS→OOS劣化が悪化する副作用が観測されており、globalモードに対する明確な優位性は確認できていない。
- **`check_exposure_limits()`は事後モニターであり強制ではない**：将来の設定変更で同種のスケール問題が再発しても、バックテスト自体は警告なく実行される。
- **Vol Arbitrageのシグナルが機能しなかった具体的な要因は特定できていない**：trailing RVという設計上の制約、`vrp_threshold`の根拠の弱さ、SVXY固有の値動きのどれが主要因かは切り分けられていない。
- **本ダッシュボードのTier 2データは凍結スナップショットである**：ライブ取得ではなく、生成時点（`scripts/generate_dashboard_data.py`実行時）の価格データに固定されている。

詳細は各戦略の`README.md`・`REQUIREMENTS.md`を参照してください。
"""
)
