# Calculus Visualization App 📐
An interactive tool for STEM students to visually understand challenging concepts in Calculus and Analysis.
理系大学生が解析学（微分積分学）の難所を視覚的に理解するためのインタラクティブ・ツールです。

## ✨ Core Features / 主な機能

This app visualizes abstract mathematical definitions dynamically.
教科書上の抽象的な概念を動的に可視化します。

1. **Taylor Series / テイラー展開**
   - Observe how polynomials approximate functions as the degree $n$ increases.
   - 次数を変えることで関数が近似される様子を観察。リアルタイムに LaTeX 数式を表示。

2. **ε-δ Definition / ε-δ 論法**
   - Dynamically adjust $\epsilon$ to see how $\delta$ responds.
   - 「連続」と「不連続」を切り替えて、極限の定義を視覚的に比較。

3. **Double Integral & Riemann Sum / 重積分とリーマン和**
   - Visualize volume approximation using rectangular prisms in 3D space.
   - 3D空間で、領域を直方体で細分化して体積を近似するプロセスを可視化。

4. **Gradient & Contour / 勾配と等高線**
   - Visualize how the gradient vector $\nabla f$ is always perpendicular to contour lines.
   - 勾配ベクトルが等高線に対して常に垂直である性質を 3D/2D で表示。

5. **Line Integral / 線積分**
   - Calculate "work" done by a vector field along a custom path.
   - ベクトル場の中を通る経路の「仕事」を計算し、追い風・向かい風を色分け表示。

## 🛠 Tech Stack / 使用技術

- **Language**: Python 3.9+
- **Framework**: [Streamlit](https://streamlit.io/)
- **Libraries**: NumPy, SciPy, Matplotlib

## 🚀 Setup & Run / 実行方法

```bash
# Install dependencies / ライブラリのインストール
python3 -m pip install streamlit numpy matplotlib scipy

# Run the app / アプリの起動
python3 -m streamlit run app.py