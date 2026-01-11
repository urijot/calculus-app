import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

st.title("解析学：テイラー展開の視覚化")

# 1. 関数の選択とパラメータ設定
target_func = st.sidebar.selectbox("関数を選択", ["sin(x)", "cos(x)", "exp(x)"])
n = st.sidebar.slider("近似の次数 (n)", 0, 10, 1)

# 2. 数学的定義とLaTeX数式の生成
def get_data(x, n, func_type):
    y = np.zeros_like(x)
    latex_formula = ""
    
    if func_type == "sin(x)":
        for i in range(n + 1):
            term_val = ((-1)**i) * (x**(2*i+1)) / factorial(2*i+1)
            y += term_val
        # LaTeX形式の数式（sinの一般項）
        latex_formula = r"P_{2n+1}(x) = \sum_{k=0}^{n} \frac{(-1)^k x^{2k+1}}{(2k+1)!}"
        return y, np.sin(x), latex_formula
    
    elif func_type == "cos(x)":
        for i in range(n + 1):
            y += ((-1)**i) * (x**(2*i)) / factorial(2*i)
        latex_formula = r"P_{2n}(x) = \sum_{k=0}^{n} \frac{(-1)^k x^{2k}}{(2k)!}"
        return y, np.cos(x), latex_formula
    
    elif func_type == "exp(x)":
        for i in range(n + 1):
            y += (x**i) / factorial(i)
        latex_formula = r"P_n(x) = \sum_{k=0}^{n} \frac{x^k}{k!}"
        return y, np.exp(x), latex_formula

# 3. データ作成
x = np.linspace(-5, 5, 400)
y_approx, y_orig, formula = get_data(x, n, target_func)

# 4. 数式の表示（ここが新機能！）
st.write(f"### {target_func} のテイラー多項式:")
st.latex(formula)

# 5. グラフ描画
fig, ax = plt.subplots()
ax.plot(x, y_orig, label=f"Original {target_func}", color="gray", linestyle="--")
ax.plot(x, y_approx, label=f"Taylor Approx (n={n})", color="green", linewidth=2)
ax.set_ylim(-3, 3)
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.title("解析学：ε-δ 論法の視覚化")
st.write(r"目標: $\lim_{x \to a} f(x) = L$ の定義を理解する")

# 1. パラメータ設定
a = 2.0  # 極限点
L = 4.0  # 極限値
epsilon = st.slider("ε (縦の許容誤差) を決めてください", 0.1, 2.0, 1.0, step=0.1)

# f(x) = x^2 において、|x^2 - 4| < eps を満たす delta を計算
# 本来は論理的に探すべきですが、視覚化のために最大の delta を逆算します
delta = np.sqrt(L + epsilon) - a

# 2. グラフ描画用のデータ
x = np.linspace(0, 4, 400)
y = x**2

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, color="black", label="f(x) = x^2")

# ε の範囲（縦軸の帯）
ax.axhspan(L - epsilon, L + epsilon, color="yellow", alpha=0.3, label="ε-band (|f(x) - L| < ε)")
ax.axhline(L, color="red", linestyle="--", alpha=0.5)

# δ の範囲（横軸の帯）
ax.axvspan(a - delta, a + delta, color="blue", alpha=0.2, label="δ-band (|x - a| < δ)")
ax.axvline(a, color="blue", linestyle="--", alpha=0.5)

# 点 (a, L)
ax.plot(a, L, 'ro')

ax.set_xlim(0, 4)
ax.set_ylim(0, 8)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

st.info(f"ε = {epsilon:.2f} のとき、δ = {delta:.2f} とすれば、青い範囲の x はすべて黄色の範囲に収まります。")