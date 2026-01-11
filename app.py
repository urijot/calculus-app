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