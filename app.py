import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# サイドバーで「どの機能を使うか」を選べるようにする
st.sidebar.title("解析学メニュー")
menu = st.sidebar.selectbox("学習テーマを選択", ["テイラー展開", "ε-δ 論法"])

# ---------------------------------------------------------
# 1. テイラー展開の画面
# ---------------------------------------------------------
if menu == "テイラー展開":
    st.title("解析学：テイラー展開の視覚化")
    
    target_func = st.sidebar.selectbox("関数を選択", ["sin(x)", "cos(x)", "exp(x)"])
    n = st.sidebar.slider("近似の次数 (n)", 0, 15, 1)

    def get_data(x, n, func_type):
        y = np.zeros_like(x)
        if func_type == "sin(x)":
            for i in range(n + 1):
                y += ((-1)**i) * (x**(2*i+1)) / factorial(2*i+1)
            return y, np.sin(x), r"P_{2n+1}(x) = \sum_{k=0}^{n} \frac{(-1)^k x^{2k+1}}{(2k+1)!}"
        elif func_type == "cos(x)":
            for i in range(n + 1):
                y += ((-1)**i) * (x**(2*i)) / factorial(2*i)
            return y, np.cos(x), r"P_{2n}(x) = \sum_{k=0}^{n} \frac{(-1)^k x^{2k}}{(2k)!}"
        elif func_type == "exp(x)":
            for i in range(n + 1):
                y += (x**i) / factorial(i)
            return y, np.exp(x), r"P_n(x) = \sum_{k=0}^{n} \frac{x^k}{k!}"

    x = np.linspace(-5, 5, 400)
    y_approx, y_orig, formula = get_data(x, n, target_func)

    st.latex(formula)
    fig, ax = plt.subplots()
    ax.plot(x, y_orig, label="Original", color="gray", linestyle="--")
    ax.plot(x, y_approx, label=f"Taylor (n={n})", color="blue")
    ax.set_ylim(-3, 3)
    ax.legend()
    st.pyplot(fig)

# ---------------------------------------------------------
# 2. ε-δ 論法の画面
# ---------------------------------------------------------
elif menu == "ε-δ 論法":
    st.title("解析学：ε-δ 論法の視覚化")
    
    mode = st.sidebar.radio("関数を選択", ["連続 (x^2)", "不連続 (Step)"])
    epsilon = st.sidebar.slider("ε (縦の許容誤差)", 0.05, 1.5, 0.5)

    a = 1.0 
    x = np.linspace(0, 2, 400)

    if mode == "連続 (x^2)":
        L = a**2
        y = x**2
        delta = np.sqrt(L + epsilon) - a
        st.success(f"連続：ε={epsilon} に対して δ={delta:.3f} が見つかります。")
    else:
        L = 1.0
        y = np.where(x < a, x, x + 1)
        delta = 0.2 
        st.error("不連続：ε を小さくすると、グラフが帯から飛び出します！")

    fig, ax = plt.subplots()
    ax.plot(x, y, 'ko', markersize=1)
    ax.axhspan(L - epsilon, L + epsilon, color="yellow", alpha=0.3)
    ax.axvspan(a - delta, a + delta, color="blue", alpha=0.2)
    ax.set_ylim(0, 3)
    st.pyplot(fig)