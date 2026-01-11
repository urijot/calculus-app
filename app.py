import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

st.title("解析学：テイラー展開の視覚化")

# 1. 関数の選択とパラメータ設定
target_func = st.sidebar.selectbox("関数を選択", ["sin(x)", "cos(x)", "exp(x)"])
n = st.sidebar.slider("近似の次数 (n)", 0, 15, 1)

# 2. 数学的定義（選択された関数に応じて計算を切り替え）
def get_taylor_series(x, n, func_type):
    y = np.zeros_like(x)
    if func_type == "sin(x)":
        for i in range(n + 1):
            y += ((-1)**i) * (x**(2*i+1)) / factorial(2*i+1)
        return y, np.sin(x)
    
    elif func_type == "cos(x)":
        for i in range(n + 1):
            y += ((-1)**i) * (x**(2*i)) / factorial(2*i)
        return y, np.cos(x)
    
    elif func_type == "exp(x)":
        for i in range(n + 1):
            y += (x**i) / factorial(i)
        return y, np.exp(x)

# 3. データ作成
x = np.linspace(-5, 5, 400)
y_approx, y_orig = get_taylor_series(x, n, target_func)

# 4. グラフ描画
fig, ax = plt.subplots()
ax.plot(x, y_orig, label=f"Original {target_func}", color="gray", linestyle="--")
ax.plot(x, y_approx, label=f"Taylor Approx (n={n})", color="blue", linewidth=2)
ax.set_ylim(-3, 3)
ax.legend()
ax.grid(True)

st.pyplot(fig)