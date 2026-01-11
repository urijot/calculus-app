import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

st.title("解析学：テイラー展開の視覚化")
st.write("次数を増やすほど、多項式が元の関数に近づく様子を観察しましょう。")

# 1. パラメータ設定（サイドバーに配置）
n = st.sidebar.slider("近似の次数 (n)", 0, 15, 1)

# 2. 数学的定義（sin(x)のテイラー展開）
def taylor_sin(x, n):
    y = 0
    for i in range(n + 1):
        # sin(x) = Σ (-1)^i * x^(2i+1) / (2i+1)!
        term = ((-1)**i) * (x**(2*i+1)) / factorial(2*i+1)
        y += term
    return y

# 3. データ作成
x = np.linspace(-10, 10, 400)
y_orig = np.sin(x)
y_approx = taylor_sin(x, n)

# 4. グラフ描画
fig, ax = plt.subplots()
ax.plot(x, y_orig, label="Original sin(x)", color="gray", linestyle="--")
ax.plot(x, y_approx, label=f"Taylor Approx (n={n})", color="red", linewidth=2)
ax.set_ylim(-3, 3)
ax.legend()
ax.grid(True)

st.pyplot(fig)