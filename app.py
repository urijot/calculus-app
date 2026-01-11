import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from mpl_toolkits.mplot3d import Axes3D

# サイドバーで「どの機能を使うか」を選べるようにする
st.sidebar.title("解析学メニュー")
menu = st.sidebar.selectbox("学習テーマを選択", ["テイラー展開", "ε-δ 論法", "重積分"])

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

# ---------------------------------------------------------
# 3. 重積分とリーマン和の画面
# ---------------------------------------------------------
elif menu == "重積分":
    st.title("解析学：重積分とリーマン和の視覚化")

    # 領域の選択
    col_x, col_y = st.columns(2)
    x_min, x_max = col_x.slider("x の範囲", -3.0, 3.0, (-1.0, 1.0))
    y_min, y_max = col_y.slider("y の範囲", -3.0, 3.0, (-1.0, 1.0))

    st.markdown(f"関数 $z = f(x, y) = x^2 + y^2$ の体積近似（領域 $[{x_min}, {x_max}] \\times [{y_min}, {y_max}]$）")

    n = st.sidebar.slider("分割数 (n)", 2, 20, 5)

    # 3Dプロットの準備
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 関数 z = x^2 + y^2 のワイヤーフレーム（滑らかな曲面）
    x_range = np.linspace(x_min, x_max, 50)
    y_range = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + Y**2
    ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.3, rstride=5, cstride=5)

    # 2. リーマン和（直方体）の描画
    dx = (x_max - x_min) / n
    dy = (y_max - y_min) / n
    
    # 格子の左下の座標を生成
    x_vals = np.linspace(x_min, x_max - dx, n)
    y_vals = np.linspace(y_min, y_max - dy, n)
    X_bar, Y_bar = np.meshgrid(x_vals, y_vals)
    
    x_flat = X_bar.flatten()
    y_flat = Y_bar.flatten()
    z_flat = np.zeros_like(x_flat)
    
    dx_flat = np.full_like(x_flat, dx)
    dy_flat = np.full_like(y_flat, dy)
    dz_flat = x_flat**2 + y_flat**2
    
    # 体積の計算結果を表示
    volume_approx = np.sum(dx * dy * dz_flat)
    
    # 理論値の計算: Int(x^2 + y^2) dx dy
    # = (x_max^3 - x_min^3)/3 * (y_max - y_min) + (x_max - x_min) * (y_max^3 - y_min^3)/3
    term1 = (x_max**3 - x_min**3) / 3.0 * (y_max - y_min)
    term2 = (x_max - x_min) * (y_max**3 - y_min**3) / 3.0
    volume_exact = term1 + term2

    # リーマン和の高さの合計（表示用）
    sum_heights = np.sum(dz_flat)

    st.subheader("計算結果と過程")

    with st.expander("計算過程の詳細を表示", expanded=True):
        st.markdown("**1. 理論値（二重積分）の計算**")
        
        # f-stringの複雑なネストを避けるために変数を定義
        t1_str = f"{term1:.4f}"
        t2_str = f"{term2:.4f}"
        ve_str = f"{volume_exact:.4f}"
        
        st.latex(rf"""
            \begin{{aligned}}
            V &= \int_{{{y_min}}}^{{{y_max}}} \int_{{{x_min}}}^{{{x_max}}} (x^2 + y^2) \, dx \, dy \\
              &= \left[ \frac{{x^3}}{{3}}y + x\frac{{y^3}}{{3}} \right]_{{ {x_min}, {y_min} }}^{{ {x_max}, {y_max} }} \\
              &= \underbrace{{ {t1_str} }}_{{\text{{xの寄与}}}} + \underbrace{{ {t2_str} }}_{{\text{{yの寄与}}}} \\
              &= {ve_str}
            \end{{aligned}}
        """)
        
        st.markdown(f"**2. リーマン和（$n={n}$）の計算**")
        
        dx_str = f"{dx:.4f}"
        dy_str = f"{dy:.4f}"
        sh_str = f"{sum_heights:.4f}"
        va_str = f"{volume_approx:.4f}"
        
        st.latex(rf"""
            \begin{{aligned}}
            \Delta x &= {dx_str}, \quad \Delta y = {dy_str} \\
            S_n &\approx \Delta x \Delta y \sum_{{i,j}} (x_i^2 + y_j^2) \\
            &= {dx_str} \times {dy_str} \times \underbrace{{ {sh_str} }}_{{\text{{高さの総和}}}} \\
            &= {va_str}
            \end{{aligned}}
        """)

    col1, col2 = st.columns(2)
    col1.metric("理論値 (積分)", f"{volume_exact:.4f}")
    col2.metric(f"リーマン和 (n={n})", f"{volume_approx:.4f}", delta=f"{volume_approx - volume_exact:.4f}")

    ax.bar3d(x_flat, y_flat, z_flat, dx_flat, dy_flat, dz_flat, shade=True, color='skyblue', edgecolor='black', alpha=0.6)
    
    ax.set_title(f"Riemann Sum (n={n})")
    ax.set_zlim(0, 3)
    st.pyplot(fig)

    