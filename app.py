import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from mpl_toolkits.mplot3d import Axes3D

# サイドバーで「どの機能を使うか」を選べるようにする
st.sidebar.title("Calculus Menu")
menu = st.sidebar.selectbox("Select Topic", ["Taylor Series", "Epsilon-Delta", "Double Integral", "Gradient & Contour", "Line Integral"])

# ---------------------------------------------------------
# 1. テイラー展開の画面
# ---------------------------------------------------------
if menu == "Taylor Series":
    st.title("Calculus: Taylor Series Visualization")
    
    target_func = st.sidebar.selectbox("Select Function", ["sin(x)", "cos(x)", "exp(x)"])
    n = st.sidebar.slider("Approximation Degree (n)", 0, 15, 1)

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
elif menu == "Epsilon-Delta":
    st.title("Calculus: Epsilon-Delta Visualization")
    
    mode = st.sidebar.radio("Select Function", ["Continuous (x^2)", "Discontinuous (Step)"])
    epsilon = st.sidebar.slider("ε (Vertical Tolerance)", 0.05, 1.5, 0.5)

    a = 1.0 
    x = np.linspace(0, 2, 400)

    if mode == "Continuous (x^2)":
        L = a**2
        y = x**2
        delta = np.sqrt(L + epsilon) - a
        st.success(f"Continuous: For ε={epsilon}, we find δ={delta:.3f}.")
    else:
        L = 1.0
        y = np.where(x < a, x, x + 1)
        delta = 0.2 
        st.error("Discontinuous: If ε is small, the graph jumps out of the band!")

    fig, ax = plt.subplots()
    ax.plot(x, y, 'ko', markersize=1)
    ax.axhspan(L - epsilon, L + epsilon, color="yellow", alpha=0.3)
    ax.axvspan(a - delta, a + delta, color="blue", alpha=0.2)
    ax.set_ylim(0, 3)
    st.pyplot(fig)

# ---------------------------------------------------------
# 3. 重積分とリーマン和の画面
# ---------------------------------------------------------
elif menu == "Double Integral":
    st.title("Calculus: Double Integral & Riemann Sum Visualization")

    # 領域の選択
    col_x, col_y = st.columns(2)
    x_min, x_max = col_x.slider("x Range", -3.0, 3.0, (-1.0, 1.0))
    y_min, y_max = col_y.slider("y Range", -3.0, 3.0, (-1.0, 1.0))

    st.markdown(f"Volume approximation of $z = f(x, y) = x^2 + y^2$ (Region $[{x_min}, {x_max}] \\times [{y_min}, {y_max}]$)")

    n = st.sidebar.slider("Partitions (n)", 2, 20, 5)

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

    st.subheader("Calculation Results & Process")

    with st.expander("Show Calculation Details", expanded=True):
        st.markdown("**1. Theoretical Value (Double Integral)**")
        
        # f-stringの複雑なネストを避けるために変数を定義
        t1_str = f"{term1:.4f}"
        t2_str = f"{term2:.4f}"
        ve_str = f"{volume_exact:.4f}"
        
        st.latex(rf"""
            \begin{{aligned}}
            V &= \int_{{{y_min}}}^{{{y_max}}} \int_{{{x_min}}}^{{{x_max}}} (x^2 + y^2) \, dx \, dy \\
              &= \left[ \frac{{x^3}}{{3}}y + x\frac{{y^3}}{{3}} \right]_{{ {x_min}, {y_min} }}^{{ {x_max}, {y_max} }} \\
              &= \underbrace{{ {t1_str} }}_{{\text{{Contribution of x}}}} + \underbrace{{ {t2_str} }}_{{\text{{Contribution of y}}}} \\
              &= {ve_str}
            \end{{aligned}}
        """)
        
        st.markdown(f"**2. Riemann Sum ($n={n}$)**")
        
        dx_str = f"{dx:.4f}"
        dy_str = f"{dy:.4f}"
        sh_str = f"{sum_heights:.4f}"
        va_str = f"{volume_approx:.4f}"
        
        st.latex(rf"""
            \begin{{aligned}}
            \Delta x &= {dx_str}, \quad \Delta y = {dy_str} \\
            S_n &\approx \Delta x \Delta y \sum_{{i,j}} (x_i^2 + y_j^2) \\
            &= {dx_str} \times {dy_str} \times \underbrace{{ {sh_str} }}_{{\text{{Sum of Heights}}}} \\
            &= {va_str}
            \end{{aligned}}
        """)

    col1, col2 = st.columns(2)
    col1.metric("Theoretical (Integral)", f"{volume_exact:.4f}")
    col2.metric(f"Riemann Sum (n={n})", f"{volume_approx:.4f}", delta=f"{volume_approx - volume_exact:.4f}")

    ax.bar3d(x_flat, y_flat, z_flat, dx_flat, dy_flat, dz_flat, shade=True, color='skyblue', edgecolor='black', alpha=0.6)
    
    ax.set_title(f"Riemann Sum (n={n})")
    ax.set_zlim(0, 3)
    st.pyplot(fig)

# ---------------------------------------------------------
# 4. 勾配（グラディエント）と等高線の画面
# ---------------------------------------------------------
elif menu == "Gradient & Contour":
    st.title("Calculus: Gradient & Contour Visualization")
    st.markdown(r"Visualizing gradient $\nabla f$ of $f(x, y) = \sin(x) + \cos(y)$.")

    # 操作用スライダー
    col_param1, col_param2 = st.columns(2)
    cx = col_param1.slider("x Coordinate", -3.0, 3.0, 0.0, 0.1)
    cy = col_param2.slider("y Coordinate", -3.0, 3.0, 0.0, 0.1)
    
    lr = st.sidebar.slider("Learning Rate (η)", 0.01, 0.5, 0.1, 0.01)
    run_anim = st.button("Start Gradient Descent (Animation)")

    # データ準備
    x = np.linspace(-3.5, 3.5, 100)
    y = np.linspace(-3.5, 3.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) + np.cos(Y)

    # 描画ロジックを関数化（アニメーションで再利用するため）
    def draw_plots(curr_x, curr_y, calc_log=None):
        # 現在地点と勾配の計算
        curr_z = np.sin(curr_x) + np.cos(curr_y)
        g_x = np.cos(curr_x)
        g_y = -np.sin(curr_y)

        col1, col2 = st.columns(2)

        # 左：3D曲面グラフ
        with col1:
            st.subheader("3D Surface")
            fig1 = plt.figure(figsize=(6, 6))
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
            ax1.scatter(curr_x, curr_y, curr_z, color='red', s=100, label='Current Point')
            ax1.set_title(f"z = {curr_z:.2f}")
            st.pyplot(fig1)
            plt.close(fig1)

        # 右：等高線図と勾配ベクトル
        with col2:
            st.subheader("Contour & Gradient Vector")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
            fig2.colorbar(contour, ax=ax2)
            
            ax2.quiver(curr_x, curr_y, g_x, g_y, color='red', scale=5, width=0.02, label=r'$\nabla f$')
            ax2.plot(curr_x, curr_y, 'ro', markersize=8)
            
            ax2.set_title(f"Gradient: ({g_x:.2f}, {g_y:.2f})")
            ax2.set_aspect('equal')
            ax2.legend()
            st.pyplot(fig2)
            plt.close(fig2)
        
        st.info(f"Current Pos: $({curr_x:.3f}, {curr_y:.3f})$  Gradient: $({g_x:.3f}, {g_y:.3f})$")

        if calc_log:
            st.markdown("##### Gradient Descent Process")
            ox, oy = calc_log["old_x"], calc_log["old_y"]
            gx, gy = calc_log["grad_x"], calc_log["grad_y"]
            eta = calc_log["lr"]
            
            st.latex(rf"""
            \begin{{aligned}}
            x_{{new}} &= {ox:.3f} - {eta} \cdot ({gx:.3f}) = {curr_x:.3f} \\
            y_{{new}} &= {oy:.3f} - {eta} \cdot ({gy:.3f}) = {curr_y:.3f}
            \end{{aligned}}
            """)

    # アニメーション表示用のプレースホルダー
    plot_placeholder = st.empty()

    if run_anim:
        curr_x, curr_y = cx, cy
        for _ in range(30):  # 30ステップ実行
            # 更新前の値を保持
            old_x, old_y = curr_x, curr_y
            
            # 勾配の計算 (更新前)
            g_x_old = np.cos(old_x)
            g_y_old = -np.sin(old_y)
            
            # 更新
            curr_x -= lr * g_x_old
            curr_y -= lr * g_y_old
            
            # 計算ログ
            calc_log = {
                "old_x": old_x, "old_y": old_y,
                "grad_x": g_x_old, "grad_y": g_y_old,
                "lr": lr
            }
            
            with plot_placeholder.container():
                draw_plots(curr_x, curr_y, calc_log)
            time.sleep(0.3) # 計算が見えるように少しゆっくりに
    else:
        with plot_placeholder.container():
            draw_plots(cx, cy)

    # 数式と値の表示
    st.latex(r"\nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right) = (\cos x, -\sin y)")

# ---------------------------------------------------------
# 5. 線積分の画面
# ---------------------------------------------------------
elif menu == "Line Integral":
    st.title("Calculus: Line Integral Visualization")
    st.markdown(r"Line integral $\int_C \mathbf{F} \cdot d\mathbf{r}$ in vector field $\mathbf{F}(x, y) = (-y, x)$")

    # 経路の設定
    path_type = st.sidebar.radio("Path Shape", ["Circle (Centered at Origin)", "Circle (Shifted Center)"])
    theta_max = st.sidebar.slider("End Angle (rad)", 0.1, 2 * np.pi, np.pi)

    # ベクトル場の準備
    x_range = np.linspace(-3, 3, 20)
    y_range = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x_range, y_range)
    U = -Y
    V = X

    # 描画
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 背景のベクトル場
    ax.quiver(X, Y, U, V, color='gray', alpha=0.3, label=r'$\mathbf{F}=(-y, x)$')

    # 経路の計算
    t = np.linspace(0, theta_max, 100)
    
    if path_type == "Circle (Centered at Origin)":
        rx = np.cos(t)
        ry = np.sin(t)
    else:
        # 中心 (1.5, 0), 半径 1
        rx = 1.5 + np.cos(t)
        ry = np.sin(t)

    # 線積分の計算
    # 1. 経路上のベクトル場 F(r(t))
    Fx = -ry
    Fy = rx
    
    # 2. 接ベクトル r'(t) (数値微分)
    drx = np.gradient(rx, t)
    dry = np.gradient(ry, t)
    
    # 3. 仕事率 (Power) = F . v
    power = Fx * drx + Fy * dry
    
    # 4. 線積分 (仕事) = int (power) dt
    # NumPy 2.0以降の互換性対応: np.trapz は削除されたため np.trapezoid を使用
    work = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(power, t)
    
    # 5. 可視化用の内積値 (F . T)
    speed = np.sqrt(drx**2 + dry**2)
    speed[speed == 0] = 1.0 # ゼロ除算回避
    f_dot_t = power / speed

    # 経路の描画 (色で内積の正負を表示: 赤=正, 青=負)
    sc = ax.scatter(rx, ry, c=f_dot_t, cmap='coolwarm', s=30, vmin=-2, vmax=2, label='Path')
    fig.colorbar(sc, ax=ax, label=r"$\mathbf{F} \cdot \mathbf{T}$ (Dot Product)")
    
    # 始点と終点
    ax.plot(rx[0], ry[0], 'go', markersize=8, label='Start')
    ax.plot(rx[-1], ry[-1], 'ro', markersize=8, label='End')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f"Line Integral: {work:.4f}")
    
    st.pyplot(fig)
    
    # 理論値の計算と過程の表示
    if path_type == "Circle (Centered at Origin)":
        theory_work = theta_max
        latex_steps = rf"""
        \begin{{aligned}}
        \mathbf{{r}}(t) &= (\cos t, \sin t) \\
        \mathbf{{F}} \cdot \mathbf{{r}}'(t) &= (-\sin t)(-\sin t) + (\cos t)(\cos t) = 1 \\
        W &= \int_0^{{{theta_max:.2f}}} 1 \, dt = [t]_0^{{{theta_max:.2f}}} = {theory_work:.4f}
        \end{{aligned}}
        """
    else:
        theory_work = theta_max + 1.5 * np.sin(theta_max)
        latex_steps = rf"""
        \begin{{aligned}}
        \mathbf{{r}}(t) &= (1.5 + \cos t, \sin t) \\
        \mathbf{{F}} \cdot \mathbf{{r}}'(t) &= 1 + 1.5\cos t \\
        W &= \int_0^{{{theta_max:.2f}}} (1 + 1.5\cos t) \, dt = [t + 1.5\sin t]_0^{{{theta_max:.2f}}} \\
          &= {theta_max:.4f} + 1.5({np.sin(theta_max):.4f}) = {theory_work:.4f}
        \end{{aligned}}
        """

    st.subheader("Calculation Results & Process")
    with st.expander("Show Calculation Details", expanded=True):
        st.latex(latex_steps)

    col1, col2 = st.columns(2)
    col1.metric("Theoretical Value", f"{theory_work:.4f}")
    col2.metric("Numerical Integral", f"{work:.4f}", delta=f"{work - theory_work:.4f}")

    st.info("Red indicates 'Tailwind (Positive Work)', Blue indicates 'Headwind (Negative Work)'.")
    