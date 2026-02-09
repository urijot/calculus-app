# Calculus Visualization App 📐
An interactive tool for STEM students to visually understand challenging concepts in Calculus and Analysis.

**🚀 Live Demo:** https://calculus-app-chnyhjrstvqsp2euxgd4jy.streamlit.app/

## ✨ Core Features

This app visualizes abstract mathematical definitions dynamically.

1. **Taylor Series**
   - Observe how polynomials approximate functions as the degree $n$ increases.

2. **ε-δ Definition**
   - Dynamically adjust $\epsilon$ to see how $\delta$ responds.

3. **Double Integral & Riemann Sum**
   - Visualize volume approximation using rectangular prisms in 3D space.

4. **Gradient & Contour**
   - Visualize how the gradient vector $\nabla f$ is always perpendicular to contour lines.

5. **Line Integral**
   - Calculate "work" done by a vector field along a custom path.

## 🛠 Tech Stack

- **Language**: Python 3.9+
- **Framework**: [Streamlit](https://streamlit.io/)
- **Libraries**: NumPy, SciPy, Matplotlib

## 🚀 Setup & Run

```bash
# Install dependencies
python3 -m pip install streamlit numpy matplotlib scipy

# Run the app
python3 -m streamlit run app.py