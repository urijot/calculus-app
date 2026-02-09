# Calculus Visualization App 📐


**🚀 Live Demo:** https://calculus-app-chnyhjrstvqsp2euxgd4jy.streamlit.app/

## 💡 Motivation

Coming from a non-STEM background, I built this project as a means to teach myself Calculus. I found that visualizing abstract equations was key to grasping complex topics like Taylor series and line integrals. I created this interactive tool to solidify my own understanding and to help other students and developers build a geometric intuition, bridging the gap between theory and visualization.

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