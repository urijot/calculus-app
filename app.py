import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("解析学 視覚化アプリ")
x = np.linspace(-10, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)