import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ======================================================
# App title and description
# ======================================================

st.title("Absolute Stability of the Euler Method")
st.write(
    r"""
We consider the linear initial--value problem  
\[
x'(t)=\lambda x(t), \qquad x(0)=1,
\]
and compare the analytical solution with the numerical solution obtained
using the explicit Euler method.
"""
)

# ======================================================
# Parameter selection
# ======================================================

st.sidebar.header("Model parameters")

# --- lambda ---
lam_slider = st.sidebar.slider(
    "λ (slider)",
    min_value=-10.0,
    max_value=10.0,
    value=-2.0,
    step=0.1
)

lam_input = st.sidebar.number_input(
    "λ (manual input)",
    value=lam_slider
)

lam = lam_input

# --- step size h ---
h_slider = st.sidebar.slider(
    "Step size h (slider)",
    min_value=0.01,
    max_value=2.0,
    value=0.4,
    step=0.01
)

h_input = st.sidebar.number_input(
    "Step size h (manual input)",
    value=h_slider,
    min_value=0.0
)

h = h_input

# ======================================================
# Fixed settings
# ======================================================

x0 = 1.0
T = 5.0

# ======================================================
# Absolute stability condition
# ======================================================

if lam != 0:
    h_crit = 2.0 / abs(lam)
    stable = h < h_crit
else:
    stable = False
    h_crit = np.inf

if stable:
    st.success("Absolute stability condition satisfied:  h < 2 / |λ|")
else:
    st.error("Absolute stability condition NOT satisfied:  h ≥ 2 / |λ|")

# ======================================================
# Analytical solution
# ======================================================

t = np.linspace(0, T, 400)
x_exact = x0 * np.exp(lam * t)

# ======================================================
# Explicit Euler method
# ======================================================

def euler(lam, h, x0, N):
    x = np.zeros(N + 1)
    x[0] = x0
    for n in range(N):
        x[n + 1] = (1 + h * lam) * x[n]
    return x

if h > 0:
    N = int(T / h)
    t_euler = np.linspace(0, N * h, N + 1)
    x_euler = euler(lam, h, x0, N)
else:
    t_euler = np.array([0.0])
    x_euler = np.array([x0])

# ======================================================
# Plot
# ======================================================

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(t, x_exact, linewidth=2, label="Analytical solution")
ax.plot(t_euler, x_euler, "o--", label="Euler method")
ax.axhline(0, color="black", linewidth=0.5)

ax.set_xlabel("Time t")
ax.set_ylabel("x(t)")
ax.set_title(r"Analytical vs numerical solution for $x'(t)=\lambda x(t)$")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ======================================================
# Additional information
# ======================================================

st.markdown(
    r"""
**Interpretation.**  
For $\lambda<0$, the continuous--time system is asymptotically stable.
The explicit Euler method reproduces this behavior only if the step size $h$
satisfies the absolute stability condition $h<2/|\lambda|$.
"""
)