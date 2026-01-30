import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ======================================================
# Page setup
# ======================================================

st.set_page_config(page_title="RK Orders Comparison", layout="centered")

st.title("Comparison of RK1, RK2, and RK4 Methods")
st.markdown(
    r"""
This application compares the accuracy of three explicit time--stepping schemes for the linear
initial--value problem

\[
x'(t) = \lambda x(t), \qquad x(0) = 1.
\]

The analytical solution is compared with:
- Explicit Euler method (RK1),
- Heun’s method (RK2),
- Classical fourth--order Runge--Kutta method (RK4).

Vary the parameter $\lambda$ and the step size $h$ to observe how the numerical solutions
converge to the exact solution at different rates.
"""
)

# ======================================================
# Parameter controls
# ======================================================

st.sidebar.header("Model parameters")

use_sliders = st.sidebar.checkbox("Use sliders", value=True)

if use_sliders:
    lam = st.sidebar.slider("λ (lambda)", min_value=-5.0, max_value=-0.1, value=-2.0, step=0.1)
    h = st.sidebar.slider("Step size h", min_value=0.05, max_value=1.0, value=0.4, step=0.05)
else:
    lam = st.sidebar.number_input("λ (lambda)", value=-2.0, step=0.1)
    h = st.sidebar.number_input("Step size h", value=0.4, step=0.05)

T = st.sidebar.number_input("Final time T", value=5.0, step=0.5)

# ======================================================
# Time grids
# ======================================================

x0 = 1.0
N = int(T / h)

t_exact = np.linspace(0, T, 500)
t_num = np.linspace(0, N * h, N + 1)

# ======================================================
# Analytical solution
# ======================================================

x_exact = x0 * np.exp(lam * t_exact)
x_exact_num = x0 * np.exp(lam * t_num)

# ======================================================
# Numerical methods
# ======================================================

def euler(lam, h, x0, N):
    x = np.zeros(N + 1)
    x[0] = x0
    for n in range(N):
        x[n + 1] = x[n] + h * lam * x[n]
    return x

def heun(lam, h, x0, N):
    x = np.zeros(N + 1)
    x[0] = x0
    for n in range(N):
        k1 = lam * x[n]
        k2 = lam * (x[n] + h * k1)
        x[n + 1] = x[n] + 0.5 * h * (k1 + k2)
    return x

def rk4(lam, h, x0, N):
    x = np.zeros(N + 1)
    x[0] = x0
    for n in range(N):
        k1 = lam * x[n]
        k2 = lam * (x[n] + 0.5 * h * k1)
        k3 = lam * (x[n] + 0.5 * h * k2)
        k4 = lam * (x[n] + h * k3)
        x[n + 1] = x[n] + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x

# ======================================================
# Compute solutions
# ======================================================

x_euler = euler(lam, h, x0, N)
x_heun = heun(lam, h, x0, N)
x_rk4 = rk4(lam, h, x0, N)

# ======================================================
# Errors
# ======================================================

error_euler = np.max(np.abs(x_euler - x_exact_num))
error_heun = np.max(np.abs(x_heun - x_exact_num))
error_rk4 = np.max(np.abs(x_rk4 - x_exact_num))

# ======================================================
# Plot
# ======================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(t_exact, x_exact, color="blue", linewidth=2, label="Analytical solution")
ax.plot(t_num, x_euler, "o--", color="red", label="Explicit Euler (RK1)")
ax.plot(t_num, x_heun, "s--", color="purple", label="Heun method (RK2)")
ax.plot(t_num, x_rk4, "d--", color="magenta", linewidth=2.5, label="RK4")

ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Time t")
ax.set_ylabel("x(t)")
ax.set_title(r"Accuracy comparison for $x'(t)=\lambda x(t)$")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ======================================================
# Error report
# ======================================================

st.subheader("Global errors (max norm)")

st.markdown(
    f"""
- **Explicit Euler (RK1):** {error_euler:.3e}  
- **Heun method (RK2):** {error_heun:.3e}  
- **RK4:** {error_rk4:.3e}
"""
)

if error_euler > error_heun > error_rk4:
    st.success("Observed ordering:  error_Euler ≫ error_Heun ≫ error_RK4")
else:
    st.warning("Error ordering differs for this choice of λ and h.")