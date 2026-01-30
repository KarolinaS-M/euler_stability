import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# App title and description
# ======================================================

st.title("Explicit vs Implicit Euler Method")
st.write(
    r"""
    This application compares the **explicit (forward) Euler method** and the
    **implicit (backward) Euler method** for the linear differential equation

    \[
    x'(t) = \lambda x(t), \qquad x(0) = 1 .
    \]

    The analytical solution is shown in blue in both panels.
    """
)

# ======================================================
# Parameters: sliders + input boxes
# ======================================================

st.sidebar.header("Model parameters")

lam_slider = st.sidebar.slider(
    "λ (slider)",
    min_value=-10.0,
    max_value=1.0,
    value=-2.0,
    step=0.1
)

h_slider = st.sidebar.slider(
    "Step size h (slider)",
    min_value=0.01,
    max_value=2.5,
    value=0.4,
    step=0.01
)

lam_input = st.sidebar.number_input(
    "λ (manual input)",
    value=lam_slider
)

h_input = st.sidebar.number_input(
    "h (manual input)",
    value=h_slider,
    min_value=0.0001
)

lam = lam_input
h = h_input

x0 = 1.0
T = 5.0

# ======================================================
# Stability condition (explicit Euler)
# ======================================================

if lam != 0:
    h_crit = 2.0 / abs(lam)
    explicit_stable = h < h_crit and lam < 0
else:
    explicit_stable = False

if explicit_stable:
    st.sidebar.success("Explicit Euler stable:  h < 2 / |λ|")
else:
    st.sidebar.warning("Explicit Euler unstable:  h ≥ 2 / |λ|")

# ======================================================
# Analytical solution
# ======================================================

t = np.linspace(0, T, 500)
x_exact = x0 * np.exp(lam * t)

# ======================================================
# Euler methods
# ======================================================

def explicit_euler(lam, h, x0, N):
    x = np.zeros(N + 1)
    x[0] = x0
    for n in range(N):
        x[n + 1] = (1 + h * lam) * x[n]
    return x

def backward_euler(lam, h, x0, N):
    x = np.zeros(N + 1)
    x[0] = x0
    for n in range(N):
        x[n + 1] = x[n] / (1 - h * lam)
    return x

N = int(T / h)
t_disc = np.linspace(0, N * h, N + 1)

x_explicit = explicit_euler(lam, h, x0, N)
x_backward = backward_euler(lam, h, x0, N)

# ======================================================
# Plot: two panels
# ======================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# --- Left panel: Explicit Euler ---
axes[0].plot(t, x_exact, color="blue", linewidth=2, label="Analytical solution")
axes[0].plot(t_disc, x_explicit, "o--", color="red", label="Explicit Euler")
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_title("Explicit Euler")
axes[0].set_xlabel("Time t")
axes[0].set_ylabel("x(t)")
axes[0].grid(True)
axes[0].legend()

# --- Right panel: Backward Euler ---
axes[1].plot(t, x_exact, color="blue", linewidth=2, label="Analytical solution")
axes[1].plot(t_disc, x_backward, "o-", color="green", label="Backward Euler")
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_title("Backward (Implicit) Euler")
axes[1].set_xlabel("Time t")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
st.pyplot(fig)