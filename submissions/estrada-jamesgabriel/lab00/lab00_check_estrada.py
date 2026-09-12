"""
Lab Assignment 0 - platform verification
Name           : James Gabriel Estrada
Student number : TUPM-23-4973
"""
import os
import sys

import numpy as np
import scipy
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt

# ---- 1. report the versions --------------------------------------------
print("Python     :", sys.version.split()[0])
print("NumPy      :", np.__version__)
print("SciPy      :", scipy.__version__)
print("Matplotlib :", matplotlib.__version__)

# ---- 2. a NumPy calculation --------------------------------------------
# 200 points from 0 to 2*pi, then the sine of every one of them at once.
x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)
print("\nmean of sin(x) over one period :", round(float(y.mean()), 12))

# ---- 3. a SciPy cross-check --------------------------------------------
# The integral of sin(x) from 0 to pi is exactly 2.
area, _ = integrate.quad(np.sin, 0, np.pi)
print("integral of sin(x) from 0 to pi :", round(area, 12), "(exact value 2)")

# ---- 4. a Matplotlib figure --------------------------------------------
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 150})

fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(x, y, color="#0b5fa5", lw=1.6, label="sin(x)")
ax.plot(x, np.cos(x), color="#8a5b00", lw=1.2, ls="--", label="cos(x)")
ax.set_xlabel("x (radians)")
ax.set_ylabel("value")
ax.set_title("Lab Assignment 0 - platform check")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("figures/lab00_check.png")

print("\nfigure written to figures/lab00_check.png")
print("PLATFORM READY.")