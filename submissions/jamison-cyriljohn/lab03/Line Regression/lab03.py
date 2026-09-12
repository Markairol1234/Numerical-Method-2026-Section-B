"""
Lab 03: Real-World Data Linear Regression
Numerical Methods, Section 3H
Student: Cyril John T. Jamison

Data source: World Health Organization (WHO)
  x = Physicians (per 1,000 people), 2019
      World Health Organization (Global Health Workforce Statistics), via
      World Bank World Development Indicators.
      https://ourworldindata.org/grapher/physicians-per-1000-people
  y = Life expectancy at birth, both sexes (years), 2019
      World Health Organization - Global Health Observatory (GHO)
      https://www.who.int/data/gho/data/indicators/indicator-details/GHO/life-expectancy

Both variables were retrieved for the same 20 countries and the same year
(2019, the last full year before COVID-19 disrupted life-expectancy trends).
"""

import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# SECTION 1: DATA - physicians per 1,000 people (x) and life
#            expectancy at birth in years (y), 20 countries, 2019
# =================================================================
countries = [
    "Afghanistan", "Albania", "Argentina", "Australia", "Bangladesh",
    "Belgium", "Brazil", "Canada", "Chile", "China",
    "Colombia", "Egypt", "Germany", "India",
    "Indonesia", "Italy", "Malaysia", "Mexico", "Nepal",
    "Philippines",
]

# x: physicians per 1,000 people
x = np.array([
    0.214, 1.645, 5.250, 3.808, 0.630,
    3.424, 2.351, 2.418, 2.627, 2.255,
    2.317, 0.697, 4.373, 0.889,
    0.462, 4.024, 2.021, 2.460, 0.815,
    0.732,
])

# y: life expectancy at birth, years
y = np.array([
    61.224, 77.941, 77.016, 82.642, 73.950,
    81.609, 75.478, 82.022, 81.028, 77.313,
    77.954, 71.590, 80.974, 70.727,
    71.401, 82.985, 74.684, 75.834, 71.361,
    69.429,
])

n = len(x)

# =================================================================
# SECTION 2: LEAST-SQUARES COEFFICIENTS a0 AND a1
#            Fit the model  y = a0 + a1*x  using the normal
#            equations (no Excel trendline, no numpy.polyfit)
# =================================================================
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x ** 2)

a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
a0 = (sum_y - a1 * sum_x) / n

print("SECTION 2: Least-squares coefficients")
print(f"  a0 (intercept) = {a0:.4f}")
print(f"  a1 (slope)     = {a1:.4f}")
print(f"  Regression equation: y = {a0:.4f} + {a1:.4f} x")
print()

# =================================================================
# SECTION 3: PREDICTED VALUES AND RESIDUALS
#            y_pred = a0 + a1*x for every data point;
#            residual = observed y minus predicted y
# =================================================================
y_pred = a0 + a1 * x
residuals = y - y_pred

print("SECTION 3: Predicted values and residuals")
print(f"{'Country':<14}{'x':>8}{'y (obs)':>10}{'y (pred)':>10}{'residual':>10}")
for c, xi, yi, ypi, ri in zip(countries, x, y, y_pred, residuals):
    print(f"{c:<14}{xi:>8.3f}{yi:>10.3f}{ypi:>10.3f}{ri:>10.3f}")
print()

# =================================================================
# SECTION 4: GOODNESS OF FIT - Sr (SSE), r^2, standard error sy/x
# =================================================================
St = np.sum((y - np.mean(y)) ** 2)   # total sum of squares
Sr = np.sum(residuals ** 2)          # sum of squared errors (SSE)
r2 = 1 - Sr / St                     # coefficient of determination
syx = np.sqrt(Sr / (n - 2))          # standard error of the estimate

print("SECTION 4: Goodness of fit")
print(f"  Sr (SSE)             = {Sr:.4f}")
print(f"  r^2                  = {r2:.4f}")
print(f"  standard error s_y/x = {syx:.4f} years")
print()

# =================================================================
# SECTION 5: PREDICTION FOR AN x NOT IN THE DATASET
# =================================================================
x_new = 3.0  # physicians per 1,000 people
y_new = a0 + a1 * x_new

print("SECTION 5: Prediction for a new x")
print(f"  x = {x_new} physicians/1,000  ->  predicted y = {y_new:.2f} years")
print()

# =================================================================
# SECTION 6: PLOTS - data with fitted line, and residual plot
# =================================================================
# Data + fitted line
plt.figure(figsize=(6, 4.5))
plt.scatter(x, y, color="#1f4e79", label="Country data (2019)")
x_line = np.linspace(min(x), max(x), 100)
plt.plot(x_line, a0 + a1 * x_line, color="#c8963e",
         label=f"y = {a0:.2f} + {a1:.2f}x")
plt.xlabel("Physicians (per 1,000 people)")
plt.ylabel("Life expectancy at birth (years)")
plt.title("Life Expectancy vs. Physician Density (WHO, 2019)")
plt.legend()
plt.tight_layout()
plt.savefig("fit_plot.png", dpi=150)
plt.close()

# Residual plot
plt.figure(figsize=(6, 4.5))
plt.scatter(x, residuals, color="#1f4e79")
plt.axhline(0, color="#c8963e", linewidth=1.5)
plt.xlabel("Physicians (per 1,000 people)")
plt.ylabel("Residual (years)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150)
plt.close()

print("SECTION 6: Plots saved as fit_plot.png and residual_plot.png")
