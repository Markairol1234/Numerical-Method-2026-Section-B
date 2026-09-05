"""
Simple Linear Regression by Least Squares - Philippines Population Example
Dataset: Philippines Population, Total (SP.POP.TOTL)
Source: World Bank WDI - Philippines (PHL) - compiled from PSA / UN World Population Prospects
        Philippine Statistics Authority (PSA) is underlying national source
URLs: https://data.worldbank.org/indicator/SP.POP.TOTL?locations=PH
      API: https://api.worldbank.org/v2/country/PH/indicator/SP.POP.TOTL?format=json&date=2010:2024
      PSA: https://psa.gov.ph/statistics/population-and-housing
Accessed: Sep 05 2026, lastupdated World Bank 2026-07-13

Independent variable x = Year [calendar year]
Dependent variable y = Total Population [persons] (also shown as millions for readability)
15 paired observations: 2010-2024 inclusive

Formulas (same as class):
    a1 = [n*sum(x*y) - sum(x)*sum(y)] / [n*sum(x^2) - (sum(x))^2]
    a0 = y_mean - a1 * x_mean
    Sr = sum((y - y_pred)^2)    # SSE
    St = sum((y - y_mean)^2)    # SST
    r2 = (St - Sr)/St
    sy/x = sqrt(Sr/(n-2))
"""

import matplotlib.pyplot as plt
import math

# --- DATA: 15 years ---
# World Bank API values for Philippines SP.POP.TOTL 2010-2024 (persons)
x = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
y = [96337125, 98248614, 100175512, 102076336, 103767130, 105312992, 106735719, 108119693, 109465287, 110804683, 112081264, 113100950, 113964338, 114891199, 115843670]  # persons
# For plots/table readability: millions
y_millions = [v/1e6 for v in y]

n = len(x)
print(f"n = {n} paired observations (Philippines)")

# Sums
sum_x = sum(x)
sum_y = sum(y)
sum_xy = sum(xi*yi for xi, yi in zip(x, y))
sum_x2 = sum(xi**2 for xi in x)
x_mean = sum_x / n
y_mean = sum_y / n

print(f"sum_x = {sum_x}")
print(f"sum_y = {sum_y} persons ({sum_y/1e6:.2f} million)")
print(f"sum_xy = {sum_xy}")
print(f"sum_x2 = {sum_x2}")
print(f"x_mean = {x_mean:.2f}")
print(f"y_mean = {y_mean:.2f} persons ({y_mean/1e6:.4f} million)")

# Least squares coefficients - persons per year
a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
a0 = y_mean - a1 * x_mean

# In millions for reporting convenience
a1_m = a1 / 1e6
a0_m = a0 / 1e6

print("\n--- Least squares fit: y = a0 + a1*x ---")
print(f"a1 (slope)     = {a1:.6f} persons/year = {a1_m:.6f} million/year")
print(f"a0 (intercept) = {a0:.6f} persons = {a0_m:.6f} million")
print(f"Equation (persons): y = {a0:.2f} + {a1:.2f}*x")
print(f"Equation (millions): y = {a0_m:.4f} + {a1_m:.6f}*x  [y in millions, x in year]")

# Centered form for interpretation: x' = Year - 2010
x_prime = [xi - 2010 for xi in x]
sum_xp = sum(x_prime)
sum_xp2 = sum(v**2 for v in x_prime)
sum_xpy = sum(xi*yi for xi, yi in zip(x_prime, y))
xp_mean = sum_xp / n
a1_prime = (n * sum_xpy - sum_xp * sum_y) / (n * sum_xp2 - sum_xp**2)
a0_prime = y_mean - a1_prime * xp_mean
print(f"\nCentered form x' = Year-2010:")
print(f"  y = {a0_prime:.2f} + {a1_prime:.2f}*(Year-2010)  [persons]")
print(f"  y = {a0_prime/1e6:.4f} + {a1_prime/1e6:.6f}*(Year-2010)  [millions]")
print(f"  -> intercept at 2010 = {a0_prime:,.0f} persons ({a0_prime/1e6:.2f} million)")

# Predicted values and residuals
y_pred = [a0 + a1*xi for xi in x]
y_pred_m = [v/1e6 for v in y_pred]
residuals = [yi - ypi for yi, ypi in zip(y, y_pred)]
residuals_m = [r/1e6 for r in residuals]

Sr = sum(r**2 for r in residuals)
St = sum((yi - y_mean)**2 for yi in y)
r2 = (St - Sr)/St
r = math.sqrt(r2) if r2 >=0 else 0
sy_x = math.sqrt(Sr/(n-2))

# Also in millions^2 for easier reading but Sr in persons^2 is standard
Sr_m = Sr / 1e12
St_m = St / 1e12
sy_x_m = sy_x / 1e6

print("\n--- Goodness of fit ---")
print(f"Sr (SSE) = {Sr:.2f} persons^2  ({Sr_m:.4f} million^2)")
print(f"St (SST) = {St:.2f} persons^2  ({St_m:.4f} million^2)")
print(f"r^2      = {r2:.6f}   (r = {r:.6f})")
print(f"sy/x     = {sy_x:.2f} persons = {sy_x_m:.4f} million")

# Residual table
print("\nYear  y_obs(m) y_pred(m) resid(m)  resid(persons)")
for xi, yi, ypi, ri in zip(x, y_millions, y_pred_m, residuals):
    print(f"{xi}  {yi:7.2f}   {ypi:7.2f}  {yi-ypi:+7.3f}   {ri:+10.0f}")

# Predictions for x not in dataset
for x_new in [2025, 2030, 2035]:
    y_new = a0 + a1*x_new
    print(f"\nPrediction Year {x_new}: y = {a0:.2f} + {a1:.2f}*{x_new} = {y_new:,.0f} persons ({y_new/1e6:.2f} million)")

# --- PLOTS ---
# 1. Data + fitted line (in millions for y-axis readability)
plt.figure(figsize=(8,5))
plt.scatter(x, y_millions, color='blue', label='Observed (World Bank/PSA)', zorder=3)
# Smooth line from 2010 to 2035 to show extrapolation
x_line = list(range(min(x), 2036))
y_line_m = [(a0 + a1*xi)/1e6 for xi in x_line]
plt.plot(x_line, y_line_m, color='red', label=f'Fit: y = {a0_m:.2f} + {a1_m:.4f}x')
plt.xlabel('Year [calendar year]')
plt.ylabel('Population [millions of persons]')
plt.title('Philippines Population vs Year\nWorld Bank (PSA source) 2010-2024 with Least Squares Fit')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('figures/lab03_espenilla_fit.png', dpi=300)
print("\nSaved: figures/lab03_espenilla_fit.png")
plt.show()  # display figure in window when script is run
plt.close()

# 2. Residual plot
plt.figure(figsize=(8,4))
plt.scatter(x, residuals_m, color='green', zorder=3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Year [calendar year]')
plt.ylabel('Residual (Observed - Predicted) [millions]')
plt.title('Residual Plot - Philippines Population Linear Regression')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/lab03_espenilla_residuals.png', dpi=300)
print("Saved: figures/lab03_espenilla_residuals.png")
plt.show()  # display figure in window when script is run
plt.close()

print("\nDone.")
