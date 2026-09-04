import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. DATA
# Year (independent variable x) and annual mean CO2 concentration
# (dependent variable y) from NOAA Mauna Loa Observatory
# ============================================================
years = np.array([
1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968,
1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978,
1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988,
1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
2019, 2020, 2021, 2022, 2023, 2024, 2025
])

co2 = np.array([
315.98, 316.91, 317.64, 318.45, 318.99, 319.62, 320.04, 321.37, 322.18, 323.05,
324.62, 325.68, 326.32, 327.46, 329.68, 330.19, 331.13, 332.03, 333.84, 335.41,
336.84, 338.76, 340.12, 341.48, 343.15, 344.87, 346.35, 347.61, 349.31, 351.69,
353.20, 354.45, 355.70, 356.54, 357.21, 358.96, 360.97, 362.74, 363.88, 366.84,
368.54, 369.71, 371.32, 373.45, 375.98, 377.70, 379.98, 382.09, 384.02, 385.83,
387.64, 390.10, 391.85, 394.06, 396.74, 398.81, 401.01, 404.41, 406.76, 408.72,
411.65, 414.21, 416.41, 418.53, 421.08, 424.61, 427.35
])

# Convert to float arrays for calculations
x = years.astype(float)   # independent variable
y = co2                   # dependent variable
n = len(x)                # number of observations

# ============================================================
# 2. LEAST-SQUARES CALCULATIONS (manual formulas)
# ============================================================

# Compute the required sums
sum_x  = np.sum(x)        # Σx
sum_y  = np.sum(y)        # Σy
sum_xy = np.sum(x * y)    # Σxy
sum_x2 = np.sum(x**2)     # Σx²

# Slope a1 using the least-squares formula:
# a1 = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

# Intercept a0 using the least-squares formula:
# a0 = (Σy - a1*Σx) / n
a0 = (sum_y - a1 * sum_x) / n

# ============================================================
# 3. PREDICTED VALUES, RESIDUALS, AND GOODNESS-OF-FIT
# ============================================================

# Predicted y values from the fitted line
y_hat = a0 + a1 * x

# Residuals = observed y - predicted y
residuals = y - y_hat

# Sr = SSE = sum of squared residuals (sum of squares of the errors)
Sr = np.sum(residuals**2)

# Total sum of squares (SST)
mean_y = np.mean(y)
SST = np.sum((y - mean_y)**2)

# Coefficient of determination r²
r2 = 1 - (Sr / SST)

# Standard error of the estimate sy/x
syx = np.sqrt(Sr / (n - 2))

# Print the key results
print(f"Number of points n = {n}")
print(f"Intercept a0 = {a0:.6f}")
print(f"Slope     a1 = {a1:.6f}")
print(f"Sr (SSE)     = {Sr:.6f}")
print(f"r^2          = {r2:.6f}")
print(f"sy/x         = {syx:.6f}")

# ============================================================
# 4. PREDICTION FOR A YEAR NOT IN THE DATA SET
# ============================================================
x_pred = 2030.0
y_pred = a0 + a1 * x_pred
print(f"Predicted CO2 in year {int(x_pred)} = {y_pred:.2f} ppm")

# ============================================================
# 5. GRAPHS
# ============================================================

# Graph 1: Data points + fitted straight line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', s=30, label='Observed data', zorder=3)
plt.plot(x, y_hat, color='red', linewidth=2,
         label=f'Fitted line: y = {a0:.2f} + {a1:.4f}x')
plt.xlabel('Year')
plt.ylabel('Annual mean CO2 concentration (ppm)')
plt.title('Linear Regression: Atmospheric CO2 at Mauna Loa (1959-2025)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('co2_fit.png', dpi=150)
plt.show()

# Graph 2: Residual plot
plt.figure(figsize=(10, 5))
plt.scatter(x, residuals, color='green', s=30)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Year')
plt.ylabel('Residual (ppm)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('co2_residuals.png', dpi=150)
plt.show()