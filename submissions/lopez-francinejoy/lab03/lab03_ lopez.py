"""
Lab 03: Real-World Data Linear Regression
------------------------------------------
Data set : NASA GISS Surface Temperature Analysis (GISTEMP v4)
Source   : NASA Goddard Institute for Space Studies
URL      : https://data.giss.nasa.gov/gistemp/
           (annual global mean table: GLB.Ts+dSST.csv)

x = Year (calendar year)
y = Global mean surface temperature anomaly (deg C, relative to the
    1951-1980 base period)

The script:
1. Stores the 20 paired observations (2006-2025) used for the fit.
2. Computes the least-squares regression y = a0 + a1*x "by hand"
   using the standard sums-of-squares formulas from class.
3. Computes Sr (SSE), r^2 and the standard error of the estimate sy/x.
4. Plots the data with the fitted line, and a residual plot.
5. Uses the fitted line to predict y for a year outside the data set.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------
# x = Year, y = Global Land-Ocean Temperature Index anomaly (deg C)
x = np.array([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
              2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
             dtype=float)

y = np.array([0.6417, 0.6658, 0.5450, 0.6567, 0.7250, 0.6075, 0.6467,
              0.6775, 0.7492, 0.8967, 1.0117, 0.9142, 0.8492, 0.9775,
              1.0067, 0.8475, 0.8908, 1.1675, 1.2842, 1.1925])

n = len(x)  # number of paired observations

# ---------------------------------------------------------------
# 2. Least-squares fit: y = a0 + a1*x
# Using the standard sums-of-squares formulas
# a1 = (n*Sum(xy) - Sum(x)*Sum(y)) / (n*Sum(x^2) - (Sum(x))^2)
# a0 = ybar - a1*xbar
# ---------------------------------------------------------------
sum_x  = np.sum(x)
sum_y  = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x ** 2)

x_bar = sum_x / n
y_bar = sum_y / n

a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
a0 = y_bar - a1 * x_bar

print(f"Number of observations, n = {n}")
print(f"Slope a1 = {a1:.6f} (deg C per year)")
print(f"Intercept a0 = {a0:.4f} (deg C)")
print(f"Regression equation: y = {a0:.4f} + {a1:.6f} * x")

# ---------------------------------------------------------------
# 3. Goodness of fit: St, Sr, r^2, and standard error sy/x
# ---------------------------------------------------------------
y_pred = a0 + a1 * x          # values predicted by the line
residuals = y - y_pred        # residuals (observed - predicted)

St = np.sum((y - y_bar) ** 2)  # total sum of squares about the mean
Sr = np.sum(residuals ** 2)   # sum of squares of residuals (SSE)

r2 = (St - Sr) / St            # coefficient of determination
r  = np.sqrt(r2)               # correlation coefficient

# standard error of the estimate (n-2 degrees of freedom for a line fit)
syx = np.sqrt(Sr / (n - 2))

print(f"St (total SS) = {St:.6f}")
print(f"Sr (SSE) = {Sr:.6f}")
print(f"r^2 = {r2:.6f}")
print(f"r = {r:.6f}")
print(f"Standard error sy/x = {syx:.6f} deg C")

# ---------------------------------------------------------------
# 4a. Plot: data points with the fitted line
# ---------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(x, y, color="tab:blue", label="Observed data")
x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, a0 + a1 * x_line, color="tab:red",
         label=f"Fit: y = {a0:.3f} + {a1:.5f}x")
plt.xlabel("Year")
plt.ylabel("Global temperature anomaly, deg C (vs. 1951-1980)")
plt.title("Global Temperature Anomaly vs. Year (GISTEMP, 2006-2025)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fit_plot.png", dpi=200)
plt.close()

# ---------------------------------------------------------------
# 4b. Residual plot
# ---------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(x, residuals, color="tab:green")
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Year")
plt.ylabel("Residual, deg C (observed - predicted)")
plt.title("Residual Plot")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=200)
plt.close()

# ---------------------------------------------------------------
# 5. Prediction for a year not in the data set
# ---------------------------------------------------------------
x_new = 2030.0
y_new = a0 + a1 * x_new
print(f"Prediction for x = {x_new:.0f}: y = {a0:.4f} + "
      f"{a1:.6f}*{x_new:.0f} = {y_new:.4f} deg C")
