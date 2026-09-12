import numpy as np
import matplotlib.pyplot as plt

# Dataset: Curing Age (x) and Compressive Strength (y)
x = np.array([3, 3, 7, 7, 14, 14, 21, 21, 28, 28, 35, 42, 56, 90, 100], dtype=float)
y = np.array([12.5, 14.1, 21.0, 23.4, 29.8, 31.2, 36.5, 38.0, 42.1, 44.3, 46.0, 48.5, 51.2, 54.0, 55.6], dtype=float)

n = len(x)

# Least-Squares Calculations
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_xx = np.sum(x ** 2)

# Calculate slope (a1) and intercept (a0)
a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
a0 = (sum_y - a1 * sum_x) / n

# Model predictions and statistics
y_pred = a0 + a1 * x
residuals = y - y_pred
Sr = np.sum(residuals**2)  # SSE

y_mean = np.mean(y)
St = np.sum((y - y_mean)**2)
r2 = (St - Sr) / St
sy_x = np.sqrt(Sr / (n - 2))

print(f"Regression Equation: y = {a0:.4f} + {a1:.4f}x")
print(f"Slope (a1): {a1:.4f}")
print(f"Intercept (a0): {a0:.4f}")
print(f"Sr (SSE): {Sr:.4f}")
print(f"r^2: {r2:.4f}")
print(f"Standard Error (sy/x): {sy_x:.4f}")

# Prediction for x = 60 days
x_new = 60.0
y_new = a0 + a1 * x_new
print(f"Prediction for x = {x_new} days: y = {y_new:.2f} MPa")

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label=f'Fit: y = {a0:.2f} + {a1:.2f}x')
plt.title('Concrete Curing Age vs Compressive Strength')
plt.xlabel('Curing Age (days)')
plt.ylabel('Compressive Strength (MPa)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.scatter(x, residuals, color='purple', label='Residuals')
plt.axhline(0, color='black', linestyle='-', linewidth=1)
plt.title('Residual Plot')
plt.xlabel('Curing Age (days)')
plt.ylabel('Residuals (y - y_pred)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('regression_analysis.pdf')
plt.show()