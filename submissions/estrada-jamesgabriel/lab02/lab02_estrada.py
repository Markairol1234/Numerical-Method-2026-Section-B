import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import curve_fit

# 1. Load Data & Derivatives
df_raw = pd.read_excel('Data 01.xlsx', sheet_name='Sensor Log', skiprows=3)
df_raw = df_raw.iloc[:, [0, 1, 4]]
df_raw.columns = ['Reading', 'Timestamp', 'Depth']
df_raw = df_raw.dropna(subset=['Depth'])

t = np.arange(len(df_raw)) * 0.25
h = df_raw['Depth'].values
dt = 0.25

dh_dt_central = np.zeros_like(h)
dh_dt_central[0] = (h[1] - h[0]) / dt
dh_dt_central[-1] = (h[-1] - h[-2]) / dt
for i in range(1, len(h) - 1):
  dh_dt_central[i] = (h[i + 1] - h[i - 1]) / (2 * dt)

max_idx = np.argmax(dh_dt_central)
max_time = df_raw['Timestamp'].iloc[max_idx]
max_val = dh_dt_central[max_idx]


# 2. Curve Fitting & Statistics
def logistic_model(t, c, a, k, t0):
  return c + a / (1.0 + np.exp(-k * (t - t0)))


p0 = [14.18, 7.0, 0.5, 29.0]
popt, pcov = curve_fit(logistic_model, t, h, p0=p0, method='lm', maxfev=2000)

n = len(h)
p = len(popt)
df_deg = n - p
h_fit = logistic_model(t, *popt)
residh = h - h_fit
sse = np.sum(residh**2)
sst = np.sum((h - h.mean()) ** 2)
r2 = 1.0 - (sse / sst)
s_est = np.sqrt(sse / df_deg)
se_params = np.sqrt(np.diag(pcov))
t_stats = popt / se_params
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_deg))

# 3. Integration
area_val, area_err = quad(logistic_model, t[0], t[-1], args=tuple(popt))
area_trapz = np.trapezoid(h, t)


# --- GENERATE PLOTS AS BASE64 IMAGES FOR HTML EMBEDDING ---
def fig_to_base64(fig):
  buf = BytesIO()
  fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
  buf.seek(0)
  encoded = base64.b64encode(buf.read()).decode('utf-8')
  plt.close(fig)
  return encoded


# Plot 1: Derivatives & Stage Time Series
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(t, h, label='Stage Log (Raw)', color='blue', alpha=0.6)
ax1.set_ylabel('Depth (m)', color='blue')
ax2 = ax1.twinx()
ax2.plot(t, dh_dt_central, label='dh/dt (m/h)', color='orange', linestyle='--')
ax2.scatter(
    t[max_idx],
    max_val,
    color='red',
    s=50,
    zorder=5,
    label=f'Max dh/dt: {max_val:.2f} m/h',
)
ax2.set_ylabel('dh/dt (m/h)', color='orange')
plt.title('Stage Time Series & First Derivative (Tab 1)')
img1 = fig_to_base64(fig)

# Plot 2: Fitted Curve & Residuals
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.scatter(t, h, color='black', s=10, label='Raw Data', alpha=0.5)
ax1.plot(
    t,
    h_fit,
    color='red',
    linewidth=2,
    label='Logistic Fit (4-Param Model)',
)
ax1.set_ylabel('Depth (m)')
ax1.legend()
ax1.set_title('Tab 2: Fitted Curve over Raw Data & Residuals')

ax2.scatter(t, residh, color='purple', s=10, alpha=0.6)
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_ylabel('Residuals (m)')
ax2.set_xlabel('Time (Hours)')
img2 = fig_to_base64(fig)

# Plot 3: Area under curve (Integration)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, h_fit, color='green', label='Fitted Curve')
ax.fill_between(
    t,
    0,
    h_fit,
    color='green',
    alpha=0.2,
    label=f'Area = {area_val:.2f} m-h',
)
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Time (Hours)')
ax.legend()
ax.set_title('Tab 3: Area Under Fitted Curve')
img3 = fig_to_base64(fig)


# --- BUILD SELF-CONTAINED HTML DASHBOARD ---
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Lab 02 Dashboard - Escranda</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f4f4f9; color: #333; }}
        h1, h2 {{ color: #0056b3; }}
        .card {{ background: #fff; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .plot-container {{ text-align: center; margin-top: 15px; }}
        .plot-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #0056b3; color: white; }}
    </style>
</head>
<body>
    <h1>Numerical Methods - Laboratory Activity 02 Dashboard</h1>
    
    <!-- HEADER: ALWAYS VISIBLE -->
    <div class="card">
        <h2>Header: Reservoir Stage Time Series</h2>
        <p><strong>Total Readings:</strong> {n} (15-minute sampling interval)</p>
        <p><strong>Maximum dh/dt Rate:</strong> {max_val:.4f} m/h at {max_time}</p>
        <div class="plot-container">
            <img src="data:image/png;base64,{img1}" alt="Stage and Derivatives Plot">
        </div>
    </div>

    <!-- TAB 2: FITTED CURVE & STATISTICS -->
    <div class="card">
        <h2>Tab 2: Curve Fitting, Residuals & Statistics</h2>
        <p><strong>Model Equation:</strong> h(t) = c + a / (1 + exp(-k * (t - t0)))</p>
        <p><strong>Coefficient of Determination (R&sup2;):</strong> {r2:.6f} | <strong>SST:</strong> {sst:.4f}</p>
        <p><strong>Standard Error of Estimate (s):</strong> {s_est:.4f} meters (Degrees of Freedom: {df_deg})</p>
        <p><strong>Sum of Squared Errors (SSE):</strong> {sse:.6f} (n={n}, p={p})</p>
        
        <table>
            <tr><th>Parameter</th><th>Value</th><th>Standard Error</th><th>t-statistic</th><th>P-value</th></tr>
            <tr><td>c (Base Level)</td><td>{popt[0]:.4f}</td><td>{se_params[0]:.4f}</td><td>{t_stats[0]:.4f}</td><td>{p_values[0]:.4e}</td></tr>
            <tr><td>a (Amplitude)</td><td>{popt[1]:.4f}</td><td>{se_params[1]:.4f}</td><td>{t_stats[1]:.4f}</td><td>{p_values[1]:.4e}</td></tr>
            <tr><td>k (Growth Rate)</td><td>{popt[2]:.4f}</td><td>{se_params[2]:.4f}</td><td>{t_stats[2]:.4f}</td><td>{p_values[2]:.4e}</td></tr>
            <tr><td>t0 (Inflection)</td><td>{popt[3]:.4f}</td><td>{se_params[3]:.4f}</td><td>{t_stats[3]:.4f}</td><td>{p_values[3]:.4e}</td></tr>
        </table>

        <div class="plot-container" style="margin-top: 20px;">
            <img src="data:image/png;base64,{img2}" alt="Fitted Curve and Residuals Plot">
        </div>
        <p style="margin-top: 10px; font-size: 14px;"><strong>Reading of Residuals:</strong> Residuals are tightly centered around zero during steady phases, with minor structured variations during the steep surge phase around July 22, confirming a strong macroscopic fit despite localized turbulence.</p>
    </div>

    <!-- TAB 3: INTEGRATION -->
    <div class="card">
        <h2>Tab 3: Area Under the Curve (Integration)</h2>
        <p><strong>Fitted Curve Integral (scipy.integrate.quad):</strong> {area_val:.4f} meter-hours</p>
        <p><strong>Raw Data Trapezoid Cross-check:</strong> {area_trapz:.4f} meter-hours</p>
        <div class="plot-container">
            <img src="data:image/png;base64,{img3}" alt="Integration Shaded Area Plot">
        </div>
    </div>
</body>
</html>
"""

with open('lab02_escranda.html', 'w', encoding='utf-8') as f:
  f.write(html_content)

print(
    'Enhanced HTML dashboard with embedded charts successfully generated as'
    ' lab02_escranda.html!'
)