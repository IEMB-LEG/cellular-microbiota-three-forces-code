
# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import math

EXCEL_PATH = "paramecium-growth-rate.xlsx"
OUT_DIR = Path("./growth_overall_logdecay_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

days = np.arange(1, 16, dtype=float)

def logistic_decay(t, K, r, t0, d, td):
    growth = K / (1.0 + np.exp(-r * (t - t0)))
    decay  = np.exp(-d * np.maximum(t - td, 0.0))
    return growth * decay

def auto_initial_guess(y, x):
    K0   = float(np.max(y)); tmax = int(x[np.argmax(y)])
    half = 0.5 * K0
    idx_half = np.argmin(np.abs(y - half))
    if idx_half == 0:
        idx_half = max(1, np.argmax(np.diff(y)))
    try:
        r0 = max(0.2, min(3.0, (y[idx_half] / max(1.0, idx_half)) / (K0/4.0)))
    except Exception:
        r0 = 1.0
    t0_0 = max(1.0, min(10.0, tmax - 2)); d0 = 0.08; td0 = max(3.0, min(12.0, tmax))
    return [K0, float(r0), float(t0_0), float(d0), float(td0)]

def r2_score(y, yhat):
    ss_res = float(np.sum((y - yhat)**2)); ss_tot = float(np.sum((y - np.mean(y))**2))
    return 1.0 - ss_res/ss_tot

def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat)**2)))
def aic(y, yhat, k):
    n = len(y); rss = float(np.sum((y - yhat)**2))
    if rss <= 0: rss = 1e-12
    return float(n*math.log(rss/n) + 2*k)

def main():
    df = pd.read_excel(EXCEL_PATH)
    samples = df.iloc[:4, 1:].to_numpy(dtype=float)                  # 4×15
    mean_row = df[df["Unnamed: 0"] == "均值"].iloc[0, 1:].to_numpy(dtype=float)

    # Fit with auto p0
    p0 = auto_initial_guess(mean_row, days)
    popt, _ = curve_fit(logistic_decay, days, mean_row, p0=p0, bounds=(0, [10000,5,15,2,15]), maxfev=20000)
    yhat = logistic_decay(days, *popt)

    # Save metrics
    params_df = pd.DataFrame([{
        "K": popt[0], "r": popt[1], "t0": popt[2], "d": popt[3], "td": popt[4],
        "R2": r2_score(mean_row, yhat),
        "RMSE": rmse(mean_row, yhat),
        "AIC": aic(mean_row, yhat, k=5),
        "Auto_p0": str(p0)
    }])
    params_df.to_csv(OUT_DIR/"overall_logdecay_params_metrics_auto_day1to12.csv", index=False)

    # Save fitted values (full Day1..Day15 for traceability)
    fit_df = pd.DataFrame({"Original_Day": days.astype(int), "Observed_Mean": mean_row, "Fitted": yhat})
    fit_df.to_csv(OUT_DIR/"overall_logdecay_fitted_values_auto_day1to12.csv", index=False)
# -------- Plot with new axis labeling --------
    fig_path = OUT_DIR/"overall_fit_with_samples_auto_day1to12.png"
    plt.figure(figsize=(8,5))

# 4 samples: low-saturation polylines
    for i in range(samples.shape[0]):
        plt.plot(days, samples[i], lw=2, alpha=0.35)
    
    days_dense = np.linspace(days[0], days[-1], 50)  
    yhat_dense = logistic_decay(days_dense, *popt)


    plt.plot(days_dense, yhat_dense, lw=6, alpha=0.3, label='Fitted curve')
    plt.plot(days_dense[days_dense > 4], yhat_dense[days_dense >4], lw=6, alpha=0.9, 
             color=plt.gca().lines[-1].get_color())


# X-axis relabel: original 4..15 -> Day1..Day12
    relabeled_days = days[3:]  # 4..15
    relabeled_names = [f"Day{i}" for i in range(1, len(relabeled_days)+1)]
    plt.xticks(relabeled_days, relabeled_names)

    #plt.xlabel("Day (relabelled: original Day4→Day1)")
    plt.ylabel("Number / mL")
    plt.title("Bi-phasic Logistic–Decay Growth  Curve")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    
    print("Saved:", fig_path)
    print(params_df.to_string(index=False))

if __name__ == "__main__":
    main()
