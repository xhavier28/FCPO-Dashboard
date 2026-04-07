import numpy as np
import pandas as pd


def run_kalman(y, x, delta: float, Ve: float) -> dict:
    """
    Time-varying Kalman filter estimating beta(t) and alpha(t).

    State: theta = [beta(t), alpha(t)]
    Observation: y[t] = beta(t)*x[t] + alpha(t) + noise
    Process noise: Q = (delta / (1-delta)) * I(2)
    Observation noise variance: Ve
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    n = len(y_arr)

    Q = (delta / (1.0 - delta)) * np.eye(2)

    # State estimate and covariance
    theta = np.array([1.0, 0.0])   # [beta, alpha]
    P = np.eye(2)

    beta_t                = np.empty(n)
    alpha_t               = np.empty(n)
    innovations           = np.empty(n)
    spread_reconstructed  = np.empty(n)
    P_trace               = np.empty(n)

    for t in range(n):
        F = np.array([x_arr[t], 1.0])  # observation vector

        # Predict
        P = P + Q

        # Innovation (uses PRIOR theta — before update)
        y_hat          = float(F @ theta)
        innov          = y_arr[t] - y_hat
        innovations[t] = innov
        S              = float(F @ P @ F) + Ve

        # Kalman gain
        K = P @ F / S

        # Update
        theta = theta + K * innov
        P = P - np.outer(K, F) @ P

        beta_t[t]               = theta[0]
        alpha_t[t]              = theta[1]
        spread_reconstructed[t] = y_arr[t] - theta[0] * x_arr[t] - theta[1]
        P_trace[t]              = float(np.trace(P))

    return {
        "beta_t":               beta_t,
        "alpha_t":              alpha_t,
        "spread":               innovations,           # Kalman residuals (near white noise)
        "spread_reconstructed": spread_reconstructed,  # y - beta_t*x - alpha_t (tradeable)
        "P_trace":              P_trace,
        "delta_used":           delta,
        "Ve_used":              Ve,
        "space":                "raw",
    }
