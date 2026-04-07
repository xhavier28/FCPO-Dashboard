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

    beta_t  = np.empty(n)
    alpha_t = np.empty(n)
    spread  = np.empty(n)
    P_trace = np.empty(n)

    spread[0] = 0.0

    for t in range(n):
        F = np.array([x_arr[t], 1.0])  # observation vector

        # Predict
        P = P + Q

        # Innovation
        y_hat = float(F @ theta)
        innov = y_arr[t] - y_hat
        S = float(F @ P @ F) + Ve

        # Kalman gain
        K = P @ F / S

        # Update
        theta = theta + K * innov
        P = P - np.outer(K, F) @ P

        beta_t[t]  = theta[0]
        alpha_t[t] = theta[1]
        spread[t]  = y_arr[t] - theta[0] * x_arr[t] - theta[1]
        P_trace[t] = float(np.trace(P))

    return {
        "beta_t":     beta_t,
        "alpha_t":    alpha_t,
        "spread":     spread,
        "P_trace":    P_trace,
        "delta_used": delta,
        "Ve_used":    Ve,
        "space":      "raw",
    }
