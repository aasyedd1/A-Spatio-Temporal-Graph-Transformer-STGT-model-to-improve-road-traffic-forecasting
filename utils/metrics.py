import numpy as np

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE) with masking for small true values (<10 km/h).
    Ignores ground truth values below 10 km/h, as in STTN best practice.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = np.abs(y_true) >= 10.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mape = np.abs((y_true - y_pred) / y_true)
    if len(mape) == 0:
        return np.nan
    return np.mean(mape) * 100
