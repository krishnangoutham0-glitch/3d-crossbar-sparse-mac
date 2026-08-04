import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Given data points
W_values = [1, 1, 1]
L_values = [1, 0.15, 0.2]
R_values = [2000, 300, 400]

# Reshape input for curve_fit
X_data = np.vstack((L_values, W_values)).T  # shape (n_samples, 2)

# Resistance model
def resistance_model(X, k):
    L, W = X[:, 0], X[:, 1]
    return k * (L / W)

# Fit the resistance model to the data
popt, _ = curve_fit(resistance_model, X_data, R_values)
k_value = popt[0]
print("Fitted k:", k_value)

# Function to calculate the minimum W and L given a resistance value
def find_min_w_l_for_resistance_fast(target_R, min_value=0.15, step=0.01, max_W=5.0, max_L=15.0):
    W_vals = np.arange(min_value, max_W, step)
    L_vals = np.arange(min_value, max_L, step)
    W_grid, L_grid = np.meshgrid(W_vals, L_vals)

    R_grid = k_value * (L_grid / W_grid)
    error_grid = np.abs(R_grid - target_R)

    # 1% relative tolerance
    tolerance = 0.01 * target_R
    valid_mask = error_grid <= tolerance

    if not np.any(valid_mask):
        return None, None

    # Get the W, L pair with the smallest area among valid options
    area_grid = W_grid * L_grid
    area_grid[~valid_mask] = np.inf  # Mask out invalid areas

    min_idx = np.unravel_index(np.argmin(area_grid), area_grid.shape)
    return W_grid[min_idx], L_grid[min_idx]



# Example usage
target_weights =pd.read_csv(r"Q_new_models/layer0_weights_neg.csv")
target_resistance = 1 / target_weights.values.flatten()
unique_resistance = np.unique(target_resistance)



dict_res = {}

for res in unique_resistance:
    W, L = find_min_w_l_for_resistance_fast(res)
    dict_res[res] = (W, L)

df = pd.DataFrame.from_dict(dict_res)

df = df.transpose()

df.to_csv("Q_new_models/side_out/layer0_unique_res_neg.csv")

# Optional: print or save the dictionary

#
# if W and L:
#     print(f"For resistance {target_resistance}Ω, calculated W = {W:.3f}, L = {L:.3f}")
# else:
#     print("No valid W and L found within the constraints.")