import numpy as soman
import pandas as sasi
import matplotlib.pyplot as plt
import matplotlib as mpl


df1=  sasi.read_csv("Q_new_models/layer0_weights_pos.csv")
df2 = sasi.read_csv("Q_new_models/layer0_weights_neg.csv")
df3 = sasi.read_csv("Q_new_models/layer1_weights_pos.csv")
df4 = sasi.read_csv("Q_new_models/layer1_weights_neg.csv")
df5 = sasi.read_csv("Q_new_models/layer2_weights_pos.csv")
df6 = sasi.read_csv("Q_new_models/layer2_weights_neg.csv")

# Separate bias (last row) and weights (rest)
df1_bias = df1.tail(1).values.flatten()
df2_bias = df2.tail(1).values.flatten()
df3_bias = df3.tail(1).values.flatten()
df4_bias = df4.tail(1).values.flatten()
df5_bias = df5.tail(1).values.flatten()
df6_bias = df6.tail(1).values.flatten()

df1_weights = df1.iloc[:-1].values.flatten()
df2_weights = df2.iloc[:-1].values.flatten()
df3_weights = df3.iloc[:-1].values.flatten()
df4_weights = df4.iloc[:-1].values.flatten()
df5_weights = df5.iloc[:-1].values.flatten()
df6_weights = df6.iloc[:-1].values.flatten()
all_weights_biases = sasi.Series(
    list(df1_weights) + list(df2_weights) + list(df3_weights) + list(df4_weights)+list(df5_weights) + list(df6_weights) + list(df1_bias) + list(df2_bias) + list(df3_bias) + list(df4_bias)
+list(df5_bias) + list(df6_bias))


unique_vals = soman.unique(all_weights_biases)
print(len(unique_vals))
print(1/unique_vals)
plt.plot(1/unique_vals)
plt.show()


# Set the IEEE conference font style
mpl.rcParams['font.family'] = 'Times New Roman'  # Font family
# mpl.rcParams['font.size'] = 10  # Font size for general text
# mpl.rcParams['axes.titles.fontsize'] = 12  # Title font size
mpl.rcParams['axes.labelsize'] = 10  # Axis labels font size
mpl.rcParams['xtick.labelsize'] = 10  # X-axis ticks font size
mpl.rcParams['ytick.labelsize'] = 10  # Y-axis ticks font size

data_dict = {
    1: (1/df1_weights).tolist(),
    2: (1/df1_bias).tolist(),
    3: (1/df2_weights).tolist(),
    4: (1/df2_bias).tolist(),
    5: (1/df3_weights).tolist(),
    6: (1/df3_bias).tolist(),
    7: (1/df4_weights).tolist(),
    8: (1/df4_bias).tolist(),
    9: (1 / df5_weights).tolist(),
    10: (1 / df5_bias).tolist(),
    11: (1 / df6_weights).tolist(),
    12: (1 / df6_bias).tolist()
}

size_constant = 25

# Create a larger figure size
plt.figure(figsize=(6,4))  # Adjusted size for better visibility

# Plotting scatter points
for xe, ye in data_dict.items():
    xAxis = [xe] * len(ye)
    sizes = [size_constant for num in ye]
    plt.scatter(xAxis, ye, s=sizes)

# Add grid and labels
plt.grid(True)
plt.title('Discrete Resistance levels used for mapping weights and biases')
plt.xlabel('Parameters Sets')
plt.ylabel('Resistance (Ω)')
plt.savefig('scatter_plot_high_res.pdf', dpi=600)
# Show the plot
plt.show()