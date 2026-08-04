from qkeras.quantizers import quantized_bits
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

#**********************************************************************************
results = []
output_folder = "Q_models"
os.makedirs(output_folder, exist_ok=True)
#**********************************************************************************

#Loading test and train data
data = pd.read_csv(r"/Users/gouthamkrishnan/Downloads/breast+cancer+wisconsin+original/breast-cancer-wisconsin.data", header=None)

data = data.replace('?', np.nan).dropna().astype(float)

# Split features and labels
X = data.iloc[:, 1:10].values  # 9 features
y = data.iloc[:, 10].values
y = np.where(y == 2, 0, 1)  # Binary classification (0 - benign, 1 - malignant)
X = X / 10.0
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#**********************************************************************************

# Load the saved model using h5py
with h5py.File("model.h5", "r") as f:
    w1_loaded = f["w1"][:]
    b1_loaded = f["b1"][:]
    w2_loaded = f["w2"][:]
    b2_loaded = f["b2"][:]
    w3_loaded = f["w_out"][:]
    b3_loaded = f["b_out"][()]  # Corrected for scalar value

#**********************************************************************************
print(w1_loaded)
#defining model parameters
def relu(x):
    return np.maximum(0, x)

# Define forward pass
def forward_pass(X, w1, b1, w2, b2, w_out, b_out):
    layer1_out = relu(np.dot(X, w1) + b1)
    layer2_out = relu(np.dot(layer1_out, w2) + b2)
    output = np.dot(layer2_out, w_out) + b_out
    return output, layer1_out, layer2_out

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#**********************************************************************************

#Grid Search for q_levels determination
q_levels=[1,2,3,4,5,6]

def quantize_weights_and_biases(weights, biases, weight_levels, bias_levels):
    quantized_weights = quantized_bits(weight_levels)(weights)
    quantized_biases = quantized_bits(bias_levels)(biases)
    return quantized_weights, quantized_biases

for w1_i in q_levels:
    for b1_i in q_levels:
        for w2_i in q_levels:
            for b2_i in q_levels:
                for w3_i in q_levels:
                    for b3_i in q_levels:
                        q_weight1, q_bias1 = quantize_weights_and_biases(w1_loaded,b1_loaded,w1_i,b1_i)
                        q_weight2, q_bias2 =quantize_weights_and_biases(w2_loaded,b2_loaded,w2_i,b2_i)
                        q_weight3, q_bias3 =quantize_weights_and_biases(w3_loaded,b3_loaded,w3_i,b3_i)
                        model_name = (f"model_l1w{w1_i}_l1b{b1_i}_l2w{w2_i}_l2b{b2_i}_l3w{w3_i}_l3b{b3_i}.h5")
                        with h5py.File(f"{output_folder}/{model_name}", "w") as f:
                            f.create_dataset("w1", data=q_weight1)
                            f.create_dataset("b1", data=q_bias1)
                            f.create_dataset("w2", data=q_weight2)
                            f.create_dataset("b2", data=q_bias2)
                            f.create_dataset("w_out", data=q_weight3)
                            f.create_dataset("b_out", data=q_bias3)

                        # Test predictions
                        test_output, _, _ = forward_pass(X_test, q_weight1,q_bias1, q_weight2, q_bias2, q_weight3,
                                                         q_bias3)
                        y_test_pred = sigmoid(test_output) > 0.5
                        accuracy = accuracy_score(y_test, y_test_pred)
                        results.append({
                            "Layer1_Weight_Levels": w1_i,
                            "Layer1_Bias_Levels": b1_i,
                            "Layer2_Weight_Levels": w2_i,
                            "Layer2_Bias_Levels": b2_i,
                            "Layer3_Weight_Levels": w3_i,
                            "Layer3_Bias_Levels": b3_i,
                            "Test_Accuracy": accuracy,
                            "Model_Path": model_name
                        })

results_df = pd.DataFrame(results)
results_df.to_csv("q_results.csv", index=False)