import pandas as pd
import numpy as np
from tensorflow.keras import backend as K
from sklearn.preprocessing import  MinMaxScaler

def clipped_relu(x):
    return K.relu(x, max_value=0.1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load weights (784 rows × N columns, where N is number of neurons)
W0_pos = pd.read_csv("Q_new_models/layer0_weights_pos.csv")
# W0_pos = W0_pos.iloc[1:, ].reset_index(drop=True).astype(float)
print("Wo_pos: ",W0_pos.shape)
W0_neg = pd.read_csv("Q_new_models/layer0_weights_neg.csv")
# W0_neg = W0_neg.iloc[1:, ].reset_index(drop=True).astype(float)
print(W0_neg.shape)
W1_pos = pd.read_csv("Q_new_models/layer1_weights_pos.csv")
# W1_pos = W1_pos.iloc[1:, ].reset_index(drop=True).astype(float)
print(W1_pos.shape)
W1_neg = pd.read_csv("Q_new_models/layer1_weights_neg.csv")
# W1_neg = W1_neg.iloc[1:, ].reset_index(drop=True).astype(float)
print(W1_neg.shape)
W2_pos = pd.read_csv("Q_new_models/layer2_weights_pos.csv")
# W2_pos = W2_pos.iloc[1:, ].reset_index(drop=True).astype(float)
print(W2_pos.shape)
W2_neg = pd.read_csv("Q_new_models/layer2_weights_neg.csv")
# W2_neg = W2_neg.iloc[1:, ].reset_index(drop=True).astype(float)
print(W2_neg.shape)
# Load only the first input sample (assumed to be in the first column)
input =  pd.read_csv("Q_new_models/side_out/HAR_sample.csv")
print(f"input shape : {input.shape}")
x_input=input.T
print(x_input.shape)

gain= 120000

########### LAYER 0 OPERATION ###############
mac_output_pos = np.dot(W0_pos.T, x_input)  # shape will be (N, 1)
mac_output_pos_df = pd.DataFrame(mac_output_pos)

mac_output_neg = np.dot(W0_neg.T, x_input)  # shape will be (N, 1)
mac_output_neg_df = pd.DataFrame(mac_output_neg)

mac_output_l0_dif = mac_output_pos - mac_output_neg
mac_output_l0_dif = gain * mac_output_l0_dif
mac_output_l0_dif_df = pd.DataFrame(mac_output_l0_dif)


mac_output_l0_final=clipped_relu(mac_output_l0_dif)
mac_output_l0_final_df =pd.DataFrame(mac_output_l0_final)

# Create a row with 0.1 repeated for each column
row_to_append = pd.DataFrame([[0.1] * mac_output_l0_final_df.shape[1]], columns=mac_output_l0_final_df.columns)

# Append the row to the DataFrame
mac_output_l0_final_df = pd.concat([mac_output_l0_final_df, row_to_append], ignore_index=True)

#################LAYER 1###################

mac_output_pos_l1 = np.dot(W1_pos.T, mac_output_l0_final_df)  # shape will be (N, 1)
mac_output_pos_l1_df = pd.DataFrame(mac_output_pos_l1)

mac_output_neg_l1 = np.dot(W1_neg.T, mac_output_l0_final_df)  # shape will be (N, 1)
mac_output_neg_l1_df = pd.DataFrame(mac_output_neg_l1)

mac_output_l1_dif = mac_output_pos_l1 - mac_output_neg_l1
mac_output_l1_dif = gain * mac_output_l1_dif
mac_output_l1_dif_df = pd.DataFrame(mac_output_l1_dif)


mac_output_l1_final=clipped_relu(mac_output_l1_dif)
mac_output_l1_final_df =pd.DataFrame(mac_output_l1_final)
print(mac_output_l1_final_df.head())
#
# Create a row with 0.1 repeated for each column
row_to_append = pd.DataFrame([[0.1] * mac_output_l1_final_df.shape[1]], columns=mac_output_l1_final_df.columns)

# Append the row to the DataFrame
mac_output_l1_final_df = pd.concat([mac_output_l1_final_df, row_to_append], ignore_index=True)


##########LAYER 2 OPERATION ###############

mac_output_pos_l2 = np.dot(W2_pos.T, mac_output_l1_final_df)  # shape will be (N, 1)
mac_output_pos_l2_df = pd.DataFrame(mac_output_pos_l2)

mac_output_neg_l2 = np.dot(W2_neg.T, mac_output_l1_final_df)  # shape will be (N, 1)
mac_output_neg_l2_df = pd.DataFrame(mac_output_neg_l2)

mac_output_l2_dif = mac_output_pos_l2 - mac_output_neg_l2
mac_output_l2_dif = gain * mac_output_l2_dif
mac_output_l2_dif_df = pd.DataFrame(mac_output_l2_dif)

mac_output_l2_final = sigmoid(mac_output_l2_dif)

# Convert to DataFrame if needed
mac_output_l2_final_df = pd.DataFrame(mac_output_l2_final)

# Save to CSV
mac_output_pos_df.to_csv("Q_new_models/side_out/mac_output_pos_l0.csv", index=False, header=False)
mac_output_neg_df.to_csv("Q_new_models/side_out/mac_output_neg_l0.csv", index=False, header=False)
mac_output_l0_dif_df.to_csv("Q_new_models/side_out/mac_output_l0_dif.csv", index=False, header=False)
mac_output_l0_final_df.to_csv("Q_new_models/side_out/mac_output_l0_final.csv", index=False, header=False)

mac_output_pos_l1_df.to_csv("Q_new_models/side_out/mac_output_pos_l1.csv", index=False, header=False)
mac_output_neg_l1_df.to_csv("Q_new_models/side_out/mac_output_neg_l1.csv", index=False, header=False)
mac_output_l1_dif_df.to_csv("Q_new_models/side_out/mac_output_l1_dif.csv", index=False, header=False)
mac_output_l1_final_df.to_csv("Q_new_models/side_out/mac_output_l1_final.csv", index=False, header=False)

mac_output_pos_l2_df.to_csv("Q_new_models/side_out/mac_output_pos_l2.csv", index=False, header=False)
mac_output_neg_l2_df.to_csv("Q_new_models/side_out/mac_output_neg_l2.csv", index=False, header=False)
mac_output_l2_dif_df.to_csv("Q_new_models/side_out/mac_output_l2_dif.csv", index=False, header=False)
mac_output_l2_final_df.to_csv("Q_new_models/side_out/mac_output_l2_final.csv", index=False, header=False)

print("done done done")
