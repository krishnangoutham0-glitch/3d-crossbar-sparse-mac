import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load features (561 columns) and activity labels
X = pd.read_csv(
    "/Users/gouthamkrishnan/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/train/X_train.txt",
    delim_whitespace=True,
    header=None
)

y = pd.read_csv(
    "/Users/gouthamkrishnan/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/train/y_train.txt",
    header=None,
    names=['activity']
)

# Ensure y and X are same length
assert len(X) == len(y), "Mismatch in number of samples between X and y"

# Concatenate features and labels
data = pd.concat([X, y], axis=1)

# Force activity column to integer
data['activity'] = data['activity'].astype(int)

# Sort to ensure consistency
data = data.sort_values(by='activity')

# Now select one sample per unique activity label
samples = []
for label in range(1, 7):  # labels 1 through 6
    sample = data[data['activity'] == label].iloc[0, :-1]  # drop label column
    samples.append(sample.values)

# Stack samples into (6, 561)
samples_array = np.vstack(samples)

# Normalize to range (0, 0.1)
scaler = MinMaxScaler(feature_range=(0, 0.1))
normalized = scaler.fit_transform(samples_array)

# Append a column of 0.1
final_array = np.hstack([normalized, np.full((6, 1), 0.1)])

# Convert to DataFrame
final_df = pd.DataFrame(final_array)

# Check shape and display
print("Final shape:", final_df.shape)  # Should be (6, 562)
final_df.to_csv("side_out/HAR_sample.csv", index=False)
