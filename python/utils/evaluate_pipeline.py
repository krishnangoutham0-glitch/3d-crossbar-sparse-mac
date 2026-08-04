import pandas as pd

def get_best_models(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Find the row with the best test accuracy
    best_accuracy_row = df.loc[df['Test_Accuracy'].idxmax()]
    best_accuracy_model = best_accuracy_row['Model_Path']

    return best_accuracy_model

model_name = get_best_models("q_results_mnist.csv")
print(model_name)