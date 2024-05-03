import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
# Load the car dataset from CSV
url = "car.csv"
df = pd.read_csv(url, header=None)
# Extract column labels from the first row
columns = df.iloc[0].tolist()
df = df[1:] # Skip the first row with column labels
df.columns = columns # Set the column names
# Convert categorical variables to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=["buying", "maint", "doors", "person","lug_boot", "safety"])
# Separate features (X) and target variable (y)
X = df_encoded.drop("class_variable", axis=1)
y = df_encoded["class_variable"]
# Number of repetitions
num_repeats = 20
# Lists to store results
f1_scores = []
conf_matrices = []
# Define a range of max_depth values to test
max_depth_values = [3, 5, 7, 10, 15, 20,25, None]
for max_depth in max_depth_values:
    f1_scores_iter = []
    conf_matrices_iter = []
    for _ in range(num_repeats):
# Split the data into training (60%) and testing (40%)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.4, stratify=y)
# Create a Decision Tree classifier with entropy as the criterion
        clf = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=max_depth)
        # Train the classifier on the training data
        clf.fit(X_train, y_train)
# Make predictions on the test data
        y_pred = clf.predict(X_test)
# Evaluate the performance using confusion matrix and F1-score
        conf_matrix = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
# Store results
        conf_matrices_iter.append(conf_matrix)
        f1_scores_iter.append(f1)
    average_f1_iter = np.mean(f1_scores_iter, axis=0)
    average_conf_matrix_iter = np.mean(conf_matrices_iter, axis=0)
    print(f"\nMax Depth: {max_depth}")
    print("Average Confusion Matrix:")
    print(average_conf_matrix_iter.astype(int))
    print("Average F1 Score:", average_f1_iter)
# Store results for all max_depth values
    f1_scores.append(average_f1_iter)
    conf_matrices.append(average_conf_matrix_iter)
    #Find the max_depth that gives the highest average F1-score
best_max_depth_index = np.argmax(f1_scores)
best_max_depth = max_depth_values[best_max_depth_index]
print("\nBest Max Depth:", best_max_depth)