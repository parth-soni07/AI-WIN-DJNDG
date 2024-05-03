import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
# Load the car dataset from CSV
url = "car.csv" # Replace with the actual path or URL of your dataset
df = pd.read_csv(url, header=None)
# Extract column labels from the first row
columns = df.iloc[0].tolist()
df = df[1:] # Skip the first row with column labels
df.columns = columns # Set the column names
# Convert categorical variables to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=["buying", "maint", "doors", "person", "lug_boot", "safety"])
# Separate features (X) and target variable (y)
X = df_encoded.drop("class_variable", axis=1)
y = df_encoded["class_variable"]
# Number of repetitions
num_repeats = 20
# Lists to store results
f1_scores = []
conf_matrices = []
for _ in range(num_repeats):
    #Split the data into training (60%) and testing (40%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,stratify=y)
    # Create a Decision Tree classifier with Gini index as the criterion
    clf = DecisionTreeClassifier(criterion="gini", random_state=42)
    # Train the classifier on the training data
    clf.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    # Evaluate the performance using confusion matrix and F1-score
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    # Store results
    conf_matrices.append(conf_matrix)
    f1_scores.append(f1)
# Calculate and print the average F1-score and confusion matrix
average_f1 = np.mean(f1_scores, axis=0)
average_conf_matrix = np.mean(conf_matrices, axis=0)
print("Average Confusion Matrix:")
print(average_conf_matrix.astype(int))
print("\nAverage F1 Score:", average_f1)