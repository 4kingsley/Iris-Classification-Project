# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the Iris dataset from the CSV file
data = pd.read_csv("Iris.csv")

# Remove the 'Id' column, as it is not needed for the model
iris_df = data.drop(columns=['Id'])

# Preview the first few rows to ensure the data is correct
print(iris_df.head())

# Prepare the data by separating the features (X) and target (y)
X = iris_df.drop('Species', axis=1)  # Features: all columns except 'Species'
y = iris_df['Species']  # Target: the 'Species' column, which contains the flower class

# Split the data into training and test sets (70% training data, 30% test data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Normalize the features using StandardScaler to scale them to a similar range
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models you want to test (Logistic Regression, KNN, Decision Tree, SVM, Random Forest)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier()
}

# Loop through each model, train it, and evaluate its performance
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    # Display confusion matrix with corrected labels
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))

    # Convert the unique species (flower classes) to a list of strings
    species_labels = iris_df['Species'].unique().tolist()

    # Create a heatmap of the confusion matrix with annotations (numbers) and color coding
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=species_labels, yticklabels=species_labels)
    plt.title(f'Confusion matrix: {name}')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()

    # Custom prediction with a new sample of flower measurements (sepal and petal measurements)
    new_sample = [[5.1, 3.5, 1.4, 0.2]]  # New sample with sepal and petal measurements
    new_pred = model.predict(scaler.transform(new_sample)) # Use the model to predict the species of this new sample
    print(f"Prediction for new sample: {new_pred[0]}")  # Output the predicted species for the new sample