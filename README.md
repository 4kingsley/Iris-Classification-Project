# Iris-Classification-Project

## Objective

This project demonstrates how to build and evaluate machine learning models using the Iris dataset. The objective was to predict the species of Iris flowers based on measurements of the sepals and petals.

## Libraries Used

- scikit-learn: Used for machine learning models.
- pandas: Used for data manipulation and handling.
- matplotlib and seaborn: Used for visualizing the results.

## Steps Followed

### Data Loading and Preprocessing

- Loaded the Iris dataset from scikit-learn
- Separated the data into features (X) and target labels (y)
- Split the data into training (70%) and testing (30%) sets using train_test_split (test_size=0.3, random_state=42) for reproducibility

### Scaling

## Model Training

- Five classification models were trained:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Support Vector Machine (SVM)
  - Random Forest
- Each model was evaluated using accuracy, confusion matrix, and classification reports

## New Prediction

A new flower sample [1.0, 2.0, 3.0, 4.0, 3.0] was tested, resulting in:

- Example output: ['Iris-virginica']

## Conclusion

All models achieved 100% accuracy on the test dataset, successfully predicting the species of the new sample as Iris-virginica.

## How to Use

1. Install required libraries:

```bash
pip install scikit-learn pandas matplotlib seaborn
```

2. Run the Python script to train models and test predictions

## Results

### Summary

All trained models achieved perfect accuracy on the test split of the Iris dataset.

### Scaling

- Features were standardized using StandardScaler so all features share the same scale, which helps improve and stabilize model performance.

### Model Training

Five classification models were trained:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest

### Evaluation

- Each model was evaluated using accuracy, a confusion matrix, and a classification report (precision, recall, F1-score).

### New Prediction

- A new sample with measurements [1.0, 2.0, 3.0, 4.0, 3.0] was passed to each trained model.
- Example output from the models:
  - ['Iris-virginica']

### Conclusion

- All models performed excellently on the test dataset, reaching 100% accuracy and correctly predicting the species for the provided example sample.

### How to Use

1. Install required libraries:

```bash
pip install scikit-learn pandas matplotlib seaborn
```

2. Run the provided Python script to train the models, evaluate them, and test new sample predictions.

### Results

All models achieved 100% accuracy on the Iris test set, demonstrating reliable classification of Iris species based on the measured features.
