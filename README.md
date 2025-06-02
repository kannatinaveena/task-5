🧠 Heart Disease Classification with Decision Trees & Random Forests
This repository contains the solution for Task 5 of my AI & ML Internship, which involves training and evaluating tree-based models on the Heart Disease Dataset using Decision Trees and Random Forests.

📂 Dataset
Source: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Target Variable: target (1 = Heart Disease, 0 = No Heart Disease)

📌 Objectives
Train a Decision Tree Classifier and visualize it.

Analyze overfitting and control tree depth.

Train a Random Forest Classifier and compare accuracy.

Interpret feature importances.

Evaluate models using cross-validation.

🛠️ Tools Used
Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

🧪 Model Evaluation
Accuracy Score

Classification Report (Precision, Recall, F1-Score)

Confusion Matrix

Feature Importance Visualization

Overfitting Graph (Depth vs Accuracy)

Cross-Validation Scores

🧾 Steps Performed
✅ 1. Data Preprocessing
Loaded the dataset from CSV.

Checked for missing values.

Split into train and test sets (80/20).

Separated features and target variable.

✅ 2. Decision Tree Training
Trained a DecisionTreeClassifier.

Visualized the decision tree using plot_tree().

Measured accuracy and evaluated metrics.

✅ 3. Overfitting Analysis
Trained trees with increasing max_depth.

Plotted train vs test accuracy to detect overfitting.

✅ 4. Random Forest Classifier
Trained a RandomForestClassifier.

Compared its performance to the Decision Tree.

✅ 5. Feature Importance
Plotted most important features based on the Random Forest model.

✅ 6. Cross-Validation
Used cross_val_score to validate models on the full dataset (5-fold CV).

📎 Files Included
heart.csv – Dataset file

decision_tree_random_forest.ipynb – Main Jupyter Notebook

README.md – Project summary (this file)

👩‍💻 Author
Name: Kannati Naveena

GitHub: https://github.com/kannatinaveena

Internship: AI & ML Internship

Task: Task 5 – Tree-Based Models

Date: June 2025
