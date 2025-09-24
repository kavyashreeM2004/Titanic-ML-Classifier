# Titanic-ML-Classifier
This project applies Logistic Regression on the Titanic dataset (from Seaborn) to predict passenger survival. It demonstrates a full machine learning workflow including preprocessing, model training, evaluation, and visualization.
#Dataset
The dataset is loaded directly from Seaborn’s Titanic dataset.
It contains demographic and travel information about Titanic passengers, such as age, gender, fare, and survival status.
Features Used:
pclass – Passenger class (1st, 2nd, 3rd)

age – Passenger age (missing values filled with median)

sibsp – Number of siblings/spouses aboard

parch – Number of parents/children aboard

fare – Passenger fare (missing values dropped if any)

sex – Encoded as sex_male (1 = male, 0 = female)

embarked – Encoded as embarked_Q, embarked_S

class – Encoded as class_Second, class_Third

Target:
survived – 0 = No, 1 = Yes

Workflow
Data Preprocessing

Removed irrelevant/highly missing columns (deck, embark_town, alive)

Imputed missing values:

age with median

embarked with mode

One-hot encoded categorical features

Selected key features for modeling

Model Training

Split data into train (80%) and test (20%) sets

Trained a Logistic Regression model with max_iter=1000

Evaluation

Accuracy score

ROC AUC score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

ROC Curve visualization

Metrics Example (will vary slightly)
Accuracy: ~0.77

ROC AUC: ~0.84

Confusion Matrix: Shows TP, TN, FP, FN distribution

ROC Curve: Plots True Positive Rate vs False Positive Rate

Installation and Usage
bash
# Clone the repository
git clone https://github.com/your-username/titanic-logistic-regression.git
cd titanic-logistic-regression

# Install dependencies
pip install pandas numpy scikit-learn seaborn matplotlib
Run the script:

bash
python titanic_logistic_regression.py
Visualization
The project plots the ROC Curve to visualize model performance across thresholds.

Future Improvements
Try advanced models (Random Forest, XGBoost)

Perform feature scaling (StandardScaler or MinMaxScaler)

Apply hyperparameter tuning (GridSearchCV)

Handle missing values with advanced imputation techniques
