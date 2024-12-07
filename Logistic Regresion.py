# Import necessary libraries
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score, classification_report, confusion_matrix, log_loss

# Load the dataset
df = pd.read_csv('heart.csv')

# Display the distribution of the target variable (optional)
#print(df["output"].value_counts())

# Optionally visualize a histogram of the cholesterol levels
# df.hist(column="chol", bins=50)

# Select feature columns and target variable
# Features: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar,
# resting electrocardiographic results, maximum heart rate, exercise induced angina,
# oldpeak, slope, number of major vessels, thalassemia
x = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
         'exng', 'oldpeak', 'slp', 'caa', 'thall']].values

# Target variable (output)
y = df['output'].values.astype(int)

# Standardize the features
X = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Initialize and train the Logistic Regression model
LR = LogisticRegression(C=0.01, solver="liblinear").fit(X_train, y_train)

# Make predictions on the test set
y_hat = LR.predict(X_test)

# Get predicted probabilities for the test set
y_hat_prob = LR.predict_proba(X_test)

# Calculate and print the Jaccard Score
print("Jaccard score =", jaccard_score(y_test, y_hat, pos_label=1))

# (Optional) Print confusion matrix and classification report for more insights
print(confusion_matrix(y_test, y_hat, labels=[1, 0]))
print(classification_report(y_test, y_hat))

# Calculate and print log loss to evaluate the model performance
print("Log loss =", log_loss(y_test, y_hat_prob))
