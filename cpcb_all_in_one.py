# ============================================================
# CPCB AIR QUALITY ANALYSIS AND PREDICTIVE ANALYSIS
# All syllabus methods and visualizations in ONE Python file
#
# Keep this Python file and cpcb_air_quality.csv in the same folder.
# Run: py cpcb_all_in_one.py
# ============================================================


# -------------------- IMPORT LIBRARIES --------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


# ============================================================
# UNIT I: INTRODUCTION, DATA LOADING AND PREPROCESSING
# ============================================================

# Supervised learning: the model learns from examples with known answers.
# Regression predicts a numerical value.
# Classification predicts a category.
# Unsupervised learning finds patterns without a known answer.
# This assignment demonstrates the supervised methods in Units II and III.

# Step 1: Load the CPCB CSV file.
data = pd.read_csv("cpcb_air_quality.csv")

print("FIRST FIVE ROWS")
print(data.head())

print("\nDATASET SIZE (rows, columns)")
print(data.shape)

print("\nCOLUMN NAMES")
print(data.columns)

# Step 2: Check missing values.
print("\nMISSING VALUES")
print(data.isnull().sum())

# Step 3: Keep columns used in this assignment.
data = data[
    ["state", "city", "station", "last_update", "pollutant_id",
     "pollutant_min", "pollutant_max", "pollutant_avg"]
].copy()

# Step 4: Remove rows where the target average reading is missing.
data = data.dropna(subset=["pollutant_avg"])

# Step 5: Fill missing minimum and maximum readings with their column mean.
data["pollutant_min"] = data["pollutant_min"].fillna(data["pollutant_min"].mean())
data["pollutant_max"] = data["pollutant_max"].fillna(data["pollutant_max"].mean())

print("\nROWS AFTER CLEANING:", len(data))


# ============================================================
# VISUALIZATION 1: NUMBER OF RECORDS FOR EACH POLLUTANT
# Bar chart
# ============================================================

pollutant_count = data["pollutant_id"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(pollutant_count.index, pollutant_count.values)
plt.title("Number of Records for Each Pollutant")
plt.xlabel("Pollutant")
plt.ylabel("Number of records")
plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 2: AVERAGE READING BY POLLUTANT
# Bar chart
# Note: pollutants may use different units. Do not compare raw values
# as if they were a single AQI scale.
# ============================================================

pollutant_average = data.groupby("pollutant_id")["pollutant_avg"].mean()

plt.figure(figsize=(8, 5))
plt.bar(pollutant_average.index, pollutant_average.values)
plt.title("Average Recorded Value by Pollutant")
plt.xlabel("Pollutant")
plt.ylabel("Average recorded value")
plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 3: DISTRIBUTION OF POLLUTANT READINGS
# Histogram
# ============================================================

plt.figure(figsize=(8, 5))
plt.hist(data["pollutant_avg"], bins=20, edgecolor="black")
plt.title("Distribution of Average Pollutant Readings")
plt.xlabel("Average pollutant reading")
plt.ylabel("Number of records")
plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 4: MINIMUM READING VS MAXIMUM READING
# Scatter plot
# ============================================================

plt.figure(figsize=(8, 5))
plt.scatter(data["pollutant_min"], data["pollutant_max"], alpha=0.5)
plt.title("Minimum Reading vs Maximum Reading")
plt.xlabel("Minimum reading")
plt.ylabel("Maximum reading")
plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 5: TOP 10 CITIES BY NUMBER OF RECORDS
# Horizontal bar chart
# This counts records; it does not rank cities by pollution.
# ============================================================

city_count = data["city"].value_counts().head(10)

plt.figure(figsize=(9, 5))
plt.barh(city_count.index, city_count.values)
plt.title("Top 10 Cities by Number of Records")
plt.xlabel("Number of records")
plt.ylabel("City")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 6: POLLUTANT DISTRIBUTION
# Box plot
# ============================================================

pollutant_names = []
pollutant_values = []

for pollutant in data["pollutant_id"].unique():
    pollutant_names.append(pollutant)
    pollutant_values.append(
        data[data["pollutant_id"] == pollutant]["pollutant_avg"]
    )

plt.figure(figsize=(9, 5))
plt.boxplot(pollutant_values, labels=pollutant_names)
plt.title("Distribution of Readings for Each Pollutant")
plt.xlabel("Pollutant")
plt.ylabel("Average recorded value")
plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 7: CORRELATION HEATMAP
# Correlation shows how numerical columns are related.
# ============================================================

correlation = data[
    ["pollutant_min", "pollutant_max", "pollutant_avg"]
].corr()

plt.figure(figsize=(6, 5))
plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(3), correlation.columns, rotation=30)
plt.yticks(range(3), correlation.columns)

# Display the correlation number inside each square.
for row in range(3):
    for column in range(3):
        plt.text(
            column, row, round(correlation.iloc[row, column], 2),
            ha="center", va="center"
        )

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# UNIT II: SUPERVISED LEARNING - REGRESSION
# Target column: pollutant_avg
# ============================================================

# Correlation: check the relationship between the numerical columns.
print("\nCORRELATION TABLE")
print(correlation)

# X contains input columns. y contains the value to be predicted.
X = data[["pollutant_min", "pollutant_max"]]
y = data["pollutant_avg"]

# Split data into 80% training data and 20% testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ---------------- SIMPLE LINEAR REGRESSION ----------------
# Simple Linear Regression uses ONE input column.
simple_model = LinearRegression()
simple_model.fit(X_train[["pollutant_max"]], y_train)
simple_prediction = simple_model.predict(X_test[["pollutant_max"]])

print("\nSIMPLE LINEAR REGRESSION")
print("MAE:", mean_absolute_error(y_test, simple_prediction))
print("MSE:", mean_squared_error(y_test, simple_prediction))
print("RMSE:", mean_squared_error(y_test, simple_prediction) ** 0.5)
print("R-squared:", r2_score(y_test, simple_prediction))

# ---------------- MULTIPLE LINEAR REGRESSION ----------------
# Multiple Linear Regression uses TWO input columns.
# LinearRegression finds coefficients using the Ordinary Least Squares (OLS) method.
multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)
multiple_prediction = multiple_model.predict(X_test)

print("\nMULTIPLE LINEAR REGRESSION / OLS")
print("Intercept:", multiple_model.intercept_)
print("Coefficients:", multiple_model.coef_)
print("MAE:", mean_absolute_error(y_test, multiple_prediction))
print("MSE:", mean_squared_error(y_test, multiple_prediction))
print("RMSE:", mean_squared_error(y_test, multiple_prediction) ** 0.5)
print("R-squared:", r2_score(y_test, multiple_prediction))

# ---------------- POLYNOMIAL REGRESSION ----------------
# Polynomial Regression adds squared terms to represent a curved relationship.
polynomial = PolynomialFeatures(degree=2)
X_train_poly = polynomial.fit_transform(X_train)
X_test_poly = polynomial.transform(X_test)

polynomial_model = LinearRegression()
polynomial_model.fit(X_train_poly, y_train)
polynomial_prediction = polynomial_model.predict(X_test_poly)

print("\nPOLYNOMIAL REGRESSION")
print("MAE:", mean_absolute_error(y_test, polynomial_prediction))
print("MSE:", mean_squared_error(y_test, polynomial_prediction))
print("RMSE:", mean_squared_error(y_test, polynomial_prediction) ** 0.5)
print("R-squared:", r2_score(y_test, polynomial_prediction))

# Regression visualization: actual values compared with predicted values.
plt.figure(figsize=(7, 5))
plt.scatter(y_test, multiple_prediction, alpha=0.5)
plt.xlabel("Actual average reading")
plt.ylabel("Predicted average reading")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.tight_layout()
plt.show()


# ============================================================
# UNIT III: SUPERVISED LEARNING - CLASSIFICATION
# Target: high_reading (1 = high, 0 = low)
# ============================================================

# The pollutants have different measurement scales.
# So, define HIGH/LOW by comparing each reading with the median
# of the same pollutant type. This is a teaching label, not an official AQI category.

data["pollutant_median"] = data.groupby(
    "pollutant_id"
)["pollutant_avg"].transform("median")

data["high_reading"] = (
    data["pollutant_avg"] >= data["pollutant_median"]
).astype(int)

# Use minimum and maximum readings to predict the HIGH/LOW label.
X = data[["pollutant_min", "pollutant_max"]]
y = data["high_reading"]

# Split the classification data into training and testing parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standardization puts the input columns on a similar scale.
# It is useful for KNN and SVM.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- LOGISTIC REGRESSION ----------------
# Logistic Regression predicts a category, such as 0 or 1.
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)
logistic_prediction = logistic_model.predict(X_test_scaled)
logistic_probability = logistic_model.predict_proba(X_test_scaled)[:, 1]

# ---------------- K-NEAREST NEIGHBORS ----------------
# KNN is lazy learning: it checks the nearest training examples.
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_prediction = knn_model.predict(X_test_scaled)
knn_probability = knn_model.predict_proba(X_test_scaled)[:, 1]

# ---------------- NAIVE BAYES ----------------
# Naive Bayes uses probability and Bayes' theorem.
naive_model = GaussianNB()
naive_model.fit(X_train_scaled, y_train)
naive_prediction = naive_model.predict(X_test_scaled)
naive_probability = naive_model.predict_proba(X_test_scaled)[:, 1]

# ---------------- DECISION TREE ----------------
# A Decision Tree learns simple decision rules from the data.
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)
tree_prediction = tree_model.predict(X_test)
tree_probability = tree_model.predict_proba(X_test)[:, 1]

# ---------------- SUPPORT VECTOR MACHINE ----------------
# SVM finds a boundary that separates the two categories.
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_prediction = svm_model.predict(X_test_scaled)
svm_probability = svm_model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# CLASSIFICATION EVALUATION
# Print every classification measure mentioned in the syllabus.
# ============================================================

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test, logistic_prediction))
print("Log Loss:", log_loss(y_test, logistic_probability))
print("AUC:", roc_auc_score(y_test, logistic_probability))
print("Precision:", precision_score(y_test, logistic_prediction, zero_division=0))
print("Recall:", recall_score(y_test, logistic_prediction, zero_division=0))
print("F1 Score:", f1_score(y_test, logistic_prediction, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, logistic_prediction))

print("\nKNN RESULTS")
print("Accuracy:", accuracy_score(y_test, knn_prediction))
print("Log Loss:", log_loss(y_test, knn_probability))
print("AUC:", roc_auc_score(y_test, knn_probability))
print("Precision:", precision_score(y_test, knn_prediction, zero_division=0))
print("Recall:", recall_score(y_test, knn_prediction, zero_division=0))
print("F1 Score:", f1_score(y_test, knn_prediction, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_prediction))

print("\nNAIVE BAYES RESULTS")
print("Accuracy:", accuracy_score(y_test, naive_prediction))
print("Log Loss:", log_loss(y_test, naive_probability))
print("AUC:", roc_auc_score(y_test, naive_probability))
print("Precision:", precision_score(y_test, naive_prediction, zero_division=0))
print("Recall:", recall_score(y_test, naive_prediction, zero_division=0))
print("F1 Score:", f1_score(y_test, naive_prediction, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, naive_prediction))

print("\nDECISION TREE RESULTS")
print("Accuracy:", accuracy_score(y_test, tree_prediction))
print("Log Loss:", log_loss(y_test, tree_probability))
print("AUC:", roc_auc_score(y_test, tree_probability))
print("Precision:", precision_score(y_test, tree_prediction, zero_division=0))
print("Recall:", recall_score(y_test, tree_prediction, zero_division=0))
print("F1 Score:", f1_score(y_test, tree_prediction, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, tree_prediction))

print("\nSUPPORT VECTOR MACHINE RESULTS")
print("Accuracy:", accuracy_score(y_test, svm_prediction))
print("Log Loss:", log_loss(y_test, svm_probability))
print("AUC:", roc_auc_score(y_test, svm_probability))
print("Precision:", precision_score(y_test, svm_prediction, zero_division=0))
print("Recall:", recall_score(y_test, svm_prediction, zero_division=0))
print("F1 Score:", f1_score(y_test, svm_prediction, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_prediction))


# ============================================================
# FINAL NOTE FOR THE ASSIGNMENT
# ============================================================

# The uploaded CPCB file has 3,801 rows but only one distinct update time.
# It is a snapshot, not historical time-series data.
# These models demonstrate the syllabus algorithms on the available data.
# They do not forecast future AQI over time.
#
# pollutant_avg is closely related to pollutant_min and pollutant_max.
# Explain this limitation when discussing the regression results.
# The HIGH/LOW classification label is based on each pollutant's median;
# it is not an official CPCB AQI category.
