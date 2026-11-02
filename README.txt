CPCB ALL-IN-ONE ASSIGNMENT

Files:
1. cpcb_all_in_one.py — one Python program with all methods and charts.
2. cpcb_air_quality.csv — your uploaded dataset.

Install once:
py -m pip install pandas matplotlib scikit-learn

Run:
py cpcb_all_in_one.py

The program includes:
- Unit I: loading, inspecting and cleaning data; explanation of supervised/unsupervised learning.
- Unit II: correlation, Simple Linear Regression, Multiple Linear Regression, OLS, Polynomial Regression,
  MAE, MSE, RMSE, R-squared.
- Unit III: Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM,
  Accuracy, Log Loss, AUC, Precision, Recall, F1 and Confusion Matrix.
- Seven descriptive charts plus an actual-vs-predicted regression chart.

Dataset limitation:
3,801 rows; one distinct last_update timestamp. This is a snapshot, not a time series,
so it cannot support honest future AQI forecasting. Raw pollutant values also have
different units/scales. The HIGH/LOW classification label is a teaching label based
on the median within each pollutant, not an official CPCB AQI category.
