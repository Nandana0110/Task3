import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\AL SharQ\Downloads\intern\Housing.csv")
df_encoded = pd.get_dummies(df, drop_first=True)

X_simple = df_encoded[['area']]
y = df_encoded['price']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)
model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_s = model_simple.predict(X_test_s)
print("---- Simple Linear Regression (using 'area') ----")
print("MAE:", mean_absolute_error(y_test_s, y_pred_s))
print("MSE:", mean_squared_error(y_test_s, y_pred_s))
print("R² Score:", r2_score(y_test_s, y_pred_s))
print("Coefficient:", model_simple.coef_[0])
print("Intercept:", model_simple.intercept_)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test_s['area'], y=y_test_s, label='Actual')
plt.plot(X_test_s['area'], y_pred_s, color='red', label='Predicted')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

X_multi = df_encoded.drop("price", axis=1)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)
print("\n---- Multiple Linear Regression (all features) ----")
print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("R² Score:", r2_score(y_test_m, y_pred_m))
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_m, y=y_pred_m)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()
coefficients = pd.Series(model_multi.coef_, index=X_multi.columns).sort_values(ascending=False)
print("\nModel Coefficients (Multiple Linear Regression):")
print(coefficients)