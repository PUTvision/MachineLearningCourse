import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Data: each row is [charging_time, usage_time]
data = np.array([
    [2.81,5.62],
    [7.14,8.00],
    [2.72,5.44],
    [3.87,7.74],
    [1.90,3.80],
    [7.82,8.00],
    [7.02,8.00],
    [5.50,8.00],
    [9.15,8.00],
    [4.87,8.00],
    [8.08,8.00],
    [5.58,8.00],
    [9.13,8.00],
    [0.14,0.28],
    [2.00,4.00],
    [5.47,8.00],
    [0.80,1.60],
    [4.37,8.00],
    [5.31,8.00],
    [0.00,0.00],
    [1.78,3.56],
    [3.45,6.90],
    [6.13,8.00],
    [3.53,7.06],
    [4.61,8.00],
    [1.76,3.52],
    [6.39,8.00],
    [0.02,0.04],
    [9.69,8.00],
    [5.33,8.00],
    [6.37,8.00],
    [5.55,8.00],
    [7.80,8.00],
    [2.06,4.12],
    [7.79,8.00],
    [2.24,4.48],
    [9.71,8.00],
    [1.11,2.22],
    [8.38,8.00],
    [2.33,4.66],
    [1.83,3.66],
    [5.94,8.00],
    [9.20,8.00],
    [1.14,2.28],
    [4.15,8.00],
    [8.43,8.00],
    [5.68,8.00],
    [8.21,8.00],
    [1.75,3.50],
    [2.16,4.32],
    [4.93,8.00],
    [5.75,8.00],
    [1.26,2.52],
    [3.97,7.94],
    [4.39,8.00],
    [7.53,8.00],
    [1.98,3.96],
    [1.66,3.32],
    [2.04,4.08],
    [11.72,8.00],
    [4.64,8.00],
    [4.71,8.00],
    [3.77,7.54],
    [9.33,8.00],
    [1.83,3.66],
    [2.15,4.30],
    [1.58,3.16],
    [9.29,8.00],
    [1.27,2.54],
    [8.49,8.00],
    [5.39,8.00],
    [3.47,6.94],
    [6.48,8.00],
    [4.11,8.00],
    [1.85,3.70],
    [8.79,8.00],
    [0.13,0.26],
    [1.44,2.88],
    [5.96,8.00],
    [3.42,6.84],
    [1.89,3.78],
    [1.98,3.96],
    [5.26,8.00],
    [0.39,0.78],
    [6.05,8.00],
    [1.99,3.98],
    [1.58,3.16],
    [3.99,7.98],
    [4.35,8.00],
    [6.71,8.00],
    [2.58,5.16],
    [7.37,8.00],
    [5.77,8.00],
    [3.97,7.94],
    [3.65,7.30],
    [4.38,8.00],
    [8.06,8.00],
    [8.05,8.00],
    [1.10,2.20],
    [6.65,8.00]
])


def first_approach():
    # Separate the charging times (X) and usage times (y)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]

    # Split the data into training (80%) and testing (20%) sets using random seed 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit a linear regression model on the training data
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print("Coefficient:", reg.coef_[0], "Intercept:", reg.intercept_)

    # Predict on the test set
    y_pred_test = reg.predict(X_test)

    # Cap predictions at 8 hours since the maximum usage time is 8
    y_pred_test_capped = np.minimum(y_pred_test, 8)

    # Calculate the Mean Absolute Error (MAE) on the test set
    mae = mean_absolute_error(y_test, y_pred_test_capped)
    print("Mean Absolute Error (MAE) on test set:", mae)

    # Visualize training data, test data, and the model predictions
    plt.figure(figsize=(8,5))
    plt.scatter(X_train, y_train, label='Training Data', color='blue', alpha=0.6)
    plt.scatter(X_test, y_test, label='Test Data', color='orange', alpha=0.8)
    # Create a range for plotting the regression line
    X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    y_range = reg.predict(X_range)
    y_range_capped = np.minimum(y_range, 8)
    plt.plot(X_range, y_range, color='red', label='Linear Prediction')
    plt.plot(X_range, y_range_capped, color='green', linestyle='--', label='Capped Prediction')
    plt.xlabel("Charge Time (hours)")
    plt.ylabel("Usage Time (hours)")
    plt.legend()
    plt.show()


def second_approach():
    # Separate the charging times (X) and usage times (y)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]

    # Split the data into training (80%) and testing (20%) sets using random seed 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ####################################
    # Approach 1: Polynomial Regression
    ####################################
    # We use a degree 2 polynomial to allow for curvature that might better capture the saturation.
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)

    # Predict on the test set and cap predictions at 8 hours
    y_poly_pred = poly_reg.predict(X_test_poly)
    y_poly_pred_capped = np.minimum(y_poly_pred, 8)

    mae_poly = mean_absolute_error(y_test, y_poly_pred_capped)
    print("Polynomial Regression MAE:", mae_poly)

    ####################################
    # Approach 2: Piecewise Regression
    ####################################
    # In the piecewise approach, we fit a linear model only on the regime where the laptop is not saturated (usage < 8).
    # We choose a threshold based on the training data. Here we take the 90th percentile of charging times where usage < 8.
    mask = y_train < 8
    if np.any(mask):
        threshold = np.percentile(X_train[mask], 90)
    else:
        threshold = np.max(X_train)

    print("Chosen threshold for piecewise model:", threshold)

    # Fit a linear model on training data with charging time <= threshold.
    mask_train = X_train.flatten() <= threshold
    X_train_piece = X_train[mask_train]
    y_train_piece = y_train[mask_train]
    piece_reg = LinearRegression()
    piece_reg.fit(X_train_piece, y_train_piece)

    # Define a prediction function for the piecewise model:
    # If the charging time is less than or equal to the threshold, use the linear model; otherwise, predict 8.
    def piecewise_predict(x):
        # x is expected to be a 2D array.
        preds = np.where(x.flatten() <= threshold, piece_reg.predict(x), 8)
        return preds

    # Predict on the test set and calculate MAE
    y_piece_pred = piecewise_predict(X_test)
    mae_piece = mean_absolute_error(y_test, y_piece_pred)
    print("Piecewise Regression MAE:", mae_piece)

    ####################################
    # Visualization
    ####################################
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='gray', label='Test Data', alpha=0.8)

    # Create a range for plotting model predictions.
    X_range = np.linspace(np.min(X_train), np.max(X_train), 200).reshape(-1, 1)

    # Polynomial model predictions over the range.
    y_range_poly = poly_reg.predict(poly.transform(X_range))
    y_range_poly_capped = np.minimum(y_range_poly, 8)
    plt.plot(X_range, y_range_poly, color='blue', label='Polynomial Prediction')
    plt.plot(X_range, y_range_poly_capped, color='blue', linestyle='--', label='Capped Poly Prediction')

    # Piecewise model predictions over the range.
    y_range_piece = piecewise_predict(X_range)
    plt.plot(X_range, y_range_piece, color='red', label='Piecewise Prediction')

    plt.xlabel("Charge Time (hours)")
    plt.ylabel("Usage Time (hours)")
    plt.legend()
    plt.title("Model Predictions Comparison")
    plt.show()


def main():
    first_approach()
    second_approach()


if __name__ == '__main__':
    main()