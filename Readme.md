# Deep Learning Final Project

## Overview

This project builds and evaluates an Artificial Neural Network (ANN) to predict house prices based on various features. The dataset is preprocessed, scaled, and used to train a deep learning model using TensorFlow/Keras.

## Dataset

The dataset contains multiple features related to real estate properties, including:

- **Square Footage**
- **Number of Rooms**
- **Location Features**
- **Historical Price Trends**
- **Other relevant real estate attributes**

## Technologies Used

- **Python**
- **TensorFlow/Keras**
- **Scikit-Learn**
- **Pandas**
- **NumPy**
- **Matplotlib/Seaborn**

## Model Architecture

The ANN consists of multiple dense layers with ReLU activation functions:

```python
ann_model = Sequential([
    Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # Output layer
])
```

## Data Preprocessing

1. **Feature Scaling:** MinMaxScaler or StandardScaler applied to input features.
2. **Handling Outliers:** Identified and removed extreme house prices using IQR.
3. **Feature Engineering:** Added new features such as `rooms_per_sqft`.

## Model Evaluation

The model's performance was evaluated using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

### Final Results:

- **Test RMSE:** ~1,357,926.85
- **Test R²:** ~0.6352

## Improvements & Next Steps

- Implement better outlier handling
- Tune hyperparameters (batch size, learning rate, regularization)
- Try alternative models (Random Forest, XGBoost) for comparison
- Enhance feature selection and engineering

## How to Run

1. Install dependencies:
   ```sh
   pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
   ```
2. Run the Jupyter Notebook to train and evaluate the model.

## Contributors

- [Your Name]

## License

This project is open-source under the MIT License.
