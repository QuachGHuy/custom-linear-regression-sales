import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data, split_data, data_scaler, add_bias
from src.model import LinearRegression

# 1. Get data file_path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
clean_data_path = os.path.join(project_root, "data/processed/cleaned_advertising.csv")

# 2. Load data
X, y = load_data(clean_data_path)

# 3. Split data to Train/Test (Test size = 0.15)
X_train, X_test, y_train, y_test = split_data(X, y, 0.15)

# 4. Split Train data to Train/Valid (Valid size = 0.15)
X_train, X_valid, y_train, y_valid = split_data(X_train, y_train, 0.15)

# 5. Data scaling
# Fit and transform X_train
X_train_scaled, mean, std = data_scaler(X_train)

# Transform X_valid, X_test
X_valid_scaled, _, _ = data_scaler(X_valid, mean, std)
X_test_scaled, _, _ = data_scaler(X_test, mean, std)

# 6. Add Bias
X_train_scaled = add_bias(X_train_scaled)
X_valid_scaled = add_bias(X_valid_scaled)
X_test_scaled = add_bias(X_test_scaled)

print("Data loaded and pre-processed successfully!")
print(f"Train shape: {X_train_scaled.shape}")
print(f"Valid shape: {X_valid_scaled.shape}")
print(f"Test shape: {X_test_scaled.shape}")

# 7. Train model
print("---Training model---")
model = LinearRegression(
    data_X=X_train_scaled,
    data_Y=y_train,
    learning_rate=0.02,
    epochs=200,
    batch_size=64
)

model.fit()

# 8. Evaluate on Validation Set
print("\n--- Validation ---")
y_pred_valid = model.predict(X_valid_scaled)

rmse_valid = np.sqrt(model.compute_mse(y_valid, y_pred_valid))
r2_valid = model.compute_r2_score(y_valid, y_pred_valid)

print(f"Validation RMSE: {rmse_valid:.4f}")
print(f"Validation R2:   {r2_valid:.4f}")

# 9. (Optional) Final Test
# Only run this step WHEN AND ONLY WHEN you are satisfied with the Validation
print("\n--- Final Test ---")
y_pred_test = model.predict(X_test_scaled)

rmse_test = np.sqrt(model.compute_mse(y_test, y_pred_test))
r2_test = model.compute_r2_score(y_test, y_pred_test)

print(f"Test RMSE: {rmse_test:.4f}")
print(f"Test R2:   {r2_test:.4f}")

# 10. Save Model Weights
model_dir = os.path.join(project_root, 'models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'weights.npy')

np.save(os.path.join(model_dir, 'mean.npy'), mean)
np.save(os.path.join(model_dir, 'std.npy'), std)
print(f"✅ Saved scaler parameters (mean, std) to: {model_dir}")

weights = model.theta
np.save(model_path, weights)
print(f"✅ Model weights saved successfully at: {model_path}")
print("Ready for Inference/Prediction!")