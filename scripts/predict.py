import sys
import os
import numpy as np
from pydantic import ValidationError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import data_scaler, add_bias
def load_model_artifacts():
    """
    Load weights, mean, std from models dir
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(project_root,"models")

    try:
        w = np.load(os.path.join(model_dir,"weights.npy"))
        mean = np.load(os.path.join(model_dir,"mean.npy"))
        std = np.load(os.path.join(model_dir,"std.npy"))
        return w, mean, std 
    
    except FileNotFoundError:
        print(f"ERROR: Model not found at {model_dir}. Please run train.py first!")
        sys.exit(1)

from src.schemas import AdvertisingInput
def get_user_input():
    """
    Get user input
    """
    print("\n--- SALES FORCASTING ---")
    print("Please input advertising budget values (unit: 1000 $):")
    while True:
        try:
            tv = float(input("TV: "))
            radio = float(input("Radio: "))
            newspaper = float(input("Newspaper: "))

            input_data = AdvertisingInput(
                    tv=tv,
                    radio=radio,
                    newspaper=newspaper
                )
            return input_data
        except (ValueError, ValidationError) as e:
            print(f"ERROR: {e}")

def predict():
    # 1. Load model
    w, mean, std  = load_model_artifacts()

    # 2. Get input value
    user_input = get_user_input()
    X_new = np.array([[user_input.tv, user_input.radio, user_input.newspaper]])

    # 3. Preprocessing
    # Scaling
    X_scaled, _, _ = data_scaler(X_new, mean, std)
    
    # Add Bias 
    X_scaled = add_bias(X_scaled)

    # 4. Predict
    prediction = X_scaled @ w

    # 5. Result
    sales = prediction[0]

    print("\n----------------RESULTS----------------")
    print(f"Predicted Sales: {sales:.2f} (1000 $)")
    print("-----------------------------------------")

if __name__ == "__main__":
    predict()