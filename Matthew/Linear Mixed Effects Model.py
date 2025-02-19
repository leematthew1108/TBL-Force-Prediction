import pandas as pd
import statsmodels.api as sm

# Step 1: Data Preparation
# Assume 'data' is your DataFrame with columns: mass, wingspan, acceleration, angular_velocity, forces, moments
# Replace 'dataset.csv' with the actual dataset path if loading from a file.

# data = pd.read_csv('dataset.csv')

# Step 3: Model Specification
formula = "forces ~ mass + wingspan + acceleration + angular_velocity"
random_effects_formula = "0 + acceleration + angular_velocity"

# Step 4: Model Fitting
mixed_model = sm.MixedLM.from_formula(
    formula, data, groups=data["subject"], re_formula=random_effects_formula
)
result = mixed_model.fit()

# Step 5: Model Evaluation
print(result.summary())
