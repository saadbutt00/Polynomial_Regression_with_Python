import streamlit as st
import numpy as np

st.set_page_config(page_title="Polynomial Regression", layout="centered")

st.title("Polynomial Regression Model")

# --- Intro Section ---
st.markdown("Polynomial Regression fits a non-linear relationship between features and output using a polynomial function of the input.")
st.markdown("**Example**: Estimating a car’s price based on mileage and age, where effects may be curved/non-linear.")

# --- Step Guide ---
st.markdown("Here, how you can predict your values:")
st.markdown("""
**Step 1**: Enter X's values - You can also name X column with your desired one.

**Step 2**: Enter Y values - It's length should be equal to the length of X feature's values.

**Step 3**: Select your desired bias value.

**Step 4**: Click 'Submit' & your model will be trained.

**Step 5**: It will appear predict value side where you can enter your value & get your predicted answer.
""")

# --- User Inputs ---
num_features = st.number_input('Enter Number of Features', min_value=1, step=1)
degree = st.number_input('Enter Polynomial Degree', min_value=1, step=1)
bias = st.number_input('Enter Bias value', value=1.0)

if num_features:
    feature_names = []
    X_input = []

    st.markdown("### ✍️ Name Your Features and Enter Values")
    for i in range(num_features):
        col_name = st.text_input(f'Enter name for feature X{i}', value=f'X{i}')
        feature_names.append(col_name)
        values = st.text_area(f'Enter values for {col_name} (space-separated)', key=f'feat_{i}')
        if values:
            try:
                X_input.append(list(map(float, values.strip().split())))
            except ValueError:
                st.error(f"Invalid input in {col_name}. Ensure all values are numbers.")

    Y_input = st.text_area('Enter Y values (space-separated)')
    if Y_input:
        try:
            Y = np.array(list(map(float, Y_input.strip().split())))
        except ValueError:
            st.error("Invalid Y values.")

    if len(X_input) == num_features and Y_input:
        X = np.transpose(np.array(X_input))
        if len(Y) != len(X):
            st.error("Mismatch: Number of samples in X and Y must be equal.")
        else:
            if st.button("Submit"):
                # Build polynomial features
                poly_features = [np.full(X.shape[0], bias)]
                for i in range(1, degree + 1):
                    for j in range(num_features):
                        poly_features.append(X[:, j] ** i)

                X_poly = np.transpose(np.vstack(poly_features))

                # Train model
                XT = np.transpose(X_poly)
                theta = np.dot(np.linalg.pinv(XT @ X_poly), XT @ Y)
                y_pred = X_poly @ theta

                # Save to session_state
                st.session_state.trained = True
                st.session_state.theta = theta
                st.session_state.feature_names = feature_names
                st.session_state.degree = degree
                st.session_state.bias = bias
                st.session_state.Y = Y
                st.session_state.y_pred = y_pred

    # --- Prediction + Results ---
    if st.session_state.get("trained", False):
        st.success("✅ Model trained!")

        st.markdown("### 📊 Model Coefficients")
        coeff_table = {
            "Term": [f"Theta {i}" for i in range(len(st.session_state.theta))],
            "Value": [round(t, 4) for t in st.session_state.theta]
        }
        st.table(coeff_table)

        st.markdown("### 🔮 Make a Prediction")
        pred_vals = []
        for name in st.session_state.feature_names:
            val = st.number_input(f'Enter value for {name}', key=f'pred_{name}')
            pred_vals.append(val)

        if st.button("Predict"):
            # Build prediction feature vector
            pred_features = [st.session_state.bias]
            for i in range(1, st.session_state.degree + 1):
                for val in pred_vals:
                    pred_features.append(val ** i)

            pred_array = np.array(pred_features)
            y_prediction = pred_array @ st.session_state.theta

            st.success(f"🎯 Predicted Y value: `{np.round(y_prediction, 4)}`")

            # R² Score (from training)
            r2_1 = np.sum((st.session_state.Y - st.session_state.y_pred) ** 2)
            r2_2 = np.sum((st.session_state.Y - np.mean(st.session_state.Y)) ** 2)
            r2 = 1 - (r2_1 / r2_2 + 1e-10)

            st.info(f"📈 R² Score: `{np.round(r2 * 100, 2)}%`")

# streamlit run "f:/ML_with_Python/Polynomial Regression/app.py"