import streamlit as st
import pandas as pd
from src.preprocess import load_dataframe, detect_problem_type, split, encoding_features, encoding_target
from src.model_selection import get_models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, confusion_matrix,mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, mean_absolute_error, silhouette_score,confusion_matrix
import joblib, os
import numpy as np

st.title(f"Welcome to Machine Learning Playground ")

uploaded_file = st.file_uploader(f"Upload CSV", type='csv')

if uploaded_file is not None:

    df = load_dataframe(uploaded_file)
    df.dropna(inplace=True)
    st.write(f"### Dataset Preview")
    st.write(df.head())
    st.write(f"### Dataset Shape")
    st.write(df.shape)
    st.write(f"### Summary Statistics")
    st.write(df.describe())

    all_features = df.columns.tolist()
    target_col = st.selectbox("Select target feature: ", ["None"]  + all_features)

    problem_type = detect_problem_type(df, target_col)

    st.write(f"Problem Type is ", problem_type)

    available_models = get_models(problem_type)
    model_choice = st.selectbox("Select a Model:", list(available_models.keys()))

    params = {}
    if model_choice == "Decision Tree":
        params["max_depth"] = st.slider("Max Depth", 1, 20, 5)
        params["min_samples_split"] = st.slider("Min Samples Split", 2, 10, 2)

    elif model_choice == "Random Forest":
        params["n_estimators"] = st.slider("Number of Trees", 10, 200, 100)
        params["max_depth"] = st.slider("Max Depth", 1, 20, 5)

    elif model_choice in ["AdaBoost", "Gradient Boosting"]:
        params["n_estimators"] = st.slider("Number of Estimators", 10, 200, 50)
        params["learning_rate"] = st.number_input("Learning Rate", 0.01, 1.0, 0.1)

    elif model_choice == "Voting Classifier":
        base_models = st.multiselect(
            "Select base models", ["Logistic Regression", "Decision Tree", "SVM"]
        )
        voting_type = st.selectbox("Voting Type", ["hard", "soft"])

        # Map names to actual sklearn classes
        base_estimators = []
        for bm in base_models:
            if bm == "Logistic Regression":
                base_estimators.append(("lr", LogisticRegression()))
            elif bm == "Decision Tree":
                base_estimators.append(("dt", DecisionTreeClassifier()))
            elif bm == "SVM":
                base_estimators.append(("svc", SVC(probability=True)))
        params["estimators"] = base_estimators
        params["voting"] = voting_type

    elif model_choice == "K-means":
        params["n_clusters"] = st.slider("Number of Clusters", 2, 10, 3)
        params["max_iter"] = st.slider("Max Iterations", 100, 1000, 300)

if st.button("Train Model!"):
    X, y = split(df, target_col)
    X_encoded = encoding_features(X)
    y_encoded = encoding_target(y)

    model_class = available_models[model_choice]

    # Voting Classifier requires special handling
    if model_choice == "Voting Classifier":
        model = model_class(**params)
    else:
        model = model_class(**params)

    model.fit(X_encoded, y_encoded)

    # Save trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "trained_model.pkl"))

    st.success(f"{model_choice} trained and saved successfully!")

st.title("ML Playground - Test Data Predictions")

# Upload Test Data
uploaded_test_file = st.file_uploader("Upload Test CSV", type='csv')

if uploaded_test_file is not None:

    test_df = pd.read_csv(uploaded_test_file)
    test_df.dropna(inplace=True)
    st.write("### Test Dataset Preview")
    st.write(test_df.head())
    
    # Load trained model
    model_path = os.path.join("models", "trained_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.warning("No trained model found. Please train a model first!")
    
    # Load saved artifacts
    ohe = joblib.load(os.path.join("artifacts", "onehot_encoder.pkl"))
    scaler = joblib.load(os.path.join("artifacts", "standard_scaler.pkl"))
    
    # If target exists in test set
    target_col = st.selectbox("Select target feature (optional, if known):", ["None"] + test_df.columns.tolist())
    if target_col != "None":
        y_test = test_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test_encoded = encoding_target(y_test)
    else:
        X_test = test_df
        y_test_encoded = None

    # Encode and scale test features
    categorical_cols = X_test.select_dtypes(include="object").columns
    numeric_cols = X_test.select_dtypes(exclude="object").columns

    X_categorical = ohe.transform(X_test[categorical_cols])
    X_numeric = scaler.transform(X_test[numeric_cols])
    X_test_processed = np.hstack([X_categorical, X_numeric])

# Predict
y_pred = model.predict(X_test_processed)

# Metrics & Visualizations if target is available
if y_test_encoded is not None:
    problem_type = detect_problem_type(test_df, target_col)
    
    if problem_type.lower() == "classification":
        # Existing metrics
        st.write(f"Accuracy: {accuracy_score(y_test_encoded, y_pred):.4f}")
        st.write(f"Precision: {precision_score(y_test_encoded, y_pred, average='weighted'):.4f}")
        st.write(f"Recall: {recall_score(y_test_encoded, y_pred, average='weighted'):.4f}")
        st.write(f"F1-score: {f1_score(y_test_encoded, y_pred, average='weighted'):.4f}")

        # Confusion matrix normalized
        cm = confusion_matrix(y_test_encoded, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # ROC Curve (binary only)
        if len(np.unique(y_test_encoded)) == 2:
            fpr, tpr, _ = roc_curve(y_test_encoded, model.predict_proba(X_test_processed)[:,1])
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0,1],[0,1],'r--')
            ax.legend()
            st.pyplot(fig)

    elif problem_type.lower() == "regression":
        st.write(f"R² Score: {r2_score(y_test_encoded, y_pred):.4f}")
        st.write(f"MSE: {mean_squared_error(y_test_encoded, y_pred):.2f}")
        st.write(f"MAE: {mean_absolute_error(y_test_encoded, y_pred):.2f}")

        # Residual plot
        residuals = y_test_encoded - y_pred
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)

        # Error distribution
        fig, ax = plt.subplots()
        sns.histplot(residuals, bins=30, kde=True, ax=ax)
        ax.set_title("Residual Distribution")
        st.pyplot(fig)

    elif problem_type.lower() == "clustering":
        st.write(f"Inertia: {model.inertia_}")
        if X_test_processed.shape[1] > 2:
            st.write("Note: Visualizing only first 2 features")
        fig, ax = plt.subplots()
        sample_idx = np.random.choice(len(X_test_processed), min(2000, len(X_test_processed)), replace=False)
        scatter = ax.scatter(X_test_processed[sample_idx, 0], 
                             X_test_processed[sample_idx, 1], 
                             c=y_pred[sample_idx], cmap="viridis")
        st.pyplot(fig)
