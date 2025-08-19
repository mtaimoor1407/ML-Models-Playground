from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans

CLASSIFICATION_MODELS = {
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "AdaBoost": AdaBoostClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "Voting Classifier": VotingClassifier
}

REGRESSION_MODELS = {
    "Decision Tree": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
}

CLUSTERING_MODELS = {
    "K-means": KMeans
}


def get_models(problem_type):
    """Return a dictionary of available models based on problem type."""
    if problem_type.lower() == "classification":
        return CLASSIFICATION_MODELS
    elif problem_type.lower() == "regression":
        return REGRESSION_MODELS
    elif problem_type.lower() == "clustering":
        return CLUSTERING_MODELS
    else:
        return {}
