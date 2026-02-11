from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def build_model(numeric_features, categorical_features=None) -> Pipeline:
    """
    Baseline model: preprocessing + Logistic Regression (class_weight balanced).
    This is a sensible baseline for rare events and interpretable.
    """
    categorical_features = categorical_features or []

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    return Pipeline(steps=[("pre", pre), ("clf", clf)])

