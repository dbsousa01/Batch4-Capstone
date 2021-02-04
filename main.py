from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Custom imports
from data_processing import processing as pc
from utils import modelling as md

df = pc.load_data()
df = pc.build_outcome_label(df)
df = pc.create_time_features(df)

# Generate train and test sets
X_train, X_test, y_train, y_test = md.create_train_test(df)

# Create pipeline for feature transformation
all_features = [
    "Type",
    "Part of a policing operation",
    "Gender",
    "Age range",
    "Officer-defined ethnicity",
    "Legislation",
    "Object of search",
    "station",
    "hour",
    "day_of_week",
]

# Features array
numerical_features = ["hour"]
standard_categorical_features = [
    "Type",
    "Part of a policing operation",
    "Age range",
    "Legislation",
    "Object of search",
    "station",
    "day_of_week",
]
other_categorical_features = [
    "Gender",
    "Officer-defined ethnicity",
]

# Categorical transformers
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

standard_categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

other_cat_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Other")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("std_cat", standard_categorical_transformer, standard_categorical_features),
        ("other_cat", other_cat_transformer, other_categorical_features),
    ]
)
