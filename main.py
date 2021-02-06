import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score

# Custom imports
from data_processing import processing as pc
from utils import modelling as md
from utils import model_validation as mv


##################################################
# Get the processed data
df = pc.load_data()
df = pc.build_outcome_label(df)
df = pc.create_time_features(df)

# Generate train and test sets
X_train, X_test, y_train, y_test = md.create_train_test(df)

##################################################
# Create pipeline for feature transformation
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
cols_used = (
    numerical_features + standard_categorical_features + other_categorical_features
)

# Numerical transformer
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Categorical transformers
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

# Put them all together
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("std_cat", standard_categorical_transformer, standard_categorical_features),
        ("other_cat", other_cat_transformer, other_categorical_features),
    ],
    remainder="drop",
)

##############################################
# Classifier
clf = RandomForestClassifier(
    max_depth=3,
    min_samples_leaf=0.03,
    class_weight="balanced",
    random_state=123,
    n_jobs=-1,
    verbose=1,
)

# Create the final pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

# Fit the model
pipeline.fit(X_train, y_train)

y_pred = md.calculate_prediction(pipeline, X_test, decision_value=0.5)

roc_score = roc_auc_score(y_test, y_pred)
print(roc_score)
print(
    classification_report(
        y_test, y_pred, target_names=["Not Successful Search", "Successful Search"]
    )
)

############################################
# Check for discrimination

(
    is_satisfied,
    problematic_stations,
    good_stations,
    global_precisions,
) = mv.verify_no_discrimination(X_test, y_test, y_pred)

precision_per_station = mv.comparison_between_stations(X_test, y_test, y_pred)

if not is_satisfied:
    print("Requirement failed 😢")
    print("Global rates: {}".format(global_precisions))
    print("Num problematic departments: {}".format(len(problematic_stations)))
    print("Num good departments: {}".format(len(good_stations)))

    print("avg diff:", np.mean([p[1] for p in problematic_stations]))

print(precision_per_station)
############################################
# Save the model

# md.feature_importance(cols_used, pipeline)
# md.save_model(pipeline, X_train)
