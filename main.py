import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Custom imports
from data_processing import processing as pc
from utils import modelling as md
from utils import model_validation as mv

##################################################
# Get the first test set
test1_df = pc.load_test_data()

# Get the processed data
df = pc.load_data()
df = pc.build_outcome_label(df)

# concat the first test set with the train dataframe
df = pd.concat([df, test1_df])
df.drop(columns=["predict"], inplace=True)

df = pc.create_time_features(df)
df = pc.build_features(df)

# Generate train and test sets
X_train, X_test, y_train, y_test = md.create_train_test(df)

##################################################
# Create pipeline for feature transformation
# Features array
numerical_features = ["hour"]
standard_categorical_features = [
    "Type",
    "Age range",
    "Object of search",
    "station",
    "day_of_week",
]
other_categorical_features = [
    "Gender",
    "Officer-defined ethnicity",
    "Legislation",
]
cols_used = (
    numerical_features
    + standard_categorical_features
    + other_categorical_features
    + ["Part of a policing operation", "Latitude", "Longitude"]
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
    min_samples_leaf=100,
    n_estimators=100,
    max_depth=9,
    min_samples_split=2,
    class_weight="balanced",
    random_state=123,
    n_jobs=4,
)

# Create the final pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

# Grid search cv
# param_grid = {
#    "classifier__n_estimators": [100, 250, 450],
#    "classifier__max_depth": np.arange(3, 10, 2),
#    "classifier__min_samples_split": np.arange(2, 53, 10),
# }

grid_search = GridSearchCV(
    pipeline,
    param_grid={},
    scoring="f1_macro",
    verbose=3,
)

# Fit the model
grid_search.fit(X_train, y_train)

y_pred = md.calculate_prediction(grid_search, X_test, decision_value=0.5)

roc_score = roc_auc_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(roc_score)
print("Best parameters {}".format(grid_search.best_params_, grid_search.best_score_))
print(
    "{0}\nTrue Negatives: {1}\nFalse Positives: {2}\nFalse Negatives: {3}\nTrue Positives: {4}".format(
        confusion_matrix(y_test, y_pred), tn, fp, fn, tp
    )
)
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

y_proba = grid_search.predict_proba(X_test)[:, 1]
mv.plot_roc_curve(y_test, y_proba)
############################################
# Save the model

md.feature_importance(cols_used, grid_search.best_estimator_)
md.save_model(grid_search, X_train)

# Gridsearchcv was done, group object of search and legislations with lower than 3000 rows grouped as Other
# latitude and longitude used, tried a geolocator for the city or location but also did not improve
