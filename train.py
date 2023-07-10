import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")


numeric_features = [
        'Age', 
        'DurationOfPitch', 
        'MonthlyIncome'
    ]
numeric_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'median')), 
            ("scaler", StandardScaler())
        ]
    )
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']

categorical_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
        ]
    )

preprocessor = ColumnTransformer(transformers = [
            ('num_tr', numeric_transformer, numeric_features),
            ('cat_tr', categorical_transformer, categorical_features)
        ], remainder = SimpleImputer(strategy = 'most_frequent')
    )
# Fit a model
depth = 2
rf_model = RandomForestClassifier(max_depth=depth)
pipe_rf = Pipeline(steps = [
            ('preprocessor', preprocessor), ('classifier', rf_model)
        ])
pipe_rf.fit(X_train, y_train)

acc = pipe_rf.score(X_test, y_test)
print(acc)
with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy: " + str(acc) + "\n")

# Plot it
disp = ConfusionMatrixDisplay.from_estimator(
    rf_model, X_test, y_test, normalize="true", cmap=plt.cm.Blues
)
plt.savefig("plot.png")
