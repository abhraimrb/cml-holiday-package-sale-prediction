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


# Read in data
X_train = pd.read_csv("data/train_features.csv")
y_train = pd.read_csv("data/train_labels.csv")
X_test = pd.read_csv("data/test_features.csv")
y_test = pd.read_csv("data/test_labels.csv")


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
n_estimators =  100

min_samples_split =  2

min_samples_leaf  =  2
rf_model = RandomForestClassifier( n_estimators = n_estimators, 
            min_samples_split = min_samples_split, 
            min_samples_leaf = min_samples_leaf, class_weight = 'balanced')
pipe_rf = Pipeline(steps = [
            ('preprocessor', preprocessor), ('classifier', rf_model)
        ])
pipe_rf.fit(X_train, y_train)

y_pred =  pipe_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)
print('Precision: %.3f' % prec)
print('Recall: %.3f' % rec )
print('F1 Score: %.3f' % f1)

with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy: " + str(acc) + "\n")
    outfile.write("Precision: " + str(prec) + "\n")
    outfile.write("Recall: " + str(rec) + "\n")
    outfile.write("F1 Score: " + str(f1) + "\n")

# Plot it
disp = ConfusionMatrixDisplay.from_estimator(
    pipe_rf, X_test, y_test, normalize="true", cmap=plt.cm.Blues
)
plt.savefig("plot.png")
