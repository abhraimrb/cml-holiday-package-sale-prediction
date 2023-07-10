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

# Fit a model
depth = 2
clf = RandomForestClassifier(max_depth=depth)
pipe = Pipeline(steps = [
            ('preprocessor', preprocessor), ('classifier', model)
        ])
pipe_rf.fit(X_train, y_train)

acc = pipe_rf.score(X_test, y_test)
print(acc)
with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy: " + str(acc) + "\n")

# Plot it
disp = ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues
)
plt.savefig("plot.png")
