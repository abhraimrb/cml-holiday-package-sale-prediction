from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import os
# Find the data in the same folder as the current file
package_sale_data = pd.read_csv('data/holiday_package_data.csv')
package_sale_data = package_sale_data.drop(columns = ['CustomerID'])
package_sale_data.replace('Fe Male','Female', inplace = True)

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

    categorical_features = [
        'TypeofContact', 
        'Occupation', 
        'Gender', 
        'ProductPitched',
        'MaritalStatus',
        'Designation'
    ]

    categorical_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers = [
            ('num_tr', numeric_transformer, numeric_features),
            ('cat_tr', categorical_transformer, categorical_features)
        ], remainder = SimpleImputer(strategy = 'most_frequent')
    )
X = package_sale_data.drop(columns = ['ProdTaken'])
y = package_sale_data['ProdTaken']# Make a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)


np.savetxt('data/train_features.csv', X_train)
np.savetxt('data/test_features.csv', X_test)
np.savetxt('data/train_labels.csv', y_train)
np.savetxt('data/test_labels.csv', y_test)
