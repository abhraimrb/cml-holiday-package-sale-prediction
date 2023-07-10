from sklearn.model_selection import train_test_split
import pandas as pd

# Find the data in the same folder as the current file
package_sale_data = pd.read_csv('data/holiday_package_data.csv')
package_sale_data = package_sale_data.drop(columns = ['CustomerID'])
package_sale_data.replace('Fe Male','Female', inplace = True)


X = package_sale_data.drop(columns = ['ProdTaken'])
y = package_sale_data['ProdTaken']# Make a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)


X_train.to_csv('data/train_features.csv', index=False)
X_test.to_csv('data/test_features.csv', index=False)
y_train.to_csv('data/train_labels.csv', index=False)
y_test.to_csv('data/test_labels.csv', index=False)
