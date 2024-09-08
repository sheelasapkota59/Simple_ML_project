import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Create a dictionary with data
data = {
    'Hours Studied': [1, 2, 3, 4, 5, 6],
    'Result': ['Fail', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass']
}
#To use this data with scikit-learn, you'll need to convert it into a DataFrame.
df = pd.DataFrame(data)
print(df)

#Machine learning models in scikit-learn generally require numerical input, so you need to encode the categorical 
#target variable.
#The map function in pandas allows you to replace values in a DataFrame or Series according to a mapping dictionary.
#Here’s how it’s used:

df['Result'] = df['Result'].map({'Fail' : 0 , 'Pass' : 1})
df

#split the data into training and test sets.
# Many machine learning libraries, including scikit-learn, expect features (X) to be in a 2D array-like structure 
# (i.e., a DataFrame or a 2D NumPy array), even if there is only one feature. Using double brackets ensures that 
# X is always a DataFrame, which satisfies these requirements.

X = df[['Hours Studied']]  #feature(for training)

Y = df['Result']     #Result(testing)


# X: This variable holds the input feature (hours studied).
# y: This variable holds the target variable (pass/fail).
# test_size=0.3: 30% of the data is reserved for testing, and 70% is used for training.
# random_state=42: This ensures that the split is the same every time you run the code.

x_train , x_test, y_train , y_test =  train_test_split( X, Y , test_size = 0.3 , random_state = 42)

# Initialize the model
log_reg = LogisticRegression()

# Initialize the Decision Tree Classifier
log_def = DecisionTreeClassifier()

# Fit the  Decision Tree Classifier model on training data
log_def.fit(x_train , y_train)

# Fit the model on training data
log_reg.fit(x_train , y_train)

#predict result
y_pred_log_Reg = log_reg.predict(x_test)
y_pred_log_Def = log_def.predict(x_test)

accuracy = accuracy_score(y_test , y_pred_log_Reg)
accuracy_def = accuracy_score(y_test , y_pred_log_Def)

print(f"Logistic Regression Accuracy: {accuracy:.2f}")
print(f"Decision Tree Accuracy: {accuracy:.2f}")

# Plot the Decision Tree
plt.figure(figsize= ( 10 , 8))
tree.plot_tree(log_def , feature_names = ['Hours studied'] , filled = True)
plt.show()
