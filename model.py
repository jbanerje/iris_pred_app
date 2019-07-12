# Import dependencies
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble  import RandomForestClassifier

# Load the dataset in a dataframe object and include only four features as mentioned
iris_df = pd.read_csv('iris.csv')


# Data Preprocessing

#Label Encoding (Gender)
iris_df['species'] = iris_df.species.map( {'setosa': 0, 'virginica': 1, 'versicolor': 2} ).astype(int)

# Select Feature and Target
Feature = iris_df.drop(columns=['species'], axis=1)
Target  = iris_df[['species']]

# Random Forest classifier
classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(Feature,Target)

# Save your model
joblib.dump(classifier, 'model.pkl') # # To load this model, use lr = joblib.load('model.pkl')
print("Model dumped!")
