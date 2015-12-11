import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skflow

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

def get_age_group(age):
    if age < 30:
        return "18-29"
    elif age < 50:
        return "30-49"
    elif age < 65:
        return "50-64"
    return "65+"

AGES = ['18-29', '30-49', '50-64', '65+']
COLUMNS = ['age', 'pial2a', 'pial2b', 'pial2c', 'pial2d', 'pial2e', 'pial3a', 'pial3b', 'pial3c', 'pial3d', 'pial4a', 'pial4b', 'pial4c', 'pial5']

data = pd.read_csv("Feb_2014_Views_Future_CSV.csv", usecols=COLUMNS)

data['age_group'] = np.array([AGES.index(get_age_group(x)) for x in data.age.values])

# Split data into training and testing groups. Functionally equivalent to scikit's train_test_split
train_data, test_data, train_target, test_target = train_test_split(data.iloc[:, 0:12].values, data.age_group.values)

# Tensor Flow Linear Classifier
clf = skflow.TensorFlowLinearClassifier(n_classes=4)
clf.fit(train_data.astype(float), train_target)

score = accuracy_score(clf.predict(test_data), test_target)
print("Score: ", score)
print()

if score < 0.33:
    print ("This classifier is about as good or worse than guessing.")
elif score < 0.50:
    print ("This classifier is better than guessing, but the difference is negligible.")
elif score < 0.75:
    print ("This classifier is significantly better than guessing, but it still isn't very accurate.")
else:
    print ("This classifier is accurate in determining an age group based on sentiment about future technology.")