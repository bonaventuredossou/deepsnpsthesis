from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve
import numpy as np
import matplotlib.pyplot as plt

training = pd.read_csv('CTCF Binding Sites/training_data_filtered.csv')
validation = pd.read_csv('CTCF Binding Sites/validation_data_filtered.csv')
testing = pd.read_csv('CTCF Binding Sites/test_data_filtered.csv')

init_training_len = len(training)
init_validation_len = len(validation)
init_testing_len = len(testing)

dataset = pd.concat([training, validation, testing])
SEED = 1234
dataset.head()


def mapping_to_lower_interval(a, b, c, d, t):
    # interval [a, b]: a is the lowest initial value, b the largest initial value
    # interval [c, d]: the new minimum in the new interval, d, the largest
    # t the value to be converted
    return c + ((d - c) / (b - a)) * (t - a)


min_dist = min(dataset.min_distance.tolist())
max_dist = max(dataset.min_distance.tolist())
min_chr_pos = min(dataset.CHR_POS.tolist())
max_chr_pos = max(dataset.CHR_POS.tolist())

new_min_dist = 0
new_max_dist = 2000

new_min_pos = 0
new_max_pos = 5000


def transform_min_distance(value):
    return mapping_to_lower_interval(min_dist, max_dist, new_min_dist, new_max_dist, value)


def transform_chr_position(value):
    return mapping_to_lower_interval(min_chr_pos, max_chr_pos, new_min_pos, new_max_pos, value)


dataset['CHR_POS'] = dataset['CHR_POS'].apply(transform_chr_position)
dataset['min_distance'] = dataset['min_distance'].apply(transform_min_distance)

training = dataset[:init_training_len]
training = training.sample(frac=1, random_state=1234)

validation = dataset[init_training_len:init_training_len + init_validation_len]
validation = validation.sample(frac=1, random_state=1234)

testing = dataset[init_validation_len:init_validation_len + init_testing_len]
testing = testing.sample(frac=1, random_state=1234)

y_train = training["is_cancer"].values
training.drop(columns=["is_cancer"], inplace=True)
X_train = training.values

y_test = testing["is_cancer"].values
testing.drop(columns=["is_cancer"], inplace=True)
X_test = testing.values

rf = RandomForestClassifier(n_estimators=2000, random_state=1234)
dt = DecisionTreeClassifier(random_state=1234)

rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

feat_importances = pd.Series(rf.feature_importances_, index=training.columns)
feat_importances.nlargest(5).plot(kind='barh', color=['darkblue', 'orange', 'green', 'red', 'brown'])
plt.ylabel('Features')
plt.xlabel('Random Forest Feature Importance')
plt.savefig('pictures/rf_feature_importance.png')
plt.show()

feat_importances = pd.Series(dt.feature_importances_, index=training.columns)
feat_importances.nlargest(5).plot(kind='barh', color=['darkblue', 'orange', 'green', 'red', 'brown'])
plt.ylabel('Features')
plt.xlabel('Decision Tree Feature Importance')
plt.savefig('pictures/dt_feature_importance.png')
plt.show()
