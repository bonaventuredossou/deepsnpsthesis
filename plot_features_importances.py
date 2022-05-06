import json
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from numpy import mean
from numpy import std
import pandas as pd

f = open('feature_importances.json')
data = json.load(f)
keys = list(data.keys())

correct_predictions = {"SNPS": [], "CONTEXT": [], "CHR_ID": [], "CHR_POS": [], "min_distance": []}
wrong_predictions = {"SNPS": [], "CONTEXT": [], "CHR_ID": [], "CHR_POS": [], "min_distance": []}
features = ["SNPS", "CONTEXT", "CHR_ID", "CHR_POS", "min_distance"]

for key in keys:
    lists = data[key]
    first, second, third, fourth, fifth = [], [], [], [], []
    for list_ in lists:
        first.append(list_[0])
        second.append(list_[1])
        third.append(list_[2])
        fourth.append(list_[3])
        fifth.append(list_[4])

    for feature in features:
        if key == 'correct_predictions':
            correct_predictions[feature].append(first.count(feature))
            correct_predictions[feature].append(second.count(feature))
            correct_predictions[feature].append(third.count(feature))
            correct_predictions[feature].append(fourth.count(feature))
            correct_predictions[feature].append(fifth.count(feature))
        else:
            if key == 'wrong_predictions':
                wrong_predictions[feature].append(first.count(feature))
                wrong_predictions[feature].append(second.count(feature))
                wrong_predictions[feature].append(third.count(feature))
                wrong_predictions[feature].append(fourth.count(feature))
                wrong_predictions[feature].append(fifth.count(feature))

correct_predictions_dataframe = pd.DataFrame.from_dict(correct_predictions)
wrong_predictions_dataframe = pd.DataFrame.from_dict(wrong_predictions)

correct_predictions_dataframe.plot.bar(rot=0)
plt.xticks(np.arange(len(features)), labels=["$1^{st}$", "$2^{nd}$", "$3^{rd}$", "$4^{th}$", "$5^{th}$"])
plt.title('Feature Importance for Correctly Predicted Samples')
plt.savefig('pictures/features_importance_correct_predictions.png')
plt.show()

wrong_predictions_dataframe.plot.bar(rot=0)
plt.xticks(np.arange(len(features)), labels=["$1^{st}$", "$2^{nd}$", "$3^{rd}$", "$4^{th}$", "$5^{th}$"])
plt.title('Feature Importance for Wrongly Predicted Samples')
plt.savefig('pictures/features_importance_wrong_predictions.png')
plt.show()
