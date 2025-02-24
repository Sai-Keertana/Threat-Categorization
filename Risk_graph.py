import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from MachineLearning_model import y_predictionRF
from Feature_selection import x_test_clean


data = pd.read_csv('Cybersecurity_Dataset.csv')
risk_data = pd.DataFrame({'Risk Level Prediction': x_test_clean['Risk Level Prediction'], 'Threat Category': y_predictionRF})

plt.figure(figsize=(10, 6))
sns.boxplot(x='Threat Category', y='Risk Level Prediction', data=risk_data)
plt.title("Risk Level Distribution by Threat Category")
plt.xlabel("Threat Category")
plt.ylabel("Risk Level Prediction")
plt.show()