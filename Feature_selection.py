import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Prepare dataset
data = pd.read_csv('Cybersecurity_Dataset.csv') # Import dataset
data = data.dropna() # Remove all null values

# Encode features into numerical values using Label Encoder
data['Threat Category'] = LabelEncoder().fit_transform(data['Threat Category'])
data['Threat Actor'] = LabelEncoder().fit_transform(data['Threat Actor'])
data['Attack Vector'] = LabelEncoder().fit_transform(data['Attack Vector'])
data['Geographical Location'] = LabelEncoder().fit_transform(data['Geographical Location'])
data['Suggested Defense Mechanism'] = LabelEncoder().fit_transform(data['Suggested Defense Mechanism'])
data['Cleaned Threat Description'] = LabelEncoder().fit_transform(data['Cleaned Threat Description'])
data['Keyword Extraction'] = LabelEncoder().fit_transform(data['Keyword Extraction'])
data['Named Entities (NER)'] = LabelEncoder().fit_transform(data['Named Entities (NER)'])
data['Topic Modeling Labels'] = LabelEncoder().fit_transform(data['Topic Modeling Labels'])

#print(data.head())


# Split data into test and train
features = ['Threat Actor','Attack Vector','Geographical Location','Named Entities (NER)','Keyword Extraction','Cleaned Threat Description','Suggested Defense Mechanism','Topic Modeling Labels','Severity Score','Sentiment in Forums','Risk Level Prediction'] 
goal = 'Threat Category' 
data[features] =  StandardScaler().fit_transform(data[features])
x = data[features] 
y = data[goal]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# Feature analysis
correlation = x.corrwith(data[goal]) # Pandas correalation
feature_correlation = pd.DataFrame(correlation, columns=["Correlation"])
#print("Correlation of features:\n",correlation)
rf = RandomForestClassifier(n_estimators=100, random_state=42) # Train the Random Forest model
rf.fit(x_train, y_train)
accuracy_before = rf.score(x_test, y_test) # Accuracy of the Random Forest model before feature selection
feature_importance = pd.DataFrame(rf.feature_importances_, index=x_train.columns, columns=['importance']).sort_values('importance', ascending=False) # Feature importance
#print(feature_importances) # Feature importance sorted in descending order
print(f'\nAccuracy using Random Forest model before feature selection: {accuracy_before:.2f}')

# Remove and/or retain features
remove_features = ['Threat Actor','Attack vector', 'Topic Modeling Labels','Suggested Defense Mechanism']
retain_features = ['Sentiment in Forums','Risk Level Prediction','Keyword Extraction','Geographical Location','Named Entities (NER)','Cleaned Threat Description','Severity Score']
x_train_clean = x_train[retain_features].copy()
x_test_clean = x_test[retain_features].copy()
x_train_clean.loc[:, retain_features] = x_train_clean[retain_features].astype(float)
x_test_clean.loc[:, retain_features] = x_test_clean[retain_features].astype(float)


# Calculate final weights
feature_correlation["Correlation"] = np.abs(feature_correlation["Correlation"])
feature_correlation["Correlation"] /= feature_correlation["Correlation"].sum()
feature_importance["importance"] /= feature_importance["importance"].sum() # Normalize feature importance (ensure it sums to 1)
feature_weights = feature_importance.merge(feature_correlation, left_index=True, right_index=True) # Merge Importance and Correlation into a single DataFrame
alpha = 0.8  # Adjustable weight factor
feature_weights["FinalWeight"] = (alpha * feature_weights["importance"]) + ((1 - alpha) * feature_weights["Correlation"])
feature_weights["FinalWeight"] /= feature_weights["FinalWeight"].sum() # Normalize Final Weights
print("\nFeature Weights Based on Importance and Correlation:")
print(feature_weights.sort_values("FinalWeight", ascending=False))

# Apply weights to features
for feature, weight in feature_weights["FinalWeight"].items(): 
    if feature in retain_features:  # Ensure we only apply to the features to retain
        if feature in x_train_clean.columns:  # Check if the feature exists in x_train_clean
            x_train_clean[feature] = x_train_clean[feature] * weight
        if feature in x_test_clean.columns:  # Check if the feature exists in x_test_clean
            x_test_clean[feature] = x_test_clean[feature] * weight

# SMOTE 
smote = SMOTE(sampling_strategy='auto', random_state=32)
x_train_res, y_train_res = smote.fit_resample(x_train_clean, y_train)

# Clean data feature analysis
rf.fit(x_train_clean, y_train)
accuracy_after = rf.score(x_test_clean, y_test) # Accuracy of the Random Forest model after feature selection
print(f'\nAccuracy using Random Forest model after feature selection 2: {accuracy_after:.2f}\n')
y_pred = rf.predict(x_test_clean)
f1 = f1_score(y_test, y_pred, average='macro')# Calculate F1 score
print(f"F1-Score: {f1:.2f}")
print("\nClassification Report:") # Full classification report
print(classification_report(y_test, y_pred))

'''
# Hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(x_train_clean, y_train)
accuracy_after_grid_search = grid_search.best_estimator_.score(x_test_clean, y_test)
#print(f'Accuracy after Grid Search: {accuracy_after_grid_search:.2f}')

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_clean, y_train)
y_pred = knn.predict(x_test_clean)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

'''

