import numpy as np
import pandas as pd
from google.colab import drive
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import pickle

drive.mount('/content/drive')

df=pd.read_csv("/content/drive/MyDrive/TTDS-Project/TTDS-Proj/Dataset/fraudTrain.csv")
df.head()

df.shape

df.info()

df.describe()

df.isnull().sum()

df.duplicated().sum()

df.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'trans_num', 'unix_time'],inplace=True)
df.head()

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

df['year'] = df['trans_date_trans_time'].dt.year
df['month'] = df['trans_date_trans_time'].dt.month
df['day'] = df['trans_date_trans_time'].dt.day
df['hour'] = df['trans_date_trans_time'].dt.hour
df['minute'] = df['trans_date_trans_time'].dt.minute
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

df.drop(columns=['trans_date_trans_time'],inplace=True)

df.sample(n=10)

from datetime import datetime

df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
current_year = datetime.now().year
df['age'] = current_year - df['dob'].dt.year

df = df.drop(columns=['dob'])
df[['age']].head()

plt.figure(figsize=(12,6))
merchant_counts = df['merchant'].value_counts().head(15)

sns.barplot(x=merchant_counts.values, y=merchant_counts.index, palette='viridis')
plt.title("Top 15 Merchants by Frequency")
plt.xlabel("Count")
plt.ylabel("Merchant")
plt.show()

plt.figure(figsize=(12,6))
category_counts = df['category'].value_counts()

sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')
plt.title("Category Frequency Distribution")
plt.xlabel("Count")
plt.ylabel("Category")
plt.show()

plt.figure(figsize=(12,6))
job_counts = df['job'].value_counts().head(15)

sns.barplot(x=job_counts.values, y=job_counts.index, palette='viridis')
plt.title("Top 15 Jobs by Frequency")
plt.xlabel("Count")
plt.ylabel("Job")
plt.show()

plt.figure(figsize=(7,7))
gender_counts = df['gender'].value_counts()

plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
plt.title("Gender Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,7))
weekend_counts = df['is_weekend'].value_counts()

labels = ['Weekday' if val == 0 else 'Weekend' for val in weekend_counts.index]

plt.pie(weekend_counts.values, labels=labels, autopct='%1.1f%%')
plt.title("Weekend vs Weekday Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))

day_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
df['day_of_week_label'] = df['day_of_week'].map({i: day_labels[i] for i in range(7)})

sns.countplot(data=df,x='day_of_week_label',hue='is_fraud',palette='viridis')

plt.title("Transaction Count vs Fraud by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Count")
plt.legend(title="Fraud", labels=["Not Fraud (0)", "Fraud (1)"])
plt.show()

fraud_rate = df.groupby('day_of_week')['is_fraud'].mean()

plt.figure(figsize=(12,6))
sns.barplot(x=fraud_rate.index, y=fraud_rate.values, palette='viridis')

plt.xticks(ticks=range(7), labels=day_labels)
plt.title("Fraud Rate (%) by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Fraud Rate")
plt.show()

plt.figure(figsize=(12,6))

age_bins = pd.cut(df['age'], bins=[0,18,30,40,50,60,70,80,100],labels=['0-18','19-30','31-40','41-50','51-60','61-70','71-80','80+'])

sns.countplot(x=age_bins, palette='viridis')
plt.title("Age Distribution (Binned)")
plt.xlabel("Age Range")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12,6))

sns.countplot(x=df['hour'], palette='viridis')
plt.title("Transaction Count by Hour of Day")
plt.xlabel("Hour (0–23)")
plt.ylabel("Count")
plt.show()

fraud_rate_gender = df.groupby('gender')['is_fraud'].mean()

plt.figure(figsize=(8,6))
sns.barplot(x=fraud_rate_gender.index, y=fraud_rate_gender.values, palette='viridis')

plt.title("Fraud Rate (%) by Gender")
plt.xlabel("Gender")
plt.ylabel("Fraud Rate")
plt.show()

age_bins = pd.cut(
    df['age'],
    bins=[0,18,30,40,50,60,70,80,100],
    labels=['0-18','19-30','31-40','41-50','51-60','61-70','71-80','80+']
)

fraud_rate_age = df.groupby(age_bins)['is_fraud'].mean()

plt.figure(figsize=(12,6))
sns.barplot(x=fraud_rate_age.index, y=fraud_rate_age.values, palette='viridis')

plt.title("Fraud Rate (%) by Age Group")
plt.xlabel("Age Range")
plt.ylabel("Fraud Rate")
plt.show()

fraud_rate_category = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False)

plt.figure(figsize=(14,6))
sns.barplot(x=fraud_rate_category.index, y=fraud_rate_category.values, palette='viridis')

plt.title("Fraud Rate (%) by Category")
plt.xlabel("Category")
plt.ylabel("Fraud Rate")
plt.xticks(rotation=90)
plt.show()

df.job.unique()
df.job.value_counts()

cat_cols = ['merchant', 'job', 'category', 'gender']
df[cat_cols].head()

df.is_fraud.value_counts()

print(df['is_fraud'].value_counts())

df_majority = df[df['is_fraud'] == 0]
df_minority = df[df['is_fraud'] == 1]

df_majority_downsampled = df_majority.sample(n=70000, random_state=42)

df = pd.concat([df_majority_downsampled, df_minority], axis=0)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df['is_fraud'].value_counts())

df.shape

label_encoders = {}

for col in ['category', 'gender', 'merchant', 'job']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df.head()

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

df.drop(columns=['day_of_week_label'],inplace=True)

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

df.head()

df.drop(columns=["city_pop", "minute", "is_weekend"], inplace = True)

df.head()

x = df.drop(columns=['is_fraud'])
y = df['is_fraud']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

"""# Decision Tree"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dt = DecisionTreeClassifier(max_depth=6, criterion='gini', random_state=42)

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

print("Decision Tree Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Not Fraud','Fraud']))

"""# Random Forest"""

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42,n_jobs=-1)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

print("Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Not Fraud','Fraud']))

!pip install xgboost lightgbm catboost

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, x_test, y_test, name="Model"):
    print(f"\n================= {name} =================")

    y_pred = model.predict(x_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

bag_model = BaggingClassifier(n_estimators=100,max_samples=0.7,random_state=42,n_jobs=-1)

bag_model.fit(x_train, y_train)

evaluate_model(bag_model, x_test, y_test, "Bagging Classifier")

xgb_model = XGBClassifier(n_estimators=300,max_depth=6,random_state=42,n_jobs=-1)

xgb_model.fit(x_train, y_train)

evaluate_model(xgb_model, x_test, y_test, "XGBoost")

lgbm_model = lgb.LGBMClassifier(n_estimators=300,max_depth=-1,random_state=42)

lgbm_model.fit(x_train, y_train)

evaluate_model(lgbm_model, x_test, y_test, "LightGBM")

results = {"Model": [],"Accuracy": [],"Precision": [],"Recall": [],"F1 Score": []}

def add_results(model, name):
    y_pred = model.predict(x_test)

    results["Model"].append(name)
    results["Accuracy"].append(accuracy_score(y_test, y_pred))
    results["Precision"].append(precision_score(y_test, y_pred))
    results["Recall"].append(recall_score(y_test, y_pred))
    results["F1 Score"].append(f1_score(y_test, y_pred))

add_results(dt, "Decision Tree")
add_results(rf, "Random Forest")
add_results(bag_model, "Bagging")
add_results(xgb_model, "XGBoost")
add_results(lgbm_model, "LightGBM")

results_df = pd.DataFrame(results)
results_df


results_df.plot(kind='bar', x='Model', y=['Accuracy','Precision','Recall','F1 Score'], figsize=(12,6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

!pip install joblib

import joblib

joblib.dump(rf, "random_forest_model.pkl")

joblib.dump(lgbm_model, "lightgbm_model.pkl")

joblib.dump(dt, "decision_tree_model.pkl")

joblib.dump(xgb_model, "xgboost_model.pkl")

joblib.dump(bag_model, "bagging_model.pkl")