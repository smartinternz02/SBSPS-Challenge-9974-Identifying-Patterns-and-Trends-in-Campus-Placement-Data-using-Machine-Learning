#By Aitham Meghana


#Importing the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the dataset
data = pd.read_csv('Placement_data_full_class.csv')

print(data.head())

data.drop(['sl_no', 'salary'], axis=1, inplace=True)
data['gender'] = data['gender'].map({'M': 0, 'F': 1})

data.dropna(inplace=True)

data = pd.get_dummies(data, drop_first=True)

X = data.drop(['status_Placed'], axis=1)
y = data['status_Placed']

placement_count = data['status_Placed'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(placement_count, labels=placement_count.index, autopct='%1.1f%%', startangle=90)
plt.title('Placement Status Distribution')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
print("Feature Importance:")
print(feature_importance)


plt.figure(figsize=(10, 6))
plt.barh(feature_importance.index, feature_importance.values)
plt.xlabel("Coefficient (Importance)")
plt.ylabel("Features")
plt.title("Feature Importance in Logistic Regression")
plt.show()


# Plotting the placement status distribution
placement_count = data['status_Placed'].value_counts()
plt.figure(figsize=(6, 6))
plt.bar(placement_count.index, placement_count.values)
plt.xticks([0, 1], ['Not Placed', 'Placed'])
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")
plt.title("Placement Status Distribution")
plt.show()

# Grouping data by gender and calculating placement success rates
gender_placement = data.groupby('gender')['status_Placed'].mean()

# Plotting the gender-wise placement success rates
plt.figure(figsize=(6, 6))
plt.bar(gender_placement.index, gender_placement.values)
plt.xticks([0, 1], ['Male', 'Female'])
plt.xlabel("Gender")
plt.ylabel("Placement Success Rate")
plt.title("Gender-wise Placement Success Rate")
plt.ylim(0, 1)
plt.show()

# Create a correlation matrix
correlation_matrix = data.corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# Create a correlation matrix
correlation_matrix = data.corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()