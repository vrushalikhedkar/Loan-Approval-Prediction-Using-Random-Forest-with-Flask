import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("loan_approval.csv")

print("Dataset Shape:") 
print(dataset.shape)


print("Missing Values:")
print(dataset.isnull().sum())

dataset = dataset.dropna()

print(dataset.shape)

print(dataset.duplicated().sum())

dataset = dataset.drop_duplicates()

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

dataset = pd.get_dummies(
    dataset,
    columns=categorical_columns,
    drop_first=True
)

dataset["Loan_Status"] = dataset["Loan_Status"].map({"Y": 1,"N": 0})

X = dataset.drop("Loan_Status", axis=1)

y = dataset["Loan_Status"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:10])

print("\nActual Values:")
print(y_test.values[:10])

from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.xticks([0, 1],["Rejected", "Approved"])
plt.yticks([0, 1],["Rejected", "Approved"])
plt.show()

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


new_application = X.iloc[[0]]
prediction = model.predict(new_application)

if prediction[0] == 1:
    print("\nNew Loan Application: Approved")
else:
    print("\nNew Loan Application: Rejected")


import pickle
pickle.dump(model, open("loan_approval_random_forest.pkl", "wb"))
print("\nModel saved successfully!")
