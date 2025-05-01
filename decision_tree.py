import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
Matplotlib is building the font cache; this may take a moment.
df = pd.read_csv("drug.csv")
display(df.head())
 Age Sex BP Cholesterol Na_to_K Drug
0 23 F HIGH HIGH 25.355 drugY
1 47 M LOW HIGH 13.093 drugC
2 47 M LOW HIGH 10.114 drugC
3 28 F NORMAL HIGH 7.798 drugX
4 61 F LOW HIGH 18.043 drugY
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
 le = LabelEncoder()
 df[column] = le.fit_transform(df[column])
 label_encoders[column] = le
X = df.drop(columns=['Drug'])
y = df['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train_scaled, y_train)
DecisionTreeClassifier(criterion='entropy', random_state=42)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

from sklearn.tree import plot_tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns,
class_names=label_encoders['Drug'].classes_)
plt.show()
