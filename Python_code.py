# Install required packages (for Colab or fresh environments)
!pip install imbalanced-learn xgboost plotly --quiet

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Load dataset (make sure it's uploaded in Colab environment)
df = pd.read_csv("healthcare_data_transformed.csv")
X = df.drop("disease_risk", axis=1)
y = df["disease_risk"]

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Feature selection using ExtraTreesClassifier
selector = ExtraTreesClassifier(n_estimators=100, random_state=42)
selector.fit(X_resampled, y_resampled)
feat_selector = SelectFromModel(selector, threshold="median", prefit=True)
X_selected = feat_selector.transform(X_resampled)

# Train-test split and scale features
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define machine learning models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=10, random_state=42),
    "SVM": SVC(kernel='rbf', C=2, gamma='scale'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "MLP (Sklearn)": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42)
}

# Train and evaluate ML models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    results[name] = accuracy_score(y_test, preds)

# Voting Classifier as ensemble
voting = VotingClassifier(
    estimators=[
        ("rf", models["Random Forest"]),
        ("gb", models["Gradient Boosting"]),
        ("svm", models["SVM"])
    ],
    voting="hard"
)
voting.fit(X_train_scaled, y_train)
y_pred_vote = voting.predict(X_test_scaled)
results["Voting Classifier"] = accuracy_score(y_test, y_pred_vote)

# Deep Learning model
y_train_dl = to_categorical(y_train)
y_test_dl = to_categorical(y_test)

model = Sequential([
    Dense(256, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train_dl, validation_data=(X_test_scaled, y_test_dl),
                    epochs=50, batch_size=64, verbose=0)
dl_loss, dl_acc = model.evaluate(X_test_scaled, y_test_dl, verbose=0)
results["Deep Learning (Keras)"] = dl_acc

# Print model accuracy summary
print("\n--- Model Accuracies ---")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.4f}")

# Identify best model
best_model_name = max(results, key=results.get)
print(f"\nBest Performing Model: {best_model_name} with Accuracy = {results[best_model_name]:.4f}")

# Confusion matrix for best model
if best_model_name == "Deep Learning (Keras)":
    y_pred_best = model.predict(X_test_scaled).argmax(axis=1)
else:
    best_model = models.get(best_model_name) if best_model_name in models else voting
    y_pred_best = best_model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=["low", "medium", "high"]))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_best, normalize="true")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["low", "medium", "high"],
            yticklabels=["low", "medium", "high"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# --- Visualization 1: Interactive Accuracy Comparison ---
accuracy_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": list(results.values())
}).sort_values("Accuracy", ascending=True)

fig = px.bar(
    accuracy_df,
    x="Accuracy",
    y="Model",
    orientation='h',
    color="Accuracy",
    color_continuous_scale="Agsunset",
    title="Model Accuracy Comparison"
)
fig.update_layout(
    xaxis_title="Accuracy Score",
    yaxis_title="ML/DL Model",
    template="plotly_white",
    title_font_size=20,
    margin=dict(l=40, r=40, t=60, b=40)
)
fig.show()

# --- Visualization 2: Prediction Correlation Heatmap ---
model_preds_df = pd.DataFrame()
for name, model in models.items():
    model_preds_df[name] = model.predict(X_test_scaled)
model_preds_df["Voting"] = voting.predict(X_test_scaled)
model_preds_df["Deep Learning"] = y_pred_best

corr = model_preds_df.corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", square=True)
plt.title("Prediction Correlation Between Models")
plt.tight_layout()
plt.show()

# --- Visualization 3: Deep Learning Training Curve ---
fig = go.Figure()
fig.add_trace(go.Scatter(y=history.history['accuracy'], mode='lines', name='Train Accuracy'))
fig.add_trace(go.Scatter(y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy'))
fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss', line=dict(dash='dash')))
fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss', line=dict(dash='dash')))

fig.update_layout(
    title="Deep Learning Training Curve",
    xaxis_title="Epoch",
    yaxis_title="Metric Value",
    legend_title="Metric",
    template="plotly_white",
    title_font_size=20
)
fig.show()

# --- Feature Importance (Top 10 Features) ---
feat_names = df.drop("disease_risk", axis=1).columns
feat_mask = feat_selector.get_support()
selected_names = feat_names[feat_mask]
importances = models["Random Forest"].feature_importances_
top_feats = pd.DataFrame({"Feature": selected_names, "Importance": importances})
top_feats = top_feats.sort_values("Importance", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_feats, x="Importance", y="Feature", palette="crest")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()
