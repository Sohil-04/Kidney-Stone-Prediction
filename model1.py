
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("S:\ml models\kidney store prediction\kindey stone urine analysis.csv")

print(df.head())

"""# understand the dataset"""

print("Dataset Shape:", df.shape)
df.info()
print(df.describe())

"""# data quality analysis"""

# missing values
print(df.isnull().sum())
# duplicated values
print(df.duplicated().sum())

# unique values
for col in df.columns:
    print(col, df[col].nunique())

"""# univariate analysis"""

# histograms
df.hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()

# boxplots
plt.figure(figsize=(12, 6))

sns.boxplot(data=df.select_dtypes(include=np.number))

plt.xticks(rotation=45)
plt.title("Feature Distribution and Outliers")
plt.tight_layout()
plt.show()

"""# target variable analysis"""

print(df["target"].value_counts())

print(df["target"].value_counts(normalize=True) * 100)

sns.countplot(data=df, x="target")

plt.title("Kidney Stone Class Distribution")
plt.show()

"""# Bivariate analysis"""

sns.boxplot(
    data=df,
    x="target",
    y="calc"
)

plt.title("Calcium vs Kidney Stone")
plt.show()

plt.figure(figsize=(10, 8))

corr = df.select_dtypes(include=np.number).corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.show()

"""#Separate features and target"""

x = df.drop("target", axis=1)
y = df["target"]

print("features:",x.shape)
print("target:",y.shape)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("train:",x_train.shape,y_train.shape)
print("test:",x_test.shape,y_test.shape)

"""# preprocessing pipeline


"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numerical_features = x_train.select_dtypes(
    include=np.number
).columns

categorical_features = x_train.select_dtypes(
    include="object"
).columns

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)

print("Training shape:", x_train_processed.shape)
print("Testing shape:", x_test_processed.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(accuracy_score,
                            confusion_matrix,
                            classification_report,
                            roc_auc_score,
                            RocCurveDisplay
                            )

model =Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs',
        C=1.0
    ))
])

model.fit(x_train, y_train)

print("Model training completed!")

y_pred = model.predict(x_test)

y_prob = model.predict_proba(x_test)[:, 1]

accuracy_score=accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy_score)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Stone", "Stone"],
    yticklabels=["No Stone", "Stone"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", auc)

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("Logistic Regression - ROC Curve")
plt.show()

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = model.named_steps[
    "classifier"
].coef_[0]

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

coef_df["Abs_Coefficient"] = abs(
    coef_df["Coefficient"]
)

coef_df = coef_df.sort_values(
    "Abs_Coefficient",
    ascending=False
)

print(coef_df)

new_patient = pd.DataFrame([{
    "gravity": 1.020,
    "ph": 6.0,
    "osmo": 500,
    "cond": 15,
    "urea": 300,
    "calc": 5
}])

prediction = model.predict(new_patient)

probability = model.predict_proba(new_patient)[0][1]

if prediction[0] == 1:
    print("Predicted Class: Stone")
else:
    print("Predicted Class: No Stone")

print("Predicted Probability:", probability)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Probability": y_prob
})

results.to_csv(
    "kidney_stone_predictions.csv",
    index=False
)

print("Predictions saved successfully!")

