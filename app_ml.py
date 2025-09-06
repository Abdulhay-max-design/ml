# app_ml.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# إعداد الصفحة
# -----------------------------
st.set_page_config(page_title="Iris Flower Classification", layout="wide")
st.title("🌸 Iris Flower Classification App")
st.write("A simple Machine Learning demo: predict iris flower species using Random Forest.")

# -----------------------------
# تحميل البيانات
# -----------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# -----------------------------
# تدريب النموذج
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# مدخلات المستخدم
# -----------------------------
st.sidebar.header("🔧 Input Flower Measurements")
sepal_length = st.sidebar.slider("Sepal length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()), float(X["sepal length (cm)"].mean()))
sepal_width = st.sidebar.slider("Sepal width (cm)", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()), float(X["sepal width (cm)"].mean()))
petal_length = st.sidebar.slider("Petal length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()), float(X["petal length (cm)"].mean()))
petal_width = st.sidebar.slider("Petal width (cm)", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()), float(X["petal width (cm)"].mean()))

# -----------------------------
# التنبؤ
# -----------------------------
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)

# -----------------------------
# عرض النتائج
# -----------------------------
st.subheader("🔮 Prediction Result")
st.write(f"**Predicted Species:** {iris.target_names[prediction]}")

st.subheader("📊 Prediction Probability")
st.bar_chart(pd.DataFrame(prediction_proba, columns=iris.target_names))
