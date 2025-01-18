# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

# Title of the app
st.title("Model Random Forest untuk Prediksi Nama Barang")

# File uploader
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("Data Awal:")
    st.write(data.head())

    # Mengecek data yang hilang
    st.write("Jumlah data yang hilang:")
    st.write(data.isnull().sum())

    # Visualisasi distribusi nama barang
    st.write("Distribusi Nama Barang:")
    sns.countplot(data['nama.barang'])
    st.pyplot(plt)

    # Preprocessing
    le = LabelEncoder()
    data['nama.barang'] = le.fit_transform(data['nama.barang'])
    X = data.drop(columns=['Unnamed: 0', 'tanggal', 'kuantum'])
    y = data['nama.barang']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Model Training
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train_balanced, y_train_balanced)

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Akurasi Model: {accuracy:.2f}")

    st.write("Laporan Klasifikasi:")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt)

    # Hyperparameter Tuning
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    model_rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(estimator=model_rf, param_distributions=param_dist, n_iter=100, cv=3, random_state=42, n_jobs=-1)
    random_search.fit(X_train_balanced, y_train_balanced)

    best_params = random_search.best_params_
    st.write("Best Parameters from RandomizedSearchCV:")
    st.write(best_params)

    # Model with best params
    best_model = random_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred_best)
    st.write(f"Akurasi Model dengan Hyperparameter Tuning: {best_accuracy:.2f}")

    st.write("Laporan Klasifikasi Model dengan Hyperparameter Tuning:")
    st.text(classification_report(y_test, y_pred_best))

    # Confusion matrix for best model
    conf_matrix_best = confusion_matrix(y_test, y_pred_best)
    st.write("Confusion Matrix - Best Model:")
    st.write(conf_matrix_best)
    sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt)
