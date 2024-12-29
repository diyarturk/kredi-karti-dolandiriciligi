import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Başlık
st.title("Kredi Kartı Dolandırıcılık Tespiti")

# Veri seti yükleme ve açıklama
@st.cache
def load_data():
    data = pd.read_csv("creditcard.csv")
    # Sütun adlarını temizle
    data.columns = data.columns.str.strip()
    return data

data = load_data()

# Veri setini açıklama
st.sidebar.subheader("Veri Seti Açıklaması")
st.sidebar.write("Bu veri seti, kredi kartı işlemlerinin dolandırıcılık olup olmadığını tespit etmek için kullanılır.")
st.sidebar.write("Özellikler:")
st.sidebar.write("- **Time**: İşlemin ilk işlemden itibaren geçen süre (saniye cinsinden).")
st.sidebar.write("- **V1-V28**: PCA ile anonimleştirilmiş özellikler.")
st.sidebar.write("- **Amount**: İşlem tutarı.")
st.sidebar.write("- **Class**: Hedef değişken (0: Normal, 1: Dolandırıcılık).")

# Veri görselleştirme
st.subheader("Veri Seti Genel Görünümü")
st.write(data.head())

# Eksik değer kontrolü ve temizleme
st.write("Eksik değerlerin kontrolü")
st.write(data.isnull().sum())
if 'Class' in data.columns and 'Amount' in data.columns:
    data = data.dropna(subset=['Class', 'Amount'])
else:
    st.error("Veri setinde 'Class' veya 'Amount' sütunu bulunamadı. Lütfen veri setini kontrol edin.")

# Görselleştirme: Dolandırıcılığın Zamanla Dağılımı
st.subheader("Dolandırıcılık İşlemlerinin Zamanla Dağılımı")
if 'Class' in data.columns and 'Time' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[data['Class'] == 1]['Time'], bins=50, kde=True, color='red', label='Dolandırıcılık')
    sns.histplot(data[data['Class'] == 0]['Time'], bins=50, kde=True, color='blue', label='Normal')
    plt.title("Zaman Bazında İşlemler")
    plt.xlabel("Zaman (saniye)")
    plt.ylabel("İşlem Sayısı")
    plt.legend()
    st.pyplot(plt)
    st.write("**Açıklama**: Bu grafik, dolandırıcılık ve normal işlemlerin zamanla nasıl dağıldığını gösterir. Dolandırıcılık işlemleri genellikle belirli zaman dilimlerinde yoğunlaşabilir.")
else:
    st.error("'Class' veya 'Time' sütunu bulunamadı. Görselleştirme oluşturulamadı.")

# Görselleştirme: İşlem Tutarlarının Dağılımı
st.subheader("İşlem Tutarlarının Dağılımı")
if 'Class' in data.columns and 'Amount' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Amount', data=data, palette='coolwarm')
    plt.title("İşlem Tutarı ve Dolandırıcılık")
    plt.xlabel("Sınıf (Class)")
    plt.ylabel("Tutar (Amount)")
    st.pyplot(plt)
    st.write("**Açıklama**: Bu kutu grafiği, dolandırıcılık ve normal işlemlerin işlem tutarlarına göre dağılımını gösterir. Dolandırıcılık işlemleri genellikle daha düşük işlem tutarlarında yoğunlaşır.")
else:
    st.error("'Class' veya 'Amount' sütunu bulunamadı. Görselleştirme oluşturulamadı.")

# Görselleştirme: Dolandırıcılıkla İlişkili Özellikler
st.subheader("Dolandırıcılık ile En İlişkili Özellikler")
if 'Class' in data.columns:
    correlation_matrix = data.corr()
    dolandırıcılık_corr = correlation_matrix['Class'].drop('Class').sort_values(key=abs, ascending=False).head(5)
    st.write(dolandırıcılık_corr)
    st.write("**Açıklama**: Bu tablo, dolandırıcılık ile en çok ilişkili olan ilk 5 özelliği göstermektedir. Bu özellikler modelin dolandırıcılığı tespit etmesinde kritik öneme sahiptir.")

    # Görselleştirme: Korelasyon Matrisi (En Önemli Özellikler)
    plt.figure(figsize=(10, 6))
    sns.heatmap(data[dolandırıcılık_corr.index].corr(), annot=True, cmap='coolwarm')
    plt.title("Dolandırıcılık ile İlgili Özelliklerin Korelasyonu")
    st.pyplot(plt)
    st.write("**Açıklama**: Bu korelasyon matrisi, dolandırıcılıkla en çok ilişkili olan özelliklerin birbirleriyle nasıl bir ilişkiye sahip olduğunu gösterir. Yüksek korelasyon, bu özelliklerin birbiriyle ilişkili olduğunu ifade eder.")
else:
    st.error("'Class' sütunu bulunamadı. Dolandırıcılıkla ilişkili özellikler gösterilemiyor.")

# Kullanıcıdan giriş alın
st.sidebar.header("İşlem Özelliklerini Girin")
def user_input_features():
    Time = st.sidebar.number_input("Zaman (Time)", min_value=0.0, value=0.0)
    Amount = st.sidebar.number_input("Tutar (Amount)", min_value=0.0, value=1.0)
    features = {f"V{i}": st.sidebar.number_input(f"V{i}", value=0.0) for i in range(1, 29)}
    features['Time'] = Time
    features['Amount'] = Amount
    return np.array([features[col] for col in ['Time', *[f"V{i}" for i in range(1, 29)], 'Amount']])

# Kullanıcı girdisini al
input_data = user_input_features()

# Veri Hazırlık (Model eğitim ve değerlendirme)
if 'Class' in data.columns:
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Eğitim ve test verilerini ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Algoritmaların Eğitim ve Değerlendirmesi
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        results[name] = {
            "Accuracy": accuracy,
            "ROC AUC": roc_auc,
            "Classification Report": classification_report(y_test, y_pred)
        }

    # Sonuçları Streamlit ile göster
    st.subheader("Model Karşılaştırmaları")
    for name, metrics in results.items():
        st.write(f"### {name}")
        st.write(f"- Doğruluk (Accuracy): {metrics['Accuracy']:.2f}")
        st.write(f"- ROC AUC: {metrics['ROC AUC']:.2f}")
        st.text(metrics['Classification Report'])
        st.write("**Açıklama**: Bu modelin doğruluk ve ROC AUC skorları, modelin dolandırıcılık tahmin etme başarısını ifade eder. Yüksek ROC AUC skoru, modelin dolandırıcılık ve normal işlemleri ayırt etme yeteneğinin iyi olduğunu gösterir.")

    # Tahmin (Önceden eğitilmiş modeli seçerek)
    selected_model = st.sidebar.selectbox("Model Seçimi", list(models.keys()))
    model = models[selected_model]
    scaled_input = scaler.transform([input_data])
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # Tahmin Sonucu Göster
    st.subheader("Tahmin Sonucu")
    result = "Dolandırıcılık" if prediction[0] == 1 else "Normal"
    st.write(f"Bu işlem {result} olarak tahmin edilmiştir.")
    st.write("**Açıklama**: Tahmin sonucu, kullanıcının girdiği özelliklere göre bu işlemin dolandırıcılık olup olmadığını belirler.")

    st.subheader("Tahmin Olasılığı")
    st.write(f"Normal: {prediction_proba[0][0]:.2f}, Dolandırıcılık: {prediction_proba[0][1]:.2f}")
    st.write("**Açıklama**: Bu olasılıklar, modelin işlemin dolandırıcılık olma ihtimaline dair güvenini ifade eder. Daha yüksek bir dolandırıcılık olasılığı, işlemin şüpheli olduğunu gösterir.")
else:
    st.error("'Class' sütunu bulunamadı. Model eğitimi gerçekleştirilemiyor.")
