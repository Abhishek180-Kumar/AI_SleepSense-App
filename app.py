import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Data banane wali function (placeholder data)
def load_data():
    data = pd.DataFrame({
        'hours_slept': np.random.normal(6, 1.5, 100),
        'deep_sleep_ratio': np.random.normal(0.3, 0.1, 100),
        'awake_times': np.random.randint(0, 5, 100),
        'snore_intensity': np.random.uniform(0, 1, 100),
        'disorder': np.random.choice(['None', 'Insomnia', 'Sleep Apnea'], 100)
    })
    return data

data = load_data()

# Model training ka function
def train_model(data):
    X = data[['hours_slept', 'deep_sleep_ratio', 'awake_times', 'snore_intensity']]
    y = data['disorder']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    return model, report

model, report = train_model(data)

# Sleep pattern clustering ke liye function
def cluster_sleep_patterns(data):
    X = data[['hours_slept', 'deep_sleep_ratio']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(X)
    return data

data = cluster_sleep_patterns(data)

# Streamlit UI shuru
st.title("AI SleepSense: Smart Sleep Advisor")

st.write("## Upload your sleep data or enter it manually")
hours = st.slider("Hours Slept (Soye kitne ghante)", 0.0, 12.0, 6.0)
deep_sleep = st.slider("Deep Sleep Ratio (Gehri neend ka hissa)", 0.0, 1.0, 0.3)
awake = st.slider("How many times did you wake up at night", 0, 10, 2)
snore = st.slider("Snore Intensity (Khana ki shor ki tezi, 0-1)", 0.0, 1.0, 0.4)

if st.button("Analyze My Sleep"):
    input_data = pd.DataFrame([[hours, deep_sleep, awake, snore]],
                              columns=['hours_slept', 'deep_sleep_ratio', 'awake_times', 'snore_intensity'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Sleep Disorder: {prediction}")

    if prediction == 'Insomnia':
        st.info("Suggestion: Fix your sleeping time and reduce screen time before going to bed. (Apna sone ka time fix karo aur sone se pehle screen time kam karo.)")
    elif prediction == 'Sleep Apnea':
        st.info("Suggestion: Consult a doctor and avoid alcohol before sleeping. (Doctor se salah lo, sone se pehle alcohol se bachao.)")
    else:
        st.info(" Your sleep seems normal. Keep maintaining good habits. (Aapki neend normal lag rahi hai. Acchi aadatein banaye rakho!)")

# Visualization dikhao
st.write("## Sleep Pattern Clusters")
fig, ax = plt.subplots()
sns.scatterplot(data=data, x='hours_slept', y='deep_sleep_ratio', hue='cluster', palette='viridis', ax=ax)
st.pyplot(fig)

st.write("---")
st.write("Made with Terminal Titans using Streamlit and AI")
