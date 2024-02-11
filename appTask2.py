import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load training data


@st.cache_data
def load_data():
    return pd.read_excel('train.xlsx')


df_train = load_data()

label_encoder = LabelEncoder()
df_train['target_encoded'] = label_encoder.fit_transform(df_train['target'])

X_train = df_train.drop(['target', 'target_encoded'], axis=1)
y_train = df_train['target_encoded']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

reverse_mapping = {val: label for val,
                   label in enumerate(label_encoder.classes_)}


def app():
    st.title("Classification App")

    st.sidebar.title("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:

        X_test = pd.read_csv(uploaded_file)
        X_test_scaled = scaler.transform(X_test)

        # Predict
        predictions_encoded = model.predict(X_test_scaled)
        predictions = [reverse_mapping[pred] for pred in predictions_encoded]

        # Display predictions
        st.subheader("Predictions")
        prediction_df = pd.DataFrame({'Prediction': predictions})
        st.dataframe(prediction_df.style.highlight_max(axis=0), width=400)


# Run the app
if __name__ == '__main__':
    app()
