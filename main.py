# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import jax.numpy as jnp
from flax import linen as nn
import tensorflow as tf
import torch
from sklearn.pipeline import Pipeline

# Configuración inicial
st.set_page_config(page_title="Framework ML Comparison", layout="wide")

@st.cache_resource
def load_artifacts():
    """Cargar todos los modelos y datos necesarios"""
    artifacts = {
        'preprocessor': joblib.load("preprocessor.joblib"),
        'times': {
            'JAX': joblib.load("jax_time.joblib"),
            'TensorFlow': joblib.load("tf_time.joblib"),
            'PyTorch': joblib.load("torch_time.joblib")
        },
        'jax_params': joblib.load("jax_params.joblib"),
        'tf_model': tf.keras.models.load_model("tf_model.keras"),
        'torch_model': torch.load("torch_model.pth"),
        'bank_data': fetch_bank_data()
    }
    return artifacts

def fetch_bank_data():
    """Obtener metadata del dataset"""
    bank = fetch_openml(name="bank-marketing", version=1, as_frame=True)
    return {
        'features': bank.data,
        'target_names': bank.target.cat.categories.tolist()
    }

# Modelo JAX usando Flax
class JAXModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# Modelo PyTorch
class TorchModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def main():
    artifacts = load_artifacts()
    preprocessor = artifacts['preprocessor']
    
    # Sidebar para inputs
    st.sidebar.title("Parámetros del Cliente")
    input_data = {}
    
    for feature in artifacts['bank_data']['features'].columns:
        col_data = artifacts['bank_data']['features'][feature]
        if np.issubdtype(col_data.dtype, np.number):
            input_data[feature] = st.sidebar.number_input(
                f"{feature} ({col_data.min()}-{col_data.max()})",
                min_value=float(col_data.min()),
                max_value=float(col_data.max()),
                value=float(col_data.median())
            )
        else:
            unique_vals = col_data.unique().tolist()
            input_data[feature] = st.sidebar.selectbox(
                feature,
                unique_vals,
                index=len(unique_vals)//2
            )

    # Procesar entrada
    input_df = pd.DataFrame([input_data])
    processed_input = preprocessor.transform(input_df).astype(np.float32)
    
    # Sección principal
    st.title("🦾 Comparación de Frameworks de ML")
    st.subheader("Predicción de Marketing Bancario", divider="rainbow")
    
    if st.button("Ejecutar Predicción", use_container_width=True):
        with st.spinner("Calculando predicciones..."):
            # Predicción JAX
            jax_model = JAXModel()
            jax_input = jnp.array(processed_input)
            jax_pred = jax_model.apply({'params': artifacts['jax_params']}, jax_input)
            prob_jax = 1 / (1 + np.exp(-jax_pred[0][0]))
            
            # Predicción TensorFlow
            tf_pred = artifacts['tf_model'].predict(processed_input, verbose=0)[0][0]
            
            # Predicción PyTorch
            input_size = processed_input.shape[1]
            torch_model = TorchModel(input_size)
            torch_model.load_state_dict(artifacts['torch_model'])
            torch_model.eval()
            with torch.no_grad():
                torch_input = torch.tensor(processed_input)
                torch_pred = torch_model(torch_input).numpy()[0][0]
            
            # Mostrar resultados
            cols = st.columns(3)
            frameworks = {
                "JAX": prob_jax,
                "TensorFlow": tf_pred,
                "PyTorch": torch_pred
            }
            
            for (name, prob), col in zip(frameworks.items(), cols):
                col.metric(
                    label=name,
                    value=f"{prob:.2%}",
                    help=f"Tiempo de entrenamiento: {artifacts['times'][name]:.2f} segundos"
                )
                col.progress(float(prob), text=f"Probabilidad de suscripción ({name})")

            # Gráfico comparativo
            st.bar_chart(pd.DataFrame({
                "Framework": frameworks.keys(),
                "Probabilidad": [float(p) for p in frameworks.values()]
            }).set_index("Framework"))

if __name__ == "__main__":
    main()
