import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline

# Configuración básica
st.set_page_config(page_title="Comparación ML", layout="centered")
st.title("🚀 Comparación JAX vs TF vs PyTorch")

# --------------------------
# Carga de Modelos y Datos
# --------------------------
@st.cache_resource
def cargar_modelos():
    """Cargar todos los recursos necesarios"""
    return {
        'prepro': joblib.load("preprocessor.joblib"),
        'tiempos': joblib.load("tiempos_entrenamiento.joblib"),
        'modelo_jax': joblib.load("jax_params.joblib"),
        'modelo_tf': tf.keras.models.load_model("tf_model.keras"),
        'modelo_torch': torch.load("torch_model.pth"),
        'dataset_info': obtener_info_dataset()
    }

def obtener_info_dataset():
    """Obtener nombres de características y tipos"""
    datos = fetch_openml(name="bank-marketing", version=1, as_frame=True)
    return {
        'nombres_features': datos.data.columns.tolist(),
        'tipos_features': datos.data.dtypes.astype(str).to_dict()
    }

# --------------------------
# Modelos
# --------------------------
# Modelo JAX (Flax)
class ModeloJAX:
    def predecir(self, params, X):
        @jax.jit
        def predict(params, x):
            return 1 / (1 + jnp.exp(-self.apply(params, x)))
        return predict(params, X)
    
    @staticmethod
    @nn.compact
    def apply(params, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x)

# Modelo PyTorch
class ModeloTorch(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# --------------------------
# Interfaz
# --------------------------
def main():
    recursos = cargar_modelos()
    
    # Formulario de entrada
    with st.form("input_form"):
        st.subheader("📋 Datos del Cliente")
        inputs = {}
        for feature in recursos['dataset_info']['nombres_features']:
            tipo = recursos['dataset_info']['tipos_features'][feature]
            
            if np.issubdtype(tipo, np.number):
                inputs[feature] = st.number_input(feature)
            else:
                inputs[feature] = st.selectbox(feature, ["Opcion1", "Opcion2"])  # Simplificado
        
        if st.form_submit_button("🔮 Predecir"):
            # Preprocesar
            X = recursos['prepro'].transform(pd.DataFrame([inputs]))
            
            # Predicciones
            resultados = {
                'JAX': ModeloJAX().predecir(recursos['modelo_jax'], X)[0][0],
                'TensorFlow': recursos['modelo_tf'].predict(X, verbose=0)[0][0],
                'PyTorch': ModeloTorch(X.shape[1]).eval()(torch.tensor(X)).detach().numpy()[0][0]
            }
            
            # Mostrar resultados
            st.subheader("📊 Resultados")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mejor Framework", max(resultados, key=resultados.get))
                st.write("Probabilidades:")
                st.json({k: f"{v:.2%}" for k, v in resultados.items()})
            
            with col2:
                st.bar_chart(pd.DataFrame({
                    'Framework': resultados.keys(),
                    'Tiempo Entrenamiento': recursos['tiempos'].values()
                }).set_index('Framework'))

if __name__ == "__main__":
    main()
