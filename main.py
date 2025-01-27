import streamlit as st
import joblib
import numpy as np
import jax.numpy as jnp
from flax.training import train_state
import tensorflow as tf
import torch
import torch.nn as nn

# Cargar preprocesador y modelos
preprocessor = joblib.load("preprocessor.joblib")
times = {
    'JAX': joblib.load("jax_time.joblib"),
    'TensorFlow': joblib.load("tf_time.joblib"),
    'PyTorch': joblib.load("torch_time.joblib")
}

# Cargar modelos
## JAX
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

jax_params = joblib.load("jax_params.joblib")
jax_model = MLP()

## TensorFlow
tf_model = tf.keras.models.load_model("tf_model.keras")

## PyTorch
class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

input_size = preprocessor.transformers_[0][2].shape[0] + \
             preprocessor.transformers_[1][1].get_feature_names_out().shape[0]
torch_model = Net(input_size)
torch_model.load_state_dict(torch.load("torch_model.pth"))

# Interfaz
st.title("Comparación de Frameworks de ML")
st.write("Predicción de Marketing Bancario")

# Inputs de usuario
input_data = {}
for feature in bank.data.columns:
    if feature in numerical_features:
        input_data[feature] = st.number_input(feature)
    else:
        categories = bank.data[feature].unique().tolist()
        input_data[feature] = st.selectbox(feature, categories)

# Preprocesar input
input_df = pd.DataFrame([input_data])
processed_input = preprocessor.transform(input_df)

# Predicciones
if st.button("Predecir"):
    # JAX
    jax_input = jnp.array(processed_input.astype(np.float32))
    jax_pred = jax_model.apply({'params': jax_params}, jax_input)
    prob_jax = 1 / (1 + np.exp(-jax_pred[0][0]))
    
    # TensorFlow
    tf_pred = tf_model.predict(processed_input, verbose=0)[0][0]
    
    # PyTorch
    torch_input = torch.tensor(processed_input.astype(np.float32))
    torch_pred = torch_model(torch_input).detach().numpy()[0][0]
    
    # Resultados
    st.subheader("Probabilidad de suscripción:")
    st.write(f"JAX: {prob_jax:.2%} (Tiempo entrenamiento: {times['JAX']:.2f}s)")
    st.write(f"TensorFlow: {tf_pred:.2%} (Tiempo entrenamiento: {times['TensorFlow']:.2f}s)")
    st.write(f"PyTorch: {torch_pred:.2%} (Tiempo entrenamiento: {times['PyTorch']:.2f}s)")
