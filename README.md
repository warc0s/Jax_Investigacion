# 🚀 Investigación Comparativa: JAX vs TensorFlow vs PyTorch

![JAX Tensorflow Pytorch](https://www.askpython.com/wp-content/uploads/2021/03/front_cover-1024x512.png.webp)

Este proyecto explora **JAX**, una potente biblioteca de computación numérica de Google, y la compara con otros frameworks populares como TensorFlow y PyTorch. El objetivo principal es evaluar el rendimiento y la eficiencia de JAX en tareas de aprendizaje automático.

## 📚 Índice
1. [Estructura ](#-estructura)
1. [¿Qué es JAX?](#-qué-es-jax)
2. [Comparación con TensorFlow y PyTorch](#-comparación-jax-vs-tensorflow-vs-pytorch)
3. [Ecosistema JAX](#-ecosistema-jax)
4. [Experimento Práctico](#-experimento-práctico)
5. [Resultados y Conclusiones](#-resultados-y-conclusiones)
6. [Recursos Utilizados](#-recursos-utilizados)

---

## 🏢 Estructura

- **JAX_Trabajo_Investigacion_MGE.ipynb:**  
  Notebook principal que detalla el desarrollo de la investigación, incluyendo teoría, comparaciones y ejemplos prácticos.

- **models/:**  
  Directorio que almacena los modelos entrenados y los tiempos de entrenamiento correspondientes. Contiene los siguientes archivos:
  - **jax_params.joblib:** Parámetros del modelo entrenado con JAX.
  - **jax_time.joblib:** Tiempo de entrenamiento del modelo con JAX.
  - **preprocessor.joblib:** Pipeline de preprocesamiento de datos utilizado en el proyecto.
  - **tf_model.keras:** Modelo entrenado con TensorFlow.
  - **tf_time.joblib:** Tiempo de entrenamiento del modelo con TensorFlow.
  - **torch_model.pth:** Modelo entrenado con PyTorch.
  - **torch_time.joblib:** Tiempo de entrenamiento del modelo con PyTorch.

## 🧠 ¿Qué es JAX?
Framework de Google para **cómputo numérico acelerado** con:
- ✅ Diferenciación automática de alto orden
- ✅ Compilación JIT (Just-In-Time)
- ✅ Programación funcional pura
- ✅ Soporte nativo para GPU/TPU

*Detalles completos en el [Jupyter Notebook](JAX_Trabajo_Investigacion_MGE.ipynb)*

---

## 🥊 Comparación: JAX vs TensorFlow vs PyTorch

| Característica       | JAX 🏎️          | TensorFlow 🏗️     | PyTorch 🔥        |
|----------------------|----------------|-------------------|------------------|
| Paradigma            | Funcional      | Híbrido           | Imperativo       |
| Entrenamiento (CPU)    | **7 segundos** | 220 segundos      | 17 segundos      |
| Autodiferenciación   | Hessianas      | Gradient Tape     | Autograd         |
| Caso ideal           | Investigación  | Producción        | Prototipado      |

**Punto clave:**  
JAX brilla en velocidad (7x más rápido que PyTorch, 31x que TensorFlow en nuestro test) gracias a XLA y JIT.

---

## 🌐 Ecosistema JAX
Librerías clave que potencian su uso:
- **Flax/Haiku** → Construcción de redes neuronales
- **Optax** → Optimizadores avanzados
- **JAX-MD** → Simulaciones científicas
- **NumPyro** → Programación probabilística

Integración con: TensorFlow Datasets 🤝 HuggingFace Transformers 🤝 JAX AI Stack

---

## 🔬 Experimento Práctico
### Bank Marketing Dataset (Clasificación binaria)

**Flujo del experimento:**
1. Preprocesamiento con scikit-learn
2. Entrenamiento idéntico en 3 frameworks
3. Comparación de tiempos y código

---

## 📊 Resultados y Conclusiones

- ⚡ JAX fue **el más rápido**: 7s vs 17s (PyTorch) y 220s (TensorFlow)
- ✍️ Sintaxis concisa y estilo funcional, aunque con más lineas de código
- 🧠 Ideal para investigación científica y optimizaciones complejas
- 🚧 Menor soporte para despliegue en producción vs TensorFlow

---

## 📖 Recursos Utilizados
- https://es.statisticseasily.com/glosario/what-is-jax-high-performance-computing/
- https://opensource.googleblog.com/2024/12/a-robust-open-ecosystem-accelerating-ai-infrastructure.html
- https://github.com/jax-ml/jax-ai-stack
- https://github.com/n2cholas/awesome-jax

---

⌨️ **Nota:** Todos los experimentos se ejecutaron en CPU usando Google Colab.  
🔍 ¡Revisa el [Jupyter Notebook](JAX_Trabajo_Investigacion_MGE.ipynb) para detalles técnicos y análisis completo!
