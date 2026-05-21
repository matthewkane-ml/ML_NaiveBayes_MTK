# Naive Bayes — Análisis de Sentimientos en Reseñas de Apps

> Pipeline de clasificación NLP que predice si una reseña de Google Play Store es positiva o negativa: preprocesamiento de texto con CountVectorizer, MultinomialNB seleccionado sobre las variantes Gaussian y Bernoulli por razones teóricas, y optimizado con RandomizedSearchCV — con una comparación de Random Forest que muestra el equilibrio entre interpretabilidad y potencia del ensemble.

---

## Problema

Clasificar reseñas de usuarios de Google Play Store como sentimiento positivo o negativo. Automatizar esto permite a los desarrolladores de apps monitorizar la satisfacción de los usuarios a escala sin leer manualmente miles de reseñas. Este es un problema de clasificación binaria de NLP.

## Dataset

- **Fuente:** Dataset de reseñas de Google Play Store (GitHub / 4Geeks)
- **Target:** `polarity` — 1 = sentimiento positivo, 0 = sentimiento negativo
- **Características originales:** `package_name` (eliminada), `review` (texto — el único predictor)

## Pipeline

| Paso | Herramienta / Enfoque |
|---|---|
| Limpieza de texto | Eliminar espacios en blanco, convertir a minúsculas |
| Vectorización | `CountVectorizer(stop_words="english")` → matriz de recuentos de palabras (Bag-of-Words) |
| División train/test | 80/20, random_state=42 |
| Selección del modelo | MultinomialNB elegido — las características basadas en recuentos se ajustan a sus supuestos probabilísticos |
| Optimización | `RandomizedSearchCV` (50 iteraciones, validación cruzada de 5 pliegues) sobre `alpha` y `fit_prior` |
| Mejores hiperparámetros | `alpha=1,918`, `fit_prior=False` |

**¿Por qué MultinomialNB sobre Gaussian o Bernoulli?**
- **MultinomialNB** está diseñado para datos de recuentos (frecuencias de palabras) — encaje natural para Bag-of-Words
- **GaussianNB** asume características continuas con distribución normal — incorrecto para recuentos enteros de palabras
- **BernoulliNB** trata cada palabra como binaria (presente/ausente) — pierde información de frecuencia
- Los tres se entrenaron y compararon; MultinomialNB confirmado como mejor empírica y teóricamente

## Comparación de Modelos

| Modelo | Notas |
|---|---|
| MultinomialNB (línea base) | alpha=1,0 por defecto |
| MultinomialNB (optimizado) | alpha=1,918, fit_prior=False — precisión mejorada |
| RandomForestClassifier (n_estimators=60) | ~80% de precisión — sólida línea base de ensemble |

El Random Forest alcanza mayor precisión bruta pero a costa de la interpretabilidad. Naive Bayes es más rápido de entrenar, requiere mucha menos memoria en datos de texto de alta dimensionalidad, y sus probabilidades a nivel de palabra son directamente inspeccionables.

## Conclusiones Clave

- **Bag-of-Words es una línea base poderosa:** Ignorar el orden de las palabras y tratar las reseñas como conjuntos de palabras sin orden aún captura suficiente señal para clasificar el sentimiento de forma fiable — un fuerte recordatorio de que las representaciones más simples suelen funcionar bien antes de recurrir a embeddings.
- **La elección del algoritmo debe seguir al tipo de datos:** La razón teórica para elegir MultinomialNB (datos de recuentos) sobre GaussianNB (datos continuos) es tan importante como la comparación empírica de precisión.
- **`fit_prior=False` aplana la distribución previa de clases:** Cuando las clases están desbalanceadas, dejar que el modelo aprenda la distribución previa de los datos de entrenamiento puede sesgarlo hacia la clase mayoritaria — establecer `fit_prior=False` asume distribuciones previas iguales y mejora el recall de la clase minoritaria.

## Stack Tecnológico

`Python` · `scikit-learn` · `pandas` · `NLTK (stop words)`

## Ejecutar Localmente

```bash
git clone https://github.com/matthewkane-ml/ML_NaiveBayes_MTK.git
cd ML_NaiveBayes_MTK
pip install -r requirements.txt
jupyter notebook src/app.ipynb
```

El modelo entrenado se guarda en `models/` mediante `pickle`.

## Próximos Pasos

- Reemplazar CountVectorizer con **TF-IDF** para penalizar las palabras comunes que aparecen en todas las reseñas y destacar los términos distintivos
- Añadir una **matriz de confusión** y precisión/recall por clase — la precisión global en datos de sentimiento desbalanceados es un número único engañoso
- Probar **word embeddings** (word2vec o sentence-transformers) para capturar el significado semántico en lugar de solo la frecuencia de palabras, y comparar con la línea base BoW

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
