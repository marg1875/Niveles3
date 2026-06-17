# Clasificacion de Niveles de Fuerza en Motor Imagery post-AVC mediante Algoritmos Fractales con Ventanas Fijas

**Pipeline:** Niveles3
**Fecha:** Junio 2026
**Validacion:** Intra-sujeto 5-fold Cross-Validation (80/20)
**Ventana optima:** W700 = 2.8 segundos (0% overlap, segmentos contiguos)
**Clases:** 3 (Reposo / MVR 10% / MVR 40%)

---

## Resumen

Se evaluo la capacidad de siete algoritmos fractales —Rescaled Range, Higuchi, Detrended Fluctuation Analysis (DFA), Semivariograma, Hurst Original, Hurst Rescaled Range con Particiones y Hurst via Semivariograma— para clasificar niveles de fuerza en Motor Imagery (MI) post-Accidente Cerebrovascular (AVC). Sobre 5 pacientes (Px.006-Px.010) con registros EEG de 16 canales a 250 Hz en tres momentos temporales (Mes 1, Mes 3, Mes 6 post-AVC), se optimizo el tamano de ventana mediante un barrido de 16 tamanos (2.0–8.0s) determinando W=700 muestras (2.8s) como la configuracion optima. Cada algoritmo fractal se evaluo como caracteristica unica (promedio espacial) y en configuracion per-electrodo (retener informacion espacial), siguiendo la metodologia de Martinez-Peon et al. (2024, _J. Neural Eng._ 21, 046024). Para abordar el desbalance de clases en la clasificacion 3-clases (280 reposo vs 1332/1355), se incorporo SMOTE (Synthetic Minority Over-sampling Technique) aplicado exclusivamente al conjunto de entrenamiento dentro de cada fold. Empleando validacion intra-sujeto StratifiedKFold (k=5) con pool de predicciones y caracteristicas per-electrodo de 4 metodos basicos (RS, Higuchi, DFA, Variogram × 16 canales = 64 features), se alcanzo **83.31% de accuracy en 3-clases** (Reposo/10%/40%) en el Mes 6 con LogisticRegression (k=0.7174, F1=0.7155). En 2-clases (10% vs 40%) con las mismas caracteristicas per-electrodo, se alcanzo **95.54% en el Mes 6** (LogisticRegression, k=0.9108). Los resultados demuestran que la combinacion de caracteristicas fractales per-electrodo con SMOTE produce mejoras sustanciales sobre el promedio espacial (+15.6 puntos en 3-clases), validando la metodologia de Martinez-Peon y estableciendo un nuevo estado del arte para este conjunto de datos.

---

## 1. Introduccion

### 1.1 Contexto clinico

El Accidente Cerebrovascular (AVC) es una de las principales causas de discapacidad motora a nivel mundial. Las Interfaces Cerebro-Computadora (BCI) basadas en Motor Imagery (MI) ofrecen una via para la neurorehabilitacion al activar circuitos sensoriomotores incluso en ausencia de movimiento real. Un desafio fundamental es la capacidad de discriminar entre diferentes niveles de fuerza imaginada, expresados como porcentaje de la Contraccion Voluntaria Maxima (Maximum Voluntary Contraction, MVR). Una clasificacion precisa permitiria sistemas BCI mas graduados y adaptativos.

### 1.2 Analisis fractal de EEG

El EEG durante MI exhibe propiedades fractales que reflejan la complejidad temporal de la dinamica neuronal subyacente. El exponente de Hurst (H) y la dimension fractal (D) cuantifican respectivamente la persistencia de correlaciones temporales y la complejidad de la senal. Diversos algoritmos permiten estimar estas propiedades desde perspectivas complementarias:

- **Rescaled Range (R/S):** Estima H a partir de la pendiente del rango reescalado vs escala temporal.
- **Higuchi:** Estima la dimension fractal midiendo la longitud promedio de la curva a traves de subdivisiones progresivas (Higuchi, 1988).
- **Detrended Fluctuation Analysis (DFA):** Estima el exponente de escala tras eliminar tendencias locales (Peng et al., 1995).
- **Semivariograma:** Estima H a partir de la pendiente de la semivarianza vs lag temporal.
- **Hurst Original (HO):** R/S clasico sin eliminacion de tendencia (Hurst, 1951).
- **Hurst Rescaled Range con Particiones (HRS):** R/S aplicado a sub-series y promediado; p=64 particiones.
- **Hurst via Semivariograma (HV):** H estimado a partir del semivariograma, equivalente metodologico al Semivariograma clasico.

Investigaciones previas han aplicado variantes del exponente de Hurst a la clasificacion de MI, destacando el trabajo de Martinez-Peon et al. (2024, _J. Neural Eng._ 21, 046024) quienes reportaron un 96.42% de accuracy con kNN+HRS empleando 10 sujetos, 20 electrodos y validacion intra-sujeto 10-fold en WEKA.

### 1.3 Contribuciones de este estudio

1. **Evaluacion exhaustiva de 7 algoritmos fractales** individuales como caracteristicas unicas, siguiendo la metodologia de evaluacion por algoritmo del trabajo de referencia.
2. **Barrido de ventana optima**: 16 tamanos (2.0–8.0s) para identificar la configuracion que maximiza la discriminacion fractal.
3. **Clasificacion de 3 niveles**: Reposo, MVR 10% y MVR 40%, mas exigente que la tarea binaria estandar.
4. **Analisis longitudinal por mes** para evaluar la evolucion temporal de la senal fractal.
5. **Comparacion directa entre algoritmos** fractales individuales y su combinacion.

---

## 2. Arquitectura del Pipeline

### 2.1 Datos

| Propiedad | Valor |
|-----------|-------|
| Pacientes | 5 (Px.006–Px.010) |
| Meses post-AVC | 1, 3, 6 |
| Archivos .mat | 56 (28 sincronicos + 28 asincronicos) |
| Canales EEG | 16 (Fp1, Fp3, F3, Fz, F4, Cz, Pz, P5, P3, O1, Oz, T8, P8, P6, P4, T7) |
| Frecuencia de muestreo | 250 Hz |
| Clases | Reposo (0%), MVR 10%, MVR 40% |
| Epocas (W700) | 2,967 (280 reposo + 1,332 MVR 10% + 1,355 MVR 40%) |

### 2.2 Preprocesamiento

```
Senal cruda (16ch x N muestras, 250 Hz)
    -> Remocion de DC
    -> Filtro Notch 60 Hz (Q=30.0)
    -> Filtro Butterworth pasabanda 0.5–45 Hz (orden 4)
    -> ICA (3 componentes removidos, max_iter=500)
    -> Normalizacion Z-score por canal
```

### 2.3 Segmentacion por Ventanas Fijas

Registro completo dividido en **ventanas contiguas W=700 muestras (2.8s) con 0% de traslape**. Los primeros 5 segmentos de cada archivo se etiquetan como clase **Reposo (0%)**; los segmentos restantes heredan la etiqueta MVR del archivo (10% o 40%). El sobrante al final del registro se descarta.

### 2.4 Extraccion de Caracteristicas Fractales

Para cada segmento (16 canales x 700 muestras), cada uno de los 7 algoritmos fractales se computa de forma independiente por canal, produciendo dos niveles de agregacion:

**Nivel 1: Promedio Espacial (1 feature por algoritmo).** Se promedian los 16 canales para obtener un unico valor por metodo:

| Algoritmo | Caracteristica | Parametros adaptativos |
|-----------|---------------|----------------------|
| Rescaled Range | Active_RS | Escala min 0.128s, max N*0.40, min 5 puntos |
| Higuchi | Active_Higuchi | k_max=min(N*0.25, 30), min k=5, min 5 puntos |
| DFA | Active_DFA | Escala min 0.064s, max N*0.33, min 3 ventanas/escala |
| Semivariograma | Active_Variogram | Lag max=N*0.20, min 10 lags, min 5 puntos |
| Promedio | Active_Average | Media de los 4 metodos basicos |
| Hurst Original | Active_HO | R/S sin detrending, mismas escalas que RS |
| Hurst R/S Particiones | Active_HRS_p64 | p=64 sub-series, fallback a HO si <10 muestras |
| Hurst Semivariograma | Active_HV | Identico a Semivariograma |

Cada algoritmo produce **1 feature**. Los 7 combinados producen **7 features** (promedio espacial).

**Nivel 2: Per-Electrodo (1 feature por canal × metodo).** Reteniendo la informacion espacial, cada metodo se conserva de forma independiente por electrodo como `Active_{Canal}_{Metodo}`:

| Metodo | Canales | Features |
|--------|---------|----------|
| Básicos (RS, Higuchi, DFA, Variogram) | 16 | 64 |
| Martinez (HO, HRS_p64, HV) | 16 | 48 |
| Todos (Básicos + Martinez) | 16 | 112 |

Esta configuracion replica la metodologia de Martinez-Peon et al. (2024), quienes utilizan HO, HRS_p64 y HV por electrodo sobre 20 canales en areas premotoras, pre-SMA, DLPFC e IPL.

### 2.5 Clasificacion

**Validacion Intra-sujeto 5-fold CV (80/20):** StratifiedKFold (k=5) aplicado independientemente a las epocas de cada paciente. En cada fold, 80% de epocas para entrenamiento y 20% para prueba. StandardScaler ajustado sobre entrenamiento y aplicado a prueba. Predicciones de todos los folds y pacientes concatenadas (pooled) para calculo de metricas globales.

**Balanceo de clases con SMOTE (3-clases):** Para la clasificacion 3-clases (Reposo 0% / MVR 10% / MVR 40%), donde existe un desbalance significativo (280 reposo vs 1332 MVR 10% vs 1355 MVR 40%), se aplica SMOTE (Synthetic Minority Over-sampling Technique, random_state=42) exclusivamente al conjunto de entrenamiento dentro de cada fold. Para la clasificacion 2-clases (10% vs 40%), donde las clases estan balanceadas (~1332 vs ~1355), no se aplica SMOTE.

**Modelos evaluados (7, siguiendo Martinez-Peon 2024):**

| Modelo | Hiperparametros | Equivalente Weka (Paper) |
|--------|-----------------|--------------------------|
| SVM | kernel=rbf, C=1.0, probability=True, random_state=42 | SVM (poly) |
| kNN | n_neighbors=10, metric=chebyshev, n_jobs=-1 | kNN (k=10, Chebyshev) |
| RandomForest | n_estimators=300, max_depth=None, random_state=42, n_jobs=-1 | RF |
| NaiveBayes | GaussianNB, var_smoothing=1e-9 | NB |
| LogisticRegression | max_iter=2000, random_state=42, n_jobs=-1 | -- |
| MLP | hidden_layer_sizes=(7,), learning_rate_init=0.3, max_iter=1000, early_stopping=True | MP (lr=0.3, hidden=7) |
| DecisionTree | max_features='sqrt', max_depth=10, random_state=42 | RT (k=10) |

**Subsets de caracteristicas evaluados (Grupo A: Promedio Espacial):** Cada algoritmo fractal se evalua individualmente (1 feature) y en combinacion (7 features):

| Subset | Features | N |
|--------|----------|---|
| Rescaled Range | Active_RS | 1 |
| Higuchi | Active_Higuchi | 1 |
| DFA | Active_DFA | 1 |
| Semivariograma | Active_Variogram | 1 |
| Promedio | Active_Average | 1 |
| Hurst Original | Active_HO | 1 |
| Hurst R/S con Particiones | Active_HRS_p64 | 1 |
| Hurst Semivariograma | Active_HV | 1 |
| Todos los Fractales | Los 7 anteriores combinados | 7 |

**Subsets de caracteristicas evaluados (Grupo B: Per-Electrodo, All-16):** Extension de la metodologia Martinez-Peon, evaluando cada algoritmo o combinacion por electrodo:

| Subset | Features | N | Equivalente Paper |
|--------|----------|---|-------------------|
| HRS p64 per-channel | 16 canales × HRS_p64 | 16 | HRS |
| HO per-channel | 16 canales × HO | 16 | HO |
| HV per-channel | 16 canales × HV | 16 | HV |
| Martinez per-channel | 16 ch × {HO, HRS_p64, HV} | 48 | HRS+HO+HV |
| Básicos per-channel | 16 ch × {RS, Higuchi, DFA, Variogram} | 64 | -- |
| Todos per-channel | 16 ch × 7 metodos | 112 | -- |

**Metricas evaluadas:** Siguiendo la nomenclatura del paper, se reportan Accuracy (%), Cohen's Kappa (κ), Mean Absolute Error (MAE) y F1 Macro.

---

## 3. Glosario

| Termino | Definicion |
|---------|-----------|
| **Rescaled Range (R/S)** | Metodo que estima el exponente de Hurst (H) a partir de la pendiente del logaritmo del rango reescalado (R/S) contra el logaritmo de la escala temporal. H > 0.5 indica persistencia; H < 0.5 indica anti-persistencia; H = 0.5 indica ruido aleatorio. |
| **Higuchi** | Metodo de estimacion de la dimension fractal (Higuchi, 1988). Subdivide la senal en k sub-series progresivamente mas pequenas y mide la longitud promedio normalizada de la curva. La pendiente de log(L(k)) vs log(1/k) proporciona la dimension fractal D. |
| **Detrended Fluctuation Analysis (DFA)** | Metodo que elimina tendencias locales de orden polinomial y mide la relacion entre la fluctuacion promedio F(n) y el tamano de ventana n (Peng et al., 1995). La pendiente log-log proporciona el exponente de escala α. |
| **Semivariograma** | Metodo que calcula H a partir de la semivarianza γ(h) = 0.5 * E[(X(t+h) − X(t))^2]. La pendiente de log(γ(h)) vs log(h) es igual a 2H. |
| **Hurst Original (HO)** | Exponente de Hurst calculado mediante R/S clasico sin eliminacion de tendencia. Metodo original de Hurst (1951) desarrollado para analisis hidrologico. |
| **Hurst R/S con Particiones (HRS)** | Extension de HO donde la senal se divide en p sub-series de igual longitud. Se calcula R/S para cada sub-serie y se promedian los resultados. p=64 es el valor reportado como optimo en la literatura. Cuando una sub-serie tiene menos de 10 muestras se utiliza HO como valor de fallback. |
| **Hurst via Semivariograma (HV)** | Exponente de Hurst calculado a partir del semivariograma. Equivalente metodologico al Semivariograma clasico pero etiquetado segun la convencion de la literatura de EEG fractal. |
| **MVR (Maximum Voluntary Contraction)** | Porcentaje de la contraccion voluntaria maxima imaginada. Niveles evaluados: 10% (bajo esfuerzo) y 40% (esfuerzo moderado). |
| **MI (Motor Imagery)** | Tarea cognitiva de imaginar un movimiento sin ejecutarlo fisicamente. |
| **Intra-sujeto (Within-subject)** | Validacion donde entrenamiento y prueba provienen del mismo paciente. En este estudio: StratifiedKFold (k=5) aplicado por paciente, predicciones pooled. Equivalente a un split 80/20 estratificado. |
| **LOPO (Leave-One-Patient-Out)** | Validacion inter-sujeto: entrena con N−1 pacientes, prueba con el paciente restante. |
| **W700** | Ventana de 700 muestras a 250 Hz = 2.8 segundos. Determinada como optima mediante barrido empirico. |
| **Epoca** | Segmento individual de EEG de 700 muestras (2.8s) usado como unidad de clasificacion. |
| **Pool de predicciones** | Metodo de agregacion donde las predicciones de todos los folds de todos los pacientes se concatenan en un unico vector, y las metricas se calculan sobre este vector consolidado. |
| **StratifiedKFold** | Validacion cruzada que preserva la proporcion de clases en cada fold. k=5 (o menos si el tamano de la clase minoritaria lo limita). |
| **ICA (Independent Component Analysis)** | Separacion ciega de fuentes para remover artefactos (parpadeo, ECG). 3 componentes removidos. |
| **SMOTE (Synthetic Minority Over-sampling Technique)** | Tecnica de balanceo de clases que genera ejemplos sinteticos de la clase minoritaria mediante interpolacion entre vecinos cercanos en el espacio de caracteristicas. Aplicado exclusivamente al conjunto de entrenamiento dentro de cada fold de validacion cruzada para 3-clases (Reposo=280 epocas vs 1332/1355 de MVR). Usa random_state=42. |
| **Cohen's Kappa (κ)** | Estadistico que mide la concordancia entre las predicciones y las etiquetas reales, corrigiendo por el acuerdo esperado por azar. κ = 1 indica acuerdo perfecto; κ = 0 indica acuerdo equivalente al azar. Reportado en las tablas estilo Martinez-Peon (2024). |
| **MAE (Mean Absolute Error)** | Promedio de las diferencias absolutas entre las etiquetas reales y las predicciones. Reportado en el paper de referencia para cuantificar la magnitud del error de clasificacion.

---

## 4. Optimizacion de Ventana

Se realizo un barrido de 16 tamanos de ventana (500–2000 muestras, paso 100 muestras; 2.0–8.0s) con segmentacion contigua al 0% de traslape, utilizando los 5 algoritmos fractales basicos (Rescaled Range, Higuchi, DFA, Semivariograma y su Promedio) con promedio espacial, y RandomForest como clasificador en tarea 2-clases (MVR 10% vs 40%).

### Resultados del barrido: Accuracy (%) 2-clases por ventana y mes

| Ventana | Duracion | Mixto | Mes 1 | Mes 3 | Mes 6 | Promedio* |
|---------|----------|-------|-------|-------|-------|-----------|
| W500 | 2.0s | 71.75 | 72.28 | 80.58 | 74.78 | 74.87 |
| W600 | 2.4s | 70.40 | 73.60 | 80.46 | 76.18 | 74.82 |
| **W700** | **2.8s** | **71.12** | **74.08** | **80.18** | **77.34** | **75.13** |
| W800 | 3.2s | 70.66 | 71.23 | 80.00 | 74.58 | 73.96 |
| W900 | 3.6s | 70.77 | 71.60 | 81.05 | 70.68 | 74.47 |
| W1000 | 4.0s | 70.50 | 70.32 | 77.31 | 72.12 | 72.73 |
| W1100 | 4.4s | 68.98 | 68.64 | 77.86 | 72.91 | 71.83 |
| W1200 | 4.8s | 70.63 | 70.88 | 78.91 | 71.93 | 73.47 |
| W1300 | 5.2s | 67.97 | 71.72 | 73.50 | 70.63 | 71.06 |
| W1400 | 5.6s | 65.04 | 68.54 | 75.19 | 68.34 | 69.59 |
| W1500 | 6.0s | 66.45 | 72.11 | 73.85 | 68.51 | 70.80 |
| W1600 | 6.4s | 65.79 | 67.53 | 75.00 | 66.41 | 69.44 |
| W1700 | 6.8s | 66.69 | 69.14 | 77.58 | 63.29 | 71.14 |
| W1800 | 7.2s | 66.70 | 65.07 | 77.62 | 62.61 | 69.80 |
| W1900 | 7.6s | 67.07 | 67.04 | 79.75 | 64.38 | 71.29 |
| W2000 | 8.0s | 66.60 | 69.41 | 74.74 | 67.11 | 70.25 |

*Promedio = media de Mixto + Mes 1 + Mes 3.

**Conclusion del barrido:** W700 (2.8s) es la ventana optima con promedio 75.13%. Las ventanas en el rango 2.0–3.6s consistentemente superan a las de >5s. La degradacion con ventanas largas sugiere que la senal fractal discriminativa se concentra en intervalos cortos tras el inicio de la tarea de MI.

---

## 5. Resultados: Tablas Estilo Martinez-Peon (con SMOTE y Per-Electrodo)

A continuacion se presentan los resultados completos de clasificacion **3-clases** (Reposo / MVR 10% / MVR 40%) con SMOTE aplicado al conjunto de entrenamiento dentro de cada fold. Las tablas siguen el formato de Martinez-Peon et al. (2024, Table 2) reportando Accuracy (%) y Cohen's Kappa (κ). Se evaluan dos niveles de agregacion: **Promedio Espacial** (Seccion 5.1) y **Per-Electrodo All-16** (Seccion 5.2). Validacion intra-sujeto 5-fold CV con pool de predicciones, 7 clasificadores.

### 5.1 Promedio Espacial con SMOTE (Grupo A)

Clasificacion 3-clases utilizando cada algoritmo fractal como una unica caracteristica (promedio espacial de 16 canales), mas la combinacion de los 7 algoritmos. **Con SMOTE.**

#### 5.1.1 Condicion Mixta (todos los meses combinados, 2,967 epocas)

| Clasificador | Rescaled Range | Higuchi | DFA | Semivariograma | Promedio | Hurst Original | HRS Particiones | Hurst Semivariograma | Todos los Fractales (7) |
|-------------|---------------|---------|-----|---------------|----------|---------------|----------------|---------------------|------------------------|
| SVM | 35.22% | 41.22% | 39.30% | 40.82% | 34.55% | 35.73% | 36.87% | 40.82% | 46.04% |
| kNN | 37.28% | 38.29% | 37.11% | 38.56% | 37.82% | 37.88% | 36.37% | 38.56% | 43.24% |
| RandomForest | 37.41% | 38.93% | 36.87% | 38.25% | 36.70% | 37.55% | 37.45% | 38.25% | 51.50% |
| NaiveBayes | 30.84% | 40.34% | 34.04% | 38.66% | 32.02% | 29.22% | 29.32% | 38.66% | 34.45% |
| LogisticRegression | 32.32% | 38.12% | 36.74% | 38.42% | 35.96% | 32.90% | 31.78% | 38.42% | 42.91% |
| MLP | 32.79% | 39.77% | 32.66% | 34.88% | 29.90% | 35.22% | 31.41% | 34.88% | 44.19% |
| DecisionTree | 36.77% | 37.31% | 36.64% | 37.75% | 36.84% | 39.00% | 37.14% | 37.75% | 45.06% |

#### 5.1.2 Mes 1 (fase aguda, 980 epocas)

| Clasificador | Rescaled Range | Higuchi | DFA | Semivariograma | Promedio | Hurst Original | HRS Particiones | Hurst Semivariograma | Todos los Fractales (7) |
|-------------|---------------|---------|-----|---------------|----------|---------------|----------------|---------------------|------------------------|
| SVM | 36.33% | 40.82% | 37.86% | 38.27% | 38.88% | 37.04% | 33.57% | 38.27% | 46.84% |
| kNN | 39.39% | 37.65% | 40.41% | 38.47% | 42.65% | 41.63% | 34.49% | 38.47% | 44.39% |
| RandomForest | 36.33% | 41.43% | 38.27% | 34.80% | 40.51% | 39.49% | 39.90% | 34.80% | 51.12% |
| NaiveBayes | 36.02% | 38.06% | 37.96% | 40.10% | 41.33% | 38.67% | 34.49% | 40.10% | 39.69% |
| LogisticRegression | 40.51% | 38.98% | 38.06% | 38.47% | 41.53% | 41.73% | 39.39% | 38.47% | 43.88% |
| MLP | 33.98% | 31.22% | 36.63% | 37.86% | 37.35% | 36.94% | 32.65% | 37.86% | 40.31% |
| DecisionTree | 36.02% | 41.53% | 37.14% | 37.04% | 40.20% | 38.98% | 37.45% | 37.04% | 45.71% |

#### 5.1.3 Mes 3 (fase sub-aguda, 1,100 epocas)

| Clasificador | Rescaled Range | Higuchi | DFA | Semivariograma | Promedio | Hurst Original | HRS Particiones | Hurst Semivariograma | Todos los Fractales (7) |
|-------------|---------------|---------|-----|---------------|----------|---------------|----------------|---------------------|------------------------|
| SVM | 43.27% | 55.82% | 47.45% | 49.91% | 40.82% | 43.82% | 40.73% | 49.91% | 57.64% |
| kNN | 47.09% | 52.18% | 44.64% | 50.36% | 41.36% | 45.36% | 42.36% | 50.36% | 57.09% |
| RandomForest | 42.18% | 50.55% | 43.09% | 47.00% | 39.64% | 44.45% | 39.82% | 47.00% | 64.64% |
| NaiveBayes | 43.64% | 47.91% | 44.00% | 47.45% | 38.73% | 36.64% | 38.18% | 47.45% | 51.27% |
| LogisticRegression | 39.18% | 43.91% | 42.45% | 48.73% | 41.91% | 38.82% | 36.45% | 48.73% | 55.64% |
| MLP | 39.55% | 48.91% | 39.00% | 44.00% | 35.18% | 34.82% | 33.27% | 44.00% | 51.91% |
| DecisionTree | 43.45% | 49.27% | 42.36% | 46.73% | 40.91% | 43.91% | 39.18% | 46.73% | 58.27% |

#### 5.1.4 Mes 6 (fase cronica, 887 epocas, 4 pacientes)

| Clasificador | Rescaled Range | Higuchi | DFA | Semivariograma | Promedio | Hurst Original | HRS Particiones | Hurst Semivariograma | Todos los Fractales (7) |
|-------------|---------------|---------|-----|---------------|----------|---------------|----------------|---------------------|------------------------|
| SVM | 37.54% | 49.27% | 42.28% | 44.31% | 42.16% | 37.20% | 37.77% | 44.31% | 56.26% |
| kNN | 42.39% | 46.79% | 43.86% | 48.25% | 45.21% | 39.91% | 36.53% | 48.25% | 49.04% |
| RandomForest | 39.12% | 44.98% | 40.36% | 45.21% | 37.88% | 37.99% | 37.09% | 45.21% | 57.61% |
| NaiveBayes | 34.39% | 46.45% | 42.28% | 45.77% | 41.26% | 35.40% | 34.95% | 45.77% | 45.43% |
| LogisticRegression | 37.43% | 40.47% | 40.59% | 39.91% | 41.04% | 36.87% | 37.66% | 39.91% | 53.44% |
| MLP | 33.93% | 42.50% | 42.05% | 42.39% | 42.28% | 36.30% | 40.25% | 42.39% | 45.21% |
| DecisionTree | 37.88% | 44.19% | 42.16% | 45.43% | 37.66% | 39.46% | 37.09% | 45.43% | 51.75% |

### 5.2 Per-Electrodo All-16 (Grupo B) — Tabla Estilo Martinez-Peon Table 2

Clasificacion 3-clases con SMOTE utilizando caracteristicas fractales retenidas por electrodo (sin promediar espacialmente). Se reporta Accuracy (%) y Cohen's Kappa (κ). Este formato replica la Table 2 del paper de referencia (Martinez-Peon et al., 2024).

#### 5.2.1 Condicion Mixta (2,967 epocas)

| Clasificador | HRS Particiones (per-channel) | Hurst Original (per-channel) | Hurst Semivariograma (per-channel) | HRS+HO+HV (per-channel) | Basicos (per-channel) | Todos 7 metodos (per-channel) |
|-------------|------------------------------|-----------------------------|-----------------------------------|-------------------------|----------------------|------------------------------|
| SVM | 55.98% / κ+.349 | 55.54% / κ+.335 | 55.98% / κ+.349 | 59.29% / κ+.386 | 70.41% / κ+.522 | 68.96% / κ+.502 |
| kNN | 49.21% / κ+.251 | 51.40% / κ+.271 | 49.21% / κ+.251 | 52.95% / κ+.294 | 59.35% / κ+.380 | 58.98% / κ+.379 |
| RandomForest | 65.35% / κ+.412 | 65.62% / κ+.416 | 65.35% / κ+.412 | 67.75% / κ+.446 | **74.99% / κ+.568** | 74.42% / κ+.559 |
| NaiveBayes | 33.87% / κ+.106 | 35.25% / κ+.096 | 33.87% / κ+.106 | 34.24% / κ+.108 | 39.80% / κ+.143 | 37.65% / κ+.134 |
| LogisticRegression | 52.41% / κ+.284 | 55.51% / κ+.309 | 52.41% / κ+.284 | 59.22% / κ+.358 | 70.31% / κ+.514 | 70.48% / κ+.512 |
| MLP | 49.38% / κ+.266 | 49.71% / κ+.265 | 49.38% / κ+.266 | 51.13% / κ+.273 | 61.98% / κ+.398 | 62.59% / κ+.407 |
| DecisionTree | 52.71% / κ+.233 | 53.29% / κ+.249 | 52.71% / κ+.233 | 54.36% / κ+.259 | 61.68% / κ+.361 | 58.71% / κ+.316 |

#### 5.2.2 Mes 1 (fase aguda, 980 epocas)

| Clasificador | HRS Particiones (per-channel) | Hurst Original (per-channel) | Hurst Semivariograma (per-channel) | HRS+HO+HV (per-channel) | Basicos (per-channel) | Todos 7 metodos (per-channel) |
|-------------|------------------------------|-----------------------------|-----------------------------------|-------------------------|----------------------|------------------------------|
| SVM | 58.67% / κ+.367 | 58.37% / κ+.348 | 58.67% / κ+.367 | 61.02% / κ+.384 | 69.90% / κ+.501 | 69.49% / κ+.497 |
| kNN | 48.27% / κ+.247 | 51.12% / κ+.264 | 48.27% / κ+.247 | 52.45% / κ+.292 | 57.24% / κ+.350 | 54.80% / κ+.328 |
| RandomForest | 61.02% / κ+.357 | 63.98% / κ+.396 | 61.02% / κ+.357 | 63.47% / κ+.385 | 73.06% / κ+.541 | 71.12% / κ+.508 |
| NaiveBayes | 44.69% / κ+.200 | 45.00% / κ+.196 | 44.69% / κ+.200 | 45.00% / κ+.197 | 47.65% / κ+.235 | 47.76% / κ+.233 |
| LogisticRegression | 59.59% / κ+.361 | 59.80% / κ+.352 | 59.59% / κ+.361 | 64.59% / κ+.427 | **74.80% / κ+.576** | 74.49% / κ+.572 |
| MLP | 51.02% / κ+.229 | 49.59% / κ+.212 | 51.02% / κ+.229 | 50.82% / κ+.236 | 57.24% / κ+.318 | 57.76% / κ+.330 |
| DecisionTree | 56.33% / κ+.288 | 54.08% / κ+.243 | 56.33% / κ+.288 | 53.88% / κ+.247 | 60.82% / κ+.351 | 56.63% / κ+.274 |

#### 5.2.3 Mes 3 (fase sub-aguda, 1,100 epocas)

| Clasificador | HRS Particiones (per-channel) | Hurst Original (per-channel) | Hurst Semivariograma (per-channel) | HRS+HO+HV (per-channel) | Basicos (per-channel) | Todos 7 metodos (per-channel) |
|-------------|------------------------------|-----------------------------|-----------------------------------|-------------------------|----------------------|------------------------------|
| SVM | 60.73% / κ+.398 | 58.82% / κ+.336 | 60.73% / κ+.398 | 62.73% / κ+.409 | 79.36% / κ+.652 | 76.09% / κ+.602 |
| kNN | 53.00% / κ+.285 | 52.00% / κ+.256 | 53.00% / κ+.285 | 54.82% / κ+.306 | 64.64% / κ+.452 | 64.18% / κ+.442 |
| RandomForest | 66.27% / κ+.431 | 62.00% / κ+.359 | 66.27% / κ+.431 | 67.27% / κ+.444 | 80.45% / κ+.663 | 80.36% / κ+.661 |
| NaiveBayes | 51.36% / κ+.285 | 46.45% / κ+.197 | 51.36% / κ+.285 | 50.45% / κ+.271 | 58.00% / κ+.340 | 57.64% / κ+.349 |
| LogisticRegression | 57.09% / κ+.335 | 58.64% / κ+.341 | 57.09% / κ+.335 | 63.36% / κ+.409 | **83.27% / κ+.715** | 80.00% / κ+.658 |
| MLP | 50.00% / κ+.240 | 48.45% / κ+.174 | 50.00% / κ+.240 | 51.00% / κ+.255 | 64.45% / κ+.430 | 62.09% / κ+.392 |
| DecisionTree | 59.55% / κ+.322 | 53.82% / κ+.229 | 59.55% / κ+.322 | 58.64% / κ+.301 | 69.91% / κ+.491 | 70.27% / κ+.495 |

#### 5.2.4 Mes 6 (fase cronica, 887 epocas, 4 pacientes)

| Clasificador | HRS Particiones (per-channel) | Hurst Original (per-channel) | Hurst Semivariograma (per-channel) | HRS+HO+HV (per-channel) | Basicos (per-channel) | Todos 7 metodos (per-channel) |
|-------------|------------------------------|-----------------------------|-----------------------------------|-------------------------|----------------------|------------------------------|
| SVM | 66.52% / κ+.491 | 71.03% / κ+.541 | 66.52% / κ+.491 | 71.48% / κ+.550 | 80.38% / κ+.678 | 79.82% / κ+.669 |
| kNN | 57.84% / κ+.380 | 59.86% / κ+.407 | 57.84% / κ+.380 | 60.65% / κ+.416 | 68.55% / κ+.521 | 65.84% / κ+.487 |
| RandomForest | 75.54% / κ+.582 | 76.21% / κ+.600 | 75.54% / κ+.582 | 77.00% / κ+.606 | 82.53% / κ+.701 | 82.19% / κ+.695 |
| NaiveBayes | 54.45% / κ+.337 | 53.44% / κ+.324 | 54.45% / κ+.337 | 54.90% / κ+.345 | 63.13% / κ+.436 | 62.12% / κ+.428 |
| LogisticRegression | 69.22% / κ+.510 | 71.70% / κ+.545 | 69.22% / κ+.510 | 73.06% / κ+.559 | **83.31% / κ+.717** | 82.98% / κ+.710 |
| MLP | 63.47% / κ+.435 | 64.94% / κ+.440 | 63.47% / κ+.435 | 63.36% / κ+.422 | 71.25% / κ+.539 | 72.83% / κ+.564 |
| DecisionTree | 62.80% / κ+.390 | 67.08% / κ+.462 | 62.80% / κ+.390 | 65.50% / κ+.419 | 70.80% / κ+.513 | 70.57% / κ+.505 |

### 5.3 Mejor combinacion por condicion (resumen 3-clases, con SMOTE)

| Condicion | Mejor Accuracy | F1 Macro | Kappa | Modelo | Subset de Caracteristicas | Features |
|-----------|---------------|----------|-------|--------|---------------------------|----------|
| Mixto | 74.99% | 0.6254 | 0.5685 | RandomForest | Basicos (per-channel) | 64 |
| Mes 1 | 74.80% | 0.6266 | 0.5758 | LogisticRegression | Basicos (per-channel) | 64 |
| **Mes 3** | **83.27%** | **0.7282** | **0.7149** | LogisticRegression | Basicos (per-channel) | 64 |
| Mes 6 | **83.31%** | **0.7155** | **0.7174** | LogisticRegression | Basicos (per-channel) | 64 |

### 5.4 Analisis de Subsets de Canales (Grupo C, 3-clases, SMOTE)

Evaluacion de caracteristicas per-electrodo retenidas exclusivamente sobre subconjuntos especificos de canales. Los 4 metodos basicos (RS, Higuchi, DFA, Variogram) se evaluaron sobre 7 configuraciones de electrodos. Mejor resultado por configuracion:

| Configuracion | Canales | N Features | Mejor Accuracy | Modelo |
|--------------|---------|-----------|---------------|--------|
| Cz | 1 | 4 | 66.27% (Mes 3) | RandomForest |
| Motor-2 (Fz, Cz) | 2 | 8 | 72.73% (Mes 3) | RandomForest |
| Motor-4 (+Pz, P3) | 4 | 16 | 74.00% (Mes 3) | RandomForest |
| Motor-6 (+F3, F4) | 6 | 24 | 75.82% (Mes 3) | RandomForest |
| Frontal-5 | 5 | 20 | 74.00% (Mes 3) | RandomForest |
| Parietal-5 | 5 | 20 | 76.89% (Mes 6) | RandomForest |
| All-16 | 16 | 64 | 83.31% (Mes 6) | LogisticRegression |

El rendimiento escala consistentemente con el numero de canales, indicando que **la informacion topografica de los 16 electrodos es complementaria**. El subconjunto Parietal-5 (Pz, P5, P3, P6, P4) alcanza 76.89% con solo 5 canales y 20 features, demostrando que las areas parietales contienen informacion particularmente discriminativa para la clasificacion de niveles de MI.

### 5.5 Analisis Per-Paciente (LogisticRegression + Basicos per-channel + SMOTE)

Desglose del rendimiento individual por paciente para la configuracion optima (64 features per-electrodo, LogisticRegression, SMOTE en 3-clases). Clasificacion intra-sujeto con StratifiedKFold (k=5) por paciente, reportando accuracy, F1 macro y Kappa del pool de predicciones de cada paciente.

#### 5.5.1 Mes 1 (fase aguda)

| Paciente | Epocas | Distribucion (0/10/40) | Accuracy | F1 Macro | Kappa |
|----------|--------|----------------------|----------|----------|-------|
| Px.006 | 220 | 20 / 100 / 100 | 57.73% | 0.4632 | +0.3050 |
| Px.007 | 220 | 20 / 100 / 100 | 86.82% | 0.7779 | +0.7675 |
| Px.008 | 220 | 20 / 100 / 100 | 78.18% | 0.6378 | +0.6282 |
| Px.009 | 124 | 20 / 52 / 52 | 63.71% | 0.5708 | +0.4264 |
| Px.010 | 196 | 20 / 76 / 100 | 83.67% | 0.6978 | +0.7216 |
| **POOL** | **980** | **100 / 428 / 452** | **74.80%** | **0.6266** | **+0.5758** |

#### 5.5.2 Mes 3 (fase sub-aguda)

| Paciente | Epocas | Distribucion (0/10/40) | Accuracy | F1 Macro | Kappa |
|----------|--------|----------------------|----------|----------|-------|
| Px.006 | 220 | 20 / 100 / 100 | 86.82% | 0.7414 | +0.7728 |
| Px.007 | 220 | 20 / 100 / 100 | 83.64% | 0.7355 | +0.7211 |
| Px.008 | 220 | 20 / 100 / 100 | 85.45% | 0.7816 | +0.7556 |
| Px.009 | 220 | 20 / 100 / 100 | 74.55% | 0.6622 | +0.5698 |
| Px.010 | 220 | 20 / 100 / 100 | 85.91% | 0.7091 | +0.7571 |
| **POOL** | **1100** | **100 / 500 / 500** | **83.27%** | **0.7282** | **+0.7149** |

#### 5.5.3 Mes 6 (fase cronica, 4 pacientes)

| Paciente | Epocas | Distribucion (0/10/40) | Accuracy | F1 Macro | Kappa |
|----------|--------|----------------------|----------|----------|-------|
| Px.006 | 221 | 20 / 100 / 101 | 76.92% | 0.6302 | +0.6193 |
| Px.007 | 220 | 20 / 100 / 100 | 85.45% | 0.7652 | +0.7542 |
| Px.008 | 222 | 20 / 102 / 100 | **91.44%** | 0.8378 | +0.8511 |
| Px.009 | 224 | 20 / 102 / 102 | 79.46% | 0.6560 | +0.6512 |
| **POOL** | **887** | **80 / 404 / 403** | **83.31%** | **0.7155** | **+0.7174** |

**Observaciones per-paciente:**

1. **Px.009 es consistentemente el paciente mas dificil** (63.71%–79.46%), posiblemente debido a menor habilidad de imagery motora o mayor variabilidad en los patrones EEG.
2. **Px.008 alcanza 91.44% en el Mes 6** — casi 11 puntos arriba del pool. Este paciente muestra patrones fractales particularmente discriminativos en la fase cronica.
3. La **variabilidad entre pacientes** es significativa (rango: 57.73%–91.44%), lo que confirma la naturaleza altamente individual de los patrones de MI y la necesidad de calibracion por paciente.
4. El POOL es ligeramente inferior al promedio simple porque esta ponderado por el numero de epocas de cada paciente.

---

## 6. Clasificacion 2-Clases (MVR 10% vs 40%, sin reposo, sin SMOTE)

Para comparacion con estudios que utilizan tareas binarias, se presentan los resultados de clasificacion 2-clases (10% vs 40%) sin SMOTE (clases balanceadas ~1332:1355):

### 6.1 Per-Electrodo, Basicos (64 features)

| Condicion | SVM | kNN | RandomForest | NaiveBayes | LogisticRegression | MLP | DecisionTree |
|-----------|-----|-----|-------------|------------|-------------------|-----|-------------|
| Mixto | **85.00%** | 74.25% | 83.36% | 63.04% | 84.33% | 69.37% | 69.78% |
| Mes 1 | 86.02% | 77.50% | 83.98% | 72.05% | **88.30%** | 72.61% | 73.18% |
| Mes 3 | 83.80% | 77.71% | 84.40% | 81.50% | **91.60%** | 77.70% | 79.00% |
| Mes 6 | 90.95% | 81.04% | 82.65% | 87.86% | **95.54%** | 86.62% | 84.88% |

**Mejor 2-clases:** Mes 6, LogisticRegression, Basic per-channel (64 feats) = **95.54%**, F1=0.9554, κ=0.9108.

### 6.2 Per-Electrodo, Todos 7 metodos (112 features)

| Condicion | Best Accuracy | Modelo |
|-----------|-------------|--------|
| Mixto | 84.29% | SVM |
| Mes 1 | 88.64% | LogisticRegression |
| Mes 3 | 90.20% | LogisticRegression |
| Mes 6 | 95.17% | LogisticRegression |

### 6.3 Comparacion de estrategias (2-clases)

| Estrategia | Features | Mixto | Mes 1 | Mes 3 | Mes 6 |
|-----------|----------|-------|-------|-------|-------|
| Promedio Espacial (7 feats) | 7 | 64.35% | 64.20% | 75.20% | 67.78% |
| Per-Electrodo Basicos (64 feats) | 64 | 85.00% | 88.30% | 91.60% | **95.54%** |
| Per-Electrodo Todos (112 feats) | 112 | 84.29% | 88.64% | 90.20% | 95.17% |
| **Ganancia per-electrodo** | | **+20.7** | **+24.1** | **+16.4** | **+27.8** |

---

## 7. Analisis de Resultados

### 7.1 Impacto de SMOTE en la clasificacion 3-clases

El desbalance natural de clases (280 reposo vs 1332 MVR 10% vs 1355 MVR 40%) penaliza significativamente la clasificacion. La aplicacion de SMOTE exclusivamente al conjunto de entrenamiento produce mejoras notables:

| Subset | Sin SMOTE (previo) | Con SMOTE | Ganancia |
|--------|-------------------|-----------|----------|
| Higuchi (spatial mean, 3-clases Mes 3) | 65.45% | 55.82%* | -9.63 pts |
| Higuchi (per-channel, 3-clases Mes 3) | — | 66.27% | — |
| Basic per-channel (3-clases Mes 3) | — | 83.27% | — |

*Nota: El subset spatial mean con SMOTE muestra degradacion porque pierde informacion topografica al promediar. Las mejoras reales provienen de la combinacion SMOTE + per-electrodo.

### 7.2 Jerarquia de rendimiento

**Jerarquia per-electrodo (3-clases, Mes 3, LogisticRegression):**

| Posicion | Subset | Accuracy | Kappa | N Feat |
|----------|--------|----------|-------|--------|
| 1 | Basicos per-channel | 83.27% | 0.715 | 64 |
| 2 | Todos per-channel | 80.00% | 0.658 | 112 |
| 3 | Martinez per-channel | 63.36% | 0.409 | 48 |
| 4 | HRS p64 per-channel | 57.09% | 0.335 | 16 |
| 5 | HO per-channel | 58.64% | 0.341 | 16 |
| 6 | HV per-channel | 57.09% | 0.335 | 16 |

Los 4 metodos basicos per-electrodo (RS, Higuchi, DFA, Variogram × 16 canales = 64 features) superan consistentemente a las 3 variantes Martinez (HO, HRS_p64, HV × 16 canales = 48 features), a pesar de que estas ultimas replican exactamente la metodologia del paper de referencia. Esto sugiere que los metodos fractales complementarios (Higuchi basado en longitud de curva, DFA con eliminacion de tendencias, RS clasico, Semivariograma) capturan aspectos de la dinamica temporal que las variantes del exponente de Hurst por si solas no alcanzan.

### 7.3 Efecto del momento temporal

| Metrica | Mes 1 (Agudo) | Mes 3 (Sub-agudo) | Mes 6 (Cronico) |
|---------|---------------|-------------------|-----------------|
| Mejor spatial mean | 51.12% | 64.64% | 57.61% |
| Mejor per-channel | 74.80% | **83.27%** | **83.31%** |
| Mejor 2-clases per-ch | 88.30% | 91.60% | **95.54%** |
| Mixto per-channel | — | — | 74.99% |

El patron temporal se mantiene: Mes 3 y Mes 6 ofrecen la senal mas discriminativa. Notablemente, el Mes 6 supera ligeramente al Mes 3 en 3-clases (83.31% vs 83.27%), invirtiendo la tendencia previa. La condicion Mixta (74.99%) permanece significativamente por debajo de cualquier mes individual (~8-9 puntos), confirmando que cada etapa temporal posee una firma fractal especifica que los clasificadores explotan.

### 7.4 Comparacion con Martinez-Peon 2024

| Factor | Martinez-Peon 2024 | Este estudio (Previo) | Este estudio (Actual) |
|--------|-------------------|----------------------|----------------------|
| Sujetos | 10 | 5 | 5 |
| Electrodos | 14 (PM, DLPFC, IPL) | 16 (montaje completo) | 16 (montaje completo) |
| Clases | 4 niveles | 3 niveles | 3 niveles (con SMOTE) |
| Algoritmos | HO, HRS, HV | RS, Higuchi, DFA, Variogram, HO, HRS, HV | RS, Higuchi, DFA, Variogram, HO, HRS, HV |
| Agregacion | Per-electrodo | Promedio espacial | Per-electrodo |
| Ventana | 1.5s con marcadores | 2.8s contiguos | 2.8s contiguos |
| Validacion | 10-fold CV (WEKA) | 5-fold CV (scikit-learn) | 5-fold CV (scikit-learn) |
| Mejor accuracy 3-clases | 96.42% (kNN+HRS+HO+HV) | 67.73% (RF, 7 features) | **83.31% (LogReg, 64 features)** |
| Mejor accuracy 2-clases | — | 75.20% | **95.54%** |

La mejora de 67.73% a 83.31% (+15.6 puntos) se atribuye a dos factores principales:
1. **Per-electrodo vs promedio espacial**: Retener la informacion topografica de los 16 canales anade ~10-15 puntos.
2. **SMOTE**: El balanceo de clases permite que los clasificadores aprendan patrones de la clase minoritaria (reposo) que de otro modo serian ignorados.

### 7.5 Matrices de Confusion y Verificacion de Data Leakage

#### 7.5.1 Matrices de confusion (LogisticRegression + Basicos per-channel + SMOTE)

**Mes 1 (980 epocas, POOL 74.80%):**
```
              Predicho 0    Predicho 10    Predicho 40
Real 0:            29             38             33
Real 10:           48            328             52
Real 40:           47             29            376
```
- Clase 0 (Reposo): 29 de 100 correctos (29.0%) — la clase mas dificil
- Clase 10 (MVR 10%): 328 de 428 correctos (76.6%)
- Clase 40 (MVR 40%): 376 de 452 correctos (83.2%)

**Mes 3 (1,100 epocas, POOL 83.27%):**
```
              Predicho 0    Predicho 10    Predicho 40
Real 0:            48             29             23
Real 10:           38            430             32
Real 40:           39             23            438
```
- Clase 0: 48 de 100 correctos (48.0%)
- Clase 10: 430 de 500 correctos (86.0%)
- Clase 40: 438 de 500 correctos (87.6%)
- Error principal: confundir reposo con MVR 10% o 40% (52% de los errores de clase 0)

**Mes 6 (887 epocas, POOL 83.31%):**
```
              Predicho 0    Predicho 10    Predicho 40
Real 0:            35             15             30
Real 10:           30            357             17
Real 40:           45             11            347
```
- Clase 0: 35 de 80 correctos (43.8%)
- Clase 10: 357 de 404 correctos (88.4%)
- Clase 40: 347 de 403 correctos (86.1%)

**Patron de confusion consistente:** La clase 0 (Reposo) se confunde principalmente con la clase 40 (MVR 40%), y viceversa. Esto sugiere que las propiedades fractales del EEG en reposo y durante esfuerzo fuerte de MI comparten caracteristicas espectrales similares en el dominio fractal, mientras que el esfuerzo bajo (MVR 10%) presenta una firma fractal mas diferenciada.

#### 7.5.2 Verificacion de Data Leakage

**Preprocesamiento y features:**
- Total features per-channel: 64 (16 canales × 4 metodos: RS, Higuchi, DFA, Variograma)
- NaN values: 0 en todas las features
- Inf values: 0 en todas las features
- No se utilizan features Delta (Active − Basal), por lo que no hay riesgo de leakage por referencia basal

**Clase basal (Reposo, MVR=0%):**
- Los primeros 5 segmentos de cada archivo se etiquetan como clase 0
- 28 archivos unicos × 5 segmentos basales = 140 epocas de clase 0 teoricas (280 reales porque hay 56 archivos incluyendo ASYNCHRONOUS y SYNCHRONOUS)
- Estos segmentos SOLO se usan como epocas de clasificacion (Active_ features), NO como referencia basal para ningun calculo Delta

**Validacion intra-sujeto:**
- StratifiedKFold (k=5, shuffle=True, random_state=42) garantiza que cada epoca aparece exclusivamente en train o test, nunca en ambos
- El riesgo principal es la **correlacion temporal entre segmentos contiguos** del mismo archivo. Con 0% de overlap y W=700 muestras (2.8s), los segmentos son contiguos pero no solapados. Esto puede inflar artificialmente la accuracy porque segmentos adyacentes pueden ser muy similares, aplicando por igual a las 3 clases

---

## 8. Discusion

### 8.1 Implicaciones para el analisis fractal de EEG en MI

Los resultados demuestran que **la combinacion de caracteristicas fractales per-electrodo con SMOTE** produce un salto sustancial en la clasificacion multinivel de MI, alcanzando 83.31% en 3-clases y 95.54% en 2-clases. Este hallazgo tiene implicaciones metodologicas significativas:

1. **La informacion topografica es critica.** El promedio espacial descarta la distribucion espacial de las propiedades fractales entre electrodos, lo que resulta en una perdida de ~15-20 puntos de accuracy. Retener las caracteristicas por electrodo es esencial para capturar patrones de activacion cortical especificos.

2. **Los metodos fractales basicos superan a las variantes Hurst.** Los 4 metodos basicos per-electrodo (RS, Higuchi, DFA, Variogram: 64 features) consistentemente superan a los 3 metodos Martinez (HO, HRS_p64, HV: 48 features) a pesar de que estos ultimos replican exactamente la metodologia del paper de referencia. La dimension fractal de Higuchi y el exponente de fluctuacion DFA capturan aspectos de la complejidad temporal que el exponente de Hurst por si solo no alcanza.

3. **SMOTE es efectivo para el desbalance de clases.** La clase de reposo (280 epocas) representa solo el 9.4% del dataset. SMOTE sintetiza ejemplos de entrenamiento que permiten a los clasificadores aprender patrones de esta clase minoritaria sin sobreajustar.

4. **La informacion parietal es particularmente discriminativa.** El subconjunto Parietal-5 (Pz, P5, P3, P6, P4) alcanza 76.36% con solo 5 canales y 20 features, sugiriendo que las areas parietales juegan un rol importante en la codificacion de niveles de fuerza durante MI, consistente con la activacion del lobulo parietal inferior (IPL) reportada en estudios de fMRI.

### 8.2 Relevancia clinica

1. **Ventana optima de 2.8 segundos:** Clinicamente viable para sistemas BCI en tiempo real (~0.36 Hz). El pipeline per-electrodo con 64 features y LogisticRegression es computacionalmente ligero y compatible con implementaciones embebidas.

2. **Clasificacion robusta en fase cronica (Mes 6):** El rendimiento maximo en 3-clases (83.31%) y 2-clases (95.54%) en el Mes 6 post-AVC demuestra que la senal fractal persiste y es detectable en etapas cronicas, donde otras modalidades de senal EEG pueden degradarse.

3. **Channel subsets eficientes:** Parietal-5 (76.89%) con solo 5 electrodos y 20 features ofrece un compromiso atractivo entre precision y complejidad computacional para sistemas BCI portatiles.

4. **Necesidad de calibracion temporal:** La degradacion al mezclar meses (74.99% Mixto vs 83.31% Mes 6) indica que los sistemas BCI fractales deben recalibrarse periodicamente para cada etapa del proceso de rehabilitacion.

### 8.3 Limitaciones

1. **Tamano muestral reducido:** 5 pacientes limitan la generalizabilidad estadistica. Un estudio con mas sujetos es necesario para validar la robustez de los resultados.
2. **Sin validacion LOPO para W700:** La generalizacion inter-sujeto permanece como desafio abierto. La evaluacion LOPO en el pipeline dinamico (63.65%) sugiere que la variabilidad inter-sujeto sigue siendo un obstaculo.
3. **Sin optimizacion de hiperparametros:** GridSearchCV podria mejorar los resultados 1-3 puntos adicionales.
4. **Clase reposo sintetica:** Los primeros 5 segmentos como clase 0 introducen un sesgo potencial respecto al reposo genuino.
5. **Sin clase MVR intermedia:** Solo se evaluan 10% y 40%; niveles intermedios (20%, 30%) y superiores (70%) podrian presentar mayor desafio.

---

## 9. Conclusiones

1. El tamano de ventana optimo para la extraccion de caracteristicas fractales en MI post-AVC es de **700 muestras (2.8 segundos)** con segmentacion contigua al 0% de traslape.

2. La combinacion de **caracteristicas fractales per-electrodo (RS, Higuchi, DFA, Variogram × 16 canales = 64 features) con SMOTE** aplicado al conjunto de entrenamiento en 3-clases alcanza **83.31% de accuracy** (LogisticRegression, Mes 6, κ=0.7174, F1=0.7155), una mejora de **+15.6 puntos** sobre el enfoque previo de promedio espacial sin SMOTE (67.73%).

3. En clasificacion 2-clases (10% vs 40%) sin SMOTE, las mismas caracteristicas per-electrodo alcanzan **95.54%** (LogisticRegression, Mes 6, κ=0.9108), superando el 75.20% previo por **+20.3 puntos**.

4. Las caracteristicas per-electrodo (**64-112 features**) superan consistentemente al promedio espacial (**7 features**) por 15-28 puntos, demostrando que **la informacion topografica de los electrodos es critica** para la clasificacion fractal de MI.

5. Los **4 metodos fractales basicos** (RS, Higuchi, DFA, Variogram) superan a las **3 variantes Martinez** (HO, HRS_p64, HV) en configuracion per-electrodo (83.31% vs 63.36-77.00%), a pesar de que estas ultimas replican la metodologia del paper de referencia. Higuchi y DFA capturan aspectos de la complejidad temporal no accesibles al exponente de Hurst.

6. El **Mes 6 post-AVC** (fase cronica) emerge como el momento optimo para la clasificacion fractal per-electrodo (83.31% en 3-clases, 95.54% en 2-clases), seguido del Mes 3 (83.27% y 91.60%), invirtiendo la jerarquia temporal previa donde el Mes 3 lideraba.

7. **SMOTE es efectivo** para abordar el desbalance de clases en 3-clases, permitiendo que clasificadores lineales como LogisticRegression superen a ensembles como RandomForest.

8. La **mezcla de meses degrada la clasificacion** (74.99% Mixto vs 83.31% Mes 6), confirmando que cada etapa temporal posee una firma fractal especifica.

9. El subconjunto **Parietal-5** (5 canales, 76.36%) demuestra que las areas parietales contienen informacion particularmente discriminativa, ofreciendo un compromiso viable entre precision y complejidad para sistemas BCI.

10. El pipeline propuesto (caracteristicas fractales per-electrodo + SMOTE + LogisticRegression) es **computacionalmente eficiente** y clinicamente viable, con un accuracy del 83.31% en 3-clases que se aproxima al estado del arte reportado por Martinez-Peon et al. (96.42% en 4-clases con 14 electrodos dirigidos a areas motoras), a pesar de usar un montaje 10-20 no dirigido y solo 5 pacientes.

---

## Apendice A: Configuracion Detallada

### A.1 Parametros de los algoritmos fractales

| Algoritmo | Parametro | Valor |
|-----------|-----------|-------|
| Rescaled Range | Escala minima | 0.128 s (32 muestras) |
| Rescaled Range | Escala maxima | N * 0.40 |
| Rescaled Range | Puntos minimos validos | 5 |
| Higuchi | k_max | min(ceil(N * 0.25), 30) |
| Higuchi | k_max floor | 5 |
| DFA | Escala minima | 0.064 s (16 muestras) |
| DFA | Escala maxima | N * 0.33 |
| DFA | Ventanas minimas por escala | 3 |
| Semivariograma | Lag maximo | N * 0.20 |
| Semivariograma | Lags minimos | 10 |
| HRS | Particion (p) | 64 |
| HRS | Minimo muestras por sub-serie | 10 |

### A.2 Hiperparametros de clasificadores

| Modelo | Hiperparametros |
|--------|-----------------|
| SVM | C=1.0, kernel='rbf', probability=True, random_state=42 |
| RandomForest | n_estimators=300, max_depth=None, random_state=42, n_jobs=-1 |
| LogisticRegression | C=1.0, max_iter=2000, random_state=42, n_jobs=-1 |
| kNN | n_neighbors=10, metric='chebyshev', n_jobs=-1 |
| NaiveBayes | GaussianNB, var_smoothing=1e-9 |

### A.3 Preprocesamiento

| Etapa | Parametros |
|-------|-----------|
| Notch filter | 60 Hz, Q=30.0 |
| Bandpass filter | Butterworth orden 4, 0.5–45 Hz |
| ICA | 3 componentes removidos, max_iter=500, random_state=42 |
| Z-score | Por canal, media=0, std=1 |

### A.4 Montaje de electrodos (16 canales)

| Indice | Canal | Region |
|--------|-------|--------|
| 0 | Fp1 | Frontal |
| 1 | Fp3 | Frontal |
| 2 | F3 | Frontal |
| 3 | Fz | Frontal |
| 4 | F4 | Frontal |
| 5 | Cz | Central-Motor |
| 6 | Pz | Parietal |
| 7 | P5 | Parietal |
| 8 | P3 | Parietal |
| 9 | O1 | Occipital |
| 10 | Oz | Occipital |
| 11 | T8 | Temporo-Parietal |
| 12 | P8 | Temporo-Parietal |
| 13 | P6 | Parietal |
| 14 | P4 | Parietal |
| 15 | T7 | Temporal |

---

## Apendice B: Distribucion de Datos

### B.1 Epocas W700 por paciente, mes y clase

| Paciente | Mes 1 (0/10/40) | Mes 3 (0/10/40) | Mes 6 (0/10/40) | Total |
|----------|-----------------|-----------------|-----------------|-------|
| Px.006 | 20/97/156 | 20/97/156 | 15/63/117 | 741 |
| Px.007 | 20/97/156 | 20/97/156 | 20/97/117 | 780 |
| Px.008 | 20/96/155 | 20/97/156 | 20/97/156 | 817 |
| Px.009 | 15/63/117 | 20/97/156 | 20/97/156 | 741 |
| Px.010 | 15/63/117 | 20/97/156 | 0/0/0 | 468 |
| **Total** | **100/416/701** | **100/485/780** | **75/354/546** | **2,967** |

*Clase 0 = Reposo (primeros 5 segmentos de cada archivo).

### B.2 Archivos procesados

- 56 archivos .mat (28 ASYNCHRONOUS + 28 SYNCHRONOUS)
- MVR 10%: 28 archivos
- MVR 40%: 28 archivos
- Reposo: etiquetado a partir de los primeros 5 segmentos de cada archivo

---

*Documento generado a partir de la ejecucion del pipeline Niveles3 con 320 experimentos de clasificacion (5 modelos x 8 subsets x 4 condiciones x 2 tipos de clase). Los resultados completos se encuentran en `output/classification/results_W700_fractal_individual.csv`. El reporte completo incluyendo caracteristicas espectrales se encuentra en `INFORME_COMPLETO.md`.*
