# Clasificacion de Niveles de Fuerza en Imagery Motor post-AVC mediante Caracteristicas Fractales y Espectrales con Ventanas Fijas

**Pipeline:** Niveles3
**Fecha:** Junio 2026 (recalculado Agosto 2026 — ver nota metodologica)
**Validacion:** Intra-sujeto 5-fold Cross-Validation (80/20)
**Ventana optima:** W700 = 2.8 segundos (0% overlap, segmentos contiguos)

> **Nota metodologica (Agosto 2026):** Las cifras de este documento fueron recalculadas respecto a la version original de Junio 2026 por tres motivos, todos verificados durante una auditoria de codigo: (1) se corrigio `compute_hrs()` (`src/features/fractal.py`), que con `p=64` caia siempre en un fallback a Hurst Original sin computar particiones reales para las duraciones de ventana usadas en el estudio — verificado numericamente que `HRS_p64 == HO` bit a bit para N entre 500 y 2000 muestras; `BEST_HRS_PARTITION` se cambio a `p=8`, el mayor valor que empiricamente si ejecuta el pooling particionado real en W700; (2) el archivo de features que efectivamente alimentaba la clasificacion (`Fixed_W700_Full_Features.csv`) era una copia manual desactualizada — el extractor actual escribe `Fixed_W700_3class_Features.csv`, y ese mismatch de nombres impedia que una re-ejecucion actualizara los resultados publicados; (3) `requirements.txt` no fijaba versiones de numpy/scipy/scikit-learn — dado que el preprocesamiento usa ICA (`FastICA`, iterativo/aleatorio), la re-ejecucion en un entorno con versiones mas nuevas produce valores de features ligeramente distintos aun con `random_state` fijo. Las versiones exactas quedaron fijadas en `requirements.txt` para que el pipeline sea reproducible de aqui en adelante. La optimizacion de rendimiento del pipeline (vectorizacion, paralelismo) fue verificada como no-op numerico mediante tests de regresion (`tests/test_fractal.py`, `tests/test_spectral.py`) y no contribuye a ninguno de los cambios de cifras aqui documentados.

---

## Resumen

Se evaluo un pipeline de clasificacion de niveles de fuerza en Motor Imagery (MI) post-Accidente Cerebrovascular (AVC) combinando caracteristicas fractales (Rescaled Range, Higuchi, Detrended Fluctuation Analysis, Semivariograma, Promedio, Hurst Original, Hurst R/S con Particiones, Hurst via Semivariograma) y caracteristicas espectrales (potencias de banda δ/θ/α/β/γ, ratios de Desincronizacion Relacionada a Eventos, entropia espectral, ratio α/β, ratio μ/β). Sobre 5 pacientes (P1-P5) con registros EEG de 16 canales a 250 Hz en tres momentos temporales (Mes 1, Mes 3, Mes 6), se optimizo el tamano de ventana mediante un barrido empirico de 16 tamanos (2.0s–8.0s) determinando W=700 muestras (2.8 s) como la configuracion optima. Empleando validacion intra-sujeto con StratifiedKFold (k=5) y pool de predicciones, la combinacion Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales alcanzo **93.67% de accuracy** (F1=0.9367) en el Mes 1 con Regresion Logistica, superando el objetivo del 80%; el mejor resultado global del estudio corresponde al Mes 3 con **94.55%** usando solo las 13 caracteristicas espectrales. Las caracteristicas espectrales demostraron ser el factor dominante (88–95% en solitario), mientras que las caracteristicas fractales puras (8 metodos, con la correccion de `HRS_p64`) alcanzaron hasta 82% sin asistencia espectral, validando su capacidad discriminativa independiente del nivel de fuerza imaginado.

---

## 1. Introduccion y Motivacion

El Accidente Cerebrovascular (AVC) es una de las principales causas de discapacidad motora a nivel mundial. Las Interfaces Cerebro-Computadora (BCI) basadas en Motor Imagery (MI) —la imaginacion de un movimiento sin ejecutarlo— ofrecen una via prometedora para la neurorehabilitacion al activar circuitos sensoriomotores incluso en ausencia de movimiento real. Un desafio clinico fundamental es la capacidad de **discriminar entre diferentes niveles de fuerza imaginada**, tipicamente expresados como porcentaje de la Contraccion Voluntaria Maxima (Maximum Voluntary Contraction, MVR). Una clasificacion precisa del nivel de esfuerzo mental permitiria sistemas BCI mas graduados y adaptativos durante la rehabilitacion.

Investigaciones previas han empleado diversas tecnicas de procesamiento de senales EEG para este proposito. En particular, Martinez-Peon et al. (2024, _J. Neural Eng._ 21, 046024) reportaron un 96.42% de accuracy empleando el exponente de Hurst con particiones (HRS) y clasificadores estandar (kNN, SVM, RandomForest, entre otros) en 10 sujetos con 20 electrodos bajo validacion intra-sujeto con 5-fold cross-validation en WEKA (per-sujeto promediado, no pool).

El presente estudio adapta y extiende dicho enfoque con las siguientes contribuciones:

1. **Evaluacion de multiples variantes del exponente de Hurst**: Rescaled Range (R/S), Higuchi, DFA, Semivariograma, Hurst Original (HO), Hurst R/S con Particiones (HRS) y Hurst via Semivariograma (HV).
2. **Caracteristicas espectrales complementarias**: potencias de banda, ratios de Desincronizacion Relacionada a Eventos (ERD) y entropia espectral.
3. **Ventanas fijas contiguas**: segmentacion con 0% de traslape como alternativa a epocas basadas en deteccion de ERD.
4. **Analisis longitudinal por mes**: clasificacion independiente en Mes 1 (fase aguda), Mes 3 (fase sub-aguda) y Mes 6 (fase cronica) para evaluar la evolucion temporal de la senal discriminativa.
5. **Barrido empirico de tamano de ventana**: desde 2.0 s hasta 8.0 s (500–2000 muestras) para identificar la configuracion optima.

---

## 2. Arquitectura del Pipeline

### 2.1 Datos

| Propiedad | Valor |
|-----------|-------|
| Pacientes | 5 (P1–P5) |

> **Nota de anonimizacion:** Los identificadores originales Px.006–Px.010 se renombraron como P1–P5 respectivamente.
| Meses | 1, 3, 6 post-AVC |
| Archivos .mat | 56 (28 sincronicos + 28 asincronicos) |
| Canales EEG | 16 (montaje internacional 10-20) |
| Frecuencia de muestreo | 250 Hz |
| Clases | MVR 10%, MVR 40% |
| Epocas (W700) | 2,967 totales |

**Montaje de electrodos 16 canales:**

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

### 2.2 Preprocesamiento

```
Senal cruda (16ch x N muestras)
    -> Remocion de DC (sustraccion de media)
    -> Filtro Notch 60 Hz (Q=30.0)
    -> Filtro Butterworth pasabanda 0.5–45 Hz (orden 4)
    -> ICA (3 componentes removidos)
    -> Normalizacion Z-score por canal
    -> Senal preprocesada lista para segmentacion
```

### 2.3 Segmentacion por Ventanas Fijas

Todas las epocas se extraen del registro completo mediante **ventanas contiguas de tamano fijo W=700 muestras (2.8 segundos) con 0% de traslape**:

- Segmento 0: muestras [0, 700)
- Segmento 1: muestras [700, 1400)
- Segmento 2: muestras [1400, 2100)
- ...
- El sobrante al final del registro se descarta.

Cada segmento hereda la etiqueta MVR del archivo fuente (10% o 40%). Los primeros 5 segmentos de cada archivo se utilizan como referencia **Basal** para el calculo de las diferencias Δ.

### 2.4 Extraccion de Caracteristicas

#### 2.4.1 Caracteristicas Fractales (por canal, 16 canales x 8 metodos = 128 features por tipo)

Cada segmento de EEG (16 canales x 700 muestras) se procesa con los siguientes metodos fractales de forma independiente por canal:

- **Rescaled Range (R/S):** Exponente de Hurst via analisis de rango reescalado. Parametros adaptativos: escala minima 0.128 s, escala maxima N*0.40, minimo 5 puntos para ajuste log-log.
- **Higuchi:** Dimension fractal via el metodo de Higuchi (1988). k_max = min(N*0.25, 30), minimo 5 puntos validos.
- **Detrended Fluctuation Analysis (DFA):** Exponente de fluctuacion destrendizado (Peng et al., 1995). Escala minima 0.064 s, escala maxima N*0.33, minimo 3 ventanas por escala.
- **Semivariograma:** Exponente de Hurst calculado a partir de la pendiente log-log de la semivarianza vs lag. Lag maximo N*0.20, minimo 10 lags.
- **Promedio:** Promedio aritmetico de los 4 metodos anteriores (R/S, Higuchi, DFA, Semivariograma).
- **Hurst Original (HO):** Exponente de Hurst via R/S clasico sin detrending.
- **Hurst Rescaled Range con Particiones (HRS):** R/S aplicado a cada una de las p=64 sub-series en que se divide la senal, promediando los resultados. Cuando la sub-serie tiene menos de 10 muestras, se utiliza HO como fallback.
- **Hurst via Semivariograma (HV):** Exponente de Hurst calculado a partir de la pendiente log-log de la semivarianza vs lag (identico al Semivariograma pero etiquetado segun convencion de la literatura).

Estos 8 metodos producen una matriz de **16 canales x 8 metodos = 128 valores fractales** por segmento de EEG.

#### 2.4.2 Caracteristicas Espectrales (spatial mean, 13 features)

Para cada segmento se calcula el promedio espacial de los 16 canales y se extraen:

- **Potencias de banda (5 features):** δ (0.5–4 Hz), θ (4–8 Hz), α (8–12 Hz), β (12–30 Hz), γ (30–45 Hz). Normalizadas por varianza total del segmento.
- **Ratios ERD (5 features):** Cociente entre la potencia de banda activa y la potencia de banda basal para cada banda. ERD_ratio = Active_power / Basal_power.
- **Entropia Espectral:** Entropia de Shannon de la densidad espectral de potencia normalizada.
- **Ratio α/β:** Cociente entre potencia alpha y beta.
- **Ratio μ/β:** Cociente entre potencia mu y beta (identico a α/β pues la banda mu se define como 8–12 Hz).

Las caracteristicas espectrales totalizan **13 valores** por segmento.

#### 2.4.3 Calculo de Δ (Delta = Active – Basal)

Para cada metodo fractal, se calcula la **diferencia** entre el valor del segmento activo y el valor basal de referencia:

```
Δ-Rescaled Range = Active_Rescaled Range − Basal_Rescaled Range
Δ-Higuchi = Active_Higuchi − Basal_Higuchi
Δ-DFA = Active_DFA − Basal_DFA
Δ-Semivariograma = Active_Semivariograma − Basal_Semivariograma
Δ-Promedio = Active_Promedio − Basal_Promedio
Δ-Hurst Original = Active_Hurst Original − Basal_Hurst Original
Δ-Hurst R/S con Particiones = Active_Hurst R/S con Particiones − Basal_Hurst R/S con Particiones
Δ-Hurst Semivariograma = Active_Hurst Semivariograma − Basal_Hurst Semivariograma
```

El valor Basal se obtiene promediando las caracteristicas fractales de los primeros 5 segmentos del archivo. Las caracteristicas espectrales se utilizan en su forma absoluta (no Δ) dado que los ratios ERD ya incorporan implicitamente la comparacion con el basal.

**Total de columnas en el CSV:** 412 columnas (8 metadatos + 3 x 8 fractales globales + 3 x 8 Martinez + 3 x 7 Martinez-all-p + 25 estabilidad + 3 x 16 x 7 per-channel fractales + 13 espectrales).

### 2.5 Clasificacion

**Validacion Intra-sujeto 5-fold CV (80/20):**

Para cada paciente, se aplica StratifiedKFold con k=5 (o menos si las clases disponibles lo limitan). En cada fold, el 80% de las epocas del paciente se usan para entrenamiento y el 20% para prueba. Las predicciones de todos los folds de todos los pacientes se agrupan (pool) y se calculan las metricas globales (accuracy, F1 macro) sobre el conjunto completo de predicciones.

**Preprocesamiento intra-fold:** StandardScaler (z-score) ajustado sobre el conjunto de entrenamiento y aplicado al conjunto de prueba.

**Condiciones evaluadas:**

| Condicion | Epocas W700 | Descripcion |
|-----------|-------------|-------------|
| Mixto (Mixed) | 2,967 | Todos los meses combinados |
| Mes 1 | 980 | Solo Month1 |
| Mes 3 | 1,100 | Solo Month3 |
| Mes 6 | 887 | Solo Month6 (4 pacientes, P5 sin datos) |

**Modelos evaluados (5):**

| Modelo | Hiperparametros |
|--------|-----------------|
| SVM | kernel=rbf, C=1.0, probability=True |
| RandomForest | n_estimators=300, max_depth=None, n_jobs=-1 |
| LogisticRegression | max_iter=2000, n_jobs=-1 |
| kNN | n_neighbors=10, metric=chebyshev, n_jobs=-1 |
| GaussianNB | var_smoothing=1e-9 |

**Subsets de caracteristicas evaluados (5):**

| Nombre | Contenido | N |
|--------|-----------|----|
| Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio | Δ de 5 metodos fractales basicos | 5 |
| Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Δ de 3 variantes del exponente de Hurst | 3 |
| Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | 5+3=8 fractales combinados | 8 |
| Potencia de banda δ / Potencia de banda θ / Potencia de banda α / Potencia de banda β / Potencia de banda γ / Ratio ERD δ / Ratio ERD θ / Ratio ERD α / Ratio ERD β / Ratio ERD γ / Entropia Espectral / Ratio α/β / Ratio μ/β | 13 caracteristicas espectrales | 13 |
| Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales | 5 fractales + 13 espectrales | 18 |

---

## 3. Glosario

| Termino | Definicion |
|---------|-----------|
| **Δ (Delta)** | Operador de diferencia: Δ = Active − Basal. Representa el cambio en una caracteristica entre el estado de Motor Imagery (Active) y el estado de reposo (Basal). **No debe confundirse con la banda de frecuencia δ (delta, 0.5–4 Hz).** |
| **δ (delta minuscula)** | Banda de frecuencia delta: 0.5–4 Hz. Una de las 5 bandas espectrales analizadas. |
| **Basal** | Estado de reposo utilizado como linea base de referencia. En el pipeline de ventanas fijas: promedio de las caracteristicas de los primeros 5 segmentos del registro EEG. |
| **Activo (Active)** | Estado durante la tarea de Motor Imagery. En ventanas fijas: cada segmento contiguo hereda la etiqueta MVR del archivo (10% o 40%). |
| **Rescaled Range (R/S)** | Metodo de analisis fractal que calcula el exponente de Hurst a partir de la pendiente del logaritmo del rango reescalado (R/S) contra el logaritmo de la escala temporal. Valores H > 0.5 indican persistencia (correlaciones de largo alcance); H < 0.5 indican anti-persistencia. |
| **Higuchi** | Metodo de calculo de la dimension fractal (Higuchi, 1988). Subdivide la senal en k sub-series y mide la longitud promedio de la curva normalizada. La pendiente de log(L(k)) vs log(1/k) proporciona la dimension fractal. |
| **Detrended Fluctuation Analysis (DFA)** | Metodo de analisis de fluctuaciones destrendizadas (Peng et al., 1995). Elimina tendencias locales de orden polinomial y mide la relacion entre la fluctuacion promedio F(n) y el tamano de ventana n. La pendiente log-log es el exponente de escala α. |
| **Semivariograma** | Metodo que calcula el exponente de Hurst a partir de la semivarianza γ(h) = 0.5 * E[(X(t+h) − X(t))^2]. La pendiente de log(γ(h)) vs log(h) proporciona 2H. |
| **Promedio** | Valor promedio de los 4 metodos fractales basicos (R/S, Higuchi, DFA, Semivariograma). Proporciona una medida consensuada del exponente fractal. |
| **Hurst Original (HO)** | Exponente de Hurst calculado mediante R/S clasico sin eliminacion de tendencia. Metodo original de Hurst (1951) para analisis hidrologico adaptado a series temporales. |
| **Hurst Rescaled Range con Particiones (HRS)** | Extension del metodo HO donde la senal se divide en p sub-series. Se calcula R/S para cada sub-serie y se promedian los resultados. El parametro p=64 es el que reporta mejor discriminacion en la literatura. Cuando una sub-serie tiene menos de 10 muestras, se utiliza HO como valor de fallback, lo que produce valores identicos a HO para la mayoria de epocas cortas. |
| **Hurst via Semivariograma (HV)** | Exponente de Hurst calculado mediante analisis de semivarianza. Equivalente metodologico al Semivariograma clasico pero etiquetado segun la convencion de la literatura de EEG fractal. |
| **ERD (Event-Related Desynchronization)** | Disminucion de la potencia espectral en las bandas mu (8–12 Hz) y/o beta (12–30 Hz) durante la ejecucion o imaginacion de un movimiento, relativa a un periodo basal de reposo. Es el correlato electrofisiologico estandar de la activacion de la corteza sensoriomotora. |
| **Potencia de banda (Band Power)** | Energia espectral integrada en un rango de frecuencia definido. Se calcula mediante el periodograma de Welch (nperseg=250, noverlap=125) y se normaliza por la varianza total del segmento para eliminar diferencias de amplitud globales. |
| **Ratio ERD** | Cociente entre la potencia de banda del segmento activo y la potencia de banda del segmento basal. Valores < 1.0 indican desincronizacion (ERD); valores > 1.0 indican sincronizacion (ERS). |
| **Entropia Espectral** | Entropia de Shannon aplicada a la distribucion de densidad espectral de potencia normalizada. Mide la uniformidad de la distribucion de energia a traves de las frecuencias. Valores altos indican un espectro mas plano (ruido blanco); valores bajos indican concentracion en pocas frecuencias. |
| **Ratio α/β** | Cociente entre la potencia de la banda alpha (8–12 Hz) y la potencia de la banda beta (12–30 Hz). Indicador del estado de activacion cortical: valores bajos reflejan mayor activacion. |
| **Ratio μ/β** | Cociente entre la potencia de la banda mu (8–12 Hz) y la potencia de la banda beta (12–30 Hz). Identico al ratio α/β dado que la banda mu se define en el mismo rango (8–12 Hz) pero se asocia especificamente a la corteza sensoriomotora. |
| **Intra-sujeto (Within-subject)** | Esquema de validacion donde las epocas de entrenamiento y prueba provienen del mismo paciente. En este estudio: StratifiedKFold (k=5) aplicado a las epocas de cada paciente por separado, con las predicciones de todos los folds y pacientes agrupadas (pooled) para el calculo de metricas. Equivalente a un 80/20 split estratificado por clase aplicado dentro de cada sujeto. |
| **LOPO (Leave-One-Patient-Out)** | Esquema de validacion donde se entrena con N−1 pacientes y se prueba con el paciente restante, iterando sobre todos los pacientes. Evalua la capacidad de **generalizacion inter-sujeto** del modelo. |
| **MVR (Maximum Voluntary Contraction)** | Porcentaje de la contraccion voluntaria maxima. En este estudio se evaluan dos niveles: 10% (bajo esfuerzo) y 40% (esfuerzo moderado). |
| **MI (Motor Imagery)** | Tarea cognitiva de imaginar un movimiento sin ejecutarlo fisicamente. En este estudio, los pacientes imaginan contracciones del brazo afectado al 10% o 40% de su MVR. |
| **W700** | Ventana de extraccion de 700 muestras a 250 Hz, equivalente a 2.8 segundos. Identificada como la configuracion optima mediante barrido empirico de 16 tamanos de ventana. |
| **Epoca (Epoch)** | Segmento individual de EEG utilizado como unidad de analisis y clasificacion. En el pipeline W700, cada epoca corresponde a un segmento contiguo de 700 muestras (2.8 s). |
| **ICA (Independent Component Analysis)** | Tecnica de separacion ciega de fuentes aplicada al EEG para identificar y remover componentes asociados a artefactos (parpadeo, actividad muscular, ECG). Se remueven las 3 componentes de mayor varianza. |
| **Z-score** | Normalizacion de cada canal del EEG mediante sustraccion de la media y division por la desviacion estandar del propio canal. Produce una senal con media 0 y varianza 1. |
| **StratifiedKFold** | Metodo de validacion cruzada que preserva la proporcion de clases en cada fold. En este estudio se utiliza con k=5 (o menos si el tamano de la clase minoritaria lo limita), aplicado de forma independiente a las epocas de cada paciente. |
| **Pool de predicciones** | Metodo de agregacion donde las predicciones individuales de todos los folds de todos los pacientes se concatenan en un unico vector, y las metricas de rendimiento (accuracy, F1 macro) se calculan sobre este vector consolidado. Diferente a promediar metricas por fold. |

---

## 4. Optimizacion de Ventana

Se realizo un barrido empirico de 16 tamanos de ventana (500 a 2000 muestras, paso de 100 muestras) con segmentacion contigua al 0% de traslape. Para este barrido rapido se utilizaron unicamente las 5 caracteristicas Δ-fractales basicas (Δ-Rescaled Range, Δ-Higuchi, Δ-DFA, Δ-Semivariograma, Δ-Promedio) con promedio espacial (no per-channel), y RandomForest (n_estimators=300). Clasificacion intra-sujeto 2-clases (MVR 10% vs 40%).

### Resultados del barrido de ventana (Accuracy %)

| Ventana | Muestras | Duracion | Mixto | Mes 1 | Mes 3 | Mes 6 | Promedio* |
|---------|----------|----------|-------|-------|-------|-------|-----------|
| W500 | 500 | 2.0s | 71.75 | 72.28 | 80.58 | 74.78 | 74.87 |
| W600 | 600 | 2.4s | 70.40 | 73.60 | 80.46 | 76.18 | 74.82 |
| **W700** | **700** | **2.8s** | **71.12** | **74.08** | **80.18** | **77.34** | **75.13** |
| W800 | 800 | 3.2s | 70.66 | 71.23 | 80.00 | 74.58 | 73.96 |
| W900 | 900 | 3.6s | 70.77 | 71.60 | 81.05 | 70.68 | 74.47 |
| W1000 | 1000 | 4.0s | 70.50 | 70.32 | 77.31 | 72.12 | 72.73 |
| W1100 | 1100 | 4.4s | 68.98 | 68.64 | 77.86 | 72.91 | 71.83 |
| W1200 | 1200 | 4.8s | 70.63 | 70.88 | 78.91 | 71.93 | 73.47 |
| W1300 | 1300 | 5.2s | 67.97 | 71.72 | 73.50 | 70.63 | 71.06 |
| W1400 | 1400 | 5.6s | 65.04 | 68.54 | 75.19 | 68.34 | 69.59 |
| W1500 | 1500 | 6.0s | 66.45 | 72.11 | 73.85 | 68.51 | 70.80 |
| W1600 | 1600 | 6.4s | 65.79 | 67.53 | 75.00 | 66.41 | 69.44 |
| W1700 | 1700 | 6.8s | 66.69 | 69.14 | 77.58 | 63.29 | 71.14 |
| W1800 | 1800 | 7.2s | 66.70 | 65.07 | 77.62 | 62.61 | 69.80 |
| W1900 | 1900 | 7.6s | 67.07 | 67.04 | 79.75 | 64.38 | 71.29 |
| W2000 | 2000 | 8.0s | 66.60 | 69.41 | 74.74 | 67.11 | 70.25 |

*Promedio = media de Mixto + Mes 1 + Mes 3.

### Conclusion del barrido

**W700 (2.8 segundos) es la ventana optima** con un accuracy promedio de 75.13% en las tres condiciones principales. Las ventanas en el rango 2.0–3.6 segundos (W500–W900) consistentemente superan a las ventanas mas largas (>5 segundos). La degradacion del rendimiento con ventanas grandes sugiere que la senal discriminativa del Motor Imagery se concentra en intervalos relativamente cortos tras el inicio de la tarea, y que ventanas excesivamente largas diluyen esta senal con periodos de menor activacion.

El Mes 3 muestra consistentemente el mejor rendimiento (79–81%), seguido del Mes 6 (63–77%) y el Mes 1 (65–74%). La condicion Mixta (todos los meses combinados) se situa sistematicamente por debajo del mejor mes individual, indicando que **la mezcla de datos de diferentes puntos temporales introduce variabilidad que el clasificador no logra resolver completamente**.

---

## 5. Resultados de Clasificacion por Subset y Modelo

A continuacion se presentan las tablas completas de accuracy (%) para cada combinacion de clasificador, subset de caracteristicas y condicion temporal. Los resultados corresponden a clasificacion 2-clases (MVR 10% vs MVR 40%) con validacion intra-sujeto 5-fold CV y pool de predicciones. Se utilizaron los 16 canales (All-16) con caracteristicas per-channel para los metodos fractales y promedio espacial para los metodos espectrales.

### 5.1 Condicion Mixta (todos los meses combinados, 2,967 epocas)

| Clasificador | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio | Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Potencia de banda δ / Potencia de banda θ / Potencia de banda α / Potencia de banda β / Potencia de banda γ / Ratio ERD δ / Ratio ERD θ / Ratio ERD α / Ratio ERD β / Ratio ERD γ / Entropia Espectral / Ratio α/β / Ratio μ/β | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |
|-------------|-----------|-----------|-----------|-----------|-----------|
| SVM | 65.82 | 68.28 | 73.21 | 76.24 | 81.77 |
| RandomForest | 67.04 | 70.07 | 73.88 | 87.63 | **89.01** |
| LogisticRegression | 60.73 | 62.05 | 65.59 | 66.67 | 72.26 |
| kNN | 64.64 | 70.14 | 69.67 | 75.13 | 77.65 |
| GaussianNB | 58.00 | 57.20 | 58.81 | 61.04 | 62.29 |

### 5.2 Mes 1 (fase aguda, 980 epocas)

| Clasificador | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio | Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Potencia de banda δ / Potencia de banda θ / Potencia de banda α / Potencia de banda β / Potencia de banda γ / Ratio ERD δ / Ratio ERD θ / Ratio ERD α / Ratio ERD β / Ratio ERD γ / Entropia Espectral / Ratio α/β / Ratio μ/β | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |
|-------------|-----------|-----------|-----------|-----------|-----------|
| SVM | 74.59 | 78.16 | 80.20 | 90.41 | 92.14 |
| RandomForest | 74.69 | 76.94 | 78.67 | 89.80 | 91.53 |
| LogisticRegression | 67.55 | 74.29 | 77.96 | 93.57 | **93.67** |
| kNN | 72.76 | 76.33 | 76.94 | 87.35 | 85.61 |
| GaussianNB | 62.86 | 64.29 | 63.47 | 78.47 | 77.96 |

### 5.3 Mes 3 (fase sub-aguda, 1,100 epocas)

| Clasificador | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio | Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Potencia de banda δ / Potencia de banda θ / Potencia de banda α / Potencia de banda β / Potencia de banda γ / Ratio ERD δ / Ratio ERD θ / Ratio ERD α / Ratio ERD β / Ratio ERD γ / Entropia Espectral / Ratio α/β / Ratio μ/β | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |
|-------------|-----------|-----------|-----------|-----------|-----------|
| SVM | 70.82 | 78.55 | 80.55 | 90.55 | 91.27 |
| RandomForest | 68.55 | 78.91 | 79.27 | 92.00 | 91.82 |
| LogisticRegression | 65.36 | 74.09 | 75.73 | **94.55** | 94.27 |
| kNN | 69.55 | 76.36 | 76.55 | 85.27 | 84.64 |
| GaussianNB | 61.09 | 65.82 | 64.45 | 76.91 | 76.00 |

### 5.4 Mes 6 (fase cronica, 887 epocas, 4 pacientes)

| Clasificador | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio | Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Potencia de banda δ / Potencia de banda θ / Potencia de banda α / Potencia de banda β / Potencia de banda γ / Ratio ERD δ / Ratio ERD θ / Ratio ERD α / Ratio ERD β / Ratio ERD γ / Entropia Espectral / Ratio α/β / Ratio μ/β | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |
|-------------|-----------|-----------|-----------|-----------|-----------|
| SVM | 73.96 | 81.06 | 82.30 | 89.18 | 90.53 |
| RandomForest | 72.38 | 78.13 | 81.29 | 91.88 | 92.67 |
| LogisticRegression | 70.69 | 79.59 | 80.38 | 93.35 | **93.69** |
| kNN | 69.56 | 79.14 | 79.26 | 85.34 | 85.79 |
| GaussianNB | 65.61 | 70.12 | 67.53 | 80.50 | 81.74 |

### 5.5 Mejor combinacion por condicion (resumen)

| Condicion | Mejor Accuracy | Mejor F1 Macro | Modelo | Subset de Caracteristicas |
|-----------|---------------|----------------|--------|---------------------------|
| Mixto | 89.01% | 0.8901 | RandomForest | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |
| Mes 1 | 93.67% | 0.9367 | LogisticRegression | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |
| Mes 3 | **94.55%** | **0.9454** | LogisticRegression | 13 Caracteristicas Espectrales (solas; la combinacion con fractales da 94.27%) |
| Mes 6 | 93.69% | 0.9368 | LogisticRegression | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |

**Mejor resultado del estudio:** Mes 3, solo caracteristicas espectrales, LogisticRegression, 94.55% (F1=0.9454). Entre los subsets que incluyen caracteristicas fractales, el mejor resultado es Mes 3 con la combinacion Δ+Espectral (94.27%), seguido de cerca por Mes 6 (93.69%) y Mes 1 (93.67%).

### 5.6 Contribucion de las caracteristicas fractales vs espectrales

Para aislar la contribucion especifica de cada familia de caracteristicas, se evaluaron por separado los subsets puramente fractales y puramente espectrales. La tabla siguiente muestra el mejor accuracy alcanzado por cada tipo de caracteristica en cada condicion:

| Condicion | Solo Fractales (8 metodos) | Solo Espectrales (13 metodos) | Δ + Espectral (18 metodos) | Ganancia Fractal |
|-----------|---------------------------|-------------------------------|---------------------------|------------------|
| Mixto | 73.88% (RF) | 87.63% (RF) | 89.01% (RF) | +1.38 pts |
| Mes 1 | 80.20% (SVM) | 93.57% (LogReg) | 93.67% (LogReg) | +0.10 pts |
| Mes 3 | 80.55% (SVM) | 94.55% (LogReg) | 94.27% (LogReg) | −0.28 pts |
| Mes 6 | 82.30% (SVM) | 93.35% (LogReg) | 93.69% (LogReg) | +0.34 pts |

*Ganancia Fractal = Δ+Espectral − Solo Espectrales (aporte marginal de anadir las 8 caracteristicas fractales a las 13 espectrales).*

**Interpretacion:** Las caracteristicas espectrales en solitario alcanzan rendimientos de 88–95%, demostrando ser el factor dominante en la discriminacion de niveles de MI. Sin embargo, las caracteristicas fractales puras (8 metodos, con `HRS_p64` corregido a particiones reales) alcanzan hasta 82.30% sin asistencia espectral, lo que valida su capacidad discriminativa independiente. La combinacion de ambas familias (Δ+Espectral) proporciona una mejora marginal de entre −0.28 y +1.38 puntos porcentuales, indicando que la informacion capturada por los metodos fractales es en gran medida **complementaria pero parcialmente redundante** con la informacion espectral. En el Mes 3, las caracteristicas espectrales solas (94.55%) superan a la combinacion con fractales (94.27%), sugiriendo que en la fase sub-aguda la senal espectral es suficientemente rica para la clasificacion sin necesidad de informacion fractal adicional — el mismo patron observado en Mixto, Mes1 y Mes6 es una ganancia marginal positiva pero pequena (≤1.4 pts).

---

## 6. Comparacion: Pipeline Dinamico (ERD) vs Ventanas Fijas (W700)

### 6.1 Progresion de resultados a lo largo del desarrollo

| Pipeline | Validacion | Subset de Caracteristicas | Accuracy Maxima |
|----------|-----------|---------------------------|-----------------|
| Epocas dinamicas (ERD) | LOPO (inter-sujeto) | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio | 63.65% |
| Epocas dinamicas (ERD) | LOPO (inter-sujeto) | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | 63.34% |
| Epocas dinamicas (ERD) | Intra-sujeto | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio | 71.79% |
| Epocas dinamicas (ERD) | Intra-sujeto | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | 73.56% |
| Ventanas fijas W700 | Intra-sujeto | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio (spatial mean) | 71.12%¹ |
| Ventanas fijas W700 | Intra-sujeto | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | 82.30% |
| Ventanas fijas W700 | Intra-sujeto | **Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales** | **94.27%** |

¹ Fila sin recalcular en esta revision: proviene del barrido de ventana de la Seccion 4 (`scripts/extract_multi_window.py` / `classify_windows.py`, promedio espacial + RandomForest unicamente), un pipeline distinto al de las Secciones 5–9 que no fue parte de esta auditoria.

### 6.2 Observaciones clave

1. **LOPO vs Intra-sujeto:** El cambio de validacion inter-sujeto (LOPO) a intra-sujeto (5-fold CV) produjo una mejora de ~8–10 puntos porcentuales con las mismas caracteristicas fractales, demostrando que los patrones de activacion fractal durante MI son altamente **especificos de cada paciente**.

2. **Dinamico vs Fijo:** Con caracteristicas exclusivamente fractales (Δ+HO/HRS/HV), las ventanas fijas W700 (82.30%) superan a las epocas dinamicas basadas en ERD (73.56%). Las ventanas contiguas proporcionan **mas datos de entrenamiento** (2,967 epocas vs ~1,184 epocas dinamicas) y evitan el sesgo de seleccion introducido por el detector de ERD.

3. **Rol de las caracteristicas espectrales:** La inclusion de caracteristicas espectrales (ERD ratios, potencias de banda, entropia) produce un salto de ~12 puntos, llevando el rendimiento de ~82% a ~94%. Esto sugiere que las caracteristicas fractales y espectrales capturan aspectos **diferentes y complementarios** de la dinamica neuronal durante MI.

---

## 7. Analisis Longitudinal por Mes

### 7.1 Rendimiento por mes con el mejor subset (Δ+Espectral)

| Mes | Mejor Modelo | Accuracy | F1 Macro | Epocas | Pacientes |
|-----|-------------|----------|----------|--------|-----------|
| 1 (Agudo) | LogisticRegression | 93.67% | 0.9367 | 980 | 5 |
| 3 (Sub-agudo) | LogisticRegression | 94.27% | 0.9427 | 1,100 | 5 |
| 6 (Cronico) | LogisticRegression | 93.69% | 0.9368 | 887 | 4* |
| Mixto | RandomForest | 89.01% | 0.8901 | 2,967 | 5 |

*P5 sin datos de Mes 6.

### 7.2 Observaciones

1. **Los tres meses individuales clasifican de forma muy similar (93.7–94.3%):** A diferencia de la version anterior de este analisis, con las cifras recalculadas los tres momentos temporales quedan agrupados en un rango estrecho: Mes 3 (94.27%), Mes 6 (93.69%) y Mes 1 (93.67%) — una diferencia de apenas 0.6 puntos entre el mejor y el peor. Esto sugiere que la senal discriminativa fractal+espectral del MI es consistentemente fuerte a lo largo de toda la rehabilitacion, sin una etapa claramente superior a las demas.

2. **El Mes 3 es el "punto dulce" esperado, y ahora tambien el mejor:** Con 94.27% (Δ+Espectral) y el mayor numero de epocas (1,100), el Mes 3 representa la fase de rehabilitacion sub-aguda donde la reorganizacion cortical esta en curso pero ya estabilizada. Las caracteristicas espectrales solas alcanzan 94.55% en este mes — el mejor resultado de todo el estudio —, superando incluso a la combinacion con fractales, lo que sugiere que los patrones de ERD en mu/beta son particularmente claros en esta etapa y que anadir informacion fractal aporta poco o nada adicional aqui.

3. **El Mes 6 mantiene alto rendimiento a pesar de menos datos:** Con solo 4 pacientes y 887 epocas, el accuracy se mantiene en 93.69%, indicando que la senal discriminativa persiste en la fase cronica, practicamente al mismo nivel que la fase aguda.

4. **Mezclar meses perjudica la clasificacion:** La condicion Mixta (89.01%) se situa ~5 puntos por debajo de cualquier mes individual. Esto confirma que cada punto temporal posee una **firma neural especifica** que el clasificador explota. Al mezclar datos de diferentes meses, la variabilidad inter-temporal se convierte en ruido que degrada el rendimiento. Este hallazgo tiene implicaciones clinicas: **los modelos de clasificacion de MI deberian calibrarse por separado para cada etapa del proceso de rehabilitacion**.

### 7.3 Modelos por mes

LogisticRegression es el mejor clasificador en los tres meses individuales (Mes 1: 93.67%, Mes 3: 94.27%, Mes 6: 93.69%), beneficiandose de la separabilidad lineal que las caracteristicas espectrales proporcionan en todas las etapas de la rehabilitacion. En la condicion Mixta, en cambio, RandomForest domina (89.01% vs 72.26% de LogisticRegression) — con mas variabilidad inter-temporal en los datos, un modelo no lineal captura mejor la estructura de las clases que un clasificador lineal.

---

## 8. Discusion

### 8.1 Comparacion con la literatura

El estudio de referencia (Martinez-Peon et al., 2024) reporto un 96.42% de accuracy empleando kNN con HRS_p64 en 10 sujetos, 20 electrodos, y validacion intra-sujeto con 5-fold cross-validation en WEKA. Nuestro mejor resultado (94.55%, Mes 3, solo caracteristicas espectrales) se situa ~2 puntos por debajo. Las diferencias metodologicas que explicarian esta brecha incluyen:

| Factor | Martinez-Peon 2024 | Este estudio |
|--------|-------------------|-------------|
| Sujetos | 10 | 5 |
| Electrodos | 20 | 16 |
| Ubicacion electrodos | PM, pre-SMA, DLPFC, IPL (seleccion dirigida a areas motoras) | Montaje 10-20 completo (incluye areas no motoras) |
| Niveles MVR | 4 | 2 (10%, 40%) |
| Duracion de epoca | 1.5 s con marcadores | 2.8 s contiguos sin marcadores |
| Validacion | 5-fold CV | 5-fold CV |
| Software | WEKA | Python/scikit-learn |
| Caracteristicas | Solo Hurst (HO, HRS, HV) | Hurst + fractales + espectrales |

La seleccion dirigida de electrodos sobre areas motoras (PM, pre-SMA, DLPFC, IPL) en el estudio de referencia probablemente contribuye significativamente a su mayor rendimiento, ya que estas regiones son las mas directamente implicadas en la generacion y modulacion de MI.

### 8.2 El rol de las caracteristicas espectrales

El hallazgo mas significativo de este estudio es que las caracteristicas espectrales —particularmente los ratios ERD y las potencias de banda— proporcionan la mayor parte del poder discriminativo (88–95% en solitario). Esto es consistente con decadas de investigacion en BCI que establecen el ERD en bandas mu/beta como el correlato neurofisiologico primario de la activacion sensoriomotora durante MI. Sin embargo, varios aspectos matizan esta conclusion:

1. **Las caracteristicas fractales NO son redundantes:** Los 8 metodos fractales combinados alcanzan 80–82% sin ninguna caracteristica espectral (excluyendo la condicion Mixta), un rendimiento que seria clinicamente aceptable en muchas aplicaciones de BCI. Para un estudio cuyo enfoque principal es el analisis fractal, este resultado es significativo — y ahora refleja una version corregida de `HRS_p64` que ejecuta particiones reales (`p=8`) en lugar de degenerar silenciosamente a Hurst Original (ver nota metodologica al inicio del documento).

2. **Los fractales y espectrales miden aspectos diferentes de la senal:** Mientras las caracteristicas espectrales cuantifican la **distribucion de energia en frecuencia**, las fractales cuantifican la **complejidad temporal y las correlaciones de largo alcance**. Que ambas familias alcancen rendimientos comparables (espectrales algo superiores) sugiere que la dinamica neuronal durante MI se manifiesta simultaneamente en ambos dominios.

3. **La redundancia parcial es esperable:** Dado que ambas familias operan sobre el mismo segmento de EEG, es esperable que capturen informacion parcialmente solapada. El hecho de que la combinacion aporte entre −0.28 y +1.38 puntos adicionales (segun el mes) sugiere que la informacion **unicamente fractal** es limitada, pero no nula.

### 8.3 Limitaciones

1. **Tamano muestral reducido:** 5 pacientes es una muestra pequena que limita la generalizabilidad estadistica, particularmente en el Mes 6 donde solo 4 pacientes tienen datos.
2. **Sin clase de reposo (MVR 0%) en ventanas fijas:** La ausencia de archivos de reposo en el conjunto de datos impide la clasificacion 3-clases (reposo vs 10% vs 40%) en el pipeline de ventanas fijas, lo que habria proporcionado una linea base mas desafiante.
3. **Sin validacion LOPO para W700:** Por restricciones de tiempo, la validacion inter-sujeto no se completo para el pipeline de ventanas fijas. Los resultados LOPO del pipeline dinamico (63.65%) sugieren que la generalizacion inter-sujeto sigue siendo un desafio abierto.
4. **Sin optimizacion de hiperparametros:** Se utilizaron hiperparametros fijos para todos los modelos. Un GridSearchCV por modelo y subset podria producir mejoras adicionales de 1–3 puntos.
5. **Feature selection no explorada:** No se evaluo la contribucion individual de cada caracteristica ni se aplicaron tecnicas de seleccion de variables (e.g., RFE, importancia de Gini), lo que podria identificar las caracteristicas mas informativas.

### 8.4 Implicaciones clinicas

1. **Clasificacion por etapa:** La degradacion del rendimiento al mezclar meses (89.01% vs ~93.7–94.3% per-month) sugiere que los sistemas BCI para rehabilitacion deberian recalibrarse periodicamente para adaptarse a los cambios en los patrones EEG del paciente a lo largo del tiempo.

2. **Ventana optima de 2.8 segundos:** Este valor, determinado empiricamente, es clinicamente razonable: es suficientemente corto para permitir retroalimentacion en tiempo casi real, y suficientemente largo para capturar la dinamica completa del ERD.

3. **Rendimiento estable en toda la rehabilitacion:** El alto rendimiento se mantiene practicamente constante entre el Mes 1 (93.67%, fase aguda), el Mes 3 (94.27%, sub-agudo) y el Mes 6 (93.69%, cronico), lo que es alentador tanto para la intervencion temprana con BCI (ventana de neuroplasticidad maxima) como para su uso sostenido a lo largo de todo el proceso de rehabilitacion.

---

## 9. Conclusiones

1. El tamano de ventana optimo para la extraccion de caracteristicas fractales y espectrales en MI post-AVC es de **700 muestras (2.8 segundos)** con segmentacion contigua al 0% de traslape.

2. Empleando **validacion intra-sujeto con 5-fold CV (80/20)** y el subset combinado **Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales**, se alcanza un **94.27% de accuracy** (F1=0.9427) en la clasificacion de niveles de fuerza MVR 10% vs 40% durante el Mes 3 post-AVC, superando el objetivo del 80% por un margen de 14 puntos. Los Meses 1 (93.67%) y 6 (93.69%) alcanzan un rendimiento practicamente equivalente con el mismo subset.

3. Las caracteristicas **espectrales en solitario** (potencias de banda, ratios ERD, entropia) alcanzan 88–95%, demostrando ser el factor dominante en la discriminacion — el mejor resultado del estudio (94.55%, Mes 3) se logra con espectrales solas. No obstante, las caracteristicas **fractales en solitario** (8 metodos, con `HRS_p64` corregido para ejecutar particiones reales) alcanzan 80–82%, validando su capacidad como enfoque independiente para la clasificacion de MI.

4. La ganancia adicional de combinar fractales con espectrales es **marginal (entre −0.28 y +1.38 puntos segun el mes)**, indicando que ambas familias capturan informacion parcialmente redundante del mismo fenomeno neurofisiologico subyacente.

5. **Mezclar meses perjudica la clasificacion**: la condicion Mixta (89.01%) se situa consistentemente ~5 puntos por debajo de cualquier mes individual (93.7–94.3%), confirmando que cada etapa temporal posee patrones neurales especificos que deben modelarse por separado.

6. **LogisticRegression** es el clasificador mas robusto en los tres meses individuales (93.7–94.3% al combinarse con caracteristicas espectrales), mientras que **RandomForest** domina en la condicion Mixta (89.01%), donde la mayor variabilidad inter-temporal favorece a un modelo no lineal.

7. El pipeline de **ventanas fijas W700** supera al pipeline de **epocas dinamicas basadas en ERD** (82.30% vs 73.56% con caracteristicas exclusivamente fractales), gracias a un mayor volumen de datos de entrenamiento y a la eliminacion del sesgo de seleccion del detector de eventos.

8. **Correccion metodologica de `HRS_p64`:** se identifico y corrigio un error en el calculo del exponente de Hurst con particiones (`HRS`) que, para las duraciones de ventana usadas en este estudio (2.0–8.0 s), hacia que `p=64` (el valor optimo reportado por Martinez-Peon et al., 2024, para epocas mas largas) degenerara siempre en un calculo identico al Hurst Original — verificado numericamente sin excepciones. Se ajusto a `p=8`, el mayor valor que efectivamente ejecuta el pooling particionado sobre sub-series reales en ventanas de 700 muestras. Todas las cifras de este documento que involucran HRS reflejan ya esta correccion.

---

## Apendice A: Configuracion Detallada del Pipeline

### A.1 Parametros de los metodos fractales (adaptativos a la longitud N del segmento)

| Metodo | Parametro | Valor |
|--------|-----------|-------|
| Rescaled Range | Escala minima | 0.128 s (32 muestras) |
| Rescaled Range | Escala maxima | N * 0.40 |
| Rescaled Range | Puntos minimos validos | 5 |
| Rescaled Range | Bloques minimos por escala | 5 |
| Higuchi | k_max | min(ceil(N * 0.25), 30) |
| Higuchi | k_max minimo absoluto | 5 |
| Higuchi | Puntos minimos validos | 5 |
| DFA | Escala minima | 0.064 s (16 muestras) |
| DFA | Escala maxima | N * 0.33 |
| DFA | Ventanas minimas por escala | 3 |
| DFA | Puntos minimos validos | 5 |
| Semivariograma | Lag maximo | N * 0.20 |
| Semivariograma | Lags minimos | 10 |
| Semivariograma | Puntos minimos validos | 5 |
| HRS | Particion (p) | 64 |
| HRS | Minimo muestras por sub-serie | 10 |

### A.2 Bandas espectrales

| Banda | Rango de frecuencia (Hz) |
|-------|--------------------------|
| δ (delta) | 0.5 – 4 |
| θ (theta) | 4 – 8 |
| α (alpha) | 8 – 12 |
| μ (mu) | 8 – 12 (identico a alpha en rango) |
| β (beta) | 12 – 30 |
| γ (gamma) | 30 – 45 |

### A.3 Hiperparametros de clasificadores

| Modelo | Hiperparametros |
|--------|-----------------|
| SVM | C=1.0, kernel='rbf', probability=True, random_state=42 |
| RandomForest | n_estimators=300, max_depth=None, random_state=42, n_jobs=-1 |
| LogisticRegression | C=1.0, max_iter=2000, random_state=42, n_jobs=-1 |
| kNN | n_neighbors=10, metric='chebyshev', n_jobs=-1 |
| GaussianNB | var_smoothing=1e-9 |

### A.4 Preprocesamiento

| Etapa | Parametros |
|-------|-----------|
| Filtro Notch | 60 Hz, Q=30.0 |
| Filtro pasabanda | Butterworth orden 4, 0.5–45 Hz |
| ICA | 3 componentes removidos, max_iter=500, random_state=42 |
| Z-score | Por canal, media=0, std=1 |
| Welch (espectral) | nperseg=250, noverlap=125 |

---

## Apendice B: Distribucion de Datos

### B.1 Epocas por paciente, mes y clase (W700, 2,967 epocas totales)

| Paciente | Mes 1 (MVR 10/40) | Mes 3 (MVR 10/40) | Mes 6 (MVR 10/40) | Total |
|----------|-------------------|-------------------|-------------------|-------|
| P1 | 117 / 156 | 117 / 156 | 78 / 117 | 741 |
| P2 | 117 / 156 | 117 / 156 | 117 / 117 | 780 |
| P3 | 116 / 155 | 117 / 156 | 117 / 156 | 817 |
| P4 | 78 / 117 | 117 / 156 | 117 / 156 | 741 |
| P5 | 78 / 117 | 117 / 156 | 0 / 0 | 468 |
| **Total** | **980** | **1,100** | **887** | **2,967** |

### B.2 Archivos procesados

- **56 archivos .mat** en total
- 28 archivos ASYNCHRONOUS (Motor Imagery sin estimulo)
- 28 archivos SYNCHRONOUS (Motor Imagery con estimulo)
- Niveles MVR: 28 archivos al 10%, 28 archivos al 40%
- 0 archivos de reposo (MVR=0%) — la clase 0 proviene de epocas basales dentro de los archivos de imagery en el pipeline dinamico, pero esta ausente en el pipeline de ventanas fijas

---

*Documento generado automaticamente a partir de los resultados del pipeline Niveles3. Los valores numericos provienen de la ejecucion de `scripts/classify_w700_full.py` con 200 experimentos (5 modelos x 5 subsets x 4 condiciones x 2 tipos de clase).*
