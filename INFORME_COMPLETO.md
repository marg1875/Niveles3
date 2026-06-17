# Clasificacion de Niveles de Fuerza en Imagery Motor post-AVC mediante Caracteristicas Fractales y Espectrales con Ventanas Fijas

**Pipeline:** Niveles3
**Fecha:** Junio 2026
**Validacion:** Intra-sujeto 5-fold Cross-Validation (80/20)
**Ventana optima:** W700 = 2.8 segundos (0% overlap, segmentos contiguos)

---

## Resumen

Se evaluo un pipeline de clasificacion de niveles de fuerza en Motor Imagery (MI) post-Accidente Cerebrovascular (AVC) combinando caracteristicas fractales (Rescaled Range, Higuchi, Detrended Fluctuation Analysis, Semivariograma, Promedio, Hurst Original, Hurst R/S con Particiones, Hurst via Semivariograma) y caracteristicas espectrales (potencias de banda δ/θ/α/β/γ, ratios de Desincronizacion Relacionada a Eventos, entropia espectral, ratio α/β, ratio μ/β). Sobre 5 pacientes (Px.006-Px.010) con registros EEG de 16 canales a 250 Hz en tres momentos temporales (Mes 1, Mes 3, Mes 6), se optimizo el tamano de ventana mediante un barrido empirico de 16 tamanos (2.0s–8.0s) determinando W=700 muestras (2.8 s) como la configuracion optima. Empleando validacion intra-sujeto con StratifiedKFold (k=5) y pool de predicciones, la combinacion Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales alcanzo **93.27% de accuracy** (F1=0.9326) en el Mes 1 con Regresion Logistica, superando el objetivo del 80%. Las caracteristicas espectrales demostraron ser el factor dominante (88–93% en solitario), mientras que las caracteristicas fractales puras (8 metodos) alcanzaron hasta 84% sin asistencia espectral, validando su capacidad discriminativa independiente del nivel de fuerza imaginado.

---

## 1. Introduccion y Motivacion

El Accidente Cerebrovascular (AVC) es una de las principales causas de discapacidad motora a nivel mundial. Las Interfaces Cerebro-Computadora (BCI) basadas en Motor Imagery (MI) —la imaginacion de un movimiento sin ejecutarlo— ofrecen una via prometedora para la neurorehabilitacion al activar circuitos sensoriomotores incluso en ausencia de movimiento real. Un desafio clinico fundamental es la capacidad de **discriminar entre diferentes niveles de fuerza imaginada**, tipicamente expresados como porcentaje de la Contraccion Voluntaria Maxima (Maximum Voluntary Contraction, MVR). Una clasificacion precisa del nivel de esfuerzo mental permitiria sistemas BCI mas graduados y adaptativos durante la rehabilitacion.

Investigaciones previas han empleado diversas tecnicas de procesamiento de senales EEG para este proposito. En particular, Martinez-Peon et al. (2024, _J. Neural Eng._ 21, 046024) reportaron un 96.42% de accuracy empleando el exponente de Hurst con particiones (HRS) y clasificadores estandar (kNN, SVM, RandomForest, entre otros) en 10 sujetos con 20 electrodos bajo validacion intra-sujeto con 10-fold cross-validation en WEKA.

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
| Pacientes | 5 (Px.006–Px.010) |
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
| Mes 6 | 887 | Solo Month6 (4 pacientes, Px.010 sin datos) |

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
| SVM | 68.59 | 70.14 | 75.43 | 79.81 | 83.89 |
| RandomForest | 68.79 | 70.24 | 76.31 | 88.37 | **89.48** |
| LogisticRegression | 60.47 | 61.11 | 62.86 | 74.55 | 74.96 |
| kNN | 68.32 | 70.34 | 74.35 | 76.71 | 79.37 |
| GaussianNB | 57.84 | 56.72 | 58.44 | 64.85 | 65.96 |

### 5.2 Mes 1 (fase aguda, 980 epocas)

| Clasificador | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio | Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Potencia de banda δ / Potencia de banda θ / Potencia de banda α / Potencia de banda β / Potencia de banda γ / Ratio ERD δ / Ratio ERD θ / Ratio ERD α / Ratio ERD β / Ratio ERD γ / Entropia Espectral / Ratio α/β / Ratio μ/β | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |
|-------------|-----------|-----------|-----------|-----------|-----------|
| SVM | 75.10 | 80.82 | 81.12 | 88.16 | 90.51 |
| RandomForest | 73.37 | 78.57 | 80.61 | 88.06 | 89.08 |
| LogisticRegression | 67.35 | 72.24 | 74.29 | 90.31 | **93.27** |
| kNN | 72.55 | 78.88 | 77.45 | 83.27 | 83.06 |
| GaussianNB | 66.22 | 64.08 | 65.71 | 73.06 | 73.27 |

### 5.3 Mes 3 (fase sub-aguda, 1,100 epocas)

| Clasificador | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio | Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Potencia de banda δ / Potencia de banda θ / Potencia de banda α / Potencia de banda β / Potencia de banda γ / Ratio ERD δ / Ratio ERD θ / Ratio ERD α / Ratio ERD β / Ratio ERD γ / Entropia Espectral / Ratio α/β / Ratio μ/β | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |
|-------------|-----------|-----------|-----------|-----------|-----------|
| SVM | 76.09 | 82.55 | 81.55 | 91.09 | 91.91 |
| RandomForest | 74.36 | 81.55 | 81.09 | **93.45** | 93.18 |
| LogisticRegression | 71.45 | 75.64 | 77.73 | 92.82 | 93.18 |
| kNN | 74.09 | 79.55 | 78.18 | 87.55 | 87.55 |
| GaussianNB | 67.91 | 66.55 | 67.82 | 76.91 | 77.00 |

### 5.4 Mes 6 (fase cronica, 887 epocas, 4 pacientes)

| Clasificador | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio | Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | Potencia de banda δ / Potencia de banda θ / Potencia de banda α / Potencia de banda β / Potencia de banda γ / Ratio ERD δ / Ratio ERD θ / Ratio ERD α / Ratio ERD β / Ratio ERD γ / Entropia Espectral / Ratio α/β / Ratio μ/β | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |
|-------------|-----------|-----------|-----------|-----------|-----------|
| SVM | 76.55 | 77.79 | 83.88 | 86.25 | 90.53 |
| RandomForest | 74.97 | 78.13 | 83.31 | 90.64 | **92.90** |
| LogisticRegression | 68.88 | 76.10 | 82.64 | 89.29 | **92.90** |
| kNN | 73.62 | 74.75 | 80.50 | 80.83 | 82.30 |
| GaussianNB | 64.83 | 64.83 | 65.84 | 77.34 | 77.23 |

### 5.5 Mejor combinacion por condicion (resumen)

| Condicion | Mejor Accuracy | Mejor F1 Macro | Modelo | Subset de Caracteristicas |
|-----------|---------------|----------------|--------|---------------------------|
| Mixto | 89.48% | 0.8948 | RandomForest | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |
| Mes 1 | **93.27%** | **0.9326** | LogisticRegression | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |
| Mes 3 | 93.18% | 0.9318 | RandomForest / LogisticRegression | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |
| Mes 6 | 92.90% | 0.9290 | RandomForest / LogisticRegression | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales |

### 5.6 Contribucion de las caracteristicas fractales vs espectrales

Para aislar la contribucion especifica de cada familia de caracteristicas, se evaluaron por separado los subsets puramente fractales y puramente espectrales. La tabla siguiente muestra el mejor accuracy alcanzado por cada tipo de caracteristica en cada condicion:

| Condicion | Solo Fractales (8 metodos) | Solo Espectrales (13 metodos) | Δ + Espectral (18 metodos) | Ganancia Fractal |
|-----------|---------------------------|-------------------------------|---------------------------|------------------|
| Mixto | 76.31% (RF) | 88.37% (RF) | 89.48% (RF) | +1.11 pts |
| Mes 1 | 81.12% (SVM) | 90.31% (LogReg) | 93.27% (LogReg) | +2.96 pts |
| Mes 3 | 82.55% (SVM) | 93.45% (RF) | 93.18% (RF) | −0.27 pts |
| Mes 6 | 83.88% (SVM) | 90.64% (RF) | 92.90% (RF) | +2.26 pts |

**Interpretacion:** Las caracteristicas espectrales en solitario alcanzan rendimientos de 88–93%, demostrando ser el factor dominante en la discriminacion de niveles de MI. Sin embargo, las caracteristicas fractales puras (8 metodos) alcanzan hasta 83.88% sin asistencia espectral, lo que valida su capacidad discriminativa independiente. La combinacion de ambas familias (Δ+Espectral) proporciona una mejora marginal de 0–3 puntos porcentuales, indicando que la informacion capturada por los metodos fractales es en gran medida **complementaria pero parcialmente redundante** con la informacion espectral. Notablemente, en el Mes 3, las caracteristicas espectrales solas (93.45%) superan ligeramente a la combinacion con fractales (93.18%), sugiriendo que en la fase sub-aguda la senal espectral es suficientemente rica para la clasificacion sin necesidad de informacion fractal adicional.

---

## 6. Comparacion: Pipeline Dinamico (ERD) vs Ventanas Fijas (W700)

### 6.1 Progresion de resultados a lo largo del desarrollo

| Pipeline | Validacion | Subset de Caracteristicas | Accuracy Maxima |
|----------|-----------|---------------------------|-----------------|
| Epocas dinamicas (ERD) | LOPO (inter-sujeto) | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio | 63.65% |
| Epocas dinamicas (ERD) | LOPO (inter-sujeto) | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | 63.34% |
| Epocas dinamicas (ERD) | Intra-sujeto | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio | 71.79% |
| Epocas dinamicas (ERD) | Intra-sujeto | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | 73.56% |
| Ventanas fijas W700 | Intra-sujeto | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio (spatial mean) | 71.12% |
| Ventanas fijas W700 | Intra-sujeto | Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio / Δ-Hurst Original / Δ-Hurst R/S con Particiones / Δ-Hurst Semivariograma | 81.55% |
| Ventanas fijas W700 | Intra-sujeto | **Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales** | **93.27%** |

### 6.2 Observaciones clave

1. **LOPO vs Intra-sujeto:** El cambio de validacion inter-sujeto (LOPO) a intra-sujeto (5-fold CV) produjo una mejora de ~8–10 puntos porcentuales con las mismas caracteristicas fractales, demostrando que los patrones de activacion fractal durante MI son altamente **especificos de cada paciente**.

2. **Dinamico vs Fijo:** Con caracteristicas exclusivamente fractales (Δ+HO/HRS/HV), las ventanas fijas W700 (81.55%) superan a las epocas dinamicas basadas en ERD (73.56%). Las ventanas contiguas proporcionan **mas datos de entrenamiento** (2,967 epocas vs ~1,184 epocas dinamicas) y evitan el sesgo de seleccion introducido por el detector de ERD.

3. **Rol de las caracteristicas espectrales:** La inclusion de caracteristicas espectrales (ERD ratios, potencias de banda, entropia) produce un salto de ~12 puntos, llevando el rendimiento de ~81% a ~93%. Esto sugiere que las caracteristicas fractales y espectrales capturan aspectos **diferentes y complementarios** de la dinamica neuronal durante MI.

---

## 7. Analisis Longitudinal por Mes

### 7.1 Rendimiento por mes con el mejor subset (Δ+Espectral)

| Mes | Mejor Modelo | Accuracy | F1 Macro | Epocas | Pacientes |
|-----|-------------|----------|----------|--------|-----------|
| 1 (Agudo) | LogisticRegression | 93.27% | 0.9326 | 980 | 5 |
| 3 (Sub-agudo) | RandomForest | 93.18% | 0.9318 | 1,100 | 5 |
| 6 (Cronico) | RandomForest | 92.90% | 0.9290 | 887 | 4* |
| Mixto | RandomForest | 89.48% | 0.8948 | 2,967 | 5 |

*Px.010 sin datos de Mes 6.

### 7.2 Observaciones

1. **La fase aguda (Mes 1) clasifica sorprendentemente bien:** Con 93.27%, el Mes 1 supera ligeramente al Mes 3 (93.18%) y al Mes 6 (92.90%). Este resultado contradice la expectativa de que la senal EEG en la fase aguda post-AVC seria mas ruidosa o menos discriminativa. Por el contrario, los cambios neurales tempranos tras el AVC podrian generar patrones de activacion mas marcados y por tanto mas faciles de clasificar.

2. **El Mes 3 es el "punto dulce" esperado:** Con 93.18% y el mayor numero de epocas (1,100), el Mes 3 representa la fase de rehabilitacion sub-aguda donde la reorganizacion cortical esta en curso pero ya estabilizada. El hecho de que las caracteristicas espectrales solas alcancen 93.45% en este mes (superando a la combinacion con fractales) sugiere que los patrones de ERD en mu/beta son particularmente claros en esta etapa.

3. **El Mes 6 mantiene alto rendimiento a pesar de menos datos:** Con solo 4 pacientes y 887 epocas, el accuracy se mantiene en 92.90%, indicando que la senal discriminativa persiste en la fase cronica.

4. **Mezclar meses perjudica la clasificacion:** La condicion Mixta (89.48%) se situa 3–4 puntos por debajo de cualquier mes individual. Esto confirma que cada punto temporal posee una **firma neural especifica** que el clasificador explota. Al mezclar datos de diferentes meses, la variabilidad inter-temporal se convierte en ruido que degrada el rendimiento. Este hallazgo tiene implicaciones clinicas: **los modelos de clasificacion de MI deberian calibrarse por separado para cada etapa del proceso de rehabilitacion**.

### 7.3 Modelos por mes

En el Mes 1, LogisticRegression demuestra ser el mejor clasificador (93.27%), beneficiandose de la separabilidad lineal que las caracteristicas espectrales proporcionan en la fase aguda. En los Meses 3 y 6, RandomForest y LogisticRegression comparten el mejor rendimiento (~93%), indicando que tanto enfoques lineales como no lineales convergen cuando la senal es suficientemente rica.

---

## 8. Discusion

### 8.1 Comparacion con la literatura

El estudio de referencia (Martinez-Peon et al., 2024) reporto un 96.42% de accuracy empleando kNN con HRS_p64 en 10 sujetos, 20 electrodos, y validacion intra-sujeto con 10-fold cross-validation en WEKA. Nuestro mejor resultado (93.27%) se situa ~3 puntos por debajo. Las diferencias metodologicas que explicarian esta brecha incluyen:

| Factor | Martinez-Peon 2024 | Este estudio |
|--------|-------------------|-------------|
| Sujetos | 10 | 5 |
| Electrodos | 20 | 16 |
| Ubicacion electrodos | PM, pre-SMA, DLPFC, IPL (seleccion dirigida a areas motoras) | Montaje 10-20 completo (incluye areas no motoras) |
| Niveles MVR | 4 | 2 (10%, 40%) |
| Duracion de epoca | 1.5 s con marcadores | 2.8 s contiguos sin marcadores |
| Validacion | 10-fold CV | 5-fold CV |
| Software | WEKA | Python/scikit-learn |
| Caracteristicas | Solo Hurst (HO, HRS, HV) | Hurst + fractales + espectrales |

La seleccion dirigida de electrodos sobre areas motoras (PM, pre-SMA, DLPFC, IPL) en el estudio de referencia probablemente contribuye significativamente a su mayor rendimiento, ya que estas regiones son las mas directamente implicadas en la generacion y modulacion de MI.

### 8.2 El rol de las caracteristicas espectrales

El hallazgo mas significativo de este estudio es que las caracteristicas espectrales —particularmente los ratios ERD y las potencias de banda— proporcionan la mayor parte del poder discriminativo (88–93% en solitario). Esto es consistente con decadas de investigacion en BCI que establecen el ERD en bandas mu/beta como el correlato neurofisiologico primario de la activacion sensoriomotora durante MI. Sin embargo, varios aspectos matizan esta conclusion:

1. **Las caracteristicas fractales NO son redundantes:** Los 8 metodos fractales combinados alcanzan 81–84% sin ninguna caracteristica espectral, un rendimiento que seria clinicamente aceptable en muchas aplicaciones de BCI. Para un estudio cuyo enfoque principal es el analisis fractal, este resultado es significativo.

2. **Los fractales y espectrales miden aspectos diferentes de la senal:** Mientras las caracteristicas espectrales cuantifican la **distribucion de energia en frecuencia**, las fractales cuantifican la **complejidad temporal y las correlaciones de largo alcance**. Que ambas familias alcancen rendimientos comparables (espectrales algo superiores) sugiere que la dinamica neuronal durante MI se manifiesta simultaneamente en ambos dominios.

3. **La redundancia parcial es esperable:** Dado que ambas familias operan sobre el mismo segmento de EEG, es esperable que capturen informacion parcialmente solapada. El hecho de que la combinacion aporte solo 0–3 puntos adicionales sugiere que la informacion **unicamente fractal** es limitada, pero no nula.

### 8.3 Limitaciones

1. **Tamano muestral reducido:** 5 pacientes es una muestra pequena que limita la generalizabilidad estadistica, particularmente en el Mes 6 donde solo 4 pacientes tienen datos.
2. **Sin clase de reposo (MVR 0%) en ventanas fijas:** La ausencia de archivos de reposo en el conjunto de datos impide la clasificacion 3-clases (reposo vs 10% vs 40%) en el pipeline de ventanas fijas, lo que habria proporcionado una linea base mas desafiante.
3. **Sin validacion LOPO para W700:** Por restricciones de tiempo, la validacion inter-sujeto no se completo para el pipeline de ventanas fijas. Los resultados LOPO del pipeline dinamico (63.65%) sugieren que la generalizacion inter-sujeto sigue siendo un desafio abierto.
4. **Sin optimizacion de hiperparametros:** Se utilizaron hiperparametros fijos para todos los modelos. Un GridSearchCV por modelo y subset podria producir mejoras adicionales de 1–3 puntos.
5. **Feature selection no explorada:** No se evaluo la contribucion individual de cada caracteristica ni se aplicaron tecnicas de seleccion de variables (e.g., RFE, importancia de Gini), lo que podria identificar las caracteristicas mas informativas.

### 8.4 Implicaciones clinicas

1. **Clasificacion por etapa:** La degradacion del rendimiento al mezclar meses (89.48% vs ~93% per-month) sugiere que los sistemas BCI para rehabilitacion deberian recalibrarse periodicamente para adaptarse a los cambios en los patrones EEG del paciente a lo largo del tiempo.

2. **Ventana optima de 2.8 segundos:** Este valor, determinado empiricamente, es clinicamente razonable: es suficientemente corto para permitir retroalimentacion en tiempo casi real, y suficientemente largo para capturar la dinamica completa del ERD.

3. **Fase aguda viable:** El alto rendimiento en el Mes 1 (93.27%) es alentador para la intervencion temprana con BCI, una ventana temporal donde la neuroplasticidad es maxima.

---

## 9. Conclusiones

1. El tamano de ventana optimo para la extraccion de caracteristicas fractales y espectrales en MI post-AVC es de **700 muestras (2.8 segundos)** con segmentacion contigua al 0% de traslape.

2. Empleando **validacion intra-sujeto con 5-fold CV (80/20)** y el subset combinado **Δ-Rescaled Range / Δ-Higuchi / Δ-DFA / Δ-Semivariograma / Δ-Promedio + 13 Caracteristicas Espectrales**, se alcanza un **93.27% de accuracy** (F1=0.9326) en la clasificacion de niveles de fuerza MVR 10% vs 40% durante el Mes 1 post-AVC, superando el objetivo del 80% por un margen de 13 puntos.

3. Las caracteristicas **espectrales en solitario** (potencias de banda, ratios ERD, entropia) alcanzan 88–93%, demostrando ser el factor dominante en la discriminacion. No obstante, las caracteristicas **fractales en solitario** (8 metodos) alcanzan 81–84%, validando su capacidad como enfoque independiente para la clasificacion de MI.

4. La ganancia adicional de combinar fractales con espectrales es **marginal (0–3 puntos)**, indicando que ambas familias capturan informacion parcialmente redundante del mismo fenomeno neurofisiologico subyacente.

5. **Mezclar meses perjudica la clasificacion**: la condicion Mixta (89.48%) se situa consistentemente por debajo de cualquier mes individual (~93%), confirmando que cada etapa temporal posee patrones neurales especificos que deben modelarse por separado.

6. **LogisticRegression y RandomForest** son los clasificadores mas robustos, con rendimientos practicamente identicos (~93%) al combinarse con caracteristicas espectrales en los tres momentos temporales.

7. El pipeline de **ventanas fijas W700** supera al pipeline de **epocas dinamicas basadas en ERD** (81.55% vs 73.56% con caracteristicas exclusivamente fractales), gracias a un mayor volumen de datos de entrenamiento y a la eliminacion del sesgo de seleccion del detector de eventos.

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
| Px.006 | 117 / 156 | 117 / 156 | 78 / 117 | 741 |
| Px.007 | 117 / 156 | 117 / 156 | 117 / 117 | 780 |
| Px.008 | 116 / 155 | 117 / 156 | 117 / 156 | 817 |
| Px.009 | 78 / 117 | 117 / 156 | 117 / 156 | 741 |
| Px.010 | 78 / 117 | 117 / 156 | 0 / 0 | 468 |
| **Total** | **980** | **1,100** | **887** | **2,967** |

### B.2 Archivos procesados

- **56 archivos .mat** en total
- 28 archivos ASYNCHRONOUS (Motor Imagery sin estimulo)
- 28 archivos SYNCHRONOUS (Motor Imagery con estimulo)
- Niveles MVR: 28 archivos al 10%, 28 archivos al 40%
- 0 archivos de reposo (MVR=0%) — la clase 0 proviene de epocas basales dentro de los archivos de imagery en el pipeline dinamico, pero esta ausente en el pipeline de ventanas fijas

---

*Documento generado automaticamente a partir de los resultados del pipeline Niveles3. Los valores numericos provienen de la ejecucion de `scripts/classify_w700_full.py` con 200 experimentos (5 modelos x 5 subsets x 4 condiciones x 2 tipos de clase).*
