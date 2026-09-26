# Análisis intra-sujeto por paciente (W700, 3 clases) — resultados del póster

**Pipeline:** Niveles3, ventanas fijas W700 (2.8 s), reposo genuino (`extract_w700_full.py`)
**Características:** Δ-fractales + espectrales (18), 16 canales
**Clasificador:** regresión logística (mejor promedio de los 3 meses en 3 clases)
**Validación:** StratifiedKFold k=5 dentro de cada paciente (igual que `classify_w700_full.py`)
**Scripts:** `scripts/analyze_intrasubject_w700.py`, `scripts/plot_poster_intrasubject.py`

---

## 1. Re-ejecución de la extracción

`Fixed_W700_3class_Features.csv` está en `.gitignore`, así que se regeneró con `scripts/extract_w700_full.py` sobre los 56 archivos `.mat` originales (numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, pandas 3.0.2).

| | INFORME_COMPLETO.md | Esta re-ejecución |
|---|---|---|
| Épocas totales | 3,977 | 3,985 |
| Épocas activas (10 % / 40 %) | 2,967 | 2,967 (1,472 / 1,495) |
| Épocas de reposo | 1,010 | 1,018 |
| LR 3 clases, mes 1 / 3 / 6 (agrupado) | 64.87 / 63.92 / 66.31 % | 67.24 / 64.53 / 66.58 % |
| LR 2 clases, mes 1 / 3 / 6 (agrupado) | 91.12 / 90.82 / 95.26 % | 93.06 / 92.73 / 95.26 % |

Las diferencias (hasta 2.4 puntos) son consistentes con la advertencia de `requirements.txt`: FastICA no es bit-reproducible entre entornos, y eso cambia ligeramente la detección de ERD y, con ella, el número de épocas de reposo. **Todas las cifras de este documento y del póster provienen de esta re-ejecución.**

## 2. Por qué se reporta exactitud balanceada

En 3 clases el modelo casi no reconoce el reposo (TPR ≈ 9–13 %). Por eso la exactitud simple depende de cuántas épocas de reposo tenga cada registro: mientras menos reposo, más alta. Se reporta la **exactitud balanceada** (promedio del acierto de cada clase), que no depende de la proporción de clases. El nivel de azar es 33.3 %.

### 2.1 Registros con protocolo distinto (P4 mes 1, P5 mes 1)

| Registro | Duración | Marcadores | Separación entre marcadores |
|---|---|---|---|
| Estándar (51 archivos) | ~156 s | 10 | ~15.6 s |
| Px.009 (P4) mes 1: los 4 archivos | ~89 s | 40 | ~2.2 s |
| Px.010 (P5) mes 1: `px010_10_1.mat` | ~89 s | 40 | ~2.2 s |

Con estímulos cada 2.2 s no hay huecos > 5 s entre eventos, así que `extract_basal_epochs` casi no encuentra reposo (en los archivos síncronos cae al *fallback* de 1 época). P4 mes 1 queda con 13 épocas de reposo de 137 (9.5 %), frente a ~26 % en los demás. Su exactitud simple (84.7 %) está inflada por eso; su exactitud balanceada es 62.4 %, igual a la de sus otros meses.

## 3. Resultados por paciente (3 clases, sin balanceo)

| Mes | Paciente | N | Reposo | Exactitud | **Exactitud balanceada** (± DE entre particiones) | TPR reposo | TPR 10 % | TPR 40 % |
|---|---|---|---|---|---|---|---|---|
| 1 | P1 | 293 | 73 | 59.4 | **53.5** ± 4.9 | 6.8 | 79.1 | 74.5 |
| 1 | P2 | 306 | 86 | 65.0 | **61.2** ± 2.0 | 12.8 | 81.8 | 89.1 |
| 1 | P3 | 301 | 81 | 64.8 | **59.9** ± 2.0 | 8.6 | 84.5 | 86.4 |
| 1 | P4 | 137 | 13 | 84.7 | **62.4** ± 3.8 | 0.0 | 90.3 | 96.8 |
| 1 | P5 | 245 | 49 | 72.7 | **62.3** ± 4.9 | 8.2 | 94.2 | 84.5 |
| 1 | Grupo | 1282 | 302 | 67.2 | **59.8** | 8.9 | 85.1 | 85.3 |
| 3 | P1 | 304 | 84 | 67.1 | **62.3** ± 4.0 | 6.0 | 92.7 | 88.2 |
| 3 | P2 | 301 | 81 | 63.5 | **59.1** ± 6.2 | 13.6 | 84.5 | 79.1 |
| 3 | P3 | 300 | 80 | 67.0 | **61.9** ± 7.5 | 11.2 | 94.5 | 80.0 |
| 3 | P4 | 292 | 72 | 66.1 | **61.5** ± 7.2 | 26.4 | 81.8 | 76.4 |
| 3 | P5 | 300 | 80 | 59.0 | **54.3** ± 5.4 | 7.5 | 85.5 | 70.0 |
| 3 | Grupo | 1497 | 397 | 64.5 | **59.7** | 12.6 | 87.8 | 78.7 |
| 6 | P1 | 295 | 74 | 69.5 | **62.6** ± 4.4 | 6.8 | 85.5 | 95.5 |
| 6 | P2 | 298 | 78 | 62.1 | **57.3** ± 3.2 | 12.8 | 82.7 | 76.4 |
| 6 | P3 | 311 | 89 | 68.8 | **64.8** ± 1.2 | 9.0 | 96.4 | 89.1 |
| 6 | P4 | 302 | 78 | 65.9 | **60.1** ± 0.4 | 9.0 | 77.7 | 93.8 |
| 6 | Grupo | 1206 | 319 | 66.6 | **61.2** | 9.4 | 85.6 | 88.7 |

P5 no tiene registros en el mes 6.

- Todos los pacientes superan el azar en todos los meses.
- P1 mejora del mes 1 al 3; P3 alcanza su máximo en el mes 6; P2 y P4 se mantienen estables; P5 baja en el mes 3 (pierde reconocimiento del 40 %: 84.5 → 70.0 %). En 2 clases P5 también baja (93.9 → 83.6 %), así que esa caída es real.
- Matriz de confusión del mes 6 (normalizada por fila por paciente y promediada): reposo 9.4 %, 10 % 85.6 %, 40 % 88.7 %. La media de la diagonal (61.2 %) coincide con la exactitud balanceada promedio de los pacientes.

## 4. Prueba de balanceo de clases (solo en el entrenamiento de cada partición)

| Mes | Método | Exactitud balanceada | F1 macro | TPR reposo | TPR 10 % | TPR 40 % |
|---|---|---|---|---|---|---|
| 1 | Sin balanceo | 59.8 | 0.563 | 8.9 | 85.1 | 85.3 |
| 1 | SMOTE | 59.3 | 0.593 | 29.1 | 75.9 | 72.9 |
| 1 | `class_weight="balanced"` | 58.8 | 0.588 | 28.1 | 75.7 | 72.5 |
| 3 | Sin balanceo | 59.7 | 0.564 | 12.6 | 87.8 | 78.7 |
| 3 | SMOTE | 57.8 | 0.574 | 27.0 | 76.7 | 69.6 |
| 3 | `class_weight="balanced"` | 56.4 | 0.561 | 25.7 | 76.5 | 67.1 |
| 6 | Sin balanceo | 61.2 | 0.569 | 9.4 | 85.6 | 88.7 |
| 6 | SMOTE | 59.7 | 0.588 | 23.2 | 73.6 | 82.4 |
| 6 | `class_weight="balanced"` | 59.4 | 0.583 | 21.9 | 74.1 | 82.2 |

El balanceo sube el reconocimiento del reposo (~9–13 % → ~22–29 %) a costa de 10 % y 40 % (~−10 puntos). La exactitud balanceada no mejora (entre −0.5 y −3.3 puntos). El reposo no es separable con estas características: el problema no es el desbalance. El póster usa la versión **sin balanceo**.

## 5. Pendientes

- Los archivos síncronos (sin "img") son de **movimiento ejecutado** (README) y se clasifican junto con los de imaginación.
- Decidir si P4 mes 1 y `px010_10_1.mat` se excluyen por protocolo distinto; en este documento se conservan.
- Actualizar INFORME_COMPLETO.md con esta re-ejecución o fijar una semilla/entorno que reproduzca sus cifras.

## Reproducción

```bash
python scripts/extract_w700_full.py          # ~30 min, regenera Fixed_W700_3class_Features.csv
python scripts/analyze_intrasubject_w700.py  # results_W700_intrasubject_per_patient.csv + matrices
python scripts/plot_poster_intrasubject.py   # figuras del póster (a) y (b)
```
