# 📋 Model Card — Motor de Micro-Préstamos Inclusivo

> Documentación técnica del modelo LightGBM Tuned  
> Siguiendo el estándar de Model Cards for Model Reporting (Mitchell et al., 2019)

---

## Información General

| Campo | Detalle |
|-------|---------|
| **Nombre del modelo** | Motor de Micro-Préstamos Inclusivo — LightGBM Tuned |
| **Versión** | v1.0 |
| **Tipo** | Clasificación binaria (default / no-default) |
| **Algoritmo** | LightGBM (Light Gradient Boosting Machine) |
| **Fecha de entrenamiento** | 2026 |
| **Autor** | Abraham Tartalos |
| **Contacto** | GitHub: AbrahamTartalos |

---

## Uso Previsto

### Caso de uso primario

Evaluación del riesgo crediticio de solicitantes de micro-préstamos, con foco en personas que carecen de historial bancario tradicional — trabajadores informales, cuentapropistas y pequeños emprendedores.

### Usuarios previstos

- Instituciones de microfinanzas y cooperativas de crédito
- Fintechs orientadas a inclusión financiera
- Organismos de desarrollo que otorgan crédito a población informal

### Casos de uso fuera del alcance

- **No recomendado** para scoring de créditos corporativos o de alto monto
- **No recomendado** como único criterio de decisión sin revisión humana
- **No recomendado** para poblaciones fuera del contexto socioeconómico argentino / latinoamericano sin reentrenamiento

---

## Dataset de Entrenamiento

| Campo | Detalle |
|-------|---------|
| **Fuente** | Home Credit Default Risk — Kaggle Competition |
| **URL** | https://www.kaggle.com/c/home-credit-default-risk |
| **Tamaño** | 307,511 registros × 122 columnas originales |
| **Período** | Datos históricos de solicitudes de crédito |
| **Variable objetivo** | `TARGET` — 1 = default, 0 = no-default |
| **Desbalance** | 91.93% no-default / 8.07% default |
| **Split** | 80% entrenamiento / 20% test (stratified, random_state=42) |

### Preprocesamiento aplicado

- Tratamiento de 67 columnas con missing values (imputación por mediana/moda)
- Codificación de variables categóricas (Label Encoding)
- Ingeniería de features: 18 nuevas variables (5 temporales, 6 ratios tradicionales, 7 alternativas)
- Balanceo de clases: SMOTE sobre el conjunto de entrenamiento
- Limpieza de nombres de columnas para compatibilidad con LightGBM

---

## Arquitectura del Modelo

### Algoritmo

LightGBM (`LGBMClassifier`) — gradient boosting basado en árboles de decisión con optimización por histogramas.

### Hiperparámetros óptimos

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `n_estimators` | 200 | Número de árboles |
| `max_depth` | 10 | Profundidad máxima por árbol |
| `learning_rate` | 0.05 | Tasa de aprendizaje |
| `num_leaves` | 31 | Número de hojas por árbol |
| `random_state` | 42 | Semilla de reproducibilidad |

### Proceso de optimización

Grid Search con validación cruzada de 3 folds sobre 16 combinaciones de hiperparámetros. Mejor ROC-AUC en CV: 0.9655 (datos balanceados con SMOTE).

### Threshold de clasificación

**0.35** — seleccionado por análisis de curva Precision-Recall como balance óptimo entre inclusión financiera y control del default rate. Ver sección de Trade-offs.

---

## Métricas de Rendimiento

### Conjunto de test (61,503 registros)

| Métrica | Valor |
|---------|-------|
| **ROC-AUC** | **0.7440** |
| Precision | 0.3435 |
| Recall | 0.0906 |
| F1-Score | 0.1434 |
| Accuracy | ~0.91 |
| Default rate en aprobados | 7.50% |

### Referencia de industria

El ROC-AUC de 0.7440 se ubica dentro del rango típico de modelos de scoring crediticio en la industria fintech (0.70–0.78), alcanzando el 98.7% del target definido de 0.75.

### Comparativa con baseline

| Modelo | ROC-AUC | Aprobaciones | Incremento |
|--------|---------|--------------|------------|
| RF Baseline (sin señales alternativas) | 0.6484 | 57,745 | — |
| **LightGBM Tuned** | **0.7440** | **60,193** | **+4.24%** |

### Matriz de confusión (threshold 0.35)

| | Predicho: No-Default | Predicho: Default |
|--|---------------------|-------------------|
| **Real: No-Default** | 55,413 (TN) ✅ | 1,125 (FP) ⚠️ |
| **Real: Default** | 4,482 (FN) ⚠️ | 483 (TP) ✅ |

---

## Explicabilidad

### Método

SHAP (SHapley Additive exPlanations) con `TreeExplainer` — método exacto para modelos basados en árboles. Calculado sobre muestra de 2,000 solicitantes del conjunto de test.

### Top 10 features por importancia SHAP

| Rank | Feature | Mean \|SHAP\| | Tipo |
|------|---------|--------------|------|
| 1 | `EXT_SOURCE_COMBINED` | 0.03528 | ⭐ Alternativo |
| 2 | `NAME_EDUCATION_TYPE_Higher_education` | 0.03240 | Tradicional |
| 3 | `NAME_EDUCATION_TYPE_Secondary_special` | 0.02470 | Tradicional |
| 4 | `ADDRESS_TENURE_SCORE` | 0.02072 | ⭐ Alternativo |
| 5 | `INCOME_STABILITY_SCORE_ADJ` | 0.01968 | ⭐ Alternativo |
| 6 | `ADDRESS_TENURE_YEARS` | 0.01460 | Temporal |
| 7 | `FINANCIAL_INCLUSION_SCORE` | 0.01454 | ⭐ Alternativo |
| 8 | `FLAG_PHONE` | 0.01410 | Tradicional |
| 9 | `NAME_INCOME_TYPE_Working` | 0.01160 | Tradicional |
| 10 | `CREDIT_BUREAU_SCORE` | 0.01139 | ⭐ Alternativo |

**5 de 7 señales alternativas aparecen en el Top 10** — validación cuantitativa del diferenciador del modelo.

### Interpretación de señales clave

- **EXT_SOURCE_COMBINED alto → menor riesgo:** el score externo combinado es el predictor más poderoso. Un valor alto (>0.5) reduce significativamente la probabilidad de default.
- **FINANCIAL_INCLUSION_SCORE alto → menor riesgo:** tener teléfono, email y documentación completa predice responsabilidad financiera.
- **ADDRESS_TENURE_SCORE alto → menor riesgo:** vivir muchos años en la misma dirección es señal robusta de estabilidad de vida.

---

## Trade-offs y Limitaciones

### Trade-off principal: inclusión vs. sostenibilidad

La selección del threshold define el balance entre incluir más personas y mantener el riesgo controlado:

| Threshold | Precision | Recall | F1 | Perfil |
|-----------|-----------|--------|----|--------|
| 0.50 | 0.3803 | 0.0234 | 0.0440 | Conservador extremo |
| **0.35** ⭐ | **0.3435** | **0.0906** | **0.1434** | **Balance elegido** |
| 0.1564 | 0.2018 | 0.4381 | 0.2763 | Máximo recall |

El threshold 0.1564 maximizaría el F1-Score pero implicaría una Precision del 20% — inviable para producción real. El threshold 0.35 representa la decisión consciente de priorizar sostenibilidad sobre volumen de aprobaciones.

### Limitaciones conocidas

**Sesgo de datos históricos:** el modelo aprende de decisiones de crédito pasadas, que pueden contener sesgos sistémicos contra poblaciones informales. Las señales alternativas mitigan parcialmente este problema pero no lo eliminan.

**Recall bajo (0.0906):** el modelo detecta solo el 9% de los defaults reales con threshold 0.35. Esto implica que muchos defaults no son detectados — trade-off aceptado para no excluir a buenos pagadores.

**Generalización geográfica:** entrenado con datos del contexto Home Credit (Europa del Este y Asia). La aplicación directa al mercado argentino requiere validación y posible reentrenamiento con datos locales.

**Estabilidad temporal:** el modelo no fue validado con datos de diferentes períodos temporales (backtesting). En producción real sería necesario monitoreo de data drift.

**Features alternativas sintéticas:** las 7 señales alternativas fueron construidas como proxies a partir de variables disponibles en el dataset, no como señales reales de campo (ej. historial de pagos de servicios). En producción, reemplazarlas por señales reales mejoraría el poder predictivo.

---

## Consideraciones Éticas

### Equidad algorítmica

El modelo no utiliza directamente variables de género, edad, etnia o nacionalidad como features predictivos. Sin embargo, variables proxy como tipo de empleo o nivel educativo pueden correlacionar con estas características demográficas e introducir sesgo indirecto.

**Recomendación:** antes de deployment en producción, realizar análisis de equidad (fairness audit) por grupos demográficos para detectar disparate impact.

### Explicabilidad regulatoria

El modelo cuenta con explicabilidad a nivel individual mediante SHAP, lo que permite cumplir con requerimientos regulatorios de transparencia algorítmica (análogos al "derecho a explicación" del GDPR europeo).

Cada decisión puede ser explicada en términos de los factores que la determinaron, tanto al solicitante como a auditores regulatorios.

### Uso responsable

- El modelo debe ser usado como **herramienta de apoyo a la decisión**, no como decisor autónomo
- Las decisiones de rechazo deben incluir un canal de apelación humana
- El modelo debe ser re-evaluado periódicamente para detectar degradación de performance

---

## Reproducibilidad

### Entorno de ejecución

| Componente | Versión |
|------------|---------|
| Python | 3.11.14 |
| LightGBM | 4.2.0 |
| Scikit-learn | 1.4.0 |
| Imbalanced-learn | 0.12.0 |
| SHAP | 0.44.1 |
| Pandas | 2.1.4 |
| NumPy | 1.26.3 |

### Semillas de reproducibilidad

- `random_state=42` en train/test split, SMOTE y LightGBM
- Pipeline completamente reproducible ejecutando los notebooks en orden

### Nota sobre el análisis SHAP

El notebook `04_SHAP_Explainability.ipynb` fue ejecutado en Google Colab (GPU T4) por requerimientos de memoria RAM. Es compatible con entorno local pero puede requerir reducir el tamaño de muestra en hardware limitado.

---

## Referencias

- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
- Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
- Lundberg, S., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP).
- Mitchell, M., et al. (2019). Model Cards for Model Reporting.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.

---

*Model Card creado como parte del proyecto Motor de Micro-Préstamos Inclusivo — Abraham Tartalos, 2026*
