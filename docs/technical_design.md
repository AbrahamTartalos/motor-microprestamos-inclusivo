# TECHNICAL DESIGN DOCUMENT
## Motor de Micro-préstamos Inclusivo

---

### 1. RESUMEN EJECUTIVO

**Proyecto:** Sistema de scoring crediticio inclusivo basado en Machine Learning

**Objetivo Técnico:** Desarrollar un clasificador binario (aprobar/rechazar) que utilice variables tradicionales y alternativas para evaluar solvencia crediticia de trabajadores informales.

**Arquitectura:** Pipeline de ML end-to-end (datos → procesamiento → modelo → explicabilidad → deployment)

**Stack Principal:** Python, Scikit-learn, XGBoost, SHAP, Streamlit

---

### 2. ARQUITECTURA DEL SISTEMA

```
┌─────────────────────────────────────────────────────────────┐
│                     ARQUITECTURA GENERAL                     │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   RAW DATA   │─────▶│DATA PIPELINE │─────▶│   TRAINING   │
│  (Kaggle)    │      │ (ETL + FE)   │      │   (Models)   │
└──────────────┘      └──────────────┘      └──────┬───────┘
                                                     │
                                                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  DASHBOARD   │◀─────│ EXPLAINABILITY│◀─────│TRAINED MODEL │
│ (Streamlit)  │      │    (SHAP)     │      │   (pickle)   │
└──────────────┘      └──────────────┘      └──────────────┘
```

---

### 3. COMPONENTES DEL SISTEMA

#### 3.1 Módulo de Datos (Data Pipeline)

**Responsabilidad:** Ingesta, limpieza, transformación y feature engineering

**Inputs:**
- `application_train.csv` - Datos principales de solicitudes
- `bureau.csv` - Historial crediticio en otras instituciones (opcional)
- `previous_application.csv` - Solicitudes previas (opcional)

**Outputs:**
- `cleaned_data.csv` - Dataset limpio
- `features_engineered.csv` - Dataset con variables tradicionales + alternativas
- `data_dictionary.csv` - Documentación de variables

**Procesos Clave:**

```python
# Pseudo-código del pipeline
def data_pipeline(raw_data):
    # 1. LIMPIEZA
    data = load_data(raw_data)
    data = handle_missing_values(data)
    data = remove_duplicates(data)
    data = fix_data_types(data)
    
    # 2. FEATURE ENGINEERING
    data = create_traditional_features(data)
    data = create_alternative_features(data)
    data = create_interaction_features(data)
    
    # 3. ENCODING
    data = encode_categorical(data)
    data = scale_numerical(data)
    
    # 4. SPLIT
    X_train, X_test, y_train, y_test = split_data(data)
    
    return X_train, X_test, y_train, y_test
```

**Decisiones Técnicas:**

| Decisión | Opción Elegida | Justificación |
|----------|----------------|---------------|
| Manejo de valores nulos | Imputación + Flag variable | Preserva información de "falta de dato" |
| Encoding categórico | Target Encoding | Mejor para variables de alta cardinalidad |
| Scaling | StandardScaler | Requerido para modelos lineales |
| Split | Stratified 80/20 | Mantiene proporción de clases |

---

#### 3.2 Módulo de Feature Engineering

**Variables Tradicionales (Baseline):**

```python
TRADITIONAL_FEATURES = [
    # Demográficas
    'AGE',
    'GENDER',
    'EDUCATION_LEVEL',
    'FAMILY_SIZE',
    
    # Financieras
    'INCOME_TOTAL',
    'CREDIT_AMOUNT',
    'LOAN_ANNUITY',
    'DEBT_TO_INCOME_RATIO',
    
    # Empleo
    'EMPLOYMENT_TYPE',
    'YEARS_EMPLOYED',
    'OCCUPATION_TYPE'
]
```

**Variables Alternativas (Innovación):**

```python
ALTERNATIVE_FEATURES = {
    # 1. ESTABILIDAD DE INGRESOS IRREGULARES
    'income_variance': 'Variabilidad de ingresos últimos 6 meses',
    'income_trend': 'Tendencia (creciente/estable/decreciente)',
    'income_floor': 'Ingreso mínimo últimos 6 meses',
    'income_stability_score': 'Score compuesto de estabilidad',
    
    # 2. HISTORIAL DE PAGOS ALTERNATIVOS
    'utility_payment_score': 'Puntualidad en servicios (0-100)',
    'consecutive_on_time_months': 'Meses consecutivos sin atrasos',
    'payment_consistency_ratio': 'Ratio pagos a tiempo / total',
    
    # 3. PATRONES DE AHORRO
    'has_savings': 'Tiene ahorros (binario)',
    'savings_consistency': 'Ahorra regularmente (binario)',
    'savings_to_income_ratio': 'Ratio ahorro/ingreso',
    
    # 4. ARRAIGO Y ESTABILIDAD
    'address_tenure_months': 'Meses en domicilio actual',
    'job_tenure_months': 'Meses en trabajo/actividad actual',
    'has_references': 'Tiene referencias verificables'
}
```

**Funciones de Feature Engineering:**

```python
def create_income_stability_features(df):
    """
    Crea features relacionadas con estabilidad de ingresos
    Para trabajadores informales con ingresos variables
    """
    # Simular ingresos mensuales (si no existen)
    if 'monthly_incomes' not in df.columns:
        df['income_variance'] = np.random.uniform(0.1, 0.5, len(df))
        df['income_trend'] = np.random.choice(['growing', 'stable', 'declining'], len(df))
        df['income_floor'] = df['AMT_INCOME_TOTAL'] * np.random.uniform(0.6, 0.9, len(df))
    
    # Score compuesto
    df['income_stability_score'] = (
        (1 - df['income_variance']) * 0.4 +  # Menos varianza = mejor
        (df['income_trend'] == 'growing') * 0.3 +
        (df['income_floor'] / df['AMT_INCOME_TOTAL']) * 0.3
    )
    
    return df

def create_alternative_payment_features(df):
    """
    Simula historial de pagos de servicios
    Dato clave para trabajadores informales
    """
    # Simular pagos puntuales (correlacionado con buen crédito)
    df['utility_payment_score'] = np.where(
        df['TARGET'] == 0,  # No default
        np.random.normal(85, 10, len(df)),  # Mejor score
        np.random.normal(65, 15, len(df))   # Peor score
    ).clip(0, 100)
    
    df['consecutive_on_time_months'] = (df['utility_payment_score'] / 10).astype(int)
    df['payment_consistency_ratio'] = df['utility_payment_score'] / 100
    
    return df
```

---

#### 3.3 Módulo de Modelado

**Estrategia Multi-Modelo:**

```
BASELINE MODEL (Variables Tradicionales)
    ↓
CANDIDATE MODELS (Todas las Variables)
    ├── Logistic Regression (interpretable)
    ├── Random Forest (robusto)
    ├── XGBoost (alto performance)
    └── LightGBM (rápido)
    ↓
BEST MODEL (Selección por ROC-AUC)
    ↓
HYPERPARAMETER TUNING
    ↓
FINAL MODEL
```

**Configuración de Modelos:**

```python
# 1. BASELINE: Logistic Regression
baseline_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

# 2. CANDIDATE: XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Balanceo
    random_state=42
)

# 3. CANDIDATE: LightGBM (Recomendado)
lgbm_model = LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    is_unbalance=True,  # Manejo automático de desbalance
    random_state=42
)
```

**Manejo de Desbalance de Clases:**

El dataset tendrá probablemente desbalance (más "buenos pagadores" que "malos pagadores"):

```python
# Opción 1: SMOTE (Synthetic Minority Over-sampling)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Opción 2: Class Weights (en el modelo)
model = XGBClassifier(scale_pos_weight=ratio_negativo_positivo)

# Opción 3: Threshold Adjustment
# Ajustar threshold de decisión de 0.5 a valor óptimo
```

**Validación:**

```python
# Stratified K-Fold Cross-Validation
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model.fit(X_tr, y_tr)
    score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    cv_scores.append(score)

print(f"CV ROC-AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
```

---

#### 3.4 Módulo de Evaluación

**Métricas Primarias:**

```python
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Evaluación completa del modelo"""
    
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics['true_negative'] = tn
    metrics['false_positive'] = fp  # Aprobamos malo (CARO)
    metrics['false_negative'] = fn  # Rechazamos bueno (OPORTUNIDAD PERDIDA)
    metrics['true_positive'] = tp
    
    return metrics
```

**Métricas de Negocio:**

```python
def calculate_business_impact(baseline_metrics, inclusive_metrics, dataset_size):
    """Calcula impacto de negocio del modelo inclusivo"""
    
    # Tasa de aprobación
    baseline_approval_rate = (baseline_metrics['tp'] + baseline_metrics['fp']) / dataset_size
    inclusive_approval_rate = (inclusive_metrics['tp'] + inclusive_metrics['fp']) / dataset_size
    
    improvement = (inclusive_approval_rate - baseline_approval_rate) / baseline_approval_rate
    
    # Tasa de default
    baseline_default_rate = baseline_metrics['fp'] / (baseline_metrics['tp'] + baseline_metrics['fp'])
    inclusive_default_rate = inclusive_metrics['fp'] / (inclusive_metrics['tp'] + inclusive_metrics['fp'])
    
    # Ingresos adicionales
    additional_approvals = (inclusive_approval_rate - baseline_approval_rate) * dataset_size
    avg_loan_amount = 300  # USD
    interest_rate = 0.15  # 15% anual
    additional_revenue = additional_approvals * avg_loan_amount * interest_rate
    
    return {
        'approval_improvement_%': improvement * 100,
        'additional_approvals': additional_approvals,
        'additional_revenue_usd': additional_revenue,
        'baseline_default_rate_%': baseline_default_rate * 100,
        'inclusive_default_rate_%': inclusive_default_rate * 100
    }
```

---

#### 3.5 Módulo de Explicabilidad

**SHAP (SHapley Additive exPlanations):**

```python
import shap

# Inicializar explainer
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

# GLOBAL: ¿Qué variables importan más en general?
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)  # Bee swarm plot

# LOCAL: ¿Por qué Juan fue aprobado/rechazado?
def explain_individual_prediction(model, explainer, X, idx):
    """Explica predicción individual"""
    
    prediction = model.predict_proba(X.iloc[[idx]])[0][1]
    
    shap_value = explainer.shap_values(X.iloc[[idx]])
    
    # Visualización
    shap.force_plot(
        explainer.expected_value,
        shap_value[0],
        X.iloc[idx],
        matplotlib=True
    )
    
    # Texto explicativo
    features_impact = pd.DataFrame({
        'feature': X.columns,
        'value': X.iloc[idx].values,
        'shap_value': shap_value[0]
    }).sort_values('shap_value', key=abs, ascending=False)
    
    return prediction, features_impact.head(5)
```

**Explicaciones en Lenguaje Natural:**

```python
def generate_explanation_text(features_impact, prediction):
    """Genera explicación legible para usuario"""
    
    if prediction >= 0.5:
        decision = "APROBADO"
        verb = "favorecen"
    else:
        decision = "NO APROBADO"
        verb = "afectan negativamente"
    
    top_features = features_impact.head(3)
    
    explanation = f"Decisión: {decision} (confianza: {prediction*100:.1f}%)\n\n"
    explanation += f"Las principales razones que {verb} esta decisión son:\n"
    
    for idx, row in top_features.iterrows():
        feature_name = row['feature']
        feature_value = row['value']
        impact = "positivo" if row['shap_value'] > 0 else "negativo"
        
        explanation += f"- {feature_name}: {feature_value} (impacto {impact})\n"
    
    return explanation
```

---

#### 3.6 Módulo de Dashboard (Streamlit)

**Arquitectura del Dashboard:**

```
HOMEPAGE
├── Simulador de Solicitud (Main Feature)
│   ├── Form: Input de datos del solicitante
│   ├── Botón: "Evaluar Solicitud"
│   └── Output: Decisión + Explicación SHAP
│
├── Análisis Comparativo
│   ├── Métricas: Baseline vs Inclusivo
│   ├── ROC Curves
│   ├── Confusion Matrices
│   └── Feature Importance
│
└── Casos de Éxito
    ├── Perfil 1: María (Vendedora)
    ├── Perfil 2: Juan (Freelancer)
    └── Perfil 3: Ana (Comerciante)
```

**Código Base del Dashboard:**

```python
import streamlit as st
import pickle
import pandas as pd

# Cargar modelo
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Título
st.title("🏦 Motor de Micro-préstamos Inclusivo")
st.markdown("Sistema de evaluación crediticia para trabajadores informales")

# Sidebar: Navegación
page = st.sidebar.selectbox(
    "Navegación",
    ["🎯 Simulador", "📊 Análisis", "⭐ Casos de Éxito"]
)

if page == "🎯 Simulador":
    st.header("Evalúa tu Solicitud de Préstamo")
    
    # Form
    with st.form("loan_application"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Edad", 18, 70, 30)
            income = st.number_input("Ingreso mensual (USD)", 100, 5000, 500)
            utility_score = st.slider("Puntualidad en pagos de servicios (últimos 12 meses)", 0, 12, 10)
        
        with col2:
            employment_years = st.number_input("Años en actividad actual", 0, 30, 2)
            has_savings = st.checkbox("¿Tienes ahorros?")
            loan_amount = st.number_input("Monto solicitado (USD)", 50, 1000, 300)
        
        submitted = st.form_submit_button("Evaluar Solicitud")
    
    if submitted:
        # Preparar features
        features = prepare_features(age, income, utility_score, ...)
        
        # Predicción
        prediction_proba = model.predict_proba([features])[0][1]
        prediction = int(prediction_proba >= 0.5)
        
        # Resultado
        if prediction == 1:
            st.error(f"❌ Solicitud NO aprobada (Riesgo: {prediction_proba*100:.1f}%)")
        else:
            st.success(f"✅ Solicitud APROBADA (Confianza: {(1-prediction_proba)*100:.1f}%)")
            st.info(f"Monto aprobado: ${loan_amount} USD")
        
        # Explicación (simulada, realmente usarías SHAP)
        with st.expander("🔍 ¿Por qué esta decisión?"):
            st.write("Las principales razones son...")
```

**Deployment:**
- Plataforma: **Streamlit Cloud** (gratuito)
- URL: `https://[nombre-app].streamlit.app`
- Requisitos: `requirements.txt` con todas las librerías

---

### 4. FLUJO DE DATOS

```
┌─────────────────────────────────────────────────────────┐
│                   FLUJO DE ENTRENAMIENTO                 │
└─────────────────────────────────────────────────────────┘

RAW DATA (CSV)
    ↓
[1] LOAD & CLEAN
    ├─ Manejo de nulos
    ├─ Corrección de tipos
    └─ Eliminación de duplicados
    ↓
[2] FEATURE ENGINEERING
    ├─ Variables tradicionales
    ├─ Variables alternativas
    └─ Interacciones
    ↓
[3] PREPROCESSING
    ├─ Encoding categórico
    ├─ Scaling numérico
    └─ Train/Test split
    ↓
[4] TRAINING
    ├─ Baseline model
    ├─ Candidate models
    └─ Best model selection
    ↓
[5] EVALUATION
    ├─ Métricas técnicas
    ├─ Métricas de negocio
    └─ SHAP analysis
    ↓
[6] SAVE MODEL
    └─ model.pkl, scaler.pkl, encoder.pkl

┌─────────────────────────────────────────────────────────┐
│                   FLUJO DE PREDICCIÓN                    │
└─────────────────────────────────────────────────────────┘

USER INPUT (Dashboard)
    ↓
[1] VALIDATE INPUT
    └─ Rangos, tipos, obligatorios
    ↓
[2] PREPROCESS
    ├─ Aplicar scaler guardado
    ├─ Aplicar encoder guardado
    └─ Crear features derivados
    ↓
[3] PREDICT
    └─ model.predict_proba()
    ↓
[4] EXPLAIN
    └─ SHAP values
    ↓
[5] FORMAT OUTPUT
    ├─ Decisión (Aprobado/Rechazado)
    ├─ Monto (si aprobado)
    └─ Explicación natural
    ↓
DISPLAY RESULT
```

---

### 5. ESTRUCTURA DEL REPOSITORIO

```
motor-microprestamos-inclusivo/
│
├── data/
│   ├── raw/                    # Datos originales (no subir a GitHub)
│   ├── processed/              # Datos procesados
│   └── data_dictionary.csv     # Documentación de variables
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Análisis exploratorio
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── clean_data.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── explain.py
│   │
│   └── utils/
│       ├── config.py           # Configuraciones
│       └── helpers.py          # Funciones auxiliares
│
├── app/
│   ├── app.py                  # Dashboard Streamlit
│   ├── components/             # Componentes reutilizables
│   └── assets/                 # Imágenes, CSS
│
├── models/
│   ├── baseline_model.pkl
│   ├── final_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── docs/
│   ├── project_charter.md
│   ├── technical_design.md
│   └── presentation.pdf
│
├── tests/
│   └── test_*.py              # Tests unitarios (opcional)
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

---

### 6. DECISIONES TÉCNICAS CLAVE

| Aspecto | Decisión | Alternativa Considerada | Justificación |
|---------|----------|-------------------------|---------------|
| **Algoritmo Principal** | XGBoost/LightGBM | Random Forest, Neural Networks | Mejor balance performance/interpretabilidad para datos tabulares |
| **Manejo Desbalance** | SMOTE + Class Weights | Under-sampling, Ensemble methods | Genera datos sintéticos sin perder información |
| **Explicabilidad** | SHAP | LIME, Partial Dependence | Estándar en fintech, matemáticamente robusto |
| **Encoding Categórico** | Target Encoding | One-Hot, Label Encoding | Eficiente para alta cardinalidad |
| **Dashboard** | Streamlit | Flask, Dash | Rápido desarrollo, deployment gratuito |
| **Deployment** | Streamlit Cloud | Render, Heroku | Gratuito, específico para Streamlit |

---

### 7. CONSIDERACIONES DE PERFORMANCE

#### Tiempos Esperados (en laptop estándar)

| Proceso | Tiempo Estimado | Optimización Posible |
|---------|-----------------|----------------------|
| Carga de datos | 10-30 segundos | Usar Parquet en vez de CSV |
| Feature engineering | 1-3 minutos | Vectorización con Numpy |
| Entrenamiento baseline | 10-30 segundos | N/A (ya es rápido) |
| Entrenamiento XGBoost | 2-5 minutos | Reducir n_estimators, usar GPU |
| SHAP cálculo (test set) | 5-10 minutos | Usar TreeExplainer (más rápido) |
| Predicción individual | <1 segundo | N/A |

#### Tamaño de Archivos

| Archivo | Tamaño Estimado | Nota |
|---------|-----------------|------|
| Raw data | 100-500 MB | No subido a GitHub |
| Processed data | 50-200 MB | No subido a GitHub |
| Model .pkl | 5-50 MB | Subido a GitHub |
| Dashboard app | <1 MB | Subido a GitHub |

---

### 8. TESTING Y VALIDACIÓN

```python
# Test de sanity checks
def test_model_sanity():
    """Verifica que modelo funciona básicamente"""
    
    # 1. Predicción básica
    sample = X_test.iloc[0:1]
    pred = model.predict(sample)
    assert pred in [0, 1], "Predicción debe ser 0 o 1"
    
    # 2. Probabilidades suman 1
    proba = model.predict_proba(sample)
    assert abs(proba.sum() - 1.0) < 1e-6, "Probabilidades deben sumar 1"
    
    # 3. Features requeridos
    required_features = ['AGE', 'INCOME_TOTAL', 'CREDIT_AMOUNT']
    assert all(f in X_test.columns for f in required_features)
    
    print("✅ Todos los tests pasaron")

# Test de sesgo
def test_model_bias():
    """Verifica que modelo no tiene sesgos evidentes"""
    
    # Analizar aprobaciones por género, edad, etc.
    for segment in ['GENDER', 'AGE_GROUP']:
        approval_rates = df.groupby(segment)['APPROVED'].mean()
        
        # No debería haber diferencias >20% entre grupos
        max_diff = approval_rates.max() - approval_rates.min()
        assert max_diff < 0.20, f"Sesgo detectado en {segment}"
```

---

### 9. MANTENIMIENTO Y EVOLUCIÓN

#### Posibles Mejoras Futuras (Fuera de alcance inicial)

1. **Modelo de monto óptimo:** No solo aprobar/rechazar, sino también sugerir monto máximo seguro
2. **Segmentación de clientes:** Identificar perfiles de riesgo (bajo, medio, alto)
3. **Monitoreo en producción:** Detectar drift de datos/modelo
4. **A/B testing:** Comparar modelo en producción con variantes
5. **Integración con APIs reales:** Verificación de identidad, bureau de crédito

#### Versionado de Modelos

```python
# Guardar modelo con metadata
model_metadata = {
    'version': '1.0.0',
    'train_date': '2024-XX-XX',
    'roc_auc': 0.76,
    'features': list(X_train.columns),
    'hyperparameters': model.get_params()
}

with open('models/model_v1.0.0.pkl', 'wb') as f:
    pickle.dump({'model': model, 'metadata': model_metadata}, f)
```

---

### 10. SEGURIDAD Y PRIVACIDAD

#### Consideraciones

- **No incluir datos personales identificables** en repositorio público
- **Anonimizar** cualquier ejemplo en dashboard
- **No hardcodear** API keys o secrets (usar variables de entorno)
- **Validar inputs** del usuario en dashboard (prevenir inyección)

```python
# Ejemplo de validación de input
def validate_user_input(age, income, loan_amount):
    """Valida inputs del usuario"""
    
    if not (18 <= age <= 100):
        raise ValueError("Edad debe estar entre 18 y 100")
    
    if not (0 < income <= 100000):
        raise ValueError("Ingreso debe ser positivo y razonable")
    
    if not (50 <= loan_amount <= 5000):
        raise ValueError("Monto solicitado fuera de rango permitido")
    
    return True
```

---

### 11. DOCUMENTACIÓN REQUERIDA

- ✅ **README.md:** Overview, cómo ejecutar, resultados
- ✅ **Notebooks comentados:** Código explicado paso a paso
- ✅ **Docstrings:** En todas las funciones
- ✅ **Data Dictionary:** Descripción de cada variable
- ✅ **requirements.txt:** Dependencias exactas
- ✅ **CHANGELOG.md:** Registro de cambios (opcional)

---

### 12. PRÓXIMOS PASOS TÉCNICOS

1. ✅ Completar esta documentación
2. ✅ Setup inicial: Crear estructura de carpetas
3. ✅ Descargar dataset Home Credit
4. ✅ Iniciar Notebook 01_EDA.ipynb
5. ⏭️ Crear functions en src/data/
6. ⏭️ Desarrollar pipeline completo
7. ⏭️ Entrenar y evaluar modelos
8. ⏭️ Crear dashboard
9. ⏭️ Deploy a Streamlit Cloud
10. ⏭️ Documentar resultados finales

---

**Versión:** 1.0  
**Autor:** Abraham Tartalos  
**Última Actualización:** 12/02/26

---