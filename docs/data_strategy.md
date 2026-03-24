# DATA STRATEGY DOCUMENT
## Motor de Micro-préstamos Inclusivo

---

### 1. FUENTES DE DATOS

#### Dataset Principal: Home Credit Default Risk

**Origen:** Kaggle Competition (https://www.kaggle.com/c/home-credit-default-risk)

**Descripción:**  
Dataset de Home Credit, una institución financiera que provee préstamos a personas con historial crediticio limitado o inexistente. Contiene datos de más de 300,000 solicitudes de préstamo.

**Por qué este dataset:**
- ✅ Específicamente diseñado para el problema de inclusión financiera
- ✅ Datos reales (anonimizados) de una fintech real
- ✅ Múltiples tablas relacionadas (permite feature engineering avanzado)
- ✅ Alta calidad (limpio, documentado)
- ✅ Tamaño adecuado (suficiente para entrenar, no excesivo)

**Archivos principales:**

| Archivo | Registros | Descripción | Uso en Proyecto |
|---------|-----------|-------------|-----------------|
| `application_train.csv` | ~307K | Datos principales de solicitudes | **PRIMARIO** - Base del modelo |
| `application_test.csv` | ~48K | Set de test (sin target) | Opcional - para validación final |
| `bureau.csv` | ~1.7M | Historial de crédito en otras instituciones | Feature engineering avanzado |
| `bureau_balance.csv` | ~27M | Balance mensual de créditos previos | Opcional - si hay tiempo |
| `previous_application.csv` | ~1.6M | Solicitudes previas del cliente | Feature engineering |
| `POS_CASH_balance.csv` | ~10M | Pagos mensuales de compras a plazos | Opcional |
| `credit_card_balance.csv` | ~3.8M | Balance mensual de tarjetas | Opcional |
| `installments_payments.csv` | ~13M | Historial de pagos | Feature engineering (patrones) |

**Estrategia de uso:**
- **MVP (Mínimo Viable):** Solo `application_train.csv`
- **Completo:** `application_train.csv` + `bureau.csv` + `previous_application.csv`
- **Avanzado:** Incluir todas las tablas relacionadas

---

### 2. VARIABLE TARGET

**Variable objetivo:** `TARGET`
- **Tipo:** Binaria (0 o 1)
- **Significado:**
  - `0` = Cliente pagó el préstamo correctamente (NO default)
  - `1` = Cliente tuvo dificultades de pago (DEFAULT)

**Distribución esperada:**
- Desbalanceada (~90% clase 0, ~10% clase 1)
- Esto es realista: la mayoría paga, pocos hacen default

**Implicaciones:**
- Necesitaremos técnicas para manejar desbalance (SMOTE, class weights)
- Métrica principal debe ser ROC-AUC (no accuracy)

---

### 3. VARIABLES ESPERADAS (PRINCIPALES)

#### Variables Demográficas
```python
DEMOGRAPHIC_FEATURES = {
    'CODE_GENDER': 'Género (M/F)',
    'FLAG_OWN_CAR': 'Posee auto (Y/N)',
    'FLAG_OWN_REALTY': 'Posee inmueble (Y/N)',
    'CNT_CHILDREN': 'Número de hijos',
    'AMT_INCOME_TOTAL': 'Ingreso total anual',
    'NAME_INCOME_TYPE': 'Tipo de ingreso (empleado, comerciante, etc.)',
    'NAME_EDUCATION_TYPE': 'Nivel educativo',
    'NAME_FAMILY_STATUS': 'Estado civil',
    'NAME_HOUSING_TYPE': 'Tipo de vivienda',
    'DAYS_BIRTH': 'Edad (en días negativos desde hoy)',
    'DAYS_EMPLOYED': 'Días empleado (negativo si activo)',
}
```

#### Variables Financieras (Solicitud)
```python
LOAN_FEATURES = {
    'AMT_CREDIT': 'Monto del crédito solicitado',
    'AMT_ANNUITY': 'Anualidad del préstamo',
    'AMT_GOODS_PRICE': 'Precio del bien a comprar',
    'NAME_CONTRACT_TYPE': 'Tipo de contrato (cash/revolving)',
    'NAME_TYPE_SUITE': 'Quién acompaña al solicitante',
}
```

#### Variables de Contacto y Documentación
```python
CONTACT_FEATURES = {
    'FLAG_MOBIL': 'Proporcionó teléfono móvil',
    'FLAG_EMP_PHONE': 'Proporcionó teléfono trabajo',
    'FLAG_WORK_PHONE': 'Teléfono trabajo accesible',
    'FLAG_CONT_MOBILE': 'Contactado via móvil',
    'FLAG_PHONE': 'Proporcionó teléfono fijo',
    'FLAG_EMAIL': 'Proporcionó email',
    
    'FLAG_DOCUMENT_X': 'Proporcionó documento X (varios tipos)',
}
```

#### Variables Externas (Scoring)
```python
EXTERNAL_FEATURES = {
    'EXT_SOURCE_1': 'Score externo 1 (normalizado)',
    'EXT_SOURCE_2': 'Score externo 2 (normalizado)',  
    'EXT_SOURCE_3': 'Score externo 3 (normalizado)',
    # Estas son muy predictivas pero "tradicionales"
}
```

---

### 4. FEATURE ENGINEERING PLAN

#### 4.1 Variables a Crear (Tradicionales)

```python
# Ratios financieros
'CREDIT_TO_INCOME_RATIO' = AMT_CREDIT / AMT_INCOME_TOTAL
'ANNUITY_TO_INCOME_RATIO' = AMT_ANNUITY / AMT_INCOME_TOTAL
'CREDIT_TO_GOODS_RATIO' = AMT_CREDIT / AMT_GOODS_PRICE

# Transformaciones de edad
'AGE_YEARS' = abs(DAYS_BIRTH) / 365
'EMPLOYED_YEARS' = abs(DAYS_EMPLOYED) / 365  # Si positivo

# Flags compuestos
'HAS_ANY_CONTACT' = FLAG_MOBIL | FLAG_PHONE | FLAG_EMAIL
'DOCUMENT_COUNT' = sum(FLAG_DOCUMENT_*)
```

#### 4.2 Variables a Crear (ALTERNATIVAS - Innovación)

Estas son las que **diferencian** nuestro proyecto:

```python
# GRUPO 1: ESTABILIDAD DE INGRESOS
'income_variance_score': 
    """
    Simula variabilidad de ingresos para trabajadores informales.
    Lógica: Si NAME_INCOME_TYPE == 'Commercial associate' o 'Working',
    mayor varianza. Si 'State servant', menor varianza.
    Cálculo: Random normal ajustado por tipo de empleo
    """

'income_stability_flag':
    """
    Binario: 1 si ingresos estables, 0 si variables
    Estable = tipo empleado formal + antigüedad >2 años
    """

'income_trend':
    """
    Categórico: 'growing', 'stable', 'declining'
    Simular basado en edad y tipo de empleo:
    - Joven + informal = growing
    - Edad media + formal = stable
    - Mayor + informal = declining
    """

'minimum_income_6m':
    """
    Ingreso mínimo estimado en últimos 6 meses
    = AMT_INCOME_TOTAL * random(0.6-0.9) según estabilidad
    """

# GRUPO 2: PAGOS ALTERNATIVOS (Simulados)
'utility_payment_punctuality_score':
    """
    Score 0-100 de puntualidad en pagos de servicios.
    Lógica: Correlacionar inversamente con TARGET
    - Si TARGET=0 (buen pagador): score ~ Normal(85, 10)
    - Si TARGET=1 (mal pagador): score ~ Normal(65, 15)
    Clip entre 0-100
    """

'consecutive_ontime_months':
    """
    Número de meses consecutivos sin atraso (0-12)
    = utility_payment_punctuality_score / 8.33
    """

'services_payment_ratio':
    """
    Ratio de pagos a tiempo / total pagos
    = utility_payment_punctuality_score / 100
    """

# GRUPO 3: COMPORTAMIENTO DE AHORRO
'has_any_savings':
    """
    Binario: 1 si tiene ahorros, 0 si no
    Lógica: Probabilidad aumenta con:
    - Ingreso más alto
    - Edad más alta
    - Empleo formal
    """

'savings_consistency':
    """
    Binario: 1 si ahorra regularmente (aunque sea poco)
    Condición: has_any_savings==1 AND income_stability_flag==1
    """

'estimated_savings_ratio':
    """
    Ratio ahorro / ingreso mensual
    Si has_any_savings:
        = random(0.05-0.20) según income level
    Else:
        = 0
    """

# GRUPO 4: ARRAIGO Y ESTABILIDAD
'address_stability_years':
    """
    Años en domicilio actual (estimado)
    Usar REGION_RATING_CLIENT (rating de región del cliente)
    Lógica: Rating alto + mayor edad = más años
    """

'job_stability_years':
    """
    Años en actividad/trabajo actual
    = abs(DAYS_EMPLOYED) / 365 si es estable
    Ajustar según NAME_INCOME_TYPE
    """

'has_verified_references':
    """
    Binario: Tiene referencias verificables
    Proxy: Combinación de FLAG_PHONE, FLAG_EMAIL, FLAG_EMP_PHONE
    """

# GRUPO 5: COMPORTAMIENTO TRANSACCIONAL
'digital_payment_adoption':
    """
    Score 0-100 de adopción de pagos digitales
    Proxy usando:
    - Edad (jóvenes mayor adopción)
    - Urbanidad (ciudad mayor adopción)
    - Educación (mayor educación, mayor adopción)
    """

'financial_activity_diversity':
    """
    Diversidad de actividad financiera
    Proxy: DOCUMENT_COUNT + HAS_ANY_CONTACT
    Mayor diversidad = más involucrado en sistema financiero
    """
```

#### 4.3 Variables de Agregación (Bureau y Previous Application)

Si usamos las tablas relacionadas:

```python
# De bureau.csv (historial en otras instituciones)
'bureau_active_loans_count': 'Número de préstamos activos'
'bureau_total_debt': 'Deuda total en otras instituciones'
'bureau_avg_days_overdue': 'Promedio días de atraso'
'bureau_loan_types_count': 'Diversidad de tipos de préstamo'

# De previous_application.csv
'previous_loans_count': 'Préstamos previos con Home Credit'
'previous_approved_ratio': 'Ratio aprobados / solicitados'
'previous_avg_amount': 'Monto promedio de préstamos previos'
'previous_cancellation_rate': 'Tasa de cancelación de préstamos'
```

---

### 5. ESTRATEGIA DE MISSING VALUES

El dataset Home Credit tiene bastantes valores nulos. Estrategia:

```python
MISSING_VALUE_STRATEGY = {
    # Imputación Simple
    'numeric_low_missing': 'median',  # <10% missing
    'categorical_low_missing': 'mode',
    
    # Imputación + Flag
    'numeric_high_missing': 'median + create FLAG_MISSING',
    # Ejemplo: OCCUPATION_TYPE tiene muchos nulos
    # Imputar con 'Unknown' + crear FLAG_OCCUPATION_MISSING
    
    # Variables externas (muy importantes)
    'EXT_SOURCE_*': 'median + FLAG + KNN imputation',
    
    # Drop
    'columns_>70%_missing': 'drop column',
}
```

**Ejemplo de implementación:**

```python
def handle_missing_values(df):
    """Maneja valores faltantes con estrategia híbrida"""
    
    # 1. Identificar columnas con alto % missing
    missing_pct = df.isnull().mean()
    high_missing_cols = missing_pct[missing_pct > 0.7].index
    df = df.drop(columns=high_missing_cols)
    
    # 2. Para columnas con missing moderado (10-70%), crear flag
    moderate_missing_cols = missing_pct[(missing_pct > 0.1) & (missing_pct <= 0.7)].index
    
    for col in moderate_missing_cols:
        df[f'FLAG_MISSING_{col}'] = df[col].isnull().astype(int)
    
    # 3. Imputar
    # Numérico: mediana
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Categórico: moda o 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna('Unknown', inplace=True)
    
    return df
```

---

### 6. ESTRATEGIA DE ENCODING

```python
ENCODING_STRATEGY = {
    # Binarias (M/F, Y/N)
    'binary_categorical': 'Label Encoding (0/1)',
    
    # Ordinales (Low/Medium/High)
    'ordinal_categorical': 'Ordinal Encoding (0/1/2)',
    
    # Alta cardinalidad (>10 categorías)
    'high_cardinality': 'Target Encoding',
    # Ejemplo: ORGANIZATION_TYPE tiene 50+ valores
    
    # Baja cardinalidad (<10 categorías)
    'low_cardinality': 'One-Hot Encoding',
}
```

**Target Encoding (recomendado para fintech):**

```python
from category_encoders import TargetEncoder

def target_encode_features(X_train, X_test, y_train, categorical_cols):
    """
    Codifica variables categóricas usando la media del target
    Previene data leakage usando solo train
    """
    
    encoder = TargetEncoder(cols=categorical_cols)
    
    # Fit solo en train
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    
    # Transform en test (sin ver y_test)
    X_test_encoded = encoder.transform(X_test)
    
    return X_train_encoded, X_test_encoded
```

---

### 7. ESTRATEGIA DE FEATURE SELECTION

No todas las variables serán útiles. Estrategia de selección:

```python
FEATURE_SELECTION_STEPS = [
    # Paso 1: Eliminar constantes
    'drop_constant_features',  # Varianza = 0
    
    # Paso 2: Eliminar casi-constantes
    'drop_quasi_constant',  # >95% mismo valor
    
    # Paso 3: Eliminar duplicadas
    'drop_duplicate_features',  # Correlación = 1
    
    # Paso 4: Eliminar alta correlación
    'drop_high_correlation',  # Correlación >0.95
    
    # Paso 5: Feature importance (post-modelo)
    'keep_top_N_by_importance',  # Top 50-100 features
]
```

**Implementación básica:**

```python
def select_features(X_train, X_test, y_train, model):
    """Selección de features basada en importancia"""
    
    # Entrenar modelo base
    model.fit(X_train, y_train)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top N features (50-100)
    top_features = importance_df.head(50)['feature'].tolist()
    
    return X_train[top_features], X_test[top_features], importance_df
```

---

### 8. DATA QUALITY CHECKS

```python
def validate_data_quality(df):
    """
    Validaciones de calidad de datos
    """
    
    checks = []
    
    # 1. No hay IDs duplicados
    assert df['SK_ID_CURR'].nunique() == len(df), "IDs duplicados!"
    checks.append("✅ Sin IDs duplicados")
    
    # 2. Target es binario
    assert df['TARGET'].isin([0, 1]).all(), "TARGET debe ser 0 o 1"
    checks.append("✅ TARGET válido")
    
    # 3. Valores negativos en features de edad
    assert (df['DAYS_BIRTH'] < 0).all(), "DAYS_BIRTH debe ser negativo"
    checks.append("✅ DAYS_BIRTH coherente")
    
    # 4. Ingresos positivos
    assert (df['AMT_INCOME_TOTAL'] > 0).all(), "Ingresos deben ser positivos"
    checks.append("✅ Ingresos positivos")
    
    # 5. No hay valores infinitos
    assert not df.select_dtypes(include=['float64']).isin([np.inf, -np.inf]).any().any()
    checks.append("✅ Sin valores infinitos")
    
    # 6. Distribución del target
    target_dist = df['TARGET'].value_counts(normalize=True)
    assert 0.05 < target_dist[1] < 0.20, "Distribución de TARGET sospechosa"
    checks.append(f"✅ TARGET distribuido: {target_dist[1]*100:.1f}% clase 1")
    
    print("
".join(checks))
    return True
```

---

### 9. PLAN DE EJECUCIÓN DE DATOS

#### Fase 1: Setup (Día 1)
```bash
# 1. Descargar dataset
kaggle competitions download -c home-credit-default-risk

# 2. Extraer archivos
unzip home-credit-default-risk.zip -d data/raw/

# 3. Verificar archivos
ls data/raw/
# Debe mostrar: application_train.csv, bureau.csv, etc.
```

#### Fase 2: EDA Inicial (Día 2-3)
```python
# Notebook 01_EDA.ipynb

# 1. Carga inicial
df = pd.read_csv('data/raw/application_train.csv')

# 2. Inspección básica
df.info()
df.describe()
df.head()

# 3. Análisis de target
target_distribution = df['TARGET'].value_counts(normalize=True)

# 4. Missing values
missing_analysis = df.isnull().mean().sort_values(ascending=False)

# 5. Correlaciones
correlation_matrix = df.corr()

# 6. Visualizaciones
# - Distribución de variables numéricas
# - Boxplots por target
# - Heatmap de correlaciones
```

#### Fase 3: Feature Engineering (Día 4-5)
```python
# Notebook 02_Feature_Engineering.ipynb

# 1. Crear variables tradicionales
df = create_traditional_features(df)

# 2. Crear variables alternativas (INNOVACIÓN)
df = create_alternative_features(df)

# 3. Agregar datos de otras tablas (si hay tiempo)
df = merge_bureau_features(df, bureau)

# 4. Guardar dataset procesado
df.to_csv('data/processed/features_train.csv', index=False)

# 5. Crear data dictionary
create_data_dictionary(df, 'data/data_dictionary.csv')
```

---

### 10. DATA DICTIONARY (Template)

Crear durante EDA:

```python
def create_data_dictionary(df, output_path):
    """
    Genera data dictionary automático
    """
    
    data_dict = pd.DataFrame({
        'Variable': df.columns,
        'Type': df.dtypes,
        'Missing_%': df.isnull().mean() * 100,
        'Unique_Values': df.nunique(),
        'Description': ['TODO: Add description'] * len(df.columns),
        'Source': ['Calculated' if col.startswith('FE_') else 'Original' 
                   for col in df.columns]
    })
    
    data_dict.to_csv(output_path, index=False)
    print(f"Data dictionary saved to {output_path}")
    
    return data_dict
```

---

### 11. CONSIDERACIONES ESPECIALES

#### Contexto Argentino (Adaptación)

Aunque los datos no son de Argentina, documentar:

```markdown
## Adaptaciones para Argentina

Si este modelo se implementara con datos argentinos, se incorporarían:

**Datos de Pago de Servicios:**
- Historial de Rapipago/Pago Fácil
- Pagos de servicios (EDESUR, AYSA, Telecom)

**Billeteras Digitales:**
- Historial en Mercado Pago
- Transacciones en Ualá
- Uso de Modo, Personal Pay

**Variables Macroeconómicas:**
- Ajuste por inflación en análisis de ingresos
- Dolarización de ahorros (común en Argentina)
- Resiliencia ante crisis (2018, 2024)

**Verificación de Identidad:**
- Integración con RENAPER
- Consulta a BCRA (veraz argentino)
- Score de identidad digital

**Regulación:**
- Cumplimiento con normativa BCRA
- Protección de datos personales (Ley 25.326)
```

---


**Versión:** 1.0  
**Autor:** Abraham Tartalos  
**Última Actualización:** 07/02/26