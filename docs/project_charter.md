# PROJECT CHARTER
## Motor de Micro-préstamos para Trabajadores Informales

---

### 1. INFORMACIÓN DEL PROYECTO

**Nombre del Proyecto:** Motor de Micro-préstamos Inclusivo para Trabajadores Informales

**Autor:** Abraham Tartalos

**Fecha de Inicio:** 05/02/2026

**Duración Estimada:** 3-4 semanas

**Tipo:** Proyecto de Portfolio / Data Science

---

### 2. PROBLEMA DE NEGOCIO

#### Contexto Global
- 1.2 mil millones de personas en el mundo carecen de acceso a servicios financieros formales
- Los trabajadores informales, freelancers y personas sin historial crediticio tradicional son sistemáticamente excluidos del sistema financiero
- Los métodos tradicionales de scoring crediticio dependen exclusivamente de historial bancario, ingresos formales y datos tradicionales

#### Contexto Argentino (Mercado Objetivo)
- **45% de la economía argentina es informal** (aprox. 7 millones de trabajadores)
- **70% de trabajadores informales no tienen acceso a crédito bancario**
- Las fintech argentinas (Ualá, Mercado Pago, Naranja X) están creciendo pero aún hay millones de personas sin acceso
- Oportunidad de mercado: millones de personas solventes pero "invisibles" para el sistema tradicional

#### Declaración del Problema
Las instituciones financieras rechazan automáticamente a personas solventes que podrían pagar préstamos, simplemente porque carecen de historial crediticio tradicional, dejando dinero sobre la mesa y excluyendo a segmentos vulnerables de la población.

---

### 3. OBJETIVOS DEL PROYECTO

#### Objetivo Principal
Desarrollar un sistema de scoring crediticio inclusivo que utilice **datos alternativos** (no tradicionales) para evaluar la solvencia de trabajadores informales, aumentando la aprobación de préstamos sin incrementar el riesgo de default.

#### Objetivos Específicos

**Objetivos Técnicos:**
1. Construir un modelo de clasificación (aprobar/rechazar préstamo) con ROC-AUC ≥ 0.75
2. Implementar feature engineering creativo usando datos alternativos (pagos de servicios, patrones de ahorro, estabilidad de ingresos irregulares)
3. Garantizar explicabilidad del modelo usando SHAP values (requisito regulatorio en fintech)
4. Desarrollar un dashboard interactivo para simulación de solicitudes

**Objetivos de Negocio:**
1. Aumentar la tasa de aprobación de préstamos en **15%+** sin incrementar la tasa de default
2. Cuantificar el impacto económico (ingresos adicionales para la institución)
3. Cuantificar el impacto social (personas adicionales con acceso a crédito)
4. Demostrar viabilidad de modelo inclusivo vs. modelo tradicional

**Objetivos de Portfolio:**
1. Demostrar habilidades avanzadas de feature engineering
2. Mostrar pensamiento de negocio (impacto cuantificable)
3. Evidenciar conocimiento del ecosistema fintech argentino
4. Destacar compromiso con inclusión financiera

---

### 4. ALCANCE DEL PROYECTO

#### Dentro del Alcance (In-Scope)
- ✅ Análisis exploratorio de datos (EDA) completo
- ✅ Feature engineering de variables tradicionales y alternativas
- ✅ Desarrollo y comparación de múltiples algoritmos ML
- ✅ Modelo baseline (solo variables tradicionales)
- ✅ Modelo inclusivo (variables tradicionales + alternativas)
- ✅ Análisis de explicabilidad (SHAP values)
- ✅ Cuantificación de impacto (negocio + social)
- ✅ Dashboard interactivo en Streamlit
- ✅ Documentación completa (README, notebooks comentados)
- ✅ Deployment en Streamlit Cloud

#### Fuera del Alcance (Out-of-Scope)
- ❌ Recolección de datos reales de Argentina
- ❌ Integración con APIs bancarias reales
- ❌ Sistema de aprobación en tiempo real con infraestructura productiva
- ❌ Análisis de segmentación de clientes avanzado
- ❌ Motor de recomendación de montos óptimos (puede ser fase 2)
- ❌ Sistema de cobranza o recuperación de cartera

---

### 5. STAKEHOLDERS Y BENEFICIARIOS

#### Stakeholders Primarios
- **Instituciones financieras:** Bancos, fintech, cooperativas de crédito que buscan expandir su base de clientes
- **Trabajadores informales:** Beneficiarios directos con acceso a micro-créditos

#### Stakeholders Secundarios
- **Reguladores financieros:** BCRA (Banco Central de Argentina) interesado en inclusión financiera
- **Organizaciones de inclusión social:** ONGs enfocadas en reducción de pobreza

#### Audiencia del Portfolio
- **Reclutadores técnicos:** Data Scientists, ML Engineers
- **Hiring Managers:** Líderes de equipos de data/fintech
- **Founders de fintech:** Startups argentinas buscando talento

---

### 6. ENTREGABLES DEL PROYECTO

#### Entregables Técnicos
1. **Código fuente:**
   - Notebooks Jupyter (EDA + Modelado)
   - Scripts Python modulares
   - Aplicación Streamlit

2. **Modelos entrenados:**
   - Modelo baseline (variables tradicionales)
   - Modelo inclusivo (variables tradicionales + alternativas)
   - Archivos pickle de modelos finales

3. **Dashboard interactivo:**
   - Simulador de solicitudes
   - Análisis comparativo de modelos
   - Casos de éxito (storytelling)

#### Entregables de Documentación
1. **README profesional** (GitHub)
2. **Notebooks comentados** con análisis detallado
3. **Presentación ejecutiva** (5-7 slides)
4. **Data Dictionary** (creado durante EDA)
5. **Reporte de impacto** (cuantificación de resultados)

#### Entregables Opcionales
- Video demo (2-3 minutos)
- Artículo técnico en Medium/LinkedIn
- Case study detallado

---

### 7. MÉTRICAS DE ÉXITO

#### Métricas Técnicas (Modelo)
| Métrica | Target | Justificación |
|---------|--------|---------------|
| ROC-AUC | ≥ 0.75 | Estándar industria para scoring crediticio |
| Precision | ≥ 0.70 | Minimizar falsos positivos (aprobar malos pagadores) |
| Recall | ≥ 0.65 | Maximizar verdaderos positivos (aprobar buenos pagadores) |
| Feature Importance | Top 10 claras | Explicabilidad regulatoria |

#### Métricas de Negocio
| Métrica | Target | Cálculo |
|---------|--------|---------|
| Incremento en aprobaciones | +15% | (Aprobados inclusivo - Aprobados baseline) / Aprobados baseline |
| Tasa de default mantenida | ≤ 5% | Default rate modelo inclusivo ≤ Default rate baseline |
| Ingreso adicional anual | Cuantificado | Nuevas aprobaciones × préstamo promedio × tasa interés |
| Personas con nuevo acceso | Cuantificado | Extrapolación al mercado argentino |

#### Métricas de Portfolio
- ⭐ Stars en GitHub: >10
- 👁️ Vistas en LinkedIn: >500
- 💼 Menciones en entrevistas: Proyecto destacado
- 📧 Contactos de reclutadores: Al menos 1

---

### 8. ESTRATEGIA DE DATOS

#### Dataset Principal
**Home Credit Default Risk (Kaggle)**
- **Razón:** Dataset de calidad profesional con 300k+ registros de personas con historial crediticio limitado
- **Ventaja:** Incluye múltiples tablas relacionadas (bureau, historial previo, pagos)
- **Limitación:** Datos no son de Argentina, pero los principios son transferibles

#### Transparencia sobre Datos
- Documentar explícitamente que datos son internacionales
- Contextualizar problema en Argentina (45% informalidad)
- Proponer adaptaciones para datos argentinos (Rapipago, MercadoPago, etc.)

#### Variables Clave a Desarrollar

**Variables Tradicionales:**
- Edad, género, educación
- Ingresos reportados
- Tipo de empleo
- Deuda actual

**Variables Alternativas (Feature Engineering):**
- **Estabilidad de ingresos irregulares:**
  - Varianza de ingresos mensuales
  - Tendencia (creciente/decreciente)
  - Piso de ingresos (mínimo últimos 6 meses)

- **Historial de pagos alternativos:**
  - Puntualidad en servicios (luz, agua, teléfono)
  - Meses consecutivos sin atrasos
  - Ratio pagos a tiempo / total pagos

- **Patrones de ahorro:**
  - Existencia de ahorros (sí/no)
  - Consistencia en ahorro mensual
  - Ratio ahorro/ingreso

- **Arraigo y estabilidad:**
  - Antigüedad en domicilio
  - Antigüedad en actividad económica
  - Referencias disponibles

---

### 9. TECNOLOGÍAS Y HERRAMIENTAS

#### Lenguaje y Librerías Core
- **Python 3.11+**
- **Pandas, NumPy:** Manipulación de datos
- **Scikit-learn:** Modelos baseline
- **XGBoost/LightGBM:** Modelos avanzados
- **Imbalanced-learn:** Manejo de desbalance de clases

#### Visualización y Explicabilidad
- **Matplotlib, Seaborn, Plotly:** Visualizaciones
- **SHAP:** Explicabilidad de modelos
- **Streamlit:** Dashboard interactivo

#### Deployment y Versionado
- **Git/GitHub:** Control de versiones
- **Streamlit Cloud:** Hosting gratuito
- **Jupyter Notebooks:** Análisis y documentación

---

### 10. CRONOGRAMA (3-4 semanas)

| Fase | Actividades | Duración |
|------|-------------|----------|
| **Fase 1:** Preparación | Definición, investigación, setup de datos | 2-3 días |
| **Fase 2:** EDA + Feature Engineering | Limpieza, análisis, creación de variables | 3-4 días |
| **Fase 3:** Modelado | Baseline, modelos avanzados, tuning | 4-5 días |
| **Fase 4:** Explicabilidad | SHAP, análisis de impacto, storytelling | 2-3 días |
| **Fase 5:** Dashboard | Desarrollo en Streamlit, deployment | 3-4 días |
| **Fase 6:** Documentación | README, presentación, pulido final | 2 días |

**Total:** 16-21 días (~3-4 semanas trabajando 3-4 horas/día)

---

### 11. RIESGOS Y SUPUESTOS

#### Riesgos Identificados

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Dataset no tiene variables necesarias | Media | Alto | Simular variables alternativas con lógica de negocio |
| Desbalance extremo de clases | Alta | Medio | SMOTE, ajuste de pesos, métricas apropiadas |
| Modelo no mejora sobre baseline | Baja | Alto | Iterar en feature engineering, probar múltiples algoritmos |
| Dashboard no deploya correctamente | Baja | Bajo | Testear localmente, usar Streamlit Cloud (probado) |
| Falta de tiempo | Media | Medio | Priorizar MVP, features opcionales al final |

#### Supuestos
1. Dataset Home Credit es representativo del problema
2. Patrones de comportamiento crediticio son transferibles entre países
3. Variables simuladas son razonables y basadas en lógica de negocio
4. 3-4 horas/día disponibles para el proyecto
5. Streamlit Cloud mantiene servicio gratuito

---

### 12. CRITERIOS DE ÉXITO FINAL

El proyecto será considerado **exitoso** si cumple:

✅ **Técnicamente:**
- Modelo funcional con ROC-AUC ≥ 0.75
- Explicabilidad implementada (SHAP)
- Dashboard deployado y accesible públicamente

✅ **De Negocio:**
- Incremento ≥15% en aprobaciones sin aumentar default
- Impacto cuantificado (económico + social)
- Comparación clara baseline vs. inclusivo

✅ **De Portfolio:**
- Repositorio GitHub profesional
- README atractivo con resultados destacados
- Al menos 1 reclutador/contacto interesado
- Proyecto presentable en entrevistas

---

**Aprobación del Proyecto:**
- **Autor:** Abraham Tartalos
- **Fecha:** 12/02/26
- **Versión:** 1.0

---

*Este Project Charter es un documento vivo y puede actualizarse según descubrimientos durante el desarrollo.*