# 🧪 ANIF - Guía de Pruebas Automatizadas

## 📊 Sistema de Reportes y Resultados

### ✅ **Respuesta a tu pregunta: SÍ, ahora los resultados se guardan automáticamente**

Los resultados de las pruebas se guardan en **múltiples formatos y ubicaciones**:

## 📁 Estructura de Archivos de Resultados

```
proyecto/
├── test_results/           # 💾 Logs detallados de cada ejecución
│   ├── pruebas_de_api_20260211_233000.log
│   ├── pruebas_del_sistema_rag_20260211_233100.log
│   └── pruebas_de_streamlit_20260211_233200.log
│
├── test_reports/           # 📊 Reportes visuales y estructurados
│   ├── test_report_20260211_233000.html      # Reporte visual interactivo
│   ├── test_results_20260211_233000.xml      # Formato JUnit (CI/CD)
│   ├── test_summary_20260211_233000.json     # Datos estructurados
│   ├── coverage_html_20260211_233000/        # Cobertura de código
│   └── index_20260211_233000.html            # Índice de todos los reportes
│
└── .pytest_cache/         # 🗄️ Cache temporal de pytest
    └── v/cache/
```

## 🚀 Cómo Ejecutar Pruebas con Resultados Guardados

### **Ejecución Básica (guarda resultados automáticamente):**
```bash
python run_tests.py --quick
```

### **Ejecución con Reportes Completos:**
```bash
python run_tests.py --reports
```

### **Ejecución Específica:**
```bash
python run_tests.py --api-only --reports     # Solo API con reportes
python run_tests.py --rag-only               # Solo RAG con logs
python run_tests.py --streamlit-only         # Solo Streamlit
```

### **Generar Solo Reportes (sin ejecutar pruebas):**
```bash
python test_reports.py --all                 # Todos los reportes
python test_reports.py --html                # Solo reporte HTML
python test_reports.py --coverage            # Solo cobertura
```

## 📋 Tipos de Resultados Guardados

### 1. **📄 Logs Detallados (.log)**
- **Ubicación:** `test_results/`
- **Contenido:** Output completo, errores, comandos ejecutados
- **Formato:** Texto plano con timestamps
- **Ejemplo:** `pruebas_de_api_20260211_233000.log`

### 2. **🌐 Reportes HTML Interactivos**
- **Ubicación:** `test_reports/`
- **Contenido:** Reporte visual con gráficos y detalles
- **Formato:** HTML navegable
- **Ejemplo:** `test_report_20260211_233000.html`

### 3. **📊 Reportes XML (JUnit)**
- **Ubicación:** `test_reports/`
- **Contenido:** Formato estándar para CI/CD
- **Formato:** XML compatible con Jenkins, GitHub Actions
- **Ejemplo:** `test_results_20260211_233000.xml`

### 4. **📋 Reportes JSON Estructurados**
- **Ubicación:** `test_reports/`
- **Contenido:** Datos para análisis programático
- **Formato:** JSON con métricas detalladas
- **Ejemplo:** `test_summary_20260211_233000.json`

### 5. **📈 Reportes de Cobertura**
- **Ubicación:** `test_reports/coverage_html_*/`
- **Contenido:** Análisis de cobertura de código
- **Formato:** HTML con líneas cubiertas/no cubiertas
- **Ejemplo:** `coverage_html_20260211_233000/index.html`

## 🔍 Cómo Ver los Resultados

### **Ver Logs de Texto:**
```bash
# Ver último resultado de API
ls -la test_results/*api*.log | tail -1 | xargs cat

# Ver todos los resultados del día
ls test_results/*20260211*.log
```

### **Ver Reportes HTML:**
```bash
# Abrir último reporte HTML
start test_reports/test_report_*.html

# Ver índice de reportes
start test_reports/index_*.html
```

### **Analizar Datos JSON:**
```bash
# Ver resumen JSON
cat test_reports/test_summary_*.json | jq .summary
```

## 📊 Ejemplo de Contenido de Resultados

### **Log Detallado (.log):**
```
=== RESULTADO DE PRUEBA ===
Descripción: Pruebas de API
Comando: python -m pytest test_api.py -v
Timestamp: 2026-02-11 23:30:00
Exit Code: 0
Estado: EXITOSO

=== STDOUT ===
test_api.py::test_health_check PASSED
test_api.py::test_chat_valid_query PASSED
...

=== STDERR ===
3 warnings about deprecation
```

### **Resumen JSON:**
```json
{
  "timestamp": "20260211_233000",
  "summary": {
    "total": 5,
    "passed": 4,
    "failed": 1,
    "skipped": 0
  },
  "duration": "10.57s",
  "test_files": ["test_api.py"]
}
```

## ⚙️ Configuración Avanzada

### **Desactivar Guardado (solo consola):**
```bash
python run_tests.py --no-save
```

### **Cambiar Directorio de Reportes:**
```bash
python test_reports.py --dir mi_directorio_reportes
```

### **Ejecutar con Cobertura:**
```bash
python run_tests.py --reports  # Incluye cobertura automáticamente
```

## 🔄 Integración con CI/CD

Los archivos XML generados son compatibles con:
- ✅ **GitHub Actions**
- ✅ **Jenkins**
- ✅ **Azure DevOps**
- ✅ **GitLab CI**

### **Ejemplo GitHub Actions:**
```yaml
- name: Run Tests
  run: python run_tests.py --reports

- name: Publish Test Results
  uses: dorny/test-reporter@v1
  with:
    name: ANIF Tests
    path: test_reports/*.xml
    reporter: java-junit
```

## 📈 Historial y Tendencias

Todos los archivos incluyen timestamps, permitiendo:
- **Comparar resultados** entre ejecuciones
- **Identificar regresiones** en el tiempo
- **Analizar tendencias** de cobertura
- **Mantener historial** completo de pruebas

## 🎯 Resumen

**✅ SÍ, los resultados se guardan automáticamente en:**
1. **Logs detallados** - `test_results/`
2. **Reportes HTML** - `test_reports/`
3. **Datos XML/JSON** - `test_reports/`
4. **Cobertura de código** - `test_reports/coverage_*/`
5. **Índices navegables** - `test_reports/index_*.html`

**🚀 Para ejecutar con guardado completo:**
```bash
python run_tests.py --reports
```
