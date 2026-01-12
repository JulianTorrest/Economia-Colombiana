@echo off
echo ========================================
echo  ANIF - Asistente de Investigacion Economica
echo  Setup automatico
echo ========================================
echo.

echo [1/4] Verificando Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    echo Por favor instala Python 3.8+ desde https://python.org
    pause
    exit /b 1
)

echo.
echo [2/4] Creando entorno virtual...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: No se pudo crear el entorno virtual
    pause
    exit /b 1
)

echo.
echo [3/4] Activando entorno virtual...
call venv\Scripts\activate.bat

echo.
echo [4/4] Instalando dependencias...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: No se pudieron instalar las dependencias
    pause
    exit /b 1
)

echo.
echo ========================================
echo  INSTALACION COMPLETADA!
echo ========================================
echo.
echo Para ejecutar la aplicacion:
echo 1. Activa el entorno virtual: venv\Scripts\activate.bat
echo 2. Ejecuta la aplicacion: streamlit run main.py
echo 3. Abre tu navegador en: http://localhost:8501
echo.
echo No olvides:
echo - Obtener tu API key gratuita de Groq en https://console.groq.com/
echo - Configurar la API key en la aplicacion
echo.
pause
