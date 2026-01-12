@echo off
echo ========================================
echo  ANIF - Asistente de Investigacion Economica
echo  Iniciando aplicacion...
echo ========================================
echo.

echo Activando entorno virtual...
call venv\Scripts\activate.bat

echo.
echo Iniciando Streamlit...
echo La aplicacion se abrira en tu navegador automaticamente
echo URL: http://localhost:8501
echo.
echo Para detener la aplicacion, presiona Ctrl+C
echo.

streamlit run main.py
