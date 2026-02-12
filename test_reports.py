#!/usr/bin/env python3
"""
Sistema de reportes y logging para las pruebas de ANIF
Genera reportes HTML, XML, JSON y logs detallados
"""

import json
import os
import datetime
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Any

class TestReporter:
    """Generador de reportes de pruebas"""
    
    def __init__(self, reports_dir: str = "test_reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_html_report(self, test_files: List[str] = None) -> str:
        """Genera reporte HTML con pytest-html"""
        if test_files is None:
            test_files = ["test_api.py", "test_rag_system.py", "test_streamlit_app.py"]
        
        html_file = self.reports_dir / f"test_report_{self.timestamp}.html"
        
        cmd = [
            sys.executable, "-m", "pytest",
            "--html", str(html_file),
            "--self-contained-html",
            "-v",
            "--tb=short"
        ] + test_files
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Guardar también el output en texto
            self.save_text_output(result, "html_generation")
            
            return str(html_file) if html_file.exists() else None
            
        except subprocess.TimeoutExpired:
            print("⏰ Timeout generando reporte HTML")
            return None
        except Exception as e:
            print(f"❌ Error generando reporte HTML: {e}")
            return None
    
    def generate_xml_report(self, test_files: List[str] = None) -> str:
        """Genera reporte XML (JUnit format)"""
        if test_files is None:
            test_files = ["test_api.py", "test_rag_system.py", "test_streamlit_app.py"]
        
        xml_file = self.reports_dir / f"test_results_{self.timestamp}.xml"
        
        cmd = [
            sys.executable, "-m", "pytest",
            "--junitxml", str(xml_file),
            "-v",
            "--tb=short"
        ] + test_files
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Guardar también el output en texto
            self.save_text_output(result, "xml_generation")
            
            return str(xml_file) if xml_file.exists() else None
            
        except subprocess.TimeoutExpired:
            print("⏰ Timeout generando reporte XML")
            return None
        except Exception as e:
            print(f"❌ Error generando reporte XML: {e}")
            return None
    
    def generate_json_report(self, test_files: List[str] = None) -> str:
        """Genera reporte JSON personalizado"""
        if test_files is None:
            test_files = ["test_api.py", "test_rag_system.py", "test_streamlit_app.py"]
        
        json_file = self.reports_dir / f"test_summary_{self.timestamp}.json"
        
        cmd = [
            sys.executable, "-m", "pytest",
            "--json-report",
            "--json-report-file", str(json_file),
            "-v"
        ] + test_files
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Si pytest-json-report no está disponible, crear reporte manual
            if result.returncode != 0 and "json-report" in result.stderr:
                return self.create_manual_json_report(test_files)
            
            return str(json_file) if json_file.exists() else None
            
        except Exception as e:
            print(f"⚠️ pytest-json-report no disponible, creando reporte manual")
            return self.create_manual_json_report(test_files)
    
    def create_manual_json_report(self, test_files: List[str]) -> str:
        """Crea reporte JSON manual ejecutando pytest y parseando output"""
        json_file = self.reports_dir / f"test_summary_{self.timestamp}.json"
        
        cmd = [
            sys.executable, "-m", "pytest",
            "-v", "--tb=short"
        ] + test_files
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Parsear output de pytest
            report_data = {
                "timestamp": self.timestamp,
                "command": " ".join(cmd),
                "exit_code": result.returncode,
                "duration": "unknown",
                "summary": {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "errors": 0
                },
                "test_files": test_files,
                "output": result.stdout,
                "errors": result.stderr
            }
            
            # Parsear resultados del output
            if result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "passed" in line and "failed" in line:
                        # Línea de resumen como "1 failed, 4 passed in 10.57s"
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "passed":
                                report_data["summary"]["passed"] = int(parts[i-1])
                            elif part == "failed":
                                report_data["summary"]["failed"] = int(parts[i-1])
                            elif part == "skipped":
                                report_data["summary"]["skipped"] = int(parts[i-1])
                            elif part.endswith("s") and "in" in parts[i-1:i+1]:
                                report_data["duration"] = part
                        
                        report_data["summary"]["total"] = (
                            report_data["summary"]["passed"] + 
                            report_data["summary"]["failed"] + 
                            report_data["summary"]["skipped"]
                        )
                        break
            
            # Guardar JSON
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            return str(json_file)
            
        except Exception as e:
            print(f"❌ Error creando reporte JSON manual: {e}")
            return None
    
    def save_text_output(self, result: subprocess.CompletedProcess, operation: str):
        """Guarda el output de texto de un comando"""
        log_file = self.reports_dir / f"{operation}_{self.timestamp}.log"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {operation.upper()} LOG ===\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"Exit Code: {result.returncode}\n")
            f.write(f"Command: {' '.join(result.args) if hasattr(result, 'args') else 'N/A'}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
    
    def generate_coverage_report(self) -> str:
        """Genera reporte de cobertura de código"""
        coverage_file = self.reports_dir / f"coverage_{self.timestamp}.html"
        
        # Instalar coverage si no está disponible
        try:
            subprocess.run([sys.executable, "-m", "coverage", "--version"], 
                         capture_output=True, check=True)
        except subprocess.CalledProcessError:
            print("📦 Instalando coverage...")
            subprocess.run([sys.executable, "-m", "pip", "install", "coverage"])
        
        try:
            # Ejecutar tests con coverage
            cmd_run = [
                sys.executable, "-m", "coverage", "run", "-m", "pytest",
                "test_api.py", "test_rag_system.py", "test_streamlit_app.py"
            ]
            
            result_run = subprocess.run(cmd_run, capture_output=True, text=True, timeout=300)
            
            # Generar reporte HTML
            cmd_html = [
                sys.executable, "-m", "coverage", "html",
                "-d", str(self.reports_dir / f"coverage_html_{self.timestamp}")
            ]
            
            result_html = subprocess.run(cmd_html, capture_output=True, text=True)
            
            # Guardar logs
            self.save_text_output(result_run, "coverage_run")
            self.save_text_output(result_html, "coverage_html")
            
            coverage_dir = self.reports_dir / f"coverage_html_{self.timestamp}"
            index_file = coverage_dir / "index.html"
            
            return str(index_file) if index_file.exists() else None
            
        except Exception as e:
            print(f"❌ Error generando reporte de cobertura: {e}")
            return None
    
    def generate_all_reports(self) -> Dict[str, str]:
        """Genera todos los tipos de reportes"""
        print(f"📊 Generando reportes de pruebas - {self.timestamp}")
        print("=" * 60)
        
        reports = {}
        
        # Reporte HTML
        print("🌐 Generando reporte HTML...")
        html_report = self.generate_html_report()
        if html_report:
            reports["html"] = html_report
            print(f"✅ Reporte HTML: {html_report}")
        else:
            print("❌ Error generando reporte HTML")
        
        # Reporte XML
        print("\n📄 Generando reporte XML...")
        xml_report = self.generate_xml_report()
        if xml_report:
            reports["xml"] = xml_report
            print(f"✅ Reporte XML: {xml_report}")
        else:
            print("❌ Error generando reporte XML")
        
        # Reporte JSON
        print("\n📋 Generando reporte JSON...")
        json_report = self.generate_json_report()
        if json_report:
            reports["json"] = json_report
            print(f"✅ Reporte JSON: {json_report}")
        else:
            print("❌ Error generando reporte JSON")
        
        # Reporte de cobertura
        print("\n📈 Generando reporte de cobertura...")
        coverage_report = self.generate_coverage_report()
        if coverage_report:
            reports["coverage"] = coverage_report
            print(f"✅ Reporte de cobertura: {coverage_report}")
        else:
            print("❌ Error generando reporte de cobertura")
        
        # Crear índice de reportes
        self.create_reports_index(reports)
        
        print(f"\n📁 Todos los reportes guardados en: {self.reports_dir}")
        return reports
    
    def create_reports_index(self, reports: Dict[str, str]):
        """Crea un archivo índice con enlaces a todos los reportes"""
        index_file = self.reports_dir / f"index_{self.timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ANIF - Reportes de Pruebas {self.timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2a5298; color: white; padding: 20px; border-radius: 5px; }}
                .report-link {{ display: block; padding: 10px; margin: 10px 0; 
                              background: #f0f2f6; border-radius: 5px; text-decoration: none; }}
                .report-link:hover {{ background: #e8f4fd; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏛️ ANIF - Reportes de Pruebas</h1>
                <p class="timestamp">Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>📊 Reportes Disponibles</h2>
        """
        
        for report_type, report_path in reports.items():
            if report_path:
                relative_path = Path(report_path).name
                html_content += f"""
                <a href="{relative_path}" class="report-link">
                    <strong>{report_type.upper()}</strong><br>
                    <small>{relative_path}</small>
                </a>
                """
        
        html_content += """
            <h2>📁 Estructura de Archivos</h2>
            <ul>
                <li><strong>HTML:</strong> Reporte visual interactivo</li>
                <li><strong>XML:</strong> Formato JUnit para CI/CD</li>
                <li><strong>JSON:</strong> Datos estructurados para análisis</li>
                <li><strong>Coverage:</strong> Cobertura de código</li>
                <li><strong>Logs:</strong> Archivos .log con detalles de ejecución</li>
            </ul>
        </body>
        </html>
        """
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📋 Índice de reportes: {index_file}")

def main():
    """Función principal para generar reportes"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generar reportes de pruebas ANIF")
    parser.add_argument("--html", action="store_true", help="Solo reporte HTML")
    parser.add_argument("--xml", action="store_true", help="Solo reporte XML")
    parser.add_argument("--json", action="store_true", help="Solo reporte JSON")
    parser.add_argument("--coverage", action="store_true", help="Solo reporte de cobertura")
    parser.add_argument("--all", action="store_true", help="Todos los reportes")
    parser.add_argument("--dir", default="test_reports", help="Directorio de reportes")
    
    args = parser.parse_args()
    
    reporter = TestReporter(args.dir)
    
    if args.all or not any([args.html, args.xml, args.json, args.coverage]):
        # Generar todos los reportes por defecto
        reporter.generate_all_reports()
    else:
        # Generar reportes específicos
        if args.html:
            reporter.generate_html_report()
        if args.xml:
            reporter.generate_xml_report()
        if args.json:
            reporter.generate_json_report()
        if args.coverage:
            reporter.generate_coverage_report()

if __name__ == "__main__":
    main()
