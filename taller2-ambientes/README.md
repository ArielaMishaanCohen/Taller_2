# Taller 2 — Ambientes virtuales de Python

Machine Learning Engineering (CC3105) · Universidad del Valle de Guatemala · Ariela Mishaan

## Estructura

```text
taller2-ambientes/
├── requirements.txt          # Punto 2: ambiente general de ML (pip + venv)
├── environment.yml           # Punto 4: ambiente equivalente para Conda
└── pipeline_sklearn/         # Punto 3: pipeline de scikit-learn empaquetado
    ├── requirements.txt      # Dependencias de ejecución del pipeline
    ├── pyproject.toml        # Empaquetado: lee dependencias de requirements.txt
    ├── MANIFEST.in           # Incluye requirements.txt y datos en el paquete
    ├── main.py · pipeline.py · Makefile
    ├── data/titanic.csv
    └── tests/test_pipeline.py
```

## 1. Documentación consultada

| Fuente | Sección |
|---|---|
| Python | [`venv` — Creation of virtual environments](https://docs.python.org/3/library/venv.html) |
| Python (tutorial) | [12. Virtual Environments and Packages](https://docs.python.org/3/tutorial/venv.html) |
| VS Code | [Python environments in VS Code](https://code.visualstudio.com/docs/python/environments) |
| pip | [Requirements File Format](https://pip.pypa.io/en/stable/reference/requirements-file-format/) |
| setuptools | [Configuring setuptools using pyproject.toml](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) y [Controlling files in the distribution](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html) |
| Conda | [Managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) |

Un ambiente virtual es un directorio con su propio intérprete de Python y su propio `site-packages`. Al activarlo, `python` y `pip` apuntan a ese directorio, de modo que los paquetes que se instalan no afectan al Python del sistema ni a otros proyectos.

## 2. Ambiente virtual con `venv` + `requirements.txt`

Por convención, el archivo de requisitos se llama **`requirements.txt`** y se ubica en la raíz del proyecto.

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Verificar que el ambiente está activo
which python          # debe apuntar a .../taller2-ambientes/.venv/bin/python
python -m pip list

# Congelar las versiones exactas instaladas (opcional, para reproducibilidad total)
python -m pip freeze > requirements.lock.txt

deactivate
```

En Windows la activación es `.venv\Scripts\activate`. En VS Code: `Cmd+Shift+P` → *Python: Create Environment* → *Venv* → seleccionar `requirements.txt`.

`requirements.txt` usa rangos (`>=x,<y`) para aceptar parches y versiones menores sin romper la API mayor. `pip freeze` genera la versión "bloqueada" con `==`, útil para reproducir exactamente un entrenamiento.

## 3. Requisitos del pipeline de scikit-learn y empaquetado (reto)

El pipeline (extracción → limpieza → entrenamiento → evaluación) tiene su propio `pipeline_sklearn/requirements.txt` con solo lo necesario para ejecutarse: `pandas`, `scikit-learn`, `numpy` y `joblib`. Herramientas de exploración como Jupyter o seaborn quedan fuera porque no se usan en producción.

Para incluir ese archivo en el paquete se necesitaron dos configuraciones adicionales:

1. **`pyproject.toml` declara las dependencias como dinámicas** y las lee de `requirements.txt`, así hay una sola fuente de verdad:

   ```toml
   [project]
   dynamic = ["dependencies"]

   [tool.setuptools.dynamic]
   dependencies = { file = ["requirements.txt"] }
   ```

2. **`MANIFEST.in`** indica qué archivos no-Python deben viajar dentro del sdist (`.tar.gz`). Por defecto setuptools solo empaqueta los módulos `.py`; sin `include data/titanic.csv` el dataset no se distribuye, y con un `setup.py` que hace `open("requirements.txt")` la instalación falla con `FileNotFoundError` si ese archivo no está en el manifiesto.

```bash
cd pipeline_sklearn
python -m pip install build
python -m build                                  # genera dist/*.tar.gz y dist/*.whl
tar tzf dist/pipeline_sklearn-1.2.0.tar.gz       # requirements.txt y data/ aparecen listados

# Instalar el paquete en un ambiente limpio y ejecutarlo
python -m pip install dist/pipeline_sklearn-1.2.0.tar.gz
titanic-pipeline --data data/titanic.csv --output-dir artifacts
python -m unittest discover -s tests -v
```

Resultado verificado: el wheel declara `Requires-Dist` para las cuatro dependencias, el sdist contiene `requirements.txt` y `data/titanic.csv`, y el pipeline instalado desde el sdist entrena el modelo con accuracy de 0.8212 y pasa las pruebas.

## 4. Ambiente con Conda + `environment.yml` (reto)

Conda sí permite definir un archivo de requisitos, pero con otro formato: **`environment.yml`** (YAML), no `requirements.txt`.

```bash
# Crear el ambiente a partir del archivo
conda env create -f environment.yml
conda activate taller2-ml

# Verificar
conda env list
which python
python -c "import sklearn, pandas; print(sklearn.__version__, pandas.__version__)"

# Ejecutar el pipeline dentro del ambiente de conda
cd pipeline_sklearn && make pipeline

# Actualizar tras editar environment.yml / exportar
conda env update -f environment.yml --prune
conda env export --from-history > environment.lock.yml

conda deactivate
```

Alternativa por comandos, sin archivo: `conda create -n taller2-ml python=3.12 scikit-learn pandas`.

### Diferencias con el `requirements.txt` del punto 2

| | `requirements.txt` (pip) | `environment.yml` (conda) |
|---|---|---|
| Formato | Texto plano, una dependencia por línea | YAML con `name`, `channels` y `dependencies` |
| Versión de Python | No la controla; usa la del intérprete con que se creó el venv | Se declara como dependencia (`python=3.12`) y conda la instala |
| Origen de paquetes | PyPI | Canales (`conda-forge`, `defaults`) y una subsección `pip:` para lo que solo existe en PyPI |
| Dependencias no-Python | No (compiladores, CUDA, BLAS, GDAL deben existir en el sistema) | Sí, conda instala binarios y librerías del sistema |
| Sintaxis de versión | `==`, `>=`, `~=` | `=` para versión, `>=,<` también válido; `==` para versión exacta |
| Nombre del ambiente | Lo define la carpeta (`.venv`) | Se define en el campo `name` |

También existe `conda list --export > spec.txt` / `conda create --file spec.txt`, que usa un formato de texto plano parecido a `requirements.txt`, pero no admite paquetes de pip ni canales, por eso se prefiere `environment.yml`.

## Conclusiones

- **Aislamiento entre proyectos.** Cada proyecto de ML puede depender de versiones distintas de las mismas librerías (por ejemplo, un modelo entrenado con scikit-learn 1.3 y otro con 1.5). Un ambiente por proyecto evita que actualizar una librería rompa otro proyecto o el Python del sistema.
- **Reproducibilidad de modelos.** Un modelo serializado con `joblib` solo se carga de forma confiable con la misma versión de scikit-learn con la que se entrenó. Congelar el ambiente (`pip freeze`, `conda env export`) es parte del artefacto del modelo, igual que los pesos y las métricas.
- **Separación entre desarrollo y producción.** El `requirements.txt` general incluye Jupyter, seaborn y pytest para experimentar; el del pipeline solo contiene lo que se necesita para entrenar e inferir. Esto reduce el tamaño de imágenes Docker, la superficie de vulnerabilidades y el tiempo de instalación en CI.
- **Base para CI/CD y contenedores.** GitHub Actions y Docker reconstruyen el ambiente desde cero en cada ejecución a partir de estos archivos; si el archivo de requisitos está incompleto, el pipeline falla en el servidor aunque funcione en la computadora del desarrollador.
- **Empaquetado con una sola fuente de verdad.** Leer las dependencias del paquete desde `requirements.txt` evita que `setup.py`/`pyproject.toml` y el archivo de requisitos se desincronicen; el costo es declarar explícitamente en `MANIFEST.in` los archivos no-Python.
- **pip/venv vs. conda.** `venv` es liviano y suficiente cuando todo está en PyPI. Conda conviene cuando el proyecto necesita una versión específica de Python o dependencias del sistema (CUDA para deep learning, librerías geoespaciales, BLAS optimizado), algo común en proyectos de ML que pasan de un prototipo a entrenar en GPU.
- **Colaboración en equipo.** Con el archivo de requisitos en el repositorio, cualquier integrante recrea el mismo ambiente con un solo comando, lo que elimina el "en mi máquina sí funciona".
