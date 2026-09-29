# pipeline_sklearn

Pipeline reproducible de scikit-learn (extracción, limpieza, entrenamiento y evaluación) sobre el dataset Titanic.

Las dependencias viven en `requirements.txt` y el empaquetado las lee desde ahí (`pyproject.toml` → `[tool.setuptools.dynamic]`). `MANIFEST.in` garantiza que ese archivo se incluya en el paquete distribuido.

```bash
python -m pip install -r requirements.txt
make pipeline
```
