import json
import tempfile
import unittest
from pathlib import Path

import joblib

from pipeline import (
    REQUIRED_COLUMNS,
    entrenar_modelo,
    extraer_datos,
    guardar_artefactos,
    limpiar_datos,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET = PROJECT_DIR / "data" / "titanic.csv"


class PipelineTest(unittest.TestCase):
    def test_extraction_and_cleaning_contract(self):
        datos = extraer_datos(DATASET)
        self.assertTrue(set(REQUIRED_COLUMNS).issubset(datos.columns))
        limpios = limpiar_datos(datos)
        self.assertFalse(limpios.empty)
        self.assertFalse(limpios["Survived"].isna().any())
        self.assertTrue(set(limpios["Survived"].unique()).issubset({0, 1}))

    def test_training_generates_loadable_artifacts(self):
        limpios = limpiar_datos(extraer_datos(DATASET))
        modelo, resultados, metricas = entrenar_modelo(limpios)
        self.assertGreaterEqual(metricas["accuracy"], 0.70)
        self.assertEqual(len(resultados), metricas["test_rows"])

        with tempfile.TemporaryDirectory() as temp_dir:
            rutas = guardar_artefactos(modelo, resultados, metricas, temp_dir)
            self.assertTrue(all(ruta.exists() for ruta in rutas.values()))
            joblib.load(rutas["modelo"])
            metricas_guardadas = json.loads(
                rutas["metricas"].read_text(encoding="utf-8")
            )
            self.assertEqual(metricas_guardadas["accuracy"], metricas["accuracy"])


if __name__ == "__main__":
    unittest.main()
