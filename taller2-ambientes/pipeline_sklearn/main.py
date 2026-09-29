"""Interfaz de línea de comandos para ejecutar cada etapa del pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from pipeline import (
    entrenar_modelo,
    ejecutar_pipeline,
    extraer_datos,
    guardar_artefactos,
    limpiar_datos,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data" / "titanic.csv"
DEFAULT_OUTPUT = BASE_DIR / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline ML reproducible para Titanic")
    parser.add_argument(
        "--stage",
        choices=["extract", "clean", "train", "all"],
        default="all",
        help="Etapa que se desea ejecutar",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV de entrada")
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT), help="Directorio de artefactos"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    salida = Path(args.output_dir)
    salida.mkdir(parents=True, exist_ok=True)
    extraidos_path = salida / "datos_extraidos.csv"
    limpios_path = salida / "datos_limpios.csv"

    if args.stage == "extract":
        datos = extraer_datos(args.data)
        datos.to_csv(extraidos_path, index=False)
        print(f"Extracción validada: {len(datos)} filas -> {extraidos_path}")
        return

    if args.stage == "clean":
        origen = extraidos_path if extraidos_path.exists() else Path(args.data)
        limpios = limpiar_datos(pd.read_csv(origen))
        limpios.to_csv(limpios_path, index=False)
        print(f"Limpieza completada: {len(limpios)} filas -> {limpios_path}")
        return

    if args.stage == "train":
        origen = limpios_path if limpios_path.exists() else Path(args.data)
        limpios = limpiar_datos(pd.read_csv(origen))
        modelo, resultados, metricas = entrenar_modelo(limpios)
        rutas = guardar_artefactos(modelo, resultados, metricas, salida)
        joblib.load(rutas["modelo"])
        json.loads(rutas["metricas"].read_text(encoding="utf-8"))
        print(f"Entrenamiento completado. Accuracy: {metricas['accuracy']:.4f}")
        return

    ejecutar_pipeline(args.data, salida)


if __name__ == "__main__":
    main()
