"""Entrena y exporta el modelo.

    python scripts/train.py --config config/config.toml
    python scripts/train.py --config config/config.toml --set training.steps=4000
"""
import argparse

from wakeforge import load_config, train


def main():
    ap = argparse.ArgumentParser(description="Entrena un wake word y exporta a ONNX.")
    ap.add_argument("--config", default="config/config.toml", help="ruta al TOML")
    ap.add_argument("--set", action="append", default=[], metavar="seccion.opcion=valor",
                    help="sobrescribe una opción; se puede repetir")
    args = ap.parse_args()
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()