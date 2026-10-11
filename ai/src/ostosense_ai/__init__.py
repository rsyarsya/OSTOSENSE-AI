"""Sensor processing, ordinal classification, and application interfaces.

Core processing and exported-model inference use the standard library.
Training/evaluation require the optional pipeline extra; HTTP serving requires
serve. The bundled model is trained on synthetic fixtures and its real-sensor
integration mode remains unvalidated. See docs/architecture.md and docs/research.md.
"""

__version__ = "0.2.0"

__all__: list[str] = []
