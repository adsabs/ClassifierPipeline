import importlib
import sys

import pytest


class DummyLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


@pytest.fixture
def dummy_logger():
    return DummyLogger()


@pytest.fixture
def base_fake_config():
    return {
        "LOGGING_LEVEL": "INFO",
        "LOG_STDOUT": False,
        "ALLOWED_CATEGORIES": [
            "Astronomy",
            "Heliophysics",
            "Planetary Science",
            "Earth Science",
            "NASA-funded Biophysics",
            "Other Physics",
            "Other",
            "Text Garbage",
        ],
        "CLASSIFICATION_THRESHOLDS": [0.6, 0.4, 0.8, 0.5, 0.7, 0.3, 0.2, 0.9],
        "ADDITIONAL_EARTH_SCIENCE_PROCESSING": "active",
        "ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD": 0.55,
        "CLASSIFICATION_PRETRAINED_MODEL": "model-x",
        "CLASSIFICATION_PRETRAINED_MODEL_REVISION": "rev-1",
        "CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER": "tok-1",
        "DELAY_MESSAGE": False,
        "FAKE_DATA": False,
        "OPERATION_STEP": "classify_verify",
        "TEST_INPUT_DATA": "ClassifierPipeline/tests/stub_data/classifier_request.json",
    }


@pytest.fixture
def clear_module():
    def _clear(module_name):
        sys.modules.pop(module_name, None)

    return _clear


@pytest.fixture
def import_fresh(clear_module):
    def _import(module_name):
        clear_module(module_name)
        return importlib.import_module(module_name)

    return _import
