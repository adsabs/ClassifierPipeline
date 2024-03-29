
API_URL = "https://api.adsabs.harvard.edu/v1" # ADS API URL


CLASSIFICATION_PRETRAINED_MODEL = "ClassifierPipeline/tests/models/checkpoint-32100/"
CLASSIFICATION_PRETRAINED_MODEL_REVISION = "SciX-Categorizer"
CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER = "adsabs/ASTROBERT"

CLASSIFICATION_THRESHOLDS = [0.06, 0.03, 0.04, 0.02, 0.99, 0.02, 0.02, 0.99]
ADDITIONAL_EARTH_SCIENCE_PROCESSING = True
ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD = 0.015
