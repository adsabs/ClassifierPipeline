SQLALCHEMY_URL = ''
SQLALCHEMY_ECHO = False
API_URL = "https://api.adsabs.harvard.edu/v1" # ADS API URL
API_TOKEN = ''
# CLASSIFICATION_PRETRAINED_MODEL = "adsabs/ASTROBERT"
# CLASSIFICATION_PRETRAINED_MODEL = "ClassifierPipeline/tests/models/checkpoint-32100/"
CLASSIFICATION_PRETRAINED_MODEL = "/app/ClassifierPipeline/tests/models/checkpoint-32100/"
CLASSIFICATION_PRETRAINED_MODEL_REVISION = "SciX-Categorizer"
CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER = "adsabs/ASTROBERT"

# Celery configuration
CELERY_INCLUDE = ["classifierpipeline.tasks"]
CELERY_BROKER = "pyamqp://test:test@localhost:5682/classifier_pipeline"

OUTPUT_CELERY_BROKER = "pyamqp://test:test@localhost:5682/master_pipeline" 
OUTPUT_TASKNAME = "adsmp.tasks.task_update_record"
# OUTPUT_TASKNAME = "ClassifierPipeline.tasks.task_handle_input_from_master"


# Thresholds for model checkpoint 32100
# [Astrophysics, Heliophysics, Planetary Science, Earth Science, Biophysics, Other Physics, Other, Garbage]
# [0.06, 0.03, 0.04, 0.02, 0.0, 0.02, 0.02, 0.0]
CLASSIFICATION_THRESHOLDS = [0.06, 0.03, 0.04, 0.02, 0.99, 0.02, 0.02, 0.99]
ASTRONOMY_THRESHOLD_DELTA = 0.06
HELIOPHYSICS_THRESHOLD_DELTA = 0.03
PLANETARY_SCIENCE_THRESHOLD_DELTA = 0.04
EARTH_SCIENCE_THRESHOLD_DELTA = 0.02
BIOPHYSICS_THRESHOLD_DELTA = 0.0
OTHER_PHYSICS_THRESHOLD_DELTA = 0.02
OTHER_THRESHOLD_DELTA = 0.02
GARBAGE_THRESHOLD_DELTA = 0.0

# To enable extra processing for Earth Science
ADDITIONAL_EARTH_SCIENCE_PROCESSING = True
ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD = 0.015

