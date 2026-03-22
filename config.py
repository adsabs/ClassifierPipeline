SQLALCHEMY_URL = ''
SQLALCHEMY_ECHO = False
API_URL = "https://api.adsabs.harvard.edu/v1" # ADS API URL
API_TOKEN = ''
CLASSIFICATION_PRETRAINED_MODEL = "adsabs/ASTROBERT"
CLASSIFICATION_PRETRAINED_MODEL_REVISION = "SciX-Categorizer"
# CLASSIFICATION_PRETRAINED_MODEL_REVISION = "/app/ClassifierPipeline/tests/models/checkpoint-32100/"
CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER = "adsabs/ASTROBERT"

# Celery configuration
CELERY_INCLUDE = ["ClassifierPipeline.tasks"]
CELERY_BROKER = "pyamqp://test:test@localhost:5682/classifier_pipeline"

OUTPUT_CELERY_BROKER = "pyamqp://test:test@localhost:5682/master_pipeline"
OUTPUT_TASKNAME = "adsmp.tasks.task_update_record"

# set to True adds .delay() or .apply_async() to the end of each task
# set to False for direct function calls
DELAY_MESSAGE = True

# Return fake data instead of running the model for testing purposes
FAKE_DATA = False

# Batching configuration
# Number of records grouped into one classify task payload.
CLASSIFY_STAGE_BATCH_SIZE = 100
# Number of real records sent into one classifier preprocessing/inference call.
CLASSIFIER_PRE_FORWARD_BATCH_SIZE = 100
# Number of prepared records grouped into one model forward micro-batch.
MODEL_INFERENCE_BATCH_SIZE = 32

#Data to Skip message from Master Pipeline
TEST_INPUT_DATA = 'ClassifierPipeline/tests/stub_data/classifier_request.json'

ALLOWED_CATEGORIES = ['astrophysics', 'heliophysics', 'planetary', 'earthscience', 'NASA-funded Biophysics', 'physics', 'general', 'Text Garbage']
# Thresholds for model checkpoint 32100
# [Astrophysics, Heliophysics, Planetary Science, Earth Science, Biophysics, Other Physics, Other, Garbage]
CLASSIFICATION_THRESHOLDS = [0.06, 0.03, 0.04, 0.02, 0.99, 0.02, 0.02, 0.99]
ASTRONOMY_THRESHOLD_DELTA = 0.06
HELIOPHYSICS_THRESHOLD_DELTA = 0.03
PLANETARY_SCIENCE_THRESHOLD_DELTA = 0.04
EARTH_SCIENCE_THRESHOLD_DELTA = 0.02
BIOPHYSICS_THRESHOLD_DELTA = 0.0
OTHER_PHYSICS_THRESHOLD_DELTA = 0.02
OTHER_THRESHOLD_DELTA = 0.02
GARBAGE_THRESHOLD_DELTA = 0.0

OPERATION_STEP = 'classify_verify'
# To enable extra processing for Earth Science
ADDITIONAL_EARTH_SCIENCE_PROCESSING = False
ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD = 0.015

# Benchmark / profiling configuration
PERF_METRICS_ENABLED = False
PERF_METRICS_PATH = ""
PERF_METRICS_OUTPUT_DIR = "logs/benchmarks"
PERF_P95_REGRESSION_LIMIT_PCT = 10.0
PERF_MIN_THROUGHPUT_IMPROVEMENT_PCT = 5.0
