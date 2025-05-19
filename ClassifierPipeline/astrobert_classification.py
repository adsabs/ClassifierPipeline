import os

from adsputils import setup_logging, load_config


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TokenClassificationPipeline
import transformers
transformers.logging.set_verbosity_debug()

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))

ALLOWED_CATEGORIES = set(['astronomy', 'planetary science', 'heliophysics', 'earth science', 'physics', 'other physics', 'other'])


proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
config = load_config(proj_home=proj_home)
logger = setup_logging('astrobert_classification.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))

# Define model paths
pretrained_model_name_or_path = config.get('CLASSIFICATION_PRETRAINED_MODEL', None)
revision = config.get('CLASSIFICATION_PRETRAINED_MODEL_REVISION',None)
tokenizer_model_name_or_path = config.get('CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER', None)

# Load model and tokenizer
labels=config['ALLOWED_CATEGORIES']
id2label = {i:c for i,c in enumerate(labels) }
label2id = {v:k for k,v in id2label.items()}


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_model_name_or_path,revision=revision, do_lower_case=False)

# load model
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,revision=revision,num_labels=len(labels),problem_type='multi_label_classification',id2label=id2label,label2id=label2id)

logger.info(f'Loaded model: {pretrained_model_name_or_path}, revision: {revision}, tokenizer: {tokenizer_model_name_or_path}')

class AstroBERTClassification():

    model = model
    tokenizer = tokenizer
    labels = labels
    id2label = id2label
    label2id = label2id

