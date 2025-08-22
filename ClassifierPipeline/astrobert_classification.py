"""
AstroBERT Classification Loader

This module sets up a fine-tuned AstroBERT model and tokenizer for multi-label
classification into SciX scientific categories. It loads a model from Hugging Face
or local cache using parameters defined in the configuration.

Loaded components are made available via the `AstroBERTClassification` class,
which stores:
    - tokenizer
    - model
    - labels
    - id2label and label2id dictionaries

This setup supports classification pipelines that leverage Hugging Face Transformers.

Configuration:
    - CLASSIFICATION_PRETRAINED_MODEL
    - CLASSIFICATION_PRETRAINED_MODEL_REVISION
    - CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER
    - ALLOWED_CATEGORIES
"""
import os

from adsputils import setup_logging, load_config


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TokenClassificationPipeline
import transformers
transformers.logging.set_verbosity_debug()

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
    """
    Static wrapper class that holds model, tokenizer, and label metadata
    for AstroBERT classification.

    Attributes:
        model (transformers.PreTrainedModel): Multi-label classification model
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model
        labels (list[str]): Allowed SciX category labels
        id2label (dict[int, str]): Mapping from index to label
        label2id (dict[str, int]): Mapping from label to index
    """
    model = model
    tokenizer = tokenizer
    labels = labels
    id2label = id2label
    label2id = label2id

