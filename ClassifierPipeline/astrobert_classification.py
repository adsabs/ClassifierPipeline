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
import torch

from adsputils import setup_logging, load_config


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers
transformers.logging.set_verbosity_warning()

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
config = load_config(proj_home=proj_home)
logger = setup_logging('astrobert_classification.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))


def _as_int(value):
    try:
        if value in (None, ""):
            return None
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _as_bool_string(value, default="false"):
    if value in (None, ""):
        return default
    return "true" if str(value).strip().lower() in {"1", "true", "yes", "on"} else "false"


def _resolve_runtime_metadata():
    requested_device = str(config.get("MODEL_DEVICE", "cpu") or "cpu").strip().lower()
    resolved_device = requested_device or "cpu"

    tokenizer_parallelism = _as_bool_string(config.get("TOKENIZERS_PARALLELISM"), default="false")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", tokenizer_parallelism)

    requested_num_threads = _as_int(config.get("MODEL_NUM_THREADS"))
    requested_num_interop_threads = _as_int(config.get("MODEL_NUM_INTEROP_THREADS"))

    if requested_num_threads is not None:
        os.environ.setdefault("OMP_NUM_THREADS", str(requested_num_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(requested_num_threads))
        if hasattr(torch, "set_num_threads"):
            try:
                torch.set_num_threads(requested_num_threads)
            except RuntimeError as exc:
                logger.warning(f"Unable to set torch num threads to {requested_num_threads}: {exc}")

    if requested_num_interop_threads is not None and hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(requested_num_interop_threads)
        except RuntimeError as exc:
            logger.warning(f"Unable to set torch interop threads to {requested_num_interop_threads}: {exc}")

    if resolved_device.startswith("cuda"):
        cuda_available = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
        if not cuda_available:
            logger.warning("MODEL_DEVICE=%s requested but CUDA is unavailable; falling back to cpu", requested_device)
            resolved_device = "cpu"

    return {
        "device": resolved_device,
        "requested_device": requested_device,
        "torch_num_threads": getattr(torch, "get_num_threads", lambda: None)(),
        "torch_num_interop_threads": getattr(torch, "get_num_interop_threads", lambda: None)(),
        "tokenizer_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", tokenizer_parallelism),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


runtime_metadata = _resolve_runtime_metadata()

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
if hasattr(model, "to"):
    model = model.to(runtime_metadata["device"])
if hasattr(model, "eval"):
    model.eval()

logger.info(f'Loaded model: {pretrained_model_name_or_path}, revision: {revision}, tokenizer: {tokenizer_model_name_or_path}')
logger.info(
    "Inference runtime configured: device=%s torch_num_threads=%s torch_num_interop_threads=%s tokenizer_parallelism=%s",
    runtime_metadata["device"],
    runtime_metadata["torch_num_threads"],
    runtime_metadata["torch_num_interop_threads"],
    runtime_metadata["tokenizer_parallelism"],
)


def get_runtime_metadata():
    return dict(runtime_metadata)

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
    runtime_metadata = runtime_metadata
