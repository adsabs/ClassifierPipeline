"""
Classifier Module for SciX Pipeline

This module defines the `Classifier` class that uses a fine-tuned AstroBERT model
to classify scientific text into predefined SciX categories.

Main Responsibilities:
    - Tokenize and preprocess input text
    - Handle long input via sliding window tokenization
    - Add required special tokens and padding
    - Perform batched model inference
    - Aggregate model outputs into category scores and final predictions

Dependencies:
    - torch
    - huggingface tokenizer and model via AstroBERTClassification
    - adsputils for logging and config

Model Assumptions:
    - A multi-label classification model with sigmoid outputs
    - Supports long input splitting and score aggregation
"""
import os
import time
from torch import no_grad, tensor
from adsputils import load_config, setup_logging
from ClassifierPipeline.astrobert_classification import AstroBERTClassification
from ClassifierPipeline import perf_metrics

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
config = load_config(proj_home=proj_home)

logger = setup_logging('classifier.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))

class Classifier:
    """
    Encapsulates logic for SciX category classification using a fine-tuned AstroBERT model.

    Attributes:
        classifier (AstroBERTClassification): Model wrapper class
        tokenizer: Huggingface tokenizer instance
        model: PyTorch model instance
        labels (list[str]): List of category labels
        id2label (dict[int, str]): Mapping from index to label
        label2id (dict[str, int]): Mapping from label to index
    """

    def __init__(self):
        self.classifier = AstroBERTClassification()
        self.tokenizer = self.classifier.tokenizer
        self.model = self.classifier.model
        self.labels = self.classifier.labels
        self.id2label = self.classifier.id2label
        self.label2id = self.classifier.label2id

    # split tokenized text into chunks for the model
    def input_ids_splitter(self, input_ids, window_size=510, window_stride=255):
        """
        Splits a long sequence of token IDs into overlapping chunks.

        Parameters:
            input_ids (list[int]): Token IDs from tokenizer
            window_size (int): Max size of each chunk
            window_stride (int): Overlap between consecutive chunks

        Returns:
            list[list[int]]: List of split token ID chunks
        """
            
        # int() rounds towards zero, so down for positive values
        # import pdb; pdb.set_trace()
        num_splits = max(1, int(len(input_ids)/window_stride))
        
        split_input_ids = [input_ids[i*window_stride:i*window_stride+window_size] for i in range(num_splits)]
        
        
        return(split_input_ids)


    def add_special_tokens_split_input_ids(self, split_input_ids, tokenizer):
        """
        Adds [CLS], [SEP], and [PAD] tokens to split token ID chunks.

        Parameters:
            split_input_ids (list[list[int]]): Chunks of token IDs
            tokenizer: Huggingface tokenizer with special token IDs

        Returns:
            list[list[int]]: Modified chunks with special tokens and padding
        """
        
        # add start and end
        split_input_ids_with_tokens = [[tokenizer.cls_token_id]+s+[tokenizer.sep_token_id] for s in split_input_ids]
        
        # add padding to the last one
        split_input_ids_with_tokens[-1] = split_input_ids_with_tokens[-1]+[tokenizer.pad_token_id 
                                                                           for _ in range(len(split_input_ids_with_tokens[0])-len(split_input_ids_with_tokens[-1]))]
        
        return(split_input_ids_with_tokens)

    def _resolve_model_inference_batch_size(self, requested_size=None):
        if requested_size is not None:
            try:
                requested_value = int(requested_size)
                if requested_value > 0:
                    return requested_value
            except (TypeError, ValueError):
                pass
        try:
            configured_value = int(config.get("MODEL_INFERENCE_BATCH_SIZE", 16) or 16)
            if configured_value > 0:
                return configured_value
        except (TypeError, ValueError):
            pass
        return 16

    def _chunk_list(self, items, batch_size):
        for index in range(0, len(items), batch_size):
            yield items[index:index + batch_size]

        
    def _emit_classifier_shape_metrics(self, run_id, context_id, configured_record_batch_size, shape_metrics):
        for name, value in shape_metrics.items():
            perf_metrics.emit_event(
                stage="classifier_batch_shape",
                run_id=run_id,
                context_id=context_id,
                record_id=None,
                duration_ms=float(value),
                extra={"name": name},
                config=config,
            )

    def batch_score_SciX_categories(
        self,
        list_of_texts,
        score_combiner='max',
        score_thresholds=None,
        window_size=510,
        window_stride=500,
        run_id=None,
        context_id=None,
        configured_record_batch_size=None,
        model_inference_batch_size=None,
    ):
        """
        Classifies each input text into SciX categories using the model.

        Parameters:
            list_of_texts (list[str]): Raw input text to classify
            score_combiner (str or function): Method for aggregating scores across chunks ('max', 'mean', or custom function)
            score_thresholds (list[float]): Threshold per category to include in results
            window_size (int): Token window size for splitting long inputs
            window_stride (int): Token stride for overlapping splits

        Returns:
            tuple:
                list[list[str]]: Predicted categories per input
                list[list[float]]: Raw category scores per input
        """
        
        logger.info(f'Classifying {len(list_of_texts)} records')
        
        # optimal default thresholds based on experimental results
        if score_thresholds is None:
            score_thresholds = [0.0 for _ in range(len(self.labels)) ]

        logger.debug('lists of texts')
        logger.debug('List of texts {}'.format(list_of_texts))

        configured_batch_size = configured_record_batch_size or len(list_of_texts)
        resolved_model_inference_batch_size = self._resolve_model_inference_batch_size(
            requested_size=model_inference_batch_size
        )

        with perf_metrics.timed_profile(
            category="classifier_timing",
            name="tokenizer_call",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            extra={"configured_record_batch_size": configured_batch_size, "record_count": len(list_of_texts)},
            config=config,
        ):
            list_of_texts_tokenized_input_ids = self.tokenizer(list_of_texts, add_special_tokens=False)['input_ids']

        logger.debug('Tokenized input ids')
        logger.debug('List of texts tokenized input ids {}'.format(list_of_texts_tokenized_input_ids))

        with perf_metrics.timed_profile(
            category="classifier_timing",
            name="input_splitting",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            extra={"configured_record_batch_size": configured_batch_size, "record_count": len(list_of_texts)},
            config=config,
        ):
            list_of_split_input_ids = [self.input_ids_splitter(t, window_size=window_size, window_stride=window_stride) for t in list_of_texts_tokenized_input_ids]
        # Full list of text
        # list_of_split_input_ids = input_ids_splitter(list_of_texts_tokenized_input_ids, window_size=window_size, window_stride=window_stride)
        
        logger.debug('Split input ids')
        with perf_metrics.timed_profile(
            category="classifier_timing",
            name="special_token_padding",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            extra={"configured_record_batch_size": configured_batch_size, "record_count": len(list_of_texts)},
            config=config,
        ):
            list_of_split_input_ids_with_tokens = [self.add_special_tokens_split_input_ids(s, self.tokenizer) for s in list_of_split_input_ids]

        logger.debug('Split input ids with tokens')
        logger.debug('List of split input ids with tokens {}'.format(list_of_split_input_ids_with_tokens))

        chunk_counts = [len(split_input_ids) for split_input_ids in list_of_split_input_ids]
        total_chunks = sum(chunk_counts)
        max_chunks = max(chunk_counts) if chunk_counts else 0
        max_tokenized_length = max((len(token_ids) for token_ids in list_of_texts_tokenized_input_ids), default=0)
        padded_tensor_rows = max((len(split_ids) for split_ids in list_of_split_input_ids_with_tokens), default=0)
        padded_tensor_cols = max((len(split_ids[0]) for split_ids in list_of_split_input_ids_with_tokens if split_ids), default=0)
        prepared_records = [
            {
                "original_index": index,
                "split_input_ids_with_tokens": split_input_ids_with_tokens,
            }
            for index, split_input_ids_with_tokens in enumerate(list_of_split_input_ids_with_tokens)
        ]

        micro_batches = list(self._chunk_list(prepared_records, resolved_model_inference_batch_size))
        micro_batch_record_counts = [len(batch) for batch in micro_batches]
        micro_batch_row_counts = [
            sum(len(record["split_input_ids_with_tokens"]) for record in batch)
            for batch in micro_batches
        ]
        self._emit_classifier_shape_metrics(
            run_id,
            context_id,
            configured_batch_size,
            {
                "configured_record_batch_size": configured_batch_size,
                "model_inference_batch_size": resolved_model_inference_batch_size,
                "effective_chunk_batch_size": total_chunks,
                "total_chunks": total_chunks,
                "mean_chunks_per_record": (float(total_chunks) / len(chunk_counts)) if chunk_counts else 0.0,
                "max_chunks_per_record": max_chunks,
                "max_tokenized_length": max_tokenized_length,
                "padded_tensor_rows": padded_tensor_rows,
                "padded_tensor_cols": padded_tensor_cols,
                "micro_batch_count": len(micro_batches),
                "max_micro_batch_records": max(micro_batch_record_counts, default=0),
                "mean_micro_batch_records": (
                    float(sum(micro_batch_record_counts)) / len(micro_batch_record_counts)
                    if micro_batch_record_counts
                    else 0.0
                ),
                "max_micro_batch_rows": max(micro_batch_row_counts, default=0),
                "mean_micro_batch_rows": (
                    float(sum(micro_batch_row_counts)) / len(micro_batch_row_counts)
                    if micro_batch_row_counts
                    else 0.0
                ),
            },
        )
        
        # list to return
        list_of_categories = [None] * len(list_of_texts)
        list_of_scores = [None] * len(list_of_texts)
        model_forward_ms = 0.0
        post_sigmoid_aggregation_ms = 0.0
        
        # forward call
        with no_grad():
            for micro_batch in micro_batches:
                flattened_rows = []
                record_row_counts = []
                for prepared_record in micro_batch:
                    split_input_ids_with_tokens = prepared_record["split_input_ids_with_tokens"]
                    flattened_rows.extend(split_input_ids_with_tokens)
                    record_row_counts.append(
                        (prepared_record["original_index"], len(split_input_ids_with_tokens))
                    )

                logger.debug('Making predictions')
                logger.debug('Predictions with model {}'.format(self.model))
                try:
                    logger.debug('Really making predictions')
                    model_start = time.perf_counter()
                    predictions = self.model(input_ids=tensor(flattened_rows))
                    model_forward_ms += (time.perf_counter() - model_start) * 1000.0
                except Exception as e:
                    logger.exception(f'Failed with: {str(e)}')
                    raise e
                try:
                    logger.debug('Really making predictions - really')
                    post_start = time.perf_counter()
                    predictions = predictions.logits.sigmoid()
                except Exception as e:
                    logger.exception(f'Failed with: {str(e)}')
                    raise e

                logger.debug('Predictions {}'.format(predictions))
                
                logger.debug('COmbining predictions')
                prediction_row_start = 0
                for original_index, row_count in record_row_counts:
                    prediction_rows = predictions[prediction_row_start:prediction_row_start + row_count]
                    prediction_row_start += row_count
                    record_predictions = prediction_rows

                    if score_combiner=='mean':
                        prediction = record_predictions.mean(dim=0)
                    elif score_combiner=='max':
                        prediction = record_predictions.max(dim=0)[0]
                    else:
                        prediction = score_combiner(record_predictions)

                    logger.debug('Assigning predictions')
                    list_of_scores[original_index] = prediction.tolist()
                    list_of_categories[original_index] = [
                        self.id2label[index]
                        for index,score in enumerate(prediction)
                        if score>=score_thresholds[index]
                    ]
                post_sigmoid_aggregation_ms += (time.perf_counter() - post_start) * 1000.0

        perf_metrics.emit_event(
            stage="classifier_timing",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            duration_ms=model_forward_ms,
            extra={"name": "model_forward", "configured_record_batch_size": configured_batch_size, "record_count": len(list_of_texts)},
            config=config,
        )
        perf_metrics.emit_event(
            stage="classifier_timing",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            duration_ms=post_sigmoid_aggregation_ms,
            extra={"name": "post_sigmoid_aggregation", "configured_record_batch_size": configured_batch_size, "record_count": len(list_of_texts)},
            config=config,
        )
        
        logger.debug('Ran forward call')
        return(list_of_categories, list_of_scores)
        
