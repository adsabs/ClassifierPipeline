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
from torch import no_grad, tensor
from adsputils import load_config, setup_logging
from ClassifierPipeline.astrobert_classification import AstroBERTClassification

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

        
    def batch_score_SciX_categories(self, list_of_texts, score_combiner='max', score_thresholds=None, window_size=510,  window_stride=500):
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
        
        list_of_texts_tokenized_input_ids = self.tokenizer(list_of_texts, add_special_tokens=False)['input_ids']

        logger.debug('Tokenized input ids')
        logger.debug('List of texts tokenized input ids {}'.format(list_of_texts_tokenized_input_ids))

        
        # split
        list_of_split_input_ids = [self.input_ids_splitter(t, window_size=window_size, window_stride=window_stride) for t in list_of_texts_tokenized_input_ids]
        # Full list of text
        # list_of_split_input_ids = input_ids_splitter(list_of_texts_tokenized_input_ids, window_size=window_size, window_stride=window_stride)
        
        logger.debug('Split input ids')
        # add special tokens
        list_of_split_input_ids_with_tokens = [self.add_special_tokens_split_input_ids(s, self.tokenizer) for s in list_of_split_input_ids]
        
        logger.debug('Split input ids with tokens')
        logger.debug('List of split input ids with tokens {}'.format(list_of_split_input_ids_with_tokens))
        
        # list to return
        list_of_categories = []
        list_of_scores = []
        
        # forward call
        with no_grad():
            # for split_input_ids_with_tokens in tqdm(list_of_split_input_ids_with_tokens):
            for split_input_ids_with_tokens in list_of_split_input_ids_with_tokens:
                # make predictions
                logger.debug('Making predictions')
                logger.debug('Predictions with model {}'.format(self.model))
                try:
                    logger.debug('Really making predictions')
                    predictions = self.model(input_ids=tensor(split_input_ids_with_tokens))
                except Exception as e:
                    logger.exception(f'Failed with: {str(e)}')
                    raise e
                try:
                    logger.debug('Really making predictions - really')
                    predictions = predictions.logits.sigmoid()
                except Exception as e:
                    logger.exception(f'Failed with: {str(e)}')

                logger.debug('Predictions {}'.format(predictions))
                
                logger.debug('COmbining predictions')
                # combine into one prediction
                if score_combiner=='mean':
                    prediction = predictions.mean(dim=0)
                elif score_combiner=='max':
                    prediction = predictions.max(dim=0)[0]
                else:
                    # should be a custom lambda function
                    prediction = score_combiner(predictions)
                

                logger.debug('Appending predictions')
                list_of_scores.append(prediction.tolist())
                # filter by scores above score_threshold

                logger.debug('Appending categories')
                list_of_categories.append([self.id2label[index] for index,score in enumerate(prediction) if score>=score_thresholds[index]])
        
        logger.debug('Ran forward call')
        return(list_of_categories, list_of_scores)
        

