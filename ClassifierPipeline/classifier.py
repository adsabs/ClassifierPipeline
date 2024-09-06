import os
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import no_grad, tensor
from adsputils import load_config, setup_logging
from ClassifierPipeline.astrobert_classification import AstroBERTClassification
# from astrobert_classification import AstroBERTClassification

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
config = load_config(proj_home=proj_home)

logger = setup_logging('classifier.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))

class Classifier:

    def __init__(self):
        self.classifier = AstroBERTClassification()
        self.tokenizer = self.classifier.tokenizer
        self.model = self.classifier.model
        self.labels = self.classifier.labels
        self.id2label = self.classifier.id2label
        self.label2id = self.classifier.label2id

    # split tokenized text into chunks for the model
    def input_ids_splitter(self, input_ids, window_size=510, window_stride=255):
        '''
        Given a list of input_ids (tokenized text ready for a model),
        returns a list with chuncks of window_size, starting and ending with the special tokens (potentially with padding)
        the chuncks will have overlap by window_size-window_stride
        '''
            
        # int() rounds towards zero, so down for positive values
        # import pdb; pdb.set_trace()
        num_splits = max(1, int(len(input_ids)/window_stride))
        
        split_input_ids = [input_ids[i*window_stride:i*window_stride+window_size] for i in range(num_splits)]
        
        
        return(split_input_ids)


    def add_special_tokens_split_input_ids(self, split_input_ids, tokenizer):
        '''
        adds the start [CLS], end [SEP] and padding [PAD] special tokens to the list of split_input_ids
        '''
        
        # add start and end
        split_input_ids_with_tokens = [[tokenizer.cls_token_id]+s+[tokenizer.sep_token_id] for s in split_input_ids]
        
        # add padding to the last one
        split_input_ids_with_tokens[-1] = split_input_ids_with_tokens[-1]+[tokenizer.pad_token_id 
                                                                           for _ in range(len(split_input_ids_with_tokens[0])-len(split_input_ids_with_tokens[-1]))]
        
        return(split_input_ids_with_tokens)

        
    def batch_assign_SciX_categories(self, list_of_texts, score_combiner='max', score_thresholds=None, window_size=510,  window_stride=500):
        '''
        Given a list of texts, assigns SciX categories to each of them.
        Returns two items:
            a list of categories of the form [[cat_1,cat2], ...] (the predicted categories for each text in the input list, texts can be in multiple categories)
            a list of detailed scores of the form [(ast_score, hp_score ...) ...] (the predicted scores for each category for each text in the input list). The scores are in order ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']

        
        Other than the required list of texts, this functions has a number of optional parameters to modify its behavior.
        pretrained_model_name_or_path: defaults to 'adsabs/astroBERT', but can replaced with a path to a different finetuned categorizer
        revision: defaults to 'SCIX-CATEGORIZER' so that huggigface knows which version of astroBERT to download. Probably never needs to be changed.
        score_combiner: Defaults to 'max'. Can be one of: 'max', 'mean', or a custom lambda function that combines a list of scores per category for each sub-sample into one score for the entire text (this is needed when the text is longer than 512 tokens, the max astroBERT can handle).
        score_thresholds: list of thresholds that scores in each category need to surpass for that category to be assigned. Defaults are from testing.
        
        # splitting params, to handle samples longer than 512 tokens.
        window_size = 510
        window_stride = 500    
        '''
        
        logger.info('Starting batch_assign_SciX_categories')
        # import pdb; pdb.set_trace()
        
        # optimal default thresholds based on experimental results
        if score_thresholds is None:
            score_thresholds = [0.0 for _ in range(len(self.labels)) ]

        
        # import pdb; pdb.set_trace()

        logger.info('lists of texts')
        logger.info('List of texts {}'.format(list_of_texts))

        # if any(isinstance(i, list) for i in list_of_texts) is False:
        #     list_of_texts = [list_of_texts]
        
        list_of_texts_tokenized_input_ids = self.tokenizer(list_of_texts, add_special_tokens=False)['input_ids']
        # list_of_texts_tokenized_input_ids = tokenizer(list_of_texts, add_special_tokens=False)['input_ids'][0]
        # import pdb; pdb.set_trace()

        logger.info('Tokenized input ids')
        logger.info('List of texts tokenized input ids {}'.format(list_of_texts_tokenized_input_ids))
        # import pdb; pdb.set_trace()

        
        # split
        list_of_split_input_ids = [self.input_ids_splitter(t, window_size=window_size, window_stride=window_stride) for t in list_of_texts_tokenized_input_ids]
        # Full list of text
        # list_of_split_input_ids = input_ids_splitter(list_of_texts_tokenized_input_ids, window_size=window_size, window_stride=window_stride)

        # import pdb; pdb.set_trace()
        
        logger.info('Split input ids')
        # add special tokens
        list_of_split_input_ids_with_tokens = [self.add_special_tokens_split_input_ids(s, self.tokenizer) for s in list_of_split_input_ids]
        
        logger.info('Split input ids with tokens')
        logger.info('List of split input ids with tokens {}'.format(list_of_split_input_ids_with_tokens))
        
        # list to return
        list_of_categories = []
        list_of_scores = []
        
        # forward call
        with no_grad():
            # for split_input_ids_with_tokens in tqdm(list_of_split_input_ids_with_tokens):
            for split_input_ids_with_tokens in list_of_split_input_ids_with_tokens:
                # make predictions
                logger.info('Making predictions')
                logger.info('Predictions with model {}'.format(self.model))
                # logger.info('split_input_ids_with_tokens {}'.format(split_input_ids_with_tokens))
                # import pdb; pdb.set_trace()
                predictions = self.model(input_ids=tensor(split_input_ids_with_tokens)).logits.sigmoid()

                logger.info('Predictions {}'.format(predictions))
                
                logger.info('COmbining predictions')
                # combine into one prediction
                if score_combiner=='mean':
                    prediction = predictions.mean(dim=0)
                elif score_combiner=='max':
                    prediction = predictions.max(dim=0)[0]
                else:
                    # should be a custom lambda function
                    prediction = score_combiner(predictions)
                

                logger.info('Appending predictions')
                list_of_scores.append(prediction.tolist())
                # filter by scores above score_threshold

                logger.info('Appending categories')
                list_of_categories.append([self.id2label[index] for index,score in enumerate(prediction) if score>=score_thresholds[index]])
        
        logger.info('Ran forward call')
        return(list_of_categories, list_of_scores)
        
    # classifier = AstroBERTClassification()
    # scores = classifier.batch_assign_SciX_categories(text)

