from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import no_grad, tensor


# split tokenized text into chunks for the model
def input_ids_splitter(input_ids, window_size=510, window_stride=255):
    '''
    Given a list of input_ids (tokenized text ready for a model),
    returns a list with chuncks of window_size, starting and ending with the special tokens (potentially with padding)
    the chuncks will have overlap by window_size-window_stride
    '''
        
    # int() rounds towards zero, so down for positive values
    num_splits = max(1, int(len(input_ids)/window_stride))
    
    split_input_ids = [input_ids[i*window_stride:i*window_stride+window_size] for i in range(num_splits)]
    
    
    return(split_input_ids)


def add_special_tokens_split_input_ids(split_input_ids, tokenizer):
    '''
    adds the start [CLS], end [SEP] and padding [PAD] special tokens to the list of split_input_ids
    '''
    
    # add start and end
    split_input_ids_with_tokens = [[tokenizer.cls_token_id]+s+[tokenizer.sep_token_id] for s in split_input_ids]
    
    # add padding to the last one
    split_input_ids_with_tokens[-1] = split_input_ids_with_tokens[-1]+[tokenizer.pad_token_id 
                                                                       for _ in range(len(split_input_ids_with_tokens[0])-len(split_input_ids_with_tokens[-1]))]
    
    return(split_input_ids_with_tokens)

    
def batch_assign_SciX_categories(list_of_texts, tokenizer, model,labels,id2label,label2id, score_combiner='max', score_thresholds=None, window_size=510,  window_stride=500):
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
    
    
    # optimal default thresholds based on experimental results
    if score_thresholds is None:
        score_thresholds = [0.0 for _ in range(len(labels)) ]

    
    # import pdb; pdb.set_trace()
    
    list_of_texts_tokenized_input_ids = tokenizer(list_of_texts, add_special_tokens=False)['input_ids']
    # list_of_texts_tokenized_input_ids = tokenizer(list_of_texts, add_special_tokens=False)['input_ids'][0]
    
    # split
    list_of_split_input_ids = [input_ids_splitter(t, window_size=window_size, window_stride=window_stride) for t in list_of_texts_tokenized_input_ids]
    
    # add special tokens
    list_of_split_input_ids_with_tokens = [add_special_tokens_split_input_ids(s, tokenizer) for s in list_of_split_input_ids]
    
    
    # list to return
    list_of_categories = []
    list_of_scores = []
    
    # forward call
    with no_grad():
        # for split_input_ids_with_tokens in tqdm(list_of_split_input_ids_with_tokens):
        for split_input_ids_with_tokens in list_of_split_input_ids_with_tokens:
            # make predictions
            predictions = model(input_ids=tensor(split_input_ids_with_tokens)).logits.sigmoid()
            
            # combine into one prediction
            if score_combiner=='mean':
                prediction = predictions.mean(dim=0)
            elif score_combiner=='max':
                prediction = predictions.max(dim=0)[0]
            else:
                # should be a custom lambda function
                prediction = score_combiner(predictions)
            
            list_of_scores.append(prediction.tolist())
            # filter by scores above score_threshold
            list_of_categories.append([id2label[index] for index,score in enumerate(prediction) if score>=score_thresholds[index]])
    
    return(list_of_categories, list_of_scores)
    

