#!/usr/bin/env python
"""
"""

# __author__ = 'tsa'
# __maintainer__ = 'tsa'
# __copyright__ = 'Copyright 2024'
# __email__ = 'ads@cfa.harvard.edu'
# __status__ = 'Production'
# __credit__ = ['T. Allen']
# __license__ = 'MIT'

import os
import csv
import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ClassifierPipeline import classifier, tasks
from run import score_record, classify_record_from_scores, prepare_records
from harvest_solr import harvest_solr

# # ============================= INITIALIZATION ==================================== #

from adsputils import setup_logging, load_config
proj_home = os.path.realpath(os.path.dirname(__file__))
global config
config = load_config(proj_home=proj_home)
logger = setup_logging('run.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', False))


# =============================== FUNCTIONS ======================================= #

def write_batch_to_csv(batch, header, filename, mode='w', include_header=True):
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)
        if include_header:
            writer.writerow(header)
        writer.writerows(batch)

# =============================== MAIN ======================================= #

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process user input.')


    parser.add_argument('-r',
                        '--records',
                        dest='records',
                        action='store',
                        help='Path to comma delimited list of new records' +
                             'to process: columns: bibcode, title, abstract')

    args = parser.parse_args()


    if args.records:
        records_path = args.records
        out_path = records_path.replace('.csv', '_classified.csv')
        print(f'Reading in {records_path} may take a minute for large input files.')
        print(f'Will write output to {out_path}.')
    else:
        print("Please provide a path to a .csv file with records to process.")
        exit()

    output_idx = 0
    output_list = []
    output_batch = 500
    header = 'bibcode,title,abstract,categories,scores,collections,collection_scores,earth_science_adjustment'


    with open(records_path, 'r') as f:
        bibcodes = f.read().splitlines()

    # If first line is 'bibcode' remove it
    if bibcodes[0]=='bibcode':
        bibcodes = bibcodes[1:]

    print('Classifying records...')
    while output_idx < len(bibcodes):

        # Harvest Title and Abstract from Solr
        bibcode_batch = bibcodes[output_idx:output_idx+output_batch]
        records = harvest_solr(bibcode_batch, start_index=0, fields='bibcode, title, abstract')

        for index, record in enumerate(records):
            record = score_record(record)
            record = classify_record_from_scores(record)
            del record['model']
            del record['text']

            collection_scores = []
            for collection in record['collections']:

                collection_scores.append(record['scores'][record['categories'].index(collection)]) 

            collection_scores, record['collections'] = zip(*sorted(zip(collection_scores, record['collections']), reverse=True))

            collection_scores = [round(score, 2) for score in collection_scores]
            record['collection_scores'] = collection_scores

            records[index] = record

            record_output = [record['bibcode'],record['title'],record['abstract'],', '.join(record['categories']),', '.join(map(str, record['scores'])),', '.join(record['collections']),', '.join(map(str, record['collection_scores'])),str(record['earth_science_adjustment'])]

            output_list.append(record_output)

        if output_idx == 0:
            include_header = True
            mode = 'w'
        else:
            include_header = False
            mode = 'a'
        write_batch_to_csv(output_list, header.split(','), out_path, mode=mode, include_header=include_header)
        output_list = []
        print(f"Processed {output_idx+index+1} records")

        output_idx += output_batch 


    print("Done")
    print(f"Results saved to {out_path}")
