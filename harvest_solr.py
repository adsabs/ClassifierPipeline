import time
import os
import json

import requests
from adsputils import setup_logging, load_config


solr_config = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__), '.')))


def harvest_solr(bibcodes_list, start_index=0, fields='bibcode, title, abstract'):
    ''' Harvests citations for an input list of bibcodes using the ADS API.

    It will perform minor cleaning of utf-8 control characters.
    Log in output_dir/logs/harvest_clean.log -> tail -f logs/harvest_clean.log .
    bibcodes_list: a list of bibcodes to harvest citations for.
    paths_list:_list: a list of paths to save the output
    start_index (optional): starting index for harvesting
    fields (optional): fields to harvest from the API. Default is 'bibcode, title, abstract'.
    '''

    logger = setup_logging('harvest_clean', proj_home=os.path.dirname('harvest_log.txt'))

    idx=start_index
    step_size = 2000
    # limit attempts to 10
    total_attempts = 10


    logger.info('Start of harvest')
    print('Harvesting titles and abstracts from Solr')

    # loop through list of bibcodes and query solr
    while idx<len(bibcodes_list):

        start_time = time.perf_counter()
        # string to log
        to_log = ''

        attempts = 0
        successful_req = False

        # extract next step_size list
        input_bibcodes = bibcodes_list[idx:idx+step_size]
        bibcodes = 'bibcode\n' + '\n'.join(input_bibcodes)

        # start attempts
        while (not successful_req) and (attempts<total_attempts):
            r_json = None
            r = requests.post(solr_config['API_URL']+'/search/bigquery',
                    params={'q':'*:*', 'wt':'json', 'fq':'{!bitset}', 'fl':fields, 'rows':len(input_bibcodes)},
                              headers={'Authorization': 'Bearer ' + solr_config['API_TOKEN'], "Content-Type": "big-query/csv"},
                              data=bibcodes)

            # check that request worked
            # proceed if r.status_code == 200
            # if fails, log r.text, then repeat for x tries
            if r.status_code==200:
                successful_req=True
            else:
                to_log += 'REQUEST {} FAILED: CODE {}\n'.format(attempts, r.status_code)
                to_log += str(r.text)+'\n'

            # inc count
            attempts+=1

        # after request
        if successful_req:
            #extract json
            r_json = r.json()


            # info to log
            to_log += 'Harvested links up to {}\n'.format(idx)


        # if not successful_req
        else:
            # add to log
            to_log += 'FAILING BIBCODES: {}\n'.format(input_bibcodes)


        # import pdb;pdb.set_trace()

        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print()
        print(f'Harvested bibcodes starting at: {idx}')

        # pause to not go over API rate limit
        if len(bibcodes_list)>step_size:
            time.sleep(45)

        idx+=step_size
        logger.info(to_log)


    return transform_r_json(r_json)


def transform_r_json(r_json):
    """
    Extract the needed information from the json response from the solr query.
    """

    # Bibcoded and titles are always present
    bibcodes = [doc['bibcode'] for doc in r_json['response']['docs']]
    titles = [doc['title'][0] for doc in r_json['response']['docs']] # without [0] it returns a list
    # Abstracts are not always present
    abstracts = []
    for doc in r_json['response']['docs']:
        if 'abstract' in doc:
            abstracts.append(doc['abstract'])
        else:
            abstracts.append('')

    record_list = [{'bibcode': bibcodes[i],
                    'title' : titles[i],
                    'abstract' : abstracts[i],
                    'text': f'{titles[i]} {abstracts[i]}'} for i in range(len(bibcodes))]

    return record_list
