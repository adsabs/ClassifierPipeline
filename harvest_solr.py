import time
import os
import json

import requests
from adsputils import setup_logging, load_config
# from tqdm import tqdm
# import ptree

#from ..utilities import remove_control_chars
# import remove_control_chars as rcc

# load API config
#solr_config = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__), '../../')))
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

    # save the log next to the bibcode files
    logger = setup_logging('harvest_clean', proj_home=os.path.dirname('harvest_log.txt'))

    # COnvert list of bibcodes to a set for comparison later

    out_path = 'data/'

    # starting index of harvesting
    idx=start_index

    # params for stats
    # number of bibcodes to harvest at once
    step_size = 2000
    # step_size = 20

    harvested_records = []
    batch_failures = []

    logger.info('Start of harvest')
    #start progress bar
    # pbar = tqdm(total=len(bibcodes_list), initial=start_index)
    # print('checkpoint harvest_solr.py')
    # import pdb;pdb.set_trace()

    # loop through list of bibcodes and query solr
    while idx<len(bibcodes_list):

        start_time = time.perf_counter()
        # string to log
        to_log = ''

        # limit attempts to 10
        attempts = 0
        successful_req = False
        last_error = None

        # extract next step_size list
        input_bibcodes = bibcodes_list[idx:idx+step_size]
        bibcodes = 'bibcode\n' + '\n'.join(input_bibcodes)

        # start attempts
        while (not successful_req) and (attempts<10):
            r_json = None
            try:
                r = requests.post(
                    solr_config['API_URL'] + '/search/bigquery',
                    params={'q': '*:*', 'wt': 'json', 'fq': '{!bitset}', 'fl': fields, 'rows': len(input_bibcodes)},
                    headers={'Authorization': 'Bearer ' + solr_config['API_TOKEN'], "Content-Type": "big-query/csv"},
                    data=bibcodes,
                    timeout=60,
                )
            except requests.RequestException as exc:
                last_error = 'REQUEST {} FAILED WITH EXCEPTION: {}'.format(attempts, exc)
                to_log += last_error + '\n'
                attempts += 1
                continue

            # check that request worked
            # proceed if r.status_code == 200
            # if fails, log r.text, then repeat for x tries
            if r.status_code==200:
                try:
                    r_json = r.json()
                except ValueError as exc:
                    last_error = 'REQUEST {} RETURNED INVALID JSON: {}'.format(attempts, exc)
                    to_log += last_error + '\n'
                    to_log += str(r.text) + '\n'
                    attempts += 1
                    continue

                docs = ((r_json.get('response') or {}).get('docs')) if isinstance(r_json, dict) else None
                if docs is None:
                    last_error = 'REQUEST {} RETURNED NO response.docs FIELD'.format(attempts)
                    to_log += last_error + '\n'
                    to_log += json.dumps(r_json)[:2000] + '\n'
                elif len(docs) == 0:
                    last_error = 'REQUEST {} RETURNED ZERO DOCS'.format(attempts)
                    to_log += last_error + '\n'
                    to_log += json.dumps(r_json)[:2000] + '\n'
                else:
                    successful_req=True
            else:
                last_error = 'REQUEST {} FAILED: CODE {}'.format(attempts, r.status_code)
                to_log += last_error + '\n'
                to_log += str(r.text)+'\n'

            # inc count
            attempts+=1

        # after request
        if successful_req:
            #extract json
            r_json = r.json()
            harvested_records.extend(transform_r_json(r_json))

            # add to stat counts
            # astronomy_count += len(r_json['response']['docs'])

            # info to log
            to_log += 'Harvested links up to {}\n'.format(idx)
            # to_log += 'Running astronomy count={}, body count={}, ack count={}\n'.format(astronomy_count,body_count,ack_count)


        # if not successful_req
        else:
            # add to log
            to_log += 'FAILING BIBCODES: {}\n'.format(input_bibcodes)
            batch_failures.append({
                'start_index': idx,
                'count': len(input_bibcodes),
                'last_error': last_error,
            })

#             # raise error
#             r.raise_for_status()

        print()
        print('index')
        print(idx)
        # import pdb;pdb.set_trace()

        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print()
        print()
        print('index')
        print(idx)
        print()
        print('Time for loop segment')
        print(total_time)

        # with open('data/abstracts/test.txt','w') as f:
        #     f.write("test")

        # import pdb;pdb.set_trace()

        # pause to not go over API rate limit
        if len(bibcodes_list)>step_size:
            time.sleep(45)

        # ALWAYS DO:
        # increment counter for next batch
        idx+=step_size
        # update progress bar
        # pbar.update(step_size)
        #update log
        logger.info(to_log)

    # print('checkpoint harvest_solr.py')
    # import pdb;pdb.set_trace()

    if batch_failures and not harvested_records:
        failure = batch_failures[0]
        raise RuntimeError(
            'Harvest failed for all batches. '
            'First failed batch start_index={} count={} error={}'.format(
                failure['start_index'],
                failure['count'],
                failure['last_error'],
            )
        )

    if batch_failures:
        logger.warning('Harvest completed with failed batches: {}'.format(batch_failures))

    if bibcodes_list and not harvested_records:
        raise RuntimeError('Harvest completed without errors but returned zero records.')

    return harvested_records


def transform_r_json(r_json):
    """
    Extract the needed information from the json response from the solr query.
    """
    if not r_json:
        return []

    response = r_json.get('response') or {}
    docs = response.get('docs') or []
    if not docs:
        return []

    # extract the needed information
    bibcodes = [doc.get('bibcode') for doc in docs]
    titles = []
    abstracts = []
    for doc in docs:
        title = doc.get('title') or ['']
        if isinstance(title, list):
            title = title[0] if title else ''
        abstract = doc.get('abstract') or ''
        titles.append(title)
        abstracts.append(abstract)

    # list of dictionaries with the bibcode, title, and abstract for each record
    record_list = [{'bibcode': bibcodes[i],
                    'title' : titles[i],
                    'abstract' : abstracts[i],
                    'text': f'{titles[i]} {abstracts[i]}'} for i in range(len(bibcodes))]

    # return bibcodes, titles, abstracts
    return record_list
