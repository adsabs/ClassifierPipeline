## Classifier Pipeline

### Short Summary

This pipeline assigns articles to Collections based on the following criteria: LLM prediction, journal provenance, and citations.  Only the first criteria is implemented currently.

This pipeline typically recieves records from the Master Pipeline and returns them to the master pipeline. 

### Required Software
```
- RabbitMQ and PostgreSQL
```

### To Run  

To run the Classifier Pipeline directory given a `.csv` input file with columns `bibcode`, `title`, and `abstract`:
```
python run.py -n -r path/to/file.csv
```

To manually verify a set of classified records:
```
python run.py -v -r path/to/file.csv
```

To resend a record or a set of records to the Master Pipeline use the following:
For a bibcode:
```
python run.py -s -b bibcode
```
For a SciX ID:
```
python run.py -s -x SciX ID
```
For a Classification batch:
```
python run.py -s -i run_id
```

### Running from the Master Pipeline
If calling the classifier from the Master Pipeline:
```
python run.py --classify --manual -n path/to/file.csv
```
If called using `--classify` the classifications are indexed immediately.  TO allow a curator inspection of the results before indexing use `--classify_verify`.  
