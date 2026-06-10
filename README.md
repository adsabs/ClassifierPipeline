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
For a SciXID:
```
python run.py -s -x SciXID
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
If called using `--classify` the classifications are indexed immediately.  To allow a curator inspection of the results before indexing use `--classify_verify`.  

### Benchmarking
Production-like throughput profiling is available via:
```
python -m ClassifierPipeline.benchmark run --dataset ClassifierPipeline/tests/stub_data/stub_new_records.csv --mode real --batch-size 100
```

Run the default fixed sweep (`batch_sizes=[50,100,250,500]`, `modes=real,fake`):
```
python -m ClassifierPipeline.benchmark sweep --dataset ClassifierPipeline/tests/stub_data/stub_new_records.csv
```

Compare a candidate run against a baseline:
```
python -m ClassifierPipeline.benchmark compare --baseline path/to/baseline.json --candidate path/to/candidate.json
```

Artifacts:
- Markdown report (human summary)
- JSON metrics payload (for diffing/comparisons)

Environment flags used by workers for profiling:
- `PERF_METRICS_ENABLED=true`
- `PERF_METRICS_PATH=/absolute/path/to/perf_events.jsonl`
- `PERF_FORCE_FAKE_DATA=true|false` (controls fake/real inference mode per worker process)

### Attached Benchmarking

There are now two wrapper layers for benchmark execution:

1. an in-container runner:
```
/app/scripts/run-in-container-benchmark.bash \
  --dataset /app/logs/2023Sci_titles_abstracts_200.csv \
  --model-inference-batch-size 16 \
  --model-num-threads 4 \
  --model-num-interop-threads 1
```

2. an attach-only host wrapper for already-running classifier containers:
```
bash /Users/thomasallen/Code/ADS/ADSIngestPipelineTestEnvironment/run-attached-classifier-benchmark.bash \
  --container classifier_pipeline \
  --model-inference-batch-size 16 \
  --model-num-threads 4 \
  --model-num-interop-threads 1
```

The host wrapper:
- requires an explicit target container
- does not restart or tear down services
- captures host context separately from benchmark runtime metadata
- writes a host-side manifest plus translated host paths for `/app` artifacts when that mount is detectable

