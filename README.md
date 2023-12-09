## Configure

```
python3.10 -m venv ~/.venv/skimmer
source ~/.venv/skimmer/bin/activate
pip install -r requirements.txt
```

```
export OPENAI_API_KEY='...'
```

## Demo
This will run the clause-level summary-based scorer, and open an HTML temp file in a browser to show color-coded clauses (red=remove / don't highlight, green=keep / highlight).

``` 
python -m skimmer.summary_matching_abridger dev-data/gutenberg-pg3300-excerpt02-trunc.txt  --method clause-summary-matching
```

## Locally test service
```
uvicorn api.fastapi_service:app --reload
```

```
curl -X POST "http://localhost:8000/score" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"text\":\"Stocks fell today. Tech stocks led the sell-off, falling 5%. A spokesperson for Microsoft said not to worry. This person reminded us that stocks go up and down all the time. Nevertheless, Microsoft fell 10% on news that bad stuff might happen.\"}"
```

```
curl -X POST "http://localhost:8000/abridge" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"text\":\"Stocks fell today. Tech stocks led the sell-off, falling 5%. A spokesperson for Microsoft said not to worry. This person reminded us that stocks go up and down all the time. Nevertheless, Microsoft fell 10% on news that bad stuff might happen.\", \"keep\": 0.5}"
```

