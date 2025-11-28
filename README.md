# synth-aud-gen

Does duplications of slots happens compulasary?

Yes, duplications happen when the sample size exceeds the number of unique attribute combinations.

**Why it happens:**

* You have **48 unique combinations** (from the Cartesian product of all variable options)
* But sample sizes are **250** and **100**
* To fill 250 slots with only 48 unique combinations, each combination must repeat based on its proportional allocation

**Example:**

* If a combination has 10% probability and sample_size is 250 → it gets ~25 slots (same attributes repeated 25 times)

**Is it compulsory?**

* **Yes** , if `sample_size > unique_combinations`
* **No** , if `sample_size ≤ unique_combinations` (each slot would be unique)

In your case with 48 combinations and 250/100 sample sizes, duplication is unavoidable to match the sample size.

python generate_audience_characteristics.py -i data/personas_input_10.json -o data/audience_output_10.json


## With File:

curl -X POST "http://localhost:8000/generate/file?max_concurrent=10"
  -F "file=@data/persona_input.json"

curl -X POST "http://localhost:8000/generate/file?max_concurrent=10" -F "file=@data/personas_input_10.json"

curl -X POST "http://localhost:8000/generate/file?max_concurrent=10"
  -F "file=@personas_input_10.json"

## Without File

curl -X POST http://localhost:8000/generate
  -H "Content-Type: application/json"
  -d '{
    "projectName": "Test Project",
    "audiences": [{
      "persona": {
        "personaName": "Tech Enthusiast",
        "about": "Loves technology and gadgets"
      },
      "screenerQuestions": [
        {"question": "Do you use smartphones?", "answer": "Yes, daily"}
      ],
      "sampleSize": 2
    }]
  }'

## Azure Blob Storage Integration

Upload generated audience characteristics to Azure Blob Storage and update the persona file URL via the middleware API.

### Setup

1. Set environment variables (see `.env.example`):
   ```bash
   export AZURE_STORAGE_ACCOUNT_NAME=your-storage-account
   export AZURE_STORAGE_ACCOUNT_KEY=your-storage-key
   export AZURE_STORAGE_CONTAINER_NAME=audience-characteristics
   export MIDDLEWARE_API_URL=https://sample-agument-middleware-dev.azurewebsites.net
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Command Line

```bash
# Upload file and update persona URL
python azure_blob_storage.py data/audience_characteristics_small.json 28

# With custom options
python azure_blob_storage.py data/audience_characteristics_small.json 28 \
  --blob-name custom_name.json \
  --container my-container \
  --expiry-days 180
```

#### Python API

```python
from azure_blob_storage import (
    AzureBlobStorageClient,
    upload_and_update_persona,
    update_persona_file_url,
)

# Full workflow: upload and update API
result = upload_and_update_persona(
    file_path="data/audience_characteristics_small.json",
    project_id=28,
)
print(result)

# Or use the client directly
client = AzureBlobStorageClient()

# Upload and get SAS URL
upload_result = client.upload_and_get_sas_url(
    file_path="data/audience_characteristics_small.json",
    expiry_days=365,
)
print(upload_result["sas_url"])

# Update persona URL via API
api_result = update_persona_file_url(
    project_id=28,
    sas_url=upload_result["sas_url"],
)
```
