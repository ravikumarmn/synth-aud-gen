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
