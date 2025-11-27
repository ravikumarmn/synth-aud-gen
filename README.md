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
