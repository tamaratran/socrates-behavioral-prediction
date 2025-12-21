# SOCSCI210 Dataset

## Overview

**SOCSCI210** is a standardized, large-scale dataset comprising **2.9 million individual responses** from **400,491 unique participants** across **210 peer-reviewed social science experiments**.

- **Source:** https://huggingface.co/datasets/socratesft/SocSci210
- **Paper:** "Finetuning LLMs for Human Behavior Prediction in Social Science Experiments" (arXiv:2509.05830)
- **Origin:** NSF's Time-sharing Experiments for the Social Sciences (TESS)
- **Downloaded:** December 19, 2025

## Key Statistics

### Participants Extracted
- **Total unique participants:** 6,335
- **Total responses in dataset:** 2,901,390
- **Average responses per participant:** ~458

### Demographics Distribution

**Gender:**
- Female: 41.0% (2,600)
- Male: 35.6% (2,257)
- Unknown: 23.3% (1,478)

**Political Ideology:**
- Moderate: 27.5% (1,742)
- Somewhat Conservative: 12.0% (763)
- Somewhat Liberal: 11.1% (705)
- Very Liberal: 6.7% (427)
- Very Conservative: 5.6% (354)
- Liberal: 5.0% (316)
- Conservative: 4.0% (256)
- Declined to Answer: 2.0% (125)
- Extremely Liberal: 1.5% (94)
- Extremely Conservative: 0.8% (51)
- Unknown: 23.7% (1,502)

## Demographic Fields

Each participant has the following demographic information:

| Field | Type | Example Values |
|-------|------|----------------|
| `participant_id` | int | 3885, 226, 3082 |
| `age` | int | 22, 53, 54 |
| `sex` | str | "Male", "Female" |
| `ethnicity` | str | null (not populated in most cases) |
| `education` | str | "Post grad study/professional degree", "Vocational/tech school" |
| `employment` | str | "Self-employed", "Looking for work", null |
| `income` | str | "50-74K", "125-149K" |
| `marital_status` | str | "Married", "Living with partner" |
| `household_size` | int | 4, 5 |
| `housing_ownership` | str | "Owned or being bought", "Rented for cash" |
| `housing_type` | str | "A one-family house detached from any other house" |
| `location` | str | "Florida", "California" (US states) |
| `metro_status` | str | "Metro Area" |
| `ideology` | str | "Very Conservative", "Somewhat Liberal" |
| `party_id` | str | "Strong Republican", "Strong Democrat" |
| `internet_access` | str | "Internet Household" |
| `phone_service` | str | "Landline telephone only", "Cellphone only" |

## Files Created

### Persona Files (JSON)

| File | Count | Size | Description |
|------|-------|------|-------------|
| `socsci210_all_participants.json` | 6,335 | 3.6 MB | All unique participants |
| `socsci210_personas_100.json` | 100 | 0.1 MB | Stratified sample (100) |
| `socsci210_personas_500.json` | 500 | 0.3 MB | Stratified sample (500) |
| `socsci210_personas_1000.json` | 1,000 | 0.6 MB | Stratified sample (1000) |
| `socsci210_personas_5000.json` | 5,000 | 2.8 MB | Stratified sample (5000) |

### Stratified Sampling Strategy

Personas were sampled using **stratified sampling by gender** to ensure representative demographics:

1. Calculate target personas per gender group
2. Sample proportionally from each group
3. Fill remaining slots with random selection
4. Seed = 42 for reproducibility

## Advantages Over Synthetic Personas (Nemotron)

### SOCSCI210 Real Participants vs Nemotron Synthetic Personas

| Feature | SOCSCI210 | Nemotron |
|---------|-----------|----------|
| **Origin** | Real human participants in peer-reviewed studies | AI-generated synthetic personas |
| **Demographics** | 17 fields (ideology, party_id, income, etc.) | 8 fields (age, sex, city, state, etc.) |
| **Political Data** | Ideology + party affiliation | None |
| **Validation** | Linked to actual human responses | No ground truth responses |
| **Size** | 6,335 unique real people | 10,000 synthetic personas |
| **Use Case** | Behavioral prediction with validation | Demographic simulation only |

**Key Advantage:** SOCSCI210 personas can be **validated against actual human responses** from the same participants in the original experiments.

## Usage

### Loading Personas

```python
import json

# Load 1000-persona sample
with open('data/socsci210_personas_1000.json', 'r') as f:
    personas = json.load(f)

print(f"Loaded {len(personas)} personas")
print(f"First persona: {personas[0]}")
```

### Adapting for Our System

The personas are already in a compatible format, but you may need to map fields:

```python
def adapt_socsci_persona(persona):
    """Convert SOCSCI210 persona to our internal format."""
    return {
        'uuid': f"socsci_{persona['participant_id']}",
        'age': persona['age'],
        'sex': persona['sex'],
        'state': persona['location'],
        'education': persona['education'],
        'income': persona['income'],
        'ideology': persona['ideology'],
        'party_id': persona['party_id'],
        # Add more fields as needed
    }
```

### Running Experiments

```python
# Use with our existing prediction system
from behavioral_simulator import run_behavioral_prediction

results = run_behavioral_prediction(
    personas_file='data/socsci210_personas_1000.json',
    question='Do you approve of Congress?',
    model='gpt-4o-mini'
)
```

## Dataset Structure in Hugging Face

The original SOCSCI210 dataset on Hugging Face contains:

- `sample_id`: Unique response ID
- `participant`: Participant ID (used to extract personas)
- `demographic`: Nested dict with 18 demographic fields
- `stimuli`: Experiment stimuli text (102-4,720 characters)
- `response`: Participant's response (int64, range -5 to 60.1M)
- `condition_num`, `task_num`: Experiment design fields
- `prompt`, `reasoning`: LLM prediction fields
- `study_id`: One of 185 distinct experiment IDs

## Future Work

1. **Extract actual questions from experiments** - Map study_id to specific social science questions
2. **Validate predictions against real responses** - Compare LLM predictions to actual human responses
3. **Identify best experiments for validation** - Select studies most relevant to our poll questions
4. **Build question-specific personas** - Group participants by experiment type

## References

- **Paper:** https://arxiv.org/abs/2509.05830
- **Dataset:** https://huggingface.co/datasets/socratesft/SocSci210
- **Project Site:** https://stanfordhci.github.io/socrates
- **TESS Repository:** https://osf.io/4547c/

## Download Script

The dataset was downloaded using `scripts/download_socsci210.py`:

```bash
python scripts/download_socsci210.py
```

This script:
1. Downloads the full SOCSCI210 dataset from Hugging Face
2. Extracts 6,335 unique participants with demographics
3. Creates stratified samples (100, 500, 1000, 5000)
4. Saves to `data/` directory

---

**Generated:** December 19, 2025
