# NFL Defensive Playcaller Assistant

CS 153 final project. Recommends a defensive call for a given 3rd/4th-down situation by combining a Random Forest policy layer (minimizes estimated offensive EPA) with an LLM coordinator that adjusts the recommendation based on injury and roster context.

## Project structure

```
defense_asst.ipynb          # end-to-end pipeline: ingestion → features → RF policy → LLM coordinator
src/
  context/injury_context.py # InjuryContext builder (gsis_id joins, nfl_data_py)
  llm/coordinator.py        # LLM coordinator with constrained output schema
app/
  inference.py              # loads exported model artifacts; scoring functions
  streamlit_app.py          # interactive demo UI
data/
  pbp_3rd4th_2024_2025.csv  # filtered play-by-play (3rd/4th conversion attempts)
  context_cache/            # cached nfl_data_py tables (injuries, rosters, depth charts)
models/                     # exported RF model + candidate set (created by export step)
```

## Setup

### 1. Install dependencies

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Or, if using Anaconda:

```powershell
pip install nfl_data_py joblib pyarrow python-dotenv openai streamlit
```

### 2. Set your API key

Create a `.env` file in the project root:

```
MODEL_API_KEY=your_key_here
```

The LLM coordinator calls a DigitalOcean inference endpoint (OpenAI-compatible). The key is loaded via `python-dotenv`.

### 3. Export model artifacts from the notebook

Open `defense_asst.ipynb` and run all cells through §3. Then run this export cell (add it at the end of §3):

```python
import os, json, joblib
os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/rf_policy.joblib")
candidates.to_parquet("models/candidate_set.parquet", index=False)
test_df.to_parquet("data/test_scenarios.parquet", index=False)

json.dump({
    "ALL_FEATURES": ALL_FEATURES,
    "NUMERIC_STATE_FEATURES": NUMERIC_STATE_FEATURES,
    "ALL_CATEGORICAL": ALL_CATEGORICAL,
    "ACTION_COMPONENTS": ACTION_COMPONENTS,
    "CANDIDATE_KEYS": CANDIDATE_KEYS,
}, open("models/feature_cols.json", "w"), indent=2)

pbp_sorted[[
    "game_id", "defteam", "play_id",
    "def_personnel", "defense_man_zone_type", "defense_coverage_type",
]].to_parquet("data/pbp_history.parquet", index=False)

print("Artifacts exported.")
```

This only needs to be run once (or again if you retrain the model).

## Running the demo app

From the project root:

```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

Or if Streamlit is installed in your global/Anaconda environment:

```powershell
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

### How to use it

1. **Select a game** from the dropdown (2025 weeks 16+ test set).
2. **Select a play** — filtered to 3rd/4th-down conversion attempts in that game.
3. Click **Analyze ▶**.

The app runs three steps in sequence:

- **RF policy scoring** (<100 ms) — scores all candidate defensive configurations by predicted offensive EPA, applies the λ=0.033 predictability penalty for repeated schemes.
- **Injury context** — joins `nfl_data_py` cached tables (injuries, inactives, depth charts) to identify material absences on the opposing offense.
- **LLM coordinator** (~2–5 s) — either affirms the RF pick or re-ranks within the same candidate set when a starter is out.

Results shown: RF candidate bar chart, injury context (material absences highlighted), LLM recommendation with rationale, and the actual defensive call + EPA outcome from the game.

## Running the notebook

Open `defense_asst.ipynb` in Jupyter or VS Code and run cells sequentially. Section markers:

- **§0** — imports and setup  
- **§1** — data ingestion and schema (Weeks 1–2)  
- **§2** — feature engineering and history features (Week 3)  
- **§3** — RF policy engine, evaluation, model card (Weeks 4–5)  
- **§4** — qualitative layer: InjuryContext + LLM coordinator (Week 6)

## AI usage

- **Gemini** — project planning and writing the initial planning document (`cs153_defensive_assistant_plan.md`).
- **Claude Code** — implementation: data ingestion pipeline, feature engineering, RF policy engine, InjuryContext builder, LLM coordinator, and the Streamlit demo UI.
