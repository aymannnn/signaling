# AI Agent Instructions — Signaling Project

Purpose: Help AI coding agents work productively in this Python research repo by documenting architecture, workflows, conventions, and gotchas specific to this codebase.

## Big Picture
- Scope: Single-folder Python project simulating residency matching with preference signaling and analyzing outcomes.
- Flow: `constants.csv` → `simulation.py` → `simulation_results.csv` → `data_analysis.py` → PNG plots.
- Core algorithm: Applicant-proposing deferred acceptance (NRMP-style). Programs rank applicants with signals prioritized over non-signals.
- Quartile model: Applicants and programs are split into 4 equal quartiles. Application targeting uses a 50/25/25 split across quartiles (own/above/below).

## Key Files
- `simulation.py`: Runs the simulation across a signal range. Key pieces:
  - `Applicant.update_signal_number(s)`: sets class-level `n_signals`/`n_non_signals` before each run.
  - `Applicant.pick_programs(...)`: enforces 50/25/25 application distribution and updates program inboxes.
  - `Program.create_final_rank_list_and_count_reviews()`: builds interview lists; always reviews all signals; fills remaining slots with non-signals.
  - `stable_match(applicants, programs)`: applicant-proposing deferred acceptance; respects each program’s final rank list.
  - Writes a combined CSV `simulation_results.csv` with a `Parameter` column and integer signal columns as strings.
- `data_analysis.py`: Reads `simulation_results.csv`, computes means and 95% CIs, and saves plots (`simulation_results.png` and per-parameter PNGs).
- `constants.csv`: Required integer parameters with columns `Variable, Value (Must be INTEGERS), Description` (the `Description` column is dropped at load).
- `docs.txt`: Extended narrative documentation; note that some parts are outdated (e.g., references to per-metric CSVs not currently written).

## Developer Workflows
- Install deps (PowerShell):
  ```powershell
  pip install pandas numpy matplotlib scipy
  ```
- Run simulation (writes `simulation_results.csv`):
  ```powershell
  python simulation.py
  ```
- Run analysis (reads CSV, writes PNGs, shows plots):
  ```powershell
  python data_analysis.py
  ```
- Tuning for speed: Edit `constants.csv` (e.g., lower `simulations_per_s` or narrow `study_min_signal`/`study_max_signal`) to iterate quickly.

## Data Contracts
- `simulation_results.csv` schema:
  - Column `Parameter` with values: `Unmatched_Applicants`, `Unfilled_Spots`, `Reviews_Per_Program` (exact strings).
  - Remaining columns are signal counts (e.g., `1, 2, ..., 20`) stored as strings; `data_analysis.py` casts them to ints for plotting.
- IDs are 0-based. Quartiles are computed by integer ranges on IDs; lower ID = better applicant/program.

## Conventions & Patterns
- Signals always outrank non-signals in program lists; within each group, lower applicant ID is preferred.
- Applicants’ final rank list is `[signaled_programs + non_signaled_programs]`, both sorted ascending by program ID.
- Review counting: programs must review all signals; if signals < interview slots, they review all non-signals to fill.
- Randomness: program picks use `numpy.random.choice`; no seed is set by default.

## Gotchas
- Long runs: With large `simulations_per_s` and ranges, total runs = `simulations_per_s × (#signal values)` and can be very slow.
- `docs.txt` references additional per-metric CSVs; current `simulation.py` only writes the combined `simulation_results.csv`.
- `Results` class in `simulation.py` is unused and has a bad reference (`unfilled_spots` not defined in `__init__`); treat as legacy.
- Infinite loop guard in `Applicant.pick_programs` will error if constraints make unique selection impossible (e.g., too many applications vs. programs per quartile).

## Safe Changes for Agents
- If modifying outputs, keep the `Parameter` values and integer-named signal columns unless you also update `data_analysis.py` accordingly.
- If altering application distribution or ranking rules, update both `Applicant.pick_programs` and `Program.create_final_rank_list_and_count_reviews()` and reflect assumptions in `docs.txt`.
- Prefer editing `constants.csv` to tune scale/experiments rather than hardcoding values.

---
Questions or unclear bits? Notably, confirm whether per-metric CSVs should be restored (as in `docs.txt`) or `simulation_results.csv` is the single source of truth. I can align code/docs based on your preference.
