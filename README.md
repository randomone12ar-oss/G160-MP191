# Tiffin Relay Delivery Time Prediction

**SCSE3040 MLOps Mini-Project**  
**Group ID:** G160  
**Problem ID:** MP191  
**Client:** Kochi Dabba Relay Union  
**Batch:** 79 | **Faculty:** Ravi Pratap  

---

## 1. Project Objective

The Dabba Relay Union of Kochi operates a relay delivery network carrying home-cooked tiffins from residential kitchens to office desks across Kochi. To minimize delays, optimize runner allocations on congested routes, and provide accurate arrival expectations to customers, this project develops an end-to-end MLOps pipeline to predict `delivery_min` (the total delivery duration in minutes from pick-up to delivery desk).

This is a regression task evaluated on **Mean Absolute Error (MAE)**.

---

## 2. Group Members & Roles

| Enrolment No. | Name | Role Assignment |
| :--- | :--- | :--- |
| **S24CSEU0869** | DAMMATIPRANEETHVISHNU | Data Lead (DVC, Validation, Cleaning) |
| **S24CSEU0702** | ARYANDANGI | Model Lead & Acting Data Lead (Architecture, Training) |
| **S24CSEU0784** | GaganTiwari | Service Lead (FastAPI, Schemas) |
| **S24CSEU0155** | VaibhavChaudhary | Ops Lead (CI/CD, Containerization) |
| **S24CSEU2151** | YuvrajSingh | Quality & Documentation Lead |

---

## 3. Project Status & Roadmap

- [x] **Stage 1 (Completed):** Data inspection, verification, DVC tracking, data quality validation (7 detectors), data cleaning pipeline, and `reports/findings.yaml` population.
- [ ] **Stage 2 (Upcoming):** Feature engineering, model training (3+ configurations), MLflow experiment tracking, champion model registration, and automated unit testing.
- [ ] **Stage 3 (Upcoming):** FastAPI serving (`POST /predict`, `GET /health`), Pydantic validation rules, structured JSON logging, Docker containerization, Kubernetes manifests, and GitHub Actions CI.
- [ ] **Stage 4 (Upcoming):** Drift monitoring (PSI on `later.csv`), route zone MAE analysis, chronological retraining (v2), model promotion evaluation, and holdout set predictions.

---

## 4. Repository Architecture (Stage 1)

```
G160-MP191/
├── .dvc/
│   ├── config                  # DVC configuration (analytics disabled)
│   ├── .gitignore              # DVC internal ignore rules
│   └── .dvcignore              # DVC ignore patterns
├── data/
│   └── raw/
│       ├── .gitignore          # Git ignores raw CSVs tracked by DVC
│       ├── train.csv           # DVC-managed training dataset (1,000 rows)
│       ├── train.csv.dvc       # DVC pointer file with train.csv hash
│       ├── later.csv           # DVC-managed production dataset (300 rows)
│       ├── later.csv.dvc       # DVC pointer file with later.csv hash
│       └── holdout_features.csv# Git-tracked holdout feature set (250 rows)
├── reports/
│   └── findings.yaml           # Authoritative findings report
├── src/
│   └── mp191/
│       ├── __init__.py         # Package initializer
│       ├── data_validation.py  # Executable detectors for 7 data quality issues
│       └── data_cleaning.py    # Reusable cleaning transformations & integrity check
├── .gitignore                  # Git ignore rules for caches, venvs, and logs
├── GROUP.md                    # Group member roster
├── params.yaml                 # Configuration parameters
├── PROBLEM.md                  # Problem specification
├── README.md                   # Project documentation
└── STUDENT_HANDBOOK.md         # Course requirements and rubric
```

---

## 5. Dataset Descriptions

The raw datasets are located in `data/raw/` and their cryptographic SHA-256 hashes are strictly verified:

| File | Rows & Cols | Purpose | SHA-256 Checksum | Tracking |
| :--- | :--- | :--- | :--- | :--- |
| `data/raw/train.csv` | 1,000 × 11 | Historical relay records (Jan–Jun 2026). Contains deliberate real-world data issues. | `e93ccf33c9481256feec9ddd6c095016597672095b80852755ee1dca6eb2e672` | DVC |
| `data/raw/later.csv` | 300 × 11 | Post-deployment records (Jul–Aug 2026). Clean monitoring & retraining dataset. | `0e32e36d7add8436b314859eb4d16ed228dad5f85a428924cdbbde6ed2212808` | DVC |
| `data/raw/holdout_features.csv` | 250 × 10 | Final evaluation features (Sep 2026) without target. | `77a6518ca85c9585547c09d631ccbafe5e67ea0db6e9ef840b20bb2b0357807f` | Git |

### Features Schema

- `record_id`: Unique identifier (e.g., `R00001`). Excluded from model features.
- `date`: Record date (`YYYY-MM-DD`). Excluded from inference features.
- `distance_km`: Route length.
- `tiffins`: Batch quantity of tiffins (5–60).
- `handovers`: Number of relay handovers (1–5).
- `wait_at_station_min`: Transit station dwell time in minutes (0–25).
- `rider_age`: Age of main runner in years (19–60).
- `temp_c`: Noon ambient temperature in degrees Celsius (10–45).
- `transport`: Primary transit mode (`local_train`, `scooter`, `bicycle`).
- `route_zone`: Route geographic sector (`Central`, `Suburb`, `Old City`).
- `delivery_min`: Target duration in minutes from pick-up to destination desk.

---

## 6. Seven Data Quality Issues & Findings

Each data quality problem defined in `STUDENT_HANDBOOK.md` was investigated programmatically in `src/mp191/data_validation.py`:

| # | Data Quality Issue | Present | Findings & Parameters | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **`leaking_column`** | **True** | `column: delivery_min` | `delivery_min` is recorded post-event, absent in holdout features, and must never be used as an input feature. |
| 2 | **`unit_mixup`** | **True** | `column: distance_km`<br>`group_value: Suburb`<br>`factor: 1000` | Suburb route distances were logged in meters ($1,100\text{--}20,000\text{ m}$) instead of kilometers ($1.1\text{--}20.0\text{ km}$). Conversion factor is exactly $1000$. |
| 3 | **`sentinel_value`** | **False** | `null` | All numeric features reside within valid physical and physiological ranges. No sentinel flags (e.g., `-999`, `-1`, `9999`) were found. |
| 4 | **`duplicate_rows`** | **False** | `null` | `df.duplicated().sum() == 0`. Every record is distinct. |
| 5 | **`stuck_sensor`** | **True** | `column: rider_age`<br>`start_date: 2026-05-02`<br>`end_date: 2026-05-19` | `rider_age` logged a continuous constant value of `23` for 105 consecutive records during this 18-day hardware/entry glitch. |
| 6 | **`target_typos`** | **True** | `count: 13` | Exactly 13 target values ($247.0\text{--}763.0\text{ min}$) sit above a wide distribution gap ($> 100\text{ min}$). Dividing by 10 maps them into $[24.7, 76.3]\text{ min}$, aligning with normal deliveries. |
| 7 | **`spelling_variants`**| **False** | `null` | Categoricals (`transport`, `route_zone`) contain exactly 3 clean levels matching `later.csv`. No casing or whitespace inconsistencies exist. |

---

## 7. Data Cleaning Methodology

The cleaning pipeline in `src/mp191/data_cleaning.py` implements the following rules:

1. **Raw Cryptographic Verification**: Verifies the binary SHA-256 hash of `train.csv` before reading.
2. **Unit Correction**: Converts Suburb distance readings in meters to kilometers:
   $$\text{distance\_km} = \frac{\text{distance\_km}}{1000} \quad \text{for } \text{route\_zone} = \text{'Suburb'} \text{ and } \text{distance\_km} > 100$$
3. **Target Typo Correction**: Corrects typographical errors in `delivery_min` by dividing values $\ge 200\text{ min}$ by 10. Does not modify clean datasets such as `later.csv`.
4. **Stuck Sensor Resolution**: For rows between `2026-05-02` and `2026-05-19` with `rider_age == 23`, replaces the faulty reading with the median age of valid readings (`38`). Valid 23-year-old riders outside this window are preserved.
5. **Record Preservation**: All 1,000 raw training rows are preserved without loss of legitimate observations.

---

## 8. Execution Commands

### 8.1 Execute Data Validation Detectors
Runs all seven independent detectors and prints the diagnostic summary:
```powershell
python -m src.mp191.data_validation
```

### 8.2 Execute Data Cleaning Verification
Runs hash verification and tests the cleaning transformations on `train.csv`:
```powershell
python -m src.mp191.data_cleaning
```

### 8.3 Verify Dataset Cryptographic Hashes
```powershell
Get-FileHash -Algorithm SHA256 data\raw\*.csv | Format-List
```

### 8.4 Git and DVC Status Commands
```powershell
# Check DVC pipeline status
python -m dvc status

# Check Git working tree
git status --short
```
