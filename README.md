# CEREBRO — Decision Reversal Pattern Detector

> **Hackathon:** Kolkata IEM Hackathon | **Problem Statement 3**
> Rule-based, local, no external AI/LLM APIs.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Definitions](#definitions)
3. [Approach & Pipeline](#approach--pipeline)
4. [Repo Structure](#repo-structure)
5. [How to Run](#how-to-run)
6. [Evaluation Results](#evaluation-results)
7. [Limitations & Future Work](#limitations--future-work)
8. [Screenshots](#screenshots)

---

## Problem Statement

In online subscription checkout flows, users frequently change their minds —
switching plans, toggling add-ons, or reconsidering payment methods.  
Understanding *when* and *how often* these reversals happen reveals friction
points in the UX that can be targeted for improvement.

**Goal:** Given a sequence of user choices recorded as events, detect when a
user repeatedly changes a previous choice, identify the funnel stages where
reversals concentrate, classify the user's behavioural pattern, and present
the findings visually.

---

## Definitions

| Term | Definition |
|------|-----------|
| **Reversal** | A choice that differs from the immediately preceding choice for the same `(user_id, decision_id)` pair. |
| **Flip-back** | A reversal where the new choice equals *any* earlier choice for that decision (i.e. the user returns to something they already tried). |
| **Correction** | Exactly one reversal on a decision, made within `CORRECTION_MAX_SECONDS` (default 60 s) of the previous choice. |
| **Reversal rate** | `total_reversals / number_of_distinct_decisions_made`. |

**Funnel stages** (in order): `Browsing → Selection → Review → Confirmation → Post-commit`

**Decision types** (subscription checkout domain): `plan`, `add_on`, `payment_method`

All thresholds are centralised in [`src/config.py`](src/config.py) and can be
changed without touching any logic module.

---

## Approach & Pipeline

```
CSV  ──▶  loader.py  ──▶  detector.py  ──▶  metrics.py
            │                  │                 │
        Validate           Find reversals     Aggregate
        schema             (flip-back flag,   per-user
        parse ts           time gap)          KPIs
        sort

metrics.py  ──▶  classifier.py  ──▶  explain.py  ──▶  visuals.py
                      │                  │                 │
               6 priority rules     Template         Plotly charts
               (decisive →          sentence         (timeline,
               single_corrector →   per pattern      stage bar,
               chronic_indecisive →                  Sankey)
               flip_flopper →
               late_reverser →
               mixed)
                      │
               evaluate.py / app.py
```

### Classifier rules (evaluated in priority order)

| Priority | Pattern | Condition |
|----------|---------|-----------|
| 1 | `decisive` | `total_reversals == 0` |
| 2 | `single_corrector` | Exactly 1 reversal, no flip-back, gap ≤ 60 s |
| 3 | `chronic_indecisive` | `reversal_rate ≥ 0.75` AND `decisions_with_reversals ≥ 2` AND `max_share_on_one_decision < 0.60` (broad) |
| 4 | `flip_flopper` | (≥ 3 reversals on one decision **OR** ≥ 2 flip-backs) AND `max_share_on_one_decision ≥ 0.60` (concentrated) |
| 5 | `late_reverser` | ≥ 60 % of reversals at Confirmation / Post-commit |
| 6 | `mixed` | Catch-all |

---

## Repo Structure

```
.
├── app.py                         # Streamlit dashboard entry point (multi-domain & upload)
├── cerebro_decision_reversal.ipynb# Self-contained, executed Jupyter notebook (Sections 0-13)
├── presentation.html              # 12-slide interactive dark-mode glassmorphic presentation
├── evaluate.py                    # CLI accuracy / confusion-matrix script (100% accuracy)
├── requirements.txt               # Complete runtime dependencies
├── README.md                      # Comprehensive documentation
├── .gitignore                     # Production Git ignore rules
│
├── data/
│   ├── generate_data.py           # Benchmark synthetic event generator (200 users, 5 personas)
│   ├── sample_events.csv          # Benchmark dataset: 1,239 events
│   ├── ground_truth.csv           # Benchmark ground truth: 200 personas
│   ├── real_retail_events.csv     # Real UCI / Databricks online retail dataset (1,968 events)
│   ├── ecommerce_events.csv       # E-Commerce cart & checkout dataset
│   ├── loan_application_events.csv# Fintech loan qualification dataset
│   ├── healthcare_appointments.csv# Telehealth consultation booking dataset
│   ├── hotel_travel_booking.csv   # Hospitality & travel reservation dataset
│   ├── cloud_infra_provisioning.csv# Cloud DevOps infrastructure dataset
│   ├── edtech_course_enrollment.csv# University course enrollment dataset
│   └── gaming_character_loadout.csv# RPG gaming loadout configuration dataset
│
├── outputs/
│   ├── user_metrics_and_patterns.csv # Per-user metrics, pattern labels & explanations
│   └── detected_reversals.csv        # Reversal audit log with deltas & flip-back flags
│
├── src/
│   ├── config.py                  # Thresholds, schema constants, and auto-detect logic
│   ├── loader.py                  # Universal CSV / BytesIO ingestion & schema validator
│   ├── detector.py                # Reversal detection (flip-back flag, Δt gap)
│   ├── metrics.py                 # Per-user KPI & reversal concentration metrics
│   ├── classifier.py              # Rule-based priority pattern classifier
│   ├── explain.py                 # Plain-English evidence sentence generator
│   └── visuals.py                 # Plotly visualisations (timeline, stage, Sankey)
│
└── tests/
    ├── test_classifier.py         # Unit tests for classifier rules & priority
    ├── test_detector.py           # Unit tests for reversal detection & flip-backs
    ├── test_metrics.py            # Unit tests for KPI calculations & concentration
    ├── test_loader.py             # Ingestion, streams, auto-detect & missing stage tests
    ├── test_explain.py            # Evidence sentence generation tests
    └── test_multi_dataset_pipeline.py # End-to-end integration across all 10 domains
```

---

## How to Run

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.10+** required.

### 2 — Generate benchmark data

```bash
python data/generate_data.py
```

Outputs `data/sample_events.csv` (1,239 rows, 200 users) and `data/ground_truth.csv` (200 rows with true personas). Fixed seed (`SEED = 42`) guarantees 100% reproducibility.

### 3 — Launch the Streamlit app

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.
- **Sidebar:** Choose between 10 industry presets or upload any custom CSV with automatic column detection and interactive schema override.
- **Overview tab:** Global KPIs, ranked user table with reversal rates and pattern distribution, global stage distribution chart.
- **User Deep-Dive tab:** Select any user to see their metrics, explanation sentence, interactive event timeline, stage bar chart, and choice-flow Sankey diagram.

### 4 — Open the Self-Contained Jupyter Notebook

```bash
jupyter notebook cerebro_decision_reversal.ipynb
```

- Fully self-contained from Section 0 through Section 13.
- Pre-executed with all output cells, interactive widgets, Plotly visualizations, and 100% accuracy metrics.
- Can be re-executed end-to-end via **Kernel -> Restart & Run All**.

### 5 — View the Interactive Presentation Slide Deck

Open [`presentation.html`](presentation.html) directly in any modern web browser (Google Chrome, Firefox, Safari, Edge).
- 12 interactive dark-mode glassmorphic slides.
- Use `Arrow Keys` or `Space` to navigate. Press `N` to toggle speaker notes. Press `F` for fullscreen.

### 6 — Run unit and integration tests

```bash
python -m pytest -v
```

**68 tests, all green (100% passing)** across all 6 test modules in `tests/`.

### 7 — Run CLI evaluation

```bash
python evaluate.py
```

Runs the full pipeline on `data/sample_events.csv` and benchmarks against `data/ground_truth.csv`. Achieves **1.0000 (100.00%) accuracy** across all 200 users and saves artifacts to `outputs/`.


---

## Dynamic Architecture for Any Dataset

CEREBRO is built to analyze decision reversals across **any product, funnel, or industry** — not just the original subscription checkout spec.

```
Arbitrary CSV
  ├── Column auto-detection (exact match → alias dictionary → keyword heuristics)
  ├── Funnel discovery (chronological stage order extraction)
  ├── Missing stage synthesis (defaults to "General" if funnel stages omitted)
  └── Dynamic late-stage derivation (final ~35-40% of active funnel steps)
```

### Supported Presets Included in the App:
1. **🚀 SaaS Onboarding / Subscription Funnel** (`data/sample_events.csv`):
   - Standard 5-stage funnel: `Browsing → Selection → Review → Confirmation → Post-commit`
   - Decision types: `plan`, `add_on`, `payment_method` (200 users, 1,239 events)
2. **🌐 Real-World Online Retail (UCI / Databricks Open Dataset)** (`data/real_retail_events.csv`):
   - Real transaction events fetched live from Databricks/UCI benchmark (1,968 transactions, 98 actual shoppers)
   - Real cancellations and quantity adjustments detected as genuine human behavioral reversals!
3. **🏥 Healthcare & Telehealth Appointments** (`data/healthcare_appointments.csv`):
   - Columns: `patient_id`, `consultation_item`, `scheduled_at`, `triage_step`, `selected_value`
   - Funnel: `Triage_Intake → Symptoms_Assessment → Provider_Match → Time_Slot → Insurance_Verification` (50 patients)
4. **🏨 Luxury Hotel & Travel Booking** (`data/hotel_travel_booking.csv`):
   - Columns: `guest_id`, `package_param`, `booking_time`, `checkout_milestone`, `option_selected`
   - Funnel: `Destination_Search → Room_Selection → Dining_Plan → Addon_Services → Final_Confirmation` (54 guests)
5. **☁️ Cloud DevOps Infrastructure Provisioning** (`data/cloud_infra_provisioning.csv`):
   - Columns: `developer_id`, `infra_parameter`, `config_time`, `deployment_phase`, `spec_value`
   - Funnel: `Cluster_Spec → Compute_Tier → Storage_Config → Network_VPC → Review_Deploy` (50 engineers)
6. **🎓 EdTech University / Masterclass Enrollment** (`data/edtech_course_enrollment.csv`):
   - Columns: `student_id`, `curriculum_field`, `event_time`, `enrollment_step`, `chosen_option`
   - Funnel: `Track_Selection → Specialization → Schedule_Pace → Mentorship → Tuition_Plan` (50 students)
7. **⚔️ RPG Gaming Character Customization & Loadout** (`data/gaming_character_loadout.csv`):
   - Columns: `gamer_tag`, `loadout_slot`, `action_timestamp`, `inventory_screen`, `equipped_gear`
   - Funnel: `Class_Origin → Primary_Weapon → Armor_Set → Elemental_Rune → Confirm_Battle_Loadout` (54 players)
8. **🛍️ E-Commerce Checkout** (`data/ecommerce_events.csv`):
   - Columns: `customer_id`, `item_type`, `event_time`, `checkout_step`, `selected_option`
   - Funnel: `Cart → Shipping → Payment_Info → Order_Review` (60 customers)
9. **🏦 Fintech & Loan Application** (`data/loan_application_events.csv`):
   - Columns: `applicant_id`, `field`, `timestamp`, `phase`, `value`
   - Funnel: `Eligibility → Vehicle_Details → Coverage_Quote → Final_Signature` (60 applicants)
10. **📂 Upload Custom CSV**:
   - Accepts clickstream or event dumps from mobile apps, web funnels, or gaming UI.
   - Interactive schema mapping panel allows instant overrides of column assignments and late-stage definitions.

### Auto-Detection Alias Table

| Canonical Field | Recognized Aliases & Keywords |
|-----------------|-------------------------------|
| `user_id` | `user_id`, `customer_id`, `client_id`, `account_id`, `visitor_id`, `applicant_id`, `patient_id`, `guest_id`, `student_id`, `developer_id`, `player_id`, `gamer_tag`, `member_id` |
| `decision_id` | `decision_id`, `item_id`, `item_type`, `feature`, `field`, `product`, `setting`, `action_type`, `stockcode`, `sku`, `consultation_item`, `package_param`, `infra_parameter`, `curriculum_field`, `loadout_slot` |
| `timestamp` | `timestamp`, `time`, `datetime`, `date`, `event_time`, `created_at`, `recorded_at`, `ts`, `invoicedate`, `scheduled_at`, `booking_time`, `config_time`, `action_timestamp` |
| `stage` | `stage`, `funnel_stage`, `step`, `funnel_step`, `checkout_step`, `phase`, `screen`, `page`, `triage_step`, `checkout_milestone`, `deployment_phase`, `enrollment_step`, `inventory_screen` (optional — synthesized if absent) |
| `choice` | `choice`, `value`, `option`, `selection`, `selected_option`, `variant`, `quantity`, `qty`, `option_selected`, `spec_value`, `chosen_option`, `equipped_gear` |

---

## Evaluation Results

Evaluated on 200 synthetic users with balanced ground-truth personas (40 each).

### Overall accuracy: **100.00 %** (200 / 200 correct)

### Confusion matrix

```
True \ Pred          chronic_indecisive  decisive  flip_flopper  late_reverser  single_corrector
──────────────────── ─────────────────── ───────── ────────────  ─────────────  ────────────────
chronic_indecisive            40             0            0             0               0
decisive                       0            40            0             0               0
flip_flopper                   0             0           40             0               0
late_reverser                  0             0            0            40               0
single_corrector               0             0            0             0              40
```

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| chronic_indecisive | 1.0000 | 1.0000 | 1.0000 | 40 |
| decisive | 1.0000 | 1.0000 | 1.0000 | 40 |
| flip_flopper | 1.0000 | 1.0000 | 1.0000 | 40 |
| late_reverser | 1.0000 | 1.0000 | 1.0000 | 40 |
| single_corrector | 1.0000 | 1.0000 | 1.0000 | 40 |
| **macro avg** | **1.0000** | **1.0000** | **1.0000** | **200** |

### Analysis

- **All 5 patterns** are detected with 100% precision, recall, and F1 (200/200 correct).
- **Concentrated vs Broad Churn Separation:**
  The distinction between `flip_flopper` and `chronic_indecisive` is formulated around decision concentration:
  - `flip_flopper`: Reversals are heavily concentrated on a single focal decision (`max_share_on_one_decision ≥ 0.60`).
  - `chronic_indecisive`: Reversals are dispersed across multiple decisions (`decisions_with_reversals ≥ 2`, `reversal_rate ≥ 0.75`, `max_share_on_one_decision < 0.60`).
  This eliminates feature overlap without requiring artificial hardcoding or constraints on churn count.

---

## Limitations & Future Work

| Limitation | Potential fix |
|-----------|--------------|
| `chronic_indecisive` ↔ `flip_flopper` overlap | Add inter-decision-spread guard to flip_flopper rule |
| Rule priority is fixed | Replace with a ranked scoring system per pattern |
| `mixed` is a catch-all with no sub-explanation | Add secondary indicators (e.g. "2 slow reversals across 2 decisions") |
| Single-session assumption | Extend loader to handle multi-session events per user |
| No time-of-day / session-length features | Add session-duration and intra-session pace signals |
| Thresholds hand-tuned | Calibrate against a larger labelled dataset |
| Sankey collapses revisited nodes | Option to show positional node disambiguation in the UI |

---

## Screenshots

> _Run the app (`streamlit run app.py`) and add screenshots here._

### Overview tab

<!-- Add screenshot: Overview_tab.png -->

### User Deep-Dive — Flip-Flopper

<!-- Add screenshot: UserTab_FlipFlopper.png -->

### User Deep-Dive — Timeline chart

<!-- Add screenshot: Timeline_chart.png -->

### Sankey flow diagram

<!-- Add screenshot: Sankey_flow.png -->
