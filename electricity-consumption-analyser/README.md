# Electricity Consumption Analysis Project

A **modular Python project** for analyzing household electricity consumption in Israel, generating interactive Plotly visualizations, and recommending the most cost-effective electricity discount scheme from raw .csv file containing summary of consumption and provided by electrical company. Designed to help users navigate Israel’s liberalized electricity market by evaluating historical consumption data against available tariffs and discount packages.
Input: a raw electricity consumption file from Israeli Electricity Company (.csv)

************************************************************************
## Quick Activation 
### 1. Clone & Setup

```bash
git clone https://github.com/YouliaXX/DS-portfolio/electricity-consumption-analyzer.git
cd electricity-consumption-analyzer

# Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 2. Install dependencies

``` bash
pip install -e . 
```

### 3. Prepare Your data

``` bash
#Create folders if they don't exist:
  data/raw/
  data/external/

#Put your IEC electricity consumption file here:
  data/raw/Electricity_consumption.csv
  
#(Optional but recommended) Put discount offers here:
  data/external/electricity_discount_offers.csv

```
#### How to generate `electricity_discount_offers.csv`

This file is not included in the repository because discount offers change over time. To create it:

1. Open `docs/kamaze_scraping_ai_prompt.md` and copy the prompt inside.
2. Go to [Claude](https://claude.ai) and select model **Claude Sonnet 4.6**.
3. Paste the prompt and run it. Claude will scrape current discount offers from the Kamaze website and return a structured CSV.
4. Save the result as `data/external/electricity_discount_offers.csv`.

> The prompt instructs Claude to extract provider names, discount percentages, applicable hours, and smart-meter requirements — exactly the fields the pipeline expects.
### 4. Run the analysis

``` bash
python pipeline.py
```

### 5. Streamlit dashboard (optional)

``` bash
streamlit run app/streamlit_electricity_usage.py
```

********************************************************************************
## **Overview**

This project was developed to provide **actionable insights** into household electricity usage patterns. With multiple providers now offering time-of-use tariffs, seasonal discounts, and consumption-based packages, this tool enables users to:

- Process raw data from the **Israel Electric Corporation (IEC) digital meter**.
- Clean, standardize, and analyze consumption data.
- Identify peak/off-peak usage, daily/weekly/monthly trends, and estimate costs under different tariff structures.
- Generate **interactive visualizations** and **exportable reports** to support informed decision-making.

The project **does not invent or simulate data**. It relies exclusively on:

1. `data/raw/Electricity_consumption.csv` (personal IEC digital meter export)
2. `data/external/electricity_discount_offers.csv` (discount offers collected via Claude Sonnet 4.6 using the prompt in `docs/kamaze_scraping_ai_prompt.md`)
3. Optional weather data from the **Open-Meteo Archive API** (for contextual analysis)
4. Derived statistics from measured consumption data.

---

## **Key Features**

- **Automated data processing**: Handles IEC CSV exports, including Hebrew headers, metadata rows, and European decimal separators.
- **Comprehensive analysis**: Hourly, daily, weekly, and monthly consumption trends, with peak/off-peak identification.
- **Tariff comparison**: Estimates costs under multiple provider tariffs and ranks discount schemes based on **measured consumption** (not forecasts).
- **Visualizations**: Interactive Plotly charts for consumption patterns
- **Reporting**: Generates a **Markdown summary report** (`reports/summary_report.md`) and CSV outputs for further analysis.
- **Streamlit dashboard**: Optional, a basic dashboard for local viewing and decision-making. 

---

## **Discount Recommendation Logic**

The recommendation system **ranks discount plans** by:

1. Filtering offers based on **time restrictions** (e.g., night-time discounts).
2. Applying the **scraped discount percentage** to the user’s measured consumption that falls within each offer’s time window.
3. Scoring plans relative to the user’s actual usage patterns.
4. Flagging **smart-meter-only plans** as `unknown_smart_meter_required` if the user’s smart-meter status is unknown.

> **Note**: This is **not a bill forecast**. It provides a **relative ranking** of plans based on historical data.
> Decision should be made in case-by-case manner.

---

## **Dataset Format Notes**

### **Source**: IEC Digital Meter CSV Export

IEC exports may vary in format. Known characteristics:

- Headers and column names may be in **Hebrew**.
- Personal account information may appear above the measurement table.
- The measurement table may not start on the first row.
- Numeric values may use **European decimal separators** (`,` instead of `.`).

The analyzer automatically:

- Detects the start of the measurement table.
- Skips metadata and personal information.
- Normalizes Hebrew-formatted columns.
- Parses date/time fields and cleans consumption values.
- Sumarizes the data by hourly/daily and weekly consumption, including mean, std values
- Analyzes outlier to detect extreme values and then understand whether they're correlate with weather or specific electrical device usage.
- Generates a series of visualizations (heatmaps, barcharts, etc) for clearity.
- 
- Generates csv, tables, html (plotly) files for further use
- Summarizes conclusions and recommendations in report file
---

## **Project Structure**

Bulk structure.

```
electricity-consumption-analyzer/
├── app                   # Optional, streamlit dashboard code
├── data/
│   ├── raw/               # Original IEC CSV exports
│   ├── external/          # Discount offers csv
│   └── processed/         # Cleaned and derived datasets
├── src/                   # Python scripts (analyses and visualizations)
├── notebook.ipynb         # EDA Jupyter notebooks
├── outputs/
│   ├── html/              # Plotly visualizations
│   ├── figures/           # .png visualizations              
│   └── tables/            # CSV outputs 
├── docs/                  # Prompts and reference documents
│   └── kamaze_scraping_ai_prompt.md  # Claude prompt for collecting discount offers
├── reports/               # Generated reports (e.g., `summary_report.md`)
├── pyproject.toml
├── pipeline.py        
└── README.md
```

---

## **Outputs**

Running the pipeline generates:

- **Processed data**:
  - `data/processed/cleaned_consumption.csv`
  - `data/processed/weekly_hourly_stats.csv`
  - `data/processed/daily_stats.csv`
  - `data/processed/outliers.csv`
- **Visualizations**: `outputs/html/*.html` (Plotly charts)
- **Reports**:
  - `reports/summary_report.md` (Markdown summary)
  - `outputs/tables/discount_scenarios.csv` (Discount plan comparisons)
- **Streamlit Dashboard**: `optional`
  - `Overview`
  - `Hourly Patterns`
  - `Trends & Outliers`
  - `Calculator`
  - **and more**
---

## Author

**Youlia Denisov**  
Data Analyst / Data Scientist / Biologist 

- GitHub: https://github.com/YouliaXX  
- LinkedIn: https://linkedin.com/in/youliadenisov-phd

## **License**

MIT License — © 2026 Youlia Denisov

Permission is hereby granted, free of charge, to any person obtaining a copy of this software to use, copy, modify, merge, publish, distribute, and/or sublicense it, subject to the following condition: this copyright notice and permission notice must be included in all copies or substantial portions of the software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.