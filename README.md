# Youlia Denisov · Data Science Portfolio

**Biologist → Data Explorer** · PhD in Biology · Turning messy real-world data into clear, useful things

[![LinkedIn](https://img.shields.io/badge/LinkedIn-youliadenisov--phd-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/youliadenisov-phd)
[![GitHub](https://img.shields.io/badge/GitHub-youlia--denisov-181717?logo=github&logoColor=white)](https://github.com/youlia-denisov/DS-portfolio)

---

## Skills

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?logo=postgresql&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?logo=powerbi&logoColor=black)
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)

---

## Projects

### 🌿 NGT Tracker — Gene Editing Dashboard
![Status](https://img.shields.io/badge/status-live-brightgreen)

**[▶ Open Dashboard](https://youlia-denisov.github.io/DS-portfolio/gene-editing-tracker/docs/)** · [Source](./gene-editing-tracker)

206 gene-edited agricultural products. 60 organisms. 17 countries. One interactive dashboard.

This started as a Power BI report and became a fully browser-based web app — no license, no server, shareable as a URL. It maps the global NGT (CRISPR and related tools) pipeline from lab discovery to market, with real-time cross-filtering by organism, country, and trait.

![Dashboard overview](./gene-editing-tracker/screenshots/dashboard_overview.png)

What I built on top of the original Power BI:
- World map showing geographic distribution (not in the original)
- Animated bar chart race — NGT product counts by organism, 2016–2024
- Organism drill-down with stage breakdown and per-product image cards

**Stack:** Plotly.js · wordcloud2.js · Vanilla JS · GitHub Pages

---

### ⚡ Electricity Consumption Analyser
![Status](https://img.shields.io/badge/status-complete-blue)

[Source](./electricity-consumption-analyser)

My first public project. Takes raw meter exports from the Israel Electric Corporation — messy CSVs with Hebrew headers — and turns them into actual insights: consumption patterns, outlier spikes, weather correlations, and a plain-language answer to "which discount plan would save me money?"

What I'm most satisfied with technically:
- KMeans clustering with cyclical time encoding (so Monday and Sunday are actually neighbours)
- Dual-method outlier detection with a recommendation based on your data's distribution
- Optional weather layer via Open-Meteo API

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Stack:** Python · pandas · scikit-learn · Plotly · Streamlit · matplotlib

---

### 🌐 WattWise — Multi-User App
![Status](https://img.shields.io/badge/status-complete-blue)

[Source](./streamlit-multiuser-app)

A full refactor of the analyser above into a proper multi-user web app — each visitor gets an isolated session, so multiple people can upload and analyse their data simultaneously without leaking state between them.

New analytical tabs added in this version:

| Tab | What it does |
|-----|-------------|
| Behavioural Fingerprint | Distils your history into readable metrics: WFH ratio, peak hour, night owl score, regularity index |
| Outlier Detection | Side-by-side 3σ vs IQR comparison with an auto-selected recommendation |
| Usage Clustering | Groups hourly readings into ranked profiles, quietest to heaviest |

**Stack:** Python · Streamlit · scikit-learn · Plotly · pandas
