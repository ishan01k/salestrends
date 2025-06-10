# 📈 E-commerce Sales Trends Prediction


## 📂 Project Structure

| File                                   | Description                                                                             |
| -------------------------------------- | --------------------------------------------------------------------------------------- |
| `Amazon_Sale_Report.csv`               | Raw sales data from Amazon platform.                                                    |
| `Cloud_Warehouse_Compersion_Chart.csv` | Comparison of various cloud data warehouse providers.                                   |
| `Expense_IIGF.csv`                     | Expense tracking dataset for an IIGF entity.                                            |
| `International_sale_Report.csv`        | E-commerce sales report segmented by international markets.                             |
| `May-2022.csv`                         | Sales report for the month of May 2022.                                                 |
| `P_L_March_2021.csv`                   | Profit & Loss statement for March 2021.                                                 |
| `Sale_Report.csv`                      | Combined general sales report.                                                          |
| `amazon_sales_cleaned.csv`             | Cleaned version of Amazon sales data used in analysis.                                  |
| `pricing_data_cleaned.csv`             | Cleaned pricing data, prepared for analysis.                                            |
| `analysis_summary.json`                | Summary of statistical and exploratory data analysis.                                   |
| `Sales-Trends-Prediction.py`           | Python script for preprocessing, training, and predicting sales trends using ML models. |
| `Visualizations.docx`                  | Charts and graphs providing visual insights on trends and patterns.                     |

---

## 🧭 Visualization Objectives & Interaction Guide

The **visualizations** in this project serve to **translate raw data into actionable business insights**. They're designed for both data analysts and decision-makers who want to understand sales patterns, cost impact, and market behavior.

### 🎯 Objectives

1. **Sales Trends Over Time**
   Understand seasonality and forecast future demand.

2. **Geographical Sales Breakdown**
   Compare domestic vs. international performance.

3. **Expense and Profit Analysis**
   Visualize where the money goes and how profits evolve.

4. **Cloud Warehouse Cost Efficiency**
   Evaluate data warehouse options for analytics scalability.

5. **Forecast Insights**
   Highlight predicted sales behavior for upcoming months.

### 🛠 How to Interact with the Visuals

* **Open `Visualizations.docx`**: This file contains embedded charts with annotations for context.
* **Hover on tooltips** *(in future interactive versions)*: When deployed in a dashboard, you'll be able to hover over chart points for real-time data insights.
* **Compare Year-Over-Year Trends**: Several visuals align month-by-month data across years to highlight patterns.
* **Look for Annotations**: Key events (e.g., marketing campaigns, peak holidays) are marked where relevant.

> **Note**: You can convert these visuals into interactive dashboards using Plotly Dash, Streamlit, or Power BI for more hands-on exploration.

---

## 📊 Highlights & Insights

Here are key takeaways from our analysis and visualizations:

### 🔸 Sales Seasonality

* **Sales peak during Q4**, especially in November and December, indicating strong seasonal impact.
* **Drop in Q1** post-holiday season is consistently observed.

### 🔸 Regional Performance

* **International markets** show rising trends, with increasing sales volume year-over-year.
* Domestic market growth is **steady but saturated**.

### 🔸 Expense & P\&L Review

* **Expenses increased** notably in marketing and logistics post-2021, cutting into profit margins.
* **March 2021 P\&L** reveals a breakeven point where revenue was slightly ahead of cost.

### 🔸 Cloud Warehouse Cost Comparison

* Snowflake and BigQuery offer competitive pricing models for large-scale analytics.
* Redshift may offer performance benefits but with higher associated costs.

---

## 🧠 Predictive Modeling

The script `Sales-Trends-Prediction.py` uses time-series forecasting and regression techniques to:

* Predict **monthly sales**.
* Detect **anomalies** or sudden drops.
* Forecast **future demand**.

Libraries used:

* `pandas`, `numpy` for data processing.
* `scikit-learn`, `statsmodels` for modeling.
* `matplotlib`, `seaborn` for visualizations.

---

## 📌 Future Enhancements

* Implement LSTM or ARIMA for more robust time-series forecasting.
* Deploy the model via Flask/Django for live dashboarding.
* Integrate external market factors (ads, inflation, etc.) for richer modeling.
* Convert visualizations into interactive dashboards.

---
