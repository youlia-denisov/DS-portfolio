"""
Main pipeline script for electricity consumption analysis.
This orchestrates the entire workflow:
1. Checks and creates necessary output directories.
2. Checks the presence of smart electric meter data and adjusts analysis accordingly.
3. Load raw consumption data from CSV.
4. Clean and preprocess the data.
5. Run KMeans clustering
6. Compute stats (daily/hourly) + detect outliers.
7. Generate and save visualizations.
8. (Optional) Perform weather analysis and discount scenario estimation.
9. Save all outputs (cleaned data, stats, visuals) to organized folders.
10. Generates report, with most important highlights and recommendations 
The main function is `run_pipeline()` which executes all steps in sequence, with conditional execution of weather module
This script is designed to be run as a standalone program, and it will create all necessary output
folders if they don't exist. 
The visualizations are saved as interactive HTML files, the cleaned data 
and stats are saved as CSV files for further analysis or reporting.

Additional option: visualization of the pipeline using streamlit (the code is also available in this project "app/streamlit_electricity_usage.py)
"""
import sys
import config
from src.loader import load_raw_csv, load_discount_offers
from src.preprocessing import clean_consumption_data
from src.aggregation import compute_hourly_stats, compute_daily_stats, compute_daily_totals, compute_summary
from src.outliers import detect_outliers_3sigma, detect_outliers_iqr, calculate_outlier_summary
from src.visualization import save_all_visuals, save_clustering_visuals
from src.weather_analysis import add_weather, summarize_weather, save_weather_plots
from src.discount_analysis import estimate_discount_scenarios, choose_recommendation, generate_side_by_side_plots, get_user_smart_meter_status
from src.reporting import write_report
from src.clustering import run_clustering


def run_pipeline(run_weather: bool = True):
    print("Starting pipeline execution...")
    print("Checking and creating output directories...")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.HTML_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Resolve smart meter presence and adjust analysis if needed
    smart_meter_status =  config.HAS_SMART_METER
    if smart_meter_status is None:
        print("\n[Input Prompt Required]")
        smart_meter_status = get_user_smart_meter_status()
    print(f"Smart Meter configuration set to: {smart_meter_status}  \n")    
    
    # File loading and preprocessing
    raw = load_raw_csv(config.CONSUMPTION_FILE)
    df = clean_consumption_data(raw)
    df.to_csv(config.PROCESSED_DIR / "cleaned_consumption.csv", index=False)

    # Clustering Analysis
    # Passing figure_dir triggers the elbow method before the final KMeans fit.
    # The elbow curve is saved to config.FIGURE_DIR for the report.
    print("\nRunning KMeans clustering...")
    df_clustered = run_clustering(figure_dir=config.FIGURE_DIR)

    # Compute average/std values and detect outliers
    hourly = compute_hourly_stats(df)
    daily = compute_daily_stats(df)
    daily_totals = compute_daily_totals(df)
    outliers_3sigma = detect_outliers_3sigma(df)
    outliers_iqr = detect_outliers_iqr(df)
    outlier_summary = calculate_outlier_summary(df, outliers_3sigma, outliers_iqr)

    # Save processed data and stats to CSV
    hourly.to_csv(config.PROCESSED_DIR / "weekly_hourly_stats.csv", index=False)
    daily.to_csv(config.PROCESSED_DIR / "daily_stats.csv", index=False)
    daily_totals.to_csv(config.PROCESSED_DIR / "daily_totals.csv", index=False)
    outliers_3sigma.to_csv(config.PROCESSED_DIR / "outliers_3sigma.csv", index=False)
    outliers_iqr.to_csv(config.PROCESSED_DIR / "outliers_iqr.csv", index=False)
    outlier_summary.to_csv(config.PROCESSED_DIR / "outlier_summary.csv", index=False)

    print("Generating visualizations...")
    save_all_visuals(df, hourly, daily, outliers_3sigma, config.HTML_DIR)
    print("Generating clustering visualizations...")
    save_clustering_visuals(df_clustered=df_clustered, figure_dir=config.FIGURE_DIR)

    # Optional weather analysis.
    weather_summary = None
    if run_weather:
        try:
            df_weather = add_weather(df)
            df_weather.to_csv(config.PROCESSED_DIR / "consumption_with_weather.csv", index=False)
            weather_summary = summarize_weather(df_weather)
            save_weather_plots(df_weather, config.HTML_DIR)
        except Exception as error:
            print(f"Weather analysis skipped: {error}")

    # Discount scenario estimation and recommendation
    offers = load_discount_offers(config.DISCOUNT_OFFERS_FILE)
    scenarios = estimate_discount_scenarios(df, offers, has_smart_meter=smart_meter_status)
    scenarios.to_csv(config.TABLE_DIR / "discount_scenarios.csv", index=False, encoding="utf-8-sig")
    recommendation = choose_recommendation(scenarios)

    # Generate side-by-side plots for the recommended plan
    print("\nGenerating tariff matrices...")
    generated_plots = generate_side_by_side_plots(has_smart_meter=smart_meter_status)

    print(f"Generated {len(generated_plots)} side-by-side plots.")
    
    #Generate final report
    summary = compute_summary(df, hourly, daily)
    report_path = write_report(
        summary=summary,
        outliers=outlier_summary,
        weather_summary=weather_summary,  
        scenarios=scenarios,
        recommendation=recommendation,
        report_dir=config.REPORT_DIR,           
        generated_plots=generated_plots
        )

    print("\nPipeline completed successfully.")
    print(f"Report: {report_path}")
    print(f"Interactive HTML visuals: {config.HTML_DIR}")
    print(f"Clustering plots & data: {config.PROCESSED_DIR}")
    print(f"Static figure comparison plots: {config.FIGURE_DIR}")
    print(f"Discount scenarios: {config.TABLE_DIR / 'discount_scenarios.csv'}")


if __name__ == "__main__":
    """ Options to disable weather analysis:
    - Command line: `python pipeline.py --no-weather`
    - Or set `run_weather=False` when calling `run_pipeline()`"""
    run_weather_flag = True
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["--no-weather", "--skip-weather"]:
        run_weather_flag = False

    run_pipeline(run_weather=run_weather_flag)