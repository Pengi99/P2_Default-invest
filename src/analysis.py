"""Analysis utilities."""

import pandas as pd
import matplotlib.pyplot as plt
from . import backtester
import config


def run_backtest_and_analyze(model):
    test_data = pd.read_csv("data/processed/test.csv")
    result1 = backtester.run_backtest(model, test_data)
    result2 = backtester.run_backtest(model, test_data)  # without filtering would require modification
    plot_results(result1, result2)
    generate_summary_table(result1, result2)


def plot_results(result1, result2):
    """Plots the cumulative returns of two backtest results."""
    plt.figure(figsize=(12, 6))
    
    # It's good practice to ensure the series exists and is not empty before plotting
    if 'cumulative_returns_series' in result1 and isinstance(result1['cumulative_returns_series'], pd.Series) and not result1['cumulative_returns_series'].empty:
        series_to_plot1 = result1['cumulative_returns_series'].copy() # Work on a copy to avoid modifying original
        # Convert index to Python datetime objects
        series_to_plot1.index = pd.to_datetime(series_to_plot1.index).to_pydatetime()
        series_to_plot1.plot(label='Strategy 1 (e.g., Filtered)', legend=True)
    else:
        print("Warning: Cumulative returns series not found, not a Series, or empty for result1.")

    if 'cumulative_returns_series' in result2 and isinstance(result2['cumulative_returns_series'], pd.Series) and not result2['cumulative_returns_series'].empty:
        series_to_plot2 = result2['cumulative_returns_series'].copy() # Work on a copy
        # Convert index to Python datetime objects
        series_to_plot2.index = pd.to_datetime(series_to_plot2.index).to_pydatetime()
        series_to_plot2.plot(label='Strategy 2 (e.g., Unfiltered)', legend=True)
    else:
        print("Warning: Cumulative returns series not found, not a Series, or empty for result2.")
        
    plt.title('Strategy Comparison: Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Normalized to 1 at start)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def generate_summary_table(result1, result2):
    """Generates and prints a formatted summary table of performance metrics."""
    # Extract metrics, excluding the series data for the table
    metrics1 = {k: v for k, v in result1.items() if k != 'cumulative_returns_series'}
    metrics2 = {k: v for k, v in result2.items() if k != 'cumulative_returns_series'}

    df = pd.DataFrame([metrics1, metrics2], index=['Strategy 1 (Filtered)', 'Strategy 2 (Unfiltered)'])
    
    # Define columns to format and their respective formatters
    columns_to_format = {
        'cumulative_return': '{:.2%}',
        'cagr': '{:.2%}',
        'sharpe': '{:.2f}',
        'mdd': '{:.2%}'
    }
    
    formatted_df = df.copy() # Create a copy for formatted display
    for col, fmt_str in columns_to_format.items():
        if col in formatted_df.columns:
            # Apply formatting, handle potential errors if data is not numeric (e.g. None or str)
            formatted_df[col] = formatted_df[col].apply(lambda x: fmt_str.format(x) if pd.notnull(x) and isinstance(x, (int, float)) else x)
            
    print("\nPerformance Summary:")
    # .to_string() is good for console display of DataFrames
    print(formatted_df.to_string())
