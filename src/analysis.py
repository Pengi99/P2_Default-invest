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
    # TODO: plot cumulative returns comparison
    plt.figure()
    # placeholder
    plt.title('Strategy Comparison')
    plt.show()


def generate_summary_table(result1, result2):
    df = pd.DataFrame([result1, result2], index=['Filtered', 'Unfiltered'])
    print(df)
