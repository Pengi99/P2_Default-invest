"""Backtesting utilities."""

import pandas as pd
import numpy as np
import config


def run_backtest(model, test_data):
    portfolio_by_year = generate_portfolio_by_year(model, test_data)
    returns = simulate_rebalancing(portfolio_by_year)
    metrics = calculate_performance_metrics(returns)
    return metrics


def generate_portfolio_by_year(model, test_data):
    portfolios = {}
    for year, data in test_data.groupby('year'):
        preds = model.predict(data.drop('bankrupt_label', axis=1))
        filtered = data[preds == 0]
        top_n = int(len(filtered) * config.QUALITY_QUANTILE)
        selected = filtered.sort_values('quality_factor', ascending=False).head(top_n)
        portfolios[year] = selected
    return portfolios


def simulate_rebalancing(portfolio_by_year, initial_capital=1e8):
    # TODO: simulate portfolio rebalancing
    return pd.Series(dtype=float)


def calculate_performance_metrics(returns):
    if returns.empty:
        return {}
    cumulative = (1 + returns).cumprod() - 1
    cagr = (1 + cumulative.iloc[-1]) ** (252/len(returns)) - 1
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    mdd = (cumulative.cummax() - cumulative).max()
    return {
        'cumulative_return': cumulative.iloc[-1],
        'cagr': cagr,
        'sharpe': sharpe,
        'mdd': mdd
    }
