"""Backtesting utilities."""

import pandas as pd
import numpy as np
import config


def run_backtest(model, test_data):
    # Ensure 'date' column is datetime type for proper operations later
    if 'date' in test_data.columns and not pd.api.types.is_datetime64_any_dtype(test_data['date']):
        test_data['date'] = pd.to_datetime(test_data['date'])
        
    portfolio_by_year = generate_portfolio_by_year(model, test_data)
    # Pass the full test_data to simulate_rebalancing for accessing all stock returns
    # Assuming initial_capital from config or a default. Let's use config.INITIAL_CAPITAL if available, else 1e8.
    initial_capital = getattr(config, 'INITIAL_CAPITAL', 1e8)
    daily_portfolio_returns = simulate_rebalancing(portfolio_by_year, test_data, initial_capital=initial_capital)
    metrics = calculate_performance_metrics(daily_portfolio_returns)
    return metrics


def generate_portfolio_by_year(model, test_data):
    portfolios = {}
    if 'date' in test_data.columns and not pd.api.types.is_datetime64_any_dtype(test_data['date']):
        test_data['date'] = pd.to_datetime(test_data['date'])

    for year_val, group_data_for_year in test_data.groupby('year'):
        # Get unique stock-year observations for model prediction (financials are annual)
        # Explicitly make it a copy to avoid SettingWithCopyWarning
        unique_stock_year_data = group_data_for_year.drop_duplicates(subset=['stock_code']).copy()
        
        if unique_stock_year_data.empty:
            portfolios[year_val] = pd.DataFrame(columns=['stock_code', 'quality_factor'])
            continue
            
        # Prepare features for the model. This must match how the model was trained.
        # Assuming the model was trained on features excluding 'bankrupt_label', 'stock_code', 'date'.
        # For robustness, explicitly select features if known, or ensure model handles extra cols.
        features_for_prediction = unique_stock_year_data.drop(['bankrupt_label', 'stock_code', 'date', 'year', 'return', 'close_price', 'market_cap', 'quality_factor'], axis=1, errors='ignore')
        # The above drop is an example. The actual features depend on model training.
        # A safer approach if model is a pipeline or has known feature names:
        # features_for_prediction = unique_stock_year_data[model.feature_names_in_] # If available
        # For now, let's assume the model was trained on X1-X4 which are in the remaining columns.
        # If the model was trained with more columns (e.g. from `data.drop('bankrupt_label', axis=1)`), 
        # then `features_for_prediction` should be `unique_stock_year_data.drop('bankrupt_label', axis=1, errors='ignore')`
        # Sticking to a more general approach for now:
        # Ensure all feature columns are present in the data for this year
        missing_cols = [col for col in config.FEATURE_COLUMNS if col not in unique_stock_year_data.columns]
        if missing_cols:
            print(f"Warning: Feature columns {missing_cols} not found in data for year {year_val}. Skipping prediction for this year.")
            portfolios[year_val] = pd.DataFrame(columns=['stock_code', 'quality_factor'])
            continue
            
        predict_features = unique_stock_year_data[config.FEATURE_COLUMNS]
        preds = model.predict(predict_features) # predict bankruptcy (1) or not (0)
        
        unique_stock_year_data['prediction'] = preds
        
        filtered_stocks = unique_stock_year_data[unique_stock_year_data['prediction'] == 0]
        
        if filtered_stocks.empty:
            portfolios[year_val] = pd.DataFrame(columns=['stock_code', 'quality_factor'])
            continue
            
        num_eligible_stocks = len(filtered_stocks)
        top_n = int(num_eligible_stocks * config.QUALITY_QUANTILE)
        
        if top_n == 0 and num_eligible_stocks > 0:
            top_n = 1 
        elif top_n == 0 and num_eligible_stocks == 0:
             portfolios[year_val] = pd.DataFrame(columns=['stock_code', 'quality_factor'])
             continue

        selected_for_portfolio = filtered_stocks.sort_values('quality_factor', ascending=False).head(top_n)
        portfolios[year_val] = selected_for_portfolio[['stock_code', 'quality_factor']]

    return portfolios


def simulate_rebalancing(portfolio_by_year, full_test_data, initial_capital=1e8):
    """Simulates portfolio rebalancing and calculates daily portfolio returns."""
    all_daily_portfolio_returns_list = [] 

    if not isinstance(full_test_data, pd.DataFrame) or full_test_data.empty:
        return pd.Series(dtype=float)
        
    # Ensure 'date' in full_test_data is datetime and set as index for pivoting
    if 'date' not in full_test_data.columns:
        # This case should ideally not happen if data collection and preprocessing are correct
        return pd.Series(dtype=float)
    if not pd.api.types.is_datetime64_any_dtype(full_test_data['date']):
        full_test_data['date'] = pd.to_datetime(full_test_data['date'])

    # Pivot stock returns for easy lookup: Index=date, Columns=stock_code, Values=return
    # Ensure no duplicate index entries before pivoting if 'date' and 'stock_code' are not unique together for 'return'
    # Assuming 'return' is daily return, so ('date', 'stock_code') should be unique key.
    try:
        daily_stock_returns_pivot = full_test_data.pivot_table(
            index='date', columns='stock_code', values='return'
        )
    except Exception as e:
        # Handle cases where pivot might fail (e.g. duplicate entries for a stock on a date)
        # A more robust way: group by date and stock_code, take mean (or first) return
        # For now, assume data is clean enough for pivot.
        print(f"Error pivoting data: {e}. Using groupby as fallback.")
        daily_stock_returns_pivot = full_test_data.groupby(['date', 'stock_code'])['return'].mean().unstack()

    daily_stock_returns_pivot.fillna(0, inplace=True) # Fill missing daily returns with 0

    sorted_years = sorted(portfolio_by_year.keys())

    for year_idx, year in enumerate(sorted_years):
        selected_stocks_info = portfolio_by_year[year]
        
        if not isinstance(selected_stocks_info, pd.DataFrame) or selected_stocks_info.empty or 'stock_code' not in selected_stocks_info.columns:
            # If no stocks selected for this year, assume 0 returns for this year's segment
            year_dates = daily_stock_returns_pivot.loc[daily_stock_returns_pivot.index.year == year].index
            if not year_dates.empty:
                year_portfolio_returns = pd.Series(0.0, index=year_dates)
                all_daily_portfolio_returns_list.append(year_portfolio_returns)
            continue

        selected_stock_codes = selected_stocks_info['stock_code'].unique().tolist()
        if not selected_stock_codes:
            year_dates = daily_stock_returns_pivot.loc[daily_stock_returns_pivot.index.year == year].index
            if not year_dates.empty:
                year_portfolio_returns = pd.Series(0.0, index=year_dates)
                all_daily_portfolio_returns_list.append(year_portfolio_returns)
            continue
        
        # Filter daily_stock_returns_pivot for the current year and selected stocks
        # Ensure selected_stock_codes exist in pivot table columns
        valid_cols = [col for col in selected_stock_codes if col in daily_stock_returns_pivot.columns]
        if not valid_cols:
            year_dates = daily_stock_returns_pivot.loc[daily_stock_returns_pivot.index.year == year].index
            if not year_dates.empty:
                year_portfolio_returns = pd.Series(0.0, index=year_dates)
                all_daily_portfolio_returns_list.append(year_portfolio_returns)
            continue
            
        current_year_stock_returns = daily_stock_returns_pivot.loc[
            daily_stock_returns_pivot.index.year == year,
            valid_cols
        ]
        
        if current_year_stock_returns.empty:
            continue

        # Calculate equal-weighted portfolio daily return
        year_portfolio_returns = current_year_stock_returns.mean(axis=1)

        # Apply transaction cost at the beginning of the period (rebalance day)
        if year_idx > 0: # Not the first year, so rebalancing occurred
            if not year_portfolio_returns.empty:
                # Cost applied to first day's return of this new period.
                # Assumes config.TRANSACTION_COST is one-way cost for simplicity.
                year_portfolio_returns.iloc[0] -= config.TRANSACTION_COST 
                
        all_daily_portfolio_returns_list.append(year_portfolio_returns)

    if not all_daily_portfolio_returns_list:
        return pd.Series(dtype=float)

    final_portfolio_returns = pd.concat(all_daily_portfolio_returns_list).sort_index()
    return final_portfolio_returns


def calculate_performance_metrics(daily_portfolio_returns):
    if not isinstance(daily_portfolio_returns, pd.Series) or daily_portfolio_returns.empty or daily_portfolio_returns.isnull().all():
        return {
            'cumulative_return': 0.0,
            'cagr': 0.0,
            'sharpe': 0.0,
            'mdd': 0.0,
            'cumulative_returns_series': pd.Series(dtype=float) # Empty series
        }

    # Ensure returns are numeric and handle potential NaNs from calculations
    daily_portfolio_returns = pd.to_numeric(daily_portfolio_returns, errors='coerce').fillna(0.0)

    # Cumulative returns series (starts from 1, representing initial capital)
    cumulative_returns_series = (1 + daily_portfolio_returns).cumprod()
    
    final_cumulative_return = cumulative_returns_series.iloc[-1] - 1
    
    num_days = len(daily_portfolio_returns)
    if num_days == 0:
        return { # Should have been caught by earlier check, but as a safeguard
            'cumulative_return': 0.0, 'cagr': 0.0, 'sharpe': 0.0, 'mdd': 0.0,
            'cumulative_returns_series': pd.Series(dtype=float)
        }
        
    num_years = num_days / 252.0 # Assuming 252 trading days per year
    
    if num_years == 0: 
        cagr = 0.0
    else:
        # CAGR: (End Value / Start Value)^(1/NumYears) - 1. Start Value is 1.
        cagr = (cumulative_returns_series.iloc[-1]) ** (1.0 / num_years) - 1

    mean_return = daily_portfolio_returns.mean()
    std_return = daily_portfolio_returns.std()

    if std_return == 0 or pd.isna(std_return): # Avoid division by zero or NaN std dev
        sharpe = 0.0 
    else:
        sharpe = np.sqrt(252) * mean_return / std_return
    
    # Max Drawdown (MDD)
    # MDD = (Peak - Trough) / Peak. Result is negative or zero. We report absolute value.
    roll_max = cumulative_returns_series.cummax()
    daily_drawdown = cumulative_returns_series / roll_max - 1.0
    mdd = daily_drawdown.min() 
    mdd = abs(mdd) if not pd.isna(mdd) else 0.0

    return {
        'cumulative_return': final_cumulative_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'mdd': mdd,
        'cumulative_returns_series': cumulative_returns_series 
    }
