import pandas as pd
import numpy as np

def calculate_financial_variables(bs_ratio_path, final_path, output_path):
    """
    Calculates financial variables using data from BS_ratio.csv and adds them to final.csv.
    Column names are converted to English, and the output is saved as UTF-8.

    Args:
        bs_ratio_path (str): Path to BS_ratio.csv
        final_path (str): Path to final.csv
        output_path (str): Path to save the resulting final.csv
    """
    try:
        # Load data with cp949 encoding, fallback to utf-8
        bs_df = pd.read_csv(bs_ratio_path, encoding='cp949')
        final_df = pd.read_csv(final_path, encoding='cp949')
    except UnicodeDecodeError:
        bs_df = pd.read_csv(bs_ratio_path, encoding='utf-8')
        final_df = pd.read_csv(final_path, encoding='utf-8')

    # --- Rename columns to English ---
    # Rename columns in final_df
    final_df_rename_map = {
        '회사명': 'company_name',
        '거래소코드': 'exchange_code',
        '회계년도': 'fiscal_year'
    }
    final_df.rename(columns=final_df_rename_map, inplace=True)

    # Rename columns in bs_df
    bs_df_rename_map = {
        '회사명': 'company_name',
        '거래소코드': 'exchange_code',
        '회계년도': 'fiscal_year',
        '[A100000000]자산(*)(IFRS연결)(천원)': 'total_assets',
        '[A600000000]자본(*)(IFRS연결)(천원)': 'total_equity',
        '[A800000000]부채(*)(IFRS연결)(천원)': 'total_liabilities',
        '[B420000000]* (정상)영업손익(보고서기재)(IFRS연결)(천원)': 'operating_income',
        '[B840000000]당기순이익(손실)(IFRS연결)(천원)': 'net_income',
        '[D100000000]영업활동으로 인한 현금흐름(간접법)(*)(IFRS연결)(천원)': 'cfo',
        '[A615000000]      이익잉여금(결손금)(*)(IFRS연결)(천원)': 'retained_earnings',
        '[D109010200]   이자지급(-)(IFRS연결)(천원)': 'interest_expense'
    }
    
    # Rename only existing columns in bs_df
    bs_df.rename(columns={k: v for k, v in bs_df_rename_map.items() if k in bs_df.columns}, inplace=True)
    
    # Select only the necessary columns from bs_df to avoid duplicate columns after merge
    bs_cols_to_merge = ['company_name', 'exchange_code', 'fiscal_year'] + [v for k, v in bs_df_rename_map.items() if k in bs_df.columns and v not in ['company_name', 'exchange_code', 'fiscal_year']]
    
    # To prevent duplicate columns, identify columns from bs_df that are also in final_df
    # We will use the data from bs_df as the source of truth.
    cols_from_bs = [col for col in bs_df[bs_cols_to_merge].columns if col not in ['company_name', 'exchange_code', 'fiscal_year']]
    final_df_subset = final_df.drop(columns=[col for col in cols_from_bs if col in final_df.columns], errors='ignore')
    
    merged_df = pd.merge(final_df_subset, bs_df[bs_cols_to_merge], 
                         on=['company_name', 'exchange_code', 'fiscal_year'], how='left')

    # Sort data for time-series calculations
    merged_df.sort_values(by=['exchange_code', 'fiscal_year'], inplace=True)

    # --- Variable Calculation ---
    created_variables = []
    not_created_variables = []

    def safe_division(numerator, denominator):
        num = pd.to_numeric(numerator, errors='coerce')
        den = pd.to_numeric(denominator, errors='coerce')
        return np.where(den != 0, num / den, np.nan)

    # 1. Debt-to-Equity Ratio
    if 'total_liabilities' in merged_df.columns and 'total_equity' in merged_df.columns:
        merged_df['debt_to_equity_ratio'] = safe_division(merged_df['total_liabilities'], merged_df['total_equity']) * 100
        created_variables.append('debt_to_equity_ratio')

    # 3. Return on Assets (ROA)
    if 'net_income' in merged_df.columns and 'total_assets' in merged_df.columns:
        merged_df['roa'] = safe_division(merged_df['net_income'], merged_df['total_assets']) * 100
        created_variables.append('return_on_assets (roa)')

    # 4. Return on Equity (ROE)
    if 'net_income' in merged_df.columns and 'total_equity' in merged_df.columns:
        merged_df['roe'] = safe_division(merged_df['net_income'], merged_df['total_equity']) * 100
        created_variables.append('return_on_equity (roe)')

    # 5. Cash Flow to Debt Ratio
    if 'cfo' in merged_df.columns and 'total_liabilities' in merged_df.columns:
        merged_df['cfo_to_debt_ratio'] = safe_division(merged_df['cfo'], merged_df['total_liabilities'])
        created_variables.append('cfo_to_debt_ratio')

    # 9. Asset Growth Rate
    if 'total_assets' in merged_df.columns:
        merged_df['total_assets_prev_year'] = merged_df.groupby('exchange_code')['total_assets'].shift(1)
        merged_df['asset_growth_rate'] = safe_division(merged_df['total_assets'] - merged_df['total_assets_prev_year'], merged_df['total_assets_prev_year']) * 100
        created_variables.append('asset_growth_rate')

    # 13. Interest Coverage Ratio
    if 'operating_income' in merged_df.columns and 'interest_expense' in merged_df.columns:
        merged_df['interest_coverage_ratio'] = safe_division(merged_df['operating_income'], merged_df['interest_expense'])
        created_variables.append('interest_coverage_ratio')

    # 15. Retained Earnings Ratio
    if 'retained_earnings' in merged_df.columns and 'total_assets' in merged_df.columns:
        merged_df['retained_earnings_ratio'] = safe_division(merged_df['retained_earnings'], merged_df['total_assets'])
        created_variables.append('retained_earnings_ratio')

    # 16. EBIT to Total Assets Ratio (assuming EBIT is operating_income)
    if 'operating_income' in merged_df.columns and 'total_assets' in merged_df.columns:
        merged_df['ebit_to_assets_ratio'] = safe_division(merged_df['operating_income'], merged_df['total_assets'])
        created_variables.append('ebit_to_assets_ratio')

    # 19. Net Income to Total Assets Ratio
    if 'net_income' in merged_df.columns and 'total_assets' in merged_df.columns:
        merged_df['net_income_to_total_assets_ratio'] = safe_division(merged_df['net_income'], merged_df['total_assets'])
        created_variables.append('net_income_to_total_assets_ratio')

    # 20. CFO to Total Liabilities Ratio (same as #5)
    if 'cfo' in merged_df.columns and 'total_liabilities' in merged_df.columns:
        merged_df['cfo_to_total_liabilities_ratio'] = safe_division(merged_df['cfo'], merged_df['total_liabilities'])
        created_variables.append('cfo_to_total_liabilities_ratio')

    # 21. Loss Dummy (INTWO)
    if 'net_income' in merged_df.columns:
        merged_df['net_income_prev_year'] = merged_df.groupby('exchange_code')['net_income'].shift(1)
        merged_df['loss_dummy_intwo'] = ((merged_df['net_income'] < 0) & (merged_df['net_income_prev_year'] < 0)).astype(int)
        created_variables.append('loss_dummy_intwo')

    # 22. Insolvency Dummy (OENEG)
    if 'total_liabilities' in merged_df.columns and 'total_assets' in merged_df.columns:
        merged_df['insolvency_dummy_oeneg'] = (merged_df['total_liabilities'] > merged_df['total_assets']).astype(int)
        created_variables.append('insolvency_dummy_oeneg')

    # 23. Net Income Change Ratio
    if 'net_income' in merged_df.columns:
        # 'net_income_prev_year' was created for #21
        numerator = merged_df['net_income'] - merged_df['net_income_prev_year']
        denominator = merged_df['net_income'].abs() + merged_df['net_income_prev_year'].abs()
        merged_df['net_income_change_ratio'] = safe_division(numerator, denominator)
        created_variables.append('net_income_change_ratio')

    # --- List of variables that could not be created ---
    not_created_variables = [
        'current_ratio', 'debt_turnover', 'gross_profit_margin',
        'operating_margin', 'market_cap', 'stock_return_1yr', 'stock_volatility',
        'working_capital_ratio', 'market_to_book_ratio', 'sales_to_total_assets_ratio'
    ]

    print("### Task Complete ###")
    print("\n[Created Variables]")
    for var in sorted(list(set(created_variables))):
        print(f"- {var}")

    print("\n[Unavailable Variables (due to missing source data)]")
    for var in sorted(list(set(not_created_variables))):
        print(f"- {var}")


    # Clean up temporary columns and save the final file
    final_cols = [col for col in merged_df.columns if not col.endswith('_prev_year')]
    merged_df[final_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to '{output_path}' with UTF-8 encoding.")

if __name__ == '__main__':
    # File paths are relative to the project root directory
    BS_RATIO_PATH = 'data/processed/BS_ratio.csv'
    FINAL_PATH = 'data/processed/final.csv'
    OUTPUT_PATH = 'data/processed/final.csv' # Overwrite original file

    # Execute function
    calculate_financial_variables(BS_RATIO_PATH, FINAL_PATH, OUTPUT_PATH)
