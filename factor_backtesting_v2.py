"""
Factor Backtesting Framework v2.2
Daily Data-Based Backtesting with 4 Core Strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import glob
import os
import yaml
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy import stats
import yfinance as yf
from pathlib import Path
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pykrx")

try:
    from pykrx import bond
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False
    print("Warning: pykrx not available. Using fallback risk-free rate.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Using matplotlib for visualization.")

# Korean font settings
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':  # Windows
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class DataHandler:
    """Data processing and master dataframe creation"""
    
    def __init__(self, config):
        self.config = config
        self.daily_price_df = None
        self.fs_df = None
        self.market_cap_df = None
        self.master_df = None
        
    def load_data(self):
        """Load and process all required data"""
        print("📊 데이터 로딩 시작...")
        
        # 1. Load daily price data
        self._load_daily_price_data()
        
        # 2. Load financial statement data
        self._load_financial_data()
        
        # 3. Load market cap data
        self._load_market_cap_data()
        
        # 4. Create master dataframe
        self._create_master_dataframe()
        
        print("✅ 데이터 로딩 완료")
        
    def _load_daily_price_data(self):
        """Load daily price data from multiple CSV files"""
        print("  📈 일간 주가 데이터 로딩 중...")
        
        price_files = glob.glob(os.path.join(self.config['data_paths']['price_data_dir'], '*.csv'))
        price_dfs = []
        
        for file in price_files:
            try:
                df = pd.read_csv(file, encoding='utf-8-sig')
                # Standardize column names
                df['date'] = pd.to_datetime(df['매매년월일'], errors='coerce')
                df['일간_시가총액'] = pd.to_numeric(df['시가총액(원)'], errors='coerce')
                df['거래소코드'] = df['거래소코드'].astype(str)
                
                # Extract year from filename or date
                if '회계년도' in df.columns:
                    df['회계년도'] = pd.to_numeric(df['회계년도'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
                
                price_dfs.append(df)
                
            except Exception as e:
                print(f"    ⚠️ 파일 로딩 실패: {file} - {e}")
                continue
        
        if price_dfs:
            self.daily_price_df = pd.concat(price_dfs, ignore_index=True)
            self.daily_price_df = self.daily_price_df.dropna(subset=['date'])
            self.daily_price_df = self.daily_price_df.sort_values(['거래소코드', 'date'])
            print(f"    ✅ 일간 데이터 로딩 완료: {len(self.daily_price_df):,}행")
        else:
            raise ValueError("일간 주가 데이터를 로드할 수 없습니다.")
            
    def _load_financial_data(self):
        """Load financial statement data"""
        print("  📊 재무제표 데이터 로딩 중...")
        
        fs_path = self.config['data_paths']['fundamental']
        self.fs_df = pd.read_csv(fs_path, encoding='utf-8-sig')
        
        # Rename '연도' to '회계년도' if '회계년도' doesn't exist but '연도' does
        if '회계년도' not in self.fs_df.columns and '연도' in self.fs_df.columns:
            self.fs_df.rename(columns={'연도': '회계년도'}, inplace=True)
        
        # Standardize account year column name
        if '회계년도' in self.fs_df.columns:
            self.fs_df['회계년도'] = pd.to_numeric(self.fs_df['회계년도'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
        
        self.fs_df['거래소코드'] = self.fs_df['거래소코드'].astype(str)
        print(f"    ✅ 재무제표 데이터 로딩 완료: {len(self.fs_df):,}행")
        
    def _load_market_cap_data(self):
        """Load annual market cap data"""
        print("  💰 시가총액 데이터 로딩 중...")
        
        cap_path = self.config['data_paths']['market_cap']
        self.market_cap_df = pd.read_csv(cap_path, encoding='utf-8-sig')
        
        # Rename market cap column to distinguish from daily data
        if '시가총액' in self.market_cap_df.columns:
            self.market_cap_df.rename(columns={'시가총액': '연간_시가총액'}, inplace=True)
        
        if '회계년도' in self.market_cap_df.columns:
            self.market_cap_df['회계년도'] = pd.to_numeric(self.market_cap_df['회계년도'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
        
        self.market_cap_df['거래소코드'] = self.market_cap_df['거래소코드'].astype(str)
        print(f"    ✅ 시가총액 데이터 로딩 완료: {len(self.market_cap_df):,}행")
        
    def _create_master_dataframe(self):
        """Create master dataframe with future information bias prevention"""
        print("  🔄 마스터 데이터프레임 생성 중...")
        
        # Merge financial data with market cap data
        fundamental_df = self.fs_df.merge(self.market_cap_df, on=['거래소코드', '회계년도'], how='left')
        
        # Create master dataframe by merging daily data with fundamental data
        # Key rule: Link fundamental data that was available at each trading date
        master_records = []
        
        for date in self.daily_price_df['date'].unique():
            # Get daily data for this date
            daily_data = self.daily_price_df[self.daily_price_df['date'] == date].copy()
            
            # Determine which fiscal year data should be used for this date
            # Rule: Use the most recent fiscal year data available before this date
            current_year = date.year
            
            # For dates from April 1 to March 31, use previous year's fundamental data
            if date.month >= 4:
                fiscal_year = current_year - 1
            else:
                fiscal_year = current_year - 2
                
            # Get fundamental data for the determined fiscal year
            fund_data = fundamental_df[fundamental_df['회계년도'] == fiscal_year]
            # Merge daily data with fundamental data
            merged_data = daily_data.merge(fund_data, on=['거래소코드', '회계년도'], how='left')
            merged_data['rebalance_date'] = date
            
            master_records.append(merged_data)
        
        if master_records:
            self.master_df = pd.concat(master_records, ignore_index=True)
            self.master_df = self.master_df.dropna(subset=['거래소코드', 'date'])
            print(f"    ✅ 마스터 데이터프레임 생성 완료: {len(self.master_df):,}행")
        else:
            raise ValueError("마스터 데이터프레임을 생성할 수 없습니다.")


class FactorEngine:
    """Factor signal calculation engine"""
    
    def __init__(self, config):
        self.config = config
        self.ff3_factors = None
        
    def compute_factors(self, df):
        """Compute all factor signals"""
        print("🔢 팩터 계산 시작...")
        
        # Compute individual factors
        df = self._compute_magic_formula(df)
        df = self._compute_fscore(df)
        df = self._compute_momentum(df)
        
        # Build FF3 factors for FF3-Alpha strategy
        self._build_ff3_factors(df)
        
        print("✅ 팩터 계산 완료")
        return df
        
    def _compute_magic_formula(self, df):
        """Magic Formula computation following existing logic"""
        print("  🪄 Magic Formula 계산 중...")
        
        # Earnings Yield = Operating Income / EV
        # EV = Market Cap + Total Debt - Cash Equivalents
        if '영업이익' in df.columns and '일간_시가총액' in df.columns:
            cash_equiv = df.get('현금및현금성자산', 0) + df.get('단기금융상품(금융기관예치금)', 0)
            ev = df['일간_시가총액'] + df.get('총부채', 0) - cash_equiv
            ev = ev.replace(0, np.nan)
            df['earnings_yield'] = df['영업이익'] / ev
        else:
            df['earnings_yield'] = np.nan
            
        # ROIC calculation
        if '경영자본영업이익률' in df.columns:
            df['roic'] = pd.to_numeric(df['경영자본영업이익률'], errors='coerce') / 100
        elif '영업이익' in df.columns and '총자산' in df.columns:
            total_assets = pd.to_numeric(df['총자산'], errors='coerce').replace(0, np.nan)
            df['roic'] = pd.to_numeric(df['영업이익'], errors='coerce') / total_assets
        else:
            df['roic'] = np.nan
            
        # Calculate rankings by fiscal year
        df['magic_signal'] = np.nan
        
        for year in df['회계년도'].unique():
            if pd.isna(year):
                continue
                
            year_mask = df['회계년도'] == year
            year_df = df[year_mask].copy()
            
            valid_mask = year_df['earnings_yield'].notna() & year_df['roic'].notna()
            if valid_mask.sum() < 5:
                continue
                
            # Rankings (higher is better)
            year_df.loc[valid_mask, 'ey_rank'] = year_df.loc[valid_mask, 'earnings_yield'].rank(ascending=False)
            year_df.loc[valid_mask, 'roic_rank'] = year_df.loc[valid_mask, 'roic'].rank(ascending=False)
            
            # Combined rank (lower is better)
            year_df.loc[valid_mask, 'combined_rank'] = (
                year_df.loc[valid_mask, 'ey_rank'] + year_df.loc[valid_mask, 'roic_rank']
            )
            year_df.loc[valid_mask, 'magic_signal'] = -year_df.loc[valid_mask, 'combined_rank']
            
            df.loc[year_mask, 'magic_signal'] = year_df['magic_signal']
            
        return df
        
    def _compute_fscore(self, df):
        """F-Score computation following existing logic"""
        print("  📊 F-Score 계산 중...")
        
        # Initialize F-Score components
        fscore_components = []
        
        # 1. ROA > 0
        if 'ROA' in df.columns:
            df['f_roa'] = (pd.to_numeric(df['ROA'], errors='coerce') > 0).astype(int)
        elif '총자산수익률' in df.columns:
            df['f_roa'] = (pd.to_numeric(df['총자산수익률'], errors='coerce') > 0).astype(int)
        elif '당기순이익' in df.columns and '총자산' in df.columns:
            total_assets = pd.to_numeric(df['총자산'], errors='coerce').replace(0, np.nan)
            roa = pd.to_numeric(df['당기순이익'], errors='coerce') / total_assets
            df['f_roa'] = (roa > 0).astype(int)
        else:
            df['f_roa'] = 0
        fscore_components.append('f_roa')
        
        # 2. CFO > 0
        cfo_col = '영업현금흐름' if '영업현금흐름' in df.columns else '영업CF'
        if cfo_col in df.columns:
            df['f_cfo'] = (pd.to_numeric(df[cfo_col], errors='coerce') > 0).astype(int)
        else:
            df['f_cfo'] = 0
        fscore_components.append('f_cfo')
        
        # 3. ΔROA (Change in ROA)
        if '총자산수익률' in df.columns:
            df['f_delta_roa'] = (df.groupby('거래소코드')['총자산수익률'].diff() > 0).astype(int)
        else:
            df['f_delta_roa'] = 0
        fscore_components.append('f_delta_roa')
        
        # 4. CFO > ROA
        if cfo_col in df.columns:
            if '총자산수익률' in df.columns:
                roa_values = pd.to_numeric(df['총자산수익률'], errors='coerce')
                cfo_values = pd.to_numeric(df[cfo_col], errors='coerce')
                total_assets = pd.to_numeric(df.get('avg_총자산', df.get('총자산', 1)), errors='coerce').replace(0, np.nan)
                cfo_ta = cfo_values / total_assets
                # ROA가 퍼센트면 100으로 나누기
                if roa_values.max() > 1:
                    roa_values = roa_values / 100
                df['f_cfo_roa'] = (cfo_ta > roa_values).astype(int)
            else:
                df['f_cfo_roa'] = 0
        else:
            df['f_cfo_roa'] = 0
        fscore_components.append('f_cfo_roa')
        
        # 5. Δ부채 (Debt decrease)
        if '총부채' in df.columns:
            df['f_debt'] = (df.groupby('거래소코드')['총부채'].diff() < 0).astype(int)
        else:
            df['f_debt'] = 0
        fscore_components.append('f_debt')
        
        # 6. Δ유동비율 (Liquidity improvement)
        if '현금및현금성자산' in df.columns and '재고자산' in df.columns and '유동부채' in df.columns:
            liquid_assets = (pd.to_numeric(df['현금및현금성자산'], errors='coerce').fillna(0) + 
                           pd.to_numeric(df['재고자산'], errors='coerce').fillna(0))
            liabilities = pd.to_numeric(df['유동부채'], errors='coerce').replace(0, np.nan)
            approx_ratio = liquid_assets / liabilities
            df['f_liquid'] = (approx_ratio.groupby(df['거래소코드']).diff() > 0).astype(int)
        else:
            df['f_liquid'] = 0
        fscore_components.append('f_liquid')
        
        # 7. 신주발행 없음 (No share issuance)
        if '자본금' in df.columns:
            capital_change = df.groupby('거래소코드')['자본금'].diff()
            df['f_shares'] = ((capital_change <= 0) & (~capital_change.isna())).astype(int)
        else:
            df['f_shares'] = 0
        fscore_components.append('f_shares')
        
        # 8. Δ마진 (Gross margin improvement)
        if '매출총이익' in df.columns and '매출액' in df.columns:
            gross_margin = pd.to_numeric(df['매출총이익'], errors='coerce') / pd.to_numeric(df['매출액'], errors='coerce').replace(0, np.nan)
            df['f_margin'] = (gross_margin.groupby(df['거래소코드']).diff() > 0).astype(int)
        else:
            df['f_margin'] = 0
        fscore_components.append('f_margin')
        
        # 9. Δ회전율 (Asset turnover improvement)
        if '총자산회전율' in df.columns:
            df['f_turnover'] = (df.groupby('거래소코드')['총자산회전율'].diff() > 0).astype(int)
        else:
            df['f_turnover'] = 0
        fscore_components.append('f_turnover')
        
        # Calculate total F-Score
        df['fscore'] = df[fscore_components].sum(axis=1)
        
        return df
    
    @staticmethod
    def _process_momentum_chunk(chunk_data):
        """Process a chunk of stock codes for momentum calculation"""
        chunk, df, lookback_months, skip_months, chunk_num, total_chunks = chunk_data
        
        print(f"    📈 모멘텀 계산 청크 {chunk_num}/{total_chunks} 처리 중... ({len(chunk)}개 종목)")
        
        chunk_results = []
        for code in chunk:
            stock_data = df[df['거래소코드'] == code].copy()
            stock_data = stock_data.sort_values('date')
            
            # Calculate momentum returns
            stock_data['momentum'] = np.nan
            
            for i in range(len(stock_data)):
                current_date = stock_data.iloc[i]['date']
                
                # Skip period
                lookback_date = current_date - relativedelta(months=lookback_months)
                
                # Find prices
                current_price = stock_data.iloc[i]['종가'] if '종가' in stock_data.columns else stock_data.iloc[i]['일간_시가총액']
                
                # Get price at lookback date
                past_data = stock_data[stock_data['date'] <= lookback_date]
                if len(past_data) > 0:
                    past_price = past_data.iloc[-1]['종가'] if '종가' in past_data.columns else past_data.iloc[-1]['일간_시가총액']
                    
                    if past_price > 0:
                        momentum = (current_price / past_price) - 1
                        stock_data.iloc[i, stock_data.columns.get_loc('momentum')] = momentum
            
            chunk_results.append(stock_data)
        
        return chunk_results
    
    @staticmethod
    def _process_ff3_alpha_chunk(chunk_data):
        """Process a chunk of stock codes for FF3 Alpha calculation"""
        chunk, df, ff3_factors, regression_window, alpha_threshold, chunk_num, total_chunks = chunk_data
        
        print(f"    📊 FF3-Alpha 계산 청크 {chunk_num}/{total_chunks} 처리 중... ({len(chunk)}개 종목)")
        
        chunk_alpha_results = []
        
        for code in chunk:
            try:
                # Get stock monthly returns for regression window
                stock_data = df[df['거래소코드'] == code].copy()
                stock_data = stock_data.sort_values('date')
                
                # Calculate monthly returns
                stock_data['month_year'] = stock_data['date'].dt.to_period('M')
                monthly_returns = stock_data.groupby('month_year').last()
                
                if len(monthly_returns) < 12:  # Minimum 12 months
                    continue
                    
                # Get last N months of data
                recent_returns = monthly_returns.tail(min(regression_window, len(monthly_returns)))
                
                if len(recent_returns) < 12:
                    continue
                
                # Calculate stock excess returns
                stock_returns = recent_returns['일간_시가총액'].pct_change().dropna()
                
                # Merge with FF3 factors
                if ff3_factors is not None and len(ff3_factors) > 0:
                    factor_data = ff3_factors.copy()
                    factor_data.index = factor_data.index.to_period('M')
                    
                    merged_data = pd.merge(stock_returns.to_frame('Stock_Return'), 
                                         factor_data, 
                                         left_index=True, 
                                         right_index=True, 
                                         how='inner')
                    
                    if len(merged_data) < 12:
                        continue
                    
                    # Calculate excess returns
                    merged_data['Stock_Excess'] = merged_data['Stock_Return'] - merged_data.get('RF', 0)
                    
                    # Run FF3 regression
                    X = merged_data[['Mkt_RF', 'SMB', 'HML']].fillna(0)
                    y = merged_data['Stock_Excess'].fillna(0)
                    
                    if len(X) >= 12 and len(y) >= 12:
                        from scipy.stats import linregress
                        slope, intercept, r_value, p_value, std_err = linregress(X.sum(axis=1), y)
                        
                        if p_value < alpha_threshold:
                            chunk_alpha_results.append({
                                'code': code,
                                'alpha': intercept,
                                'p_value': p_value,
                                'r_squared': r_value**2
                            })
                
            except Exception as e:
                continue
        
        return chunk_alpha_results
        
    def _compute_momentum(self, df):
        """Momentum computation using daily price data with multiprocessing"""
        print("  📈 모멘텀 계산 중...")
        
        lookback_months = self.config['strategy_params']['momentum']['lookback_period']
        skip_months = self.config['strategy_params']['momentum']['skip_period']

        # Calculate momentum returns
        df = df.sort_values(['거래소코드', 'date'])
        
        unique_codes = df['거래소코드'].unique()
        total_stocks = len(unique_codes)
        
        print(f"    📊 총 {total_stocks}개 종목의 모멘텀 계산 시작...")
        
        # Use multiprocessing to calculate momentum for each stock
        num_processes = min(cpu_count() - 1, 8)  # Use all cores except 1, max 8 processes
        print(f"    🔄 {num_processes}개 프로세스 사용")
        
        # Split stock codes into chunks for progress tracking
        chunk_size = max(1, total_stocks // (num_processes * 4))
        code_chunks = [unique_codes[i:i+chunk_size] for i in range(0, len(unique_codes), chunk_size)]
        
        with Pool(processes=num_processes) as pool:
            chunk_results = pool.map(self._process_momentum_chunk, 
                                   [(chunk, df, lookback_months, skip_months, i+1, len(code_chunks)) 
                                    for i, chunk in enumerate(code_chunks)])
        
        # Combine results
        momentum_results = []
        for chunk_result in chunk_results:
            momentum_results.extend(chunk_result)
            
        if momentum_results:
            df = pd.concat(momentum_results)
            df['mom'] = df['momentum'].fillna(0)
        else:
            df['mom'] = 0
            
        return df
        
    def _build_ff3_factors(self, df):
        """Build Fama-French 3-Factor time series"""
        print("  📊 FF3 팩터 구축 중...")
        
        try:
            # Get market returns (KOSPI)
            market_returns = self._get_market_returns()
            
            # Get risk-free rate
            rf_returns = self._get_risk_free_rate()
            
            # Build SMB and HML factors
            smb_hml_factors = self._build_smb_hml_factors(df)
            
            # Combine all factors
            self.ff3_factors = pd.merge(market_returns, rf_returns, left_index=True, right_index=True, how='outer')
            self.ff3_factors = pd.merge(self.ff3_factors, smb_hml_factors, left_index=True, right_index=True, how='outer')
            
            # Calculate market premium
            self.ff3_factors['Mkt_RF'] = self.ff3_factors['Market_Return'] - self.ff3_factors['RF']
            
            print("    ✅ FF3 팩터 구축 완료")
            
        except Exception as e:
            print(f"    ⚠️ FF3 팩터 구축 실패: {e}")
            
    def _get_market_returns(self):
        """Get KOSPI market returns using yfinance"""
        try:
            kospi = yf.download(self.config['benchmark_ticker'], 
                              start='2010-01-01', 
                              end='2024-12-31',
                              progress=False)
            if 'Adj Close' not in kospi.columns:
                print("    ⚠️ 'Adj Close' 컬럼이 없음, 'Close' 컬럼 사용")
                kospi['Adj Close'] = kospi['Close']
            
            monthly_prices = kospi['Adj Close'].resample('M').last()
            monthly_returns = monthly_prices.pct_change()
            return monthly_returns.dropna().to_frame('Market_Return')
        except Exception as e:
            print(f"    ⚠️ 시장 수익률 데이터 로딩 실패: {e}")
            return pd.DataFrame()
            
    def _get_risk_free_rate(self):
        """Get risk-free rate using pykrx or fallback"""
        try:
            if PYKRX_AVAILABLE:
                # Try to get CD 91-day rate from pykrx
                rf_data = bond.get_otc_treasury_yields('2010-01-01', '2024-12-31')
                if 'CD(91일)' in rf_data.columns:
                    rf_monthly = rf_data['CD(91일)'].resample('M').last() / 100 / 12
                    return rf_monthly.to_frame('RF')
            
            # Fallback to config value
            fallback_rate = self.config.get('risk_free_rate_fallback', 0.02) / 12
            dates = pd.date_range('2010-01-01', '2024-12-31', freq='M')
            return pd.DataFrame({'RF': fallback_rate}, index=dates)
            
        except Exception as e:
            print(f"    ⚠️ 무위험 이자율 데이터 로딩 실패: {e}")
            fallback_rate = self.config.get('risk_free_rate_fallback', 0.02) / 12
            dates = pd.date_range('2010-01-01', '2024-12-31', freq='M')
            return pd.DataFrame({'RF': fallback_rate}, index=dates)
            
    def _build_smb_hml_factors(self, df):
        """Build SMB and HML factors"""
        try:
            # Portfolio formation happens every June
            smb_hml_data = []
            
            for year in range(2012, 2024):
                if 'date' not in df.columns:
                    df['date'] = pd.to_datetime(df['회계년도'].astype(str) + '-12-31')
                
                june_data = df[df['회계년도'] == year]
                
                if len(june_data) == 0:
                    continue
                    
                # Size portfolios (50% split) - use 시가총액 if available
                size_col = '시가총액' if '시가총액' in june_data.columns else 'avg_시가총액'
                if size_col not in june_data.columns:
                    print(f"    ⚠️ 시가총액 컬럼 없음: {year}년")
                    continue
                    
                size_median = june_data[size_col].median()
                june_data = june_data.copy()
                june_data['size_group'] = np.where(june_data[size_col] >= size_median, 'Big', 'Small')
                
                # Value portfolios (30-40-30 split) - book-to-market ratio
                if '총자본' in june_data.columns and size_col in june_data.columns:
                    june_data['bm_ratio'] = june_data['총자본'] / june_data[size_col]
                    june_data['bm_ratio'] = june_data['bm_ratio'].replace([np.inf, -np.inf], np.nan)
                    june_data = june_data.dropna(subset=['bm_ratio'])
                    
                    if len(june_data) == 0:
                        continue
                        
                    bm_30 = june_data['bm_ratio'].quantile(0.3)
                    bm_70 = june_data['bm_ratio'].quantile(0.7)
                    
                    if bm_30 >= bm_70:
                        print(f"    ⚠️ BM 비율 정렬 오류: {year}년")
                        continue
                    
                    june_data['value_group'] = pd.cut(june_data['bm_ratio'], 
                                                    bins=[-np.inf, bm_30, bm_70, np.inf],
                                                    labels=['Low', 'Medium', 'High'])
                    
                    # Calculate portfolio returns (simplified annual returns)
                    portfolio_returns = {}
                    
                    for size in ['Small', 'Big']:
                        for value in ['Low', 'Medium', 'High']:
                            portfolio_stocks = june_data[
                                (june_data['size_group'] == size) & 
                                (june_data['value_group'] == value)
                            ]
                            
                            if len(portfolio_stocks) > 0:
                                # Use total asset return as proxy for portfolio return
                                if '총자산수익률' in portfolio_stocks.columns:
                                    portfolio_return = portfolio_stocks['총자산수익률'].mean()
                                else:
                                    portfolio_return = 0.0
                                portfolio_returns[f'{size}_{value}'] = portfolio_return
                    
                    # Calculate SMB and HML
                    if len(portfolio_returns) >= 4:
                        small_avg = np.mean([portfolio_returns.get(f'Small_{v}', 0) for v in ['Low', 'Medium', 'High']])
                        big_avg = np.mean([portfolio_returns.get(f'Big_{v}', 0) for v in ['Low', 'Medium', 'High']])
                        
                        high_avg = np.mean([portfolio_returns.get(f'{s}_High', 0) for s in ['Small', 'Big']])
                        low_avg = np.mean([portfolio_returns.get(f'{s}_Low', 0) for s in ['Small', 'Big']])
                        
                        smb = small_avg - big_avg
                        hml = high_avg - low_avg
                        
                        current_date = pd.to_datetime(f'{year}-12-31')
                        smb_hml_data.append({
                            'date': current_date,
                            'SMB': smb,
                            'HML': hml
                        })
            
            if smb_hml_data:
                smb_hml_df = pd.DataFrame(smb_hml_data)
                smb_hml_df.set_index('date', inplace=True)
                return smb_hml_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"    ⚠️ SMB/HML 팩터 계산 실패: {e}")
            return pd.DataFrame()


class StrategyBuilder:
    """Portfolio construction for different strategies"""
    
    def __init__(self, config):
        self.config = config
        self.portfolio_size = config['portfolio_params']['portfolio_size']
        self.weighting_scheme = config['portfolio_params']['weighting_scheme']
        
    def build_portfolios(self, df, factor_engine):
        """Build portfolios for all strategies"""
        print("🎯 포트폴리오 구성 시작...")
        
        strategies = {
            'Magic_Formula': self._build_magic_formula_portfolio,
            'F_Score': self._build_fscore_portfolio,
            'Momentum': self._build_momentum_portfolio,
            'FF3_Alpha': self._build_ff3_alpha_portfolio
        }
        
        portfolio_results = {}
        
        for strategy_name, strategy_func in strategies.items():
            print(f"  📊 {strategy_name} 포트폴리오 구성 중...")
            try:
                portfolio_results[strategy_name] = strategy_func(df, factor_engine)
                print(f"    ✅ {strategy_name} 완료")
            except Exception as e:
                print(f"    ❌ {strategy_name} 실패: {e}")
                continue
        
        return portfolio_results
        
    def _build_magic_formula_portfolio(self, df, factor_engine):
        """Build Magic Formula portfolio"""
        portfolios = {'All': [], 'Normal': []}
        
        # Magic Formula rebalances on April 1st
        rebalance_dates = pd.date_range(start=self.config['start_date'], 
                                      end=self.config['end_date'], 
                                      freq='AS-APR')
        
        for rebalance_date in rebalance_dates:
            # Get data for rebalancing date
            rebalance_data = df[df['date'] == rebalance_date]
            
            if len(rebalance_data) == 0:
                continue
                
            # All firms portfolio
            valid_data = rebalance_data[rebalance_data['magic_signal'].notna()]
            if len(valid_data) >= self.portfolio_size:
                top_stocks = valid_data.nlargest(self.portfolio_size, 'magic_signal')
                portfolios['All'].append({
                    'date': rebalance_date,
                    'stocks': top_stocks['거래소코드'].tolist(),
                    'signals': top_stocks['magic_signal'].tolist()
                })
            
            # Normal firms portfolio (default == 0)
            normal_data = valid_data[valid_data['default'] == 0]
            if len(normal_data) >= self.portfolio_size:
                top_normal_stocks = normal_data.nlargest(self.portfolio_size, 'magic_signal')
                portfolios['Normal'].append({
                    'date': rebalance_date,
                    'stocks': top_normal_stocks['거래소코드'].tolist(),
                    'signals': top_normal_stocks['magic_signal'].tolist()
                })
        
        return portfolios
        
    def _build_fscore_portfolio(self, df, factor_engine):
        """Build F-Score portfolio"""
        portfolios = {'All': [], 'Normal': []}
        min_score = self.config['strategy_params']['f_score']['min_score']
        
        # F-Score rebalances on April 1st
        rebalance_dates = pd.date_range(start=self.config['start_date'], 
                                      end=self.config['end_date'], 
                                      freq='AS-APR')
        
        for rebalance_date in rebalance_dates:
            rebalance_data = df[df['date'] == rebalance_date]
            
            if len(rebalance_data) == 0:
                continue
                
            # All firms portfolio
            valid_data = rebalance_data[rebalance_data['fscore'] >= min_score]
            if len(valid_data) > 0:
                portfolios['All'].append({
                    'date': rebalance_date,
                    'stocks': valid_data['거래소코드'].tolist(),
                    'signals': valid_data['fscore'].tolist()
                })
            
            # Normal firms portfolio
            normal_data = valid_data[valid_data['default'] == 0]
            if len(normal_data) > 0:
                portfolios['Normal'].append({
                    'date': rebalance_date,
                    'stocks': normal_data['거래소코드'].tolist(),
                    'signals': normal_data['fscore'].tolist()
                })
        
        return portfolios
        
    def _build_momentum_portfolio(self, df, factor_engine):
        """Build Momentum portfolio"""
        portfolios = {'All': [], 'Normal': []}
        
        # Momentum rebalances on April 1st
        rebalance_dates = pd.date_range(start=self.config['start_date'], 
                                      end=self.config['end_date'], 
                                      freq='AS-APR')
        
        for rebalance_date in rebalance_dates:
            rebalance_data = df[df['date'] == rebalance_date]
            
            if len(rebalance_data) == 0:
                continue
                
            # All firms portfolio
            valid_data = rebalance_data[rebalance_data['mom'].notna()]
            if len(valid_data) >= self.portfolio_size:
                top_stocks = valid_data.nlargest(self.portfolio_size, 'mom')
                portfolios['All'].append({
                    'date': rebalance_date,
                    'stocks': top_stocks['거래소코드'].tolist(),
                    'signals': top_stocks['mom'].tolist()
                })
            
            # Normal firms portfolio
            normal_data = valid_data[valid_data['default'] == 0]
            if len(normal_data) >= self.portfolio_size:
                top_normal_stocks = normal_data.nlargest(self.portfolio_size, 'mom')
                portfolios['Normal'].append({
                    'date': rebalance_date,
                    'stocks': top_normal_stocks['거래소코드'].tolist(),
                    'signals': top_normal_stocks['mom'].tolist()
                })
        
        return portfolios
        
    def _build_ff3_alpha_portfolio(self, df, factor_engine):
        """Build FF3-Alpha portfolio"""
        portfolios = {'All': [], 'Normal': []}
        
        if factor_engine.ff3_factors is None or len(factor_engine.ff3_factors) == 0:
            print("    ⚠️ FF3 팩터 데이터가 없어 FF3-Alpha 포트폴리오 구성을 건너뜁니다.")
            return portfolios
        
        # FF3-Alpha rebalances on July 1st
        rebalance_dates = pd.date_range(start=self.config['start_date'], 
                                      end=self.config['end_date'], 
                                      freq='AS-JUL')
        
        regression_window = self.config['strategy_params']['ff3_alpha']['regression_window']
        alpha_threshold = self.config['strategy_params']['ff3_alpha']['alpha_pvalue_threshold']
        
        total_dates = len(rebalance_dates)
        
        for idx, rebalance_date in enumerate(rebalance_dates):
            print(f"    📊 FF3-Alpha 포트폴리오 구성 진행률: {idx+1}/{total_dates} ({(idx+1)/total_dates*100:.1f}%)")
            
            # Get unique stock codes for this rebalance date
            unique_codes = df['거래소코드'].unique()
            total_stocks = len(unique_codes)
            
            # Use multiprocessing for FF3 alpha calculation
            num_processes = min(cpu_count() - 1, 6)  # Use fewer processes for alpha calculation
            print(f"    🔄 {num_processes}개 프로세스로 {total_stocks}개 종목 FF3-Alpha 계산 중...")
            
            # Split stock codes into chunks
            chunk_size = max(1, total_stocks // (num_processes * 2))
            code_chunks = [unique_codes[i:i+chunk_size] for i in range(0, len(unique_codes), chunk_size)]
            
            # Prepare data for multiprocessing
            chunk_data = [(chunk, df, factor_engine.ff3_factors, regression_window, alpha_threshold, i+1, len(code_chunks)) 
                         for i, chunk in enumerate(code_chunks)]
            
            with Pool(processes=num_processes) as pool:
                chunk_alpha_results = pool.map(self._process_ff3_alpha_chunk, chunk_data)
            
            # Combine results
            alpha_results = []
            for chunk_result in chunk_alpha_results:
                alpha_results.extend(chunk_result)
            
            # Continue with the rest of the FF3 Alpha logic
            if len(alpha_results) < 10:
                continue
            
            if len(alpha_results) > 0:
                alpha_df = pd.DataFrame(alpha_results)
                alpha_df = alpha_df.sort_values('alpha', ascending=False)
                
                # Get stock data for this rebalance date
                rebalance_data = df[df['date'] == rebalance_date]
                
                # All firms portfolio
                top_alpha_stocks = alpha_df.head(self.portfolio_size)
                valid_stocks = rebalance_data[rebalance_data['거래소코드'].isin(top_alpha_stocks['code'])]
                
                if len(valid_stocks) > 0:
                    portfolios['All'].append({
                        'date': rebalance_date,
                        'stocks': valid_stocks['거래소코드'].tolist(),
                        'signals': top_alpha_stocks['alpha'].tolist()
                    })
                
                # Normal firms portfolio
                normal_stocks = valid_stocks[valid_stocks['default'] == 0]
                normal_alpha_stocks = alpha_df[alpha_df['code'].isin(normal_stocks['거래소코드'])]
                
                if len(normal_alpha_stocks) > 0:
                    portfolios['Normal'].append({
                        'date': rebalance_date,
                        'stocks': normal_alpha_stocks['code'].tolist(),
                        'signals': normal_alpha_stocks['alpha'].tolist()
                    })
        
        return portfolios


class BacktestEngine:
    """Backtesting simulation engine"""
    
    def __init__(self, config):
        self.config = config
        self.transaction_costs = config['transaction_costs']
        
    def run_backtest(self, portfolios, price_data):
        """Run backtesting simulation"""
        print("🔄 백테스팅 시뮬레이션 시작...")
        
        backtest_results = {}
        
        for strategy_name, strategy_portfolios in portfolios.items():
            print(f"  📊 {strategy_name} 백테스팅 중...")
            
            strategy_results = {}
            for universe in ['All', 'Normal']:
                if universe in strategy_portfolios:
                    universe_results = self._backtest_universe(
                        strategy_portfolios[universe], 
                        price_data,
                        f"{strategy_name}_{universe}"
                    )
                    strategy_results[universe] = universe_results
            
            backtest_results[strategy_name] = strategy_results
            
        return backtest_results
        
    def _backtest_universe(self, portfolio_list, price_data, strategy_name):
        """Run backtest for a specific universe"""
        if len(portfolio_list) == 0:
            return None
            
        # Initialize portfolio
        portfolio_value = 1000000  # Initial capital
        cash = portfolio_value
        holdings = {}
        daily_returns = []
        portfolio_values = []
        
        # Sort price data for efficient lookup
        price_data = price_data.sort_values(['date', '거래소코드'])
        
        for i, portfolio in enumerate(portfolio_list):
            rebalance_date = portfolio['date']
            target_stocks = portfolio['stocks']
            
            # Calculate target position size
            if len(target_stocks) > 0:
                position_size = portfolio_value / len(target_stocks)
                
                # Close existing positions
                for stock_code in list(holdings.keys()):
                    if stock_code not in target_stocks:
                        # Find closing price
                        closing_price_data = price_data[
                            (price_data['거래소코드'] == stock_code) & 
                            (price_data['date'] <= rebalance_date)
                        ]
                        
                        if len(closing_price_data) > 0:
                            closing_price = closing_price_data.iloc[-1]['종가'] if '종가' in closing_price_data.columns else closing_price_data.iloc[-1]['일간_시가총액']
                            
                            # Sell position
                            shares = holdings[stock_code]
                            sale_value = shares * closing_price
                            
                            # Apply transaction costs
                            transaction_cost = sale_value * (
                                self.transaction_costs['commission_rate'] + 
                                self.transaction_costs['tax_rate'] + 
                                self.transaction_costs['slippage_rate']
                            )
                            
                            cash += sale_value - transaction_cost
                            del holdings[stock_code]
                
                # Open new positions
                for stock_code in target_stocks:
                    if stock_code not in holdings:
                        # Find opening price
                        opening_price_data = price_data[
                            (price_data['거래소코드'] == stock_code) & 
                            (price_data['date'] >= rebalance_date)
                        ]
                        
                        if len(opening_price_data) > 0:
                            opening_price = opening_price_data.iloc[0]['종가'] if '종가' in opening_price_data.columns else opening_price_data.iloc[0]['일간_시가총액']
                            
                            if opening_price > 0:
                                # Buy position
                                shares = position_size / opening_price
                                purchase_value = shares * opening_price
                                
                                # Apply transaction costs
                                transaction_cost = purchase_value * (
                                    self.transaction_costs['commission_rate'] + 
                                    self.transaction_costs['slippage_rate']
                                )
                                
                                if cash >= purchase_value + transaction_cost:
                                    cash -= purchase_value + transaction_cost
                                    holdings[stock_code] = shares
            
            # Calculate portfolio value until next rebalance
            next_rebalance_date = portfolio_list[i + 1]['date'] if i + 1 < len(portfolio_list) else pd.to_datetime(self.config['end_date'])
            
            # Get all dates between rebalances
            holding_period_data = price_data[
                (price_data['date'] > rebalance_date) & 
                (price_data['date'] <= next_rebalance_date)
            ]
            
            for date in holding_period_data['date'].unique():
                daily_portfolio_value = cash
                
                for stock_code, shares in holdings.items():
                    stock_price_data = holding_period_data[
                        (holding_period_data['거래소코드'] == stock_code) & 
                        (holding_period_data['date'] == date)
                    ]
                    
                    if len(stock_price_data) > 0:
                        current_price = stock_price_data.iloc[0]['종가'] if '종가' in stock_price_data.columns else stock_price_data.iloc[0]['일간_시가총액']
                        daily_portfolio_value += shares * current_price
                
                portfolio_values.append(daily_portfolio_value)
                daily_returns.append(daily_portfolio_value / portfolio_value - 1 if portfolio_value > 0 else 0)
                portfolio_value = daily_portfolio_value
        
        return {
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns
        }


class PerformanceAnalyzer:
    """Performance analysis and reporting"""
    
    def __init__(self, config):
        self.config = config
        
    def analyze_performance(self, backtest_results):
        """Analyze performance metrics"""
        print("📊 성과 분석 시작...")
        
        performance_metrics = {}
        
        for strategy_name, strategy_results in backtest_results.items():
            print(f"  📈 {strategy_name} 성과 분석 중...")
            
            strategy_metrics = {}
            
            for universe, universe_results in strategy_results.items():
                if universe_results is not None:
                    metrics = self._calculate_metrics(universe_results)
                    strategy_metrics[universe] = metrics
            
            performance_metrics[strategy_name] = strategy_metrics
        
        return performance_metrics
        
    def _calculate_metrics(self, results):
        """Calculate performance metrics"""
        returns = np.array(results['daily_returns'])
        
        if len(returns) == 0:
            return {}
            
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'Total_Return': total_return,
            'CAGR': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Calmar_Ratio': calmar_ratio,
            'Sortino_Ratio': sortino_ratio
        }
        
    def generate_report(self, performance_metrics, output_dir):
        """Generate performance report"""
        print("📋 성과 보고서 생성 중...")
        
        # Create comparison table
        comparison_data = []
        
        for strategy_name, strategy_metrics in performance_metrics.items():
            for universe, metrics in strategy_metrics.items():
                row = {
                    'Strategy': strategy_name,
                    'Universe': universe,
                    **metrics
                }
                comparison_data.append(row)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Save to CSV
            csv_path = os.path.join(output_dir, 'performance_comparison.csv')
            comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # Create HTML report
            html_path = os.path.join(output_dir, 'performance_report.html')
            self._create_html_report(comparison_df, html_path)
            
            print(f"  ✅ 성과 보고서 저장: {csv_path}")
            print(f"  ✅ HTML 보고서 저장: {html_path}")
            
    def _create_html_report(self, comparison_df, html_path):
        """Create HTML performance report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Factor Backtesting Results v2.2</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Factor Backtesting Results v2.2</h1>
            <h2>Performance Comparison</h2>
            {comparison_df.to_html(index=False, classes='table')}
            
            <h2>Report Generated</h2>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


class FactorBacktesterV2:
    """Main Factor Backtesting Framework v2.2"""
    
    def __init__(self, config_path='config.yaml'):
        self.config = self._load_config(config_path)
        self.output_dir = self.config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.data_handler = DataHandler(self.config)
        self.factor_engine = FactorEngine(self.config)
        self.strategy_builder = StrategyBuilder(self.config)
        self.backtest_engine = BacktestEngine(self.config)
        self.performance_analyzer = PerformanceAnalyzer(self.config)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return {
                'data_paths': {
                    'price_data_dir': 'data/raw',
                    'fundamental': 'data/processed/FS2_default.csv',
                    'market_cap': 'data/processed/시가총액.csv'
                },
                'output_dir': 'outputs/backtesting_v2',
                'start_date': '2013-04-01',
                'end_date': '2023-12-31',
                'benchmark_ticker': '^KS11',
                'risk_free_rate_fallback': 0.025,
                'portfolio_params': {
                    'portfolio_size': 20,
                    'weighting_scheme': 'Equal'
                },
                'strategy_params': {
                    'f_score': {
                        'min_score': 8
                    },
                    'ff3_alpha': {
                        'regression_window': 24,
                        'alpha_pvalue_threshold': 0.1
                    },
                    'momentum': {
                        'lookback_period': 12,
                        'skip_period': 1
                    }
                },
                'transaction_costs': {
                    'commission_rate': 0.00015,
                    'tax_rate': 0.0018,
                    'slippage_rate': 0.0005
                }
            }
    
    def run_backtest(self):
        """Run complete backtesting process"""
        print("🚀 Factor Backtesting v2.2 시작")
        print("=" * 50)
        
        try:
            # 1. Data loading and processing
            self.data_handler.load_data()
            
            # 2. Factor computation
            master_df = self.factor_engine.compute_factors(self.data_handler.master_df)
            
            # 3. Portfolio construction
            portfolios = self.strategy_builder.build_portfolios(master_df, self.factor_engine)
            
            # 4. Backtesting simulation
            backtest_results = self.backtest_engine.run_backtest(portfolios, self.data_handler.daily_price_df)
            
            # 5. Performance analysis
            performance_metrics = self.performance_analyzer.analyze_performance(backtest_results)
            
            # 6. Report generation
            self.performance_analyzer.generate_report(performance_metrics, self.output_dir)
            
            print("=" * 50)
            print("✅ 백테스팅 완료!")
            print(f"📁 결과 저장 위치: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ 백테스팅 실패: {e}")
            raise


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Factor Backtesting v2.2')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize and run backtester
    backtester = FactorBacktesterV2(args.config)
    backtester.run_backtest()


if __name__ == "__main__":
    main()