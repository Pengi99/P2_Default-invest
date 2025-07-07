"""
Factor Backtesting Framework v3.0
Daily Data-Based Backtesting with 4 Core Strategies
Enhanced with business day handling, forward-fill, and robust return calculations
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
from pathlib import Path
from multiprocessing import Pool, cpu_count
import multiprocessing

# Optimize multiprocessing start method for better performance on macOS
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Already set

# Performance optimization libraries
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
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
        print("  📈 일간 가격 데이터 로딩...")
        
        # Look for yearly price data files (2013.csv, 2014.csv, etc.)
        price_files = glob.glob(os.path.join(self.config['data_paths']['price_data_dir'], "[0-9][0-9][0-9][0-9].csv"))
        if not price_files:
            print("  ⚠️ 가격 데이터 파일이 없음 - 기본 가격 사용")
            self.daily_price_df = pd.DataFrame()
            return
            
        all_price_data = []
        
        for file in sorted(price_files):  # Process all yearly files
            try:
                print(f"    📁 {os.path.basename(file)} 로딩 중...")
                df = pd.read_csv(file, encoding='utf-8-sig')
                
                # Handle different date column names
                if '매매년월일' in df.columns:
                    df['date'] = pd.to_datetime(df['매매년월일'])
                elif 'Date' in df.columns:
                    df['date'] = pd.to_datetime(df['Date'])
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                elif '날짜' in df.columns:
                    df['date'] = pd.to_datetime(df['날짜'])
                else:
                    print(f"    ⚠️ {file}: 날짜 컬럼을 찾을 수 없음")
                    continue
                
                # Check required columns exist
                if '거래소코드' not in df.columns:
                    print(f"    ⚠️ {file}: 거래소코드 컬럼 없음")
                    continue
                    
                # Handle different price column names
                if '종가(원)' in df.columns:
                    df['종가'] = df['종가(원)']
                elif '종가' in df.columns:
                    pass  # Already has correct column
                elif 'Adj Close' in df.columns:
                    df['종가'] = df['Adj Close']
                elif 'Close' in df.columns:
                    df['종가'] = df['Close']
                else:
                    print(f"    ⚠️ {file}: 종가 컬럼 없음")
                    continue
                    
                # Select necessary columns
                df = df[['거래소코드', 'date', '종가']].copy()
                
                # Remove invalid data
                df = df.dropna(subset=['거래소코드', 'date', '종가'])
                df = df[df['종가'] > 0]  # Remove zero or negative prices
                
                all_price_data.append(df)
                
            except Exception as e:
                print(f"    ⚠️ 파일 로딩 실패 {file}: {e}")
                continue
        
        if all_price_data:
            self.daily_price_df = pd.concat(all_price_data, ignore_index=True)
            self.daily_price_df = self.daily_price_df.sort_values(['거래소코드', 'date'])
            print(f"    ✅ 가격 데이터: {len(self.daily_price_df):,}행, {self.daily_price_df['거래소코드'].nunique()}개 종목")
        else:
            self.daily_price_df = pd.DataFrame()
            print("    ⚠️ 유효한 가격 데이터 없음")
    
    def _load_financial_data(self):
        """Load financial statement data"""
        print("  📋 재무제표 데이터 로딩...")
        
        fs_path = self.config['data_paths']['fundamental']
        try:
            self.fs_df = pd.read_csv(fs_path, encoding='utf-8-sig')
            print(f"    ✅ 재무 데이터: {len(self.fs_df):,}행, {self.fs_df['거래소코드'].nunique()}개 종목")
        except Exception as e:
            print(f"    ❌ 재무 데이터 로딩 실패: {e}")
            self.fs_df = pd.DataFrame()
    
    def _load_market_cap_data(self):
        """Load market cap data"""
        print("  💰 시가총액 데이터 로딩...")
        
        cap_path = self.config['data_paths']['market_cap']
        try:
            self.market_cap_df = pd.read_csv(cap_path, encoding='utf-8-sig')
            
            # Convert date column
            if 'date' in self.market_cap_df.columns:
                self.market_cap_df['date'] = pd.to_datetime(self.market_cap_df['date'])
            elif '날짜' in self.market_cap_df.columns:
                self.market_cap_df['date'] = pd.to_datetime(self.market_cap_df['날짜'])
                
            print(f"    ✅ 시가총액 데이터: {len(self.market_cap_df):,}행")
        except Exception as e:
            print(f"    ❌ 시가총액 데이터 로딩 실패: {e}")
            self.market_cap_df = pd.DataFrame()
    
    def _create_master_dataframe(self):
        """Create master dataframe with all data"""
        print("  🔗 마스터 데이터프레임 생성...")
        
        if self.fs_df.empty:
            print("    ❌ 재무 데이터가 없어 마스터 데이터프레임 생성 불가")
            return
        
        # Start with financial data
        self.master_df = self.fs_df.copy()
        
        # Add date handling - convert year to date
        if '연도' in self.master_df.columns:
            self.master_df['date'] = pd.to_datetime(self.master_df['연도'].astype(str) + '-12-31')
        elif '회계년도' in self.master_df.columns:
            self.master_df['date'] = pd.to_datetime(self.master_df['회계년도'].astype(str) + '-12-31')
        else:
            print("    ⚠️ 연도 컬럼 없음 - 기본 날짜 사용")
            self.master_df['date'] = pd.to_datetime('2023-12-31')
        
        # Simplified master dataframe - avoid memory explosion
        # Just use annual data without daily expansion
        if not self.market_cap_df.empty and '거래소코드' in self.market_cap_df.columns:
            print("    📈 시가총액 데이터 병합...")
            # Simple merge on stock code and year
            self.master_df = pd.merge(self.master_df, self.market_cap_df, 
                                    on=['거래소코드'], how='left', suffixes=('', '_mcap'))
        
        # Skip daily price expansion to avoid memory issues
        # Keep only annual financial data
        print("    💰 가격 데이터는 백테스팅 시 직접 활용...")
            
        # Ensure date column is datetime
        self.master_df['date'] = pd.to_datetime(self.master_df['date'])
        
        # Sort by stock code and date
        self.master_df = self.master_df.sort_values(['거래소코드', 'date']).reset_index(drop=True)
        
        print(f"    ✅ 마스터 데이터: {len(self.master_df):,}행, {self.master_df['거래소코드'].nunique()}개 종목")


class FactorEngine:
    """Factor calculation engine with optimizations"""
    
    def __init__(self, config):
        self.config = config
        self.ff3_factors = None
        
    def compute_factors(self, df):
        """Compute all factors for the given dataframe"""
        print("⚙️ 팩터 계산 시작...")
        
        # Check optimization availability
        self._check_optimizations()
        
        # Compute factors in order
        print("  🪄 Magic Formula 계산...")
        df = self._compute_magic_formula(df)
        
        print("  📊 F-Score 계산...")
        df = self._compute_fscore(df)
        
        print("  📈 모멘텀 계산...")
        df = self._compute_momentum(df)
        
        print("  📊 FF3 팩터 구축...")
        df = self._build_ff3_factors(df)
        
        print("  📊 FF3-Alpha 계산...")
        df = self._compute_ff3_alpha(df)
        
        print("✅ 팩터 계산 완료")
        return df
    
    def _check_optimizations(self):
        """Check which optimizations are available"""
        optimizations = []
        if POLARS_AVAILABLE:
            optimizations.append("Polars")
        if NUMBA_AVAILABLE:
            optimizations.append("Numba")
        if DASK_AVAILABLE:
            optimizations.append("Dask")
            
        if optimizations:
            print(f"    🚀 사용 가능한 최적화: {', '.join(optimizations)}")
        else:
            print("    ⚠️ 최적화 라이브러리 없음 - 기본 pandas 사용")
    
    def _compute_magic_formula(self, df):
        """
        Compute Magic Formula factor
        
        수정 사항:
        1. Earnings Yield = EBIT/EV (기존: 영업이익/시가총액)
           - EV = Market Cap + Total Debt - Cash로 정확한 기업가치 반영
        2. ROIC = EBIT / Invested Capital (기존: 영업이익 / (총자산-총부채-현금))
           - Invested Capital = Total Assets - Cash로 수정하여 더 정확한 투하자본 계산
        """
        
        # Use year column (handle both 연도 and 회계년도)
        year_col = '회계년도' if '회계년도' in df.columns else '연도'
        
        # Calculate Earnings Yield (EBIT/EV) - corrected formula
        if '영업이익' in df.columns and '시가총액' in df.columns:
            ebit = pd.to_numeric(df['영업이익'], errors='coerce')
            market_cap = pd.to_numeric(df['시가총액'], errors='coerce')
            
            # Calculate Enterprise Value (EV) = Market Cap + Total Debt - Cash
            if '총부채' in df.columns and '현금및현금성자산' in df.columns:
                total_debt = pd.to_numeric(df['총부채'], errors='coerce')
                cash = pd.to_numeric(df['현금및현금성자산'], errors='coerce')
                enterprise_value = market_cap + total_debt - cash
            else:
                enterprise_value = market_cap  # Fallback to market cap if debt/cash not available
            
            df['earnings_yield'] = ebit / enterprise_value.replace(0, np.nan)
        else:
            df['earnings_yield'] = 0
        
        # Calculate ROIC (Return on Invested Capital) - corrected formula
        if '투하자본수익률(ROIC)' in df.columns:
            df['roic'] = pd.to_numeric(df['투하자본수익률(ROIC)'], errors='coerce')
        elif '영업이익' in df.columns and '총자산' in df.columns and '총부채' in df.columns and '현금및현금성자산' in df.columns:
            # Calculate ROIC = EBIT / Invested Capital
            ebit = pd.to_numeric(df['영업이익'], errors='coerce')
            total_assets = pd.to_numeric(df['총자산'], errors='coerce')
            total_debt = pd.to_numeric(df['총부채'], errors='coerce')
            cash = pd.to_numeric(df['현금및현금성자산'], errors='coerce')
            
            # Invested Capital = Total Assets - Non-interest bearing liabilities
            # Approximation: Invested Capital = Total Assets - Cash - Non-interest bearing debt
            invested_capital = total_assets - cash
            invested_capital = invested_capital.replace(0, np.nan)  # Avoid division by zero
            
            df['roic'] = ebit / invested_capital
        else:
            df['roic'] = 0
        
        # Calculate ranks within each year (1 = best, higher number = worse)
        df['ey_rank'] = df.groupby(year_col)['earnings_yield'].rank(ascending=False, method='average')
        df['roic_rank'] = df.groupby(year_col)['roic'].rank(ascending=False, method='average')
        
        # Combined Magic Formula rank (lower rank sum = better)
        df['magic_rank'] = df['ey_rank'] + df['roic_rank']
        
        # Magic Formula signal: lower rank sum = higher signal (직접 역순 변환)
        df['magic_signal'] = df.groupby(year_col)['magic_rank'].transform(lambda x: x.max() - x + 1)
        
        return df
    
    def _compute_fscore(self, df):
        """
        Compute Piotroski F-Score
        
        수정 사항:
        1. f_shares (주식 희석 방지): 납입자본금 기준으로 변경 (기존: 발행주식수)
           - 납입자본금이 전년대비 감소하거나 동일하면 1점, 증가하면 0점
           - 전년도 데이터가 없으면 기본적으로 1점 부여 (보수적 접근)
        """
        
        # F-Score 구성 요소들을 하나씩 계산
        fscore_components = []
        
        # 1. ROA > 0
        if '총자산수익률(ROA)' in df.columns:
            df['f_roa'] = (pd.to_numeric(df['총자산수익률(ROA)'], errors='coerce') > 0).astype(int)
        else:
            df['f_roa'] = 0
        fscore_components.append('f_roa')
        
        # 2. Operating Cash Flow > 0  
        if '영업현금흐름' in df.columns:
            df['f_cfo'] = (pd.to_numeric(df['영업현금흐름'], errors='coerce') > 0).astype(int)
        else:
            df['f_cfo'] = 0
        fscore_components.append('f_cfo')
        
        # 3. Change in ROA > 0 (compared to previous year)
        if '총자산수익률(ROA)' in df.columns:
            roa_values = pd.to_numeric(df['총자산수익률(ROA)'], errors='coerce')
            prev_roa_values = df.groupby('거래소코드')['총자산수익률(ROA)'].shift(1)
            prev_roa_values = pd.to_numeric(prev_roa_values, errors='coerce')
            df['f_droa'] = (roa_values > prev_roa_values).astype(int)
        else:
            df['f_droa'] = 0
        fscore_components.append('f_droa')
        
        # 4. Operating Cash Flow > Net Income
        if '영업현금흐름' in df.columns and '당기순이익' in df.columns:
            df['f_accrual'] = (pd.to_numeric(df['영업현금흐름'], errors='coerce') > pd.to_numeric(df['당기순이익'], errors='coerce')).astype(int)
        else:
            df['f_accrual'] = 0
        fscore_components.append('f_accrual')
        
        # 5. Decrease in Long-term Debt ratio
        if '총부채' in df.columns and '총자산' in df.columns:
            df['debt_ratio'] = pd.to_numeric(df['총부채'], errors='coerce') / pd.to_numeric(df['총자산'], errors='coerce')
            df['prev_debt_ratio'] = df.groupby('거래소코드')['debt_ratio'].shift(1)
            df['f_leverage'] = (df['debt_ratio'] < df['prev_debt_ratio']).astype(int)
        else:
            df['f_leverage'] = 0
        fscore_components.append('f_leverage')
        
        # 6. Increase in Current Ratio  
        if '유동자산' in df.columns and '유동부채' in df.columns:
            df['current_ratio'] = pd.to_numeric(df['유동자산'], errors='coerce') / pd.to_numeric(df['유동부채'], errors='coerce').replace(0, np.nan)
            df['prev_current_ratio'] = df.groupby('거래소코드')['current_ratio'].shift(1)
            df['f_liquid'] = (df['current_ratio'] > df['prev_current_ratio']).astype(int)
        else:
            df['f_liquid'] = 0
        fscore_components.append('f_liquid')
        
        # 7. No new shares issued (check for decrease in paid-in capital)
        if '납입자본금' in df.columns:
            df['납입자본금'] = pd.to_numeric(df['납입자본금'], errors='coerce')
            df['prev_capital'] = df.groupby('거래소코드')['납입자본금'].shift(1)
            # Score 1 if paid-in capital decreased or stayed same (no dilution), or if no previous year data
            df['f_shares'] = ((df['납입자본금'] <= df['prev_capital']) | (df['prev_capital'].isna())).astype(int)
        else:
            df['f_shares'] = 1  # Default to 1 (no dilution) if no paid-in capital data
        fscore_components.append('f_shares')
        
        # 8. Increase in Gross Margin
        if '매출총이익' in df.columns and '매출액' in df.columns:
            df['gross_margin'] = pd.to_numeric(df['매출총이익'], errors='coerce') / pd.to_numeric(df['매출액'], errors='coerce').replace(0, np.nan)
            df['prev_gross_margin'] = df.groupby('거래소코드')['gross_margin'].shift(1)
            df['f_margin'] = (df['gross_margin'] > df['prev_gross_margin']).astype(int)
        else:
            df['f_margin'] = 0
        fscore_components.append('f_margin')
        
        # 9. Increase in Asset Turnover
        if '총자산회전율' in df.columns:
            df['prev_turnover'] = df.groupby('거래소코드')['총자산회전율'].shift(1)
            df['f_turnover'] = (pd.to_numeric(df['총자산회전율'], errors='coerce') > pd.to_numeric(df['prev_turnover'], errors='coerce')).astype(int)
        elif '매출액' in df.columns and '총자산' in df.columns:
            # Calculate asset turnover manually
            df['asset_turnover'] = pd.to_numeric(df['매출액'], errors='coerce') / pd.to_numeric(df['총자산'], errors='coerce').replace(0, np.nan)
            df['prev_asset_turnover'] = df.groupby('거래소코드')['asset_turnover'].shift(1)
            df['f_turnover'] = (df['asset_turnover'] > df['prev_asset_turnover']).astype(int)
        else:
            df['f_turnover'] = 0
        fscore_components.append('f_turnover')
        
        # Calculate total F-Score
        df['fscore'] = df[fscore_components].sum(axis=1)
        
        return df
    
    @staticmethod
    def _process_momentum_chunk_optimized(chunk_data):
        """Optimized momentum calculation for pre-filtered chunk data"""
        chunk, chunk_df, lookback_months, skip_months, chunk_num, total_chunks = chunk_data
        
        # Remove print to reduce I/O overhead in worker processes
        
        chunk_results = []
        for code in chunk:
            stock_data = chunk_df[chunk_df['거래소코드'] == code].copy()
            stock_data = stock_data.sort_values('date')
            
            # Calculate momentum returns
            stock_data['momentum'] = np.nan
            
            for i in range(len(stock_data)):
                current_date = stock_data.iloc[i]['date']
                
                # Skip period
                lookback_date = current_date - relativedelta(months=lookback_months)
                
                # Find prices
                # Use best available price column
                if '종가' in stock_data.columns:
                    current_price = stock_data.iloc[i]['종가']
                elif '일간_시가총액' in stock_data.columns:
                    current_price = stock_data.iloc[i]['일간_시가총액']
                else:
                    current_price = stock_data.iloc[i]['시가총액']
                
                # Get price at lookback date
                past_data = stock_data[stock_data['date'] <= lookback_date]
                if len(past_data) > 0:
                    # Use best available price column for past price
                    if '종가' in past_data.columns:
                        past_price = past_data.iloc[-1]['종가']
                    elif '일간_시가총액' in past_data.columns:
                        past_price = past_data.iloc[-1]['일간_시가총액']
                    else:
                        past_price = past_data.iloc[-1]['시가총액']
                    
                    if past_price > 0:
                        momentum = (current_price / past_price) - 1
                        stock_data.iloc[i, stock_data.columns.get_loc('momentum')] = momentum
            
            chunk_results.append(stock_data)
        
        return chunk_results
    
    def _compute_momentum(self, df):
        """
        Compute momentum factor with vectorized calculation
        
        수정 사항:
        1. 멀티프로세싱에서 벡터화 계산으로 변경
        2. 년도별 모멘텀 계산으로 단순화
        3. 극단값 필터링 추가 (±200% 초과 모멘텀 제외)
        4. 연간 데이터 특성에 맞게 최적화
        """
        
        print("    📊 모멘텀 설정 확인...")
        momentum_params = self.config.get('strategy_params', {}).get('momentum', {})
        lookback_months = momentum_params.get('lookback_months', 12)
        skip_months = momentum_params.get('skip_months', 1)
        
        # Vectorized momentum calculation
        df = df.sort_values(['거래소코드', 'date'])
        
        # Determine price column to use (시가총액은 부정확하므로 피함)
        if '종가' in df.columns:
            price_col = '종가'
        elif '일간_시가총액' in df.columns:
            price_col = '일간_시가총액'
        else:
            # 시가총액은 마지막 선택지로만 사용
            price_col = '시가총액'
            print("    ⚠️ 종가 데이터 없음 - 시가총액 사용 (부정확할 수 있음)")
        
        # Convert price column to numeric
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        
        # Calculate momentum using vectorized operations
        df['momentum'] = np.nan
        
        # For annual data, calculate year-over-year momentum
        if '연도' in df.columns:
            year_col = '연도'
        elif '회계년도' in df.columns:
            year_col = '회계년도'
        else:
            # Extract year from date
            df['year'] = df['date'].dt.year
            year_col = 'year'
        
        # Calculate momentum as price change from previous year
        for code in df['거래소코드'].unique():
            stock_mask = df['거래소코드'] == code
            stock_data = df[stock_mask].copy()
            
            if len(stock_data) < 2:
                continue
                
            # Sort by year/date
            stock_data = stock_data.sort_values(year_col)
            
            # Calculate year-over-year momentum
            current_prices = stock_data[price_col].values
            lagged_prices = np.roll(current_prices, 1)  # 1-year lag
            
            # Calculate momentum returns, skip first observation (no lag available)
            momentum_values = np.where(
                (lagged_prices > 0) & (current_prices > 0) & (np.arange(len(current_prices)) > 0),
                (current_prices / lagged_prices) - 1,
                np.nan
            )
            
            # 합리적 모멘텀 필터링: 투자 가능한 범위로 제한
            momentum_values = np.where(
                (momentum_values > 0.8) | (momentum_values < -0.6),  # ±80/60% 초과 제거
                np.nan,
                momentum_values
            )
            
            # Update momentum values in original dataframe
            df.loc[stock_mask, 'momentum'] = momentum_values
        
        # Clean up momentum values
        df['momentum'] = df['momentum'].replace([np.inf, -np.inf], np.nan)
        
        # 추가 극단값 필터링
        valid_momentum = df['momentum'].notna()
        extreme_mask = (df['momentum'].abs() > 2.0) & valid_momentum
        df.loc[extreme_mask, 'momentum'] = np.nan
        
        df['mom'] = df['momentum'].fillna(0)
        
        # 통계 출력
        valid_count = df['momentum'].notna().sum()
        mean_momentum = df['momentum'].mean()
        print(f"    📊 유효 모멘텀: {valid_count}개, 평균: {mean_momentum:.2%}")
        
        print("    ✅ 벡터화된 모멘텀 계산 완료 (극단값 필터링 적용)")
        return df
        
    def _build_ff3_factors(self, df):
        """
        Build Fama-French 3-factor model
        
        수정 사항:
        1. 월말 가격을 월간 수익률(pct_change())로 변환 후 사용
        2. NYSE 방식 6-포트폴리오(Size × Value)로 SMB/HML 산출
        3. 길이 불일치·shift() NaN 오류 제거
        """
        print("    📊 FF3 팩터 데이터 생성...")
        
        try:
            # Determine price column
            if '종가' in df.columns:
                price_col = '종가'
            elif '일간_시가총액' in df.columns:
                price_col = '일간_시가총액'
            else:
                price_col = '시가총액'
            
            # Create monthly data with returns
            df['month_year'] = df['date'].dt.to_period('M')
            monthly_data = df.groupby(['거래소코드', 'month_year']).last().reset_index()
            monthly_data[price_col] = pd.to_numeric(monthly_data[price_col], errors='coerce')
            
            # Calculate monthly returns for each stock
            monthly_data = monthly_data.sort_values(['거래소코드', 'month_year'])
            monthly_data['monthly_return'] = monthly_data.groupby('거래소코드')[price_col].pct_change()
            
            # Remove first period (NaN returns) and invalid returns
            monthly_data = monthly_data.dropna(subset=['monthly_return'])
            monthly_data = monthly_data[monthly_data['monthly_return'].replace([np.inf, -np.inf], np.nan).notna()]
            
            if len(monthly_data) == 0:
                print("    ⚠️ 유효한 월간 수익률 데이터 없음")
                self.ff3_factors = pd.DataFrame()
                return df
            
            # Risk-free rate - 연도별 무위험 수익률 데이터 활용
            annual_risk_free_rates = {
                2012: 3.42, 2013: 3.37, 2014: 3.05, 2015: 2.37, 2016: 1.73,
                2017: 2.32, 2018: 2.60, 2019: 1.74, 2020: 1.41, 2021: 2.08,
                2022: 3.37, 2023: 3.64
            }
            
            # 월별 무위험 수익률 계산 함수
            def get_monthly_risk_free_rate(period):
                year = period.year
                if year in annual_risk_free_rates:
                    return annual_risk_free_rates[year] / 100 / 12  # 연율을 월율로 변환
                else:
                    return 0.02 / 12  # 기본값: 2% 연율
            
            # Calculate market return (value-weighted)
            monthly_data['market_cap'] = pd.to_numeric(monthly_data['시가총액'], errors='coerce')
            monthly_market = monthly_data.groupby('month_year').apply(
                lambda x: np.average(x['monthly_return'], weights=x['market_cap']) 
                if not x['market_cap'].isna().all() else x['monthly_return'].mean()
            )
            
            # NYSE-style 6-portfolio construction
            smb_returns = []
            hml_returns = []
            
            # Determine value measure
            if 'PBR' in monthly_data.columns:
                value_col = 'PBR'
                value_ascending = True  # Low PBR = High value
            elif '주당순자산가치(BPS)' in monthly_data.columns and '시가총액' in monthly_data.columns:
                monthly_data['pb_ratio'] = monthly_data['시가총액'] / monthly_data['주당순자산가치(BPS)']
                value_col = 'pb_ratio'
                value_ascending = True
            else:
                # Use book-to-market proxy
                if '총자산' in monthly_data.columns:
                    monthly_data['bm_proxy'] = monthly_data['총자산'] / monthly_data['시가총액']
                    value_col = 'bm_proxy'
                    value_ascending = False  # High B/M = High value
                else:
                    value_col = None
            
            for period in monthly_market.index:
                period_data = monthly_data[monthly_data['month_year'] == period].copy()
                
                if len(period_data) < 6:  # Need minimum stocks for 6 portfolios
                    smb_returns.append(0)
                    hml_returns.append(0)
                    continue
                
                # Size breakpoint (median)
                size_median = period_data['market_cap'].median()
                period_data['size_group'] = np.where(period_data['market_cap'] <= size_median, 'Small', 'Big')
                
                # Value breakpoints (30th and 70th percentiles)
                if value_col and value_col in period_data.columns:
                    value_30 = period_data[value_col].quantile(0.3)
                    value_70 = period_data[value_col].quantile(0.7)
                    
                    if value_ascending:
                        period_data['value_group'] = pd.cut(period_data[value_col], 
                                                          bins=[-np.inf, value_30, value_70, np.inf],
                                                          labels=['High', 'Medium', 'Low'])
                    else:
                        period_data['value_group'] = pd.cut(period_data[value_col], 
                                                          bins=[-np.inf, value_30, value_70, np.inf],
                                                          labels=['Low', 'Medium', 'High'])
                else:
                    period_data['value_group'] = 'Medium'  # Neutral if no value measure
                
                # Create 6 portfolios
                portfolios = {}
                for size in ['Small', 'Big']:
                    for value in ['Low', 'Medium', 'High']:
                        portfolio_stocks = period_data[
                            (period_data['size_group'] == size) & 
                            (period_data['value_group'] == value)
                        ]
                        if len(portfolio_stocks) > 0:
                            # Value-weighted returns
                            portfolio_return = np.average(
                                portfolio_stocks['monthly_return'], 
                                weights=portfolio_stocks['market_cap']
                            )
                            portfolios[f'{size}_{value}'] = portfolio_return
                        else:
                            portfolios[f'{size}_{value}'] = 0
                
                # Calculate SMB (Small Minus Big)
                small_avg = (portfolios.get('Small_Low', 0) + 
                           portfolios.get('Small_Medium', 0) + 
                           portfolios.get('Small_High', 0)) / 3
                big_avg = (portfolios.get('Big_Low', 0) + 
                         portfolios.get('Big_Medium', 0) + 
                         portfolios.get('Big_High', 0)) / 3
                smb = small_avg - big_avg
                smb_returns.append(smb)
                
                # Calculate HML (High Minus Low)
                high_avg = (portfolios.get('Small_High', 0) + portfolios.get('Big_High', 0)) / 2
                low_avg = (portfolios.get('Small_Low', 0) + portfolios.get('Big_Low', 0)) / 2
                hml = high_avg - low_avg
                hml_returns.append(hml)
            
            # Create FF3 factors dataframe
            if len(monthly_market) > 0:
                # 각 기간별 무위험 수익률 계산
                period_risk_free_rates = [get_monthly_risk_free_rate(period) for period in monthly_market.index]
                
                ff3_data = {
                    'date': monthly_market.index,
                    'Mkt_RF': monthly_market - period_risk_free_rates,
                    'SMB': smb_returns[:len(monthly_market)],
                    'HML': hml_returns[:len(monthly_market)],
                    'RF': period_risk_free_rates
                }
                
                self.ff3_factors = pd.DataFrame(ff3_data)
                self.ff3_factors = self.ff3_factors.fillna(0)
                
                print(f"    ✅ FF3 팩터 생성 완료: {len(self.ff3_factors)}개 월")
            else:
                self.ff3_factors = pd.DataFrame()
                print("    ⚠️ FF3 팩터 생성 실패: 시장 수익률 데이터 없음")
            
        except Exception as e:
            print(f"    ⚠️ FF3 팩터 생성 실패: {e}")
            self.ff3_factors = pd.DataFrame()
        
        return df
    
    def _compute_ff3_alpha(self, df):
        """
        Compute FF3-Alpha for each stock using Fama-French 3-factor model
        
        수정 사항:
        1. FF3 팩터와 개별 종목 수익률을 연간 데이터로 회귀분석
        2. Alpha = 회귀 절편 (Intercept)
        3. P-value로 유의성 검증
        4. 연간 데이터 특성에 맞게 최적화
        """
        
        # Initialize FF3-Alpha columns
        df['ff3_alpha'] = np.nan
        df['ff3_alpha_pvalue'] = np.nan
        df['ff3_r_squared'] = np.nan
        
        # Check if FF3 factors are available
        if self.ff3_factors is None or len(self.ff3_factors) == 0:
            print("    ⚠️ FF3 팩터가 없어 FF3-Alpha 계산 불가")
            return df
        
        print(f"    📊 FF3 팩터 데이터: {len(self.ff3_factors)}개 기간")
        
        # Determine price column
        if '종가' in df.columns:
            price_col = '종가'
        elif '일간_시가총액' in df.columns:
            price_col = '일간_시가총액'
        else:
            price_col = '시가총액'
        
        # Use annual data approach
        if '연도' in df.columns:
            year_col = '연도'
        elif '회계년도' in df.columns:
            year_col = '회계년도'
        else:
            df['year'] = df['date'].dt.year
            year_col = 'year'
        
        # Calculate annual returns for each stock
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df = df.sort_values(['거래소코드', year_col])
        df['annual_return'] = df.groupby('거래소코드')[price_col].pct_change()
        
        # Convert FF3 factors to annual
        if not self.ff3_factors.empty:
            ff3_annual = self.ff3_factors.copy()
            ff3_annual['year'] = ff3_annual['date'].astype(str).str[:4].astype(int)
            
            # Aggregate monthly to annual (compound returns)
            ff3_annual_agg = ff3_annual.groupby('year').agg({
                'Mkt_RF': lambda x: (1 + x).prod() - 1,
                'SMB': lambda x: (1 + x).prod() - 1, 
                'HML': lambda x: (1 + x).prod() - 1,
                'RF': 'mean'
            }).reset_index()
            
            print(f"    📊 연간 FF3 팩터: {len(ff3_annual_agg)}개 년도")
            
            alpha_count = 0
            total_stocks = df['거래소코드'].nunique()
            
            # Calculate FF3-Alpha for each stock
            for code in df['거래소코드'].unique():
                try:
                    stock_data = df[df['거래소코드'] == code].copy()
                    stock_data = stock_data.dropna(subset=['annual_return'])
                    
                    if len(stock_data) < 3:  # Need minimum observations
                        continue
                    
                    # Merge with FF3 factors
                    merged_data = pd.merge(
                        stock_data[[year_col, 'annual_return']], 
                        ff3_annual_agg,
                        left_on=year_col, 
                        right_on='year',
                        how='inner'
                    )
                    
                    if len(merged_data) < 3:
                        continue
                    
                    # Calculate excess returns
                    merged_data['Stock_Excess'] = merged_data['annual_return'] - merged_data['RF']
                    
                    # Prepare regression variables
                    X = merged_data[['Mkt_RF', 'SMB', 'HML']].fillna(0)
                    y = merged_data['Stock_Excess'].fillna(0)
                    
                    # Remove infinite and NaN values
                    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
                    X = X[mask]
                    y = y[mask]
                    
                    if len(X) >= 3 and len(y) >= 3:
                        try:
                            # Use statsmodels for proper regression
                            import statsmodels.api as sm
                            X_with_const = sm.add_constant(X)
                            model = sm.OLS(y, X_with_const)
                            results = model.fit()
                            
                            alpha = results.params['const']
                            alpha_pvalue = results.pvalues['const']
                            r_squared = results.rsquared
                            
                        except ImportError:
                            # Fallback to simple linear regression
                            from sklearn.linear_model import LinearRegression
                            from scipy import stats
                            
                            reg = LinearRegression()
                            reg.fit(X, y)
                            alpha = reg.intercept_
                            
                            # Simple p-value approximation
                            y_pred = reg.predict(X)
                            mse = np.mean((y - y_pred) ** 2)
                            alpha_pvalue = 0.05 if abs(alpha) > np.sqrt(mse) else 0.5
                            r_squared = reg.score(X, y)
                        
                        # Update dataframe with FF3-Alpha values
                        mask = df['거래소코드'] == code
                        df.loc[mask, 'ff3_alpha'] = alpha
                        df.loc[mask, 'ff3_alpha_pvalue'] = alpha_pvalue
                        df.loc[mask, 'ff3_r_squared'] = r_squared
                        
                        alpha_count += 1
                        
                except Exception:
                    continue
            
            print(f"    ✅ FF3-Alpha 계산 완료: {alpha_count}/{total_stocks}개 종목")
        
        return df


class StrategyBuilder:
    """Portfolio construction for different strategies"""
    
    def __init__(self, config):
        self.config = config
        self.portfolio_size = config['portfolio_params']['portfolio_size']
        self.weighting_scheme = config['portfolio_params']['weighting_scheme']
    
    @staticmethod  
    def _process_ff3_alpha_chunk_optimized(chunk_data):
        """
        Optimized FF3 Alpha calculation with statsmodels OLS
        
        수정 사항:
        - statsmodels.api.OLS로 다중회귀 수행
        - alpha = results.params['const'], p_value = results.pvalues['const'] 저장
        """
        chunk, chunk_df, ff3_factors, regression_window, alpha_threshold = chunk_data[:5]
        
        try:
            import statsmodels.api as sm
        except ImportError:
            print("    ⚠️ statsmodels 라이브러리가 없습니다. scipy로 대체합니다.")
            sm = None
        
        chunk_alpha_results = []
        
        # Determine price column
        if '종가' in chunk_df.columns:
            price_col = '종가'
        elif '일간_시가총액' in chunk_df.columns:
            price_col = '일간_시가총액'
        else:
            price_col = '시가총액'
        
        for code in chunk:
            try:
                # Get stock data for regression window
                stock_data = chunk_df[chunk_df['거래소코드'] == code].copy()
                stock_data = stock_data.sort_values('date')
                
                if len(stock_data) < 12:  # Minimum 12 observations
                    continue
                
                # Calculate monthly returns
                stock_data['month_year'] = stock_data['date'].dt.to_period('M')
                monthly_data = stock_data.groupby('month_year').last().reset_index()
                monthly_data[price_col] = pd.to_numeric(monthly_data[price_col], errors='coerce')
                monthly_data = monthly_data.sort_values('month_year')
                
                # Calculate monthly returns
                monthly_data['monthly_return'] = monthly_data[price_col].pct_change()
                monthly_data = monthly_data.dropna(subset=['monthly_return'])
                
                if len(monthly_data) < 12:
                    continue
                
                # Get last N months of data
                recent_data = monthly_data.tail(min(regression_window, len(monthly_data)))
                
                if len(recent_data) < 12:
                    continue
                
                # Merge with FF3 factors
                if ff3_factors is not None and len(ff3_factors) > 0:
                    factor_data = ff3_factors.copy()
                    factor_data['month_year'] = factor_data['date']
                    
                    # Merge on month_year
                    merged_data = pd.merge(recent_data[['month_year', 'monthly_return']], 
                                         factor_data[['month_year', 'Mkt_RF', 'SMB', 'HML', 'RF']], 
                                         on='month_year', 
                                         how='inner')
                    
                    if len(merged_data) < 12:
                        continue
                    
                    # Calculate excess returns
                    merged_data['Stock_Excess'] = merged_data['monthly_return'] - merged_data['RF']
                    
                    # Prepare regression variables
                    X = merged_data[['Mkt_RF', 'SMB', 'HML']].fillna(0)
                    y = merged_data['Stock_Excess'].fillna(0)
                    
                    # Remove infinite values
                    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
                    X = X[mask]
                    y = y[mask]
                    
                    if len(X) >= 12 and len(y) >= 12:
                        if sm is not None:
                            # Use statsmodels OLS for proper multi-factor regression
                            X_with_const = sm.add_constant(X)
                            model = sm.OLS(y, X_with_const)
                            results = model.fit()
                            
                            alpha = results.params['const']
                            p_value = results.pvalues['const']
                            r_squared = results.rsquared
                            
                        else:
                            # Fallback to scipy
                            from scipy.stats import linregress
                            # Simple approximation using market factor only
                            _, alpha, r_value, p_value, _ = linregress(X['Mkt_RF'], y)
                            r_squared = r_value**2
                        
                        if p_value < alpha_threshold:
                            chunk_alpha_results.append({
                                'code': code,
                                'alpha': alpha,
                                'p_value': p_value,
                                'r_squared': r_squared
                            })
                
            except Exception as e:
                # Skip problematic stocks
                continue
        
        return chunk_alpha_results
        
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
                portfolio_results[strategy_name] = {'All': [], 'Normal': []}
        
        return portfolio_results
    
    def _build_magic_formula_portfolio(self, df, factor_engine):
        """Build Magic Formula portfolio"""
        portfolios = {'All': [], 'Normal': []}
        
        # Get rebalancing dates - Magic Formula rebalances on April 1st each year
        start_date = pd.to_datetime(self.config['start_date'])
        end_date = pd.to_datetime(self.config['end_date'])
        
        # Generate April 1st dates
        start_year = start_date.year
        end_year = end_date.year
        rebalance_dates = [pd.Timestamp(f'{year}-04-01') for year in range(start_year, end_year + 1)]
        rebalance_dates = [date for date in rebalance_dates if start_date <= date <= end_date]
        
        for rebalance_date in rebalance_dates:
            print(f"    🔄 Magic Formula 리밸런싱: {rebalance_date.strftime('%Y-%m-%d')}")
            
            # Get data for rebalancing date (use annual data)
            year = rebalance_date.year
            rebalance_data = df[
                (df['연도'] == year) if '연도' in df.columns else 
                (df['회계년도'] == year) if '회계년도' in df.columns else 
                (df['date'].dt.year == year)
            ]
            
            if len(rebalance_data) == 0:
                print(f"      ⚠️ {year}년 데이터 없음")
                continue
            
            print(f"      🔍 분석 대상: {len(rebalance_data)}개 종목")
            
            # Filter stocks with valid magic signal
            valid_data = rebalance_data[rebalance_data['magic_signal'].notna()]
            print(f"      🔍 Magic 시그널 유효: {len(valid_data)}개 종목")
            
            if len(valid_data) >= self.portfolio_size:
                # Select top stocks by magic signal
                top_stocks = valid_data.nlargest(self.portfolio_size, 'magic_signal')
                print(f"      ✅ All 포트폴리오: {len(top_stocks)}개 종목 선택")
                
                portfolios['All'].append({
                    'date': rebalance_date,
                    'stocks': top_stocks['거래소코드'].tolist(),
                    'signals': top_stocks['magic_signal'].tolist()
                })
            
            # Normal firms (non-default) portfolio
            normal_data = valid_data[valid_data.get('default', 0) == 0]
            if len(normal_data) >= self.portfolio_size:
                top_normal_stocks = normal_data.nlargest(self.portfolio_size, 'magic_signal')
                print(f"      ✅ Normal 포트폴리오: {len(top_normal_stocks)}개 종목 선택")
                
                portfolios['Normal'].append({
                    'date': rebalance_date,
                    'stocks': top_normal_stocks['거래소코드'].tolist(),
                    'signals': top_normal_stocks['magic_signal'].tolist()
                })
        
        return portfolios
    
    def _build_fscore_portfolio(self, df, factor_engine):
        """Build F-Score portfolio"""
        portfolios = {'All': [], 'Normal': []}
        
        # Get rebalancing dates - F-Score rebalances on April 1st each year
        start_date = pd.to_datetime(self.config['start_date'])
        end_date = pd.to_datetime(self.config['end_date'])
        
        # Generate April 1st dates
        start_year = start_date.year
        end_year = end_date.year
        rebalance_dates = [pd.Timestamp(f'{year}-04-01') for year in range(start_year, end_year + 1)]
        rebalance_dates = [date for date in rebalance_dates if start_date <= date <= end_date]
        
        min_score = self.config['strategy_params']['f_score']['min_score']
        
        for rebalance_date in rebalance_dates:
            print(f"    🔄 F-Score 리밸런싱: {rebalance_date.strftime('%Y-%m-%d')}")
            
            # Get data for rebalancing date (use annual data)
            year = rebalance_date.year
            rebalance_data = df[
                (df['연도'] == year) if '연도' in df.columns else 
                (df['회계년도'] == year) if '회계년도' in df.columns else 
                (df['date'].dt.year == year)
            ]
            
            if len(rebalance_data) == 0:
                print(f"      ⚠️ {year}년 데이터 없음")
                continue
                
            print(f"      🔍 분석 대상: {len(rebalance_data)}개 종목")
            
            # Filter stocks with F-Score >= minimum threshold
            valid_data = rebalance_data[rebalance_data['fscore'] >= min_score]
            print(f"      🔍 F-Score >= {min_score}: {len(valid_data)}개 후보")
            
            if len(valid_data) >= self.portfolio_size:
                # Select top stocks by F-Score
                top_stocks = valid_data.nlargest(self.portfolio_size, 'fscore')
                print(f"      ✅ All 포트폴리오: {len(top_stocks)}개 종목 선택")
                
                portfolios['All'].append({
                    'date': rebalance_date,
                    'stocks': top_stocks['거래소코드'].tolist(),
                    'signals': top_stocks['fscore'].tolist()
                })
            
            # Normal firms (non-default) portfolio
            normal_data = valid_data[valid_data.get('default', 0) == 0]
            if len(normal_data) >= self.portfolio_size:
                top_normal_stocks = normal_data.nlargest(self.portfolio_size, 'fscore')
                print(f"      ✅ Normal 포트폴리오: {len(top_normal_stocks)}개 종목 선택")
                
                portfolios['Normal'].append({
                    'date': rebalance_date,
                    'stocks': top_normal_stocks['거래소코드'].tolist(),
                    'signals': top_normal_stocks['fscore'].tolist()
                })
        
        return portfolios
    
    def _build_momentum_portfolio(self, df, factor_engine):
        """Build Momentum portfolio"""
        portfolios = {'All': [], 'Normal': []}
        
        # Get rebalancing dates - Momentum rebalances on April 1st each year
        start_date = pd.to_datetime(self.config['start_date'])
        end_date = pd.to_datetime(self.config['end_date'])
        
        # Generate April 1st dates
        start_year = start_date.year
        end_year = end_date.year
        rebalance_dates = [pd.Timestamp(f'{year}-04-01') for year in range(start_year, end_year + 1)]
        rebalance_dates = [date for date in rebalance_dates if start_date <= date <= end_date]
        
        for rebalance_date in rebalance_dates:
            print(f"    🔄 모멘텀 리밸런싱: {rebalance_date.strftime('%Y-%m-%d')}")
            
            # Get data for rebalancing year (use annual data like other strategies)
            year = rebalance_date.year
            rebalance_data = df[
                (df['연도'] == year) if '연도' in df.columns else 
                (df['회계년도'] == year) if '회계년도' in df.columns else 
                (df['date'].dt.year == year)
            ]
            
            if len(rebalance_data) == 0:
                print(f"      ⚠️ {year}년 데이터 없음")
                continue
                
            print(f"      🔍 분석 대상: {len(rebalance_data)}개 종목")
                
            # All firms portfolio
            valid_data = rebalance_data[rebalance_data['momentum'].notna() & (rebalance_data['momentum'] != 0)]
            print(f"      🔍 유효 모멘텀: {len(valid_data)}개 종목")
            
            if len(valid_data) >= self.portfolio_size:
                # 균형잡힌 모멘텀 포트폴리오 선택
                # 너무 극단적인 모멘텀은 제외하고 안정적인 범위에서 선택
                moderate_momentum = valid_data[
                    (valid_data['momentum'] > 0.1) & (valid_data['momentum'] <= 0.6)
                ]  # 10-60% 범위
                
                if len(moderate_momentum) >= self.portfolio_size:
                    # 적당한 모멘텀 범위에서 상위 종목 선택
                    top_stocks = moderate_momentum.nlargest(self.portfolio_size, 'momentum')
                    print(f"      ✅ All 포트폴리오: {len(top_stocks)}개 종목 선택 (적당한 모멘텀)")
                else:
                    # 적당한 범위에 충분한 종목이 없으면 전체에서 선택하되 극단값 제한
                    valid_data_filtered = valid_data[valid_data['momentum'] <= 0.8]  # 80% 이하만
                    if len(valid_data_filtered) >= self.portfolio_size:
                        top_stocks = valid_data_filtered.nlargest(self.portfolio_size, 'momentum')
                        print(f"      ✅ All 포트폴리오: {len(top_stocks)}개 종목 선택 (필터링됨)")
                    else:
                        top_stocks = valid_data.nlargest(self.portfolio_size, 'momentum')
                        print(f"      ✅ All 포트폴리오: {len(top_stocks)}개 종목 선택 (원본)")
                
                portfolios['All'].append({
                    'date': rebalance_date,
                    'stocks': top_stocks['거래소코드'].tolist(),
                    'signals': top_stocks['momentum'].tolist()
                })
            
            # Normal firms portfolio
            normal_data = valid_data[valid_data.get('default', 0) == 0]
            if len(normal_data) >= self.portfolio_size:
                top_normal_stocks = normal_data.nlargest(self.portfolio_size, 'momentum')
                print(f"      ✅ Normal 포트폴리오: {len(top_normal_stocks)}개 종목 선택")
                
                portfolios['Normal'].append({
                    'date': rebalance_date,
                    'stocks': top_normal_stocks['거래소코드'].tolist(),
                    'signals': top_normal_stocks['momentum'].tolist()
                })
        
        return portfolios
        
    def _build_ff3_alpha_portfolio(self, df, factor_engine):
        """Build FF3-Alpha portfolio"""
        portfolios = {'All': [], 'Normal': []}
        
        # Check if FF3 factors are available
        if factor_engine.ff3_factors is None or factor_engine.ff3_factors.empty:
            print("    ⚠️ FF3 팩터가 없어 FF3-Alpha 포트폴리오 생성 불가")
            return portfolios
        
        # Get FF3 parameters
        ff3_params = self.config['strategy_params'].get('ff3_alpha', {})
        regression_window = ff3_params.get('regression_window', 5)  # Use 5 years for annual data
        alpha_threshold = ff3_params.get('alpha_pvalue_threshold', 0.1)
        
        # Get rebalancing dates - FF3-Alpha rebalances on July 1st each year
        start_date = pd.to_datetime(self.config['start_date'])
        end_date = pd.to_datetime(self.config['end_date'])
        
        # Generate July 1st dates
        start_year = start_date.year + regression_window  # Start after regression window
        end_year = end_date.year
        rebalance_dates = [pd.Timestamp(f'{year}-07-01') for year in range(start_year, end_year + 1)]
        rebalance_dates = [date for date in rebalance_dates if start_date <= date <= end_date]
        
        for rebalance_date in rebalance_dates:
            print(f"    🔄 FF3-Alpha 리밸런싱: {rebalance_date.strftime('%Y-%m-%d')}")
            
            # Get data for regression window (annual data)
            window_end_year = rebalance_date.year - 1  # Use previous year's data
            window_start_year = window_end_year - regression_window + 1
            
            year_col = '회계년도' if '회계년도' in df.columns else '연도'
            window_data = df[
                (df[year_col] >= window_start_year) & 
                (df[year_col] <= window_end_year)
            ]
            
            if len(window_data) == 0:
                print(f"      ⚠️ {window_start_year}-{window_end_year}년 데이터 없음")
                continue
            
            print(f"      🔍 회귀 윈도우: {window_start_year}-{window_end_year}년")
            
            unique_codes = window_data['거래소코드'].unique()
            total_stocks = len(unique_codes)
            print(f"      🔍 분석 대상: {total_stocks}개 종목")
            
            # Use rebalancing year data for portfolio selection
            rebalance_year = rebalance_date.year - 1  # Use previous year's calculated FF3-Alpha
            rebalance_data = df[df[year_col] == rebalance_year].copy()
            
            if len(rebalance_data) == 0:
                print(f"      ⚠️ {rebalance_year}년 데이터 없음")
                continue
            
            # Filter stocks with valid FF3-Alpha data
            valid_alpha_data = rebalance_data.dropna(subset=['ff3_alpha', 'ff3_alpha_pvalue'])
            print(f"      🔍 유효 FF3-Alpha: {len(valid_alpha_data)}개 종목")
            
            # Filter by p-value threshold for significant alphas
            significant_alpha_data = valid_alpha_data[valid_alpha_data['ff3_alpha_pvalue'] < alpha_threshold]
            print(f"      🔍 유의미한 알파 (p<{alpha_threshold}): {len(significant_alpha_data)}개 종목")
            
            # Build portfolio if we have enough significant stocks
            if len(significant_alpha_data) >= self.portfolio_size:
                # Select top alpha stocks (highest alpha values)
                top_alpha = significant_alpha_data.nlargest(self.portfolio_size, 'ff3_alpha')
                print(f"      ✅ All 포트폴리오: {len(top_alpha)}개 종목 선택")
                
                portfolios['All'].append({
                    'date': rebalance_date,
                    'stocks': [{'code': code, 'alpha': alpha} for code, alpha in 
                             zip(top_alpha['거래소코드'], top_alpha['ff3_alpha'])]
                })
                
                # Normal firms (non-default) portfolio
                normal_alpha_data = significant_alpha_data[significant_alpha_data.get('default', 0) == 0]
                if len(normal_alpha_data) >= self.portfolio_size:
                    top_normal_alpha = normal_alpha_data.nlargest(self.portfolio_size, 'ff3_alpha')
                    print(f"      ✅ Normal 포트폴리오: {len(top_normal_alpha)}개 종목 선택")
                    
                    portfolios['Normal'].append({
                        'date': rebalance_date,
                        'stocks': [{'code': code, 'alpha': alpha} for code, alpha in 
                                 zip(top_normal_alpha['거래소코드'], top_normal_alpha['ff3_alpha'])]
                    })
                else:
                    # Use same as All portfolio if not enough normal firms
                    print(f"      ✅ Normal 포트폴리오: {len(top_alpha)}개 종목 선택 (All과 동일)")
                    portfolios['Normal'].append({
                        'date': rebalance_date,
                        'stocks': [{'code': code, 'alpha': alpha} for code, alpha in 
                                 zip(top_alpha['거래소코드'], top_alpha['ff3_alpha'])]
                    })
            else:
                print(f"      ⚠️ 유의미한 알파 종목 부족: {len(significant_alpha_data)}개 < {self.portfolio_size}개")
        
        return portfolios


class BacktestEngine:
    """Backtesting engine with performance calculation"""
    
    def __init__(self, config):
        self.config = config
        self.start_date = pd.to_datetime(config['start_date'])
        self.end_date = pd.to_datetime(config['end_date'])
        self.initial_capital = config['portfolio_params'].get('initial_capital', 1000000)
        
    def run_backtest(self, portfolios, price_data):
        """Run backtest for all strategies and universes"""
        print("🚀 백테스트 실행 시작...")
        
        # Determine strategy processing method
        universes_to_process = ['All', 'Normal']
        total_combinations = sum(len(strategy_portfolios.get(universe, [])) 
                               for strategy_portfolios in portfolios.values() 
                               for universe in universes_to_process)
        
        print(f"    📊 총 {total_combinations}개 전략-유니버스 조합 처리")
        
        if total_combinations > 20:
            print("    ⚡ 대용량 처리 - 멀티프로세싱 사용")
            return self._backtest_with_multiprocessing(portfolios, price_data, universes_to_process)
        else:
            print("    🔄 소규모 처리 - 순차 처리 사용")
            return self._backtest_sequential(portfolios, price_data, universes_to_process)
    
    def _backtest_sequential(self, portfolios, price_data, universes_to_process):
        """Sequential backtesting for small datasets"""
        results = {}
        
        for strategy_name, strategy_portfolios in portfolios.items():
            print(f"  📈 {strategy_name} 백테스트...")
            results[strategy_name] = {}
            
            for universe in universes_to_process:
                if universe not in strategy_portfolios:
                    continue
                    
                print(f"    🌍 {universe} 유니버스...")
                portfolio_list = strategy_portfolios[universe]
                
                if not portfolio_list:
                    print(f"      ⚠️ 포트폴리오 없음")
                    results[strategy_name][universe] = self._create_empty_results()
                    continue
                
                # Calculate portfolio performance
                performance = self._calculate_portfolio_performance(
                    portfolio_list, price_data, strategy_name, universe
                )
                results[strategy_name][universe] = performance
        
        return results
    
    def _backtest_with_multiprocessing(self, portfolios, price_data, universes_to_process):
        """Multiprocessing backtesting for large datasets"""
        
        # Prepare tasks for multiprocessing
        tasks = []
        for strategy_name, strategy_portfolios in portfolios.items():
            for universe in universes_to_process:
                if universe in strategy_portfolios and strategy_portfolios[universe]:
                    tasks.append((strategy_name, universe, strategy_portfolios[universe], price_data))
        
        if not tasks:
            print("    ⚠️ 백테스트할 작업 없음")
            return {}
        
        # Use multiprocessing for parallel execution
        num_processes = min(len(universes_to_process), 2)
        print(f"    🔄 {num_processes}개 프로세스로 병렬 처리...")
        
        with Pool(processes=min(len(universes_to_process), 2)) as pool:
            results_list = pool.map(self._backtest_universe, tasks)
        
        # Reorganize results
        results = {}
        for (strategy_name, universe), performance in results_list:
            if strategy_name not in results:
                results[strategy_name] = {}
            results[strategy_name][universe] = performance
        
        return results
    
    def _backtest_universe(self, task_data):
        """Backtest a single strategy-universe combination"""
        strategy_name, universe, portfolio_list, price_data = task_data
        
        print(f"    🔄 {strategy_name}-{universe} 백테스트 처리 중...")
        
        try:
            performance = self._calculate_portfolio_performance(
                portfolio_list, price_data, strategy_name, universe
            )
            return (strategy_name, universe), performance
            
        except Exception as e:
            print(f"    ❌ {strategy_name}-{universe} 백테스트 실패: {e}")
            return (strategy_name, universe), self._create_empty_results()
    
    def _create_price_lookup(self, price_data):
        """Create optimized price lookup structure with forward fill"""
        print("    📊 가격 조회 구조 생성 중...")
        
        if price_data.empty:
            print("      ⚠️ 가격 데이터 없음")
            return {}
        
        # Determine price column
        if '종가' in price_data.columns:
            price_col = '종가'
        elif '일간_시가총액' in price_data.columns:
            # Check if we have share count data to calculate per-share price
            if '발행주식수' in price_data.columns or '상장주식수' in price_data.columns:
                share_count_col = '발행주식수' if '발행주식수' in price_data.columns else '상장주식수'
                # Calculate Adjusted Close price
                price_data['Adj_Close'] = pd.to_numeric(price_data['일간_시가총액'], errors='coerce') / pd.to_numeric(price_data[share_count_col], errors='coerce').replace(0, np.nan)
                price_col = 'Adj_Close'
                print(f"      📊 주당 가격 계산: 시가총액 / {share_count_col}")
            else:
                print("      ⚠️ 주식수 정보 없음 - 시가총액 직접 사용")
                price_col = '일간_시가총액'
        else:
            print("      ❌ 가격 컬럼을 찾을 수 없음")
            return {}
        
        # Clean price data
        price_data[price_col] = pd.to_numeric(price_data[price_col], errors='coerce')
        price_data = price_data.replace([np.inf, -np.inf], np.nan)
        
        # Remove stocks with no valid price data
        valid_stocks = price_data.groupby('거래소코드')[price_col].count()
        valid_stock_codes = valid_stocks[valid_stocks > 0].index
        price_data = price_data[price_data['거래소코드'].isin(valid_stock_codes)]
        
        if price_data.empty:
            print("      ⚠️ 유효한 가격 데이터 없음")
            return {}
        
        # Create pivot table and forward fill
        print(f"      📊 {price_col} 컬럼 사용하여 피벗 테이블 생성...")
        price_pivot = price_data.pivot_table(
            index='date', 
            columns='거래소코드', 
            values=price_col, 
            aggfunc='last'
        ).sort_index().ffill()  # Forward fill missing values
        
        # Create O(1) lookup dictionary
        price_lookup = price_pivot.stack().to_dict()
        
        print(f"      ✅ 가격 조회 구조 완성: {len(price_lookup):,}개 가격 포인트")
        return price_lookup
    
    def _calculate_portfolio_performance(self, portfolio_list, price_data, strategy_name, universe):
        """
        Calculate performance for a single portfolio
        
        수정 사항:
        1. 리밸런스 시 종목별 shares = w * portfolio_value / entry_price로 고정
        2. entry_price = 리밸런스 날짜 첫 거래일 가격
        3. 보유기간 동안 shares 불변, daily_value = Σ(shares_i × price_i)로 계산
        4. 거래비용: 매도·매수 체결 금액 기준으로 차감
        5. holding_dates = price pivot 인덱스로 대체하여 중복 필터 제거
        """
        print(f"      📊 {strategy_name}-{universe} 성과 계산 중...")
        
        if not portfolio_list:
            print(f"        ⚠️ 포트폴리오 리스트 비어있음")
            return self._create_empty_results()
        
        # Create price lookup structure and get available dates
        price_lookup = self._create_price_lookup(price_data)
        if not price_lookup:
            print(f"        ❌ 가격 데이터 없음")
            return self._create_empty_results()
        
        # Get available trading dates from price data
        available_dates = sorted(set(date for date, _ in price_lookup.keys()))
        
        # Get transaction costs
        transaction_costs = self.config['portfolio_params'].get('transaction_cost', 0.003)
        
        # Initialize portfolio tracking
        portfolio_values = []
        daily_returns = []
        portfolio_value = self.initial_capital
        current_shares = {}  # Track shares held for each stock
        
        print(f"        📅 일일 성과 계산 시작...")
        
        for i, portfolio in enumerate(portfolio_list):
            rebalance_date = portfolio['date']
            stocks = portfolio['stocks']
            
            print(f"        🔄 리밸런싱 {i+1}/{len(portfolio_list)}: {rebalance_date.strftime('%Y-%m-%d')} ({len(stocks)}개 종목)")
            
            # Determine holding period using available trading dates
            if i < len(portfolio_list) - 1:
                next_rebalance = portfolio_list[i + 1]['date']
                holding_dates = [d for d in available_dates if rebalance_date <= d < next_rebalance]
            else:
                # Last portfolio - hold for 1 year or until end date
                one_year_later = rebalance_date + pd.DateOffset(years=1)
                end_holding = min(one_year_later, self.end_date)
                holding_dates = [d for d in available_dates if rebalance_date <= d <= end_holding]
            
            if not holding_dates:
                print(f"          ⚠️ 보유 기간 없음")
                continue
            
            print(f"          📅 보유 기간: {len(holding_dates)}일")
            
            # Calculate entry prices and shares at rebalancing
            if stocks:
                # Equal weight allocation
                weight_per_stock = 1.0 / len(stocks)
                allocation_per_stock = portfolio_value * weight_per_stock
                
                # Calculate transaction costs for selling old positions
                if current_shares:
                    sell_value = 0
                    for old_stock, shares in current_shares.items():
                        # Get price for selling
                        price_key = (pd.Timestamp(rebalance_date), old_stock)
                        if price_key in price_lookup and shares > 0:
                            sell_price = price_lookup[price_key]
                            if pd.notna(sell_price) and sell_price > 0:
                                sell_value += shares * sell_price
                    
                    # Apply selling transaction costs
                    sell_cost = sell_value * transaction_costs
                    portfolio_value -= sell_cost
                    print(f"          💸 매도 거래비용: {sell_cost:,.0f}원")
                
                # Buy new positions
                new_shares = {}
                buy_value = 0
                
                for stock in stocks:
                    # Get entry price (first available price on or after rebalance date)
                    entry_price = None
                    for date in holding_dates:
                        price_key = (pd.Timestamp(date), stock)
                        if price_key in price_lookup:
                            price = price_lookup[price_key]
                            if pd.notna(price) and price > 0:
                                entry_price = price
                                break
                    
                    if entry_price and entry_price > 0:
                        # Calculate shares to buy
                        shares = allocation_per_stock / entry_price
                        new_shares[stock] = shares
                        buy_value += allocation_per_stock
                        
                        # Check for negative or infinite shares
                        if shares <= 0 or not np.isfinite(shares):
                            print(f"          ⚠️ {stock}: 비정상 주식수 {shares}")
                            new_shares[stock] = 0
                    else:
                        print(f"          ⚠️ {stock}: 진입 가격 없음")
                        new_shares[stock] = 0
                
                # Apply buying transaction costs
                buy_cost = buy_value * transaction_costs
                portfolio_value -= buy_cost
                print(f"          💸 매수 거래비용: {buy_cost:,.0f}원")
                
                current_shares = new_shares
            else:
                print(f"          ⚠️ 선택된 종목 없음")
                current_shares = {}
                continue
            
            # Calculate daily portfolio values during holding period
            prev_value = portfolio_value
            
            for date in holding_dates:
                daily_value = 0
                valid_positions = 0
                
                # Calculate portfolio value as sum of position values
                for stock, shares in current_shares.items():
                    if shares > 0:
                        price_key = (pd.Timestamp(date), stock)
                        if price_key in price_lookup:
                            price = price_lookup[price_key]
                            if pd.notna(price) and price > 0:
                                position_value = shares * price
                                daily_value += position_value
                                valid_positions += 1
                
                if valid_positions > 0 and daily_value > 0:
                    # Calculate return
                    if prev_value > 0:
                        daily_return = daily_value / prev_value - 1
                        
                        # Check for abnormal returns
                        if np.isfinite(daily_return):
                            daily_returns.append(daily_return)
                        else:
                            daily_returns.append(0)  # Replace infinite returns with 0
                    else:
                        daily_returns.append(0)
                        
                    portfolio_values.append({
                        'date': date,
                        'value': daily_value,
                        'return': daily_returns[-1] if daily_returns else 0
                    })
                    prev_value = daily_value
                else:
                    print(f"          ⚠️ {date.strftime('%Y-%m-%d')} - 유효 포지션 없음")
                    daily_returns.append(0)  # Use 0 instead of NaN for missing data
            
            # Update portfolio value for next rebalancing
            if portfolio_values:
                portfolio_value = portfolio_values[-1]['value']
        
        # Clean returns - filter extreme values
        daily_returns = [r for r in daily_returns if np.isfinite(r) and abs(r) < 1.0]  # Remove >100% daily returns
        
        if not daily_returns:
            print(f"        ⚠️ 유효한 수익률 없음")
            return self._create_empty_results()
        
        # Check for flat returns (all zeros)
        if all(r == 0 for r in daily_returns):
            print(f"        ⚠️ 모든 수익률이 0 - 계산 오류 의심")
        
        print(f"        ✅ 성과 계산 완료: {len(daily_returns)}일 수익률 데이터")
        
        # Calculate performance metrics with robust error handling
        returns_series = pd.Series(daily_returns)
        
        if len(returns_series) == 0:
            return self._create_empty_results()
        
        # Calculate total return with error handling
        try:
            total_return = (1 + returns_series).prod() - 1
            if not np.isfinite(total_return):
                total_return = 0
        except:
            total_return = 0
        
        # Annualized return
        try:
            if len(returns_series) > 0:
                annual_return = (1 + total_return) ** (252 / len(returns_series)) - 1
                if not np.isfinite(annual_return):
                    annual_return = 0
            else:
                annual_return = 0
        except:
            annual_return = 0
        
        # Volatility
        try:
            volatility = returns_series.std() * np.sqrt(252)
            if not np.isfinite(volatility):
                volatility = 0
        except:
            volatility = 0
        
        # Sharpe ratio
        sharpe = annual_return / volatility if volatility > 0 else 0
        if not np.isfinite(sharpe):
            sharpe = 0
        
        # Drawdown calculation with error handling
        try:
            cumulative_returns = (1 + returns_series).cumprod()
            if len(cumulative_returns) > 0:
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - rolling_max) / rolling_max.replace(0, np.nan)
                max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
                if not np.isfinite(max_drawdown):
                    max_drawdown = 0
            else:
                max_drawdown = 0
        except:
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'daily_returns': returns_series.tolist()
        }
    
    def _create_empty_results(self):
        """Create empty results structure"""
        return {
            'total_return': 0,
            'annual_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'portfolio_values': [],
            'daily_returns': []
        }
    
    def _validate_backtest_results(self, results, portfolios, master_df):
        """
        백테스팅 오류·실수 검출 가이드 구현
        
        검증 항목:
        - 룩어헤드(미래정보) 검증
        - 데이터 누수 검사
        - 롤링 윈도 갱신 확인
        - 일일 수익률 평탄화 검사
        - 거래일 불일치 확인
        - 거래비용 검증
        - Shares 음수/무한대 검사
        - 최대 낙폭 계산 검증
        """
        print("  🔍 백테스팅 검증 시작...")
        
        validation_issues = []
        
        # 1. 룩어헤드 바이어스 검증
        print("    📅 룩어헤드 바이어스 검사...")
        for strategy_name, strategy_portfolios in portfolios.items():
            for universe, portfolio_list in strategy_portfolios.items():
                for i, portfolio in enumerate(portfolio_list):
                    rebalance_date = portfolio['date']
                    
                    # 재무제표 데이터가 리밸런스 날짜 이후 것을 사용했는지 확인
                    if '연도' in master_df.columns:
                        year_col = '연도'
                    elif '회계년도' in master_df.columns:
                        year_col = '회계년도'
                    else:
                        continue
                    
                    # 리밸런스 연도 이후 데이터 사용 여부 확인
                    rebalance_year = rebalance_date.year
                    future_data = master_df[master_df[year_col] > rebalance_year]
                    if not future_data.empty:
                        validation_issues.append(f"⚠️ {strategy_name}-{universe}: {rebalance_date.strftime('%Y-%m-%d')} 리밸런싱에서 미래 데이터 사용 의심")
        
        # 2. 일일 수익률 평탄화 검사
        print("    📊 수익률 평탄화 검사...")
        for strategy_name, strategy_results in results.items():
            for universe, performance in strategy_results.items():
                daily_returns = performance.get('daily_returns', [])
                
                if len(daily_returns) > 10:  # 충분한 데이터가 있을 때만 검사
                    # 모든 수익률이 0인지 확인
                    if all(r == 0 for r in daily_returns):
                        validation_issues.append(f"⚠️ {strategy_name}-{universe}: 모든 일일 수익률이 0 - 계산 오류 의심")
                    
                    # 동일한 값이 90% 이상 반복되는지 확인
                    from collections import Counter
                    value_counts = Counter(daily_returns)
                    max_count = max(value_counts.values()) if value_counts else 0
                    if max_count > len(daily_returns) * 0.9:
                        validation_issues.append(f"⚠️ {strategy_name}-{universe}: 수익률이 과도하게 반복됨 - 계산 오류 의심")
                    
                    # 극단적 수익률 확인 (일일 100% 이상)
                    extreme_returns = [r for r in daily_returns if abs(r) > 1.0]
                    if extreme_returns:
                        validation_issues.append(f"⚠️ {strategy_name}-{universe}: 극단적 일일 수익률 {len(extreme_returns)}개 발견")
        
        # 3. 성과 지표 검증
        print("    📈 성과 지표 검증...")
        for strategy_name, strategy_results in results.items():
            for universe, performance in strategy_results.items():
                # 무한대값 또는 NaN 확인
                metrics = ['total_return', 'annual_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
                for metric in metrics:
                    value = performance.get(metric, 0)
                    if not np.isfinite(value):
                        validation_issues.append(f"⚠️ {strategy_name}-{universe}: {metric}이 비정상값 ({value})")
                
                # 샤프 비율 범위 확인 (일반적으로 -5 ~ 5 범위)
                sharpe = performance.get('sharpe_ratio', 0)
                if abs(sharpe) > 10:
                    validation_issues.append(f"⚠️ {strategy_name}-{universe}: 샤프 비율이 비현실적 ({sharpe:.2f})")
                
                # 변동성 검사 (0% ~ 200% 범위)
                volatility = performance.get('volatility', 0)
                if volatility > 2.0 or volatility < 0:
                    validation_issues.append(f"⚠️ {strategy_name}-{universe}: 변동성이 비현실적 ({volatility:.2%})")
        
        # 4. 포트폴리오 구성 검증
        print("    🎯 포트폴리오 구성 검증...")
        for strategy_name, strategy_portfolios in portfolios.items():
            for universe, portfolio_list in strategy_portfolios.items():
                if not portfolio_list:
                    validation_issues.append(f"⚠️ {strategy_name}-{universe}: 포트폴리오가 생성되지 않음")
                    continue
                
                # 각 포트폴리오의 종목 수 확인
                for i, portfolio in enumerate(portfolio_list):
                    stocks = portfolio.get('stocks', [])
                    if len(stocks) == 0:
                        validation_issues.append(f"⚠️ {strategy_name}-{universe}: {portfolio['date'].strftime('%Y-%m-%d')} 포트폴리오에 종목 없음")
                    elif len(stocks) < 5:
                        validation_issues.append(f"⚠️ {strategy_name}-{universe}: {portfolio['date'].strftime('%Y-%m-%d')} 포트폴리오 종목수 부족 ({len(stocks)}개)")
        
        # 5. 결과 요약
        if validation_issues:
            print("  ❌ 검증 이슈 발견:")
            for issue in validation_issues[:10]:  # 최대 10개만 표시
                print(f"    {issue}")
            if len(validation_issues) > 10:
                print(f"    ... 총 {len(validation_issues)}개 이슈 발견")
        else:
            print("  ✅ 백테스팅 검증 통과 - 주요 오류 없음")
        
        print("  🔍 백테스팅 검증 완료")


class ResultsAnalyzer:
    """Analysis and visualization of backtest results"""
    
    def __init__(self, config):
        self.config = config
        
    def analyze_results(self, results):
        """Analyze and display backtest results"""
        print("📊 결과 분석 시작...")
        
        # Create performance summary
        summary_df = self._create_performance_summary(results)
        
        # Display results
        self._display_summary(summary_df)
        
        # Save results
        self._save_results(summary_df, results)
        
        return summary_df
    
    def _create_performance_summary(self, results):
        """Create performance summary dataframe"""
        summary_data = []
        
        for strategy, universes in results.items():
            for universe, performance in universes.items():
                summary_data.append({
                    'Strategy': strategy,
                    'Universe': universe,
                    'Total_Return': performance['total_return'],
                    'Annual_Return': performance['annual_return'],
                    'Volatility': performance['volatility'],
                    'Sharpe_Ratio': performance['sharpe_ratio'],
                    'Max_Drawdown': performance['max_drawdown']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by Sharpe ratio
        summary_df = summary_df.sort_values('Sharpe_Ratio', ascending=False)
        
        return summary_df
    
    def _display_summary(self, summary_df):
        """Display performance summary"""
        print("\n📈 백테스트 결과 요약:")
        print("=" * 80)
        
        for _, row in summary_df.iterrows():
            print(f"🎯 {row['Strategy']} ({row['Universe']}):")
            print(f"   총 수익률: {row['Total_Return']:.2%}")
            print(f"   연간 수익률: {row['Annual_Return']:.2%}")
            print(f"   변동성: {row['Volatility']:.2%}")
            print(f"   샤프 비율: {row['Sharpe_Ratio']:.3f}")
            print(f"   최대 낙폭: {row['Max_Drawdown']:.2%}")
            print("-" * 40)
    
    def _save_results(self, summary_df, detailed_results):
        """Save results to files"""
        output_dir = Path(self.config.get('output_dir', 'outputs/backtesting_v3'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_path = output_dir / 'performance_summary.csv'
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"📄 요약 결과 저장: {summary_path}")
        
        # Save detailed results
        detailed_path = output_dir / 'detailed_results.yaml'
        with open(detailed_path, 'w', encoding='utf-8') as f:
            yaml.dump(detailed_results, f, default_flow_style=False, allow_unicode=True)
        print(f"📄 상세 결과 저장: {detailed_path}")


def main():
    """Main execution function"""
    print("🚀 Factor Backtesting Framework v3.0 시작")
    print("=" * 60)
    
    # Load configuration
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ 설정 파일 로드 완료")
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        return
    
    try:
        # 1. Data Loading
        print("\n" + "="*60)
        data_handler = DataHandler(config)
        data_handler.load_data()
        
        if data_handler.master_df is None or data_handler.master_df.empty:
            print("❌ 마스터 데이터프레임이 비어있음 - 실행 중단")
            return
        
        # 2. Factor Calculation
        print("\n" + "="*60)
        factor_engine = FactorEngine(config)
        master_df = factor_engine.compute_factors(data_handler.master_df)
        
        # 3. Portfolio Construction
        print("\n" + "="*60)
        strategy_builder = StrategyBuilder(config)
        portfolios = strategy_builder.build_portfolios(master_df, factor_engine)
        
        # Check if any portfolios were created
        total_portfolios = sum(len(strategy_portfolios.get('All', [])) + len(strategy_portfolios.get('Normal', []))
                             for strategy_portfolios in portfolios.values())
        
        if total_portfolios == 0:
            print("❌ 생성된 포트폴리오 없음 - 백테스트 중단")
            return
        
        print(f"✅ 총 {total_portfolios}개 포트폴리오 생성됨")
        
        # 4. Backtesting
        print("\n" + "="*60)
        backtest_engine = BacktestEngine(config)
        
        # Use price data for backtesting
        if not data_handler.daily_price_df.empty:
            price_data = data_handler.daily_price_df
        else:
            # Fallback to master dataframe price data
            price_columns = ['거래소코드', 'date']
            if '종가' in master_df.columns:
                price_columns.append('종가')
            elif '일간_시가총액' in master_df.columns:
                price_columns.append('일간_시가총액')
            else:
                print("❌ 가격 데이터 없음 - 백테스트 불가")
                return
                
            price_data = master_df[price_columns].dropna()
        
        results = backtest_engine.run_backtest(portfolios, price_data)
        
        # 4.5. Backtesting Error Detection
        print("\n" + "="*50)
        print("🔍 백테스팅 오류 검출 수행...")
        backtest_engine._validate_backtest_results(results, portfolios, master_df)
        
        # 5. Results Analysis
        print("\n" + "="*60)
        analyzer = ResultsAnalyzer(config)
        analysis_summary = analyzer.analyze_results(results)
        
        print("\n🎉 백테스트 완료!")
        
    except Exception as e:
        print(f"\n💥 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()