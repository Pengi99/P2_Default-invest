"""
Factor Backtesting Framework v4.0
Simplified architecture using pre-processed master_df.csv
Focus on proper momentum strategy implementation without extreme value filtering
Enhanced with verified formulas and improved logic from v3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import yaml
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy import stats
from pathlib import Path

# Statistical modeling imports
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    from sklearn.linear_model import LinearRegression
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Using sklearn for regression.")

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


class FactorBacktestingV4:   
    def __init__(self, config_path='config.yaml'):
        """Initialize backtesting framework"""
        self.config = self._load_config(config_path)
        self.master_df = None
        self.portfolio_size = self.config.get('portfolio_params', {}).get('portfolio_size', 100)
        self.results = {}
        
        print("🚀 Factor Backtesting v4.0 초기화 완료")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"⚠️ 설정 파일 로딩 실패: {e}")
            return {}
    
    def load_master_data(self, master_df_path='data/final/master_df_realworld.csv'):
        """Load pre-processed master dataframe"""
        print("📊 마스터 데이터 로딩...")
        
        try:
            self.master_df = pd.read_csv(master_df_path, encoding='utf-8-sig')
            
            # Data type conversions
            self.master_df['매매년월일'] = pd.to_datetime(self.master_df['매매년월일'])
            self.master_df['종가'] = pd.to_numeric(self.master_df['종가'], errors='coerce')
            self.master_df['시가총액'] = pd.to_numeric(self.master_df['시가총액'], errors='coerce')
            
            # Remove invalid data
            self.master_df = self.master_df.dropna(subset=['거래소코드', '매매년월일', '종가'])
            self.master_df = self.master_df[self.master_df['종가'] > 0]
            
            # Sort by stock code and date
            self.master_df = self.master_df.sort_values(['거래소코드', '매매년월일'])
            
            print(f"✅ 마스터 데이터 로딩 완료:")
            print(f"   - 총 {len(self.master_df):,}행")
            print(f"   - 종목 수: {self.master_df['거래소코드'].nunique()}")
            print(f"   - 기간: {self.master_df['매매년월일'].min().strftime('%Y-%m-%d')} ~ {self.master_df['매매년월일'].max().strftime('%Y-%m-%d')}")
            
        except Exception as e:
            print(f"❌ 마스터 데이터 로딩 실패: {e}")
            self.master_df = None
    
    def calculate_12_1_momentum(self):
        """
        Calculate proper 12-1 momentum strategy
        - master_df는 이미 매월 첫날 데이터만 포함
        - 12개월 수익률 계산 (최근 1개월 제외)
        - 극단값 필터링 없음 (사용자 피드백 반영)
        """
        print("📈 12-1 모멘텀 계산 시작...")
        
        if self.master_df is None:
            print("❌ 마스터 데이터가 없습니다")
            return
        
        # Add momentum column
        self.master_df['momentum_12_1'] = np.nan
        
        # Sort by stock and date for easier calculation
        self.master_df = self.master_df.sort_values(['거래소코드', '매매년월일'])
        
        # Calculate for each stock
        for stock_code in self.master_df['거래소코드'].unique():
            stock_data = self.master_df[self.master_df['거래소코드'] == stock_code].copy()
            stock_data = stock_data.sort_values('매매년월일')
            
            if len(stock_data) < 13:  # Need at least 13 months for 12-1 calculation
                continue
            
            # Since master_df already has monthly first-day data, calculate directly
            for i in range(12, len(stock_data)):
                # Price 1 month ago (t-1) - skip most recent month
                price_t_minus_1 = stock_data.iloc[i-1]['종가']
                # Price 12 months ago (t-12)
                price_t_minus_12 = stock_data.iloc[i-12]['종가']
                
                if price_t_minus_12 > 0 and price_t_minus_1 > 0:
                    momentum = (price_t_minus_1 / price_t_minus_12) - 1
                    
                    # Update current row
                    current_idx = stock_data.iloc[i].name
                    self.master_df.loc[current_idx, 'momentum_12_1'] = momentum
        
        # Report momentum statistics
        valid_momentum = self.master_df.dropna(subset=['momentum_12_1'])
        print(f"✅ 12-1 모멘텀 계산 완료:")
        print(f"   - 유효 데이터: {len(valid_momentum):,}개")
        print(f"   - 평균 모멘텀: {valid_momentum['momentum_12_1'].mean():.4f}")
        print(f"   - 중앙값: {valid_momentum['momentum_12_1'].median():.4f}")
        print(f"   - 표준편차: {valid_momentum['momentum_12_1'].std():.4f}")
        print(f"   - 최댓값: {valid_momentum['momentum_12_1'].max():.4f}")
        print(f"   - 최솟값: {valid_momentum['momentum_12_1'].min():.4f}")
        
        # 극단값 통계 (필터링하지 않지만 모니터링)
        extreme_high = valid_momentum[valid_momentum['momentum_12_1'] > 1.0]
        extreme_low = valid_momentum[valid_momentum['momentum_12_1'] < -0.5]
        print(f"   - 극단값 분포: 100%+ 수익률 {len(extreme_high)}개, -50%+ 손실 {len(extreme_low)}개")
    
    def build_momentum_portfolio(self):
        """Build momentum portfolio with monthly rebalancing (every first day of month)"""
        print("🎯 모멘텀 포트폴리오 구성...")
        
        if self.master_df is None:
            print("❌ 마스터 데이터가 없습니다")
            return None
        
        # Get unique dates (already first day of each month)
        unique_dates = sorted(self.master_df['매매년월일'].unique())
        
        portfolios = {'All': [], 'Normal': []}
        
        for date in unique_dates:
            month_data = self.master_df[self.master_df['매매년월일'] == date].copy()
            
            # Filter valid momentum data
            valid_data = month_data.dropna(subset=['momentum_12_1', '시가총액'])
            
            if len(valid_data) < self.portfolio_size:
                continue
            
            # 유동성 필터링 (시가총액 하위 20% 제외)
            market_cap_threshold = valid_data['시가총액'].quantile(0.2)
            liquid_stocks = valid_data[valid_data['시가총액'] >= market_cap_threshold]
            
            if len(liquid_stocks) < self.portfolio_size:
                liquid_stocks = valid_data  # 필터링 후 부족하면 전체 사용
            
            # All firms portfolio
            top_momentum_all = liquid_stocks.nlargest(self.portfolio_size, 'momentum_12_1')
            
            portfolios['All'].append({
                'date': date,
                'stocks': top_momentum_all['거래소코드'].tolist(),
                'momentum': top_momentum_all['momentum_12_1'].tolist(),
                'market_caps': top_momentum_all['시가총액'].tolist(),
                'prices': top_momentum_all['종가'].tolist()
            })
            
            # Normal firms (non-default) portfolio
            normal_stocks = liquid_stocks[liquid_stocks.get('default', 0) == 0]
            if len(normal_stocks) >= self.portfolio_size:
                top_momentum_normal = normal_stocks.nlargest(self.portfolio_size, 'momentum_12_1')
                
                portfolios['Normal'].append({
                    'date': date,
                    'stocks': top_momentum_normal['거래소코드'].tolist(),
                    'momentum': top_momentum_normal['momentum_12_1'].tolist(),
                    'market_caps': top_momentum_normal['시가총액'].tolist(),
                    'prices': top_momentum_normal['종가'].tolist()
                })
        
        print(f"✅ 모멘텀 포트폴리오 구성 완료: All {len(portfolios['All'])}개월, Normal {len(portfolios['Normal'])}개월")
        return portfolios
    
    def backtest_momentum(self, portfolios):
        """Backtest momentum strategy with optimized monthly data access"""
        print("💰 모멘텀 백테스팅 실행...")
        
        if not portfolios:
            print("❌ 포트폴리오가 없습니다")
            return None
        
        portfolio_values = [1.0]  # 초기 자본
        monthly_returns = []
        
        for i in range(len(portfolios) - 1):
            current_portfolio = portfolios[i]
            next_portfolio = portfolios[i + 1]
            
            # 현재 포트폴리오의 다음 달 수익률 계산
            portfolio_return = 0
            valid_stocks = 0
            
            for j, stock_code in enumerate(current_portfolio['stocks']):
                current_price = current_portfolio['prices'][j]
                
                # 다음 달 가격 찾기 (master_df는 이미 매월 첫날 데이터)
                next_month_data = self.master_df[
                    (self.master_df['거래소코드'] == stock_code) & 
                    (self.master_df['매매년월일'] == next_portfolio['date'])
                ]
                
                if len(next_month_data) > 0:
                    next_price = next_month_data.iloc[0]['종가']
                    
                    if current_price > 0 and next_price > 0:
                        stock_return = (next_price / current_price) - 1
                        portfolio_return += stock_return
                        valid_stocks += 1
            
            if valid_stocks > 0:
                portfolio_return = portfolio_return / valid_stocks  # 동일가중평균
                monthly_returns.append(portfolio_return)
                new_value = portfolio_values[-1] * (1 + portfolio_return)
                portfolio_values.append(new_value)
            else:
                monthly_returns.append(0)
                portfolio_values.append(portfolio_values[-1])
        
        # 성과 지표 계산
        results = self._calculate_performance_metrics(monthly_returns, portfolio_values)
        
        print(f"📊 모멘텀 백테스팅 결과:")
        print(f"   - 총 수익률: {results['total_return']:.4f} ({results['total_return']*100:.2f}%)")
        print(f"   - 연간 수익률: {results['annual_return']:.4f} ({results['annual_return']*100:.2f}%)")
        print(f"   - 연간 변동성: {results['volatility']:.4f} ({results['volatility']*100:.2f}%)")
        print(f"   - 샤프 비율: {results['sharpe_ratio']:.4f}")
        print(f"   - 소티노 비율: {results['sortino_ratio']:.4f}")
        print(f"   - 정보 비율: {results['information_ratio']:.4f}")
        print(f"   - 최대 낙폭: {results['max_drawdown']:.4f} ({results['max_drawdown']*100:.2f}%)")
        
        return results
    
    def _calculate_performance_metrics(self, monthly_returns, portfolio_values):
        """Calculate comprehensive performance metrics with corrected formulas"""
        monthly_returns_array = np.array(monthly_returns)
        
        # Basic returns - CORRECTED FORMULA
        if len(portfolio_values) > 1:
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1  # 수정된 공식
        else:
            total_return = 0
            
        n_periods = len(monthly_returns)
        
        # Annualized metrics - periods could be monthly or annual
        if n_periods > 0 and total_return > -1:
            # Determine if this is monthly or annual data based on number of periods
            if n_periods > 24:  # Monthly data (more than 2 years)
                periods_per_year = 12
                annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
                annual_volatility = np.std(monthly_returns_array) * np.sqrt(periods_per_year)
            else:  # Annual data (less than 24 periods)
                periods_per_year = 1
                annual_return = (1 + total_return) ** (1 / (n_periods / periods_per_year)) - 1
                annual_volatility = np.std(monthly_returns_array) * np.sqrt(1)
        else:
            annual_return = 0
            annual_volatility = 0
            
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Sortino Ratio - using downside deviation
        # 연도별 무위험 수익률 데이터 활용 (2023년 기준)
        annual_risk_free_rates = {
            2012: 3.42, 2013: 3.37, 2014: 3.05, 2015: 2.37, 2016: 1.73,
            2017: 2.32, 2018: 2.60, 2019: 1.74, 2020: 1.41, 2021: 2.08,
            2022: 3.37, 2023: 3.64
        }
        risk_free_rate = annual_risk_free_rates.get(2023, 0.02) / 100  # 2023년 기준 또는 기본값 2%
        negative_returns = monthly_returns_array[monthly_returns_array < 0]
        if len(negative_returns) > 0:
            if n_periods > 24:  # Monthly data
                downside_deviation = np.std(negative_returns) * np.sqrt(periods_per_year)
                sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
            else:  # Annual data
                downside_deviation = np.std(negative_returns)
                sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
        else:
            sortino_ratio = 0
        
        # Information Ratio (vs market benchmark)
        # Using a simple market return assumption of 8% annually
        market_return = 0.08  # 8% annual market return
        excess_return = annual_return - market_return
        if annual_volatility > 0:
            information_ratio = excess_return / annual_volatility
        else:
            information_ratio = 0
        
        # Maximum drawdown
        if len(portfolio_values) > 1:
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (np.array(portfolio_values) - peak) / peak
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown,
            'monthly_returns': monthly_returns,
            'portfolio_values': portfolio_values,
            'n_months': n_periods
        }
    
    def calculate_magic_formula(self):
        """Calculate Magic Formula factors with verified formulas"""
        print("🪄 Magic Formula 계산 시작...")
        
        if self.master_df is None:
            print("❌ 마스터 데이터가 없습니다")
            return
        
        # Add Magic Formula columns
        self.master_df['earnings_yield'] = np.nan
        self.master_df['roic'] = np.nan
        self.master_df['magic_signal'] = np.nan
        
        # Calculate for each year
        for year in self.master_df['연도'].unique():
            year_data = self.master_df[self.master_df['연도'] == year].copy()
            
            if len(year_data) == 0:
                continue
            
            # 1. Calculate Earnings Yield = EBIT / Enterprise Value
            # Enterprise Value = Market Cap + Net Debt (Total Debt - Cash)
            if '영업이익' in year_data.columns and '시가총액' in year_data.columns:
                # Basic components
                market_cap = year_data['시가총액'].fillna(0)
                ebit = year_data['영업이익'].fillna(0)
                
                # Debt calculation (use available debt columns)
                total_debt = pd.Series(0, index=year_data.index)
                if '총부채' in year_data.columns:
                    total_debt = year_data['총부채'].fillna(0)
                elif '유동부채' in year_data.columns and '비유동부채' in year_data.columns:
                    current_debt = year_data['유동부채'].fillna(0)
                    non_current_debt = year_data['비유동부채'].fillna(0)
                    total_debt = current_debt + non_current_debt
                
                cash = year_data.get('현금및현금성자산', pd.Series(0, index=year_data.index)).fillna(0)
                
                # Enterprise Value = Market Cap + Net Debt
                net_debt = total_debt - cash
                enterprise_value = market_cap + net_debt
                
                # Filter valid data (EV > 0, EBIT exists)
                valid_mask = (enterprise_value > 0) & (market_cap > 0) & (ebit.notna())
                enterprise_value = enterprise_value.where(valid_mask, np.nan)
                
                # Earnings Yield = EBIT / Enterprise Value
                earnings_yield = ebit / enterprise_value
                
                # Update master dataframe
                self.master_df.loc[year_data.index, 'earnings_yield'] = earnings_yield
            else:
                earnings_yield = pd.Series(np.nan, index=year_data.index)
            
            # 2. Calculate ROIC = EBIT / Invested Capital
            if '투하자본수익률(ROIC)' in year_data.columns:
                # Use existing ROIC if available and positive
                roic = year_data['투하자본수익률(ROIC)'].copy()
                roic = roic.where(roic.notna(), np.nan)
            elif '영업이익' in year_data.columns and '총자산' in year_data.columns:
                # Calculate ROIC = EBIT / (Total Assets - Current Liabilities)
                # This approximates invested capital
                total_assets = year_data['총자산'].fillna(0)
                current_liabilities = year_data.get('유동부채', pd.Series(0, index=year_data.index)).fillna(0)
                
                # Invested Capital ≈ Total Assets - Current Liabilities
                invested_capital = total_assets - current_liabilities
                
                # Filter valid data
                valid_mask = (invested_capital > 0) & (total_assets > 0)
                invested_capital = invested_capital.where(valid_mask, np.nan)
                
                ebit = year_data['영업이익'].fillna(0)
                roic = ebit / invested_capital
            else:
                roic = pd.Series(np.nan, index=year_data.index)
            
            # Update master dataframe
            self.master_df.loc[year_data.index, 'roic'] = roic
            
            # 3. Calculate Magic Formula Ranking (Greenblatt's method)
            # Both metrics must be positive and valid
            valid_mask = (earnings_yield > 0) & (roic > 0) & earnings_yield.notna() & roic.notna()
            
            if valid_mask.sum() >= 10:  # Need minimum stocks for ranking
                # Filter to valid stocks only
                valid_ey = earnings_yield[valid_mask]
                valid_roic = roic[valid_mask]
                
                # Rank within year (1 = best, higher number = worse)
                # Higher earnings yield is better (rank ascending=False)
                ey_rank = valid_ey.rank(ascending=False, method='min')
                
                # Higher ROIC is better (rank ascending=False)
                roic_rank = valid_roic.rank(ascending=False, method='min')
                
                # Combined rank (lower combined rank = better)
                combined_rank = ey_rank + roic_rank
                
                # Convert to signal (higher signal = better stock)
                # Best stock has lowest combined rank
                max_combined_rank = combined_rank.max()
                magic_signal_valid = max_combined_rank - combined_rank + 1
                
                # Create full series with NaN for invalid stocks
                magic_signal = pd.Series(np.nan, index=year_data.index)
                magic_signal[valid_mask] = magic_signal_valid
            else:
                magic_signal = pd.Series(np.nan, index=year_data.index)
            
            # Update master dataframe
            self.master_df.loc[year_data.index, 'magic_signal'] = magic_signal
        
        # Report statistics
        valid_magic = self.master_df.dropna(subset=['magic_signal'])
        valid_ey = self.master_df.dropna(subset=['earnings_yield'])
        valid_roic = self.master_df.dropna(subset=['roic'])
        
        print(f"✅ Magic Formula 계산 완룈:")
        print(f"   - Magic Signal 유효 데이터: {len(valid_magic):,}개")
        print(f"   - Earnings Yield 유효 데이터: {len(valid_ey):,}개")
        print(f"   - ROIC 유효 데이터: {len(valid_roic):,}개")
        if len(valid_ey) > 0:
            print(f"   - 평균 Earnings Yield: {valid_ey['earnings_yield'].mean():.4f}")
            print(f"   - Earnings Yield 범위: {valid_ey['earnings_yield'].min():.4f} ~ {valid_ey['earnings_yield'].max():.4f}")
        if len(valid_roic) > 0:
            print(f"   - 평균 ROIC: {valid_roic['roic'].mean():.4f}")
            print(f"   - ROIC 범위: {valid_roic['roic'].min():.4f} ~ {valid_roic['roic'].max():.4f}")
        if len(valid_magic) > 0:
            print(f"   - 평균 Magic Signal: {valid_magic['magic_signal'].mean():.2f}")
            print(f"   - Magic Signal 범위: {valid_magic['magic_signal'].min():.0f} ~ {valid_magic['magic_signal'].max():.0f}")
    
    def build_magic_formula_portfolio(self):
        """Build Magic Formula portfolio with April 1st annual rebalancing"""
        print("🪄 Magic Formula 포트폴리오 구성...")
        
        if self.master_df is None or 'magic_signal' not in self.master_df.columns:
            print("❌ Magic Formula 데이터가 없습니다")
            return None
        
        # Get annual rebalancing dates - use first available date in April each year
        unique_dates = sorted(self.master_df['매매년월일'].unique())
        all_april_dates = [date for date in unique_dates if date.month == 4]
        april_years = list(set([date.year for date in all_april_dates]))
        april_dates = []
        for year in sorted(april_years):
            year_april_dates = [date for date in all_april_dates if date.year == year]
            if year_april_dates:
                april_dates.append(min(year_april_dates))
        
        portfolios = {'All': [], 'Normal': []}
        
        for rebalance_date in april_dates:
            # Get data for this April date
            date_data = self.master_df[self.master_df['매매년월일'] == rebalance_date].copy()
            
            # All firms portfolio - select stocks with valid magic signal
            valid_data = date_data.dropna(subset=['magic_signal'])
            
            # Additional filter: ensure positive fundamentals
            positive_fundamentals = valid_data[
                (valid_data.get('earnings_yield', 0) > 0) & 
                (valid_data.get('roic', 0) > 0)
            ]
            
            if len(positive_fundamentals) >= self.portfolio_size:
                # Select top stocks by magic signal (higher = better)
                top_stocks = positive_fundamentals.nlargest(self.portfolio_size, 'magic_signal')
            elif len(valid_data) >= self.portfolio_size:
                # Fallback to any valid magic signal
                top_stocks = valid_data.nlargest(self.portfolio_size, 'magic_signal')
            else:
                top_stocks = None
            
            if top_stocks is not None and len(top_stocks) >= self.portfolio_size:
                
                portfolios['All'].append({
                    'date': rebalance_date,
                    'stocks': top_stocks['거래소코드'].tolist(),
                    'signals': top_stocks['magic_signal'].tolist()
                })
            
            # Normal firms (non-default) portfolio
            normal_data = positive_fundamentals[positive_fundamentals.get('default', 0) == 0] if len(positive_fundamentals) >= self.portfolio_size else valid_data[valid_data.get('default', 0) == 0]
            
            if len(normal_data) >= self.portfolio_size:
                top_normal_stocks = normal_data.nlargest(self.portfolio_size, 'magic_signal')
            else:
                top_normal_stocks = None
            
            if top_normal_stocks is not None and len(top_normal_stocks) >= self.portfolio_size:
                
                portfolios['Normal'].append({
                    'date': rebalance_date,
                    'stocks': top_normal_stocks['거래소코드'].tolist(),
                    'signals': top_normal_stocks['magic_signal'].tolist()
                })
        
        print(f"✅ Magic Formula 포트폴리오 구성 완료: All {len(portfolios['All'])}개년, Normal {len(portfolios['Normal'])}개년")
        return portfolios
    
    def calculate_fscore(self):
        """Calculate Piotroski F-Score with simplified approach"""
        print("📊 F-Score 계산 시작...")
        
        if self.master_df is None:
            print("❌ 마스터 데이터가 없습니다")
            return
        
        # Simple F-Score calculation using available ratios
        self.master_df['fscore'] = 0
        
        # 1. ROA > 0
        roa_condition = self.master_df['총자산수익률(ROA)'].fillna(0) > 0
        self.master_df.loc[roa_condition, 'fscore'] += 1
        
        # 2. Operating Cash Flow > 0  
        cfo_condition = self.master_df['영업현금흐름'].fillna(0) > 0
        self.master_df.loc[cfo_condition, 'fscore'] += 1
        
        # 3. ROE > 0
        roe_condition = self.master_df['자기자본수익률(ROE)'].fillna(0) > 0
        self.master_df.loc[roe_condition, 'fscore'] += 1
        
        # 4. ROIC > 0
        roic_condition = self.master_df['투하자본수익률(ROIC)'].fillna(0) > 0
        self.master_df.loc[roic_condition, 'fscore'] += 1
        
        # 5. Low debt ratio (< 0.4)
        debt_condition = self.master_df['부채자본비율'].fillna(1) < 0.4
        self.master_df.loc[debt_condition, 'fscore'] += 1
        
        # 6. High asset turnover (> 0.5)
        turnover_condition = self.master_df['총자산회전율'].fillna(0) > 0.5
        self.master_df.loc[turnover_condition, 'fscore'] += 1
        
        # 7. Positive operating margin (use earnings yield as proxy)
        if 'earnings_yield' in self.master_df.columns:
            margin_condition = self.master_df['earnings_yield'].fillna(0) > 0
            self.master_df.loc[margin_condition, 'fscore'] += 1
        
        # 8. Low PBR (< 1.5)
        pbr_condition = self.master_df['PBR'].fillna(10) < 1.5
        self.master_df.loc[pbr_condition, 'fscore'] += 1
        
        # 9. Reasonable PER (0 < PER < 15)
        per_condition = (self.master_df['PER'].fillna(100) > 0) & (self.master_df['PER'].fillna(100) < 15)
        self.master_df.loc[per_condition, 'fscore'] += 1
        
        # Report statistics
        valid_fscore = self.master_df[self.master_df['fscore'] >= 0]
        print(f"✅ F-Score 계산 완료:")
        print(f"   - 유효 데이터: {len(valid_fscore):,}개")
        print(f"   - 평균 F-Score: {valid_fscore['fscore'].mean():.2f}")
        print(f"   - F-Score 분포:")
        for score in range(10):
            count = len(valid_fscore[valid_fscore['fscore'] == score])
            print(f"     F-Score {score}: {count:,}개 ({count/len(valid_fscore)*100:.1f}%)")
    
    def build_fscore_portfolio(self):
        """Build F-Score portfolio with April 1st annual rebalancing"""
        print("📊 F-Score 포트폴리오 구성...")
        
        if self.master_df is None or 'fscore' not in self.master_df.columns:
            print("❌ F-Score 데이터가 없습니다")
            return None
        
        # Get annual rebalancing dates - use first available date in April each year
        unique_dates = sorted(self.master_df['매매년월일'].unique())
        all_april_dates = [date for date in unique_dates if date.month == 4]
        april_years = list(set([date.year for date in all_april_dates]))
        april_dates = []
        for year in sorted(april_years):
            year_april_dates = [date for date in all_april_dates if date.year == year]
            if year_april_dates:
                april_dates.append(min(year_april_dates))
        
        min_score = self.config.get('strategy_params', {}).get('f_score', {}).get('min_score', 8)  # Lowered to 2 for more portfolios
        
        portfolios = {'All': [], 'Normal': []}
        
        for rebalance_date in april_dates:
            # Get data for this April date
            date_data = self.master_df[self.master_df['매매년월일'] == rebalance_date].copy()
            
            # Filter stocks with F-Score >= minimum threshold
            valid_data = date_data[date_data['fscore'] >= min_score]
            
            # All firms portfolio
            if len(valid_data) >= self.portfolio_size:
                top_stocks = valid_data.nlargest(self.portfolio_size, 'fscore')
                
                portfolios['All'].append({
                    'date': rebalance_date,
                    'stocks': top_stocks['거래소코드'].tolist(),
                    'signals': top_stocks['fscore'].tolist()
                })
            
            # Normal firms (non-default) portfolio
            normal_data = valid_data[valid_data.get('default', 0) == 0]
            if len(normal_data) >= self.portfolio_size:
                top_normal_stocks = normal_data.nlargest(self.portfolio_size, 'fscore')
                
                portfolios['Normal'].append({
                    'date': rebalance_date,
                    'stocks': top_normal_stocks['거래소코드'].tolist(),
                    'signals': top_normal_stocks['fscore'].tolist()
                })
        
        print(f"✅ F-Score 포트폴리오 구성 완료: All {len(portfolios['All'])}개년, Normal {len(portfolios['Normal'])}개년")
        return portfolios
    
    def calculate_ff3_factors(self):
        """Calculate Fama-French 3-Factor model factors optimized for monthly first-day data"""
        print("📈 FF3 팩터 구축 시작...")
        
        if self.master_df is None:
            print("❌ 마스터 데이터가 없습니다")
            return None
        
        monthly_data = []
        
        # Get sorted unique dates (already first day of each month)
        unique_dates = sorted(self.master_df['매매년월일'].unique())
        
        # Calculate monthly returns first
        monthly_returns = {}  # {(stock_code, date): return}
        
        for i in range(1, len(unique_dates)):
            current_date = unique_dates[i]
            prev_date = unique_dates[i-1]
            
            current_data = self.master_df[self.master_df['매매년월일'] == current_date]
            prev_data = self.master_df[self.master_df['매매년월일'] == prev_date]
            
            # Merge to calculate returns
            merged = pd.merge(
                current_data[['거래소코드', '종가', '시가총액']], 
                prev_data[['거래소코드', '종가']], 
                on='거래소코드', 
                suffixes=('_curr', '_prev')
            )
            
            merged['monthly_return'] = (merged['종가_curr'] / merged['종가_prev']) - 1
            
            for _, row in merged.iterrows():
                monthly_returns[(row['거래소코드'], current_date)] = row['monthly_return']
        
        # Calculate FF3 factors for each month
        for i in range(1, len(unique_dates)):
            current_date = unique_dates[i]
            current_data = self.master_df[self.master_df['매매년월일'] == current_date].copy()
            
            if len(current_data) == 0:
                continue
            
            # Add monthly returns to current data
            current_data['monthly_return'] = current_data['거래소코드'].apply(
                lambda x: monthly_returns.get((x, current_date), 0)
            )
            
            # Calculate market return (value-weighted)
            total_market_cap = current_data['시가총액'].sum()
            if total_market_cap > 0:
                current_data['weight'] = current_data['시가총액'] / total_market_cap
                market_return = (current_data['monthly_return'] * current_data['weight']).sum()
            else:
                market_return = 0
            
            # Create size and value groups
            if '시가총액' in current_data.columns:
                # Size breakpoint: median market cap
                size_median = current_data['시가총액'].median()
                current_data['size_group'] = current_data['시가총액'].apply(
                    lambda x: 'Small' if x <= size_median else 'Big'
                )
            else:
                current_data['size_group'] = 'Big'
            
            # Value groups using PBR (lower PBR = higher value)
            if 'PBR' in current_data.columns:
                value_30th = current_data['PBR'].quantile(0.3)
                value_70th = current_data['PBR'].quantile(0.7)
                current_data['value_group'] = current_data['PBR'].apply(
                    lambda x: 'High' if x <= value_30th else 'Low' if x >= value_70th else 'Medium'
                )
            else:
                current_data['value_group'] = 'Medium'
            
            # Calculate 6-portfolio returns
            portfolios = {}
            for size in ['Small', 'Big']:
                for value in ['Low', 'Medium', 'High']:
                    portfolio_stocks = current_data[
                        (current_data['size_group'] == size) & 
                        (current_data['value_group'] == value)
                    ]
                    if len(portfolio_stocks) > 0:
                        # Value-weighted return
                        portfolio_market_cap = portfolio_stocks['시가총액'].sum()
                        if portfolio_market_cap > 0:
                            weights = portfolio_stocks['시가총액'] / portfolio_market_cap
                            portfolio_return = (portfolio_stocks['monthly_return'] * weights).sum()
                        else:
                            portfolio_return = portfolio_stocks['monthly_return'].mean()
                        portfolios[f'{size}_{value}'] = portfolio_return
                    else:
                        portfolios[f'{size}_{value}'] = 0
            
            # Calculate SMB and HML factors
            small_avg = (portfolios.get('Small_Low', 0) + 
                        portfolios.get('Small_Medium', 0) + 
                        portfolios.get('Small_High', 0)) / 3
            big_avg = (portfolios.get('Big_Low', 0) + 
                      portfolios.get('Big_Medium', 0) + 
                      portfolios.get('Big_High', 0)) / 3
            smb = small_avg - big_avg
            
            high_avg = (portfolios.get('Small_High', 0) + portfolios.get('Big_High', 0)) / 2
            low_avg = (portfolios.get('Small_Low', 0) + portfolios.get('Big_Low', 0)) / 2
            hml = high_avg - low_avg
            
            # 연도별 무위험 수익률 데이터 활용
            annual_risk_free_rates = {
                2012: 3.42, 2013: 3.37, 2014: 3.05, 2015: 2.37, 2016: 1.73,
                2017: 2.32, 2018: 2.60, 2019: 1.74, 2020: 1.41, 2021: 2.08,
                2022: 3.37, 2023: 3.64
            }
            
            year = current_date.year
            monthly_rf = annual_risk_free_rates.get(year, 2.0) / 100 / 12  # 연율을 월율로 변환
            
            monthly_data.append({
                'date': current_date,
                'Mkt_RF': market_return - monthly_rf,
                'SMB': smb,
                'HML': hml,
                'RF': monthly_rf
            })
        
        if monthly_data:
            ff3_factors = pd.DataFrame(monthly_data)
            ff3_factors = ff3_factors.sort_values('date')
            print(f"✅ FF3 팩터 구축 완료: {len(ff3_factors)}개 월")
            return ff3_factors
        else:
            print("❌ FF3 팩터 구축 실패")
            return None
    
    def calculate_ff3_alpha(self):
        """Calculate FF3-Alpha using 2-year rolling window regression"""
        print("📊 FF3-Alpha 계산 시작 (2년 Rolling Window)...")
        
        if self.master_df is None:
            print("❌ 마스터 데이터가 없습니다")
            return
        
        # Build FF3 factors first
        ff3_factors = self.calculate_ff3_factors()
        if ff3_factors is None:
            return
        
        # Add alpha columns
        self.master_df['ff3_alpha'] = np.nan
        self.master_df['ff3_alpha_pvalue'] = np.nan
        self.master_df['ff3_r_squared'] = np.nan
        
        # Convert FF3 factors to annual (July to July for FF3-Alpha)
        ff3_factors['ff3_year'] = ff3_factors['date'].apply(
            lambda x: x.year if x.month >= 7 else x.year - 1
        )
        annual_ff3 = ff3_factors.groupby('ff3_year').agg({
            'Mkt_RF': lambda x: (1 + x).prod() - 1,
            'SMB': lambda x: (1 + x).prod() - 1,
            'HML': lambda x: (1 + x).prod() - 1,
            'RF': lambda x: (1 + x).prod() - 1
        }).reset_index()
        
        # Get all available years for rolling window
        all_years = sorted(annual_ff3['ff3_year'].unique())
        
        # Calculate alpha for each stock using 2-year rolling window
        for stock_code in self.master_df['거래소코드'].unique():
            stock_data = self.master_df[self.master_df['거래소코드'] == stock_code].copy()
            stock_data = stock_data.sort_values('매매년월일')
            
            if len(stock_data) < 24:  # Need at least 2 years of monthly data
                continue
            
            # Calculate annual returns (July to July) for this stock
            stock_annual_returns = {}
            
            for year in range(stock_data['연도'].min(), stock_data['연도'].max()):
                # July to July period
                start_date = pd.Timestamp(f'{year}-07-01')
                end_date = pd.Timestamp(f'{year+1}-07-01')
                
                # Find closest dates in data
                period_data = stock_data[
                    (stock_data['매매년월일'] >= start_date) & 
                    (stock_data['매매년월일'] < end_date)
                ]
                
                if len(period_data) >= 6:  # Need at least 6 months of data
                    first_price = period_data.iloc[0]['종가']
                    last_price = period_data.iloc[-1]['종가']
                    
                    if first_price > 0 and last_price > 0:
                        annual_return = (last_price / first_price) - 1
                        stock_annual_returns[year] = annual_return
            
            # Rolling 2-year window regression
            for target_year in all_years:
                if target_year < 2014:  # Need 2012-2013 data for 2014 alpha
                    continue
                    
                # Use previous 2 years of data
                window_years = [target_year - 2, target_year - 1]
                
                # Get stock returns for window years
                window_returns = []
                window_year_list = []
                
                for wy in window_years:
                    if wy in stock_annual_returns:
                        window_returns.append(stock_annual_returns[wy])
                        window_year_list.append(wy)
                
                if len(window_returns) < 2:  # Need 2 years minimum
                    continue
                
                # Create regression dataset
                regression_data = pd.DataFrame({
                    'ff3_year': window_year_list,
                    'annual_return': window_returns
                })
                
                # Merge with FF3 factors
                stock_ff3 = pd.merge(regression_data, annual_ff3, on='ff3_year', how='inner')
                
                if len(stock_ff3) < 2:
                    continue
                
                # Prepare regression data
                X = stock_ff3[['Mkt_RF', 'SMB', 'HML']].values
                y = (stock_ff3['annual_return'] - stock_ff3['RF']).values
                
                try:
                    if STATSMODELS_AVAILABLE:
                        # Use statsmodels for p-values
                        X_with_const = sm.add_constant(X)
                        model = sm.OLS(y, X_with_const).fit()
                        alpha = model.params[0]
                        alpha_pvalue = model.pvalues[0]
                        r_squared = model.rsquared
                    else:
                        # Use sklearn
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression().fit(X, y)
                        alpha = model.intercept_
                        alpha_pvalue = 0.05  # Default p-value
                        r_squared = model.score(X, y)
                    
                    # Update target year for this stock
                    mask = (self.master_df['거래소코드'] == stock_code) & (self.master_df['연도'] == target_year)
                    self.master_df.loc[mask, 'ff3_alpha'] = alpha
                    self.master_df.loc[mask, 'ff3_alpha_pvalue'] = alpha_pvalue
                    self.master_df.loc[mask, 'ff3_r_squared'] = r_squared
                    
                except Exception as e:
                    continue
        
        # Report statistics
        valid_alpha = self.master_df.dropna(subset=['ff3_alpha'])
        print(f"✅ FF3-Alpha 계산 완료:")
        print(f"   - 유효 데이터: {len(valid_alpha):,}개")
        print(f"   - 평균 Alpha: {valid_alpha['ff3_alpha'].mean():.4f}")
        print(f"   - Alpha 표준편차: {valid_alpha['ff3_alpha'].std():.4f}")
    
    def build_ff3_alpha_portfolio(self):
        """Build FF3-Alpha portfolio with July first available date annual rebalancing"""
        print("📈 FF3-Alpha 포트폴리오 구성...")
        
        if self.master_df is None or 'ff3_alpha' not in self.master_df.columns:
            print("❌ FF3-Alpha 데이터가 없습니다")
            return None
        
        # Get annual rebalancing dates - use first available date in July each year  
        unique_dates = sorted(self.master_df['매매년월일'].unique())
        all_july_dates = [date for date in unique_dates if date.month == 7]
        july_years = list(set([date.year for date in all_july_dates]))
        july_dates = []
        for year in sorted(july_years):
            year_july_dates = [date for date in all_july_dates if date.year == year]
            if year_july_dates:
                july_dates.append(min(year_july_dates))  # 7월 첫째 날 (가장 이른 날짜)
        
        # FF3-Alpha 전략 수정: 알파 임계값을 낮춰서 더 많은 포트폴리오 생성
        min_alpha = self.config.get('strategy_params', {}).get('ff3_alpha', {}).get('min_alpha', 0.0)  # 기본값을 0.0으로 완화
        
        portfolios = {'All': [], 'Normal': []}
        
        for rebalance_date in july_dates:
            # Get data for this July date
            date_data = self.master_df[self.master_df['매매년월일'] == rebalance_date].copy()
            
            # Filter stocks with alpha >= min_alpha
            valid_data = date_data.dropna(subset=['ff3_alpha'])
            significant_alpha = valid_data[valid_data['ff3_alpha'] >= min_alpha]
            
            # All firms portfolio - 포트폴리오 사이즈에 만족하지 못해도 가능한 종목으로 구성
            if len(significant_alpha) > 0:
                # 가능한 종목 수와 원하는 포트폴리오 사이즈 중 작은 값 선택
                actual_portfolio_size = min(len(significant_alpha), self.portfolio_size)
                top_stocks = significant_alpha.nlargest(actual_portfolio_size, 'ff3_alpha')
                
                portfolios['All'].append({
                    'date': rebalance_date,
                    'stocks': top_stocks['거래소코드'].tolist(),
                    'signals': top_stocks['ff3_alpha'].tolist(),
                    'actual_size': actual_portfolio_size
                })
            
            # Normal firms (non-default) portfolio
            normal_data = significant_alpha[significant_alpha.get('default', 0) == 0]
            if len(normal_data) > 0:
                # 가능한 종목 수와 원하는 포트폴리오 사이즈 중 작은 값 선택
                actual_normal_size = min(len(normal_data), self.portfolio_size)
                top_normal_stocks = normal_data.nlargest(actual_normal_size, 'ff3_alpha')
                
                portfolios['Normal'].append({
                    'date': rebalance_date,
                    'stocks': top_normal_stocks['거래소코드'].tolist(),
                    'signals': top_normal_stocks['ff3_alpha'].tolist(),
                    'actual_size': actual_normal_size
                })
        
        print(f"✅ FF3-Alpha 포트폴리오 구성 완료: All {len(portfolios['All'])}개년, Normal {len(portfolios['Normal'])}개년")
        print(f"   - 최소 알파 임계값: {min_alpha:.4f}")
        print(f"   - 목표 포트폴리오 사이즈: {self.portfolio_size}")
        
        # 실제 포트폴리오 사이즈 통계 출력
        if portfolios['All']:
            all_sizes = [p.get('actual_size', len(p['stocks'])) for p in portfolios['All']]
            print(f"   - All 포트폴리오 실제 사이즈: 평균 {np.mean(all_sizes):.1f}, 범위 {min(all_sizes)}-{max(all_sizes)}")
        
        if portfolios['Normal']:
            normal_sizes = [p.get('actual_size', len(p['stocks'])) for p in portfolios['Normal']]
            print(f"   - Normal 포트폴리오 실제 사이즈: 평균 {np.mean(normal_sizes):.1f}, 범위 {min(normal_sizes)}-{max(normal_sizes)}")
        
        return portfolios
    
    def backtest_strategy(self, portfolios, strategy_name):
        """Universal backtesting method for any strategy"""
        print(f"💰 {strategy_name} 백테스팅 실행...")
        
        if not portfolios:
            print(f"❌ {strategy_name} 포트폴리오가 없습니다")
            return None
        
        # Handle both dict (All/Normal) and list formats
        if isinstance(portfolios, dict):
            all_results = {}
            for universe, portfolio_list in portfolios.items():
                if portfolio_list:
                    results = self._backtest_portfolio_list(portfolio_list, f"{strategy_name}_{universe}")
                    all_results[universe] = results
            return all_results
        else:
            # Single portfolio list
            return self._backtest_portfolio_list(portfolios, strategy_name)
    
    def _backtest_portfolio_list(self, portfolio_list, strategy_name):
        """Backtest a single portfolio list optimized for monthly first-day data"""
        portfolio_values = [1.0]  # 초기 자본
        returns = []
        
        for i in range(len(portfolio_list) - 1):
            current_portfolio = portfolio_list[i]
            next_portfolio = portfolio_list[i + 1]
            
            # Calculate holding period return
            portfolio_return = 0
            valid_stocks = 0
            
            for stock_code in current_portfolio['stocks']:
                # Find current and next period prices (exact date matching)
                current_date = current_portfolio['date']
                next_date = next_portfolio['date']
                
                # Get prices from master dataframe (exact date matching)
                current_data = self.master_df[
                    (self.master_df['거래소코드'] == stock_code) & 
                    (self.master_df['매매년월일'] == current_date)
                ]
                
                next_data = self.master_df[
                    (self.master_df['거래소코드'] == stock_code) & 
                    (self.master_df['매매년월일'] == next_date)
                ]
                
                if len(current_data) > 0 and len(next_data) > 0:
                    current_price = current_data.iloc[0]['종가']
                    next_price = next_data.iloc[0]['종가']
                    
                    if current_price > 0 and next_price > 0:
                        stock_return = (next_price / current_price) - 1
                        portfolio_return += stock_return
                        valid_stocks += 1
            
            if valid_stocks > 0:
                portfolio_return = portfolio_return / valid_stocks  # Equal weighting
                returns.append(portfolio_return)
                new_value = portfolio_values[-1] * (1 + portfolio_return)
                portfolio_values.append(new_value)
            else:
                returns.append(0)
                portfolio_values.append(portfolio_values[-1])
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(returns, portfolio_values)
        
        print(f"📊 {strategy_name} 백테스팅 결과:")
        print(f"   - 총 수익률: {results['total_return']:.4f} ({results['total_return']*100:.2f}%)")
        print(f"   - 연간 수익률: {results['annual_return']:.4f} ({results['annual_return']*100:.2f}%)")
        print(f"   - 연간 변동성: {results['volatility']:.4f} ({results['volatility']*100:.2f}%)")
        print(f"   - 샤프 비율: {results['sharpe_ratio']:.4f}")
        print(f"   - 소티노 비율: {results['sortino_ratio']:.4f}")
        print(f"   - 정보 비율: {results['information_ratio']:.4f}")
        print(f"   - 최대 낙폭: {results['max_drawdown']:.4f} ({results['max_drawdown']*100:.2f}%)")
        
        return results
    
    def run_full_backtest(self):
        """Run complete backtesting for all strategies"""
        print("🚀 전체 백테스팅 실행...")
        
        # 1. Load master data
        self.load_master_data()
        
        if self.master_df is None:
            print("❌ 마스터 데이터 로딩 실패")
            return {}
        
        # 2. Calculate all factors
        print("\n📊 팩터 계산 단계...")
        
        try:
            self.calculate_12_1_momentum()
        except Exception as e:
            print(f"⚠️ 모멘텀 계산 실패: {e}")
            
        try:
            self.calculate_magic_formula()
        except Exception as e:
            print(f"⚠️ Magic Formula 계산 실패: {e}")
            
        try:
            self.calculate_fscore()
        except Exception as e:
            print(f"⚠️ F-Score 계산 실패: {e}")
            
        try:
            self.calculate_ff3_alpha()
        except Exception as e:
            print(f"⚠️ FF3-Alpha 계산 실패: {e}")
        
        # 3. Build portfolios and backtest
        print("\n🎯 포트폴리오 구성 및 백테스팅...")
        
        # Momentum strategy
        try:
            momentum_portfolios = self.build_momentum_portfolio()
            if momentum_portfolios:
                momentum_results = self.backtest_strategy(momentum_portfolios, 'Momentum')
                self.results['Momentum'] = momentum_results
            else:
                print("⚠️ 모멘텀 포트폴리오 구성 실패")
        except Exception as e:
            print(f"⚠️ 모멘텀 백테스팅 실패: {e}")
        
        # Magic Formula strategy
        try:
            magic_portfolios = self.build_magic_formula_portfolio()
            if magic_portfolios:
                magic_results = self.backtest_strategy(magic_portfolios, 'Magic_Formula')
                self.results['Magic_Formula'] = magic_results
            else:
                print("⚠️ Magic Formula 포트폴리오 구성 실패")
        except Exception as e:
            print(f"⚠️ Magic Formula 백테스팅 실패: {e}")
        
        # F-Score strategy
        try:
            fscore_portfolios = self.build_fscore_portfolio()
            if fscore_portfolios:
                fscore_results = self.backtest_strategy(fscore_portfolios, 'F_Score')
                self.results['F_Score'] = fscore_results
            else:
                print("⚠️ F-Score 포트폴리오 구성 실패")
        except Exception as e:
            print(f"⚠️ F-Score 백테스팅 실패: {e}")
        
        # FF3-Alpha strategy
        try:
            ff3_portfolios = self.build_ff3_alpha_portfolio()
            if ff3_portfolios:
                ff3_results = self.backtest_strategy(ff3_portfolios, 'FF3_Alpha')
                self.results['FF3_Alpha'] = ff3_results
            else:
                print("⚠️ FF3-Alpha 포트폴리오 구성 실패")
        except Exception as e:
            print(f"⚠️ FF3-Alpha 백테스팅 실패: {e}")
        
        print("✅ 전체 백테스팅 완료!")
        return self.results
    
    def save_results(self, output_dir='outputs/backtesting_v4'):
        """Save backtesting results with improved structure"""
        print(f"💾 결과 저장: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results summary
        if self.results:
            summary_data = []
            detailed_results = {}
            
            for strategy, results in self.results.items():
                if isinstance(results, dict) and 'All' in results:
                    # Handle multi-universe results (All/Normal)
                    for universe, universe_results in results.items():
                        if universe_results:
                            summary_data.append({
                                'Strategy': strategy,
                                'Universe': universe,
                                'Total_Return': universe_results['total_return'],
                                'Annual_Return': universe_results['annual_return'],
                                'Volatility': universe_results['volatility'],
                                'Sharpe_Ratio': universe_results['sharpe_ratio'],
                                'Sortino_Ratio': universe_results['sortino_ratio'],
                                'Information_Ratio': universe_results['information_ratio'],
                                'Max_Drawdown': universe_results['max_drawdown'],
                                'N_Periods': universe_results.get('n_months', len(universe_results.get('monthly_returns', [])))
                            })
                            detailed_results[f"{strategy}_{universe}"] = universe_results
                else:
                    # Handle single universe results
                    if results:
                        summary_data.append({
                            'Strategy': strategy,
                            'Universe': 'All',
                            'Total_Return': results['total_return'],
                            'Annual_Return': results['annual_return'],
                            'Volatility': results['volatility'],
                            'Sharpe_Ratio': results['sharpe_ratio'],
                            'Sortino_Ratio': results['sortino_ratio'],
                            'Information_Ratio': results['information_ratio'],
                            'Max_Drawdown': results['max_drawdown'],
                            'N_Periods': results.get('n_months', len(results.get('monthly_returns', [])))
                        })
                        detailed_results[strategy] = results
            
            # Save summary
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
                print("✅ 성과 요약 저장 완료")
                
                # Display summary
                print("\n📊 백테스팅 성과 요약:")
                print(summary_df.to_string(index=False, float_format='%.4f'))
            
            # Save detailed results
            import yaml
            with open(os.path.join(output_dir, 'detailed_results.yaml'), 'w', encoding='utf-8') as f:
                yaml.dump(detailed_results, f, default_flow_style=False, allow_unicode=True)
            print("✅ 상세 결과 저장 완료")
        
        print("💾 결과 저장 완료!")


def main():
    """Main execution function"""
    print("🚀 Factor Backtesting v4.0 시작")
    
    # Initialize backtesting framework
    backtester = FactorBacktestingV4()
    
    # Run complete backtesting
    results = backtester.run_full_backtest()
    
    # Save results
    backtester.save_results()
    
    print("🎉 Factor Backtesting v4.0 완료!")


if __name__ == "__main__":
    main()