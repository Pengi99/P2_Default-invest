"""
Factor Investing Backtesting Framework
8개 팩터 전략 백테스트 및 성과 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import glob
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (OS별 자동 감지)
import platform

if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':  # Windows
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux and others
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-v0_8')

class FactorBacktester:
    """팩터 투자 백테스트 클래스"""
    
    def __init__(self, data_path=None):
        # 프로젝트 루트 디렉토리 설정
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 데이터 경로 설정
        if data_path is None:
            self.data_path = os.path.join(base_dir, 'data', 'raw')
        else:
            self.data_path = data_path
        
        # 재무 데이터 경로 설정
        self.fs_path = os.path.join(base_dir, 'data', 'processed', 'FS.csv')
        
        # 출력 디렉토리 설정
        self.output_dir = os.path.join(base_dir, 'outputs', 'backtesting')
        os.makedirs(self.output_dir, exist_ok=True)
        self.prices_df = None
        self.fs_df = None
        self.df = None
        self.predictions_df = None
        self.factor_returns = {}
        self.performance_stats = {}
        
    def load_data(self):
        """1단계: 데이터 로딩"""
        print("📊 데이터 로딩 중...")
        
        # 가격 데이터 로딩 (2012~2023)
        price_files = sorted(glob.glob(os.path.join(self.data_path, '20*.csv')))
        price_dfs = []
        
        if not price_files:
            raise FileNotFoundError(f"No price files found in directory: {self.data_path}")
        
        for file in price_files:
            df_temp = pd.read_csv(file, encoding='utf-8')
            
            # 컬럼 rename (실제 컬럼명에 맞춰 수정)
            df_temp = df_temp.rename(columns={
                '매매년월일': 'date',
                '종가(원)': 'price',
                '상장주식수(주)': 'shares_out'
            })
            
            # 수익률 계산 (일간 수익률)
            df_temp = df_temp.sort_values(['회사명', '거래소코드', 'date'])
            df_temp['ret'] = df_temp.groupby(['회사명', '거래소코드'])['price'].pct_change()
            
            # ticker_key 생성
            df_temp['ticker_key'] = df_temp['거래소코드'].astype(str)
            
            # 날짜 변환 (YYYY/MM/DD 형식)
            df_temp['date'] = pd.to_datetime(df_temp['date'], format='%Y/%m/%d', errors='coerce')
            
            # 회계년도 정보 추가
            df_temp['year'] = df_temp['date'].dt.year
            
            price_dfs.append(df_temp)
        
        self.prices_df = pd.concat(price_dfs, ignore_index=True)
        print(f"✅ 가격 데이터 로딩 완료: {len(self.prices_df):,}행")
        
        # 재무 데이터 로딩 (processed 폴더에서)
        self.fs_df = pd.read_csv(self.fs_path, encoding='utf-8-sig')
        print(f"✅ 재무 데이터 로딩 완료: {len(self.fs_df):,}행")
        
        return self
    
    def preprocess(self):
        """2단계: 데이터 전처리"""
        print("🔄 데이터 전처리 중...")
        
        # 회계년도를 기준으로 재무 데이터 매핑
        # 회계년도 컬럼명 확인 및 정리
        if '회계년도' in self.fs_df.columns:
            # YYYY/MM 형식을 년도로 변환
            self.fs_df['year'] = self.fs_df['회계년도']
        
        # ticker_key를 fs_df에도 추가
        self.fs_df['ticker_key'] = self.fs_df['거래소코드'].astype(str)
        
        # 가격 데이터에 연말 기준으로 재무 데이터 병합
        # 각 년도의 12월 데이터만 사용 (연간 팩터 계산용)
        monthly_prices = self.prices_df.copy()
        monthly_prices['month'] = monthly_prices['date'].dt.month
        
        # 각 월 마지막 거래일 데이터만 추출
        monthly_prices = monthly_prices.loc[monthly_prices.groupby(['ticker_key', 'year', 'month'])['date'].idxmax()]
        
        # 재무 데이터와 병합
        self.df = monthly_prices.merge(
            self.fs_df, 
            on=['ticker_key', 'year'], 
            how='left'
        )
        
        # 시가총액 계산
        self.df['mktcap'] = self.df['price'] * self.df['shares_out']

        # 무한값과 결측치 제거
        self.df = self.df.replace([np.inf, -np.inf], np.nan).dropna()

        print(f"✅ 전처리 완료: {len(self.df):,}행")
        return self
    
    def _compute_annual_returns(self):
        """연간 리밸런싱 데이터 및 수익률 계산 (4월 1일 기준)"""
        # 4월 1일 기준 리밸런싱 데이터 추출
        rebalance_dates = []
        
        # 각 년도의 4월 1일 또는 가장 가까운 거래일 찾기
        years = sorted(self.df['date'].dt.year.unique())
        for year in years:
            # 4월 1일 또는 그 이후 가장 가까운 거래일
            target_date = pd.Timestamp(f'{year}-04-01')
            year_data = self.df[self.df['date'].dt.year == year]
            april_data = year_data[year_data['date'] >= target_date]
            
            if not april_data.empty:
                # 4월 1일 이후 가장 빠른 날짜
                rebalance_date = april_data['date'].min()
                rebalance_dates.append(rebalance_date)
        
        # 리밸런싱 날짜에 해당하는 데이터만 추출
        rebalance_data = self.df[self.df['date'].isin(rebalance_dates)].copy()
        rebalance_data = rebalance_data.sort_values(['ticker_key', 'date'])
        
        # 연간 수익률 계산 (리밸런싱 기간 간 수익률)
        rebalance_data['annual_ret'] = rebalance_data.groupby('ticker_key')['price'].pct_change()
        
        # 연간 수익률을 원본 데이터에 병합
        annual_ret_df = rebalance_data[['ticker_key', 'date', 'annual_ret']]
        self.df = self.df.merge(annual_ret_df, on=['ticker_key', 'date'], how='left')
        self.df['annual_ret'] = self.df['annual_ret'].fillna(0)
        
        # 리밸런싱 날짜 표시
        self.df['is_rebalance_date'] = self.df['date'].isin(rebalance_dates)
    
    def _compute_magic_formula(self):
        """매직 포뮬라 - 랭크 기반 점수 사용"""
        # 영업이익 수익률 = 영업이익 / 기업가치
        self.df['operating_income'] = self.df['영업이익']
        self.df['enterprise_value'] = self.df['기업가치']
        self.df['earnings_yield'] = self.df['operating_income'] / self.df['enterprise_value']
        
        # ROIC = 경영자본영업이익률 / 100
        self.df['roic'] = self.df['경영자본영업이익률'] / 100
        
        # 날짜별로 랭크 계산
        magic_scores = []
        for date in self.df['date'].unique():
            date_mask = self.df['date'] == date
            date_df = self.df.loc[date_mask].copy()
            
            if len(date_df) < 10:
                continue
            
            # 유효한 데이터만 사용
            valid_mask = date_df['earnings_yield'].notna() & date_df['roic'].notna()
            valid_df = date_df[valid_mask].copy()
            
            if len(valid_df) < 5:
                continue
            
            # 랭크 계산 (높을수록 좋음)
            valid_df['ey_rank'] = valid_df['earnings_yield'].rank(ascending=False)
            valid_df['roic_rank'] = valid_df['roic'].rank(ascending=False)
            
            # 랭크 합계 (낮을수록 좋음)
            valid_df['magic_rank_sum'] = valid_df['ey_rank'] + valid_df['roic_rank']
            valid_df['magic'] = -valid_df['magic_rank_sum']  # 낮은 랭크 합이 좋음
            
            magic_scores.append(valid_df[['ticker_key', 'date', 'magic']])
        
        if magic_scores:
            magic_df = pd.concat(magic_scores)
            self.df = self.df.merge(magic_df, on=['ticker_key', 'date'], how='left')
        else:
            self.df['magic'] = 0
    
    def _compute_momentum(self):
        """모멘텀 (12-1) - 연간 수익률 사용"""
        # 리밸런싱 날짜에만 모멘텀 계산
        self.df['mom'] = self.df.groupby('ticker_key')['annual_ret'].apply(
            lambda x: x.shift(1).rolling(3).sum()  # 3년 모멘텀
        ).reset_index(0, drop=True)
    
    def _compute_low_volatility(self):
        """낮은 변동성 - 연간 수익률 변동성 사용"""
        self.df['vol3y'] = self.df.groupby('ticker_key')['annual_ret'].apply(
            lambda x: x.rolling(3).std()  # 3년 변동성
        ).reset_index(0, drop=True)
        self.df['lovol'] = -self.df['vol3y']
    
    def compute_signals(self):
        """3단계: 8개 팩터 시그널 계산"""
        print("🎯 팩터 시그널 계산 중...")
        
        # 데이터 정렬
        self.df = self.df.sort_values(['ticker_key', 'date']).reset_index(drop=True)
        
        # 먼저 연간 리밸런싱 데이터 및 수익률 계산
        self._compute_annual_returns()
        
        # 1. Magic Formula - Rank-based scoring
        self._compute_magic_formula()
        
        # 2. EV/EBITDA - 실제 컬럼에 이미 계산되어 있음
        self.df['ev_ebitda'] = self.df['EV_EBITDA배수']
        self.df['ev_ebitda_signal'] = -self.df['ev_ebitda']  # 낮을수록 좋음
        
        # 3. B/M (Book-to-Market) - 총자본/시가총액
        self.df['book_value'] = self.df['총자본']
        self.df['market_value'] = self.df['mktcap']
        self.df['bm'] = self.df['book_value'] / self.df['market_value']
        
        # 4. 12-1 Momentum (using annual returns)
        self._compute_momentum()
        
        # 5. Piotroski F-Score (updated with accruals)
        self._compute_fscore()
        
        # 6. QMJ (Quality Minus Junk) - improved
        self._compute_qmj()
        
        # 7. Low Volatility (using annual returns)
        self._compute_low_volatility()
        
        # 8. SMB & HML (Fama-French 3-Factor) - separate factors
        self._compute_ff_factors()
        
        print("✅ 팩터 시그널 계산 완료")
        return self
    
    def _compute_fscore(self):
        """Piotroski F-Score 계산 - Updated with accruals signal"""
        # 기본 수익성 지표 - processed FS.csv 컬럼명 사용
        self.df['f_roa'] = (self.df['총자산수익률'] > 0).astype(int)
        self.df['f_cfo'] = (self.df['영업현금흐름'] > 0).astype(int)
        
        # Accruals signal: (영업현금흐름 / 총자산) > (영업이익 / 총자산)
        cfo_ta = self.df['영업현금흐름'] / self.df['총자산']
        oi_ta = self.df['영업이익'] / self.df['총자산']
        self.df['f_accrual'] = (cfo_ta > oi_ta).astype(int)
        
        # 레버리지, 유동성, 자금조달 지표
        self.df['f_debt'] = (self.df.groupby('ticker_key')['부채비율'].pct_change() < 0).astype(int)
        self.df['f_liquid'] = (self.df.groupby('ticker_key')['유동비율'].pct_change() > 0).astype(int)
        
        # 신주발행 여부 - Updated: 발행주식총수 증가율 <= 0
        shares_change = self.df.groupby('ticker_key')['발행주식총수'].pct_change()
        self.df['f_shares'] = (shares_change <= 0).astype(int)
        
        # 운영 효율성 지표
        self.df['f_margin'] = (self.df.groupby('ticker_key')['매출액총이익률'].pct_change() > 0).astype(int)
        self.df['f_turn'] = (self.df.groupby('ticker_key')['총자본회전률'].pct_change() > 0).astype(int)
        
        # ROA 개선 지표
        self.df['f_roa_chg'] = (self.df.groupby('ticker_key')['총자산수익률'].pct_change() > 0).astype(int)
        
        # F-Score 합계 (9개 지표)
        fscore_cols = ['f_roa', 'f_cfo', 'f_accrual', 'f_debt', 'f_liquid', 'f_shares', 'f_margin', 'f_turn', 'f_roa_chg']
        self.df['fscore'] = self.df[fscore_cols].sum(axis=1)
    
    def _compute_qmj(self):
        """QMJ (Quality Minus Junk) 계산 - Improved with distinct measures"""
        # 수익성 지표들
        profitability_cols = ['자기자본순이익률', '총자산수익률']
        
        # 안정성 지표들 - 추가: 배당비율, 장기 레버리지
        safety_cols = ['부채비율', 'vol12m']
        
        # 성장성 지표들 - 중복 제거, 5년 매출 CAGR 대신 매출액증가율 사용
        growth_cols = ['매출액증가율']  # 매출액총이익률 제거
        
        # 각 그룹별로 Z-score 표준화
        quality_scores = []
        
        for date in self.df['date'].unique():
            date_mask = self.df['date'] == date
            date_df = self.df.loc[date_mask].copy()
            
            if len(date_df) < 10:
                continue
                
            # 수익성 (높을수록 좋음)
            prof_score = 0
            for col in profitability_cols:
                if col in date_df.columns and date_df[col].notna().sum() > 5:
                    prof_score += stats.zscore(date_df[col].fillna(date_df[col].median()))
            
            # 안정성 (부채비율, 변동성은 낮을수록 좋음)
            safety_score = 0
            for col in safety_cols:
                if col in date_df.columns and date_df[col].notna().sum() > 5:
                    safety_score -= stats.zscore(date_df[col].fillna(date_df[col].median()))
            
            # 배당비율 추가 (낮을수록 좋음)
            if '배당비율' in date_df.columns and date_df['배당비율'].notna().sum() > 5:
                safety_score -= stats.zscore(date_df['배당비율'].fillna(date_df['배당비율'].median()))
            
            # 성장성 (전년 대비 증가율)
            growth_score = 0
            for col in growth_cols:
                if col in date_df.columns and date_df[col].notna().sum() > 5:
                    growth_score += stats.zscore(date_df[col].fillna(date_df[col].median()))
            
            date_df['qmj'] = (prof_score + safety_score + growth_score) / 3
            quality_scores.append(date_df[['ticker_key', 'date', 'qmj']])
        
        if quality_scores:
            qmj_df = pd.concat(quality_scores)
            self.df = self.df.merge(qmj_df, on=['ticker_key', 'date'], how='left')
        else:
            self.df['qmj'] = 0
    
    def _compute_ff_factors(self):
        """Fama-French SMB & HML 팩터 계산 - 연간 리밸런싱"""
        # 연간 리밸런싱 날짜별로 size와 value 기준 포트폴리오 구성
        ff_returns = []
        
        # 리밸런싱 날짜만 사용
        rebalance_data = self.df[self.df['is_rebalance_date'] == True]
        annual_dates = rebalance_data['date'].unique()
        
        for rebal_date in annual_dates:
            annual_df = rebalance_data[rebalance_data['date'] == rebal_date].copy()
            
            if len(annual_df) < 20:
                continue
            
            # Size와 B/M 기준으로 정렬 (annual_ret 사용)
            annual_df = annual_df.dropna(subset=['mktcap', 'bm', 'annual_ret'])
            
            if len(annual_df) < 6:
                continue
                
            # 시가총액 중위수 기준 분할
            size_median = annual_df['mktcap'].median()
            
            # B/M 30%, 70% 분위수 기준 분할
            bm_30 = annual_df['bm'].quantile(0.3)
            bm_70 = annual_df['bm'].quantile(0.7)
            
            # 6개 포트폴리오 구성
            portfolios = {
                'SL': annual_df[(annual_df['mktcap'] <= size_median) & (annual_df['bm'] <= bm_30)],
                'SM': annual_df[(annual_df['mktcap'] <= size_median) & (annual_df['bm'] > bm_30) & (annual_df['bm'] <= bm_70)],
                'SH': annual_df[(annual_df['mktcap'] <= size_median) & (annual_df['bm'] > bm_70)],
                'BL': annual_df[(annual_df['mktcap'] > size_median) & (annual_df['bm'] <= bm_30)],
                'BM': annual_df[(annual_df['mktcap'] > size_median) & (annual_df['bm'] > bm_30) & (annual_df['bm'] <= bm_70)],
                'BH': annual_df[(annual_df['mktcap'] > size_median) & (annual_df['bm'] > bm_70)]
            }
            
            # 각 포트폴리오 수익률 계산 (동일가중)
            port_returns = {}
            for name, port in portfolios.items():
                if len(port) > 0:
                    port_returns[name] = port['annual_ret'].mean()
                else:
                    port_returns[name] = 0
            
            # SMB와 HML 계산
            if all(name in port_returns for name in ['SL', 'SM', 'SH', 'BL', 'BM', 'BH']):
                smb = (port_returns['SL'] + port_returns['SM'] + port_returns['SH']) / 3 - \
                      (port_returns['BL'] + port_returns['BM'] + port_returns['BH']) / 3
                
                hml = (port_returns['SH'] + port_returns['BH']) / 2 - \
                      (port_returns['SL'] + port_returns['BL']) / 2
                
                ff_returns.append({
                    'date': rebal_date,
                    'smb': smb,
                    'hml': hml,
                    'smb_hml': smb * hml  # SMB×HML 교차항 (옵션)
                })
        
        if ff_returns:
            ff_df = pd.DataFrame(ff_returns)
            
            # 원본 데이터에 병합 - SMB, HML, SMB×HML 모두 포함
            self.df = self.df.merge(ff_df[['date', 'smb', 'hml', 'smb_hml']], on='date', how='left')
        else:
            self.df['smb'] = 0
            self.df['hml'] = 0
            self.df['smb_hml'] = 0
    
    def get_factor_returns(self, signal_col, universe_mask=None, top_pct=0.3):
        """4단계: 팩터 수익률 계산 - 연간 리밸런싱 (4월 1일)"""
        if universe_mask is None:
            universe_mask = self.df.index
        
        # 리밸런싱 날짜에만 팩터 계산
        tmp = self.df.loc[universe_mask].copy()
        tmp = tmp[tmp['is_rebalance_date'] == True]  # 리밸런싱 날짜만
        tmp = tmp.dropna(subset=[signal_col, 'annual_ret'])
        
        factor_returns = []
        portfolio_holdings = []  # 턴오버 계산을 위한 보유 종목
        
        # 년도별로 그룹화
        tmp['year'] = tmp['date'].dt.year
        
        for year, grp in tmp.groupby('year'):
            if len(grp) < 10:
                continue
                
            n = max(1, int(len(grp) * top_pct))
            
            # Long: 상위 30%
            long_stocks = grp.nlargest(n, signal_col)
            long_ret = long_stocks['annual_ret'].mean()
            
            # Short: 하위 30%  
            short_stocks = grp.nsmallest(n, signal_col)
            short_ret = short_stocks['annual_ret'].mean()
            
            # 대표 날짜 (그 년도의 4월 1일)
            rebalance_date = grp['date'].iloc[0]
            
            factor_returns.append({
                'date': rebalance_date,
                'long_ret': long_ret,
                'short_ret': short_ret,
                'factor_ret': long_ret - short_ret,
                'n_stocks': len(grp)
            })
            
            # 포트폴리오 보유 종목 저장
            portfolio_holdings.append({
                'date': rebalance_date,
                'long_tickers': set(long_stocks['ticker_key']),
                'short_tickers': set(short_stocks['ticker_key'])
            })
        
        factor_df = pd.DataFrame(factor_returns).set_index('date')
        
        # 턴오버 계산 (연간 리밸런싱)
        if len(portfolio_holdings) > 1:
            turnovers = []
            for i in range(1, len(portfolio_holdings)):
                prev_long = portfolio_holdings[i-1]['long_tickers']
                curr_long = portfolio_holdings[i]['long_tickers']
                prev_short = portfolio_holdings[i-1]['short_tickers']
                curr_short = portfolio_holdings[i]['short_tickers']
                
                # 전체 보유 종목 대비 변경된 비율
                total_prev = len(prev_long) + len(prev_short)
                changed = len(prev_long.symmetric_difference(curr_long)) + len(prev_short.symmetric_difference(curr_short))
                turnover = changed / total_prev if total_prev > 0 else 0
                
                turnovers.append(turnover)
            
            factor_df['turnover'] = [np.nan] + turnovers
        else:
            factor_df['turnover'] = np.nan
        
        return factor_df
    
    def backtest(self):
        """5단계: 백테스트 실행"""
        print("📈 백테스트 실행 중...")
        
        # 2013년 이후 데이터만 사용 (2012년은 warm-up)
        backtest_mask = self.df['date'] >= '2013-01-01'
        
        factor_signals = {
            'Magic Formula': 'magic',
            'EV/EBITDA': 'ev_ebitda_signal', 
            'Book-to-Market': 'bm',
            'Momentum (12-1)': 'mom',
            'Piotroski F-Score': 'fscore',
            'Quality (QMJ)': 'qmj',
            'Low Volatility': 'lovol',
            'SMB (Size)': 'smb',
            'HML (Value)': 'hml',
            'SMB×HML': 'smb_hml'
        }
        
        for strategy_name, signal_col in factor_signals.items():
            print(f"  🔄 {strategy_name} 계산 중...")
            
            # 전체 유니버스
            full_universe = backtest_mask
            full_returns = self.get_factor_returns(signal_col, full_universe)
            
            # 우량기업 유니버스 (default == 0)
            quality_universe = backtest_mask & (self.df['default'] == 0)
            quality_returns = self.get_factor_returns(signal_col, quality_universe)
            
            self.factor_returns[f"{strategy_name}_Full"] = full_returns
            self.factor_returns[f"{strategy_name}_Quality"] = quality_returns
        
        print("✅ 백테스트 완료")
        return self
    
    def calc_performance_stats(self):
        """6단계: 성과지표 계산"""
        print("📊 성과지표 계산 중...")
        
        for strategy_name, returns_df in self.factor_returns.items():
            if len(returns_df) == 0:
                continue
                
            ret_series = returns_df['factor_ret'].dropna()
            
            if len(ret_series) < 3:  # 최소 3년 데이터 필요
                continue
            
            # 기본 통계 - 연간 리밸런싱 반영
            # ann_ret = ret_series.mean() * 12  # Removed: arithmetic annualized return
            ann_vol = ret_series.std()  # 연간 데이터이므로 루트 없이 사용
            
            # CAGR (Compound Annual Growth Rate) calculation
            # CAGR represents compound growth rather than arithmetic average
            cum_ret = (1 + ret_series).cumprod()  # Portfolio value series
            start_value = cum_ret.iloc[0]
            end_value = cum_ret.iloc[-1]
            n_periods = len(cum_ret)  # number of data points (years)
            periods_per_year = 1  # annual data
            n_years = n_periods / periods_per_year
            
            if n_years > 0 and start_value > 0:
                cagr = (end_value / start_value) ** (1 / n_years) - 1
            else:
                cagr = 0
            
            # Updated metrics using CAGR instead of ann_ret
            sharpe = cagr / ann_vol if ann_vol > 0 else 0
            
            # 누적수익률과 최대낙폭
            running_max = cum_ret.cummax()
            drawdown = (cum_ret / running_max - 1)
            max_dd = drawdown.min()
            
            # 정보비율 - 벤치마크 대비 계산 (시장 평균 수익률 가정: 10% 연간)
            benchmark_ret = 0.10  # 연간 10%
            excess_ret = ret_series - benchmark_ret
            tracking_error = excess_ret.std()  # 연간 데이터
            info_ratio = excess_ret.mean() / tracking_error if tracking_error > 0 else 0
            
            # 칼마비율 - using CAGR
            calmar = cagr / abs(max_dd) if max_dd != 0 else 0
            
            # M² 측도 (시장 변동성 대비 조정수익률) - using CAGR
            market_vol = 0.15  # 가정: 시장 연변동성 15%
            m2_measure = cagr * (market_vol / ann_vol) if ann_vol > 0 else 0
            
            # 최대낙폭 지속기간
            dd_periods = []
            in_drawdown = False
            dd_start = None
            
            for i, dd_val in enumerate(drawdown):
                if dd_val < -0.001 and not in_drawdown:  # 낙폭 시작
                    in_drawdown = True
                    dd_start = i
                elif dd_val >= -0.001 and in_drawdown:  # 낙폭 회복
                    in_drawdown = False
                    if dd_start is not None:
                        dd_periods.append(i - dd_start)
            
            dd_duration = max(dd_periods) if dd_periods else 0
            
            # VaR & CVaR (95% 신뢰수준)
            var_95 = np.percentile(ret_series, 5)
            cvar_95 = ret_series[ret_series <= var_95].mean()
            
            # 실제 턴오버율 계산
            if 'turnover' in returns_df.columns:
                turnover = returns_df['turnover'].mean()
            else:
                turnover = 0.20  # 기본값
            
            # 상승/하락 캡처 비율 - 벤치마크 대비 계산 (연간 기준)
            benchmark_returns = pd.Series([benchmark_ret] * len(ret_series), index=ret_series.index)
            up_periods = ret_series > benchmark_ret  # 벤치마크 대비 상승
            down_periods = ret_series <= benchmark_ret  # 벤치마크 대비 하락
            
            if up_periods.sum() > 0:
                up_capture = ret_series[up_periods].mean() / benchmark_returns[up_periods].mean()
            else:
                up_capture = 1.0
                
            if down_periods.sum() > 0:
                down_capture = ret_series[down_periods].mean() / benchmark_returns[down_periods].mean()
            else:
                down_capture = 1.0
            
            self.performance_stats[strategy_name] = {
                'CAGR': cagr,  # Updated from AnnRet to CAGR
                'AnnVol': ann_vol, 
                'Sharpe': sharpe,
                'MaxDD': max_dd,
                'IR': info_ratio,
                'Calmar': calmar,
                'M2': m2_measure,
                'DD_Duration': dd_duration,
                'Turnover': turnover,
                'UpCapture': up_capture,
                'DownCapture': down_capture,
                'VaR95': var_95,
                'CVaR95': cvar_95
            }
        
        print("✅ 성과지표 계산 완료")
        return self
    
    def plot_results(self):
        """7단계: 시각화"""
        print("📈 시각화 생성 중...")
        
        # 1) 누적수익률 차트
        self._plot_cumulative_returns()
        
        # 2) 성과지표 막대차트
        self._plot_performance_bars()
        
        # 3) 히트맵
        self._plot_heatmap()
        
        # 4) 월간 수익률 박스플롯
        self._plot_monthly_boxplot()
        
        # 5) 낙폭 곡선
        self._plot_drawdown_curves()
        
        print("✅ 시각화 완료")
        return self
    
    def _plot_cumulative_returns(self):
        """누적수익률 차트"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('전체 유니버스', '우량기업 유니버스'),
            vertical_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (strategy_name, returns_df) in enumerate(self.factor_returns.items()):
            if len(returns_df) == 0:
                continue
                
            cum_ret = (1 + returns_df['factor_ret']).cumprod()
            
            row = 1 if '_Full' in strategy_name else 2
            name = strategy_name.replace('_Full', '').replace('_Quality', '')
            
            fig.add_trace(
                go.Scatter(
                    x=cum_ret.index,
                    y=cum_ret.values,
                    name=name,
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=(row == 1)
                ),
                row=row, col=1
            )
        
        fig.update_layout(
            title="팩터 전략별 누적수익률",
            font=dict(family='AppleGothic'),  # 한글 폰트 설정
            height=800,
            hovermode='x unified'
        )
        
        fig.show()
        
    def _plot_performance_bars(self):
        """성과지표 막대차트"""
        if not self.performance_stats:
            return
            
        stats_df = pd.DataFrame(self.performance_stats).T
        
        metrics = ['CAGR', 'Sharpe', 'Calmar', 'MaxDD']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if metric in stats_df.columns:
                ax = axes[i]
                
                # Full vs Quality 분리
                full_data = stats_df[stats_df.index.str.contains('_Full')][metric]
                quality_data = stats_df[stats_df.index.str.contains('_Quality')][metric]
                
                x_labels = [name.replace('_Full', '') for name in full_data.index]
                
                x = np.arange(len(x_labels))
                width = 0.35
                
                ax.bar(x - width/2, full_data.values, width, label='전체', alpha=0.8)
                ax.bar(x + width/2, quality_data.values, width, label='우량기업', alpha=0.8)
                
                ax.set_title(f'{metric}')
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'factor_performance_bars.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_heatmap(self):
        """성과지표 히트맵"""
        if not self.performance_stats:
            return
            
        stats_df = pd.DataFrame(self.performance_stats).T
        
        # 숫자형 컬럼만 선택
        numeric_cols = ['CAGR', 'AnnVol', 'Sharpe', 'MaxDD', 'IR', 'Calmar']
        heatmap_data = stats_df[numeric_cols].fillna(0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0,
            cbar_kws={'label': '값'}
        )
        plt.title('팩터 전략별 성과지표 히트맵')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'factor_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_monthly_boxplot(self):
        """월간 수익률 박스플롯"""
        monthly_data = []
        
        for strategy_name, returns_df in self.factor_returns.items():
            if len(returns_df) == 0:
                continue
                
            for date, row in returns_df.iterrows():
                monthly_data.append({
                    'Strategy': strategy_name.replace('_Full', '').replace('_Quality', ''),
                    'Universe': 'Full' if '_Full' in strategy_name else 'Quality',
                    'Return': row['factor_ret'],
                    'Date': date
                })
        
        if not monthly_data:
            return
            
        monthly_df = pd.DataFrame(monthly_data)
        
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=monthly_df, x='Strategy', y='Return', hue='Universe')
        plt.title('팩터별 월간 수익률 분포')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'monthly_returns_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_drawdown_curves(self):
        """낙폭 곡선"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        colors = plt.cm.Set1(np.linspace(0, 1, 8))
        
        for i, (strategy_name, returns_df) in enumerate(self.factor_returns.items()):
            if len(returns_df) == 0:
                continue
                
            cum_ret = (1 + returns_df['factor_ret']).cumprod()
            drawdown = (cum_ret / cum_ret.cummax() - 1) * 100
            
            ax = ax1 if '_Full' in strategy_name else ax2
            name = strategy_name.replace('_Full', '').replace('_Quality', '')
            
            ax.fill_between(
                drawdown.index, 
                drawdown.values, 
                0, 
                alpha=0.6,
                color=colors[i % len(colors)],
                label=name
            )
        
        ax1.set_title('Drawdown Curves - 전체 유니버스')
        ax1.set_ylabel('Drawdown (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Drawdown Curves - 우량기업 유니버스')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'drawdown_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """결과 저장"""
        print("💾 결과 저장 중...")
        
        # 성과 통계 저장
        if self.performance_stats:
            stats_df = pd.DataFrame(self.performance_stats).T
            stats_df.to_csv(os.path.join(self.output_dir, 'factor_performance_stats.csv'), encoding='utf-8-sig')
            print("✅ 성과통계 저장: factor_performance_stats.csv")
        
        # 팩터 수익률 저장
        for strategy_name, returns_df in self.factor_returns.items():
            if len(returns_df) > 0:
                # 파일 이름에 사용 불가능 문자가 있을 경우 대체
                safe_name = strategy_name.replace('/', '_')
                filename = os.path.join(self.output_dir, f"factor_returns_{safe_name}.csv")
                returns_df.to_csv(filename, encoding='utf-8-sig')
        
        print("✅ 팩터 수익률 저장 완료")
        
        # 요약 출력
        print("\n" + "="*60)
        print("📊 FACTOR BACKTESTING SUMMARY")
        print("="*60)
        
        if self.performance_stats:
            for strategy, stats in self.performance_stats.items():
                print(f"\n🎯 {strategy}")
                print(f"   CAGR: {stats['CAGR']:.2%}")
                print(f"   샤프비율: {stats['Sharpe']:.3f}")
                print(f"   최대낙폭: {stats['MaxDD']:.2%}")
                print(f"   칼마비율: {stats['Calmar']:.3f}")
        
        return self

def main():
    """메인 실행 함수"""
    print("🚀 Factor Investing Backtesting 시작")
    print("="*60)
    
    # 백테스터 초기화 및 실행
    backtester = FactorBacktester()
    
    backtester.load_data() \
              .preprocess() \
              .compute_signals() \
              .backtest() \
              .calc_performance_stats() \
              .plot_results() \
              .save_results()
    
    print("\n🎉 백테스트 완료!")
    return backtester

if __name__ == "__main__":
    # 실행
    results = main()