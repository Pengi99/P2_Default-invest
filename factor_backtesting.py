"""
Factor Investing Backtesting Framework - Full Rewrite
11개 팩터 전략 백테스트 및 성과 분석 (Long-Only Top-10 Equal-Weight)
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
import argparse
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
    """팩터 투자 백테스트 클래스 - Long-Only Top-10 Equal-Weight"""
    
    def __init__(self, data_path=None, output_dir=None):
        # 단순한 경로 설정 - 현재 디렉토리 기준
        self.data_path = data_path or 'data/processed'
        self.df = None
        self.factor_returns = {}
        self.performance_stats = {}
        
        # 출력 디렉토리 설정
        self.output_dir = output_dir or 'outputs/backtesting'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """1단계: 데이터 로딩 및 병합"""
        print("📊 데이터 로딩 중...")
        
        # 1) 재무제표 원본 (FS2.csv) - data/processed에서 찾기
        fs_path = os.path.join(self.data_path, 'FS2.csv')
        if not os.path.exists(fs_path):
            # 대안 경로들 시도
            alternative_paths = ['data/processed/FS2.csv', 'data/FS2.csv', 'FS2.csv']
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    fs_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"FS2.csv 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        
        df_fs = pd.read_csv(fs_path, encoding='utf-8-sig')
        print(f"✅ 재무제표 데이터 로딩 완료: {len(df_fs):,}행")
        
        # 2) 연도별 주가·거래량·발행주식수·주당배당금 (2012.csv ~ 2023.csv)
        price_files = []
        for data_dir in ['data/raw', 'data', '.']:
            pattern = os.path.join(data_dir, '20*.csv')
            found_files = sorted(glob.glob(pattern))
            if found_files:
                price_files = found_files
                break
        
        if not price_files:
            raise FileNotFoundError("연도별 주가 데이터 파일들(20XX.csv)을 찾을 수 없습니다.")
        
        price_dfs = []
        for f in price_files:
            df_temp = pd.read_csv(f, encoding='utf-8-sig')
            price_dfs.append(df_temp)
        
        df_price = pd.concat(price_dfs, ignore_index=True)
        print(f"✅ 주가 데이터 로딩 완료: {len(df_price):,}행")
        
        # 3) 시가총액 (시가총액.csv)
        mkt_path = None
        for data_dir in ['data/processed', 'data/raw', 'data', '.']:
            test_path = os.path.join(data_dir, '시가총액.csv')
            if os.path.exists(test_path):
                mkt_path = test_path
                break
        
        if not mkt_path:
            raise FileNotFoundError("시가총액.csv 파일을 찾을 수 없습니다.")
        
        df_mkt = pd.read_csv(mkt_path, encoding='utf-8-sig')
        print(f"✅ 시가총액 데이터 로딩 완료: {len(df_mkt):,}행")
        
        # 4) 병합 (거래소코드 + 연도)
        self.df = (df_fs
                   .merge(df_price, how='left', on=['거래소코드', '연도'])
                   .merge(df_mkt, how='left', on=['거래소코드', '연도']))
        
        # 5) fallback market-cap
        na_mask = self.df['시가총액'].isna()
        if '종가' in self.df.columns and '발행주식총수' in self.df.columns:
            self.df.loc[na_mask, '시가총액'] = self.df.loc[na_mask, '종가'] * self.df.loc[na_mask, '발행주식총수']
        
        # 정렬
        self.df = self.df.sort_values(['거래소코드', '연도']).reset_index(drop=True)
        print(f"✅ 데이터 병합 완료: {len(self.df):,}행")
        
        return self
    
    def compute_features(self):
        """2단계: Balance-Sheet Flow Convention & 팩터 특성 계산"""
        print("🔄 특성 계산 중...")
        
        # B/S 항목들의 평균값 계산 (당기말 + 전기말) / 2
        bs_cols = ['총자산','총부채','총자본','유동자산','유동부채',
                   '단기차입금','장기차입금','유형자산','무형자산',
                   '재고자산','현금및현금성자산','단기금융상품(금융기관예치금)']
        
        for c in bs_cols:
            if c in self.df.columns:
                self.df[f'avg_{c}'] = (self.df[c] + self.df.groupby('거래소코드')[c].shift(1)) / 2
        
        # 총이자비용 = 이자비용 (already in FS2)
        if '이자비용' in self.df.columns:
            self.df['총이자비용'] = self.df['이자비용']
        
        # 기본 계산용 컬럼들
        if '매출액' in self.df.columns:
            self.df['매출액증가율'] = self.df.groupby('거래소코드')['매출액'].pct_change()
        if '영업이익' in self.df.columns:
            self.df['영업이익증가율'] = self.df.groupby('거래소코드')['영업이익'].pct_change()
        
        print("✅ 특성 계산 완료")
        return self
    
    def compute_factor_signals(self):
        """3단계: 11개 팩터 시그널 계산"""
        print("🎯 팩터 시그널 계산 중...")
        
        # 데이터 정렬
        self.df = self.df.sort_values(['거래소코드', '연도']).reset_index(drop=True)
        
        # 1. Magic Formula (그린블라트)
        self._compute_magic_formula()
        
        # 2. EV/EBITDA
        self._compute_ev_ebitda()
        
        # 3. Book-to-Market (BM)
        self._compute_book_to_market()
        
        # 4. Momentum (12-1)
        self._compute_momentum()
        
        # 5. Piotroski F-Score
        self._compute_fscore()
        
        # 6. QMJ (Quality Minus Junk)
        self._compute_qmj()
        
        # 7. Low Volatility
        self._compute_low_volatility()
        
        # 8. SMB & HML (Fama-French)
        self._compute_ff_factors()
        
        # 9. DOL & DFL (Leverage - ONLY TWO)
        self._compute_leverage_factors()
        
        print("✅ 팩터 시그널 계산 완료")
        return self
    
    def _compute_magic_formula(self):
        """Magic Formula (그린블라트)"""
        # Earnings Yield = EBIT / EV
        if 'EBIT' in self.df.columns and 'EV' in self.df.columns:
            self.df['earnings_yield'] = self.df['EBIT'] / self.df['EV']
        elif '영업이익' in self.df.columns:
            # EV = 시가총액 + 총부채 - (현금 + 단기금융상품)
            cash_equiv = self.df.get('현금및현금성자산', 0) + self.df.get('단기금융상품(금융기관예치금)', 0)
            ev = self.df['시가총액'] + self.df.get('총부채', 0) - cash_equiv
            self.df['earnings_yield'] = self.df['영업이익'] / ev
        
        # ROIC = EBIT / (순운전자본 + 순유형자산)
        if '경영자본영업이익률' in self.df.columns:
            self.df['roic'] = self.df['경영자본영업이익률'] / 100
        elif 'EBIT' in self.df.columns and '순운전자본' in self.df.columns and '순유형자산' in self.df.columns:
            invested_capital = self.df['순운전자본'] + self.df['순유형자산']
            self.df['roic'] = self.df['EBIT'] / invested_capital
        
        # 연도별 랭킹 계산
        magic_scores = []
        for year in self.df['연도'].unique():
            year_df = self.df[self.df['연도'] == year].copy()
            
            if len(year_df) < 10:
                continue
            
            valid_mask = year_df['earnings_yield'].notna() & year_df['roic'].notna()
            valid_df = year_df[valid_mask].copy()
            
            if len(valid_df) < 5:
                continue
            
            # 랭킹 계산 (높을수록 좋음)
            valid_df['ey_rank'] = valid_df['earnings_yield'].rank(ascending=False)
            valid_df['roic_rank'] = valid_df['roic'].rank(ascending=False)
            
            # Magic Formula Rank = 두 랭킹의 합 (낮을수록 좋음)
            valid_df['MF_Rank'] = valid_df['ey_rank'] + valid_df['roic_rank']
            valid_df['magic_signal'] = -valid_df['MF_Rank']  # 낮은 랭크가 좋음 → 음수 부호
            
            magic_scores.append(valid_df[['거래소코드', '연도', 'magic_signal']])
        
        if magic_scores:
            magic_df = pd.concat(magic_scores)
            self.df = self.df.merge(magic_df, on=['거래소코드', '연도'], how='left')
        else:
            self.df['magic_signal'] = 0
    
    def _compute_ev_ebitda(self):
        """EV/EBITDA"""
        # EV = 시가총액 + 총부채 - (현금 + 단기금융상품)
        cash_equiv = self.df.get('현금및현금성자산', 0) + self.df.get('단기금융상품(금융기관예치금)', 0)
        ev = self.df['시가총액'] + self.df.get('총부채', 0) - cash_equiv
        
        # EBITDA = EBIT + 감가상각비 + 무형자산상각비
        if 'EBIT' in self.df.columns:
            ebit = self.df['EBIT']
        else:
            ebit = self.df.get('영업이익', 0)
        
        ebitda = ebit + self.df.get('감가상각비', 0) + self.df.get('무형자산상각비', 0)
        
        self.df['EV_EBITDA'] = ev / ebitda
        self.df['ev_ebitda_signal'] = -self.df['EV_EBITDA']  # 낮을수록 좋음
    
    def _compute_book_to_market(self):
        """Book-to-Market (BM)"""
        self.df['bm'] = self.df['총자본'] / self.df['시가총액']  # 높을수록 좋음
    
    def _compute_momentum(self):
        """Momentum (12-1)"""
        # 월수익률이 없으므로 연간 수익률 기반으로 계산
        # 가정: 전년도 대비 수익률을 momentum으로 사용
        if '주가수익률' in self.df.columns:
            self.df['mom'] = self.df.groupby('거래소코드')['주가수익률'].shift(1)
        else:
            # 종가 기반 수익률 계산
            price_col = '종가' if '종가' in self.df.columns else '시가총액'
            self.df['mom'] = self.df.groupby('거래소코드')[price_col].pct_change().shift(1)
    
    def _compute_fscore(self):
        """Piotroski F-Score (0~9점)"""
        # 1. ROA > 0
        roa_col = 'ROA' if 'ROA' in self.df.columns else '총자산수익률'
        self.df['f_roa'] = (self.df[roa_col] > 0).astype(int)
        
        # 2. CFO > 0
        cfo_col = '영업현금흐름' if '영업현금흐름' in self.df.columns else '영업CF'
        if cfo_col in self.df.columns:
            self.df['f_cfo'] = (self.df[cfo_col] > 0).astype(int)
        else:
            self.df['f_cfo'] = 0
        
        # 3. ΔROA
        self.df['f_delta_roa'] = (self.df.groupby('거래소코드')[roa_col].diff() > 0).astype(int)
        
        # 4. CFO > ROA
        if cfo_col in self.df.columns:
            total_assets = self.df.get('avg_총자산', self.df.get('총자산', 1))
            cfo_ta = self.df[cfo_col] / total_assets
            roa_ratio = self.df[roa_col] / 100  # 퍼센트를 비율로 변환
            self.df['f_cfo_roa'] = (cfo_ta > roa_ratio).astype(int)
        else:
            self.df['f_cfo_roa'] = 0
        
        # 5. Δ부채
        self.df['f_debt'] = (self.df.groupby('거래소코드')['총부채'].diff() < 0).astype(int)
        
        # 6. Δ유동비율
        if '유동비율' in self.df.columns:
            self.df['f_liquid'] = (self.df.groupby('거래소코드')['유동비율'].diff() > 0).astype(int)
        else:
            # 유동비율 = 유동자산 / 유동부채
            current_ratio = self.df['유동자산'] / self.df['유동부채']
            self.df['f_liquid'] = (current_ratio.groupby(self.df['거래소코드']).diff() > 0).astype(int)
        
        # 7. 신주발행 (납입자본금 기준으로 판단, 첫 연도는 신주발행으로 간주)
        if '자본금' in self.df.columns:
            capital_change = self.df.groupby('거래소코드')['자본금'].diff()
            # 첫 연도(diff가 NaN인 경우)는 신주발행으로 간주하여 0점 부여
            # 납입자본금이 증가하지 않은 경우(감소하거나 변화없음)에 1점 부여
            self.df['f_shares'] = ((capital_change <= 0) & (~capital_change.isna())).astype(int)
        else:
            self.df['f_shares'] = 0
        
        # 8. Δ마진
        if '매출총이익률' in self.df.columns:
            self.df['f_margin'] = (self.df.groupby('거래소코드')['매출총이익률'].diff() > 0).astype(int)
        else:
            self.df['f_margin'] = 0
        
        # 9. Δ회전율
        if '총자산회전율' in self.df.columns:
            self.df['f_turnover'] = (self.df.groupby('거래소코드')['총자산회전율'].diff() > 0).astype(int)
        else:
            self.df['f_turnover'] = 0
        
        # F-Score 합계
        fscore_cols = ['f_roa', 'f_cfo', 'f_delta_roa', 'f_cfo_roa', 'f_debt', 
                       'f_liquid', 'f_shares', 'f_margin', 'f_turnover']
        available_cols = [col for col in fscore_cols if col in self.df.columns]
        self.df['fscore'] = self.df[available_cols].sum(axis=1)
    
    def _compute_qmj(self):
        """QMJ (Quality Minus Junk)"""
        # 수익성 지표들
        profit_cols = ['ROE', '자기자본순이익률', 'ROA', '총자산수익률']
        available_profit_cols = [col for col in profit_cols if col in self.df.columns]
        
        # 안정성 지표들 (낮을수록 좋음)
        safety_cols = ['부채비율', '부채자본비율']
        available_safety_cols = [col for col in safety_cols if col in self.df.columns]
        
        # 성장성 지표들
        growth_cols = ['매출액증가율', '영업이익증가율']
        available_growth_cols = [col for col in growth_cols if col in self.df.columns]
        
        # 연도별 Z-score 표준화
        qmj_scores = []
        for year in self.df['연도'].unique():
            year_df = self.df[self.df['연도'] == year].copy()
            
            if len(year_df) < 10:
                continue
            
            # 수익성 점수 (높을수록 좋음)
            profit_z = 0
            for col in available_profit_cols:
                if year_df[col].notna().sum() > 5:
                    profit_z += stats.zscore(year_df[col].fillna(year_df[col].median()))
            
            # 안정성 점수 (부채비율 등은 낮을수록 좋음)
            safety_z = 0
            for col in available_safety_cols:
                if year_df[col].notna().sum() > 5:
                    safety_z -= stats.zscore(year_df[col].fillna(year_df[col].median()))
            
            # 성장성 점수 (높을수록 좋음)
            growth_z = 0
            for col in available_growth_cols:
                if year_df[col].notna().sum() > 5:
                    growth_z += stats.zscore(year_df[col].fillna(year_df[col].median()))
            
            year_df['qmj'] = (profit_z + safety_z + growth_z) / 3
            qmj_scores.append(year_df[['거래소코드', '연도', 'qmj']])
        
        if qmj_scores:
            qmj_df = pd.concat(qmj_scores)
            self.df = self.df.merge(qmj_df, on=['거래소코드', '연도'], how='left')
        else:
            self.df['qmj'] = 0
    
    def _compute_low_volatility(self):
        """Low Volatility"""
        # 과거 3년간 수익률 변동성 계산
        if '주가수익률' in self.df.columns:
            ret_col = '주가수익률'
        else:
            # 종가 기반 수익률 계산
            price_col = '종가' if '종가' in self.df.columns else '시가총액'
            self.df['returns'] = self.df.groupby('거래소코드')[price_col].pct_change()
            ret_col = 'returns'
        
        # 36개월(3년) 변동성 계산 (연간 데이터이므로 3년)
        self.df['vol_3y'] = self.df.groupby('거래소코드')[ret_col].rolling(3).std().reset_index(0, drop=True)
        self.df['lowvol'] = -self.df['vol_3y']  # 변동성 낮을수록 좋음
    
    def _compute_ff_factors(self):
        """SMB & HML (Fama-French 3-Factor)"""
        # 연도별 Size median, BM 30/70% 분위수로 2×3 포트폴리오 구성
        ff_scores = []
        
        for year in self.df['연도'].unique():
            year_df = self.df[self.df['연도'] == year].copy()
            
            if len(year_df) < 20 or 'bm' not in year_df.columns:
                continue
            
            year_df = year_df.dropna(subset=['시가총액', 'bm'])
            
            if len(year_df) < 6:
                continue
            
            # Size median
            size_median = year_df['시가총액'].median()
            
            # B/M 30%, 70% 분위수
            bm_30 = year_df['bm'].quantile(0.3)
            bm_70 = year_df['bm'].quantile(0.7)
            
            # 6개 포트폴리오 구성
            portfolios = {}
            portfolios['SL'] = year_df[(year_df['시가총액'] <= size_median) & (year_df['bm'] <= bm_30)]
            portfolios['SM'] = year_df[(year_df['시가총액'] <= size_median) & (year_df['bm'] > bm_30) & (year_df['bm'] <= bm_70)]
            portfolios['SH'] = year_df[(year_df['시가총액'] <= size_median) & (year_df['bm'] > bm_70)]
            portfolios['BL'] = year_df[(year_df['시가총액'] > size_median) & (year_df['bm'] <= bm_30)]
            portfolios['BM'] = year_df[(year_df['시가총액'] > size_median) & (year_df['bm'] > bm_30) & (year_df['bm'] <= bm_70)]
            portfolios['BH'] = year_df[(year_df['시가총액'] > size_median) & (year_df['bm'] > bm_70)]
            
            # 다음해 수익률 계산을 위한 준비 (현재는 단순히 factor 값으로 대체)
            # SMB = (S/H + S/M + S/L)/3 - (B/H + B/M + B/L)/3
            small_factor = sum([len(portfolios[p]) for p in ['SL', 'SM', 'SH']]) / 3
            big_factor = sum([len(portfolios[p]) for p in ['BL', 'BM', 'BH']]) / 3
            smb_factor = small_factor - big_factor
            
            # HML = (H/S + H/B)/2 - (L/S + L/B)/2
            high_factor = (len(portfolios['SH']) + len(portfolios['BH'])) / 2
            low_factor = (len(portfolios['SL']) + len(portfolios['BL'])) / 2
            hml_factor = high_factor - low_factor
            
            # 각 종목에 SMB, HML 할당
            year_df['smb'] = smb_factor / len(year_df)  # 정규화
            year_df['hml'] = hml_factor / len(year_df)  # 정규화
            
            ff_scores.append(year_df[['거래소코드', '연도', 'smb', 'hml']])
        
        if ff_scores:
            ff_df = pd.concat(ff_scores)
            self.df = self.df.merge(ff_df, on=['거래소코드', '연도'], how='left')
        else:
            self.df['smb'] = 0
            self.df['hml'] = 0
    
    def _compute_leverage_factors(self):
        """Leverage Factors - DOL & DFL만 계산"""
        # DOL (Degree of Operating Leverage)
        if '매출액증가율' in self.df.columns and '영업이익증가율' in self.df.columns:
            # DOL = 영업이익증가율 / 매출액증가율
            self.df['DOL'] = self.df['영업이익증가율'] / self.df['매출액증가율']
            # 무한값 처리
            self.df['DOL'] = self.df['DOL'].replace([np.inf, -np.inf], np.nan)
        else:
            self.df['DOL'] = np.nan
        
        # DFL (Degree of Financial Leverage)
        if '영업이익' in self.df.columns and '총이자비용' in self.df.columns:
            # DFL = 영업이익 / (영업이익 - 이자비용)
            denominator = self.df['영업이익'] - self.df['총이자비용']
            # 분모가 0 이하인 경우 NaN 처리
            self.df['DFL'] = np.where(denominator > 0, 
                                     self.df['영업이익'] / denominator, 
                                     np.nan)
        else:
            self.df['DFL'] = np.nan
    
    def construct_long_portfolio(self, df, signal_col, date, top_n=10):
        """Long-Only Top-10 Equal-Weight 포트폴리오 구성"""
        universe = df[df['rebal_date'] == date].copy()
        
        # Piotroski F-Score의 경우 F-Score >= 8 필터 적용
        if signal_col == 'fscore':
            universe = universe[universe['fscore'] >= 8]
        
        # 상위 top_n개 종목 선택
        if len(universe) == 0:
            return pd.Series(dtype=float)
        
        winners = universe.sort_values(signal_col, ascending=False).head(top_n)
        n = len(winners)
        
        if n == 0:
            return pd.Series(dtype=float)
        
        # Equal weight
        return pd.Series(1/n, index=winners.index)
    
    def backtest(self):
        """4단계: 백테스트 실행 - Long-Only Top-10"""
        print("📈 백테스트 실행 중...")
        
        # 리밸런싱 날짜 설정 (연말 기준)
        self.df['rebal_date'] = pd.to_datetime(self.df['연도'].astype(str) + '-12-31')
        
        # 다음해 수익률 계산 (t+1 수익률)
        self.df = self.df.sort_values(['거래소코드', '연도'])
        if '주가수익률' in self.df.columns:
            self.df['next_ret'] = self.df.groupby('거래소코드')['주가수익률'].shift(-1)
        else:
            # 종가 기반 수익률
            price_col = '종가' if '종가' in self.df.columns else '시가총액'
            self.df['next_ret'] = self.df.groupby('거래소코드')[price_col].pct_change(periods=-1)
        
        # 2013년 이후 데이터만 사용
        backtest_data = self.df[self.df['연도'] >= 2013].copy()
        
        # 팩터 시그널 매핑
        factor_signals = {
            'Magic Formula': 'magic_signal',
            'EV/EBITDA': 'ev_ebitda_signal',
            'Book-to-Market': 'bm',
            'Momentum 12-1': 'mom',
            'Piotroski': 'fscore',
            'QMJ': 'qmj',
            'LowVol': 'lowvol',
            'SMB': 'smb',
            'HML': 'hml',
            'DOL': 'DOL',
            'DFL': 'DFL'
        }
        
        for strategy_name, signal_col in factor_signals.items():
            if signal_col not in backtest_data.columns:
                print(f"  ⚠️ {strategy_name} 시그널 컬럼 '{signal_col}' 없음, 건너뜀")
                continue
                
            print(f"  🔄 {strategy_name} 계산 중...")
            
            portfolio_returns = []
            rebal_dates = sorted(backtest_data['rebal_date'].unique())
            
            for date in rebal_dates:
                # 포트폴리오 구성
                weights = self.construct_long_portfolio(backtest_data, signal_col, date, top_n=10)
                
                if len(weights) == 0:
                    continue
                
                # 다음 기간 수익률 계산
                date_data = backtest_data[backtest_data['rebal_date'] == date]
                next_returns = date_data.loc[weights.index, 'next_ret']
                
                # 포트폴리오 수익률 = weight * return의 합
                port_ret = (weights * next_returns).sum()
                
                portfolio_returns.append({
                    'date': date,
                    'return': port_ret,
                    'n_stocks': len(weights)
                })
            
            if portfolio_returns:
                ret_df = pd.DataFrame(portfolio_returns).set_index('date')
                self.factor_returns[strategy_name] = ret_df
        
        print("✅ 백테스트 완료")
        return self
    
    def calc_performance_stats(self):
        """5단계: 성과지표 계산"""
        print("📊 성과지표 계산 중...")
        
        for strategy_name, returns_df in self.factor_returns.items():
            if len(returns_df) == 0:
                continue
            
            ret_series = returns_df['return'].dropna()
            
            if len(ret_series) < 3:
                continue
            
            # CAGR 계산
            cum_ret = (1 + ret_series).cumprod()
            n_years = len(cum_ret)
            if n_years > 0 and cum_ret.iloc[0] > 0:
                cagr = (cum_ret.iloc[-1] / cum_ret.iloc[0]) ** (1 / n_years) - 1
            else:
                cagr = 0
            
            # 연간 변동성
            ann_vol = ret_series.std()
            
            # 샤프 비율
            sharpe = cagr / ann_vol if ann_vol > 0 else 0
            
            # 최대 낙폭
            running_max = cum_ret.cummax()
            drawdown = (cum_ret / running_max - 1)
            max_dd = drawdown.min()
            
            # 칼마 비율
            calmar = cagr / abs(max_dd) if max_dd != 0 else 0
            
            self.performance_stats[strategy_name] = {
                'CAGR': cagr,
                'AnnVol': ann_vol,
                'Sharpe': sharpe,
                'MaxDD': max_dd,
                'Calmar': calmar
            }
        
        print("✅ 성과지표 계산 완료")
        return self
    
    def plot_results(self):
        """6단계: 시각화"""
        print("📈 시각화 생성 중...")
        
        if not self.factor_returns:
            print("  ⚠️ 백테스트 결과가 없습니다.")
            return self
        
        # 누적수익률 차트
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (strategy_name, returns_df) in enumerate(self.factor_returns.items()):
            if len(returns_df) == 0:
                continue
            
            cum_ret = (1 + returns_df['return']).cumprod()
            
            fig.add_trace(
                go.Scatter(
                    x=cum_ret.index,
                    y=cum_ret.values,
                    name=strategy_name,
                    line=dict(color=colors[i % len(colors)])
                )
            )
        
        fig.update_layout(
            title="팩터 전략별 누적수익률 (Long-Only Top-10 Equal-Weight)",
            xaxis_title="연도",
            yaxis_title="누적수익률",
            font=dict(family='AppleGothic'),
            height=600,
            hovermode='x unified'
        )
        
        fig.show()
        
        # 성과지표 테이블
        if self.performance_stats:
            stats_df = pd.DataFrame(self.performance_stats).T
            print("\n📊 성과지표 요약:")
            print(stats_df.round(4))
        
        print("✅ 시각화 완료")
        return self
    
    def save_results(self):
        """7단계: 결과 저장"""
        print("💾 결과 저장 중...")
        
        # 피처 데이터 저장 (FS2_features.csv) - data/processed에 저장
        os.makedirs('data/processed', exist_ok=True)
        feature_output_path = 'data/processed/FS2_features.csv'
        self.df.to_csv(feature_output_path, encoding='utf-8', index=False)
        print(f"✅ 피처 데이터 저장: {feature_output_path}")
        
        # 성과 통계 저장
        if self.performance_stats:
            stats_df = pd.DataFrame(self.performance_stats).T
            stats_output_path = os.path.join(self.output_dir, 'factor_performance_stats.csv')
            stats_df.to_csv(stats_output_path, encoding='utf-8-sig')
            print(f"✅ 성과통계 저장: {stats_output_path}")
        
        # 팩터 수익률 저장
        for strategy_name, returns_df in self.factor_returns.items():
            if len(returns_df) > 0:
                safe_name = strategy_name.replace('/', '_').replace(' ', '_')
                filename = os.path.join(self.output_dir, f"factor_returns_{safe_name}.csv")
                returns_df.to_csv(filename, encoding='utf-8-sig')
        
        print("✅ 팩터 수익률 저장 완료")
        
        # 요약 출력
        print("\n" + "="*60)
        print("📊 FACTOR BACKTESTING SUMMARY (Long-Only Top-10)")
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
    parser = argparse.ArgumentParser(description='Factor Backtesting - Long-Only Top-10 Equal-Weight')
    parser.add_argument('--data_path', type=str, default='data/processed', 
                       help='Data directory path (default: data/processed)')
    parser.add_argument('--output_dir', type=str, default='outputs/backtesting',
                       help='Output directory path (default: outputs/backtesting)')
    
    args = parser.parse_args()
    
    print("🚀 Factor Investing Backtesting 시작 (Long-Only Top-10)")
    print("="*60)
    
    # 백테스터 초기화 및 실행
    backtester = FactorBacktester(data_path=args.data_path, output_dir=args.output_dir)
    
    backtester.load_data() \
              .compute_features() \
              .compute_factor_signals() \
              .backtest() \
              .calc_performance_stats() \
              .plot_results() \
              .save_results()
    
    print("\n🎉 백테스트 완료!")
    return backtester

if __name__ == "__main__":
    results = main()