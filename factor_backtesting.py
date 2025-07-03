"""
Factor Investing Backtesting Framework - Updated Version
FF3 통합 전략, B/M 제거, DOL/DFL 제거 버전
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Visualization will be limited.")

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some features may be limited.")

from scipy import stats
import warnings
import glob
import os
import argparse
try:
    from pykrx import stock, bond
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False
    print("Warning: pykrx not available. FF3 factor builder will use mock data.")
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
    """팩터 투자 백테스트 클래스 - Updated with FF3 Integration"""
    
    def __init__(self, data_path=None, output_dir=None, top_n=10, fscore_min_score=8, momentum_period=12):
        # 단순한 경로 설정 - 현재 디렉토리 기준
        self.data_path = data_path or 'data/processed'
        self.df = None
        self.factor_returns = {}
        self.performance_stats = {}
        
        # 포트폴리오 설정
        self.top_n = top_n  # 상위 n개 종목 선택
        self.fscore_min_score = fscore_min_score  # F-score 최소 점수
        self.momentum_period = momentum_period  # 모멘텀 기간 (개월)
        
        # 출력 디렉토리 설정 - 파라미터 조합별로 하위 폴더 생성
        base_output_dir = output_dir or 'outputs/backtesting'
        param_suffix = f"top{top_n}_f{fscore_min_score}_mom{momentum_period}m"
        self.output_dir = os.path.join(base_output_dir, param_suffix)
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"📁 결과 저장 경로: {self.output_dir}")
        
    def load_data(self):
        """1단계: 데이터 로딩 및 병합"""
        print("📊 데이터 로딩 중...")
        
        # 1) 재무제표 원본 (FS2.csv) - data/processed에서 찾기
        fs_path = os.path.join(self.data_path, 'FS2_default.csv')
        if not os.path.exists(fs_path):
            # 대안 경로들 시도
            alternative_paths = ['data/processed/FS2_default.csv', 'data/FS2_default.csv', 'FS2_default.csv']
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    fs_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"FS2_default.csv 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        
        df_fs = pd.read_csv(fs_path, encoding='utf-8-sig')
        # FS2_default.csv의 시가총액 컬럼 제거 (시가총액.csv의 정확한 값 사용 위해)
        if '시가총액' in df_fs.columns:
            df_fs = df_fs.drop(columns=['시가총액'])
            print("⚠️  FS2_default.csv의 시가총액 컬럼 제거 (시가총액.csv 사용 예정)")
        if '로그시가총액' in df_fs.columns:
            df_fs = df_fs.drop(columns=['로그시가총액'])
            print("⚠️  FS2_default.csv의 로그시가총액 컬럼 제거")
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
        
        # 3) 시가총액 (시가총액.csv) - 우선주 포함한 정확한 시가총액
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
        
        # 컬럼명 통일 (회계년도 → 연도) 및 데이터 타입 통일
        if '회계년도' in df_fs.columns:
            df_fs['연도'] = pd.to_numeric(df_fs['회계년도'], errors='coerce').astype('Int64')
        if '회계년도' in df_price.columns:
            df_price['연도'] = pd.to_numeric(df_price['회계년도'], errors='coerce').astype('Int64')
        if '회계년도' in df_mkt.columns:
            df_mkt['연도'] = pd.to_numeric(df_mkt['회계년도'], errors='coerce').astype('Int64')
        
        # 4) 병합 (거래소코드 + 연도)
        self.df = (df_fs
                   .merge(df_price, how='left', on=['거래소코드', '연도'])
                   .merge(df_mkt, how='left', on=['거래소코드', '연도']))
        
        # 5) 시가총액 컬럼 확인 및 결측치 보완
        if '시가총액' in self.df.columns:
            print("✅ 시가총액 컬럼 확인됨")
            na_mask = self.df['시가총액'].isna()
            na_count = na_mask.sum()
            if na_count > 0:
                print(f"⚠️  시가총액 결측치 {na_count:,}개 발견")
                if '종가' in self.df.columns and '발행주식총수' in self.df.columns:
                    self.df.loc[na_mask, '시가총액'] = self.df.loc[na_mask, '종가'] * self.df.loc[na_mask, '발행주식총수']
                    print(f"✅ 결측치 {na_count:,}개를 종가 × 발행주식총수로 보완")
        else:
            print("❌ 시가총액 컬럼이 없습니다!")
            return self
        
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
        """3단계: 팩터 시그널 계산 (업데이트된 버전)"""
        print("🎯 팩터 시그널 계산 중...")
        
        # 데이터 정렬
        self.df = self.df.sort_values(['거래소코드', '연도']).reset_index(drop=True)
        
        # default 컬럼 확인 및 검증
        if 'default' not in self.df.columns:
            print("  ❌ 'default' 컬럼이 없습니다!")
            print("  💡 부실/정상 기업 구분 분석을 위해 default 컬럼이 필요합니다.")
            print("  💡 대안: 재무비율 기반으로 기업을 분류하겠습니다.")
            
            # 재무비율 기반 분류 (예: 부채비율이 높고 수익성이 낮은 기업을 위험기업으로 분류)
            self._create_risk_classification()
        else:
            print("  ✅ 'default' 컬럼 발견")
            
        # 부실/정상 기업 통계
        total_count = len(self.df)
        default_count = (self.df['default'] == 1).sum()
        normal_count = total_count - default_count
        
        print(f"  📊 데이터 구성:")
        print(f"     - 전체: {total_count:,}개 관측치")
        print(f"     - 부실기업: {default_count:,}개 ({default_count/total_count*100:.1f}%)")
        print(f"     - 정상기업: {normal_count:,}개 ({normal_count/total_count*100:.1f}%)")
        
        # 부실기업 데이터 충분성 검증
        if default_count == 0:
            print("  ⚠️ 부실기업 데이터가 없습니다! 부실/정상 비교 분석이 불가능합니다.")
            print("  💡 전체기업 대상으로만 백테스트를 진행합니다.")
        elif default_count < 50:
            print(f"  ⚠️ 부실기업 수가 적습니다 ({default_count}개). 통계적 유의성이 제한적일 수 있습니다.")
            
        # 연도별 부실기업 분포 확인
        yearly_defaults = self.df[self.df['default'] == 1].groupby('연도').size()
        if len(yearly_defaults) > 0:
            print(f"  📅 연도별 부실기업 분포:")
            for year, count in yearly_defaults.items():
                print(f"     - {year}년: {count}개")
        
        # 팩터 시그널 계산
        print("  🔄 팩터 시그널 계산 시작...")
        
        # 1. Magic Formula (그린블라트)
        self._compute_magic_formula()
        
        # 2. EV/EBITDA
        self._compute_ev_ebitda()
        
        # 3. Momentum (customizable period)
        self._compute_momentum()
        
        # 4. Piotroski F-Score
        self._compute_fscore()
        
        # 5. QMJ (Quality Minus Junk)
        self._compute_qmj()
        
        # 6. Low Volatility
        self._compute_low_volatility()
        
        # 7. BM 계산 (FF3에서 사용)
        self._compute_book_to_market()
        
        # 8. Fama-French 3Factor (통합 전략)
        self._compute_ff3_factors()
        
        print("  ✅ 팩터 시그널 계산 완료")
        
        return self
        
    def _create_risk_classification(self):
        """default 컬럼이 없을 때 재무비율 기반으로 위험기업 분류"""
        print("  🔄 재무비율 기반 위험기업 분류 중...")
        
        # 위험 지표들 확인
        risk_indicators = []
        
        # 1. 부채비율
        if '부채비율' in self.df.columns:
            risk_indicators.append('부채비율')
        
        # 2. 수익성 지표
        profitability_cols = ['ROE', '자기자본순이익률', 'ROA', '총자산수익률', '영업이익률']
        available_profit_col = None
        for col in profitability_cols:
            if col in self.df.columns:
                available_profit_col = col
                break
        
        if available_profit_col:
            risk_indicators.append(available_profit_col)
        
        # 3. 유동성 지표
        if '유동비율' in self.df.columns:
            risk_indicators.append('유동비율')
        elif '유동자산' in self.df.columns and '유동부채' in self.df.columns:
            # 유동비율 계산
            liabilities = pd.to_numeric(self.df['유동부채'], errors='coerce').replace(0, np.nan)
            self.df['유동비율'] = pd.to_numeric(self.df['유동자산'], errors='coerce') / liabilities
            risk_indicators.append('유동비율')
        
        if len(risk_indicators) == 0:
            print("    ❌ 위험분류를 위한 재무지표가 부족합니다. 모든 기업을 정상으로 분류합니다.")
            self.df['default'] = 0
            return
        
        print(f"    📊 사용 지표: {risk_indicators}")
        
        # 연도별로 위험기업 분류 (상위/하위 20%로 분류)
        self.df['default'] = 0  # 기본값
        
        for year in self.df['연도'].unique():
            year_mask = self.df['연도'] == year
            year_data = self.df[year_mask].copy()
            
            if len(year_data) < 20:  # 최소 20개 기업 필요
                continue
            
            risk_score = 0
            valid_indicators = 0
            
            # 부채비율 (높을수록 위험)
            if '부채비율' in risk_indicators:
                debt_ratio = pd.to_numeric(year_data['부채비율'], errors='coerce')
                if debt_ratio.notna().sum() > 5:
                    debt_percentile = debt_ratio.rank(pct=True)
                    risk_score += debt_percentile
                    valid_indicators += 1
            
            # 수익성 (낮을수록 위험)
            if available_profit_col in risk_indicators:
                profit_ratio = pd.to_numeric(year_data[available_profit_col], errors='coerce')
                if profit_ratio.notna().sum() > 5:
                    profit_percentile = profit_ratio.rank(pct=True, ascending=False)  # 낮을수록 높은 순위
                    risk_score += profit_percentile
                    valid_indicators += 1
            
            # 유동비율 (낮을수록 위험)
            if '유동비율' in risk_indicators:
                liquid_ratio = pd.to_numeric(year_data['유동비율'], errors='coerce')
                if liquid_ratio.notna().sum() > 5:
                    liquid_percentile = liquid_ratio.rank(pct=True, ascending=False)  # 낮을수록 높은 순위
                    risk_score += liquid_percentile
                    valid_indicators += 1
            
            if valid_indicators > 0:
                risk_score = risk_score / valid_indicators
                # 상위 20%를 위험기업으로 분류
                risk_threshold = risk_score.quantile(0.8)
                high_risk_mask = risk_score >= risk_threshold
                self.df.loc[year_mask & high_risk_mask, 'default'] = 1
        
        final_default_count = (self.df['default'] == 1).sum()
        print(f"    ✅ 재무비율 기반 분류 완료: {final_default_count}개 기업을 위험기업으로 분류")
    
    def _compute_magic_formula(self):
        """Magic Formula (그린블라트)"""
        # Earnings Yield = EBIT / EV
        if 'EBIT' in self.df.columns and 'EV' in self.df.columns:
            self.df['earnings_yield'] = self.df['EBIT'] / self.df['EV']
        elif '영업이익' in self.df.columns and '시가총액' in self.df.columns:
            # EV = 시가총액 + 총부채 - (현금 + 단기금융상품)
            cash_equiv = self.df.get('현금및현금성자산', 0) + self.df.get('단기금융상품(금융기관예치금)', 0)
            ev = self.df['시가총액'] + self.df.get('총부채', 0) - cash_equiv
            ev = ev.replace(0, np.nan)  # 0으로 나누기 방지
            self.df['earnings_yield'] = self.df['영업이익'] / ev
        else:
            self.df['earnings_yield'] = np.nan
        
        # ROIC = EBIT / (순운전자본 + 순유형자산)
        if '경영자본영업이익률' in self.df.columns:
            self.df['roic'] = pd.to_numeric(self.df['경영자본영업이익률'], errors='coerce') / 100
        elif 'EBIT' in self.df.columns and '순운전자본' in self.df.columns and '순유형자산' in self.df.columns:
            invested_capital = self.df['순운전자본'] + self.df['순유형자산']
            invested_capital = invested_capital.replace(0, np.nan)  # 0으로 나누기 방지
            self.df['roic'] = self.df['EBIT'] / invested_capital
        elif '영업이익' in self.df.columns and '총자산' in self.df.columns:
            # 대안: ROA 기반 ROIC 근사값 사용
            total_assets = pd.to_numeric(self.df['총자산'], errors='coerce')
            total_assets = total_assets.replace(0, np.nan)
            self.df['roic'] = pd.to_numeric(self.df['영업이익'], errors='coerce') / total_assets
        else:
            self.df['roic'] = np.nan
        
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
        """Book-to-Market (BM) - FF3에서만 사용"""
        self.df['bm'] = self.df['총자본'] / self.df['시가총액']  # 높을수록 좋음
    
    def _compute_momentum(self):
        """Momentum (customizable period) - 개선된 버전 with 실제 일별 데이터"""
        print(f"  🔄 모멘텀 계산 ({self.momentum_period}개월 기간)")
        
        # 실제 일별 데이터를 로딩하여 정확한 모멘텀 계산
        try:
            momentum_data = self._load_daily_momentum_data()
            if momentum_data is not None and len(momentum_data) > 0:
                print(f"    📊 실제 일별 데이터 기반 {self.momentum_period}개월 모멘텀 계산")
                self._compute_daily_momentum(momentum_data)
                return
        except Exception as e:
            print(f"    ⚠️ 일별 데이터 로딩 실패: {e}")
        
        # Fallback: 연간 데이터 기반 근사 모멘텀 계산
        print(f"    📊 연간 데이터 기반 모멘텀 계산")
        
        # 모멘텀 기간을 연도 단위로 변환 (더 정확한 로직)
        if self.momentum_period <= 6:
            # 6개월 이하: 당년도 vs 전년도
            shift_periods = 1
            period_desc = f"{self.momentum_period}개월(1년전 대비)"
        elif self.momentum_period <= 18:
            # 6-18개월: 1년 전 대비
            shift_periods = 1
            period_desc = f"{self.momentum_period}개월(1년전 대비)"
        elif self.momentum_period <= 30:
            # 18-30개월: 2년 전 대비
            shift_periods = 2
            period_desc = f"{self.momentum_period}개월(2년전 대비)"
        else:
            # 30개월 이상: 해당 연도 수만큼 과거
            shift_periods = max(2, self.momentum_period // 12)
            period_desc = f"{self.momentum_period}개월({shift_periods}년전 대비)"
        
        print(f"    📊 실제 적용: {period_desc}")
        
            # 종가 기반 수익률 계산
        price_col = None
        for col in ['종가', '시가총액', 'market_cap']:
            if col in self.df.columns:
                price_col = col
                break
        
        if price_col is None:
            print("    ❌ 가격 데이터를 찾을 수 없습니다.")
            self.df['mom'] = 0
            return
        
        # 과거 대비 수익률 계산
        self.df = self.df.sort_values(['거래소코드', '연도'])
        price_lag = self.df.groupby('거래소코드')[price_col].shift(shift_periods)
        self.df['mom'] = (self.df[price_col] / price_lag - 1)
        
        # 결측값 처리
        self.df['mom'] = self.df['mom'].fillna(0)
        
        # 통계 출력
        valid_momentum = self.df['mom'].dropna()
        if len(valid_momentum) > 0:
            print(f"    📈 모멘텀 통계: 평균 {valid_momentum.mean():.4f}, "
                  f"중앙값 {valid_momentum.median():.4f}, "
                  f"표준편차 {valid_momentum.std():.4f}")
        else:
            print("    ⚠️ 유효한 모멘텀 데이터가 없습니다.")
    
    def _load_daily_momentum_data(self):
        """실제 일별 주가 데이터 로딩 (모멘텀 계산용)"""
        try:
            # 연도별 주가 파일들 찾기
            price_files = []
            for data_dir in ['data/raw', 'data', '.']:
                pattern = os.path.join(data_dir, '20*.csv')
                found_files = sorted(glob.glob(pattern))
                if found_files:
                    price_files = found_files
                    break
            
            if not price_files:
                return None
            
            # 필요한 연도만 로딩 (성능 최적화)
            df_list = []
            for file_path in price_files:
                try:
                    df_temp = pd.read_csv(file_path, encoding='utf-8-sig')
                    
                    # 실제 컬럼명에 맞게 매핑
                    column_mapping = {
                        'date_cols': ['매매년월일', '날짜', 'date', 'Date'],
                        'price_cols': ['종가(원)', '종가', 'close', 'Close'],
                        'code_cols': ['거래소코드', 'code', 'ticker']
                    }
                    
                    # 필요한 컬럼 찾기
                    date_col = None
                    price_col = None
                    code_col = None
                    
                    for col in column_mapping['date_cols']:
                        if col in df_temp.columns:
                            date_col = col
                            break
                    
                    for col in column_mapping['price_cols']:
                        if col in df_temp.columns:
                            price_col = col
                            break
                    
                    for col in column_mapping['code_cols']:
                        if col in df_temp.columns:
                            code_col = col
                            break
                    
                    if date_col and price_col and code_col:
                        df_selected = df_temp[[code_col, date_col, price_col]].copy()
                        
                        # 컬럼명 표준화
                        df_selected.columns = ['거래소코드', 'date_str', '종가']
                        
                        # 날짜 컬럼 처리 (매매년월일은 2012/01/02 형태)
                        df_selected['date'] = pd.to_datetime(df_selected['date_str'], errors='coerce')
                        
                        # 종가 컬럼 숫자 변환
                        df_selected['종가'] = pd.to_numeric(df_selected['종가'], errors='coerce')
                        
                        # 유효한 데이터만 선택
                        df_selected = df_selected.dropna(subset=['date', '종가'])
                        df_selected = df_selected[df_selected['종가'] > 0]  # 0 이하 가격 제외
                        
                        if len(df_selected) > 0:
                            df_list.append(df_selected[['거래소코드', 'date', '종가']])
                            
                except Exception as e:
                    print(f"    ⚠️ {file_path} 로딩 실패: {e}")
                    continue
            
            if df_list:
                daily_data = pd.concat(df_list, ignore_index=True)
                daily_data = daily_data.sort_values(['거래소코드', 'date']).reset_index(drop=True)
                print(f"    ✅ 일별 데이터 로딩 성공: {len(daily_data):,}행")
                return daily_data
            else:
                return None
                
        except Exception as e:
            print(f"    ❌ 일별 데이터 로딩 중 오류: {e}")
            return None
    
    def _compute_daily_momentum(self, daily_data):
        """일별 데이터를 이용한 정확한 모멘텀 계산"""
        try:
            # 연도별 모멘텀 결과를 저장할 딕셔너리
            momentum_results = {}
            
            for year in self.df['연도'].unique():
                if pd.isna(year):
                    continue
                    
                year = int(year)
                
                # 해당 연도 3월 말 기준일 (회계년도 기준)
                current_date = pd.Timestamp(f"{year}-03-31")
                # N개월 전 기준일
                past_date = current_date - pd.DateOffset(months=self.momentum_period)
                
                # 각 종목별 모멘텀 계산
                year_momentum = {}
                
                for code in self.df[self.df['연도'] == year]['거래소코드'].unique():
                    try:
                        stock_data = daily_data[daily_data['거래소코드'] == code].copy()
                        
                        if len(stock_data) < 2:
                            continue
                        
                        # 현재 시점 가격 (가장 가까운 날짜)
                        current_prices = stock_data[stock_data['date'] <= current_date]
                        if len(current_prices) == 0:
                            continue
                        current_price = current_prices.iloc[-1]['종가']
                        
                        # 과거 시점 가격 (N개월 전 가장 가까운 날짜)
                        past_prices = stock_data[stock_data['date'] <= past_date]
                        if len(past_prices) == 0:
                            continue
                        past_price = past_prices.iloc[-1]['종가']
                        
                        # 모멘텀 계산 (수익률)
                        if past_price > 0:
                            momentum = (current_price / past_price) - 1
                            year_momentum[code] = momentum
                            
                    except Exception as e:
                        continue
                
                momentum_results[year] = year_momentum
                
            # 결과를 원본 DataFrame에 병합
            self.df['mom'] = 0.0
            
            for _, row in self.df.iterrows():
                year = row['연도']
                code = row['거래소코드']
                
                if year in momentum_results and code in momentum_results[year]:
                    self.df.loc[self.df.index == row.name, 'mom'] = momentum_results[year][code]
            
            # 통계 출력
            valid_momentum = self.df[self.df['mom'] != 0]['mom']
            if len(valid_momentum) > 0:
                print(f"    📈 일별 모멘텀 통계: 평균 {valid_momentum.mean():.4f}, "
                      f"중앙값 {valid_momentum.median():.4f}, "
                      f"표준편차 {valid_momentum.std():.4f}")
                print(f"    📊 유효한 모멘텀 계산: {len(valid_momentum):,}개 ({len(valid_momentum)/len(self.df)*100:.1f}%)")
            else:
                print("    ⚠️ 유효한 일별 모멘텀 데이터가 없습니다.")
                
        except Exception as e:
            print(f"    ❌ 일별 모멘텀 계산 실패: {e}")
            # Fallback to 0
            self.df['mom'] = 0
    
    def _compute_fscore(self):
        """Piotroski F-Score (0~9점)"""
        # 1. ROA > 0
        if 'ROA' in self.df.columns:
            roa_col = 'ROA'
        elif '총자산수익률' in self.df.columns:
            roa_col = '총자산수익률'
        else:
            # ROA 직접 계산: 당기순이익 / 총자산
            if '당기순이익' in self.df.columns and '총자산' in self.df.columns:
                total_assets = pd.to_numeric(self.df['총자산'], errors='coerce').replace(0, np.nan)
                self.df['ROA'] = pd.to_numeric(self.df['당기순이익'], errors='coerce') / total_assets
                roa_col = 'ROA'
            else:
                self.df['f_roa'] = 0
                roa_col = None
        
        if roa_col:
            self.df['f_roa'] = (pd.to_numeric(self.df[roa_col], errors='coerce') > 0).astype(int)
        
        # 2. CFO > 0
        cfo_col = '영업현금흐름' if '영업현금흐름' in self.df.columns else '영업CF'
        if cfo_col in self.df.columns:
            self.df['f_cfo'] = (self.df[cfo_col] > 0).astype(int)
        else:
            self.df['f_cfo'] = 0
        
        # 3. ΔROA
        if roa_col:
            self.df['f_delta_roa'] = (self.df.groupby('거래소코드')[roa_col].diff() > 0).astype(int)
        else:
            self.df['f_delta_roa'] = 0
        
        # 4. CFO > ROA
        if cfo_col in self.df.columns and roa_col:
            total_assets = self.df.get('avg_총자산', self.df.get('총자산', 1))
            total_assets = pd.to_numeric(total_assets, errors='coerce').replace(0, np.nan)
            cfo_ta = pd.to_numeric(self.df[cfo_col], errors='coerce') / total_assets
            roa_ratio = pd.to_numeric(self.df[roa_col], errors='coerce')
            # ROA가 이미 비율이면 그대로, 퍼센트면 100으로 나누기
            if roa_ratio.max() > 1:  # 퍼센트로 추정
                roa_ratio = roa_ratio / 100
            self.df['f_cfo_roa'] = (cfo_ta > roa_ratio).astype(int)
        else:
            self.df['f_cfo_roa'] = 0
        
        # 5. Δ부채
        self.df['f_debt'] = (self.df.groupby('거래소코드')['총부채'].diff() < 0).astype(int)
        
        # 6. Δ유동비율
        if '유동비율' in self.df.columns:
            self.df['f_liquid'] = (self.df.groupby('거래소코드')['유동비율'].diff() > 0).astype(int)
        elif '유동자산' in self.df.columns and '유동부채' in self.df.columns:
            # 유동비율 = 유동자산 / 유동부채
            liabilities = pd.to_numeric(self.df['유동부채'], errors='coerce').replace(0, np.nan)
            current_ratio = pd.to_numeric(self.df['유동자산'], errors='coerce') / liabilities
            self.df['f_liquid'] = (current_ratio.groupby(self.df['거래소코드']).diff() > 0).astype(int)
        else:
            # 유동자산이 없으면 현금+재고자산으로 근사
            if '현금및현금성자산' in self.df.columns and '재고자산' in self.df.columns and '유동부채' in self.df.columns:
                liquid_assets = (pd.to_numeric(self.df['현금및현금성자산'], errors='coerce').fillna(0) + 
                               pd.to_numeric(self.df['재고자산'], errors='coerce').fillna(0))
                liabilities = pd.to_numeric(self.df['유동부채'], errors='coerce').replace(0, np.nan)
                approx_ratio = liquid_assets / liabilities
                self.df['f_liquid'] = (approx_ratio.groupby(self.df['거래소코드']).diff() > 0).astype(int)
            else:
                self.df['f_liquid'] = 0
        
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
    
    # ─────────────────────────────────────────────────────────


    def _build_market_factor(self, start="20000103"):
        """
        ▍MKT_RF (시장 위험 프리미엄) 월·연 시계열 생성
        • KOSPI 가격지수 종가 → 월말 수익률 (Yahoo Finance 또는 pykrx)
        • 무위험수익률 → 월평균 (한국 국고채 3개월 또는 고정값)
        • 월초과수익률 → 4/1 ~ 다음 3/31 누적 → 연 MKT_RF
        """
        try:
            import yfinance as yf
            
            # 시작일과 종료일을 datetime 형태로 변환
            start_dt = pd.to_datetime(start, format='%Y%m%d')
            end_dt = pd.Timestamp.today()
            
            print(f"  📊 시장 팩터 계산: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}")

            # ── 1. KOSPI 데이터 가져오기 ──────────────────────────
            kospi_success = False
            try:
                # Yahoo Finance로 KOSPI 데이터 가져오기 (^KS11)
                kospi_data = yf.download('^KS11', start=start_dt, end=end_dt, progress=False)
                if not kospi_data.empty and len(kospi_data) > 0:
                    # Multi-level columns 처리
                    if isinstance(kospi_data.columns, pd.MultiIndex):
                        kospi_close = kospi_data['Close'].iloc[:, 0]  # 첫 번째 컬럼 선택
                    else:
                        kospi_close = kospi_data['Close']
                    
                    # 월말 데이터로 리샘플링
                    kospi_monthly = kospi_close.resample("M").last()
                    mkt_ret_m = kospi_monthly.pct_change().dropna()
                    kospi_success = True
                    print(f"  ✅ Yahoo Finance KOSPI 데이터 성공: {len(mkt_ret_m)}개월")
                else:
                    raise ValueError("Yahoo Finance에서 빈 데이터 반환")
                    
            except Exception as e:
                print(f"  ⚠️ Yahoo Finance KOSPI 실패: {e}")
                
                # pykrx 시도
                try:
                    kospi = (stock.get_index_ohlcv(start, end_dt.strftime("%Y%m%d"), "1001")["종가"]
                            .resample("M").last())
                    mkt_ret_m = kospi.pct_change().dropna()
                    kospi_success = True
                    print(f"  ✅ pykrx KOSPI 데이터 성공: {len(mkt_ret_m)}개월")
                except Exception as e2:
                    print(f"  ⚠️ pykrx KOSPI도 실패: {e2}")

            # KOSPI 데이터 가져오기 실패 시 Mock 데이터 생성
            if not kospi_success:
                print("  💡 Mock KOSPI 데이터로 대체")
                date_range = pd.date_range(start=start_dt, end=end_dt, freq='M')
                np.random.seed(42)  # 재현 가능한 결과
                # 연 8% 수익률, 월 변동성 4%로 가정한 가상 KOSPI 데이터
                monthly_returns = np.random.normal(0.08/12, 0.04, len(date_range))
                mkt_ret_m = pd.Series(monthly_returns, index=date_range)

            # ── 2. 무위험수익률 데이터 ──────────────────────────
            rf_success = False
            try:
                # pykrx로 CD(91일) 시도
                cd91 = (bond.get_otc_treasury_yields(start, end_dt.strftime("%Y%m%d"), "CD(91일)")["수익률"]
                       .resample("M").mean() / 100)
                if not cd91.empty and len(cd91) > 0:
                    rf_m = cd91.reindex(mkt_ret_m.index).fillna(method="ffill")
                    rf_success = True
                    print(f"  ✅ pykrx CD(91일) 데이터 성공: {len(cd91)}개월")
                else:
                    raise ValueError("CD(91일) 빈 데이터")
                    
            except Exception as e:
                print(f"  ⚠️ CD(91일) 데이터 실패: {e}")

            # 무위험수익률 실패 시 고정값 사용
            if not rf_success:
                print("  💡 고정 무위험수익률 사용 (연 2.5%)")
                # 한국의 역사적 평균 단기금리 고려 (연 2.5%)
                rf_m = pd.Series(0.025/12, index=mkt_ret_m.index)

            # ── 3. 시장 위험 프리미엄 계산 ──────────────────────────
            mkt_rf_m = (mkt_ret_m - rf_m).dropna()

            # 4/1 ~ 다음 해 3/31 누적 (PeriodIndex freq='A-APR')
            mkt_rf_y = ((1 + mkt_rf_m).groupby(
                pd.PeriodIndex(mkt_rf_m.index, freq="A-APR")).prod() - 1)
            
            print(f"  ✅ 연간 시장 위험 프리미엄 계산 완료: {len(mkt_rf_y)}년")
            return mkt_rf_y.rename("MKT_RF")
            
        except Exception as e:
            print(f"  ❌ 시장 팩터 계산 전체 실패: {e}")
            print("  💡 완전 Mock 데이터로 대체")
            # 완전 Mock 데이터 생성
            years = range(2000, pd.Timestamp.today().year + 1)
            periods = [pd.Period(f"{year}-04", freq="A-APR") for year in years]
            np.random.seed(42)
            # 한국 주식시장 역사적 평균 (연 6% 수익률, 15% 변동성)
            mock_returns = np.random.normal(0.06, 0.15, len(periods))
            return pd.Series(mock_returns, index=periods, name="MKT_RF")


    def _compute_ff3_factors(self):
        """
        ▍진짜 Fama-French 3 factor(연·APR) 계산
        – SMB, HML : 2×3 포트폴리오 수익률 스프레드(동일가중)
        – MKT_RF   : _build_market_factor() 결과
        → self.ff_factors  (index = Period['YYYY-APR'])
        """
        try:
            print("  🔄 FF3 팩터 계산 중...")
            
            # ── 1. 시장 팩터 ───────────────────────────────────────
            mkt_rf_y = self._build_market_factor()

            # ── 2. SMB·HML (연 4/1 리밸런스) ───────────────────────
            factor_rows = []
            rebalance_years = sorted(self.df['연도'].unique())

            for yr in rebalance_years:
                snap = self.df[self.df['연도'] == yr - 1].copy()   # 전년도 재무정보(3월 말 가정)
                if snap[['시가총액', 'bm']].isna().any(axis=None) or len(snap) < 6:
                    continue

                # ① Size·BM 컷
                size_median = snap['시가총액'].median()
                bm30, bm70 = snap['bm'].quantile([.3, .7])

                # ② 6개 포트라벨
                size_grp = np.where(snap['시가총액'] <= size_median, 'S', 'B')
                bm_grp   = np.where(snap['bm'] <= bm30, 'L',
                                    np.where(snap['bm'] > bm70, 'H', 'M'))
                snap['grp'] = [a+b for a, b in zip(size_grp, bm_grp)]

                # ③ 다음 12개월(해당 회계연도) 수익률 ▸ R_{g,yr}
                hold_ret = (self.df.loc[self.df['연도'] == yr,
                                       ['거래소코드', '주가수익률']]
                           .set_index('거래소코드')['주가수익률'])
                snap = snap.join(hold_ret, on='거래소코드').dropna(subset=['주가수익률'])
                if len(snap) < 6:               # 포트별 최소 1종목 확보
                    continue

                port_ret = snap.groupby('grp')['주가수익률'].mean()

                # 필요한 포트폴리오가 모두 존재하는지 확인
                required_portfolios = ['SL', 'SM', 'SH', 'BL', 'BM', 'BH']
                missing_portfolios = [p for p in required_portfolios if p not in port_ret.index]
                
                if missing_portfolios:
                    print(f"    ⚠️ {yr}년: 필요한 포트폴리오 누락 {missing_portfolios}, 건너뜀")
                    continue

                SMB = port_ret[['SL', 'SM', 'SH']].mean() - port_ret[['BL', 'BM', 'BH']].mean()
                HML = port_ret[['SH', 'BH']].mean() - port_ret[['SL', 'BL']].mean()

                factor_rows.append({'연도': yr, 'SMB': SMB, 'HML': HML})

            if factor_rows:
                smb_hml_y = (pd.DataFrame(factor_rows)
                            .set_index(pd.PeriodIndex([r['연도'] for r in factor_rows],
                                                     freq="A-APR")))

                # ── 3. 세 팩터 합치기 ─────────────────────────────────
                self.ff_factors = pd.concat([mkt_rf_y, smb_hml_y], axis=1).dropna()
                print(f"  ✅ FF3 팩터 계산 완료: {len(self.ff_factors)}개 연도")
            else:
                print("  ⚠️ SMB/HML 팩터 계산 실패: 유효한 포트폴리오 없음")
                # Mock FF3 팩터 생성
                years = sorted(self.df['연도'].unique())
                periods = [pd.Period(f"{year}-04", freq="A-APR") for year in years]
                np.random.seed(42)
                mock_data = {
                    'MKT_RF': np.random.normal(0.06, 0.15, len(periods)),
                    'SMB': np.random.normal(0.02, 0.10, len(periods)),
                    'HML': np.random.normal(0.03, 0.12, len(periods))
                }
                self.ff_factors = pd.DataFrame(mock_data, index=periods)
                print("  💡 Mock FF3 팩터로 대체")
                
        except Exception as e:
            print(f"  ❌ FF3 팩터 계산 실패: {e}")
            print("  💡 Mock FF3 팩터로 대체합니다.")
            # 완전 Mock FF3 팩터 생성
            years = sorted(self.df['연도'].unique()) if hasattr(self, 'df') and self.df is not None else range(2013, 2024)
            periods = [pd.Period(f"{year}-04", freq="A-APR") for year in years]
            np.random.seed(42)
            mock_data = {
                'MKT_RF': np.random.normal(0.06, 0.15, len(periods)),
                'SMB': np.random.normal(0.02, 0.10, len(periods)),
                'HML': np.random.normal(0.03, 0.12, len(periods))
            }
            self.ff_factors = pd.DataFrame(mock_data, index=periods)
            
        # FF3 시그널 생성 (간단한 동일가중 조합)
        if hasattr(self, 'ff_factors') and len(self.ff_factors) > 0:
            # 각 연도의 FF3 팩터를 기업 데이터와 매칭
            ff3_signals = []
            for year in self.df['연도'].unique():
                year_period = pd.Period(f"{year}-04", freq="A-APR")
                if year_period in self.ff_factors.index:
                    year_data = self.df[self.df['연도'] == year].copy()
                    if len(year_data) > 0:
                        # 간단한 FF3 시그널: SMB + HML (작은기업 + 가치주 선호)
                        ff3_score = self.ff_factors.loc[year_period, 'SMB'] + self.ff_factors.loc[year_period, 'HML']
                        year_data['ff3_signal'] = ff3_score
                        ff3_signals.append(year_data[['거래소코드', '연도', 'ff3_signal']])
            
            if ff3_signals:
                ff3_df = pd.concat(ff3_signals)
                self.df = self.df.merge(ff3_df, on=['거래소코드', '연도'], how='left')
            else:
                self.df['ff3_signal'] = 0
        else:
            self.df['ff3_signal'] = 0


    
    def build_signal(self, factor_cols, weights=None, winsorize_pct=0.005, 
                    sector_map=None, direction_map=None):
        """SIGNAL BUILDER - 멀티팩터 시그널 구성"""
        print(f"Building composite signal from factors: {factor_cols}")
        
        if weights is None:
            weights = {col: 1.0 for col in factor_cols}
        
        if direction_map is None:
            direction_map = {col: 1 for col in factor_cols}  # 1: 높을수록 좋음, -1: 낮을수록 좋음
        
        signals_by_year = []
        
        for year in self.df['연도'].unique():
            year_data = self.df[self.df['연도'] == year].copy()
            
            if len(year_data) < 20:
                continue
            
            composite_signal = pd.Series(0, index=year_data.index)
            valid_factors = 0
            
            for factor_col in factor_cols:
                if factor_col not in year_data.columns:
                    continue
                
                factor_values = pd.to_numeric(year_data[factor_col], errors='coerce')
                
                if factor_values.notna().sum() < 5:
                    continue
                
                # Winsorization
                lower_bound = factor_values.quantile(winsorize_pct)
                upper_bound = factor_values.quantile(1 - winsorize_pct)
                factor_values = factor_values.clip(lower_bound, upper_bound)
                
                # Z-score standardization
                factor_zscore = (factor_values - factor_values.mean()) / factor_values.std()
                
                # Direction adjustment
                factor_zscore *= direction_map.get(factor_col, 1)
                
                # Weight and add to composite
                weight = weights.get(factor_col, 1.0)
                composite_signal += factor_zscore.fillna(0) * weight
                valid_factors += 1
            
            if valid_factors > 0:
                composite_signal /= valid_factors  # Normalize by number of factors
                
                # Convert to percentile (0~1)
                composite_percentile = composite_signal.rank(pct=True)
                
                # Apply sector adjustment if provided
                if sector_map is not None and 'sector' in year_data.columns:
                    for sector, adjustment in sector_map.items():
                        sector_mask = year_data['sector'] == sector
                        composite_percentile[sector_mask] *= adjustment
                
                # Create MultiIndex Series
                signal_series = pd.Series(
                    composite_percentile.values,
                    index=pd.MultiIndex.from_tuples(
                        [(pd.to_datetime(f'{year}-04-01'), ticker) for ticker in year_data['거래소코드']],
                        names=['date', 'ticker']
                    )
                )
                signals_by_year.append(signal_series)
        
        if signals_by_year:
            return pd.concat(signals_by_year).sort_index()
        else:
            return pd.Series(dtype=float, name='signal')
    
    def construct_long_portfolio(self, df, signal_col, date, top_n=None, signal_df=None):
        """Long-Only Top-N Equal-Weight 포트폴리오 구성"""
        if top_n is None:
            top_n = self.top_n
            
        universe = df[df['rebal_date'] == date].copy()
        
        # 팩터별 특수 필터링 로직
        if signal_col == 'fscore':
            # F-Score의 경우: 최소 점수 이상만 선택
            universe = universe[universe['fscore'] >= self.fscore_min_score]
            # F-Score에서는 조건을 만족하는 모든 종목을 선택 (top_n 제한 없음)
            if len(universe) == 0:
                return pd.Series(dtype=float)
            winners = universe.sort_values(signal_col, ascending=False)
            n = len(winners)
            print(f"  F-Score {self.fscore_min_score}점 이상: {n}개 종목 선택 (요청: {top_n}개)")
        else:
            # 다른 팩터들: 상위 top_n개 종목 선택
            if len(universe) == 0:
                return pd.Series(dtype=float)
            
            # 팩터 값이 높을수록 좋은지 낮을수록 좋은지 판단
            ascending = False
            if signal_col in ['pbr', 'per', 'ev_ebitda', 'debt_to_equity']:
                ascending = True  # 낮을수록 좋은 팩터들
                
            winners = universe.sort_values(signal_col, ascending=ascending).head(top_n)
            n = len(winners)
        
        if n == 0:
            return pd.Series(dtype=float)
        
        # Equal weight
        return pd.Series(1/n, index=winners.index)
    
    def backtest(self, signal_df=None, price_df=None, top_n=30):
        """BACKTEST LOGIC with signal_df support"""
        if signal_df is not None:
            return self._backtest_with_signal(signal_df, price_df, top_n)
        else:
            return self._backtest_original()
    
    def _backtest_with_signal(self, signal_df, price_df, top_n):
        """Backtest using external signal_df"""
        print("📈 백테스트 실행 중 (External Signal)...")
        
        portfolio_returns = []
        rebal_dates = sorted(signal_df.index.get_level_values('date').unique())
        
        for date in rebal_dates:
            # Get signal for this date (use t-1 to prevent look-ahead bias)
            prev_date = date - pd.DateOffset(years=1)
            if prev_date not in signal_df.index.get_level_values('date'):
                continue
                
            date_signals = signal_df.xs(prev_date, level='date').sort_values(ascending=False)
            top_stocks = date_signals.head(top_n)
            
            if len(top_stocks) == 0:
                continue
            
            # Equal weight portfolio
            weights = pd.Series(1/len(top_stocks), index=top_stocks.index)
            
            # Calculate portfolio return (simplified - using price_df if provided)
            if price_df is not None:
                returns = price_df.loc[date, top_stocks.index] if date in price_df.index else 0
                port_ret = (weights * returns).sum() if hasattr(returns, 'sum') else 0
            else:
                port_ret = 0  # Placeholder
            
            portfolio_returns.append({
                'date': date,
                'return': port_ret,
                'n_stocks': len(top_stocks)
            })
        
        if portfolio_returns:
            ret_df = pd.DataFrame(portfolio_returns).set_index('date')
            self.factor_returns['Signal_Strategy'] = ret_df
        
        return self
    
    def _backtest_original(self):
        """4단계: 백테스트 실행 - 부실기업 vs 정상기업 구분"""
        print("📈 백테스트 실행 중...")
        
        # 리밸런싱 날짜 설정 (4월 1일 기준)
        self.df['rebal_date'] = pd.to_datetime(self.df['연도'].astype(str) + '-04-01')
        
        # 다음해 수익률 계산 (t+1 수익률)
        self.df = self.df.sort_values(['거래소코드', '연도'])
        if '주가수익률' in self.df.columns:
            self.df['next_ret'] = self.df.groupby('거래소코드')['주가수익률'].shift(-1)
        else:
            # 종가 기준 수익률
            price_col = '종가' if '종가' in self.df.columns else '시가총액'
            self.df['next_ret'] = self.df.groupby('거래소코드')[price_col].pct_change(periods=-1)
        
        # 2013년 이후 데이터만 사용
        backtest_data = self.df[self.df['연도'] >= 2013].copy()
        
        # 부실/정상 기업 구분
        default_data = backtest_data[backtest_data['default'] == 1].copy()
        normal_data = backtest_data[backtest_data['default'] == 0].copy()
        
        print(f"  📊 백테스트 대상:")
        print(f"     - 부실기업: {len(default_data):,}개 관측치")
        print(f"     - 정상기업: {len(normal_data):,}개 관측치")
        print(f"     - 전체기업: {len(backtest_data):,}개 관측치")
        
        # 백테스트 유효성 검증
        if len(default_data) == 0:
            print("  ⚠️ 부실기업 데이터가 없어 부실기업 백테스트를 건너뜁니다.")
        if len(normal_data) == 0:
            print("  ⚠️ 정상기업 데이터가 없어 정상기업 백테스트를 건너뜁니다.")
        if len(default_data) == len(backtest_data):
            print("  ⚠️ 모든 기업이 부실기업으로 분류되었습니다.")
        if len(normal_data) == len(backtest_data):
            print("  ⚠️ 모든 기업이 정상기업으로 분류되었습니다. 구분 분석이 의미가 없습니다.")
        
        # 팩터 시그널 매핑 (업데이트된 버전)
        factor_signals = {
            'Magic Formula': 'magic_signal',
            'EV/EBITDA': 'ev_ebitda_signal',
            f'Momentum {self.momentum_period}m': 'mom',
            'F-score': 'fscore',
            'QMJ': 'qmj',
            'LowVol': 'lowvol',
            'FF3 Strategy': 'ff3_signal'
        }
        
        # 각 팩터별로 부실기업, 정상기업, 전체기업 백테스트
        for strategy_name, signal_col in factor_signals.items():
            if signal_col not in backtest_data.columns:
                print(f"  ⚠️ {strategy_name} 시그널 컬럼 '{signal_col}' 없음, 건너뜀")
                continue
                
            print(f"  🔄 {strategy_name} 백테스트 중...")
            
            # 1) 정상기업 백테스트 (충분한 데이터가 있을 때만)
            if len(normal_data) > 50:  # 최소 50개 관측치 필요
                self._run_group_backtest(normal_data, signal_col, f"{strategy_name}_정상기업")
            else:
                print(f"    ⚠️ 정상기업 데이터 부족 ({len(normal_data)}개), 건너뜀")
            
            # 2) 부실기업 백테스트 SKIPPED - 정상기업, 전체기업만 분석
            # if len(default_data) > 50:  # 최소 50개 관측치 필요
            #     self._run_group_backtest(default_data, signal_col, f"{strategy_name}_부실기업")
            # else:
            #     print(f"    ⚠️ 부실기업 데이터 부족 ({len(default_data)}개), 건너뜀")
            
            # 3) 전체기업 백테스트 (항상 실행)
            self._run_group_backtest(backtest_data, signal_col, f"{strategy_name}_전체기업")
            
            # 4) 구분 분석이 의미있는지 확인
            if len(normal_data) > 0 and len(default_data) > 0:
                normal_ratio = len(normal_data) / len(backtest_data)
                default_ratio = len(default_data) / len(backtest_data)
                if min(normal_ratio, default_ratio) < 0.05:  # 어느 한 그룹이 5% 미만이면 경고
                    print(f"    ⚠️ 그룹 간 불균형 심함 (정상:{normal_ratio:.1%}, 부실:{default_ratio:.1%})")
        
        print("✅ 백테스트 완료")
        return self
    
    def _run_group_backtest(self, data, signal_col, strategy_name):
        """그룹별 백테스트 실행"""
        portfolio_returns = []
        rebal_dates = sorted(data['rebal_date'].unique())
        
        for date in rebal_dates:
            # 포트폴리오 구성
            weights = self.construct_long_portfolio(data, signal_col, date)
            
            if len(weights) == 0:
                continue
            
            # 다음 기간 수익률 계산
            date_data = data[data['rebal_date'] == date]
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
            
            # 최종 누적수익률 (새로 추가)
            final_cum_return = cum_ret.iloc[-1] - 1  # 누적수익률 (%)
            
            self.performance_stats[strategy_name] = {
                'CAGR': cagr,
                'CumulativeReturn': final_cum_return,  # 누적수익률 추가
                'AnnVol': ann_vol,
                'Sharpe': sharpe,
                'MaxDD': max_dd,
                'Calmar': calmar
            }
        
        # 성과지표 출력 (새로 추가)
        if self.performance_stats:
            print("\n" + "="*80)
            print("📊 성과지표 요약")
            print("="*80)
            
            stats_df = pd.DataFrame(self.performance_stats).T
            
            # 콘솔 출력용: CAGR과 CumulativeReturn을 %로 표시하고 색상 추가
            stats_df_display = stats_df.copy()
            
            # CAGR과 CumulativeReturn을 색상과 부호로 포맷팅
            for col in ['CAGR', 'CumulativeReturn']:
                if col in stats_df_display.columns:
                    formatted_values = []
                    for val in stats_df_display[col] * 100:
                        if val > 0:
                            formatted_values.append(f"\033[91m+{val:.2f}%\033[0m")  # 빨간색 + 양수
                        elif val < 0:
                            formatted_values.append(f"\033[94m{val:.2f}%\033[0m")   # 파란색 + 음수
                        else:
                            formatted_values.append(f"{val:.2f}%")  # 회색 + 0
                    stats_df_display[col] = formatted_values
            
            # 컬럼명을 한글로 변경
            column_mapping = {
                'CAGR': 'CAGR(%)',
                'CumulativeReturn': '누적수익률(%)',
                'AnnVol': '연간변동성',
                'Sharpe': '샤프비율',
                'MaxDD': '최대낙폭',
                'Calmar': '칼마비율'
            }
            stats_df_display = stats_df_display.rename(columns=column_mapping)
            
            # 숫자 컬럼들만 반올림 (이미 포맷된 컬럼 제외)
            numeric_cols = ['연간변동성', '샤프비율', '최대낙폭', '칼마비율']
            for col in numeric_cols:
                if col in stats_df_display.columns:
                    stats_df_display[col] = stats_df_display[col].round(3)
            
            print(stats_df_display)
            
            # 정상기업 vs 전체기업 비교 분석
            self._print_performance_comparison(stats_df_display)
        
        print("✅ 성과지표 계산 완료")
        return self
    
    def _print_performance_comparison(self, stats_df_display):
        """정상기업 vs 전체기업 성과 비교 분석"""
        print("\n" + "="*80)
        print("🔍 정상기업 vs 전체기업 성과 비교")
        print("="*80)
        
        # 전략별로 정상기업과 전체기업 비교
        normal_strategies = [idx for idx in stats_df_display.index if '_정상기업' in idx]
        
        if not normal_strategies:
            print("   정상기업 전략 데이터가 없습니다.")
            return
        
        comparison_data = []
        
        for normal_strategy in normal_strategies:
            factor_name = normal_strategy.replace('_정상기업', '')
            all_strategy = f"{factor_name}_전체기업"
            
            if all_strategy in stats_df_display.index:
                normal_stats = stats_df_display.loc[normal_strategy]
                all_stats = stats_df_display.loc[all_strategy]
                
                # 문자열에서 숫자 추출 (CAGR(%)에서 %와 + 제거)
                def extract_numeric(val_str):
                    if isinstance(val_str, str):
                        # +43.64% 형태에서 숫자만 추출
                        import re
                        numbers = re.findall(r'[-+]?\d*\.?\d+', val_str)
                        return float(numbers[0]) if numbers else 0
                    return float(val_str)
                
                normal_cagr = extract_numeric(normal_stats['CAGR(%)'])
                all_cagr = extract_numeric(all_stats['CAGR(%)'])
                normal_sharpe = extract_numeric(normal_stats['샤프비율'])
                all_sharpe = extract_numeric(all_stats['샤프비율'])
                
                # 주요 지표 비교
                cagr_diff = normal_cagr - all_cagr
                sharpe_diff = normal_sharpe - all_sharpe
                
                comparison_data.append({
                    '전략': factor_name,
                    '정상기업_CAGR(%)': f"{normal_cagr:+.2f}%",
                    '전체기업_CAGR(%)': f"{all_cagr:+.2f}%",
                    'CAGR_차이(%)': f"{cagr_diff:+.2f}%p",
                    '정상기업_샤프': f"{normal_sharpe:.3f}",
                    '전체기업_샤프': f"{all_sharpe:.3f}",
                    '샤프_차이': f"{sharpe_diff:+.3f}"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df)
            
            # 요약 분석을 위한 숫자 데이터 수집
            numeric_data = []
            for normal_strategy in normal_strategies:
                factor_name = normal_strategy.replace('_정상기업', '')
                all_strategy = f"{factor_name}_전체기업"
                
                if all_strategy in stats_df_display.index:
                    normal_stats = stats_df_display.loc[normal_strategy]
                    all_stats = stats_df_display.loc[all_strategy]
                    
                    def extract_numeric(val_str):
                        if isinstance(val_str, str):
                            import re
                            numbers = re.findall(r'[-+]?\d*\.?\d+', val_str)
                            return float(numbers[0]) if numbers else 0
                        return float(val_str)
                    
                    normal_cagr = extract_numeric(normal_stats['CAGR(%)'])
                    all_cagr = extract_numeric(all_stats['CAGR(%)'])
                    normal_sharpe = extract_numeric(normal_stats['샤프비율'])
                    all_sharpe = extract_numeric(all_stats['샤프비율'])
                    
                    numeric_data.append({
                        '전략': factor_name,
                        'CAGR_차이': normal_cagr - all_cagr,
                        '샤프_차이': normal_sharpe - all_sharpe
                    })
            
            if numeric_data:
                numeric_df = pd.DataFrame(numeric_data)
                
                # 요약 분석
                print(f"\n📈 분석 결과:")
                avg_cagr_diff = numeric_df['CAGR_차이'].mean()
                avg_sharpe_diff = numeric_df['샤프_차이'].mean()
                
                print(f"   평균 CAGR 차이: {avg_cagr_diff:.2f}%p ({'정상기업 우세' if avg_cagr_diff > 0 else '전체기업 우세'})")
                print(f"   평균 샤프비율 차이: {avg_sharpe_diff:.3f} ({'정상기업 우세' if avg_sharpe_diff > 0 else '전체기업 우세'})")
                
                # 가장 큰 차이를 보이는 전략
                if len(numeric_df) > 0:
                    max_cagr_diff_idx = numeric_df['CAGR_차이'].abs().idxmax()
                    max_sharpe_diff_idx = numeric_df['샤프_차이'].abs().idxmax()
                    
                    best_cagr_strategy = numeric_df.loc[max_cagr_diff_idx]
                    best_sharpe_strategy = numeric_df.loc[max_sharpe_diff_idx]
            
                    print(f"   CAGR 차이 최대: {best_cagr_strategy['전략']} ({best_cagr_strategy['CAGR_차이']:.2f}%p)")
            print(f"   샤프비율 차이 최대: {best_sharpe_strategy['전략']} ({best_sharpe_strategy['샤프_차이']:.3f})")
    
    def _generate_comparison_html(self, stats_df):
        """정상기업 vs 전체기업 비교표 HTML 생성"""
        normal_strategies = [idx for idx in stats_df.index if '_정상기업' in idx]
        
        if not normal_strategies:
            return ""
        
        comparison_data = []
        
        for normal_strategy in normal_strategies:
            factor_name = normal_strategy.replace('_정상기업', '')
            all_strategy = f"{factor_name}_전체기업"
            
            if all_strategy in stats_df.index:
                normal_stats = stats_df.loc[normal_strategy]
                all_stats = stats_df.loc[all_strategy]
                
                # %로 변환
                normal_cagr = normal_stats['CAGR'] * 100
                all_cagr = all_stats['CAGR'] * 100
                cagr_diff = normal_cagr - all_cagr
                sharpe_diff = normal_stats['Sharpe'] - all_stats['Sharpe']
                
                comparison_data.append({
                    '전략': factor_name,
                    '정상기업_CAGR(%)': f"{normal_cagr:.2f}%",
                    '전체기업_CAGR(%)': f"{all_cagr:.2f}%",
                    'CAGR_차이': f"{cagr_diff:+.2f}%p",
                    '정상기업_샤프': f"{normal_stats['Sharpe']:.3f}",
                    '전체기업_샤프': f"{all_stats['Sharpe']:.3f}",
                    '샤프_차이': f"{sharpe_diff:+.3f}"
                })
        
        if not comparison_data:
            return ""
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return f"""
        <div style="margin-top: 30px;">
            <h3 style="color: #333; font-family: Arial, sans-serif;">🔍 정상기업 vs 전체기업 성과 비교</h3>
            {comparison_df.to_html(classes='table table-striped', table_id='comparison-table', escape=False, index=False)}
        </div>
        """
    
    def plot_results(self):
        """6단계: 시각화"""
        print("📈 시각화 생성 중...")
        
        if not self.factor_returns:
            print("  ⚠️ 백테스트 결과가 없습니다.")
            return self
        
        if not PLOTLY_AVAILABLE:
            return self._plot_results_matplotlib()
        else:
            return self._plot_results_plotly()
    
    def _plot_results_matplotlib(self):
        """Matplotlib을 사용한 시각화 (plotly 대안)"""
        print("  📊 Matplotlib을 사용한 시각화 생성 중...")
        
        # 정상기업 vs 전체기업 비교 차트 생성
        factor_groups = {}
        for strategy_name in self.factor_returns.keys():
            if '_정상기업' in strategy_name:
                factor_name = strategy_name.replace('_정상기업', '')
                if factor_name not in factor_groups:
                    factor_groups[factor_name] = {}
                factor_groups[factor_name]['정상기업'] = strategy_name
            elif '_전체기업' in strategy_name:
                factor_name = strategy_name.replace('_전체기업', '')
                if factor_name not in factor_groups:
                    factor_groups[factor_name] = {}
                factor_groups[factor_name]['전체기업'] = strategy_name
        
        n_factors = len(factor_groups)
        if n_factors == 0:
            print("  ⚠️ 플롯할 팩터 그룹이 없습니다.")
            return self
        
        # 서브플롯 생성
        cols = 2
        rows = (n_factors + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # 사용하지 않는 서브플롯 숨기기
        for i in range(n_factors, len(axes)):
            axes[i].set_visible(False)
        
        colors = {'정상기업': 'blue', '전체기업': 'gray'}
        
        for i, (factor_name, group_strategies) in enumerate(factor_groups.items()):
            ax = axes[i]
            
            # 각 그룹별 누적수익률 차트 추가
            for group_name, strategy_name in group_strategies.items():
                if strategy_name in self.factor_returns:
                    returns_df = self.factor_returns[strategy_name]
                    if len(returns_df) > 0:
                        cum_ret = (1 + returns_df['return']).cumprod()
                        ax.plot(cum_ret.index, cum_ret.values, 
                               label=f"{group_name}", 
                               color=colors.get(group_name, 'black'),
                               linewidth=2)
            
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='기준선 (0%)')
            ax.set_title(factor_name)
            ax.set_xlabel('연도')
            ax.set_ylabel('누적수익률')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 파일 저장
        output_path = os.path.join(self.output_dir, 'factor_performance_charts.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  📊 차트 저장: {output_path}")
        
        # plt.show() 주석 처리 - 콘솔 출력 방지
        # plt.show()
        plt.close()  # 메모리 정리
        
        return self
    
    def _plot_results_plotly(self):
        """Plotly를 사용한 시각화"""
        # 정상기업 vs 전체기업 비교 차트 생성
        factor_groups = {}
        for strategy_name in self.factor_returns.keys():
            if '_정상기업' in strategy_name:
                factor_name = strategy_name.replace('_정상기업', '')
                if factor_name not in factor_groups:
                    factor_groups[factor_name] = {}
                factor_groups[factor_name]['정상기업'] = strategy_name
            elif '_전체기업' in strategy_name:
                factor_name = strategy_name.replace('_전체기업', '')
                if factor_name not in factor_groups:
                    factor_groups[factor_name] = {}
                factor_groups[factor_name]['전체기업'] = strategy_name
        
        n_factors = len(factor_groups)
        if n_factors == 0:
            return self
        
        cols = 2
        rows = (n_factors + cols - 1) // cols
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=list(factor_groups.keys()),
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        colors = {'정상기업': 'blue', '전체기업': 'gray'}
        
        for i, (factor_name, group_strategies) in enumerate(factor_groups.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # 각 그룹별 누적수익률 차트 추가
            for group_name, strategy_name in group_strategies.items():
                if strategy_name in self.factor_returns:
                    returns_df = self.factor_returns[strategy_name]
                    if len(returns_df) > 0:
                        cum_ret = (1 + returns_df['return']).cumprod()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=cum_ret.index,
                                y=cum_ret.values,
                                name=f"{factor_name}_{group_name}",
                                line=dict(color=colors.get(group_name, 'black'), width=2),
                                showlegend=(i == 0)
                            ),
                            row=row, col=col
                        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title="팩터별 정상기업 vs 전체기업 누적수익률 비교 (FF3 통합 버전)",
            font=dict(family='AppleGothic'),
            height=350 * rows,
            showlegend=True
        )
        
        # y축 제목 업데이트
        fig.update_yaxes(title_text="누적수익률")
        
        # 파일 저장
        output_path = os.path.join(self.output_dir, 'factor_performance_charts.html')
        fig.write_html(output_path)
        
        # 성과지표 테이블을 HTML에 추가
        if self.performance_stats:
            import pandas as pd
            stats_df = pd.DataFrame(self.performance_stats).T
            
            # CAGR과 CumulativeReturn을 %로 변환
            stats_df_formatted = stats_df.copy()
            if 'CAGR' in stats_df_formatted.columns:
                stats_df_formatted['CAGR'] = (stats_df_formatted['CAGR'] * 100).round(2).astype(str) + '%'
            if 'CumulativeReturn' in stats_df_formatted.columns:
                stats_df_formatted['CumulativeReturn'] = (stats_df_formatted['CumulativeReturn'] * 100).round(2).astype(str) + '%'
            
            # HTML 파일 읽기
            with open(output_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # 컬럼명을 한글로 변경 (HTML용)
            column_mapping = {
                'CAGR': 'CAGR',
                'CumulativeReturn': '누적수익률',
                'AnnVol': '연간변동성',
                'Sharpe': '샤프비율',
                'MaxDD': '최대낙폭',
                'Calmar': '칼마비율'
            }
            stats_df_formatted = stats_df_formatted.rename(columns=column_mapping)
            
            # 정상기업 vs 전체기업 비교표 생성
            comparison_html = self._generate_comparison_html(stats_df)
            
            # 성과지표 테이블 HTML 생성
            stats_table_html = f"""
            <div style="margin-top: 50px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
                <h2 style="color: #333; font-family: Arial, sans-serif;">📊 성과지표 요약</h2>
                {stats_df_formatted.to_html(classes='table table-striped', table_id='performance-table', escape=False)}
                
                {comparison_html}
                <style>
                    #performance-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 15px;
                        font-family: Arial, sans-serif;
                    }}
                    #performance-table th, #performance-table td {{
                        padding: 8px 12px;
                        text-align: right;
                        border: 1px solid #ddd;
                    }}
                    #performance-table th {{
                        background-color: #e9ecef;
                        font-weight: bold;
                    }}
                    #performance-table tr:nth-child(even) {{
                        background-color: #f8f9fa;
                    }}
                </style>
            </div>
            """
            
            # HTML에 테이블 추가
            html_content = html_content.replace('</body>', f'{stats_table_html}</body>')
            
            # 수정된 HTML 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        print(f"  📊 차트 및 성과지표 저장: {output_path}")
        
        # fig.show() 주석 처리 - 콘솔에 HTML 코드 출력 방지
        # fig.show()
        
        print("✅ 시각화 완료")
        return self
    
    def save_results(self):
        """7단계: 결과 저장"""
        print("💾 결과 저장 중...")
        
        # 피처 데이터 저장 (FS2_features.csv) - data/processed에 저장
        os.makedirs('data/processed', exist_ok=True)
        feature_output_path = 'data/processed/FS2_backtesting.csv'
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
        print("📊 FACTOR BACKTESTING SUMMARY (Updated Version)")
        print("="*60)
        print("🎯 전략 구성 (7개):")
        print("   1. Magic Formula")
        print("   2. EV/EBITDA") 
        print("   3. Momentum")
        print("   4. F-score")
        print("   5. QMJ")
        print("   6. Low Volatility")
        print("   7. FF3 Strategy (통합) ⭐")
        
        if self.performance_stats:
            print("\n📈 성과 요약:")
            for strategy, stats in self.performance_stats.items():
                if 'FF3' in strategy:
                    print(f"\n🎯 {strategy}")
                    print(f"   CAGR: {stats['CAGR']:.2%}")
                    print(f"   샤프비율: {stats['Sharpe']:.3f}")
                    print(f"   최대낙폭: {stats['MaxDD']:.2%}")
                    print(f"   칼마비율: {stats['Calmar']:.3f}")
        
        return self

    def build_ff3_factors(self, start_date, end_date, smb_series, hml_series):
        """FF-3 FACTOR BUILDER (monthly → annual)"""
        print(f"Building FF3 factors from {start_date} to {end_date}")
        
        # Convert dates for pykrx API calls
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        if PYKRX_AVAILABLE:
            # KOSPI price index (monthly close)
            kospi_data = stock.get_index_ohlcv(
                start_dt.strftime('%Y%m%d'), 
                end_dt.strftime('%Y%m%d'), 
                "1001"
            )["종가"]
            
            # CD(91일) daily yields
            cd_data = bond.get_otc_treasury_yields(
                start_dt.strftime('%Y%m%d'),
                end_dt.strftime('%Y%m%d'), 
                "CD(91일)"
            )["수익률"] / 100
        else:
            # Mock data for testing when pykrx is not available
            print("Using mock data for testing (pykrx not available)")
            date_range = pd.date_range(start_dt, end_dt, freq='M')
            kospi_data = pd.Series(
                np.random.normal(0.01, 0.05, len(date_range)) * 100 + 2000,
                index=date_range
            )
            cd_data = pd.Series(
                np.random.normal(0.02, 0.01, len(date_range)),
                index=date_range
            )
        
        # Convert to monthly data (end of month)
        kospi_monthly = kospi_data.resample('M').last().pct_change()
        cd_monthly = cd_data.resample('M').last()
        
        # Align all series to same monthly index
        common_idx = kospi_monthly.index.intersection(cd_monthly.index)
        common_idx = common_idx.intersection(smb_series.index)
        common_idx = common_idx.intersection(hml_series.index)
        
        kospi_monthly = kospi_monthly.reindex(common_idx)
        cd_monthly = cd_monthly.reindex(common_idx)
        smb_monthly = smb_series.reindex(common_idx)
        hml_monthly = hml_series.reindex(common_idx)
        
        # Create annual periods (Apr-Mar)
        monthly_df = pd.DataFrame({
            'MKT': kospi_monthly,
            'RF': cd_monthly,
            'SMB': smb_monthly,
            'HML': hml_monthly
        })
        
        # Group by annual periods (Apr-Mar)
        monthly_df.index = pd.to_datetime(monthly_df.index)
        annual_groups = monthly_df.groupby(pd.Grouper(freq='A-APR'))
        
        # Calculate annual cumulative returns
        annual_factors = []
        for period, group in annual_groups:
            if len(group) >= 6:  # Minimum 6 months of data
                mkt_annual = (1 + group['MKT'].fillna(0)).prod() - 1
                rf_annual = (1 + group['RF'].fillna(0)).prod() - 1
                smb_annual = (1 + group['SMB'].fillna(0)).prod() - 1
                hml_annual = (1 + group['HML'].fillna(0)).prod() - 1
                
                annual_factors.append({
                    'period': period,
                    'MKT_RF': mkt_annual - rf_annual,
                    'SMB': smb_annual,
                    'HML': hml_annual
                })
        
        ff3_df = pd.DataFrame(annual_factors).set_index('period')
        ff3_df.index = pd.PeriodIndex(ff3_df.index, freq='A-APR')
        
        return ff3_df

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Factor Backtesting - Updated Version with FF3 Integration')
    parser.add_argument('--data_path', type=str, default='data/processed', 
                       help='Data directory path (default: data/processed)')
    parser.add_argument('--output_dir', type=str, default='outputs/backtesting',
                       help='Output directory path (default: outputs/backtesting)')
    parser.add_argument('--top_n', '-t', type=int, default=10,
                       help='Number of top stocks to select (default: 10)')
    parser.add_argument('--fscore_min_score', '-f', type=int, default=8,
                       help='Minimum F-Score for selection (default: 8)')
    parser.add_argument('--momentum_period', '-m', type=int, default=12,
                       help='Momentum period in months (default: 12)')
    
    args = parser.parse_args()
    
    print(f"🚀 Factor Investing Backtesting 시작 (Updated with FF3 Integration)")
    print(f"📊 F-Score 최소 점수: {args.fscore_min_score}점")
    print(f"📈 모멘텀 기간: {args.momentum_period}개월")
    print("="*60)
    
    # 백테스터 초기화 및 실행
    backtester = FactorBacktester(
        data_path=args.data_path, 
        output_dir=args.output_dir,
        top_n=args.top_n,
        fscore_min_score=args.fscore_min_score,
        momentum_period=args.momentum_period
    )
    
    backtester.load_data() \
              .compute_features() \
              .compute_factor_signals() \
              .backtest() \
              .calc_performance_stats() \
              .plot_results() \
              .save_results()
    
    print("\n🎉 백테스트 완료!")
    return backtester

def example_ff3():
    """FF3 팩터 사용 예시"""
    print("FF3 Factor Builder Example")
    
    # 예시: 월간 SMB, HML 시리즈 로드 (CSV 파일에서)
    try:
        sm = pd.read_csv('smb.csv', index_col=0).iloc[:, 0]  # squeeze=True 대신
        hm = pd.read_csv('hml.csv', index_col=0).iloc[:, 0]  # squeeze=True 대신
        
        # FF3 팩터 구축
        backtester = FactorBacktester()
        ff3 = backtester.build_ff3_factors("2000-01", "2025-06", sm, hm)
        print(ff3.tail())
    except FileNotFoundError:
        print("SMB/HML CSV files not found. Please provide monthly factor series.")

if __name__ == "__main__":
    results = main()