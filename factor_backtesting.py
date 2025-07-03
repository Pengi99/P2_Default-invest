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
        """3단계: 11개 팩터 시그널 계산"""
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
        
        # 9. DOL & DFL (Leverage)
        self._compute_leverage_factors()
        
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
        """4단계: 백테스트 실행 - 부실기업 vs 정상기업 구분"""
        print("📈 백테스트 실행 중...")
        
        # 리밸런싱 날짜 설정 (연말 기준)
        self.df['rebal_date'] = pd.to_datetime(self.df['연도'].astype(str) + '-12-31')
        
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
            
            # 2) 부실기업 백테스트 (충분한 데이터가 있을 때만)
            if len(default_data) > 50:  # 최소 50개 관측치 필요
                self._run_group_backtest(default_data, signal_col, f"{strategy_name}_부실기업")
            else:
                print(f"    ⚠️ 부실기업 데이터 부족 ({len(default_data)}개), 건너뜀")
            
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
            weights = self.construct_long_portfolio(data, signal_col, date, top_n=10)
            
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
        
        # 부실기업 vs 정상기업 비교 차트 생성
        # 팩터별로 그룹화
        factor_groups = {}
        for strategy_name in self.factor_returns.keys():
            # 전략명에서 팩터명 추출 (예: "Magic Formula_정상기업" -> "Magic Formula")
            if '_정상기업' in strategy_name:
                factor_name = strategy_name.replace('_정상기업', '')
                if factor_name not in factor_groups:
                    factor_groups[factor_name] = {}
                factor_groups[factor_name]['정상기업'] = strategy_name
            elif '_부실기업' in strategy_name:
                factor_name = strategy_name.replace('_부실기업', '')
                if factor_name not in factor_groups:
                    factor_groups[factor_name] = {}
                factor_groups[factor_name]['부실기업'] = strategy_name
            elif '_전체기업' in strategy_name:
                factor_name = strategy_name.replace('_전체기업', '')
                if factor_name not in factor_groups:
                    factor_groups[factor_name] = {}
                factor_groups[factor_name]['전체기업'] = strategy_name
        
        # 팩터별 서브플롯 생성
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
        
        # 색상 설정
        colors = {'정상기업': 'blue', '부실기업': 'red', '전체기업': 'gray'}
        
        for i, (factor_name, group_strategies) in enumerate(factor_groups.items()):
            # 행, 열 위치 계산
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
                                line=dict(color=colors.get(group_name, 'black')),
                                showlegend=(i == 0)  # 첫 번째 차트에만 범례 표시
                            ),
                            row=row, col=col
                        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title="팩터별 부실기업 vs 정상기업 누적수익률 비교",
            font=dict(family='AppleGothic'),
            height=300 * rows,
            showlegend=True
        )
        
        # X축, Y축 라벨 설정
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                fig.update_xaxes(title_text="연도", row=i, col=j)
                fig.update_yaxes(title_text="누적수익률", row=i, col=j)
        
        # 차트 표시 및 저장
        output_path = os.path.join(self.output_dir, 'factor_performance_charts.html')
        fig.write_html(output_path)
        print(f"  📊 차트 저장: {output_path}")
        
        fig.show()
        
        # 그룹별 성과지표 비교 출력
        if self.performance_stats:
            print("\n📊 부실기업 vs 정상기업 성과 비교:")
            print("=" * 100)
            
            # 팩터별로 그룹화하여 출력
            for factor_name in factor_groups.keys():
                print(f"\n🎯 {factor_name}")
                print("-" * 60)
                
                # 해당 팩터의 각 그룹별 성과지표 출력
                for group_name in ['정상기업', '부실기업', '전체기업']:
                    strategy_name = f"{factor_name}_{group_name}"
                    if strategy_name in self.performance_stats:
                        stats = self.performance_stats[strategy_name]
                        print(f"   {group_name:>6}: CAGR {stats['CAGR']:>7.2%} | "
                              f"변동성 {stats['AnnVol']:>6.2%} | "
                              f"샤프 {stats['Sharpe']:>6.3f} | "
                              f"MDD {stats['MaxDD']:>7.2%} | "
                              f"칼마 {stats['Calmar']:>6.3f}")
            
            # 전체 성과지표 테이블
            stats_df = pd.DataFrame(self.performance_stats).T
            print(f"\n📋 전체 성과지표 요약표:")
            print(stats_df.round(4))
        
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