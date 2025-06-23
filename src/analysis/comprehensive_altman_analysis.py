"""
종합적인 Altman Z-Score 및 K2-Score 분석
FS_flow.csv와 FS_ratio_flow.csv를 결합하여 모든 점수 계산

작성자: AI Assistant
작성일: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveAltmanAnalyzer:
    def __init__(self, fs_flow_path, fs_ratio_path, output_dir):
        """
        종합적인 Altman 분석기 초기화
        
        Args:
            fs_flow_path: FS_flow.csv 파일 경로
            fs_ratio_path: FS_ratio_flow.csv 파일 경로
            output_dir: 결과 저장 디렉토리
        """
        self.fs_flow_path = fs_flow_path
        self.fs_ratio_path = fs_ratio_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 시각화 디렉토리도 생성
        self.viz_dir = Path("outputs/visualizations")
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        self.fs_flow = None
        self.fs_ratio = None
        self.combined_data = None
        
    def load_data(self):
        """데이터 로드 및 전처리"""
        print("데이터 로딩 중...")
        
        # FS_flow.csv 로드
        self.fs_flow = pd.read_csv(self.fs_flow_path)
        print(f"FS_flow 데이터: {len(self.fs_flow):,}개 레코드, {len(self.fs_flow.columns)}개 컬럼")
        
        # FS_ratio_flow.csv 로드
        self.fs_ratio = pd.read_csv(self.fs_ratio_path)
        print(f"FS_ratio 데이터: {len(self.fs_ratio):,}개 레코드, {len(self.fs_ratio.columns)}개 컬럼")
        
        # 데이터 결합 (거래소코드와 회계년도 기준)
        self.combined_data = pd.merge(
            self.fs_flow, 
            self.fs_ratio, 
            on=['거래소코드', '회계년도'], 
            how='inner',
            suffixes=('_flow', '_ratio')
        )
        print(f"결합된 데이터: {len(self.combined_data):,}개 레코드")
        
        return self.combined_data
    
    def calculate_original_altman_zscore(self, df):
        """
        Original Altman Z-Score 계산 (1968)
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets  
        X3 = EBIT / Total Assets
        X4 = Market Value of Equity / Total Liabilities
        X5 = Sales / Total Assets
        """
        print("Original Altman Z-Score 계산 중...")
        
        # 필요한 변수들 계산
        working_capital = df['유동자산_당기말'] - df['유동부채_당기말']
        total_assets = df['자산_당기말']
        retained_earnings = df['이익잉여금_당기말']
        ebit = df['영업손익']  # 영업이익을 EBIT로 사용
        total_liabilities = df['부채_당기말']
        sales = df['매출액']
        
        # 시가총액 계산 (발행주식수 * 주가는 없으므로 기업가치 사용)
        market_value_equity = df['기업가치']  # 또는 다른 방법으로 계산
        
        # 각 비율 계산
        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_value_equity / total_liabilities
        x5 = sales / total_assets
        
        # Z-Score 계산
        z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
        
        return z_score, {
            'X1_WC_TA': x1,
            'X2_RE_TA': x2, 
            'X3_EBIT_TA': x3,
            'X4_MVE_TL': x4,
            'X5_S_TA': x5
        }
    
    def calculate_modified_altman_zscore(self, df):
        """
        Modified Altman Z-Score 계산 (1983, 비상장기업용)
        Z = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
        
        X4 = Book Value of Equity / Total Liabilities (시가 대신 장부가)
        """
        print("Modified Altman Z-Score 계산 중...")
        
        # 필요한 변수들 계산
        working_capital = df['유동자산_당기말'] - df['유동부채_당기말']
        total_assets = df['자산_당기말']
        retained_earnings = df['이익잉여금_당기말']
        ebit = df['영업손익']
        total_liabilities = df['부채_당기말']
        sales = df['매출액']
        book_value_equity = df['자본_당기말']
        
        # 각 비율 계산
        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = book_value_equity / total_liabilities
        x5 = sales / total_assets
        
        # Z-Score 계산
        z_score = 0.717*x1 + 0.847*x2 + 3.107*x3 + 0.420*x4 + 0.998*x5
        
        return z_score, {
            'X1_WC_TA': x1,
            'X2_RE_TA': x2,
            'X3_EBIT_TA': x3,
            'X4_BVE_TL': x4,
            'X5_S_TA': x5
        }
    
    def calculate_emerging_market_zscore(self, df):
        """
        Emerging Market Z-Score 계산 (신흥시장용)
        Z = 3.25 + 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
        """
        print("Emerging Market Z-Score 계산 중...")
        
        # 필요한 변수들 계산
        working_capital = df['유동자산_당기말'] - df['유동부채_당기말']
        total_assets = df['자산_당기말']
        retained_earnings = df['이익잉여금_당기말']
        ebit = df['영업손익']
        sales = df['매출액']
        
        # 각 비율 계산
        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = sales / total_assets
        
        # Z-Score 계산
        z_score = 3.25 + 6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4
        
        return z_score, {
            'X1_WC_TA': x1,
            'X2_RE_TA': x2,
            'X3_EBIT_TA': x3,
            'X4_S_TA': x4
        }
    
    def calculate_k2_scores(self, df):
        """K2-Score 계산 (기존 비율 데이터 활용)"""
        print("K2-Score 계산 중...")
        
        # Original K2-Score
        k2_original = (
            1.28 * df['WC_TA'] +
            0.18 * df['RE_TA'] +
            16.72 * df['EBIT_TA'] +
            0.12 * df['MVE_TL'] +
            0.39 * df['S_TA']
        )
        
        # Alternative K2-Score (CFO 기반)
        k2_alternative = (
            1.28 * df['WC_TA'] +
            0.18 * df['RE_TA'] +
            16.72 * df['CFO_TA'] +  # EBIT 대신 CFO 사용
            0.12 * df['MVE_TL'] +
            0.39 * df['S_TA']
        )
        
        return k2_original, k2_alternative
    
    def calculate_all_scores(self):
        """모든 점수 계산"""
        if self.combined_data is None:
            raise ValueError("데이터를 먼저 로드해주세요.")
        
        df = self.combined_data.copy()
        
        # 무한값과 NaN 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Altman Z-Scores 계산
        z_original, original_components = self.calculate_original_altman_zscore(df)
        z_modified, modified_components = self.calculate_modified_altman_zscore(df)
        z_emerging, emerging_components = self.calculate_emerging_market_zscore(df)
        
        # K2-Scores 계산
        k2_original, k2_alternative = self.calculate_k2_scores(df)
        
        # 결과 데이터프레임에 추가
        df['Altman_Z_Original'] = z_original
        df['Altman_Z_Modified'] = z_modified
        df['Altman_Z_Emerging'] = z_emerging
        df['K2_Score_Original'] = k2_original
        df['K2_Score_Alternative'] = k2_alternative
        
        # 컴포넌트들도 추가
        for key, value in original_components.items():
            df[f'Original_{key}'] = value
        for key, value in modified_components.items():
            df[f'Modified_{key}'] = value
        for key, value in emerging_components.items():
            df[f'Emerging_{key}'] = value
        
        self.combined_data = df
        return df
    
    def analyze_score_distributions(self):
        """점수 분포 분석"""
        if self.combined_data is None:
            raise ValueError("점수를 먼저 계산해주세요.")
        
        score_columns = [
            'Altman_Z_Original', 'Altman_Z_Modified', 'Altman_Z_Emerging',
            'K2_Score_Original', 'K2_Score_Alternative'
        ]
        
        print("\n=== 점수 분포 통계 ===")
        for col in score_columns:
            if col in self.combined_data.columns:
                valid_data = self.combined_data[col].dropna()
                if len(valid_data) > 0:
                    print(f"\n{col}:")
                    print(f"  유효 데이터: {len(valid_data):,}개")
                    print(f"  평균: {valid_data.mean():.4f}")
                    print(f"  표준편차: {valid_data.std():.4f}")
                    print(f"  최솟값: {valid_data.min():.4f}")
                    print(f"  최댓값: {valid_data.max():.4f}")
                    print(f"  25%: {valid_data.quantile(0.25):.4f}")
                    print(f"  50%: {valid_data.quantile(0.50):.4f}")
                    print(f"  75%: {valid_data.quantile(0.75):.4f}")
    
    def create_visualizations(self):
        """시각화 생성"""
        if self.combined_data is None:
            raise ValueError("점수를 먼저 계산해주세요.")
        
        score_columns = [
            'Altman_Z_Original', 'Altman_Z_Modified', 'Altman_Z_Emerging',
            'K2_Score_Original', 'K2_Score_Alternative'
        ]
        
        # 1. 점수 분포 히스토그램
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(score_columns):
            if col in self.combined_data.columns:
                valid_data = self.combined_data[col].dropna()
                if len(valid_data) > 0:
                    # 이상치 제거 (1%, 99% 분위수 기준)
                    q1, q99 = valid_data.quantile([0.01, 0.99])
                    filtered_data = valid_data[(valid_data >= q1) & (valid_data <= q99)]
                    
                    axes[i].hist(filtered_data, bins=50, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{col} 분포\n(유효 데이터: {len(valid_data):,}개)')
                    axes[i].set_xlabel('점수')
                    axes[i].set_ylabel('빈도')
                    axes[i].grid(True, alpha=0.3)
        
        # 빈 subplot 제거
        if len(score_columns) < len(axes):
            axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'comprehensive_score_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 점수 간 상관관계
        score_data = self.combined_data[score_columns].dropna()
        if len(score_data) > 0:
            plt.figure(figsize=(10, 8))
            correlation_matrix = score_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.3f', square=True)
            plt.title('점수 간 상관관계')
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'score_correlations.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Altman Z-Score 해석 기준별 분포
        if 'Altman_Z_Original' in self.combined_data.columns:
            z_original = self.combined_data['Altman_Z_Original'].dropna()
            
            # 해석 기준
            safe_zone = (z_original > 2.99).sum()
            grey_zone = ((z_original >= 1.81) & (z_original <= 2.99)).sum()
            distress_zone = (z_original < 1.81).sum()
            
            plt.figure(figsize=(10, 6))
            categories = ['안전구간\n(Z > 2.99)', '회색구간\n(1.81 ≤ Z ≤ 2.99)', '위험구간\n(Z < 1.81)']
            counts = [safe_zone, grey_zone, distress_zone]
            colors = ['green', 'orange', 'red']
            
            bars = plt.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
            plt.title('Original Altman Z-Score 구간별 기업 분포')
            plt.ylabel('기업 수')
            
            # 막대 위에 숫자 표시
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{count:,}개\n({count/len(z_original)*100:.1f}%)',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'altman_zone_distribution.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self):
        """결과 저장"""
        if self.combined_data is None:
            raise ValueError("점수를 먼저 계산해주세요.")
        
        # 점수만 포함된 결과 저장
        score_columns = [
            '거래소코드', '회계년도', '연도', '회사명_flow',
            'Altman_Z_Original', 'Altman_Z_Modified', 'Altman_Z_Emerging',
            'K2_Score_Original', 'K2_Score_Alternative'
        ]
        
        # 컴포넌트 컬럼들도 추가
        component_columns = [col for col in self.combined_data.columns 
                           if col.startswith(('Original_', 'Modified_', 'Emerging_'))]
        
        all_columns = score_columns + component_columns
        result_df = self.combined_data[all_columns].copy()
        
        # 결과 저장
        output_file = self.output_dir / 'comprehensive_altman_k2_scores.csv'
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n결과 저장: {output_file}")
        
        # 요약 통계 저장
        summary_stats = {}
        for col in ['Altman_Z_Original', 'Altman_Z_Modified', 'Altman_Z_Emerging',
                   'K2_Score_Original', 'K2_Score_Alternative']:
            if col in result_df.columns:
                valid_data = result_df[col].dropna()
                if len(valid_data) > 0:
                    summary_stats[col] = {
                        'count': len(valid_data),
                        'mean': valid_data.mean(),
                        'std': valid_data.std(),
                        'min': valid_data.min(),
                        'max': valid_data.max(),
                        'q25': valid_data.quantile(0.25),
                        'q50': valid_data.quantile(0.50),
                        'q75': valid_data.quantile(0.75)
                    }
        
        summary_df = pd.DataFrame(summary_stats).T
        summary_file = self.output_dir / 'score_summary_statistics.csv'
        summary_df.to_csv(summary_file, encoding='utf-8-sig')
        print(f"요약 통계 저장: {summary_file}")
        
        return result_df

def main():
    """메인 실행 함수"""
    # 경로 설정
    fs_flow_path = "data/processed/FS_flow.csv"
    fs_ratio_path = "data/final/FS_ratio_flow.csv"
    output_dir = "outputs/reports"
    
    # 분석기 초기화
    analyzer = ComprehensiveAltmanAnalyzer(fs_flow_path, fs_ratio_path, output_dir)
    
    try:
        # 데이터 로드
        combined_data = analyzer.load_data()
        
        # 모든 점수 계산
        scored_data = analyzer.calculate_all_scores()
        
        # 분포 분석
        analyzer.analyze_score_distributions()
        
        # 시각화 생성
        analyzer.create_visualizations()
        
        # 결과 저장
        result_df = analyzer.save_results()
        
        print(f"\n=== 종합 분석 완료 ===")
        print(f"총 데이터: {len(result_df):,}개")
        print(f"결과 파일: outputs/reports/comprehensive_altman_k2_scores.csv")
        print(f"시각화 파일: outputs/visualizations/ 디렉토리")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        raise

if __name__ == "__main__":
    main() 