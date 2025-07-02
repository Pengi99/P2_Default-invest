#!/usr/bin/env python3
"""
다중공선성 분석 스크립트
FS_100 데이터셋의 다중공선성을 분석하고 결과를 저장합니다.

작성자: AI Assistant
목적: 재무비율 간 다중공선성 확인 및 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings
from datetime import datetime
import os
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class MulticollinearityAnalyzer:
    """다중공선성 분석 클래스"""
    
    def __init__(self, data_path):
        """
        초기화
        
        Args:
            data_path (str): 데이터 파일 경로
        """
        self.data_path = data_path
        self.data = None
        self.numeric_columns = None
        self.correlation_matrix = None
        self.vif_results = None
        self.condition_indices = None
        
    def load_data(self):
        """데이터 로드 및 전처리"""
        print("데이터 로딩 중...")
        self.data = pd.read_csv(self.data_path, encoding='utf-8')
        
        # 수치형 컬럼만 선택 (회사명, 거래소코드, 회계년도, default 제외)
        exclude_columns = ['회사명', '거래소코드', '회계년도', 'default']
        self.numeric_columns = [col for col in self.data.columns 
                               if col not in exclude_columns]
        
        print(f"분석 대상 컬럼 수: {len(self.numeric_columns)}")
        print(f"총 관측치 수: {len(self.data)}")
        
        # 결측치 확인
        missing_counts = self.data[self.numeric_columns].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"결측치 발견: {missing_counts.sum()}개")
            print("결측치가 있는 컬럼:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count}개")
                
    def calculate_correlation_matrix(self):
        """상관계수 행렬 계산"""
        print("상관계수 행렬 계산 중...")
        
        # 결측치 제거
        clean_data = self.data[self.numeric_columns].dropna()
        
        # 상관계수 계산
        self.correlation_matrix = clean_data.corr()
        
        # 높은 상관관계 쌍 찾기 (절댓값 0.8 이상)
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_val = self.correlation_matrix.iloc[i, j]
                if abs(corr_val) >= 0.8:
                    high_corr_pairs.append({
                        'variable1': self.correlation_matrix.columns[i],
                        'variable2': self.correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        print(f"높은 상관관계 쌍 (|r| >= 0.8): {len(high_corr_pairs)}개")
        
        return high_corr_pairs
    
    def calculate_vif(self):
        """VIF (Variance Inflation Factor) 계산"""
        print("VIF 계산 중...")
        
        # 결측치 제거
        clean_data = self.data[self.numeric_columns].dropna()
        
        # 무한값 제거
        clean_data = clean_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) == 0:
            print("VIF 계산을 위한 유효한 데이터가 없습니다.")
            return None
            
        vif_data = []
        
        try:
            # 상수항 추가
            X = add_constant(clean_data)
            
            # 각 변수에 대해 VIF 계산
            for i, col in enumerate(clean_data.columns):
                try:
                    vif_value = variance_inflation_factor(X.values, i+1)  # +1 for constant
                    vif_data.append({
                        'Variable': col,
                        'VIF': vif_value
                    })
                except Exception as e:
                    print(f"VIF 계산 오류 ({col}): {e}")
                    vif_data.append({
                        'Variable': col,
                        'VIF': np.nan
                    })
                    
        except Exception as e:
            print(f"VIF 계산 중 오류: {e}")
            return None
            
        self.vif_results = pd.DataFrame(vif_data)
        self.vif_results = self.vif_results.sort_values('VIF', ascending=False)
        
        # VIF 해석 기준 추가
        def interpret_vif(vif):
            if pd.isna(vif):
                return '계산불가'
            elif vif < 5:
                return '낮음'
            elif vif < 10:
                return '보통'
            else:
                return '높음'
                
        self.vif_results['VIF_해석'] = self.vif_results['VIF'].apply(interpret_vif)
        
        print(f"VIF 10 이상 변수: {len(self.vif_results[self.vif_results['VIF'] >= 10])}개")
        
        return self.vif_results
    
    def calculate_condition_indices(self):
        """조건지수 계산"""
        print("조건지수 계산 중...")
        
        # 결측치 제거
        clean_data = self.data[self.numeric_columns].dropna()
        clean_data = clean_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) == 0:
            print("조건지수 계산을 위한 유효한 데이터가 없습니다.")
            return None
            
        try:
            # 상관계수 행렬의 고유값 계산
            eigenvalues = np.linalg.eigvals(clean_data.corr().values)
            eigenvalues = eigenvalues[eigenvalues > 0]  # 양수 고유값만
            
            # 조건지수 계산
            max_eigenvalue = np.max(eigenvalues)
            condition_indices = np.sqrt(max_eigenvalue / eigenvalues)
            
            self.condition_indices = {
                'eigenvalues': eigenvalues,
                'condition_indices': condition_indices,
                'max_condition_index': np.max(condition_indices)
            }
            
            print(f"최대 조건지수: {self.condition_indices['max_condition_index']:.2f}")
            
            return self.condition_indices
            
        except Exception as e:
            print(f"조건지수 계산 중 오류: {e}")
            return None
    
    def create_correlation_heatmap(self, save_path):
        """상관계수 히트맵 생성"""
        plt.figure(figsize=(20, 16))
        
        # 마스크 생성 (상삼각형 숨기기)
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        
        # 히트맵 생성
        sns.heatmap(self.correlation_matrix, 
                   mask=mask,
                   annot=False,  # 숫자 표시 안함 (너무 많아서)
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': '상관계수'})
        
        plt.title('재무비율 간 상관계수 행렬', fontsize=16, pad=20)
        plt.xlabel('변수', fontsize=12)
        plt.ylabel('변수', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"상관계수 히트맵 저장: {save_path}")
    
    def create_vif_plot(self, save_path):
        """VIF 결과 시각화"""
        if self.vif_results is None:
            return
            
        plt.figure(figsize=(12, 8))
        
        # VIF 값이 유효한 데이터만 필터링
        valid_vif = self.vif_results.dropna(subset=['VIF'])
        
        if len(valid_vif) == 0:
            print("시각화할 유효한 VIF 데이터가 없습니다.")
            return
            
        # 상위 20개 변수만 표시
        top_20 = valid_vif.head(20)
        
        # 색상 설정
        colors = ['red' if vif >= 10 else 'orange' if vif >= 5 else 'green' 
                 for vif in top_20['VIF']]
        
        # 막대그래프
        bars = plt.barh(range(len(top_20)), top_20['VIF'], color=colors)
        
        # y축 레이블 설정
        plt.yticks(range(len(top_20)), top_20['Variable'])
        
        # 기준선 추가
        plt.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='VIF = 5')
        plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='VIF = 10')
        
        plt.xlabel('VIF 값', fontsize=12)
        plt.ylabel('변수', fontsize=12)
        plt.title('변수별 VIF (Variance Inflation Factor) - Top 20', fontsize=14)
        plt.legend()
        plt.grid(axis='x', alpha=0.3)
        
        # 값 표시
        for i, (bar, vif) in enumerate(zip(bars, top_20['VIF'])):
            plt.text(vif + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{vif:.1f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"VIF 차트 저장: {save_path}")
    
    def create_high_correlation_plot(self, high_corr_pairs, save_path):
        """높은 상관관계 쌍 시각화"""
        if not high_corr_pairs:
            print("높은 상관관계 쌍이 없어 시각화를 건너뜁니다.")
            return
            
        # 상위 20개 쌍만 표시
        top_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:20]
        
        plt.figure(figsize=(12, 8))
        
        # 변수 쌍 레이블 생성
        pair_labels = [f"{pair['variable1'][:10]}...\nvs\n{pair['variable2'][:10]}..." 
                      for pair in top_pairs]
        correlations = [pair['correlation'] for pair in top_pairs]
        
        # 색상 설정
        colors = ['red' if corr > 0 else 'blue' for corr in correlations]
        
        # 막대그래프
        bars = plt.barh(range(len(top_pairs)), correlations, color=colors, alpha=0.7)
        
        plt.yticks(range(len(top_pairs)), pair_labels, fontsize=8)
        plt.xlabel('상관계수', fontsize=12)
        plt.ylabel('변수 쌍', fontsize=12)
        plt.title('높은 상관관계 변수 쌍 (|r| ≥ 0.8) - Top 20', fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        
        # 값 표시
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            plt.text(corr + (0.01 if corr > 0 else -0.01), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{corr:.3f}', va='center', ha='left' if corr > 0 else 'right',
                    fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"높은 상관관계 차트 저장: {save_path}")
    
    def save_results(self, output_dir):
        """분석 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 결과 딕셔너리 생성
        results = {
            'analysis_timestamp': timestamp,
            'dataset_info': {
                'total_observations': len(self.data),
                'numeric_variables': len(self.numeric_columns),
                'variable_names': self.numeric_columns
            }
        }
        
        # 상관계수 결과
        if self.correlation_matrix is not None:
            # 높은 상관관계 쌍
            high_corr_pairs = self.calculate_correlation_matrix()
            results['correlation_analysis'] = {
                'high_correlation_pairs_count': len(high_corr_pairs),
                'high_correlation_pairs': high_corr_pairs
            }
            
            # 상관계수 행렬 저장
            corr_path = os.path.join(output_dir, f'correlation_matrix_{timestamp}.csv')
            self.correlation_matrix.to_csv(corr_path, encoding='utf-8')
            print(f"상관계수 행렬 저장: {corr_path}")
        
        # VIF 결과
        if self.vif_results is not None:
            vif_high_count = len(self.vif_results[self.vif_results['VIF'] >= 10])
            results['vif_analysis'] = {
                'variables_with_high_vif': vif_high_count,
                'max_vif': float(self.vif_results['VIF'].max()) if not self.vif_results['VIF'].isna().all() else None,
                'mean_vif': float(self.vif_results['VIF'].mean()) if not self.vif_results['VIF'].isna().all() else None
            }
            
            # VIF 결과 저장
            vif_path = os.path.join(output_dir, f'vif_results_{timestamp}.csv')
            self.vif_results.to_csv(vif_path, index=False, encoding='utf-8')
            print(f"VIF 결과 저장: {vif_path}")
        
        # 조건지수 결과
        if self.condition_indices is not None:
            results['condition_index_analysis'] = {
                'max_condition_index': float(self.condition_indices['max_condition_index']),
                'multicollinearity_severity': 'High' if self.condition_indices['max_condition_index'] > 30 else 
                                            'Moderate' if self.condition_indices['max_condition_index'] > 15 else 'Low'
            }
        
        # 종합 결과 저장
        results_path = os.path.join(output_dir, f'multicollinearity_analysis_{timestamp}.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"종합 분석 결과 저장: {results_path}")
        
        return results
    
    def run_analysis(self, output_dir):
        """전체 분석 실행"""
        print("=== 다중공선성 분석 시작 ===")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 상관계수 분석
        high_corr_pairs = self.calculate_correlation_matrix()
        
        # 3. VIF 계산
        self.calculate_vif()
        
        # 4. 조건지수 계산
        self.calculate_condition_indices()
        
        # 5. 시각화
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 상관계수 히트맵
        corr_heatmap_path = os.path.join(output_dir, f'correlation_heatmap_{timestamp}.png')
        self.create_correlation_heatmap(corr_heatmap_path)
        
        # VIF 차트
        vif_plot_path = os.path.join(output_dir, f'vif_analysis_{timestamp}.png')
        self.create_vif_plot(vif_plot_path)
        
        # 높은 상관관계 차트
        high_corr_plot_path = os.path.join(output_dir, f'high_correlation_pairs_{timestamp}.png')
        self.create_high_correlation_plot(high_corr_pairs, high_corr_plot_path)
        
        # 6. 결과 저장
        results = self.save_results(output_dir)
        
        # 7. 요약 출력
        self.print_summary(results)
        
        print("=== 다중공선성 분석 완료 ===")
        
        return results
    
    def print_summary(self, results):
        """분석 결과 요약 출력"""
        print("\n" + "="*50)
        print("다중공선성 분석 결과 요약")
        print("="*50)
        
        # 데이터셋 정보
        dataset_info = results['dataset_info']
        print(f"분석 대상 관측치 수: {dataset_info['total_observations']:,}")
        print(f"분석 대상 변수 수: {dataset_info['numeric_variables']}")
        
        # 상관계수 분석
        if 'correlation_analysis' in results:
            corr_info = results['correlation_analysis']
            print(f"높은 상관관계 쌍 (|r| ≥ 0.8): {corr_info['high_correlation_pairs_count']}개")
        
        # VIF 분석
        if 'vif_analysis' in results:
            vif_info = results['vif_analysis']
            print(f"높은 VIF 변수 (≥ 10): {vif_info['variables_with_high_vif']}개")
            if vif_info['max_vif']:
                print(f"최대 VIF: {vif_info['max_vif']:.2f}")
            if vif_info['mean_vif']:
                print(f"평균 VIF: {vif_info['mean_vif']:.2f}")
        
        # 조건지수 분석
        if 'condition_index_analysis' in results:
            ci_info = results['condition_index_analysis']
            print(f"최대 조건지수: {ci_info['max_condition_index']:.2f}")
            print(f"다중공선성 수준: {ci_info['multicollinearity_severity']}")
        
        print("\n해석 기준:")
        print("- 상관계수: |r| ≥ 0.8이면 높은 상관관계")
        print("- VIF: 5-10은 보통, ≥10은 높은 다중공선성")
        print("- 조건지수: >15는 보통, >30은 높은 다중공선성")
        print("="*50)


def main():
    """메인 함수"""
    # 경로 설정
    data_path = "data/processed/FS2_filtered.csv"
    output_dir = "outputs/analysis"
    
    # 분석 실행
    analyzer = MulticollinearityAnalyzer(data_path)
    results = analyzer.run_analysis(output_dir)
    
    return results


if __name__ == "__main__":
    main() 