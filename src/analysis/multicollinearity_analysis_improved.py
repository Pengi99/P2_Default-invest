#!/usr/bin/env python3
"""
개선된 다중공선성 분석 스크립트
FS_100 데이터셋의 다중공선성을 분석하고 VIF 무한대 문제를 해결합니다.

작성자: AI Assistant
목적: 재무비율 간 다중공선성 확인 및 분석 (VIF 무한대 문제 해결)
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class ImprovedMulticollinearityAnalyzer:
    """개선된 다중공선성 분석 클래스"""
    
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
        self.problematic_variables = []
        self.clean_variables = None
        
    def load_data(self):
        """데이터 로드 및 전처리"""
        print("데이터 로딩 중...")
        self.data = pd.read_csv(self.data_path, encoding='utf-8')
        
        # 수치형 컬럼만 선택
        exclude_columns = ['회사명', '거래소코드', '회계년도', 'default']
        self.numeric_columns = [col for col in self.data.columns 
                               if col not in exclude_columns]
        
        print(f"분석 대상 컬럼 수: {len(self.numeric_columns)}")
        print(f"총 관측치 수: {len(self.data)}")
        
        # 결측치 및 무한값 정리
        clean_data = self.data[self.numeric_columns].replace([np.inf, -np.inf], np.nan)
        missing_counts = clean_data.isnull().sum()
        
        if missing_counts.sum() > 0:
            print(f"결측치/무한값 발견: {missing_counts.sum()}개")
            print("문제가 있는 컬럼:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count}개")
                
    def detect_perfect_multicollinearity(self):
        """완전한 다중공선성 탐지"""
        print("완전한 다중공선성 탐지 중...")
        
        clean_data = self.data[self.numeric_columns].replace([np.inf, -np.inf], np.nan).dropna()
        
        # 상관계수 행렬 계산
        corr_matrix = clean_data.corr()
        
        # 완전한 상관관계 (절댓값 1.0) 찾기
        perfect_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= 0.999:  # 거의 완전한 상관관계
                    perfect_pairs.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        # 특이값 분해로 랭크 확인
        try:
            U, s, Vt = np.linalg.svd(clean_data.values)
            rank = np.sum(s > 1e-10)  # 수치적 허용오차
            
            print(f"데이터 행렬의 실제 랭크: {rank} / {clean_data.shape[1]}")
            
            if rank < clean_data.shape[1]:
                print(f"선형 종속성 탐지: {clean_data.shape[1] - rank}개 변수가 다른 변수들의 선형결합입니다.")
                
                # 가장 작은 특이값을 가진 변수들 식별
                small_singular_indices = np.where(s < 1e-8)[0]
                if len(small_singular_indices) > 0:
                    print("매우 작은 특이값을 가진 성분이 발견되었습니다.")
                    
        except Exception as e:
            print(f"특이값 분해 중 오류: {e}")
            
        return perfect_pairs
    
    def calculate_vif_iterative(self, max_vif_threshold=10):
        """반복적 VIF 계산 (높은 VIF 변수 제거)"""
        print(f"반복적 VIF 계산 중 (임계값: {max_vif_threshold})...")
        
        clean_data = self.data[self.numeric_columns].replace([np.inf, -np.inf], np.nan).dropna()
        remaining_vars = list(clean_data.columns)
        removed_vars = []
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n반복 {iteration}: {len(remaining_vars)}개 변수")
            
            if len(remaining_vars) <= 1:
                print("변수가 1개 이하로 남아 중단합니다.")
                break
                
            current_data = clean_data[remaining_vars]
            
            # 상수항 추가가 가능한지 확인
            try:
                X = add_constant(current_data)
                if X.shape[1] != len(remaining_vars) + 1:
                    print("상수항 추가 중 문제 발생")
                    break
            except Exception as e:
                print(f"상수항 추가 실패: {e}")
                break
            
            # VIF 계산
            vif_data = []
            max_vif = 0
            max_vif_var = None
            
            for i, col in enumerate(remaining_vars):
                try:
                    vif_value = variance_inflation_factor(X.values, i+1)
                    
                    if np.isfinite(vif_value):
                        vif_data.append({
                            'Variable': col,
                            'VIF': vif_value
                        })
                        
                        if vif_value > max_vif:
                            max_vif = vif_value
                            max_vif_var = col
                    else:
                        # 무한대 VIF인 경우
                        vif_data.append({
                            'Variable': col,
                            'VIF': np.inf
                        })
                        max_vif = np.inf
                        max_vif_var = col
                        
                except Exception as e:
                    print(f"VIF 계산 오류 ({col}): {e}")
                    vif_data.append({
                        'Variable': col,
                        'VIF': np.nan
                    })
            
            # 결과 출력
            current_vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
            print(f"최대 VIF: {max_vif:.2f} ({max_vif_var})")
            
            # 임계값 확인
            if max_vif <= max_vif_threshold or not np.isfinite(max_vif):
                if np.isfinite(max_vif):
                    print(f"모든 변수의 VIF가 {max_vif_threshold} 이하입니다.")
                    self.vif_results = current_vif_df
                    self.clean_variables = remaining_vars
                    break
                else:
                    # 무한대 VIF 변수 제거
                    print(f"무한대 VIF 변수 제거: {max_vif_var}")
                    remaining_vars.remove(max_vif_var)
                    removed_vars.append(max_vif_var)
            else:
                # 가장 높은 VIF 변수 제거
                print(f"높은 VIF 변수 제거: {max_vif_var} (VIF: {max_vif:.2f})")
                remaining_vars.remove(max_vif_var)
                removed_vars.append(max_vif_var)
            
            if iteration > 50:  # 무한루프 방지
                print("최대 반복 횟수 초과")
                break
        
        self.problematic_variables = removed_vars
        
        print(f"\n제거된 변수 ({len(removed_vars)}개): {removed_vars}")
        print(f"남은 변수 ({len(remaining_vars)}개): {remaining_vars}")
        
        return self.vif_results
    
    def calculate_correlation_matrix(self):
        """상관계수 행렬 계산 및 높은 상관관계 쌍 탐지"""
        print("상관계수 행렬 계산 중...")
        
        clean_data = self.data[self.numeric_columns].replace([np.inf, -np.inf], np.nan).dropna()
        self.correlation_matrix = clean_data.corr()
        
        # 높은 상관관계 쌍 찾기
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
    
    def perform_pca_analysis(self):
        """주성분 분석을 통한 차원 축소 분석"""
        print("주성분 분석 수행 중...")
        
        clean_data = self.data[self.numeric_columns].replace([np.inf, -np.inf], np.nan).dropna()
        
        # 표준화
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_data)
        
        # PCA 수행
        pca = PCA()
        pca.fit(scaled_data)
        
        # 설명된 분산 비율
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # 95% 분산을 설명하는 성분 수
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        pca_results = {
            'total_components': len(self.numeric_columns),
            'components_for_95_variance': n_components_95,
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist()
        }
        
        print(f"95% 분산 설명에 필요한 성분 수: {n_components_95} / {len(self.numeric_columns)}")
        
        return pca_results
    
    def create_comprehensive_visualizations(self, output_dir, timestamp):
        """포괄적인 시각화 생성"""
        
        # 1. 상관계수 히트맵
        self.create_correlation_heatmap(
            os.path.join(output_dir, f'correlation_heatmap_{timestamp}.png')
        )
        
        # 2. VIF 결과 (정리된 변수들)
        if self.vif_results is not None:
            self.create_vif_plot(
                os.path.join(output_dir, f'vif_analysis_cleaned_{timestamp}.png')
            )
        
        # 3. 변수 제거 과정 시각화
        self.create_variable_removal_plot(
            os.path.join(output_dir, f'variable_removal_process_{timestamp}.png')
        )
        
        # 4. PCA 설명 분산 시각화
        pca_results = self.perform_pca_analysis()
        self.create_pca_plot(
            pca_results,
            os.path.join(output_dir, f'pca_variance_explained_{timestamp}.png')
        )
        
        return pca_results
    
    def create_correlation_heatmap(self, save_path):
        """상관계수 히트맵 생성"""
        plt.figure(figsize=(16, 14))
        
        # 마스크 생성 (상삼각형 숨기기)
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        
        # 히트맵 생성
        sns.heatmap(self.correlation_matrix, 
                   mask=mask,
                   annot=True,
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': '상관계수'},
                   annot_kws={'size': 8})
        
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
        """정리된 VIF 결과 시각화"""
        if self.vif_results is None:
            return
            
        plt.figure(figsize=(12, 8))
        
        # 유효한 VIF 데이터만 필터링
        valid_vif = self.vif_results[np.isfinite(self.vif_results['VIF'])]
        
        if len(valid_vif) == 0:
            print("시각화할 유효한 VIF 데이터가 없습니다.")
            return
        
        # 색상 설정
        colors = ['red' if vif >= 10 else 'orange' if vif >= 5 else 'green' 
                 for vif in valid_vif['VIF']]
        
        # 막대그래프
        bars = plt.barh(range(len(valid_vif)), valid_vif['VIF'], color=colors)
        
        # y축 레이블 설정
        plt.yticks(range(len(valid_vif)), valid_vif['Variable'])
        
        # 기준선 추가
        plt.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='VIF = 5')
        plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='VIF = 10')
        
        plt.xlabel('VIF 값', fontsize=12)
        plt.ylabel('변수', fontsize=12)
        plt.title('다중공선성 정리 후 VIF 결과', fontsize=14)
        plt.legend()
        plt.grid(axis='x', alpha=0.3)
        
        # 값 표시
        for i, (bar, vif) in enumerate(zip(bars, valid_vif['VIF'])):
            plt.text(vif + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{vif:.1f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"정리된 VIF 차트 저장: {save_path}")
    
    def create_variable_removal_plot(self, save_path):
        """변수 제거 과정 시각화"""
        if not self.problematic_variables:
            return
            
        plt.figure(figsize=(12, 6))
        
        # 제거된 변수들과 이유
        removed_vars = self.problematic_variables[::-1]  # 역순으로 표시
        reasons = ['높은 VIF/완전한 다중공선성'] * len(removed_vars)
        
        plt.barh(range(len(removed_vars)), [1] * len(removed_vars), 
                color='red', alpha=0.7)
        
        plt.yticks(range(len(removed_vars)), removed_vars)
        plt.xlabel('제거 단계', fontsize=12)
        plt.ylabel('제거된 변수', fontsize=12)
        plt.title(f'다중공선성으로 인한 변수 제거 과정 ({len(removed_vars)}개 변수)', fontsize=14)
        
        # 제거 순서 표시
        for i, var in enumerate(removed_vars):
            plt.text(0.5, i, f'{len(removed_vars)-i}번째', 
                    va='center', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"변수 제거 과정 차트 저장: {save_path}")
    
    def create_pca_plot(self, pca_results, save_path):
        """PCA 설명 분산 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        n_components = len(pca_results['explained_variance_ratio'])
        components = range(1, n_components + 1)
        
        # 개별 설명 분산
        ax1.bar(components, pca_results['explained_variance_ratio'], alpha=0.7)
        ax1.set_xlabel('주성분 번호')
        ax1.set_ylabel('설명 분산 비율')
        ax1.set_title('각 주성분의 설명 분산 비율')
        ax1.grid(alpha=0.3)
        
        # 누적 설명 분산
        ax2.plot(components, pca_results['cumulative_variance'], 
                marker='o', linewidth=2, markersize=4)
        ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95%')
        ax2.axvline(x=pca_results['components_for_95_variance'], 
                   color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('주성분 번호')
        ax2.set_ylabel('누적 설명 분산 비율')
        ax2.set_title('누적 설명 분산 비율')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PCA 분석 차트 저장: {save_path}")
    
    def save_comprehensive_results(self, output_dir, pca_results):
        """포괄적인 분석 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 기본 결과
        results = {
            'analysis_timestamp': timestamp,
            'dataset_info': {
                'total_observations': len(self.data),
                'original_variables': len(self.numeric_columns),
                'final_variables': len(self.clean_variables) if self.clean_variables else 0,
                'removed_variables': len(self.problematic_variables),
                'variable_names': self.numeric_columns,
                'clean_variable_names': self.clean_variables,
                'removed_variable_names': self.problematic_variables
            }
        }
        
        # 상관계수 분석
        high_corr_pairs = self.calculate_correlation_matrix()
        results['correlation_analysis'] = {
            'high_correlation_pairs_count': len(high_corr_pairs),
            'high_correlation_pairs': high_corr_pairs
        }
        
        # VIF 분석
        if self.vif_results is not None:
            valid_vif = self.vif_results[np.isfinite(self.vif_results['VIF'])]
            results['vif_analysis'] = {
                'final_variables_count': len(valid_vif),
                'max_vif': float(valid_vif['VIF'].max()) if len(valid_vif) > 0 else None,
                'mean_vif': float(valid_vif['VIF'].mean()) if len(valid_vif) > 0 else None,
                'variables_with_high_vif': len(valid_vif[valid_vif['VIF'] >= 10])
            }
        
        # PCA 분석
        results['pca_analysis'] = pca_results
        
        # 완전한 다중공선성 탐지
        perfect_pairs = self.detect_perfect_multicollinearity()
        results['perfect_multicollinearity'] = {
            'perfect_correlation_pairs': perfect_pairs,
            'perfect_pairs_count': len(perfect_pairs)
        }
        
        # 결과 저장
        results_path = os.path.join(output_dir, f'comprehensive_multicollinearity_analysis_{timestamp}.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 상세 결과 CSV 저장
        if self.vif_results is not None:
            vif_path = os.path.join(output_dir, f'final_vif_results_{timestamp}.csv')
            self.vif_results.to_csv(vif_path, index=False, encoding='utf-8')
        
        if self.correlation_matrix is not None:
            corr_path = os.path.join(output_dir, f'final_correlation_matrix_{timestamp}.csv')
            self.correlation_matrix.to_csv(corr_path, encoding='utf-8')
        
        print(f"포괄적 분석 결과 저장: {results_path}")
        return results
    
    def print_comprehensive_summary(self, results):
        """포괄적인 분석 결과 요약 출력"""
        print("\n" + "="*60)
        print("포괄적 다중공선성 분석 결과")
        print("="*60)
        
        # 데이터셋 정보
        dataset_info = results['dataset_info']
        print(f"총 관측치 수: {dataset_info['total_observations']:,}")
        print(f"원본 변수 수: {dataset_info['original_variables']}")
        print(f"최종 변수 수: {dataset_info['final_variables']}")
        print(f"제거된 변수 수: {dataset_info['removed_variables']}")
        
        if dataset_info['removed_variable_names']:
            print(f"제거된 변수: {', '.join(dataset_info['removed_variable_names'])}")
        
        # 상관분석
        corr_info = results['correlation_analysis']
        print(f"\n높은 상관관계 쌍 (|r| ≥ 0.8): {corr_info['high_correlation_pairs_count']}개")
        
        # VIF 분석
        if 'vif_analysis' in results and results['vif_analysis']['max_vif'] is not None:
            vif_info = results['vif_analysis']
            print(f"정리 후 최대 VIF: {vif_info['max_vif']:.2f}")
            print(f"정리 후 평균 VIF: {vif_info['mean_vif']:.2f}")
            print(f"높은 VIF 변수 (≥ 10): {vif_info['variables_with_high_vif']}개")
        
        # PCA 분석
        pca_info = results['pca_analysis']
        print(f"\n차원 축소 관점:")
        print(f"95% 분산 설명 필요 성분: {pca_info['components_for_95_variance']} / {pca_info['total_components']}")
        reduction_rate = (1 - pca_info['components_for_95_variance'] / pca_info['total_components']) * 100
        print(f"잠재적 차원 축소율: {reduction_rate:.1f}%")
        
        # 완전한 다중공선성
        perfect_info = results['perfect_multicollinearity']
        print(f"\n완전한 다중공선성:")
        print(f"완전 상관 쌍: {perfect_info['perfect_pairs_count']}개")
        
        print("\n" + "="*60)
        print("권장사항:")
        
        if dataset_info['removed_variables'] > 0:
            print("• 다중공선성으로 인해 제거된 변수들을 모델링에서 제외하세요.")
        
        if corr_info['high_correlation_pairs_count'] > 0:
            print("• 높은 상관관계를 가진 변수 쌍 중 하나를 제거 고려하세요.")
        
        if reduction_rate > 30:
            print("• PCA를 통한 차원 축소를 고려해보세요.")
        
        print("• 정규화(Ridge) 또는 변수 선택 기법 적용을 권장합니다.")
        print("="*60)
    
    def run_comprehensive_analysis(self, output_dir):
        """포괄적인 다중공선성 분석 실행"""
        print("=== 포괄적 다중공선성 분석 시작 ===")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 완전한 다중공선성 탐지
        self.detect_perfect_multicollinearity()
        
        # 3. 반복적 VIF 계산
        self.calculate_vif_iterative(max_vif_threshold=10)
        
        # 4. 상관계수 분석
        self.calculate_correlation_matrix()
        
        # 5. 시각화 및 PCA 분석
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pca_results = self.create_comprehensive_visualizations(output_dir, timestamp)
        
        # 6. 결과 저장
        results = self.save_comprehensive_results(output_dir, pca_results)
        
        # 7. 요약 출력
        self.print_comprehensive_summary(results)
        
        print("=== 포괄적 다중공선성 분석 완료 ===")
        
        return results


def main():
    """메인 함수"""
    # 경로 설정
    data_path = "data/final/FS_100_complete.csv"
    output_dir = "outputs/analysis"
    
    # 포괄적 분석 실행
    analyzer = ImprovedMulticollinearityAnalyzer(data_path)
    results = analyzer.run_comprehensive_analysis(output_dir)
    
    return results


if __name__ == "__main__":
    main() 