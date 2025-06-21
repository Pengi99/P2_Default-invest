"""
Altman Z-Score 및 K2-Score 부실예측 성능 분석
기존 ML 모델과의 성능 비교 및 전통적 재무지표의 효과성 검증

Author: AI Assistant
Date: 2025-06-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class ScorePerformanceAnalyzer:
    """Altman Z-Score 및 K2-Score 성능 분석 클래스"""
    
    def __init__(self):
        """초기화"""
        self.scores_df = None
        self.labeled_df = None
        
    def load_data(self) -> pd.DataFrame:
        """데이터 로드 및 부실 라벨 결합"""
        print("📂 데이터 로드 중...")
        
        # 점수 데이터 로드
        scores_path = 'outputs/reports/altman_k2_scores.csv'
        if not os.path.exists(scores_path):
            raise FileNotFoundError(f"점수 파일을 찾을 수 없습니다: {scores_path}")
        
        self.scores_df = pd.read_csv(scores_path)
        print(f"✅ 점수 데이터 로드: {self.scores_df.shape}")
        
        # 라벨된 데이터 로드
        labeled_path = 'data_new/final/FS_ratio_flow_labeled.csv'
        if not os.path.exists(labeled_path):
            raise FileNotFoundError(f"라벨 파일을 찾을 수 없습니다: {labeled_path}")
        
        labeled_df = pd.read_csv(labeled_path)
        print(f"✅ 라벨 데이터 로드: {labeled_df.shape}")
        
        # 데이터 결합
        merge_cols = ['회사명', '거래소코드', '회계년도']
        self.labeled_df = pd.merge(
            self.scores_df, 
            labeled_df[merge_cols + ['default']],
            on=merge_cols,
            how='inner'
        )
        print(f"✅ 데이터 결합 완료: {self.labeled_df.shape}")
        print(f"📊 부실 기업 비율: {self.labeled_df['default'].mean():.2%}")
        
        return self.labeled_df
    
    def analyze_score_thresholds(self) -> dict:
        """각 점수별 최적 임계값 분석"""
        print("\n🎯 점수별 최적 임계값 분석")
        print("="*60)
        
        score_columns = [col for col in self.labeled_df.columns 
                        if 'K2_Score' in col and col != 'default']
        
        results = {}
        
        for score_col in score_columns:
            # 결측값 제거
            valid_data = self.labeled_df[[score_col, 'default']].dropna()
            if len(valid_data) == 0:
                continue
                
            y_true = valid_data['default']
            y_scores = valid_data[score_col]
            
            # ROC 곡선 계산
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
            
            # Youden's J statistic으로 최적 임계값 찾기
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # 최적 임계값에서의 성능
            y_pred = (y_scores >= optimal_threshold).astype(int)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            results[score_col] = {
                'auc': auc,
                'optimal_threshold': optimal_threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_samples': len(valid_data)
            }
            
            print(f"\n📊 {score_col}:")
            print(f"   AUC: {auc:.4f}")
            print(f"   최적 임계값: {optimal_threshold:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   샘플 수: {len(valid_data):,}")
        
        return results
    
    def analyze_traditional_thresholds(self) -> dict:
        """전통적인 Altman Z-Score 임계값 분석"""
        print("\n📚 전통적인 Altman Z-Score 임계값 분석")
        print("="*60)
        
        # 전통적인 임계값들
        traditional_thresholds = {
            'K2_Score_Original': {
                'safe': 2.6,      # 안전 구간
                'distress': 1.1   # 위험 구간
            }
        }
        
        results = {}
        
        for score_col, thresholds in traditional_thresholds.items():
            if score_col not in self.labeled_df.columns:
                continue
                
            valid_data = self.labeled_df[[score_col, 'default']].dropna()
            if len(valid_data) == 0:
                continue
            
            y_true = valid_data['default']
            scores = valid_data[score_col]
            
            # 구간별 분류
            safe_zone = scores >= thresholds['safe']
            gray_zone = (scores >= thresholds['distress']) & (scores < thresholds['safe'])
            distress_zone = scores < thresholds['distress']
            
            # 각 구간별 부실률
            safe_default_rate = y_true[safe_zone].mean() if safe_zone.sum() > 0 else 0
            gray_default_rate = y_true[gray_zone].mean() if gray_zone.sum() > 0 else 0
            distress_default_rate = y_true[distress_zone].mean() if distress_zone.sum() > 0 else 0
            
            # 위험 구간을 부실 예측으로 사용
            y_pred = (scores < thresholds['distress']).astype(int)
            
            precision = precision_score(y_true, y_pred) if y_pred.sum() > 0 else 0
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred) if y_pred.sum() > 0 else 0
            auc = roc_auc_score(y_true, -scores)  # 점수가 낮을수록 위험하므로 음수
            
            results[score_col] = {
                'safe_zone_count': safe_zone.sum(),
                'gray_zone_count': gray_zone.sum(),
                'distress_zone_count': distress_zone.sum(),
                'safe_default_rate': safe_default_rate,
                'gray_default_rate': gray_default_rate,
                'distress_default_rate': distress_default_rate,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
            
            print(f"\n📊 {score_col} (전통적 임계값):")
            print(f"   안전 구간 (≥{thresholds['safe']}): {safe_zone.sum():,}개, 부실률: {safe_default_rate:.2%}")
            print(f"   회색지대 ({thresholds['distress']}~{thresholds['safe']}): {gray_zone.sum():,}개, 부실률: {gray_default_rate:.2%}")
            print(f"   위험 구간 (<{thresholds['distress']}): {distress_zone.sum():,}개, 부실률: {distress_default_rate:.2%}")
            print(f"   AUC: {auc:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
        
        return results
    
    def compare_with_ml_models(self, score_results: dict) -> pd.DataFrame:
        """ML 모델과 성능 비교"""
        print("\n🤖 ML 모델과 성능 비교")
        print("="*60)
        
        # ML 모델 결과 로드
        ml_results_path = 'outputs/reports/model_comparison_report.json'
        if os.path.exists(ml_results_path):
            import json
            with open(ml_results_path, 'r', encoding='utf-8') as f:
                ml_data = json.load(f)
            
            ml_performance = pd.DataFrame(ml_data['performance_summary'])
            ml_performance = ml_performance.set_index('Model')
            
            print("📊 ML 모델 성능:")
            print(ml_performance[['AUC', 'Precision', 'Recall', 'F1-Score']].round(4))
        else:
            print("⚠️ ML 모델 결과를 찾을 수 없습니다.")
            ml_performance = pd.DataFrame()
        
        # 전통적 점수 성능
        score_performance = []
        for score_name, metrics in score_results.items():
            score_performance.append({
                'Model': score_name,
                'AUC': metrics['auc'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            })
        
        score_df = pd.DataFrame(score_performance).set_index('Model')
        
        print("\n📊 전통적 점수 성능:")
        print(score_df.round(4))
        
        # 통합 비교
        if not ml_performance.empty:
            combined_df = pd.concat([ml_performance[['AUC', 'Precision', 'Recall', 'F1-Score']], 
                                   score_df], axis=0)
            
            print("\n🏆 통합 성능 비교:")
            print(combined_df.round(4))
            
            return combined_df
        
        return score_df
    
    def visualize_performance(self, score_results: dict) -> None:
        """성능 시각화"""
        print("\n📈 성능 시각화")
        print("="*60)
        
        # ROC 곡선 그리기
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Altman Z-Score 및 K2-Score ROC 곡선 분석', fontsize=16, fontweight='bold')
        
        score_columns = [col for col in self.labeled_df.columns 
                        if 'K2_Score' in col and col != 'default']
        
        for i, score_col in enumerate(score_columns[:4]):
            row, col = divmod(i, 2)
            ax = axes[row, col]
            
            # 결측값 제거
            valid_data = self.labeled_df[[score_col, 'default']].dropna()
            if len(valid_data) == 0:
                ax.set_visible(False)
                continue
            
            y_true = valid_data['default']
            y_scores = valid_data[score_col]
            
            # ROC 곡선
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
            
            ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{score_col}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 빈 subplot 숨기기
        for i in range(len(score_columns), 4):
            row, col = divmod(i, 2)
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # 저장
        os.makedirs('outputs/visualizations', exist_ok=True)
        plt.savefig('outputs/visualizations/traditional_scores_roc_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 성능 비교 차트
        self._plot_performance_comparison(score_results)
    
    def _plot_performance_comparison(self, score_results: dict) -> None:
        """성능 비교 차트"""
        # 성능 데이터 준비
        metrics = ['auc', 'precision', 'recall', 'f1']
        score_names = list(score_results.keys())
        
        performance_data = []
        for metric in metrics:
            for score_name in score_names:
                if metric in score_results[score_name]:
                    performance_data.append({
                        'Score': score_name.replace('K2_Score_', ''),
                        'Metric': metric.upper(),
                        'Value': score_results[score_name][metric]
                    })
        
        if not performance_data:
            return
        
        perf_df = pd.DataFrame(performance_data)
        
        # 히트맵
        pivot_df = perf_df.pivot(index='Score', columns='Metric', values='Value')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0.5, vmin=0, vmax=1)
        plt.title('전통적 점수 모델 성능 비교', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig('outputs/visualizations/traditional_scores_performance_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """메인 실행 함수"""
    print("🏢 Altman Z-Score 및 K2-Score 성능 분석")
    print("="*60)
    
    # 분석기 초기화
    analyzer = ScorePerformanceAnalyzer()
    
    try:
        # 데이터 로드
        df = analyzer.load_data()
        
        # 최적 임계값 분석
        score_results = analyzer.analyze_score_thresholds()
        
        # 전통적 임계값 분석
        traditional_results = analyzer.analyze_traditional_thresholds()
        
        # ML 모델과 비교
        comparison_df = analyzer.compare_with_ml_models(score_results)
        
        # 시각화
        analyzer.visualize_performance(score_results)
        
        print("\n🎉 전통적 점수 성능 분석 완료!")
        print("📊 모든 결과는 outputs/visualizations/ 폴더에 저장되었습니다.")
        
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        print("💡 먼저 altman_score_analysis.py를 실행해주세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main() 