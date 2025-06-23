"""
모델 성능 비교 및 분석 스크립트
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_curve, auc
import joblib

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic' if plt.matplotlib.get_backend() != 'Agg' else 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ModelComparison:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.test_data = None
        
    def load_models_and_results(self, models_path='outputs/models/'):
        """저장된 모델들과 결과를 로드"""
        print("📂 모델 및 결과 로드")
        print("="*60)
        
        model_files = {
            'LogisticRegression': 'logistic_regression_best_model.joblib',
            'RandomForest': 'random_forest_best_model.joblib',
            'XGBoost': 'xgboost_best_model.joblib'
        }
        
        result_files = {
            'LogisticRegression': 'logistic_regression_results.json',
            'RandomForest': 'random_forest_results.json',
            'XGBoost': 'xgboost_results.json'
        }
        
        # 모델 로드
        for model_name, model_file in model_files.items():
            model_path = os.path.join(models_path, model_file)
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"✅ {model_name} 모델 로드 완료")
            else:
                print(f"⚠️ {model_name} 모델 파일을 찾을 수 없습니다: {model_path}")
        
        # 결과 로드
        for model_name, result_file in result_files.items():
            result_path = os.path.join(models_path, result_file)
            if os.path.exists(result_path):
                with open(result_path, 'r', encoding='utf-8') as f:
                    self.results[model_name] = json.load(f)
                print(f"✅ {model_name} 결과 로드 완료")
            else:
                print(f"⚠️ {model_name} 결과 파일을 찾을 수 없습니다: {result_path}")
        
        print(f"\n📊 로드된 모델: {len(self.models)}개")
        print(f"📊 로드된 결과: {len(self.results)}개")
    
    def load_test_data(self, data_path='data/final/'):
        """테스트 데이터 로드"""
        print("\n📂 테스트 데이터 로드")
        print("="*60)
        
        X_test = pd.read_csv(os.path.join(data_path, 'X_test_smote.csv'))
        y_test = pd.read_csv(os.path.join(data_path, 'y_test_smote.csv')).iloc[:, 0]
        
        self.test_data = {'X': X_test, 'y': y_test}
        
        print(f"✅ 테스트 데이터 로드 완료")
        print(f"   형태: {X_test.shape}")
        print(f"   부실 비율: {y_test.mean():.2%}")
    
    def create_performance_comparison(self):
        """성능 지표 비교표 생성"""
        print("\n📊 성능 지표 비교")
        print("="*60)
        
        metrics_df = []
        
        for model_name, result in self.results.items():
            if 'test_metrics' in result:
                metrics = result['test_metrics']
                metrics_df.append({
                    'Model': model_name,
                    'AUC': metrics.get('auc', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1', 0),
                    'CV_Score': result.get('cv_best_score', 0)
                })
        
        self.performance_df = pd.DataFrame(metrics_df)
        
        if not self.performance_df.empty:
            print("🏆 모델 성능 비교:")
            print(self.performance_df.round(4))
            
            # 최고 성능 모델 찾기
            best_auc_model = self.performance_df.loc[self.performance_df['AUC'].idxmax(), 'Model']
            best_f1_model = self.performance_df.loc[self.performance_df['F1-Score'].idxmax(), 'Model']
            
            print(f"\n🥇 최고 AUC: {best_auc_model}")
            print(f"🥇 최고 F1-Score: {best_f1_model}")
        
        return self.performance_df
    
    def plot_roc_comparison(self, save_path='outputs/visualizations/'):
        """ROC 곡선 비교"""
        print("\n📈 ROC 곡선 비교")
        print("="*60)
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i, (model_name, model) in enumerate(self.models.items()):
            if self.test_data is not None:
                # 예측 확률
                y_proba = model.predict_proba(self.test_data['X'])[:, 1]
                
                # ROC 곡선 계산
                fpr, tpr, _ = roc_curve(self.test_data['y'], y_proba)
                roc_auc = auc(fpr, tpr)
                
                # 플롯
                plt.plot(fpr, tpr, color=colors[i], lw=2, 
                        label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # 대각선 (랜덤 분류기)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC 곡선 비교', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 저장
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/model_roc_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ ROC 비교 저장: {save_path}/model_roc_comparison.png")
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path='outputs/visualizations/'):
        """성능 지표 비교 차트"""
        print("\n📊 성능 지표 비교 차트")
        print("="*60)
        
        if hasattr(self, 'performance_df') and not self.performance_df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('모델 성능 지표 비교', fontsize=16, fontweight='bold')
            
            metrics = ['AUC', 'Precision', 'Recall', 'F1-Score']
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
            
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                bars = ax.bar(self.performance_df['Model'], self.performance_df[metric], 
                             color=colors[i], alpha=0.7, edgecolor='black')
                
                # 값 표시
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10)
                
                ax.set_title(f'{metric} 비교', fontsize=12, fontweight='bold')
                ax.set_ylabel(metric, fontsize=10)
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3)
                
                # x축 레이블 회전
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 저장
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f'{save_path}/model_metrics_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✅ 지표 비교 저장: {save_path}/model_metrics_comparison.png")
            
            plt.show()
    
    def plot_feature_importance_comparison(self, save_path='outputs/visualizations/'):
        """특성 중요도 비교"""
        print("\n🔍 특성 중요도 비교")
        print("="*60)
        
        # Random Forest와 XGBoost의 특성 중요도만 비교 (로지스틱 회귀는 계수)
        tree_models = ['RandomForest', 'XGBoost']
        available_models = [model for model in tree_models if model in self.results]
        
        if len(available_models) >= 2:
            fig, axes = plt.subplots(1, len(available_models), figsize=(15, 8))
            if len(available_models) == 1:
                axes = [axes]
            
            fig.suptitle('특성 중요도 비교 (Tree-based 모델)', fontsize=16, fontweight='bold')
            
            for i, model_name in enumerate(available_models):
                if 'feature_importances' in self.results[model_name]:
                    importances = self.results[model_name]['feature_importances']
                    
                    # 상위 10개 특성
                    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
                    features, values = zip(*sorted_features)
                    
                    # 색상 설정
                    color = 'green' if model_name == 'RandomForest' else 'purple'
                    
                    axes[i].barh(features, values, color=color, alpha=0.7)
                    axes[i].set_title(f'{model_name}\n특성 중요도 (Top 10)', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel('중요도', fontsize=10)
                    
                    # 값 표시
                    for j, v in enumerate(values):
                        axes[i].text(v + 0.001, j, f'{v:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            
            # 저장
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f'{save_path}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✅ 특성 중요도 비교 저장: {save_path}/feature_importance_comparison.png")
            
            plt.show()
        else:
            print("⚠️ 특성 중요도 비교를 위한 충분한 Tree-based 모델이 없습니다.")
    
    def generate_summary_report(self, save_path='outputs/reports/'):
        """종합 분석 리포트 생성"""
        print("\n📋 종합 분석 리포트 생성")
        print("="*60)
        
        os.makedirs(save_path, exist_ok=True)
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'models_compared': list(self.models.keys()),
            'performance_summary': self.performance_df.to_dict('records') if hasattr(self, 'performance_df') else [],
            'best_models': {},
            'recommendations': []
        }
        
        if hasattr(self, 'performance_df') and not self.performance_df.empty:
            # 최고 성능 모델들
            report['best_models'] = {
                'best_auc': {
                    'model': self.performance_df.loc[self.performance_df['AUC'].idxmax(), 'Model'],
                    'score': float(self.performance_df['AUC'].max())
                },
                'best_precision': {
                    'model': self.performance_df.loc[self.performance_df['Precision'].idxmax(), 'Model'],
                    'score': float(self.performance_df['Precision'].max())
                },
                'best_recall': {
                    'model': self.performance_df.loc[self.performance_df['Recall'].idxmax(), 'Model'],
                    'score': float(self.performance_df['Recall'].max())
                },
                'best_f1': {
                    'model': self.performance_df.loc[self.performance_df['F1-Score'].idxmax(), 'Model'],
                    'score': float(self.performance_df['F1-Score'].max())
                }
            }
            
            # 추천사항
            best_auc_model = report['best_models']['best_auc']['model']
            best_auc_score = report['best_models']['best_auc']['score']
            
            if best_auc_score > 0.8:
                report['recommendations'].append(f"{best_auc_model}이 우수한 성능(AUC: {best_auc_score:.4f})을 보여 운영 환경에 적합합니다.")
            elif best_auc_score > 0.7:
                report['recommendations'].append(f"{best_auc_model}이 양호한 성능을 보이나 추가 튜닝이 필요할 수 있습니다.")
            else:
                report['recommendations'].append("모든 모델의 성능이 기대에 미치지 못합니다. 데이터 전처리 또는 특성 엔지니어링 재검토가 필요합니다.")
            
            # 불균형 데이터 관련 추천
            avg_precision = self.performance_df['Precision'].mean()
            avg_recall = self.performance_df['Recall'].mean()
            
            if avg_precision > avg_recall:
                report['recommendations'].append("Precision이 Recall보다 높습니다. 실제 부실 기업 탐지율 향상을 위한 임계값 조정을 고려해보세요.")
            else:
                report['recommendations'].append("Recall이 상대적으로 높습니다. False Positive 감소를 위한 모델 정밀도 향상을 고려해보세요.")
        
        # 리포트 저장
        report_path = os.path.join(save_path, 'model_comparison_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 종합 리포트 저장: {report_path}")
        
        # 요약 출력
        print(f"\n📊 분석 요약:")
        if 'best_models' in report and report['best_models']:
            print(f"   🥇 최고 AUC: {report['best_models']['best_auc']['model']} ({report['best_models']['best_auc']['score']:.4f})")
            print(f"   🎯 최고 F1: {report['best_models']['best_f1']['model']} ({report['best_models']['best_f1']['score']:.4f})")
        
        if report['recommendations']:
            print(f"\n💡 추천사항:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        return report

def main():
    """메인 실행 함수"""
    print("🏢 한국 기업 부실예측 - 모델 성능 비교")
    print("="*60)
    
    # 비교 객체 생성
    comparison = ModelComparison()
    
    # 모델 및 결과 로드
    comparison.load_models_and_results()
    
    # 테스트 데이터 로드
    comparison.load_test_data()
    
    # 성능 비교표 생성
    performance_df = comparison.create_performance_comparison()
    
    # 시각화
    comparison.plot_roc_comparison()
    comparison.plot_metrics_comparison()
    comparison.plot_feature_importance_comparison()
    
    # 종합 리포트 생성
    report = comparison.generate_summary_report()
    
    print("\n🎉 모델 비교 분석 완료!")
    print("📊 모든 결과는 outputs/ 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 