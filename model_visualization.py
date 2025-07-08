"""
모델별 성능 시각화 도구
==============================

폴더별로 저장된 모델들을 불러와서 테스트 데이터로 성능을 평가하고
AUC, F1-Score, Precision, Recall을 시각화합니다.

사용법:
python model_visualization.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':  # Windows
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False


class ModelVisualizer:
    """모델별 성능 시각화 클래스"""
    
    def __init__(self, models_dir: str, data_dir: str = "data/final"):
        """
        초기화
        
        Args:
            models_dir: 모델이 저장된 디렉토리 경로
            data_dir: 테스트 데이터가 저장된 디렉토리 경로
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models = {}
        self.results = {}
        
        # 실제 훈련된 모델들은 모두 특성 선택으로 동일한 5개 특성 사용
        self.selected_features = [
            "매출액증가율", "자본비율", "이자부담차입금비율", 
            "총자산수익률(ROA)", "로그총자산"
        ]
        
        # 모델 분류 정의 (예시 이미지 기준)
        self.model_categories = {
            'logistic': ['logistic_regression'],
            'RF': ['random_forest'], 
            'XGboost': ['xgboost']
        }
        
        # 기존 modeling pipeline에서 계산된 optimal threshold 사용
        self.optimal_thresholds = {
            'logistic_regression': {
                'normal': 0.7658325538018845,
                'smote': 0.8097071105720812,
                'undersampling': 0.7718764133112883,
                'combined': 0.9919230523872429
            },
            'random_forest': {
                'normal': 0.38369141074694035,
                'smote': 0.6427416447841382,
                'undersampling': 0.43599113112432086,
                'combined': 0.5553100264299876
            },
            'xgboost': {
                'normal': 0.4256463646888733,
                'smote': 0.5636553168296814,
                'undersampling': 0.343456894159317,
                'combined': 0.6189958453178406
            }
        }
        
        # 데이터 로드
        self._load_test_data()
        
        print("🚀 모델 시각화 도구 초기화 완료")
        print(f"📁 모델 디렉토리: {self.models_dir}")
        print(f"📊 데이터 디렉토리: {self.data_dir}")
    
    def _load_test_data(self):
        """테스트 데이터 로드 및 전처리"""
        print("📊 테스트 데이터 로딩 중...")
        
        # 데이터 로드
        X_test_full = pd.read_csv(self.data_dir / "X_test.csv")
        self.y_test = pd.read_csv(self.data_dir / "y_test.csv").iloc[:, 0]  # 첫 번째 컬럼만
        
        # 훈련 데이터도 로드 (스케일링을 위해)
        X_train_full = pd.read_csv(self.data_dir / "X_train.csv")
        
        # 선택된 특성만 사용
        self.X_test = X_test_full[self.selected_features]
        X_train = X_train_full[self.selected_features]
        
        # Robust 스케일링 적용
        print("🔧 Robust 스케일링 적용 중...")
        self.scaler = RobustScaler()
        self.scaler.fit(X_train)
        
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        print(f"✅ 데이터 로딩 완료:")
        print(f"   - X_test shape: {self.X_test.shape}")
        print(f"   - y_test shape: {self.y_test.shape}")
        print(f"   - 선택된 특성: {', '.join(self.selected_features)}")
        print(f"   - 양성 클래스 비율: {self.y_test.mean():.3%}")
    
    def _load_models(self):
        """모델 파일들을 로드"""
        print("🤖 모델 로딩 중...")
        
        model_files = list(self.models_dir.glob("*.joblib"))
        
        for model_file in model_files:
            # 파일명에서 데이터 타입과 모델 타입 추출
            # 예: "smote__logistic_regression_model.joblib"
            filename = model_file.stem
            
            if "__" in filename:
                data_type, model_info = filename.split("__", 1)
                model_type = model_info.replace("_model", "")
                
                # 앙상블 모델 제외
                if "ensemble" in model_type:
                    continue
                    
                try:
                    model = joblib.load(model_file)
                    key = f"{data_type}__{model_type}"
                    self.models[key] = {
                        'model': model,
                        'data_type': data_type,
                        'model_type': model_type,
                        'file_path': model_file
                    }
                    print(f"   ✅ {key}")
                    
                except Exception as e:
                    print(f"   ❌ {model_file.name}: {e}")
        
        print(f"📈 총 {len(self.models)}개 모델 로드 완료")
    
    def _find_optimal_f1_threshold(self, y_true, y_pred_proba):
        """F1 스코어를 최대화하는 임계값 찾기 (기존 modeling pipeline 방식과 동일)"""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            if len(np.unique(y_pred)) > 1:  # 예측이 한 클래스로만 나오지 않는 경우
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        return best_threshold, best_f1
    
    def _evaluate_models(self):
        """모든 모델 평가"""
        print("📊 모델 성능 평가 중...")
        
        for key, model_info in self.models.items():
            model = model_info['model']
            data_type = model_info['data_type']
            model_type = model_info['model_type']
            
            try:
                # 예측 (모든 모델이 동일한 특성 사용)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                # 기존 modeling pipeline에서 계산된 optimal threshold 사용
                optimal_threshold = self.optimal_thresholds.get(model_type, {}).get(data_type, 0.5)
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                
                # 메트릭 계산
                metrics = {
                    'auc': roc_auc_score(self.y_test, y_pred_proba),
                    'f1': f1_score(self.y_test, y_pred, zero_division=0),
                    'precision': precision_score(self.y_test, y_pred, zero_division=0),
                    'recall': recall_score(self.y_test, y_pred, zero_division=0),
                    'average_precision': average_precision_score(self.y_test, y_pred_proba),
                    'optimal_threshold': optimal_threshold,
                    'y_pred_proba': y_pred_proba,
                    'y_pred': y_pred
                }
                
                self.results[key] = {
                    **model_info,
                    **metrics
                }
                
                print(f"   ✅ {key}: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}, Threshold={optimal_threshold:.3f}")
                
            except Exception as e:
                print(f"   ❌ {key}: {e}")
    
    def _create_category_visualization(self, category: str, model_types: List[str]):
        """카테고리별 시각화 생성"""
        print(f"🎨 {category} 시각화 생성 중...")
        
        # 해당 카테고리의 모델들 필터링
        category_models = {}
        for key, result in self.results.items():
            if result['model_type'] in model_types:
                category_models[key] = result
        
        if not category_models:
            print(f"   ⚠️ {category} 카테고리에 해당하는 모델이 없습니다.")
            return
        
        # 데이터 타입별로 정렬 (normal, smote, undersampling, combined 순서)
        data_type_order = ['normal', 'smote', 'undersampling', 'combined']
        sorted_models = sorted(category_models.items(), 
                             key=lambda x: data_type_order.index(x[1]['data_type']) 
                             if x[1]['data_type'] in data_type_order else 999)
        
        # 데이터 준비
        model_names = []
        metrics_data = {'AUC': [], 'F1-Score': [], 'Precision': [], 'Recall': []}
        
        for key, result in sorted_models:
            display_name = f"{result['model_type']}_{result['data_type']}"
            model_names.append(display_name)
            
            metrics_data['AUC'].append(result['auc'])
            metrics_data['F1-Score'].append(result['f1'])
            metrics_data['Precision'].append(result['precision'])
            metrics_data['Recall'].append(result['recall'])
        
        # 시각화 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 색상 설정 (예시 이미지와 유사하게)
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'plum', 'wheat', 'lightgray']
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[i]
            
            # 바 차트 생성
            bars = ax.bar(range(len(model_names)), values, 
                         color=colors[:len(model_names)], alpha=0.8)
            
            # 값 표시
            for j, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=10)
            
            # 축 설정
            ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            
            # x축 레이블 설정
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # 전체 제목
        fig.suptitle(f'{category} Models Performance Comparison', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # 저장
        output_path = f"{category}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ {category} 시각화 저장: {output_path}")
    
    def create_all_visualizations(self):
        """모든 시각화 생성"""
        print("🎨 모델별 시각화 생성 시작...")
        
        # 모델 로드 및 평가
        self._load_models()
        self._evaluate_models()
        
        # 카테고리별 시각화 생성
        for category, model_types in self.model_categories.items():
            self._create_category_visualization(category, model_types)
        
        print("✅ 모든 시각화 생성 완료!")
        
        # 결과 요약 출력
        print("\n📊 성능 요약:")
        for key, result in self.results.items():
            print(f"   {key}:")
            print(f"     AUC: {result['auc']:.3f}, F1: {result['f1']:.3f}, "
                  f"Precision: {result['precision']:.3f}, Recall: {result['recall']:.3f}, "
                  f"Optimal Threshold: {result['optimal_threshold']:.3f}")


def main():
    """메인 실행 함수"""
    print("🚀 모델 성능 시각화 도구 시작")
    
    # 경로 설정
    models_dir = "/Users/jojongho/KDT/P2_Default-invest/outputs/modeling_runs/default_modeling_run_20250706_131209/models"
    data_dir = "/Users/jojongho/KDT/P2_Default-invest/data/final"
    
    # 시각화 도구 초기화 및 실행
    visualizer = ModelVisualizer(models_dir, data_dir)
    visualizer.create_all_visualizations()
    
    print("🎉 시각화 완료!")


if __name__ == "__main__":
    main()