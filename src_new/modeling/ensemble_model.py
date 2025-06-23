"""
앙상블 모델 구현
다양한 기본 모델들을 결합하여 성능 향상
"""

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
from pathlib import Path
import joblib
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic' if plt.matplotlib.get_backend() != 'Agg' else 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class EnsembleModel:
    """
    다양한 기본 모델들을 결합한 앙상블 모델
    """
    
    def __init__(self, config, base_models=None):
        """
        앙상블 모델 초기화
        
        Args:
            config (dict): 앙상블 설정
            base_models (dict): 기본 모델들 {model_name: model_object}
        """
        self.config = config
        self.base_models = base_models or {}
        self.ensemble_config = config.get('ensemble', {})
        self.method = self.ensemble_config.get('method', 'weighted_average')
        self.weights = self.ensemble_config.get('weights', {})
        self.auto_weight = self.ensemble_config.get('auto_weight', False)
        
        # 결과 저장
        self.predictions = {}
        self.final_prediction = None
        self.performance_metrics = {}
        
        print(f"🎭 앙상블 모델 초기화")
        print(f"📊 방법: {self.method}")
        print(f"⚖️ 가중치: {self.weights}")
        print(f"🤖 자동 가중치: {self.auto_weight}")
    
    def add_model(self, model_name, model, weight=None):
        """
        기본 모델 추가
        
        Args:
            model_name (str): 모델 이름
            model: 훈련된 모델 객체
            weight (float): 가중치 (선택사항)
        """
        self.base_models[model_name] = model
        if weight is not None:
            self.weights[model_name] = weight
        
        print(f"✅ 모델 추가: {model_name} (가중치: {weight})")
    
    def predict_proba_individual(self, X):
        """
        각 기본 모델의 예측 확률 계산
        
        Args:
            X: 입력 특성
            
        Returns:
            dict: {model_name: prediction_probabilities}
        """
        predictions = {}
        expected_length = len(X)
        
        for model_name, model in self.base_models.items():
            try:
                # 예측 확률 계산 (클래스 1의 확률만 사용)
                pred_proba = model.predict_proba(X)[:, 1]
                
                # 예측 결과 크기 확인
                if len(pred_proba) != expected_length:
                    print(f"⚠️ {model_name} 예측 크기 불일치: 예상={expected_length}, 실제={len(pred_proba)}")
                    # 크기가 다르면 기본값으로 채움
                    pred_proba = np.full(expected_length, 0.5)
                
                predictions[model_name] = pred_proba
                print(f"✅ {model_name} 예측 완료: {len(pred_proba)}개 샘플")
                
            except Exception as e:
                print(f"⚠️ {model_name} 예측 실패: {e}")
                # 실패한 경우 0.5로 채움 (중립적 예측)
                predictions[model_name] = np.full(expected_length, 0.5)
                print(f"✅ {model_name} 기본값으로 대체: {expected_length}개 샘플")
        
        return predictions
    
    def calculate_auto_weights(self, X_valid, y_valid):
        """
        검증 데이터 기반 자동 가중치 계산
        
        Args:
            X_valid: 검증 특성
            y_valid: 검증 라벨
            
        Returns:
            dict: 최적 가중치
        """
        print("\n🤖 자동 가중치 계산")
        print("="*50)
        
        # 각 모델의 검증 성능 계산
        individual_predictions = self.predict_proba_individual(X_valid)
        model_scores = {}
        
        for model_name, pred_proba in individual_predictions.items():
            try:
                # F1 score 기반 성능 평가
                pred_binary = (pred_proba >= 0.5).astype(int)
                f1 = f1_score(y_valid, pred_binary, zero_division=0)
                auc = roc_auc_score(y_valid, pred_proba)
                
                # 복합 점수 (F1과 AUC의 조화평균)
                if f1 > 0 and auc > 0:
                    composite_score = 2 * (f1 * auc) / (f1 + auc)
                else:
                    composite_score = 0
                
                model_scores[model_name] = composite_score
                print(f"{model_name}: F1={f1:.4f}, AUC={auc:.4f}, 복합={composite_score:.4f}")
                
            except Exception as e:
                print(f"⚠️ {model_name} 성능 계산 실패: {e}")
                model_scores[model_name] = 0
        
        # 성능 기반 가중치 계산 (소프트맥스)
        scores = np.array(list(model_scores.values()))
        if scores.sum() > 0:
            # 소프트맥스로 가중치 정규화
            exp_scores = np.exp(scores - np.max(scores))
            weights_array = exp_scores / exp_scores.sum()
            
            auto_weights = dict(zip(model_scores.keys(), weights_array))
        else:
            # 모든 성능이 0인 경우 균등 가중치
            auto_weights = {name: 1.0/len(model_scores) for name in model_scores.keys()}
        
        print(f"\n🎯 자동 계산된 가중치:")
        for name, weight in auto_weights.items():
            print(f"  {name}: {weight:.4f}")
        
        return auto_weights
    
    def ensemble_predict_proba(self, X, X_valid=None, y_valid=None):
        """
        앙상블 예측 수행
        
        Args:
            X: 예측할 데이터
            X_valid: 검증 데이터 (자동 가중치용)
            y_valid: 검증 라벨 (자동 가중치용)
            
        Returns:
            np.array: 앙상블 예측 확률
        """
        print(f"\n🎭 앙상블 예측 수행 ({self.method})")
        print("="*50)
        
        # 개별 모델 예측
        individual_predictions = self.predict_proba_individual(X)
        self.predictions = individual_predictions
        
        if len(individual_predictions) == 0:
            raise ValueError("사용 가능한 모델이 없습니다.")
        
        # 자동 가중치 계산
        if self.auto_weight and X_valid is not None and y_valid is not None:
            auto_weights = self.calculate_auto_weights(X_valid, y_valid)
            # 기존 가중치와 자동 가중치 결합
            final_weights = auto_weights
        else:
            final_weights = self.weights.copy()
        
        # 가중치 정규화
        if final_weights:
            # 설정된 모델만 사용
            available_models = set(individual_predictions.keys())
            final_weights = {k: v for k, v in final_weights.items() if k in available_models}
            
            if final_weights:
                total_weight = sum(final_weights.values())
                final_weights = {k: v/total_weight for k, v in final_weights.items()}
            else:
                # 가중치가 없으면 균등 가중치
                final_weights = {name: 1.0/len(individual_predictions) 
                               for name in individual_predictions.keys()}
        else:
            # 가중치가 없으면 균등 가중치
            final_weights = {name: 1.0/len(individual_predictions) 
                           for name in individual_predictions.keys()}
        
        print(f"🎯 최종 가중치:")
        for name, weight in final_weights.items():
            print(f"  {name}: {weight:.4f}")
        
        # 앙상블 방법에 따른 예측
        if self.method == 'weighted_average':
            ensemble_pred = self._weighted_average(individual_predictions, final_weights)
        elif self.method == 'voting':
            ensemble_pred = self._majority_voting(individual_predictions, final_weights)
        elif self.method == 'stacking':
            ensemble_pred = self._stacking_prediction(individual_predictions, X)
        else:
            raise ValueError(f"지원하지 않는 앙상블 방법: {self.method}")
        
        self.final_prediction = ensemble_pred
        return ensemble_pred
    
    def _weighted_average(self, predictions, weights):
        """가중 평균 앙상블"""
        if not predictions:
            raise ValueError("예측 결과가 없습니다.")
        
        # 첫 번째 예측 결과의 길이를 기준으로 설정
        first_pred = next(iter(predictions.values()))
        expected_length = len(first_pred)
        ensemble_pred = np.zeros(expected_length)
        
        print(f"🔍 가중 평균 계산: 예상 길이={expected_length}")
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0)
            
            # 예측 결과 크기 확인
            if len(pred) != expected_length:
                print(f"⚠️ {model_name} 예측 크기 불일치: 예상={expected_length}, 실제={len(pred)}")
                # 크기가 다르면 기본값으로 대체
                pred = np.full(expected_length, 0.5)
            
            ensemble_pred += weight * pred
            print(f"  {model_name}: 가중치={weight:.4f}, 예측길이={len(pred)}")
        
        print(f"✅ 가중 평균 완료: 결과 길이={len(ensemble_pred)}")
        return ensemble_pred
    
    def _majority_voting(self, predictions, weights):
        """가중 다수결 투표"""
        ensemble_pred = np.zeros(len(next(iter(predictions.values()))))
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0)
            # 0.5 기준으로 이진 분류 후 가중치 적용
            binary_pred = (pred >= 0.5).astype(float)
            ensemble_pred += weight * binary_pred
        
        return ensemble_pred
    
    def _stacking_prediction(self, predictions, X):
        """스태킹 앙상블 (간단한 선형 결합)"""
        # 여기서는 간단한 선형 결합으로 구현
        # 실제로는 메타 모델을 훈련해야 함
        return self._weighted_average(predictions, self.weights)
    
    def evaluate_ensemble(self, X, y, threshold=0.5):
        """
        앙상블 모델 성능 평가
        
        Args:
            X: 테스트 특성
            y: 테스트 라벨
            threshold: 분류 임계값
            
        Returns:
            dict: 성능 메트릭
        """
        print(f"\n📊 앙상블 모델 평가 (threshold={threshold})")
        print("="*50)
        
        # 항상 새로 예측 수행 (데이터 크기 불일치 방지)
        ensemble_proba = self.ensemble_predict_proba(X)
        
        # 데이터 크기 확인
        print(f"🔍 데이터 크기 확인: X={len(X)}, y={len(y)}, ensemble_proba={len(ensemble_proba)}")
        
        # 크기가 다르면 오류 발생
        if len(ensemble_proba) != len(y):
            raise ValueError(f"예측 결과와 라벨의 크기가 다릅니다: 예측={len(ensemble_proba)}, 라벨={len(y)}")
        
        # 이진 예측
        ensemble_pred = (ensemble_proba >= threshold).astype(int)
        
        # 성능 메트릭 계산
        metrics = {
            'auc': roc_auc_score(y, ensemble_proba),
            'precision': precision_score(y, ensemble_pred, zero_division=0),
            'recall': recall_score(y, ensemble_pred, zero_division=0),
            'f1': f1_score(y, ensemble_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y, ensemble_pred),
            'average_precision': average_precision_score(y, ensemble_proba)
        }
        
        self.performance_metrics = metrics
        
        print(f"🎯 앙상블 성능:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def find_optimal_threshold(self, X_valid, y_valid, metric='f1'):
        """
        최적 임계값 찾기
        
        Args:
            X_valid: 검증 특성
            y_valid: 검증 라벨
            metric: 최적화할 메트릭
            
        Returns:
            tuple: (최적 임계값, 성능 메트릭)
        """
        print(f"\n🎯 앙상블 최적 Threshold 탐색 ({metric})")
        print("="*50)
        
        # 앙상블 예측 확률
        ensemble_proba = self.ensemble_predict_proba(X_valid)
        
        # 다양한 임계값에서 성능 계산
        thresholds = np.arange(0.05, 0.95, 0.05)
        threshold_results = []
        
        for threshold in thresholds:
            pred = (ensemble_proba >= threshold).astype(int)
            
            if len(np.unique(pred)) == 1:
                continue
            
            try:
                metrics = {
                    'threshold': threshold,
                    'precision': precision_score(y_valid, pred, zero_division=0),
                    'recall': recall_score(y_valid, pred, zero_division=0),
                    'f1': f1_score(y_valid, pred, zero_division=0),
                    'balanced_accuracy': balanced_accuracy_score(y_valid, pred)
                }
                threshold_results.append(metrics)
            except:
                continue
        
        if not threshold_results:
            print("⚠️ 최적 threshold 찾기 실패, 기본값 0.5 사용")
            return 0.5, {}
        
        # 최적 임계값 선택
        threshold_df = pd.DataFrame(threshold_results)
        best_idx = threshold_df[metric].idxmax()
        optimal_threshold = threshold_df.loc[best_idx, 'threshold']
        best_metrics = threshold_df.loc[best_idx].to_dict()
        
        print(f"✅ 최적 Threshold: {optimal_threshold:.3f}")
        print(f"📊 최적 성능:")
        for m, v in best_metrics.items():
            if m != 'threshold':
                print(f"  {m.upper()}: {v:.4f}")
        
        return optimal_threshold, best_metrics
    
    def save_model(self, filepath):
        """앙상블 모델 저장"""
        ensemble_data = {
            'config': self.config,
            'method': self.method,
            'weights': self.weights,
            'auto_weight': self.auto_weight,
            'performance_metrics': self.performance_metrics,
            'model_names': list(self.base_models.keys())
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(ensemble_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 앙상블 모델 저장: {filepath}")
    
    def create_ensemble_report(self, output_dir):
        """앙상블 결과 리포트 생성"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 개별 모델 vs 앙상블 비교 시각화
        if self.predictions and self.final_prediction is not None:
            self._plot_model_comparison(output_dir)
        
        # 앙상블 가중치 시각화
        if self.weights:
            self._plot_weights(output_dir)
    
    def _plot_model_comparison(self, output_dir):
        """개별 모델 vs 앙상블 예측 비교"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('개별 모델 vs 앙상블 예측 비교', fontsize=16, fontweight='bold')
        
        # 예측 분포 비교
        ax = axes[0, 0]
        for model_name, pred in self.predictions.items():
            ax.hist(pred, alpha=0.6, bins=30, label=model_name, density=True)
        ax.hist(self.final_prediction, alpha=0.8, bins=30, label='Ensemble', 
                density=True, color='red', linewidth=2, histtype='step')
        ax.set_xlabel('예측 확률')
        ax.set_ylabel('밀도')
        ax.set_title('예측 확률 분포')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 예측 상관관계
        ax = axes[0, 1]
        model_names = list(self.predictions.keys())
        if len(model_names) >= 2:
            x_pred = self.predictions[model_names[0]]
            y_pred = self.predictions[model_names[1]]
            ax.scatter(x_pred, y_pred, alpha=0.6, s=20)
            ax.plot([0, 1], [0, 1], 'r--', alpha=0.8)
            ax.set_xlabel(f'{model_names[0]} 예측')
            ax.set_ylabel(f'{model_names[1]} 예측')
            ax.set_title('모델 간 예측 상관관계')
            ax.grid(True, alpha=0.3)
        
        # 앙상블 vs 개별 모델
        ax = axes[1, 0]
        for model_name, pred in self.predictions.items():
            ax.scatter(pred, self.final_prediction, alpha=0.6, s=20, label=model_name)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.8)
        ax.set_xlabel('개별 모델 예측')
        ax.set_ylabel('앙상블 예측')
        ax.set_title('앙상블 vs 개별 모델')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 가중치 막대 그래프
        ax = axes[1, 1]
        if self.weights:
            names = list(self.weights.keys())
            weights = list(self.weights.values())
            bars = ax.bar(names, weights, alpha=0.7, color='skyblue')
            ax.set_ylabel('가중치')
            ax.set_title('앙상블 가중치')
            ax.tick_params(axis='x', rotation=45)
            
            # 가중치 값 표시
            for bar, weight in zip(bars, weights):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{weight:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ensemble_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 앙상블 분석 저장: ensemble_analysis.png")
    
    def _plot_weights(self, output_dir):
        """가중치 시각화"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        names = list(self.weights.keys())
        weights = list(self.weights.values())
        
        # 파이 차트
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        wedges, texts, autotexts = ax.pie(weights, labels=names, autopct='%1.2f%%',
                                         colors=colors, startangle=90)
        
        ax.set_title('앙상블 모델 가중치 분포', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ensemble_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 가중치 분포 저장: ensemble_weights.png")


def create_ensemble_from_results(results_dict, config):
    """
    실행 결과에서 앙상블 모델 생성
    
    Args:
        results_dict (dict): 모델 실행 결과
        config (dict): 앙상블 설정
        
    Returns:
        EnsembleModel: 생성된 앙상블 모델
    """
    ensemble = EnsembleModel(config)
    
    # 기본 모델들을 앙상블에 추가
    ensemble_config = config.get('ensemble', {})
    enabled_models = ensemble_config.get('models', [])
    
    for model_key, model_obj in results_dict.items():
        if any(enabled_model in model_key for enabled_model in enabled_models):
            ensemble.add_model(model_key, model_obj)
    
    return ensemble 