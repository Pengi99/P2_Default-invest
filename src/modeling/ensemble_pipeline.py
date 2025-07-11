"""
앙상블 파이프라인
===============================

기능:
1. 개별 모델들의 예측 결합
2. 가중치 자동 계산
3. 앙상블 성능 평가
4. Threshold 최적화

다양한 앙상블 방법 지원:
- Simple Average
- Weighted Average  
- Stacking (Meta-learner)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, balanced_accuracy_score,
    average_precision_score, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


class EnsemblePipeline:
    """
    앙상블 파이프라인 클래스
    
    여러 개별 모델의 예측을 결합하여 더 강력한 예측 모델을 생성
    """
    
    def __init__(self, config: Dict, models: Dict):
        """
        앙상블 파이프라인 초기화
        
        Args:
            config: 설정 딕셔너리
            models: 개별 모델들의 딕셔너리 {model_key: model_object}
        """
        self.config = config
        self.models = models
        self.ensemble_config = config.get('ensemble', {})
        
        # 앙상블 방법
        self.method = self.ensemble_config.get('method', 'weighted_average')
        self.auto_weight = self.ensemble_config.get('auto_weight', True)
        
        # 가중치 저장
        self.weights = {}
        self.meta_learner = None
        
        print(f"🎭 앙상블 파이프라인 초기화")
        print(f"📊 방법: {self.method}")
        print(f"🔄 자동 가중치: {self.auto_weight}")
        print(f"🤖 포함 모델 수: {len(models)}")
    
    def calculate_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """
        검증 데이터 기반으로 각 모델의 가중치 계산 (F1 스코어 우선)
        
        Args:
            X_val: 검증 특성 데이터
            y_val: 검증 타겟 데이터
            
        Returns:
            Dict[str, float]: 모델별 가중치
        """
        if not self.auto_weight:
            # 동일 가중치
            equal_weight = 1.0 / len(self.models)
            return {model_key: equal_weight for model_key in self.models.keys()}
        
        # YAML에서 가중치 계산 메트릭 읽기
        weight_metric = self.ensemble_config.get('weight_metric', 'f1')  # 기본값을 f1으로 변경
        print(f"🔍 검증 데이터 기반 가중치 계산 중... (메트릭: {weight_metric.upper()})")
        
        model_scores = {}
        
        for model_key, model in self.models.items():
            try:
                # 검증 데이터로 예측
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # 설정된 메트릭에 따라 점수 계산
                if weight_metric == 'f1':
                    # F1의 경우 최적 임계값 찾기
                    score = self._find_best_f1_score(y_val, y_pred_proba)
                elif weight_metric == 'f1_composite':
                    # F1 스코어와 AUC의 조합 (F1에 더 큰 가중치)
                    f1_score_val = self._find_best_f1_score(y_val, y_pred_proba)
                    auc_score = roc_auc_score(y_val, y_pred_proba)
                    score = 0.7 * f1_score_val + 0.3 * auc_score  # F1에 70% 가중치
                elif weight_metric == 'average_precision':
                    score = average_precision_score(y_val, y_pred_proba)
                elif weight_metric == 'balanced_accuracy':
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    score = balanced_accuracy_score(y_val, y_pred)
                else:  # roc_auc
                    score = roc_auc_score(y_val, y_pred_proba)
                
                model_scores[model_key] = score
                
                print(f"  📊 {model_key}: {weight_metric.upper()} = {score:.4f}")
                
            except Exception as e:
                print(f"  ⚠️ {model_key} 평가 실패: {e}")
                model_scores[model_key] = 0.1  # F1의 경우 더 낮은 기본값
        
        # F1 스코어 기반 가중치 계산 (더 강한 차별화)
        scores = np.array(list(model_scores.values()))
        
        # F1의 경우 최소값을 0.1로 설정 (성능이 매우 낮은 모델 페널티)
        min_score = 0.1 if weight_metric in ['f1', 'f1_composite'] else 0.5
        scores = np.maximum(scores, min_score)
        
        # F1 스코어에 대해 더 강한 차별화를 위한 지수 가중치
        if weight_metric in ['f1', 'f1_composite']:
            # F1 스코어에 제곱을 적용하여 높은 성능 모델에 더 큰 가중치 부여
            scores_normalized = (scores - min_score) / (1.0 - min_score)  # 0-1 정규화
            scores_squared = scores_normalized ** 2  # 제곱으로 차별화 강화
            weights_array = scores_squared / np.sum(scores_squared)
        else:
            # 다른 메트릭의 경우 소프트맥스 사용
            exp_scores = np.exp((scores - np.max(scores)) * 10)
            weights_array = exp_scores / np.sum(exp_scores)
        
        weights = dict(zip(model_scores.keys(), weights_array))
        
        print("✅ 최종 가중치 (F1 스코어 우선):")
        for model_key, weight in weights.items():
            print(f"  🎯 {model_key}: {weight:.4f}")
        
        return weights
    
    def _find_best_f1_score(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """
        주어진 예측 확률에서 최적의 F1 스코어 찾기
        
        Args:
            y_true: 실제 레이블
            y_pred_proba: 예측 확률
            
        Returns:
            float: 최대 F1 스코어
        """
        # 다양한 임계값에서 F1 스코어 계산
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0.0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            if len(np.unique(y_pred)) > 1:  # 예측이 한 클래스로만 나오지 않는 경우
                f1 = f1_score(y_true, y_pred, zero_division=0)
                best_f1 = max(best_f1, f1)
        
        return best_f1
    
    def ensemble_predict_proba(self, X: pd.DataFrame, X_val: Optional[pd.DataFrame] = None, 
                              y_val: Optional[pd.Series] = None) -> np.ndarray:
        """
        앙상블 예측 확률 계산
        
        Args:
            X: 예측할 특성 데이터
            X_val: 가중치 계산용 검증 특성 데이터 (선택사항)
            y_val: 가중치 계산용 검증 타겟 데이터 (선택사항)
            
        Returns:
            np.ndarray: 앙상블 예측 확률
        """
        # 가중치 계산 (아직 계산되지 않은 경우)
        if not self.weights and X_val is not None and y_val is not None:
            self.weights = self.calculate_weights(X_val, y_val)
        elif not self.weights:
            # 동일 가중치 사용
            equal_weight = 1.0 / len(self.models)
            self.weights = {model_key: equal_weight for model_key in self.models.keys()}
        
        if self.method == 'simple_average':
            return self._simple_average_predict(X)
        elif self.method == 'weighted_average':
            return self._weighted_average_predict(X)
        elif self.method == 'stacking':
            return self._stacking_predict(X, X_val, y_val)
        else:
            raise ValueError(f"지원하지 않는 앙상블 방법: {self.method}")
    
    def _simple_average_predict(self, X: pd.DataFrame) -> np.ndarray:
        """단순 평균 앙상블"""
        predictions = []
        
        for model_key, model in self.models.items():
            y_pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(y_pred_proba)
        
        return np.mean(predictions, axis=0)
    
    def _weighted_average_predict(self, X: pd.DataFrame) -> np.ndarray:
        """가중 평균 앙상블"""
        predictions = []
        weights = []
        
        for model_key, model in self.models.items():
            y_pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(y_pred_proba)
            weights.append(self.weights.get(model_key, 1.0 / len(self.models)))
        
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # 가중 평균 계산
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        
        return weighted_pred
    
    def _stacking_predict(self, X: pd.DataFrame, X_val: Optional[pd.DataFrame] = None, 
                         y_val: Optional[pd.Series] = None) -> np.ndarray:
        """스태킹 앙상블 (메타 러너 사용)"""
        # 메타 러너가 없고 검증 데이터가 있는 경우 훈련
        if self.meta_learner is None and X_val is not None and y_val is not None:
            self._train_meta_learner(X_val, y_val)
        
        # 개별 모델들의 예측을 메타 특성으로 사용
        meta_features = []
        for model_key, model in self.models.items():
            y_pred_proba = model.predict_proba(X)[:, 1]
            meta_features.append(y_pred_proba)
        
        meta_features = np.column_stack(meta_features)
        
        if self.meta_learner is not None:
            return self.meta_learner.predict_proba(meta_features)[:, 1]
        else:
            # 메타 러너가 없으면 단순 평균 사용
            return np.mean(meta_features, axis=1)
    
    def _train_meta_learner(self, X_val: pd.DataFrame, y_val: pd.Series):
        """메타 러너 훈련 (스태킹용)"""
        print("🔄 메타 러너 훈련 중...")
        
        # 개별 모델들의 검증 예측을 메타 특성으로 사용
        meta_features = []
        for model_key, model in self.models.items():
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            meta_features.append(y_pred_proba)
        
        meta_features = np.column_stack(meta_features)
        
        # 로지스틱 회귀를 메타 러너로 사용
        self.meta_learner = LogisticRegression(
            random_state=self.config.get('random_state', 42),
            max_iter=1000
        )
        self.meta_learner.fit(meta_features, y_val)
        
        print("✅ 메타 러너 훈련 완료")
    
    def find_optimal_threshold(self, X_val: pd.DataFrame, y_val: pd.Series, 
                              metric: str = 'f1') -> Tuple[float, Dict]:
        """
        앙상블의 최적 threshold 찾기
        
        Args:
            X_val: 검증 특성 데이터
            y_val: 검증 타겟 데이터
            metric: 최적화할 메트릭
            
        Returns:
            Tuple[float, Dict]: (최적 threshold, threshold 분석 결과)
        """
        print(f"🎯 앙상블 최적 Threshold 탐색 ({metric.upper()} 기준)")
        
        # 앙상블 예측
        y_pred_proba = self.ensemble_predict_proba(X_val, X_val, y_val)
        
        # 다양한 threshold에서의 성능 계산
        thresholds = np.arange(0.05, 0.5, 0.05)
        threshold_results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if len(np.unique(y_pred)) == 1:
                continue
            
            try:
                metrics = {
                    'threshold': threshold,
                    'precision': precision_score(y_val, y_pred, zero_division=0),
                    'recall': recall_score(y_val, y_pred, zero_division=0),
                    'f1': f1_score(y_val, y_pred, zero_division=0),
                    'balanced_accuracy': balanced_accuracy_score(y_val, y_pred)
                }
                threshold_results.append(metrics)
            except:
                continue
        
        if not threshold_results:
            print("⚠️ 최적 threshold 찾기 실패, 기본값 0.5 사용")
            return 0.5, {}
        
        # 결과를 DataFrame으로 변환
        threshold_df = pd.DataFrame(threshold_results)
        
        # 최적 threshold 찾기
        if metric == 'average_precision':
            # average_precision은 임계값과 무관하므로 F1 기준으로 threshold 선택
            optimal_value = average_precision_score(y_val, y_pred_proba)
            if 'f1' in threshold_df.columns:
                best_idx = threshold_df['f1'].idxmax()
                optimal_threshold = threshold_df.loc[best_idx, 'threshold']
            else:
                optimal_threshold = 0.5
        else:
            best_idx = threshold_df[metric].idxmax()
            optimal_threshold = threshold_df.loc[best_idx, 'threshold']
            optimal_value = threshold_df.loc[best_idx, metric]
        
        print(f"✅ 최적 Threshold: {optimal_threshold:.3f} ({metric.upper()}: {optimal_value:.4f})")
        
        # Precision-Recall 곡선 데이터
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_val, y_pred_proba)
        
        threshold_analysis = {
            'all_thresholds': threshold_results,
            'optimal_threshold': optimal_threshold,
            'optimal_metric': metric,
            'optimal_value': optimal_value,
            'pr_curve': {
                'precision': precision_vals.tolist(),
                'recall': recall_vals.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        }
        
        return optimal_threshold, threshold_analysis
    
    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series, 
                         threshold: float = 0.5) -> Dict[str, float]:
        """
        앙상블 모델 평가
        
        Args:
            X_test: 테스트 특성 데이터
            y_test: 테스트 타겟 데이터
            threshold: 분류 threshold
            
        Returns:
            Dict[str, float]: 성능 메트릭들
        """
        print(f"📊 앙상블 모델 평가 (Threshold: {threshold:.3f})")
        
        # 예측
        y_pred_proba = self.ensemble_predict_proba(X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # 성능 메트릭 계산 - modeling_pipeline 호환 키 이름 사용
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision_optimal': precision_score(y_test, y_pred, zero_division=0),
            'recall_optimal': recall_score(y_test, y_pred, zero_division=0),
            'f1_optimal': f1_score(y_test, y_pred, zero_division=0),
            'balanced_accuracy_optimal': balanced_accuracy_score(y_test, y_pred),
            'average_precision': average_precision_score(y_test, y_pred_proba),
            # 기존 키들도 유지 (하위 호환성)
            'auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
        }
        
        print("📈 앙상블 성능:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.upper()}: {value:.4f}")
        
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        시각화용 predict_proba 메서드
        
        Args:
            X: 예측할 특성 데이터
            
        Returns:
            np.ndarray: 앙상블 예측 확률 (시각화 호환을 위해 확률값만 반환)
        """
        return self.ensemble_predict_proba(X)
    
    def create_ensemble_report(self, output_dir):
        """앙상블 시각화 리포트 생성"""
        print("📊 앙상블 리포트 생성 중...")
        
        # 가중치 시각화
        if self.weights:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            models = list(self.weights.keys())
            weights = list(self.weights.values())
            
            # 모델명 간소화
            model_names = [model.split('_')[0] for model in models]
            
            bars = ax.bar(model_names, weights, color=['blue', 'red', 'green', 'orange', 'purple'][:len(models)])
            
            # 값 표시
            for bar, weight in zip(bars, weights):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('앙상블 모델 가중치', fontsize=14, fontweight='bold')
            ax.set_ylabel('가중치')
            ax.set_xlabel('모델')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(weights) * 1.2)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'ensemble_weights.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ 앙상블 가중치 시각화 저장: ensemble_weights.png")
        
        print("✅ 앙상블 리포트 생성 완료")