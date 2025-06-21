"""
XGBoost 모델 - Optuna 하이퍼파라미터 튜닝 및 K-fold CV
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import optuna
from optuna.samplers import TPESampler
import joblib
from typing import Dict, Any
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic' if plt.matplotlib.get_backend() != 'Agg' else 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class XGBoostOptimizer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.study = None
        self.cv_results = {}
        
    def objective(self, trial, X_train, y_train, cv_folds=5):
        """Optuna 목적 함수"""
        # 하이퍼파라미터 탐색 공간
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': 'gbtree',
            'random_state': self.random_state,
            'verbosity': 0,
            
            # 학습률 및 부스팅 관련
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            
            # 트리 구조 관련
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            
            # 정규화 관련
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            
            # 샘플링 관련
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            
            # 불균형 데이터 처리
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }
        
        # 모델 생성
        model = XGBClassifier(**params)
        
        # Stratified K-fold Cross Validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # AUC 점수 계산
        auc_scores = cross_val_score(
            model, X_train, y_train, 
            cv=skf, scoring='roc_auc', n_jobs=-1
        )
        
        return auc_scores.mean()
    
    def optimize(self, X_train, y_train, n_trials=100, cv_folds=5):
        """하이퍼파라미터 최적화"""
        print("🚀 XGBoost 하이퍼파라미터 최적화 시작")
        print("="*60)
        
        # Optuna study 생성
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='xgboost_optimization'
        )
        
        # 최적화 실행
        self.study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, cv_folds),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # 최적 파라미터 저장
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print(f"✅ 최적화 완료!")
        print(f"📊 최적 AUC: {self.best_score:.4f}")
        print(f"🎯 최적 하이퍼파라미터:")
        for key, value in self.best_params.items():
            print(f"   {key}: {value}")
        
        return self.best_params, self.best_score
    
    def train_best_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_val: pd.DataFrame, y_val: pd.Series, 
                        best_params: Dict[str, Any]) -> XGBClassifier:
        """Train the best model with optimized parameters."""
        print("🎯 최적 파라미터로 모델 훈련")
        print(f"   파라미터: {best_params}")
        
        # 최적 모델 훈련
        model = XGBClassifier(**best_params)
        
        # Early stopping을 위한 evaluation set
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        print("✅ 모델 훈련 완료")
        return model
    
    def evaluate_test(self, X_test: pd.DataFrame, y_test: pd.Series, model: XGBClassifier = None) -> Dict[str, float]:
        """테스트 데이터 평가"""
        if model is None:
            model = self.best_model
            
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # 성능 지표 계산
        metrics = {
            'auc': roc_auc_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        print(f"🎯 테스트 데이터 성능:")
        print(f"   AUC: {metrics['auc']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1']:.4f}")
        
        return metrics
    
    def plot_results(self, X_test, y_test, feature_names, save_path='outputs/visualizations/'):
        """결과 시각화"""
        print("\n📊 결과 시각화 생성")
        print("="*60)
        
        # 예측 결과
        y_test_pred = self.best_model.predict(X_test)
        y_test_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # 시각화 설정
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('XGBoost 모델 결과', fontsize=16, fontweight='bold')
        
        # 1. ROC 곡선
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.4f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC 곡선')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True)
        
        # 2. 혼동 행렬
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title('혼동 행렬')
        axes[0, 1].set_xlabel('예측값')
        axes[0, 1].set_ylabel('실제값')
        
        # 3. 특성 중요도 (상위 10개)
        importances = self.best_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False).head(10)
        
        axes[1, 0].barh(importance_df['feature'], importance_df['importance'], color='purple')
        axes[1, 0].set_title('특성 중요도 (상위 10개)')
        axes[1, 0].set_xlabel('중요도')
        
        # 4. 최적화 히스토리
        if hasattr(self.study, 'trials'):
            trial_values = [trial.value for trial in self.study.trials if trial.value is not None]
            axes[1, 1].plot(trial_values, 'b-', alpha=0.7)
            axes[1, 1].axhline(y=max(trial_values), color='r', linestyle='--', 
                              label=f'Best: {max(trial_values):.4f}')
            axes[1, 1].set_title('최적화 히스토리')
            axes[1, 1].set_xlabel('Trial')
            axes[1, 1].set_ylabel('AUC Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/xgboost_results.png', dpi=300, bbox_inches='tight')
        print(f"✅ 시각화 저장: {save_path}/xgboost_results.png")
        
        plt.show()
    
    def plot_learning_curves(self, save_path='outputs/visualizations/'):
        """학습 곡선 시각화"""
        if hasattr(self.best_model, 'evals_result_'):
            print("\n📈 학습 곡선 시각화")
            
            results = self.best_model.evals_result_
            epochs = len(results['train']['auc'])
            x_axis = range(0, epochs)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_axis, results['train']['auc'], label='Train')
            ax.plot(x_axis, results['valid']['auc'], label='Valid')
            ax.axvline(x=self.best_model.best_iteration, color='red', linestyle='--', 
                      label=f'Best Iteration ({self.best_model.best_iteration})')
            ax.legend()
            ax.set_ylabel('AUC')
            ax.set_xlabel('Epoch')
            ax.set_title('XGBoost 학습 곡선')
            ax.grid(True)
            
            # 저장
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f'{save_path}/xgboost_learning_curves.png', dpi=300, bbox_inches='tight')
            print(f"✅ 학습 곡선 저장: {save_path}/xgboost_learning_curves.png")
            
            plt.show()
    
    def save_results(self, test_metrics: Dict[str, float], feature_names: list, 
                    model: XGBClassifier = None, best_params: Dict[str, Any] = None):
        """결과 저장"""
        if model is None:
            model = self.best_model
            
        # 모델 저장
        model_path = 'outputs/models/xgboost_best_model.joblib'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"✅ 모델 저장: {model_path}")
        
        # 결과 저장
        results = {
            'model_type': 'XGBoost',
            'best_params': best_params if best_params else self.best_params,
            'test_metrics': test_metrics,
            'feature_names': feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = 'outputs/models/xgboost_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 결과 저장: {results_path}")
        
        return results

def main():
    """메인 실행 함수"""
    print("🏢 한국 기업 부실예측 - XGBoost 모델")
    print("="*60)
    
    # 1. 데이터 로드
    print("📂 데이터 로드")
    
    # SMOTE 버전 데이터 로드
    X_train = pd.read_csv('data_new/final/X_train_smote.csv')
    X_valid = pd.read_csv('data_new/final/X_valid_smote.csv')
    X_test = pd.read_csv('data_new/final/X_test_smote.csv')
    
    y_train = pd.read_csv('data_new/final/y_train_smote.csv').iloc[:, 0]
    y_valid = pd.read_csv('data_new/final/y_valid_smote.csv').iloc[:, 0]
    y_test = pd.read_csv('data_new/final/y_test_smote.csv').iloc[:, 0]
    
    feature_names = X_train.columns.tolist()
    
    print(f"✅ 데이터 로드 완료")
    print(f"   훈련: {X_train.shape}, 검증: {X_valid.shape}, 테스트: {X_test.shape}")
    print(f"   특성 수: {len(feature_names)}")
    print(f"   훈련 부실 비율: {y_train.mean():.2%}")
    
    # 2. 모델 최적화
    optimizer = XGBoostOptimizer(random_state=42)
    best_params, best_score = optimizer.optimize(X_train, y_train, n_trials=100, cv_folds=5)
    
    # 최적 모델 훈련
    print("\n🚀 최적 모델 훈련 (Early Stopping)")
    print("="*60)
    best_model = optimizer.train_best_model(X_train, y_train, X_valid, y_valid, best_params)
    
    # 검증 데이터 평가
    print("\n📊 검증 데이터 성능 평가")
    print("="*60)
    y_valid_pred = best_model.predict(X_valid)
    y_valid_proba = best_model.predict_proba(X_valid)[:, 1]
    
    valid_metrics = {
        'auc': roc_auc_score(y_valid, y_valid_proba),
        'precision': precision_score(y_valid, y_valid_pred),
        'recall': recall_score(y_valid, y_valid_pred),
        'f1': f1_score(y_valid, y_valid_pred),
        'cv_score': best_score
    }
    
    print(f"📈 검증 데이터 성능:")
    print(f"   AUC: {valid_metrics['auc']:.4f}")
    print(f"   Precision: {valid_metrics['precision']:.4f}")
    print(f"   Recall: {valid_metrics['recall']:.4f}")
    print(f"   F1-Score: {valid_metrics['f1']:.4f}")
    print(f"   CV Score: {valid_metrics['cv_score']:.4f}")
    
    # 테스트 데이터 평가
    print("\n🎯 테스트 데이터 최종 평가")
    print("="*60)
    test_metrics = optimizer.evaluate_test(X_test, y_test, best_model)
    
    # 모델 저장
    print("\n💾 모델 및 결과 저장")
    print("="*60)
    optimizer.save_results(test_metrics, feature_names, best_model, best_params)
    
    print("\n🎉 XGBoost 모델링 완료!")
    print("📊 모든 결과는 outputs/ 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
