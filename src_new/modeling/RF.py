"""
Random Forest 모델 - Optuna 하이퍼파라미터 튜닝 및 K-fold CV
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import optuna
from optuna.samplers import TPESampler
import joblib

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic' if plt.matplotlib.get_backend() != 'Agg' else 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class RandomForestOptimizer:
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
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': self.random_state,
            'n_jobs': -1,
            'class_weight': 'balanced'  # 불균형 데이터 처리
        }
        
        # bootstrap=False인 경우 oob_score는 사용할 수 없음
        if params['bootstrap']:
            params['oob_score'] = trial.suggest_categorical('oob_score', [True, False])
        
        # 모델 생성
        model = RandomForestClassifier(**params)
        
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
        print("🌳 Random Forest 하이퍼파라미터 최적화 시작")
        print("="*60)
        
        # Optuna study 생성
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='random_forest_optimization'
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
    
    def train_best_model(self, X_train, y_train, X_valid, y_valid):
        """최적 파라미터로 모델 훈련"""
        print("\n🚀 최적 모델 훈련")
        print("="*60)
        
        # 최적 모델 생성
        self.best_model = RandomForestClassifier(**self.best_params)
        self.best_model.fit(X_train, y_train)
        
        # 검증 데이터 예측
        y_valid_pred = self.best_model.predict(X_valid)
        y_valid_proba = self.best_model.predict_proba(X_valid)[:, 1]
        
        # 성능 평가
        metrics = {
            'auc': roc_auc_score(y_valid, y_valid_proba),
            'precision': precision_score(y_valid, y_valid_pred),
            'recall': recall_score(y_valid, y_valid_pred),
            'f1': f1_score(y_valid, y_valid_pred)
        }
        
        print(f"📈 검증 데이터 성능:")
        print(f"   AUC: {metrics['auc']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1']:.4f}")
        
        # OOB Score 출력 (가능한 경우)
        if hasattr(self.best_model, 'oob_score_') and self.best_model.oob_score_ is not None:
            print(f"   OOB Score: {self.best_model.oob_score_:.4f}")
        
        return metrics
    
    def evaluate_test(self, X_test, y_test):
        """테스트 데이터 평가"""
        print("\n🎯 테스트 데이터 최종 평가")
        print("="*60)
        
        # 예측
        y_test_pred = self.best_model.predict(X_test)
        y_test_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # 성능 지표
        test_metrics = {
            'auc': roc_auc_score(y_test, y_test_proba),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred)
        }
        
        print(f"📊 테스트 성능:")
        for metric, value in test_metrics.items():
            print(f"   {metric.upper()}: {value:.4f}")
        
        # 분류 리포트
        print(f"\n📋 상세 분류 리포트:")
        print(classification_report(y_test, y_test_pred))
        
        return test_metrics, y_test_pred, y_test_proba
    
    def plot_results(self, X_test, y_test, feature_names, save_path='outputs/visualizations/'):
        """결과 시각화"""
        print("\n📊 결과 시각화 생성")
        print("="*60)
        
        # 예측 결과
        y_test_pred = self.best_model.predict(X_test)
        y_test_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # 시각화 설정
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Random Forest 모델 결과', fontsize=16, fontweight='bold')
        
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
        
        axes[1, 0].barh(importance_df['feature'], importance_df['importance'], color='green')
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
        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/random_forest_results.png', dpi=300, bbox_inches='tight')
        print(f"✅ 시각화 저장: {save_path}/random_forest_results.png")
        
        plt.show()
    
    def save_results(self, test_metrics, feature_names, save_path='outputs/models/'):
        """결과 저장"""
        print("\n💾 결과 저장")
        print("="*60)
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 모델 저장
        model_path = f'{save_path}/random_forest_best_model.joblib'
        joblib.dump(self.best_model, model_path)
        print(f"✅ 모델 저장: {model_path}")
        
        # 2. 결과 정보 저장
        results = {
            'model_name': 'RandomForest',
            'optimization_date': datetime.now().isoformat(),
            'best_params': self.best_params,
            'cv_best_score': self.best_score,
            'test_metrics': test_metrics,
            'feature_names': feature_names,
            'feature_importances': {
                name: float(importance) for name, importance in 
                zip(feature_names, self.best_model.feature_importances_)
            }
        }
        
        # OOB Score 추가 (가능한 경우)
        if hasattr(self.best_model, 'oob_score_') and self.best_model.oob_score_ is not None:
            results['oob_score'] = float(self.best_model.oob_score_)
        
        results_path = f'{save_path}/random_forest_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ 결과 저장: {results_path}")
        
        return results

def main():
    """메인 실행 함수"""
    print("🏢 한국 기업 부실예측 - Random Forest 모델")
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
    optimizer = RandomForestOptimizer(random_state=42)
    best_params, best_score = optimizer.optimize(X_train, y_train, n_trials=100, cv_folds=5)
    
    # 3. 최적 모델 훈련
    valid_metrics = optimizer.train_best_model(X_train, y_train, X_valid, y_valid)
    
    # 4. 테스트 평가
    test_metrics, y_pred, y_proba = optimizer.evaluate_test(X_test, y_test)
    
    # 5. 시각화
    optimizer.plot_results(X_test, y_test, feature_names)
    
    # 6. 결과 저장
    results = optimizer.save_results(test_metrics, feature_names)
    
    print("\n🎉 Random Forest 모델링 완료!")
    print(f"📊 최종 테스트 AUC: {test_metrics['auc']:.4f}")

if __name__ == "__main__":
    main()
