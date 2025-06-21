"""
XGBoost 모델 - 100% 완성도 데이터용 Optuna 하이퍼파라미터 튜닝
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

class XGBoostOptimizer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.study = None
        
    def objective(self, trial, X_train, y_train, cv_folds=5):
        """Optuna 목적 함수"""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # booster에 따른 추가 파라미터
        if params['booster'] == 'gbtree' or params['booster'] == 'dart':
            params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)
            params['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
        
        if params['booster'] == 'dart':
            params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            params['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
            params['one_drop'] = trial.suggest_categorical('one_drop', [0, 1])
        
        # 모델 생성
        model = xgb.XGBClassifier(**params)
        
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
        print("🔍 XGBoost 하이퍼파라미터 최적화 시작")
        print("="*60)
        
        # Optuna study 생성
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='xgboost_100_optimization'
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
        self.best_model = xgb.XGBClassifier(**self.best_params)
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

def run_optimization(data_type='smote'):
    """모델 최적화 실행"""
    print(f"=== 100% 완성도 데이터 XGBoost 최적화 ({data_type.upper()}) ===")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. 데이터 로드
    print("📂 데이터 로드 중...")
    
    X_train = pd.read_csv(f'data_new/final/X_train_100_{data_type}.csv')
    X_valid = pd.read_csv(f'data_new/final/X_valid_100_{data_type}.csv')
    X_test = pd.read_csv(f'data_new/final/X_test_100_{data_type}.csv')
    
    y_train = pd.read_csv(f'data_new/final/y_train_100_{data_type}.csv').iloc[:, 0]
    y_valid = pd.read_csv(f'data_new/final/y_valid_100_{data_type}.csv').iloc[:, 0]
    y_test = pd.read_csv(f'data_new/final/y_test_100_{data_type}.csv').iloc[:, 0]
    
    print(f"✅ 데이터 로드 완료")
    print(f"   Train: {X_train.shape}, 부실 비율: {y_train.mean():.4f}")
    print(f"   Valid: {X_valid.shape}, 부실 비율: {y_valid.mean():.4f}")
    print(f"   Test: {X_test.shape}, 부실 비율: {y_test.mean():.4f}")
    
    # 2. 모델 최적화
    optimizer = XGBoostOptimizer(random_state=42)
    
    # 하이퍼파라미터 최적화
    best_params, best_score = optimizer.optimize(X_train, y_train, n_trials=100)
    
    # 최적 모델 훈련
    valid_metrics = optimizer.train_best_model(X_train, y_train, X_valid, y_valid)
    
    # 테스트 평가
    test_metrics, y_test_pred, y_test_proba = optimizer.evaluate_test(X_test, y_test)
    
    # 3. 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 모델 저장
    model_path = f'outputs/models/xgboost_100_{data_type}_{timestamp}.joblib'
    joblib.dump(optimizer.best_model, model_path)
    
    # 결과 저장
    results = {
        'model_name': 'XGBoost',
        'data_type': data_type,
        'timestamp': timestamp,
        'best_params': best_params,
        'cv_score': best_score,
        'validation_metrics': valid_metrics,
        'test_metrics': test_metrics,
        'data_info': {
            'train_shape': X_train.shape,
            'valid_shape': X_valid.shape,
            'test_shape': X_test.shape,
            'train_default_ratio': float(y_train.mean()),
            'valid_default_ratio': float(y_valid.mean()),
            'test_default_ratio': float(y_test.mean())
        }
    }
    
    results_path = f'outputs/reports/xgboost_100_{data_type}_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 결과 저장 완료:")
    print(f"   모델: {model_path}")
    print(f"   결과: {results_path}")
    
    return results

def main():
    """메인 실행 함수"""
    # Normal 데이터로 최적화
    print("🔄 Normal 데이터 최적화 시작")
    normal_results = run_optimization('normal')
    
    print("\n" + "="*80)
    
    # SMOTE 데이터로 최적화  
    print("🔄 SMOTE 데이터 최적화 시작")
    smote_results = run_optimization('smote')
    
    # 최종 비교
    print("\n" + "="*80)
    print("📊 최종 결과 비교")
    print("="*80)
    
    print(f"Normal 데이터:")
    print(f"   CV AUC: {normal_results['cv_score']:.4f}")
    print(f"   Test AUC: {normal_results['test_metrics']['auc']:.4f}")
    print(f"   Test F1: {normal_results['test_metrics']['f1']:.4f}")
    
    print(f"\nSMOTE 데이터:")
    print(f"   CV AUC: {smote_results['cv_score']:.4f}")
    print(f"   Test AUC: {smote_results['test_metrics']['auc']:.4f}")
    print(f"   Test F1: {smote_results['test_metrics']['f1']:.4f}")
    
    print(f"\n완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 