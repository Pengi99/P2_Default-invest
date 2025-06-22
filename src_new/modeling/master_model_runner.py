"""
마스터 모델 실행 스크립트
Normal/SMOTE 데이터에 대해 3개 모델(LogisticRegression, RandomForest, XGBoost)을 자동 실행
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic' if plt.matplotlib.get_backend() != 'Agg' else 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, balanced_accuracy_score
)
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

class MasterModelRunner:
    def __init__(self, config):
        """
        마스터 모델 러너 초기화
        
        Args:
            config (dict): 설정 딕셔너리
        """
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{config['run_name']}_{self.timestamp}"
        
        # 결과 저장 경로 설정
        self.output_dir = Path(config['output_base_dir']) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 하위 디렉토리 생성
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # 데이터 저장 변수
        self.data = {}
        self.models = {}
        self.results = {}
        self.selected_features = None
        
        print(f"🚀 마스터 모델 러너 초기화 완료")
        print(f"📁 실행 이름: {self.run_name}")
        print(f"📁 출력 경로: {self.output_dir}")
        
    def load_data(self):
        """데이터 로드"""
        print("\n📂 데이터 로드")
        print("="*60)
        
        data_path = Path(self.config['data_path'])
        
        # Normal 데이터
        self.data['normal'] = {
            'X_train': pd.read_csv(data_path / 'X_train_100_normal.csv'),
            'X_valid': pd.read_csv(data_path / 'X_valid_100_normal.csv'),
            'X_test': pd.read_csv(data_path / 'X_test_100_normal.csv'),
            'y_train': pd.read_csv(data_path / 'y_train_100_normal.csv').iloc[:, 0],
            'y_valid': pd.read_csv(data_path / 'y_valid_100_normal.csv').iloc[:, 0],
            'y_test': pd.read_csv(data_path / 'y_test_100_normal.csv').iloc[:, 0]
        }
        
        # SMOTE 데이터
        self.data['smote'] = {
            'X_train': pd.read_csv(data_path / 'X_train_100_smote.csv'),
            'X_valid': pd.read_csv(data_path / 'X_valid_100_smote.csv'),
            'X_test': pd.read_csv(data_path / 'X_test_100_smote.csv'),
            'y_train': pd.read_csv(data_path / 'y_train_100_smote.csv').iloc[:, 0],
            'y_valid': pd.read_csv(data_path / 'y_valid_100_smote.csv').iloc[:, 0],
            'y_test': pd.read_csv(data_path / 'y_test_100_smote.csv').iloc[:, 0]
        }
        
        for data_type in ['normal', 'smote']:
            data = self.data[data_type]
            print(f"✅ {data_type.upper()} 데이터:")
            print(f"   Train: {data['X_train'].shape}, 부실비율: {data['y_train'].mean():.2%}")
            print(f"   Valid: {data['X_valid'].shape}, 부실비율: {data['y_valid'].mean():.2%}")
            print(f"   Test: {data['X_test'].shape}, 부실비율: {data['y_test'].mean():.2%}")
    
    def apply_lasso_feature_selection(self, data_type):
        """Lasso 특성 선택 적용"""
        if not self.config['lasso']['enabled']:
            return
            
        print(f"\n🔍 Lasso 특성 선택 적용 ({data_type.upper()})")
        print("="*60)
        
        X_train = self.data[data_type]['X_train']
        y_train = self.data[data_type]['y_train']
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Lasso CV
        lasso_cv = LassoCV(
            alphas=self.config['lasso']['alphas'],
            cv=self.config['lasso']['cv_folds'],
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        lasso_cv.fit(X_train_scaled, y_train)
        
        # 특성 선택
        threshold = self.config['lasso']['threshold']
        if threshold == 'median':
            threshold_value = np.median(np.abs(lasso_cv.coef_[lasso_cv.coef_ != 0]))
        elif threshold == 'mean':
            threshold_value = np.mean(np.abs(lasso_cv.coef_[lasso_cv.coef_ != 0]))
        else:
            threshold_value = float(threshold)
        
        selected_mask = np.abs(lasso_cv.coef_) >= threshold_value
        selected_features = X_train.columns[selected_mask].tolist()
        
        print(f"✅ 최적 alpha: {lasso_cv.alpha_:.6f}")
        print(f"📊 선택된 특성: {len(selected_features)}/{len(X_train.columns)}")
        print(f"🎯 선택된 특성: {selected_features}")
        
        # 데이터 업데이트
        for split in ['X_train', 'X_valid', 'X_test']:
            self.data[data_type][split] = self.data[data_type][split][selected_features]
        
        # 결과 저장
        lasso_results = {
            'optimal_alpha': float(lasso_cv.alpha_),
            'threshold_value': float(threshold_value),
            'original_features': len(X_train.columns),
            'selected_features': len(selected_features),
            'selected_feature_names': selected_features,
            'feature_coefficients': dict(zip(X_train.columns, lasso_cv.coef_))
        }
        
        # results 디렉토리 존재 확인 및 생성
        results_dir = self.output_dir / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / f'lasso_selection_{data_type}.json', 'w') as f:
            json.dump(lasso_results, f, indent=2, ensure_ascii=False)
    
    def optimize_logistic_regression(self, data_type):
        """로지스틱 회귀 최적화"""
        print(f"\n🔍 로지스틱 회귀 최적화 ({data_type.upper()})")
        print("="*60)
        
        X_train = self.data[data_type]['X_train']
        y_train = self.data[data_type]['y_train']
        
        def objective(trial):
            # penalty와 solver를 하나의 조합으로 선택 (동적 선택지 문제 해결)
            penalty_solver_combinations = []
            
            # 가능한 모든 penalty-solver 조합 생성
            for penalty in self.config['models']['logistic']['penalty']:
                if penalty == 'l1':
                    for solver in ['liblinear', 'saga']:
                        penalty_solver_combinations.append(f"{penalty}_{solver}")
                elif penalty == 'l2':
                    for solver in self.config['models']['logistic']['l2_solvers']:
                        if solver in ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']:
                            penalty_solver_combinations.append(f"{penalty}_{solver}")
                elif penalty == 'elasticnet':
                    penalty_solver_combinations.append(f"{penalty}_saga")
            
            # 조합 선택
            combination = trial.suggest_categorical('penalty_solver', penalty_solver_combinations)
            penalty, solver = combination.split('_', 1)
            
            C = trial.suggest_float('C', *self.config['models']['logistic']['C_range'], log=True)
            max_iter = trial.suggest_int('max_iter', *self.config['models']['logistic']['max_iter_range'])
            
            params = {
                'penalty': penalty,
                'C': C,
                'solver': solver,
                'max_iter': max_iter,
                'random_state': self.config['random_state']
            }
            
            if penalty == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', *self.config['models']['logistic']['l1_ratio_range'])
            
            model = LogisticRegression(**params)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['random_state'])
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config['random_state']))
        study.optimize(objective, n_trials=self.config['models']['logistic']['n_trials'])
        
        # 최적 모델 훈련
        best_params = study.best_params.copy()
        
        # penalty_solver 조합을 분리
        penalty_solver = best_params.pop('penalty_solver')
        penalty, solver = penalty_solver.split('_', 1)
        
        # 분리된 penalty와 solver 추가
        best_params['penalty'] = penalty
        best_params['solver'] = solver
        
        model = LogisticRegression(**best_params)
        model.fit(X_train, y_train)
        
        # 저장
        model_key = f'LogisticRegression_{data_type}'
        self.models[model_key] = model
        self.results[model_key] = {
            'best_params': best_params,
            'cv_score': study.best_value,
            'data_type': data_type
        }
        
        print(f"✅ 최적 AUC: {study.best_value:.4f}")
        print(f"📊 최적 파라미터: {best_params}")
        
        return model
    
    def optimize_random_forest(self, data_type):
        """랜덤 포레스트 최적화"""
        print(f"\n🌲 랜덤 포레스트 최적화 ({data_type.upper()})")
        print("="*60)
        
        X_train = self.data[data_type]['X_train']
        y_train = self.data[data_type]['y_train']
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *self.config['models']['random_forest']['n_estimators_range']),
                'max_depth': trial.suggest_int('max_depth', *self.config['models']['random_forest']['max_depth_range']),
                'min_samples_split': trial.suggest_int('min_samples_split', *self.config['models']['random_forest']['min_samples_split_range']),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', *self.config['models']['random_forest']['min_samples_leaf_range']),
                'max_features': trial.suggest_float('max_features', *self.config['models']['random_forest']['max_features_range']),
                'random_state': self.config['random_state'],
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['random_state'])
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config['random_state']))
        study.optimize(objective, n_trials=self.config['models']['random_forest']['n_trials'])
        
        # 최적 모델 훈련
        best_params = study.best_params
        model = RandomForestClassifier(**best_params)
        model.fit(X_train, y_train)
        
        # 저장
        model_key = f'RandomForest_{data_type}'
        self.models[model_key] = model
        self.results[model_key] = {
            'best_params': best_params,
            'cv_score': study.best_value,
            'data_type': data_type,
            'feature_importances': dict(zip(X_train.columns, model.feature_importances_))
        }
        
        print(f"✅ 최적 AUC: {study.best_value:.4f}")
        print(f"📊 최적 파라미터: {best_params}")
        
        return model
    
    def optimize_xgboost(self, data_type):
        """XGBoost 최적화"""
        print(f"\n🚀 XGBoost 최적화 ({data_type.upper()})")
        print("="*60)
        
        X_train = self.data[data_type]['X_train']
        y_train = self.data[data_type]['y_train']
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_estimators': trial.suggest_int('n_estimators', *self.config['models']['xgboost']['n_estimators_range']),
                'max_depth': trial.suggest_int('max_depth', *self.config['models']['xgboost']['max_depth_range']),
                'learning_rate': trial.suggest_float('learning_rate', *self.config['models']['xgboost']['learning_rate_range']),
                'subsample': trial.suggest_float('subsample', *self.config['models']['xgboost']['subsample_range']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *self.config['models']['xgboost']['colsample_bytree_range']),
                'reg_alpha': trial.suggest_float('reg_alpha', *self.config['models']['xgboost']['reg_alpha_range']),
                'reg_lambda': trial.suggest_float('reg_lambda', *self.config['models']['xgboost']['reg_lambda_range']),
                'random_state': self.config['random_state'],
                'n_jobs': -1,
                'verbosity': 0
            }
            
            model = xgb.XGBClassifier(**params)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['random_state'])
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config['random_state']))
        study.optimize(objective, n_trials=self.config['models']['xgboost']['n_trials'])
        
        # 최적 모델 훈련
        best_params = study.best_params
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        
        # 저장
        model_key = f'XGBoost_{data_type}'
        self.models[model_key] = model
        self.results[model_key] = {
            'best_params': best_params,
            'cv_score': study.best_value,
            'data_type': data_type,
            'feature_importances': dict(zip(X_train.columns, model.feature_importances_))
        }
        
        print(f"✅ 최적 AUC: {study.best_value:.4f}")
        print(f"📊 최적 파라미터: {best_params}")
        
        return model
    
    def find_optimal_threshold(self, model_key):
        """각 모델별 최적 threshold 찾기"""
        print(f"\n🎯 {model_key} 최적 Threshold 탐색")
        print("="*60)
        
        model = self.models[model_key]
        data_type = self.results[model_key]['data_type']
        
        X_valid = self.data[data_type]['X_valid']
        y_valid = self.data[data_type]['y_valid']
        
        # 검증 데이터에 대한 예측 확률
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        
        # Precision-Recall 곡선과 다양한 threshold에서의 성능 계산
        thresholds = np.arange(0.1, 0.9, 0.05)  # 0.1부터 0.85까지 0.05 간격
        
        threshold_results = []
        
        for threshold in thresholds:
            y_valid_pred = (y_valid_proba >= threshold).astype(int)
            
            # 예측값이 모두 0이거나 1인 경우 스킵
            if len(np.unique(y_valid_pred)) == 1:
                continue
            
            try:
                metrics = {
                    'threshold': threshold,
                    'precision': precision_score(y_valid, y_valid_pred, zero_division=0),
                    'recall': recall_score(y_valid, y_valid_pred, zero_division=0),
                    'f1': f1_score(y_valid, y_valid_pred, zero_division=0),
                    'balanced_accuracy': balanced_accuracy_score(y_valid, y_valid_pred)
                }
                threshold_results.append(metrics)
            except:
                continue
        
        if not threshold_results:
            print("⚠️ 최적 threshold 찾기 실패, 기본값 0.5 사용")
            return 0.5, {}
        
        # 결과를 DataFrame으로 변환
        threshold_df = pd.DataFrame(threshold_results)
        
        # 주요 메트릭별 최적 threshold 찾기
        metric_priority = self.config.get('threshold_optimization', {}).get('metric_priority', 'f1')
        
        optimal_thresholds = {}
        for metric in ['f1', 'precision', 'recall', 'balanced_accuracy']:
            if metric in threshold_df.columns:
                best_idx = threshold_df[metric].idxmax()
                optimal_thresholds[metric] = {
                    'threshold': threshold_df.loc[best_idx, 'threshold'],
                    'value': threshold_df.loc[best_idx, metric]
                }
        
        # 우선순위 메트릭으로 최종 threshold 선택
        if metric_priority in optimal_thresholds:
            final_threshold = optimal_thresholds[metric_priority]['threshold']
            final_value = optimal_thresholds[metric_priority]['value']
        else:
            final_threshold = optimal_thresholds['f1']['threshold']
            final_value = optimal_thresholds['f1']['value']
        
        print(f"📈 Threshold 최적화 결과:")
        for metric, result in optimal_thresholds.items():
            marker = "🎯" if metric == metric_priority else "  "
            print(f"{marker} {metric.upper()}: {result['threshold']:.3f} (값: {result['value']:.4f})")
        
        print(f"\n✅ 최종 선택: {final_threshold:.3f} ({metric_priority.upper()}: {final_value:.4f})")
        
        # Precision-Recall 곡선 데이터 저장
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_valid, y_valid_proba)
        
        threshold_analysis = {
            'all_thresholds': threshold_results,
            'optimal_by_metric': optimal_thresholds,
            'final_threshold': final_threshold,
            'final_metric': metric_priority,
            'final_value': final_value,
            'pr_curve': {
                'precision': precision_vals.tolist(),
                'recall': recall_vals.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        }
        
        return final_threshold, threshold_analysis
    
    def evaluate_model(self, model_key):
        """모델 평가 (최적 threshold 사용)"""
        # 최적 threshold 찾기
        optimal_threshold, threshold_analysis = self.find_optimal_threshold(model_key)
        
        model = self.models[model_key]
        data_type = self.results[model_key]['data_type']
        
        X_valid = self.data[data_type]['X_valid']
        y_valid = self.data[data_type]['y_valid']
        X_test = self.data[data_type]['X_test']
        y_test = self.data[data_type]['y_test']
        
        # 예측 (최적 threshold 사용)
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        y_valid_pred = (y_valid_proba >= optimal_threshold).astype(int)
        
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # 검증 성능
        valid_metrics = {
            'auc': roc_auc_score(y_valid, y_valid_proba),
            'precision': precision_score(y_valid, y_valid_pred, zero_division=0),
            'recall': recall_score(y_valid, y_valid_pred, zero_division=0),
            'f1': f1_score(y_valid, y_valid_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_valid, y_valid_pred),
            'average_precision': average_precision_score(y_valid, y_valid_proba)
        }
        
        # 테스트 성능
        test_metrics = {
            'auc': roc_auc_score(y_test, y_test_proba),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
            'average_precision': average_precision_score(y_test, y_test_proba)
        }
        
        # 결과 업데이트
        self.results[model_key].update({
            'optimal_threshold': optimal_threshold,
            'threshold_analysis': threshold_analysis,
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics,
            'predictions': {
                'y_valid_proba': y_valid_proba.tolist(),
                'y_test_proba': y_test_proba.tolist()
            }
        })
        
        print(f"\n📊 {model_key} 최종 평가 (Threshold: {optimal_threshold:.3f}):")
        print(f"   검증 - AUC: {valid_metrics['auc']:.4f}, F1: {valid_metrics['f1']:.4f}, Precision: {valid_metrics['precision']:.4f}, Recall: {valid_metrics['recall']:.4f}")
        print(f"   테스트 - AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
    
    def run_all_models(self):
        """모든 모델 실행"""
        print("\n🚀 모든 모델 실행 시작")
        print("="*80)
        
        # 데이터 타입별로 실행
        for data_type in ['normal', 'smote']:
            print(f"\n📊 {data_type.upper()} 데이터 처리")
            print("="*60)
            
            # Lasso 특성 선택
            if self.config['lasso']['enabled']:
                self.apply_lasso_feature_selection(data_type)
            
            # 모델별 최적화
            models_to_run = []
            if self.config['models']['logistic']['enabled']:
                models_to_run.append(('logistic', self.optimize_logistic_regression))
            if self.config['models']['random_forest']['enabled']:
                models_to_run.append(('random_forest', self.optimize_random_forest))
            if self.config['models']['xgboost']['enabled']:
                models_to_run.append(('xgboost', self.optimize_xgboost))
            
            for model_name, optimize_func in models_to_run:
                optimize_func(data_type)
        
        # 모든 모델 평가 (최적 threshold 자동 탐색)
        print(f"\n📊 모든 모델 평가 및 Threshold 최적화")
        print("="*60)
        
        for model_key in self.models.keys():
            self.evaluate_model(model_key)
    
    def save_all_results(self):
        """모든 결과 저장"""
        print(f"\n💾 결과 저장")
        print("="*60)
        
        # 모델 저장
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for model_key, model in self.models.items():
            model_path = models_dir / f'{model_key.lower()}_model.joblib'
            joblib.dump(model, model_path)
            print(f"✅ {model_key} 모델 저장: {model_path}")
        
        # 결과 저장 (JSON 직렬화 가능하도록 변환)
        def convert_to_serializable(obj):
            """JSON 직렬화 가능한 형태로 변환"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            else:
                return obj
        
        serializable_results = convert_to_serializable(self.results)
        results_dir = self.output_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        results_path = results_dir / 'all_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"✅ 전체 결과 저장: {results_path}")
        
        # 설정 저장
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        print(f"✅ 설정 저장: {config_path}")
        
        # 요약 테이블 생성
        self.create_summary_table()
        
        # 시각화 생성
        self.create_visualizations()
    
    def create_summary_table(self):
        """요약 테이블 생성"""
        summary_data = []
        
        for model_key, result in self.results.items():
            model_name = model_key.split('_')[0]
            data_type = result['data_type']
            
            summary_data.append({
                'Model': model_name,
                'Data_Type': data_type.upper(),
                'Optimal_Threshold': result.get('optimal_threshold', 0.5),
                'CV_AUC': result['cv_score'],
                'Valid_AUC': result['valid_metrics']['auc'],
                'Valid_F1': result['valid_metrics']['f1'],
                'Test_AUC': result['test_metrics']['auc'],
                'Test_Precision': result['test_metrics']['precision'],
                'Test_Recall': result['test_metrics']['recall'],
                'Test_F1': result['test_metrics']['f1'],
                'Test_Balanced_Acc': result['test_metrics'].get('balanced_accuracy', 0),
                'Average_Precision': result['test_metrics'].get('average_precision', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 저장
        results_dir = self.output_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        summary_path = results_dir / 'summary_table.csv'
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        print(f"\n🏆 실행 결과 요약:")
        print(summary_df.round(4))
        print(f"✅ 요약 테이블 저장: {summary_path}")
        
        return summary_df
    
    def create_visualizations(self):
        """모든 시각화 생성"""
        print(f"\n📈 시각화 생성")
        print("="*60)
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 요약 테이블 생성 (시각화에서 사용)
        summary_df = self.create_summary_table()
        
        # 1. 성능 지표 비교 차트
        self.plot_performance_comparison(summary_df, viz_dir)
        
        # 2. ROC 곡선 비교
        self.plot_roc_curves(viz_dir)
        
        # 3. 특성 중요도 비교 (RF, XGBoost)
        self.plot_feature_importance_comparison(viz_dir)
        
        # 4. Normal vs SMOTE 비교
        self.plot_normal_vs_smote_comparison(summary_df, viz_dir)
        
        # 5. CV vs Test 성능 비교
        self.plot_cv_vs_test_comparison(summary_df, viz_dir)
        
        # 6. Threshold 최적화 결과 시각화
        self.plot_threshold_optimization(viz_dir)
        
        # 7. Precision-Recall 곡선
        self.plot_precision_recall_curves(viz_dir)
        
        print(f"✅ 모든 시각화 완료: {viz_dir}")
    
    def plot_performance_comparison(self, summary_df, viz_dir):
        """성능 지표 비교 차트"""
        print("📊 성능 지표 비교 차트 생성...")
        
        # 지표별 비교
        metrics = ['CV_AUC', 'Test_AUC', 'Test_Precision', 'Test_Recall', 'Test_F1']
        metric_names = ['CV AUC', 'Test AUC', 'Test Precision', 'Test Recall', 'Test F1']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('모델 성능 지표 비교', fontsize=16, fontweight='bold')
        
        # 색상 설정
        colors = {'NORMAL': 'skyblue', 'SMOTE': 'lightcoral'}
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            # 데이터 준비
            pivot_data = summary_df.pivot(index='Model', columns='Data_Type', values=metric)
            
            # 바 차트
            x = np.arange(len(pivot_data.index))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pivot_data['NORMAL'], width, 
                          label='Normal', alpha=0.8, color=colors['NORMAL'])
            bars2 = ax.bar(x + width/2, pivot_data['SMOTE'], width, 
                          label='SMOTE', alpha=0.8, color=colors['SMOTE'])
            
            # 값 표시
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.set_ylabel(name)
            ax.set_xlabel('모델')
            ax.set_xticks(x)
            ax.set_xticklabels(pivot_data.index, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1 if metric != 'Test_F1' else max(summary_df[metric].max() * 1.2, 0.5))
        
        # 마지막 subplot 제거
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 성능 비교 차트 저장: performance_comparison.png")
    
    def plot_roc_curves(self, viz_dir):
        """ROC 곡선 비교"""
        print("📈 ROC 곡선 생성...")
        
        from sklearn.metrics import roc_curve, auc
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('ROC 곡선 비교', fontsize=16, fontweight='bold')
        
        data_types = ['normal', 'smote']
        titles = ['Normal 데이터', 'SMOTE 데이터']
        colors = ['blue', 'red', 'green']
        
        for idx, (data_type, title) in enumerate(zip(data_types, titles)):
            ax = axes[idx]
            
            X_test = self.data[data_type]['X_test']
            y_test = self.data[data_type]['y_test']
            
            model_names = ['LogisticRegression', 'RandomForest', 'XGBoost']
            
            for i, model_name in enumerate(model_names):
                model_key = f'{model_name}_{data_type}'
                if model_key in self.models:
                    model = self.models[model_key]
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, color=colors[i], lw=2, 
                           label=f'{model_name} (AUC = {roc_auc:.4f})')
            
            # 대각선 (랜덤 분류기)
            ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
                   label='Random (AUC = 0.5000)')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(title)
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ ROC 곡선 저장: roc_curves_comparison.png")
    
    def plot_feature_importance_comparison(self, viz_dir):
        """특성 중요도 비교 (RF, XGBoost만)"""
        print("🔍 특성 중요도 비교 생성...")
        
        tree_models = ['RandomForest', 'XGBoost']
        data_types = ['normal', 'smote']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('특성 중요도 비교 (Tree-based 모델)', fontsize=16, fontweight='bold')
        
        for i, model_name in enumerate(tree_models):
            for j, data_type in enumerate(data_types):
                ax = axes[i, j]
                model_key = f'{model_name}_{data_type}'
                
                if model_key in self.results and 'feature_importances' in self.results[model_key]:
                    importances = self.results[model_key]['feature_importances']
                    
                    # 상위 10개 특성
                    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
                    features, values = zip(*sorted_features)
                    
                    # 색상 설정
                    color = 'green' if model_name == 'RandomForest' else 'purple'
                    
                    bars = ax.barh(features, values, color=color, alpha=0.7)
                    
                    # 값 표시
                    for k, v in enumerate(values):
                        ax.text(v + 0.001, k, f'{v:.3f}', va='center', fontsize=9)
                    
                    ax.set_title(f'{model_name} - {data_type.upper()}\n특성 중요도 (Top 10)', 
                               fontsize=12, fontweight='bold')
                    ax.set_xlabel('중요도')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{model_name} - {data_type.upper()}')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 특성 중요도 비교 저장: feature_importance_comparison.png")
    
    def plot_normal_vs_smote_comparison(self, summary_df, viz_dir):
        """Normal vs SMOTE 상세 비교"""
        print("⚖️ Normal vs SMOTE 비교 생성...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Normal vs SMOTE 데이터 성능 비교', fontsize=16, fontweight='bold')
        
        models = summary_df['Model'].unique()
        metrics = ['Test_AUC', 'Test_F1', 'Test_Precision', 'Test_Recall']
        metric_names = ['Test AUC', 'Test F1', 'Test Precision', 'Test Recall']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            normal_values = []
            smote_values = []
            
            for model in models:
                normal_val = summary_df[(summary_df['Model'] == model) & 
                                      (summary_df['Data_Type'] == 'NORMAL')][metric].iloc[0]
                smote_val = summary_df[(summary_df['Model'] == model) & 
                                     (summary_df['Data_Type'] == 'SMOTE')][metric].iloc[0]
                normal_values.append(normal_val)
                smote_values.append(smote_val)
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, normal_values, width, label='Normal', 
                          alpha=0.8, color='skyblue')
            bars2 = ax.bar(x + width/2, smote_values, width, label='SMOTE', 
                          alpha=0.8, color='lightcoral')
            
            # 개선도 표시
            for j, (normal, smote) in enumerate(zip(normal_values, smote_values)):
                improvement = ((smote - normal) / normal * 100) if normal > 0 else 0
                ax.text(j, max(normal, smote) + 0.02, f'{improvement:+.1f}%', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 값 표시
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'{name} 비교', fontsize=12, fontweight='bold')
            ax.set_ylabel(name)
            ax.set_xlabel('모델')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'normal_vs_smote_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Normal vs SMOTE 비교 저장: normal_vs_smote_detailed.png")
    
    def plot_cv_vs_test_comparison(self, summary_df, viz_dir):
        """CV vs Test 성능 비교 (과적합 확인)"""
        print("📊 CV vs Test 성능 비교 생성...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('CV vs Test AUC 비교 (과적합 확인)', fontsize=16, fontweight='bold')
        
        data_types = ['NORMAL', 'SMOTE']
        colors = ['skyblue', 'lightcoral']
        
        for idx, data_type in enumerate(data_types):
            ax = axes[idx]
            
            subset = summary_df[summary_df['Data_Type'] == data_type]
            models = subset['Model'].tolist()
            cv_scores = subset['CV_AUC'].tolist()
            test_scores = subset['Test_AUC'].tolist()
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, cv_scores, width, label='CV AUC', 
                          alpha=0.8, color='green')
            bars2 = ax.bar(x + width/2, test_scores, width, label='Test AUC', 
                          alpha=0.8, color=colors[idx])
            
            # 과적합 정도 표시
            for i, (cv, test) in enumerate(zip(cv_scores, test_scores)):
                overfitting = cv - test
                color = 'red' if overfitting > 0.05 else 'orange' if overfitting > 0.02 else 'green'
                ax.text(i, max(cv, test) + 0.01, f'{overfitting:.3f}', 
                       ha='center', va='bottom', fontsize=10, 
                       fontweight='bold', color=color)
            
            # 값 표시
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'{data_type} 데이터', fontsize=12, fontweight='bold')
            ax.set_ylabel('AUC Score')
            ax.set_xlabel('모델')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.8, 1.0)
            
            # 과적합 기준선
            ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, 
                      label='High Performance')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'cv_vs_test_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ CV vs Test 비교 저장: cv_vs_test_comparison.png")
    
    def plot_threshold_optimization(self, viz_dir):
        """Threshold 최적화 결과 시각화"""
        print("🎯 Threshold 최적화 결과 시각화...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('모델별 최적 Threshold 분석', fontsize=16, fontweight='bold')
        
        model_names = []
        thresholds = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        # 각 모델의 threshold 최적화 결과 수집
        for model_key, result in self.results.items():
            if 'threshold_analysis' in result:
                analysis = result['threshold_analysis']
                model_name = model_key.split('_')[0]
                data_type = result['data_type']
                label = f"{model_name}\n({data_type.upper()})"
                
                model_names.append(label)
                thresholds.append(analysis.get('final_threshold', 0.5))
                
                # 최적 threshold에서의 성능
                valid_metrics = result.get('valid_metrics', {})
                f1_scores.append(valid_metrics.get('f1', 0))
                precision_scores.append(valid_metrics.get('precision', 0))
                recall_scores.append(valid_metrics.get('recall', 0))
        
        if not model_names:
            print("  ⚠️ Threshold 최적화 데이터 없음")
            return
        
        # 1. 모델별 최적 Threshold
        ax1 = axes[0, 0]
        bars = ax1.bar(model_names, thresholds, color=['blue', 'red', 'green', 'orange', 'purple', 'brown'][:len(model_names)])
        ax1.set_title('모델별 최적 Threshold', fontweight='bold')
        ax1.set_ylabel('Threshold')
        ax1.set_ylim(0, 1)
        
        # 값 표시
        for bar, thresh in zip(bars, thresholds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{thresh:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Threshold vs F1 Score
        ax2 = axes[0, 1]
        scatter = ax2.scatter(thresholds, f1_scores, c=range(len(model_names)), 
                             s=100, alpha=0.7, cmap='viridis')
        ax2.set_title('Threshold vs F1 Score', fontweight='bold')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('F1 Score')
        ax2.grid(True, alpha=0.3)
        
        # 모델명 표시
        for i, (thresh, f1, name) in enumerate(zip(thresholds, f1_scores, model_names)):
            ax2.annotate(name.split('\n')[0], (thresh, f1), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 3. Precision vs Recall (최적 threshold에서)
        ax3 = axes[1, 0]
        scatter = ax3.scatter(recall_scores, precision_scores, c=range(len(model_names)),
                             s=100, alpha=0.7, cmap='viridis')
        ax3.set_title('Precision vs Recall (최적 Threshold)', fontweight='bold')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.grid(True, alpha=0.3)
        
        # 모델명 표시
        for i, (recall, precision, name) in enumerate(zip(recall_scores, precision_scores, model_names)):
            ax3.annotate(name.split('\n')[0], (recall, precision),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 4. 종합 성능 비교 (Radar Chart)
        ax4 = axes[1, 1]
        
        # 간단한 바 차트로 대체 (radar chart는 복잡함)
        x = np.arange(len(model_names))
        width = 0.25
        
        bars1 = ax4.bar(x - width, f1_scores, width, label='F1', alpha=0.8)
        bars2 = ax4.bar(x, precision_scores, width, label='Precision', alpha=0.8)
        bars3 = ax4.bar(x + width, recall_scores, width, label='Recall', alpha=0.8)
        
        ax4.set_title('최적 Threshold에서의 성능', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels([name.split('\n')[0] for name in model_names], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'threshold_optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Threshold 최적화 분석 저장: threshold_optimization_analysis.png")
    
    def plot_precision_recall_curves(self, viz_dir):
        """Precision-Recall 곡선 시각화"""
        print("📈 Precision-Recall 곡선 생성...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Precision-Recall 곡선 비교', fontsize=16, fontweight='bold')
        
        data_types = ['normal', 'smote']
        titles = ['Normal 데이터', 'SMOTE 데이터']
        colors = ['blue', 'red', 'green']
        
        for idx, (data_type, title) in enumerate(zip(data_types, titles)):
            ax = axes[idx]
            
            model_names = ['LogisticRegression', 'RandomForest', 'XGBoost']
            
            for i, model_name in enumerate(model_names):
                model_key = f'{model_name}_{data_type}'
                if model_key in self.results and 'threshold_analysis' in self.results[model_key]:
                    analysis = self.results[model_key]['threshold_analysis']
                    
                    if 'pr_curve' in analysis:
                        pr_data = analysis['pr_curve']
                        precision_vals = pr_data['precision']
                        recall_vals = pr_data['recall']
                        
                        # Average Precision 계산
                        valid_metrics = self.results[model_key].get('valid_metrics', {})
                        avg_precision = valid_metrics.get('average_precision', 0)
                        
                        ax.plot(recall_vals, precision_vals, color=colors[i], lw=2,
                               label=f'{model_name} (AP = {avg_precision:.3f})')
                        
                        # 최적 threshold 포인트 표시
                        optimal_threshold = analysis.get('final_threshold', 0.5)
                        
                        # 최적 threshold에서의 precision, recall 찾기
                        opt_precision = valid_metrics.get('precision', 0)
                        opt_recall = valid_metrics.get('recall', 0)
                        
                        ax.scatter([opt_recall], [opt_precision], color=colors[i], 
                                 s=100, marker='*', edgecolors='black', linewidth=1,
                                 label=f'{model_name} 최적점 (T={optimal_threshold:.3f})')
            
            # 기준선 (Random Classifier)
            y_true_ratio = self.data[data_type]['y_valid'].mean()
            ax.axhline(y=y_true_ratio, color='gray', linestyle='--', alpha=0.7,
                      label=f'Random (AP = {y_true_ratio:.3f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(title)
            ax.legend(loc="lower left", fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Precision-Recall 곡선 저장: precision_recall_curves.png")

def load_config(config_path='master_config.json'):
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """메인 실행 함수"""
    print("🏢 한국 기업 부실예측 - 마스터 모델 러너")
    print("="*80)
    
    # 설정 로드
    config = load_config()
    
    # 러너 생성 및 실행
    runner = MasterModelRunner(config)
    runner.load_data()
    runner.run_all_models()
    runner.save_all_results()
    
    print(f"\n🎉 모든 모델 실행 완료!")
    print(f"📁 결과 저장 위치: {runner.output_dir}")
    print("="*80)

if __name__ == "__main__":
    main() 