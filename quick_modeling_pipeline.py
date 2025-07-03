"""
빠른 모델링 파이프라인
====================

성능을 위해 최적화된 간단한 모델링 파이프라인
- 필수 기능만 포함
- 빠른 실행 시간
- 3개 모델 (LogisticRegression, RandomForest, XGBoost)
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import joblib
warnings.filterwarnings('ignore')

# 모델링 관련 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score, balanced_accuracy_score
)
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler


class QuickModelingPipeline:
    """빠른 모델링 파이프라인 클래스"""
    
    def __init__(self, data_path: str = "data/final"):
        """파이프라인 초기화"""
        self.data_path = Path(data_path)
        self.project_root = Path(__file__).parent
        
        # 실행 정보 설정
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"quick_modeling_{self.timestamp}"
        
        # 출력 디렉토리 설정
        self.output_dir = self.project_root / "outputs" / "quick_modeling" / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로거 설정
        self.logger = self._setup_logging()
        
        # 결과 저장용
        self.results = {}
        self.models = {}
        
    def _setup_logging(self):
        """로깅 설정"""
        logger = logging.getLogger(f"QuickModeling_{self.timestamp}")
        logger.setLevel(logging.INFO)
        
        # 핸들러가 이미 있으면 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(self.output_dir / "quick_modeling.log")
        file_handler.setLevel(logging.INFO)
        
        # 포매터
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_data(self):
        """데이터 로드"""
        self.logger.info("데이터 로드 시작")
        
        try:
            # 데이터 파일 로드
            self.X_train = pd.read_csv(self.data_path / "X_train.csv", index_col=0)
            self.X_val = pd.read_csv(self.data_path / "X_val.csv", index_col=0)
            self.X_test = pd.read_csv(self.data_path / "X_test.csv", index_col=0)
            self.y_train = pd.read_csv(self.data_path / "y_train.csv", index_col=0).squeeze()
            self.y_val = pd.read_csv(self.data_path / "y_val.csv", index_col=0).squeeze()
            self.y_test = pd.read_csv(self.data_path / "y_test.csv", index_col=0).squeeze()
            
            self.logger.info(f"데이터 로드 완료:")
            self.logger.info(f"  Train: {self.X_train.shape}, {self.y_train.value_counts().to_dict()}")
            self.logger.info(f"  Val: {self.X_val.shape}, {self.y_val.value_counts().to_dict()}")
            self.logger.info(f"  Test: {self.X_test.shape}, {self.y_test.value_counts().to_dict()}")
            
        except Exception as e:
            self.logger.error(f"데이터 로드 실패: {e}")
            raise
    
    def optimize_model(self, model_name: str, n_trials: int = 15):
        """모델 하이퍼파라미터 최적화"""
        self.logger.info(f"{model_name} 최적화 시작 (trials: {n_trials})")
        
        def objective(trial):
            if model_name == 'logistic_regression':
                model = LogisticRegression(
                    C=trial.suggest_float('C', 0.01, 10.0, log=True),
                    penalty=trial.suggest_categorical('penalty', ['l2']),
                    solver=trial.suggest_categorical('solver', ['lbfgs', 'saga']),
                    class_weight='balanced',
                    random_state=42,
                    max_iter=1000
                )
            elif model_name == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    min_samples_split=trial.suggest_int('min_samples_split', 5, 15),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 2, 8),
                    max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
            elif model_name == 'xgboost':
                # scale_pos_weight 계산
                neg_count = (self.y_train == 0).sum()
                pos_count = (self.y_train == 1).sum()
                scale_pos_weight = neg_count / pos_count
                
                model = xgb.XGBClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 3, 8),
                    learning_rate=trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
                    subsample=trial.suggest_float('subsample', 0.7, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    reg_alpha=trial.suggest_float('reg_alpha', 0.0, 0.5),
                    reg_lambda=trial.suggest_float('reg_lambda', 0.5, 2.0),
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            
            # 3-fold CV로 빠른 평가
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1', n_jobs=-1)
            return scores.mean()
        
        # Optuna 최적화
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # 최적 모델 생성
        best_params = study.best_params
        best_score = study.best_value
        
        if model_name == 'logistic_regression':
            model = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                **best_params
            )
        elif model_name == 'random_forest':
            model = RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                **best_params
            )
        elif model_name == 'xgboost':
            neg_count = (self.y_train == 0).sum()
            pos_count = (self.y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count
            
            model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                **best_params
            )
        
        # 모델 훈련
        model.fit(self.X_train, self.y_train)
        
        self.logger.info(f"{model_name} 최적화 완료 - CV Score: {best_score:.4f}")
        return model, best_params, best_score
    
    def find_best_threshold(self, model, X_val, y_val):
        """최적 임계값 찾기"""
        y_prob = model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 41)
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1
    
    def evaluate_model(self, model, model_name: str, best_threshold: float):
        """모델 평가"""
        # 예측
        y_val_prob = model.predict_proba(self.X_val)[:, 1]
        y_test_prob = model.predict_proba(self.X_test)[:, 1]
        
        y_val_pred = (y_val_prob >= best_threshold).astype(int)
        y_test_pred = (y_test_prob >= best_threshold).astype(int)
        
        # 검증 세트 성능
        val_metrics = {
            'roc_auc': roc_auc_score(self.y_val, y_val_prob),
            'f1': f1_score(self.y_val, y_val_pred),
            'precision': precision_score(self.y_val, y_val_pred),
            'recall': recall_score(self.y_val, y_val_pred),
            'balanced_accuracy': balanced_accuracy_score(self.y_val, y_val_pred),
            'average_precision': average_precision_score(self.y_val, y_val_prob)
        }
        
        # 테스트 세트 성능
        test_metrics = {
            'roc_auc': roc_auc_score(self.y_test, y_test_prob),
            'f1': f1_score(self.y_test, y_test_pred),
            'precision': precision_score(self.y_test, y_test_pred),
            'recall': recall_score(self.y_test, y_test_pred),
            'balanced_accuracy': balanced_accuracy_score(self.y_test, y_test_pred),
            'average_precision': average_precision_score(self.y_test, y_test_prob)
        }
        
        return val_metrics, test_metrics
    
    def run_pipeline(self):
        """전체 파이프라인 실행"""
        start_time = datetime.now()
        self.logger.info("빠른 모델링 파이프라인 시작")
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 모델 훈련 및 평가
        model_names = ['logistic_regression', 'random_forest', 'xgboost']
        
        for model_name in model_names:
            try:
                # 모델 최적화
                model, best_params, cv_score = self.optimize_model(model_name, n_trials=15)
                
                # 최적 임계값 찾기
                best_threshold, val_f1 = self.find_best_threshold(model, self.X_val, self.y_val)
                
                # 모델 평가
                val_metrics, test_metrics = self.evaluate_model(model, model_name, best_threshold)
                
                # 결과 저장
                self.results[model_name] = {
                    'model': model,
                    'best_params': best_params,
                    'cv_score': cv_score,
                    'best_threshold': best_threshold,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics
                }
                
                self.logger.info(f"{model_name} 완료:")
                self.logger.info(f"  CV Score: {cv_score:.4f}")
                self.logger.info(f"  Best Threshold: {best_threshold:.3f}")
                self.logger.info(f"  Val F1: {val_metrics['f1']:.4f}")
                self.logger.info(f"  Test F1: {test_metrics['f1']:.4f}")
                self.logger.info(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
                
            except Exception as e:
                self.logger.error(f"{model_name} 실행 중 오류: {e}")
                continue
        
        # 3. 결과 저장
        self.save_results()
        
        # 실행 시간 출력
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.logger.info(f"파이프라인 완료 - 실행 시간: {duration:.1f}초")
        
        return self.output_dir
    
    def save_results(self):
        """결과 저장"""
        self.logger.info("결과 저장 시작")
        
        # 모델 저장
        for model_name, result in self.results.items():
            model_path = self.output_dir / f"{model_name}_model.pkl"
            joblib.dump(result['model'], model_path)
        
        # 결과 요약 저장
        summary = {}
        for model_name, result in self.results.items():
            summary[model_name] = {
                'best_params': result['best_params'],
                'cv_score': result['cv_score'],
                'best_threshold': result['best_threshold'],
                'val_metrics': result['val_metrics'],
                'test_metrics': result['test_metrics']
            }
        
        # JSON 저장
        import json
        with open(self.output_dir / "results_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # CSV 저장
        df_results = []
        for model_name, result in self.results.items():
            row = {
                'model': model_name,
                'cv_score': result['cv_score'],
                'best_threshold': result['best_threshold'],
                **{f'val_{k}': v for k, v in result['val_metrics'].items()},
                **{f'test_{k}': v for k, v in result['test_metrics'].items()}
            }
            df_results.append(row)
        
        df_results = pd.DataFrame(df_results)
        df_results.to_csv(self.output_dir / "results_summary.csv", index=False)
        
        self.logger.info(f"결과 저장 완료: {self.output_dir}")


# 사용 예시
if __name__ == "__main__":
    pipeline = QuickModelingPipeline(data_path="data/final")
    output_dir = pipeline.run_pipeline()
    print(f"결과 저장 위치: {output_dir}")