"""
모델링 파이프라인
===============================

기능:
1. 전처리된 데이터 로드
2. 다양한 샘플링 전략 적용 (Normal, SMOTE, Undersampling, Combined)
3. 모델 최적화 (LogisticRegression, RandomForest, XGBoost)
4. Threshold 최적화
5. 앙상블 모델 생성
6. 성능 평가 및 시각화

Config를 통한 커스터마이징 지원
"""

import pandas as pd
import numpy as np
import yaml
import json
import pickle
import logging
import os
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic' if plt.matplotlib.get_backend() != 'Agg' else 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 모델링 관련 라이브러리
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

# 샘플링 라이브러리
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks


class ModelingPipeline:
    """
    모델링 파이프라인 클래스
    
    Config 파일을 통해 모든 설정을 관리하며,
    데이터 로드부터 모델 훈련, 평가까지 전체 과정을 수행
    """
    
    def __init__(self, config_path: str):
        """
        파이프라인 초기화
        
        Args:
            config_path: config YAML/JSON 파일 경로
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 프로젝트 루트 디렉토리 설정
        self.project_root = Path(__file__).parent.parent.parent
        
        # 실행 정보 설정
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.config['experiment']['name']}_{self.timestamp}"
        
        # 출력 디렉토리 설정 (로거 설정 전에 해야 함)
        self.output_dir = self.project_root / self.config['output']['base_dir'] / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 하위 디렉토리 생성
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # 로거 설정 (output_dir 설정 후에 해야 함)
        self.logger = self._setup_logging()
        
        # 결과 저장용 딕셔너리
        self.results = {
            'experiment_info': {},
            'data_info': {},
            'modeling_steps': {},
            'model_performance': {},
            'ensemble_results': {}
        }
        
        # 데이터 및 모델 저장용
        self.data = {}
        self.models = {}
        self.model_results = {}
        
        self.logger.info("모델링 파이프라인이 초기화되었습니다.")
        self.logger.info(f"실행 이름: {self.run_name}")
        self.logger.info(f"출력 경로: {self.output_dir}")
    
    def _load_config(self) -> Dict:
        """Config 파일 로드"""
        try:
            # YAML 우선 시도
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError:
            # YAML 실패 시 JSON 시도
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('ModelingPipeline')
        logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        # 핸들러 제거 (중복 방지)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러
        if self.config['logging']['save_to_file']:
            log_dir = self.output_dir / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"modeling_{self.timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """전처리된 데이터 로드"""
        self.logger.info("데이터 로드 시작")
        
        data_path = self.project_root / self.config['data']['input_path']
        
        # 데이터 로드
        self.data['normal'] = {
            'X_train': pd.read_csv(data_path / self.config['data']['files']['X_train']),
            'X_val': pd.read_csv(data_path / self.config['data']['files']['X_val']),
            'X_test': pd.read_csv(data_path / self.config['data']['files']['X_test']),
            'y_train': pd.read_csv(data_path / self.config['data']['files']['y_train']).iloc[:, 0],
            'y_val': pd.read_csv(data_path / self.config['data']['files']['y_val']).iloc[:, 0],
            'y_test': pd.read_csv(data_path / self.config['data']['files']['y_test']).iloc[:, 0]
        }
        
        # 활성화된 데이터 타입별로 복사 (동적 샘플링을 위해)
        enabled_data_types = [dt for dt, config in self.config['sampling']['data_types'].items() if config['enabled']]
        for data_type in enabled_data_types:
            if data_type != 'normal':
                self.data[data_type] = {k: v.copy() for k, v in self.data['normal'].items()}
        
        # 데이터 정보 출력
        data = self.data['normal']
        self.logger.info(f"Train: {data['X_train'].shape}, 부실비율: {data['y_train'].mean():.2%}")
        self.logger.info(f"Validation: {data['X_val'].shape}, 부실비율: {data['y_val'].mean():.2%}")
        self.logger.info(f"Test: {data['X_test'].shape}, 부실비율: {data['y_test'].mean():.2%}")
        
        # 데이터 정보 저장
        self.results['data_info'] = {
            'shapes': {
                'train': data['X_train'].shape,
                'val': data['X_val'].shape,
                'test': data['X_test'].shape
            },
            'target_distribution': {
                'train': data['y_train'].value_counts().to_dict(),
                'val': data['y_val'].value_counts().to_dict(),
                'test': data['y_test'].value_counts().to_dict()
            },
            'feature_count': data['X_train'].shape[1],
            'feature_names': list(data['X_train'].columns)
        }
        
        return self.data
    
    def apply_sampling_strategy(self, X: pd.DataFrame, y: pd.Series, data_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        데이터 타입에 따른 샘플링 전략 적용
        
        Args:
            X: 특성 데이터
            y: 타겟 데이터
            data_type: 데이터 타입 ('normal', 'smote', 'undersampling', 'combined')
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        if data_type == 'normal':
            return X, y
        
        config = self.config['sampling']['data_types'][data_type]
        
        if data_type == 'smote':
            smote = BorderlineSMOTE(
                sampling_strategy=config['sampling_strategy'],
                random_state=config.get('random_state', self.config['random_state']),
                k_neighbors=config.get('k_neighbors', 5),
                m_neighbors=config.get('m_neighbors', 10),
                kind=config.get('kind', 'borderline-1')
            )
            return smote.fit_resample(X, y)
        
        elif data_type == 'undersampling':
            method = config.get('method', 'random')
            
            if method == 'random':
                undersampler = RandomUnderSampler(
                    sampling_strategy=config['sampling_strategy'],
                    random_state=config.get('random_state', self.config['random_state'])
                )
            elif method == 'edited_nearest_neighbours':
                from imblearn.under_sampling import EditedNearestNeighbours
                undersampler = EditedNearestNeighbours(
                    sampling_strategy='auto',
                    n_neighbors=3
                )
            elif method == 'tomek':
                from imblearn.under_sampling import TomekLinks
                undersampler = TomekLinks(sampling_strategy='auto')
            else:
                self.logger.warning(f"지원하지 않는 언더샘플링 방법: {method}, RandomUnderSampler 사용")
                undersampler = RandomUnderSampler(
                    sampling_strategy=config['sampling_strategy'],
                    random_state=config.get('random_state', self.config['random_state'])
                )
            
            return undersampler.fit_resample(X, y)
        
        elif data_type == 'combined':
            # 1단계: SMOTE 적용
            smote_config = config['smote']
            smote = BorderlineSMOTE(
                sampling_strategy=smote_config['sampling_strategy'],
                random_state=smote_config.get('random_state', self.config['random_state']),
                k_neighbors=smote_config.get('k_neighbors', 5),
                m_neighbors=smote_config.get('m_neighbors', 10),
                kind=smote_config.get('kind', 'borderline-1')
            )
            X_smote, y_smote = smote.fit_resample(X, y)
            
            # 2단계: 언더샘플링 적용
            undersampling_config = config['undersampling']
            
            # 언더샘플링 방법에 따른 선택
            method = undersampling_config.get('method', 'random')
            if method == 'random':
                undersampler = RandomUnderSampler(
                    sampling_strategy=undersampling_config['sampling_strategy'],
                    random_state=undersampling_config.get('random_state', self.config['random_state'])
                )
            elif method == 'edited_nearest_neighbours':
                from imblearn.under_sampling import EditedNearestNeighbours
                undersampler = EditedNearestNeighbours(
                    sampling_strategy='auto',
                    n_neighbors=3
                )
            elif method == 'tomek':
                from imblearn.under_sampling import TomekLinks
                undersampler = TomekLinks(sampling_strategy='auto')
            else:
                self.logger.warning(f"지원하지 않는 언더샘플링 방법: {method}, RandomUnderSampler 사용")
                undersampler = RandomUnderSampler(
                    sampling_strategy=undersampling_config['sampling_strategy'],
                    random_state=undersampling_config.get('random_state', self.config['random_state'])
                )
            
            X_combined, y_combined = undersampler.fit_resample(X_smote, y_smote)
            
            return X_combined, y_combined
        
        else:
            self.logger.warning(f"알 수 없는 데이터 타입: {data_type}, 원본 데이터 반환")
            return X, y
    
    def apply_scaling(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        컬럼별 스케일링 적용 (존재하지 않는 컬럼은 자동으로 건너뛰기)
        
        Args:
            X_train: 훈련 데이터
            X_val: 검증 데이터
            X_test: 테스트 데이터
            data_type: 데이터 타입
            
        Returns:
            tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scalers)
        """
        if not self.config.get('scaling', {}).get('enabled', False):
            self.logger.info(f"스케일링이 비활성화되어 있습니다 ({data_type.upper()})")
            return X_train, X_val, X_test, {}
        
        self.logger.info(f"스케일링 적용 시작 ({data_type.upper()})")
        
        scaling_config = self.config['scaling']
        scalers = {}
        total_available_columns = set(X_train.columns)
        total_processed_columns = set()
        
        # 각 스케일링 방법별로 컬럼 처리
        for scaler_type, columns in scaling_config.get('column_groups', {}).items():
            if not columns:
                self.logger.info(f"{scaler_type} 그룹에 설정된 컬럼이 없습니다")
                continue
            
            # 실제 존재하는 컬럼과 존재하지 않는 컬럼 구분
            existing_columns = [col for col in columns if col in total_available_columns]
            missing_columns = [col for col in columns if col not in total_available_columns]
            
            # 존재하지 않는 컬럼 정보 출력 (경고가 아닌 정보 레벨)
            if missing_columns:
                self.logger.info(f"{scaler_type} 그룹 - 데이터에 없는 컬럼 ({len(missing_columns)}개): {missing_columns[:5]}{'...' if len(missing_columns) > 5 else ''}")
            
            # 존재하는 컬럼이 없으면 건너뛰기
            if not existing_columns:
                self.logger.info(f"{scaler_type} 그룹 - 적용 가능한 컬럼이 없어 건너뜁니다")
                continue
            
            # 중복 처리 방지 체크
            overlapping_columns = set(existing_columns) & total_processed_columns
            if overlapping_columns:
                self.logger.warning(f"{scaler_type} 그룹 - 이미 다른 스케일러로 처리된 컬럼들 제외: {list(overlapping_columns)}")
                existing_columns = [col for col in existing_columns if col not in overlapping_columns]
            
            if not existing_columns:
                self.logger.info(f"{scaler_type} 그룹 - 중복 제거 후 적용 가능한 컬럼이 없습니다")
                continue
            
            try:
                # 스케일러 생성
                if scaler_type.lower() == 'standard':
                    scaler = StandardScaler()
                elif scaler_type.lower() == 'robust':
                    scaler = RobustScaler()
                elif scaler_type.lower() == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    self.logger.warning(f"지원하지 않는 스케일링 방법: {scaler_type}, Standard 스케일링을 사용합니다")
                    scaler = StandardScaler()
                
                # 훈련 데이터로 스케일러 피팅
                scaler.fit(X_train[existing_columns])
                
                # 스케일링 적용
                X_train.loc[:, existing_columns] = scaler.transform(X_train[existing_columns])
                X_val.loc[:, existing_columns] = scaler.transform(X_val[existing_columns])
                X_test.loc[:, existing_columns] = scaler.transform(X_test[existing_columns])
                
                # 스케일러 정보 저장
                scalers[scaler_type] = {
                    'scaler': scaler,
                    'columns': existing_columns,
                    'missing_columns': missing_columns
                }
                
                # 처리된 컬럼 추적
                total_processed_columns.update(existing_columns)
                
                self.logger.info(f"{scaler_type} 스케일링 적용 완료: {len(existing_columns)}개 컬럼 처리")
                
            except Exception as e:
                self.logger.error(f"{scaler_type} 스케일링 적용 중 오류 발생: {e}")
                continue
        
        # 전체 스케일링 결과 요약
        total_scaled_columns = len(total_processed_columns)
        unscaled_columns = total_available_columns - total_processed_columns
        
        self.logger.info(f"스케일링 완료 요약 ({data_type.upper()}):")
        self.logger.info(f"  - 총 컬럼 수: {len(total_available_columns)}")
        self.logger.info(f"  - 스케일링 적용: {total_scaled_columns}개 컬럼")
        self.logger.info(f"  - 스케일링 미적용: {len(unscaled_columns)}개 컬럼")
        
        if unscaled_columns and len(unscaled_columns) <= 10:
            self.logger.info(f"  - 미적용 컬럼: {list(unscaled_columns)}")
        elif unscaled_columns:
            self.logger.info(f"  - 미적용 컬럼 (일부): {list(unscaled_columns)[:10]}...")
        
        # 스케일링 정보 저장
        scaling_info = {
            'enabled': True,
            'scalers_used': list(scalers.keys()),
            'total_columns': len(total_available_columns),
            'total_scaled_columns': total_scaled_columns,
            'total_unscaled_columns': len(unscaled_columns),
            'scaling_details': {
                scaler_type: {
                    'method': scaler_type,
                    'applied_columns': info['columns'],
                    'applied_count': len(info['columns']),
                    'missing_columns': info['missing_columns'],
                    'missing_count': len(info['missing_columns'])
                }
                for scaler_type, info in scalers.items()
            }
        }
        
        return X_train, X_val, X_test, scalers
    
    def apply_lasso_feature_selection(self, data_type: str = 'normal'):
        """Lasso 특성 선택 적용"""
        if not self.config['feature_selection']['enabled']:
            return
        
        self.logger.info(f"Lasso 특성 선택 적용 ({data_type.upper()})")
        
        X_train = self.data[data_type]['X_train']
        y_train = self.data[data_type]['y_train']
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Lasso CV
        lasso_config = self.config['feature_selection']['lasso']
        lasso_cv = LassoCV(
            alphas=lasso_config['alphas'],
            cv=lasso_config['cv_folds'],
            random_state=self.config['random_state'],
            n_jobs=self.config.get('performance', {}).get('n_jobs', 1)
        )
        lasso_cv.fit(X_train_scaled, y_train)
        
        # 특성 선택
        threshold = lasso_config['threshold']
        if threshold == 'median':
            threshold_value = np.median(np.abs(lasso_cv.coef_[lasso_cv.coef_ != 0]))
        elif threshold == 'mean':
            threshold_value = np.mean(np.abs(lasso_cv.coef_[lasso_cv.coef_ != 0]))
        else:
            threshold_value = float(threshold)
        
        selected_mask = np.abs(lasso_cv.coef_) >= threshold_value
        selected_features = X_train.columns[selected_mask].tolist()
        
        self.logger.info(f"최적 alpha: {lasso_cv.alpha_:.6f}")
        self.logger.info(f"선택된 특성: {len(selected_features)}/{len(X_train.columns)}")
        
        # 모든 데이터 타입에 동일한 특성 적용
        enabled_data_types = [dt for dt, config in self.config['sampling']['data_types'].items() if config['enabled']]
        for dt in enabled_data_types:
            for split in ['X_train', 'X_val', 'X_test']:
                self.data[dt][split] = self.data[dt][split][selected_features]
        
        # 결과 저장
        lasso_results = {
            'optimal_alpha': float(lasso_cv.alpha_),
            'threshold_value': float(threshold_value),
            'original_features': len(X_train.columns),
            'selected_features': len(selected_features),
            'selected_feature_names': selected_features,
            'feature_coefficients': dict(zip(X_train.columns, lasso_cv.coef_))
        }
        
        self.results['modeling_steps']['feature_selection'] = lasso_results
        
        # 파일로 저장
        with open(self.output_dir / 'results' / 'lasso_selection.json', 'w', encoding='utf-8') as f:
            json.dump(lasso_results, f, indent=2, ensure_ascii=False)
    
    def optimize_model(self, model_name: str, data_type: str):
        """모델 최적화"""
        self.logger.info(f"{model_name} 최적화 시작 ({data_type.upper()})")
        
        X_train = self.data[data_type]['X_train']
        y_train = self.data[data_type]['y_train']
        
        if model_name == 'logistic_regression':
            return self._optimize_logistic_regression(X_train, y_train, data_type)
        elif model_name == 'random_forest':
            return self._optimize_random_forest(X_train, y_train, data_type)
        elif model_name == 'xgboost':
            return self._optimize_xgboost(X_train, y_train, data_type)
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
    
    def _optimize_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series, data_type: str):
        """로지스틱 회귀 최적화"""
        config = self.config['models']['logistic_regression']
        
        def objective(trial):
            # penalty와 solver 조합 선택 (수렴성이 좋은 조합 우선)
            penalty_solver_combinations = []
            
            for penalty in config['penalty']:
                if penalty == 'l1':
                    for solver in ['liblinear', 'saga']:
                        penalty_solver_combinations.append(f"{penalty}_{solver}")
                elif penalty == 'l2':
                    # 수렴성이 좋은 solver 순서로 정렬
                    preferred_solvers = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
                    for solver in preferred_solvers:
                        if solver in config['l2_solvers']:
                            penalty_solver_combinations.append(f"{penalty}_{solver}")
                elif penalty == 'elasticnet':
                    penalty_solver_combinations.append(f"{penalty}_saga")
            
            combination = trial.suggest_categorical('penalty_solver', penalty_solver_combinations)
            penalty, solver = combination.split('_', 1)
            
            C = trial.suggest_float('C', *config['C_range'], log=True)
            
            # solver별로 적절한 max_iter 설정
            if solver in ['newton-cg', 'lbfgs']:
                max_iter = trial.suggest_int('max_iter', 500, 2000)  # 더 많은 반복
            elif solver == 'liblinear':
                max_iter = trial.suggest_int('max_iter', 300, 1500)
            else:  # saga
                max_iter = trial.suggest_int('max_iter', 1000, 3000)  # SAGA는 더 많이 필요
            
            params = {
                'penalty': penalty,
                'C': C,
                'solver': solver,
                'max_iter': max_iter,
                'random_state': self.config['random_state'],
                'tol': 1e-4  # 수렴 기준 완화
            }
            
            if penalty == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', *config['l1_ratio_range'])
            
            model = LogisticRegression(**params)
            
            try:
                # Cross validation (이미 전처리된 데이터 사용)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['random_state'])
                
                # 모든 경고를 캐치하여 처리 (LineSearchWarning, ConvergenceWarning 포함)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=1)  # n_jobs=1로 안정성 증대
                    
                    # 수렴 관련 경고가 있으면 낮은 점수 반환
                    for warning in w:
                        if 'converge' in str(warning.message).lower() or 'line search' in str(warning.message).lower():
                            return 0.5
                
                return scores.mean()
                
            except (Warning, Exception):
                # 모든 예외에 대해 낮은 점수 반환
                return 0.5
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config['random_state']))
        
        # optuna 기본 로그만 억제 (trial finished 메시지 등)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # 콜백 함수로 진행상황 표시
        def progress_callback(study, trial):
            if trial.number % 5 == 0 or trial.number == config['n_trials'] - 1:
                best_value = study.best_value if study.best_value else 0
                self.logger.info(f"Trial {trial.number + 1}/{config['n_trials']} - Current: {trial.value:.4f}, Best: {best_value:.4f}")
        
        study.optimize(objective, n_trials=config['n_trials'], callbacks=[progress_callback])
        
        # 최적 모델 훈련
        best_params = study.best_params.copy()
        penalty_solver = best_params.pop('penalty_solver')
        penalty, solver = penalty_solver.split('_', 1)
        best_params['penalty'] = penalty
        best_params['solver'] = solver
        best_params['tol'] = 1e-4
        
        model = LogisticRegression(**best_params)
        
        # 모델 훈련 (수렴하지 않을 경우 max_iter 증가 및 파라미터 조정)
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    model.fit(X_train, y_train)
                    
                    # 경고가 있는지 확인
                    has_convergence_warning = any(
                        'converge' in str(warning.message).lower() or 'line search' in str(warning.message).lower()
                        for warning in w
                    )
                    
                    if not has_convergence_warning:
                        break  # 경고가 없으면 성공
                    else:
                        raise Warning("Convergence warning detected")
                        
            except (Warning, Exception) as e:
                retry_count += 1
                if retry_count >= max_retries:
                    self.logger.warning(f"최대 재시도 횟수 도달. 현재 모델을 사용합니다. (최종 에러: {e})")
                    break
                
                self.logger.warning(f"수렴 실패 (재시도 {retry_count}/{max_retries}). 파라미터를 조정합니다.")
                
                # 파라미터 조정 전략
                if retry_count == 1:
                    # 1차 시도: max_iter 증가 및 tolerance 완화
                    best_params['max_iter'] = min(best_params.get('max_iter', 1000) * 2, 10000)
                    best_params['tol'] = 1e-3
                elif retry_count == 2:
                    # 2차 시도: 더 안정적인 solver로 변경
                    if best_params['solver'] in ['saga', 'liblinear']:
                        best_params['solver'] = 'lbfgs'
                        if best_params['penalty'] == 'l1':
                            best_params['penalty'] = 'l2'  # lbfgs는 l1을 지원하지 않음
                    best_params['max_iter'] = 5000
                    best_params['tol'] = 1e-2
                elif retry_count == 3:
                    # 3차 시도: 가장 안정적인 설정
                    best_params.update({
                        'solver': 'newton-cg',
                        'penalty': 'l2',
                        'max_iter': 10000,
                        'tol': 1e-2
                    })
                
                model = LogisticRegression(**best_params)
        
        self.logger.info(f"최적 AUC: {study.best_value:.4f}")
        self.logger.info(f"최적 파라미터: {best_params}")
        
        return model, best_params, study.best_value
    
    def _optimize_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, data_type: str):
        """랜덤 포레스트 최적화"""
        config = self.config['models']['random_forest']
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *config['n_estimators_range']),
                'max_depth': trial.suggest_int('max_depth', *config['max_depth_range']),
                'min_samples_split': trial.suggest_int('min_samples_split', *config['min_samples_split_range']),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', *config['min_samples_leaf_range']),
                'max_features': trial.suggest_float('max_features', *config['max_features_range']),
                'random_state': self.config['random_state'],
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            
            # Cross validation (이미 전처리된 데이터 사용)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['random_state'])
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config['random_state']))
        
        # 진행상황 콜백 함수
        def progress_callback(study, trial):
            if trial.number % 5 == 0 or trial.number == config['n_trials'] - 1:
                best_value = study.best_value if study.best_value else 0
                self.logger.info(f"Trial {trial.number + 1}/{config['n_trials']} - Current: {trial.value:.4f}, Best: {best_value:.4f}")
        
        study.optimize(objective, n_trials=config['n_trials'], callbacks=[progress_callback])
        
        # 최적 모델 훈련
        best_params = study.best_params
        model = RandomForestClassifier(**best_params)
        
        # 모델 훈련 (이미 전처리된 데이터 사용)
        model.fit(X_train, y_train)
        
        self.logger.info(f"최적 AUC: {study.best_value:.4f}")
        self.logger.info(f"최적 파라미터: {best_params}")
        
        return model, best_params, study.best_value
    
    def _optimize_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, data_type: str):
        """XGBoost 최적화"""
        config = self.config['models']['xgboost']
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_estimators': trial.suggest_int('n_estimators', *config['n_estimators_range']),
                'max_depth': trial.suggest_int('max_depth', *config['max_depth_range']),
                'learning_rate': trial.suggest_float('learning_rate', *config['learning_rate_range']),
                'subsample': trial.suggest_float('subsample', *config['subsample_range']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *config['colsample_bytree_range']),
                'reg_alpha': trial.suggest_float('reg_alpha', *config['reg_alpha_range']),
                'reg_lambda': trial.suggest_float('reg_lambda', *config['reg_lambda_range']),
                'random_state': self.config['random_state'],
                'n_jobs': -1,
                'verbosity': 0
            }
            
            model = xgb.XGBClassifier(**params)
            
            # Cross validation (이미 전처리된 데이터 사용)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['random_state'])
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config['random_state']))
        
        # 진행상황 콜백 함수
        def progress_callback(study, trial):
            if trial.number % 5 == 0 or trial.number == config['n_trials'] - 1:
                best_value = study.best_value if study.best_value else 0
                self.logger.info(f"Trial {trial.number + 1}/{config['n_trials']} - Current: {trial.value:.4f}, Best: {best_value:.4f}")
        
        study.optimize(objective, n_trials=config['n_trials'], callbacks=[progress_callback])
        
        # 최적 모델 훈련
        best_params = study.best_params
        model = xgb.XGBClassifier(**best_params)
        
        # 모델 훈련 (이미 전처리된 데이터 사용)
        model.fit(X_train, y_train)
        
        self.logger.info(f"최적 AUC: {study.best_value:.4f}")
        self.logger.info(f"최적 파라미터: {best_params}")
        
        return model, best_params, study.best_value
    
    def _proper_cv_with_sampling(self, model, X: pd.DataFrame, y: pd.Series, data_type: str, cv_folds: int = 5):
        """
        샘플링 Data Leakage를 방지하는 올바른 Cross Validation
        각 CV fold마다 샘플링을 별도로 적용
        """
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config['random_state'])
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # 각 fold마다 별도로 분할
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 훈련 fold에만 샘플링 적용 (Data Leakage 방지)
            try:
                X_fold_train_resampled, y_fold_train_resampled = self.apply_sampling_strategy(
                    X_fold_train, y_fold_train, data_type
                )
                
                # 모델 복사 및 훈련
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_fold_train_resampled, y_fold_train_resampled)
                
                # 검증 fold에서 평가 (원본 데이터만 사용)
                y_pred_proba = model_copy.predict_proba(X_fold_val)[:, 1]
                score = roc_auc_score(y_fold_val, y_pred_proba)
                scores.append(score)
                
            except Exception as e:
                self.logger.warning(f"Fold {fold+1} {data_type.upper()} 적용 실패: {e}")
                # 샘플링 실패 시 원본 데이터로 훈련
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_fold_train, y_fold_train)
                y_pred_proba = model_copy.predict_proba(X_fold_val)[:, 1]
                score = roc_auc_score(y_fold_val, y_pred_proba)
                scores.append(score)
        
        return np.array(scores)
    
    def find_optimal_threshold(self, model_key: str):
        """각 모델별 최적 threshold 찾기"""
        self.logger.info(f"{model_key} 최적 Threshold 탐색")
        
        model = self.models[model_key]
        data_type = self.model_results[model_key]['data_type']
        
        X_val = self.data[data_type]['X_val']
        y_val = self.data[data_type]['y_val']
        
        # 검증 데이터에 대한 예측 확률
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        # 다양한 threshold에서의 성능 계산
        thresholds = np.arange(0.05, 0.5, 0.05)
        threshold_results = []
        
        for threshold in thresholds:
            y_val_pred = (y_val_proba >= threshold).astype(int)
            
            if len(np.unique(y_val_pred)) == 1:
                continue
            
            try:
                metrics = {
                    'threshold': threshold,
                    'precision': precision_score(y_val, y_val_pred, zero_division=0),
                    'recall': recall_score(y_val, y_val_pred, zero_division=0),
                    'f1': f1_score(y_val, y_val_pred, zero_division=0),
                    'balanced_accuracy': balanced_accuracy_score(y_val, y_val_pred)
                }
                threshold_results.append(metrics)
            except:
                continue
        
        if not threshold_results:
            self.logger.warning("최적 threshold 찾기 실패, 기본값 0.5 사용")
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
        
        self.logger.info(f"최종 선택: {final_threshold:.3f} ({metric_priority.upper()}: {final_value:.4f})")
        
        # Precision-Recall 곡선 데이터 저장
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_val, y_val_proba)
        
        # 시각화를 위한 임계값별 성능 데이터 추가
        threshold_analysis = {
            'all_thresholds': threshold_results,
            'optimal_by_metric': optimal_thresholds,
            'final_threshold': final_threshold,
            'final_metric': metric_priority,
            'final_value': final_value,
            'thresholds': [t['threshold'] for t in threshold_results],
            'f1_scores': [t['f1'] for t in threshold_results],
            'precisions': [t['precision'] for t in threshold_results],
            'recalls': [t['recall'] for t in threshold_results],
            'pr_curve': {
                'precision': precision_vals.tolist(),
                'recall': recall_vals.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        }
        
        return final_threshold, threshold_analysis
    
    def evaluate_model(self, model_key: str):
        """모델 평가 (최적 threshold 사용)"""
        # 최적 threshold 찾기
        optimal_threshold, threshold_analysis = self.find_optimal_threshold(model_key)
        
        model = self.models[model_key]
        data_type = self.model_results[model_key]['data_type']
        
        X_train = self.data[data_type]['X_train']
        y_train = self.data[data_type]['y_train']
        X_val = self.data[data_type]['X_val']
        y_val = self.data[data_type]['y_val']
        X_test = self.data[data_type]['X_test']
        y_test = self.data[data_type]['y_test']
        
        # 예측 (최적 threshold 사용)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_proba >= optimal_threshold).astype(int)
        
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_proba >= optimal_threshold).astype(int)
        
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # 훈련 성능
        train_metrics = {
            'auc': roc_auc_score(y_train, y_train_proba),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
            'average_precision': average_precision_score(y_train, y_train_proba)
        }
        
        # 검증 성능
        val_metrics = {
            'auc': roc_auc_score(y_val, y_val_proba),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, zero_division=0),
            'f1': f1_score(y_val, y_val_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_val, y_val_pred),
            'average_precision': average_precision_score(y_val, y_val_proba)
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
        self.model_results[model_key].update({
            'optimal_threshold': optimal_threshold,
            'threshold_analysis': threshold_analysis,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'predictions': {
                'y_train_proba': y_train_proba.tolist(),
                'y_val_proba': y_val_proba.tolist(),
                'y_test_proba': y_test_proba.tolist()
            }
        })
        
        self.logger.info(f"{model_key} 최종 평가 (Threshold: {optimal_threshold:.3f}):")
        self.logger.info(f"검증 - AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
        self.logger.info(f"테스트 - AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    def run_all_models(self):
        """모든 모델 실행"""
        self.logger.info("모든 모델 실행 시작")
        
        # Lasso 특성 선택
        if self.config['feature_selection']['enabled']:
            self.apply_lasso_feature_selection('normal')
        
        # 활성화된 데이터 타입 확인
        enabled_data_types = [dt for dt, config in self.config['sampling']['data_types'].items() if config['enabled']]
        enabled_models = [model for model, config in self.config['models'].items() if config['enabled']]
        
        self.logger.info(f"활성화된 데이터 타입: {enabled_data_types}")
        self.logger.info(f"활성화된 모델: {enabled_models}")
        
        # 데이터 타입별로 모델 실행
        for data_type in enabled_data_types:
            self.logger.info(f"{data_type.upper()} 데이터 처리")
            
            # 샘플링 적용 (SMOTE 등)
            if data_type != 'normal':
                X_train_resampled, y_train_resampled = self.apply_sampling_strategy(
                    self.data[data_type]['X_train'], 
                    self.data[data_type]['y_train'], 
                    data_type
                )
                self.data[data_type]['X_train'] = X_train_resampled
                self.data[data_type]['y_train'] = y_train_resampled
                self.logger.info(f"{data_type.upper()} 샘플링 적용 완료: {len(X_train_resampled):,}개 샘플")
            
            # 스케일링 적용 (SMOTE 후)
            X_train_scaled, X_val_scaled, X_test_scaled, scalers = self.apply_scaling(
                self.data[data_type]['X_train'].copy(),
                self.data[data_type]['X_val'].copy(), 
                self.data[data_type]['X_test'].copy(),
                data_type
            )
            
            # 스케일링된 데이터로 업데이트
            self.data[data_type]['X_train'] = X_train_scaled
            self.data[data_type]['X_val'] = X_val_scaled  
            self.data[data_type]['X_test'] = X_test_scaled
            
            # 스케일러 정보 저장
            if scalers:
                self.results['modeling_steps'][f'scaling_{data_type}'] = {
                    'scalers_info': {
                        k: {
                            'applied_columns': v['columns'],
                            'missing_columns': v.get('missing_columns', [])
                        } for k, v in scalers.items()
                    }
                }
            
            for model_name in enabled_models:
                model_key = f"{model_name}_{data_type}"
                
                # 모델 최적화 (이미 샘플링과 스케일링이 적용된 데이터 사용)
                model, best_params, cv_score = self.optimize_model(model_name, data_type)
                
                # 모델 및 결과 저장
                self.models[model_key] = model
                self.model_results[model_key] = {
                    'model_name': model_name,
                    'data_type': data_type,
                    'best_params': best_params,
                    'cv_score': cv_score
                }
                
                # 특성 중요도 저장 (tree 기반 모델의 경우)
                if hasattr(model, 'feature_importances_'):
                    feature_names = self.data[data_type]['X_train'].columns
                    self.model_results[model_key]['feature_importances'] = dict(
                        zip(feature_names, model.feature_importances_)
                    )
        
        # 모든 모델 평가
        self.logger.info("모든 모델 평가 및 Threshold 최적화")
        for model_key in self.models.keys():
            self.evaluate_model(model_key)
    
    def save_results(self):
        """결과 저장"""
        self.logger.info("결과 저장 시작")
        
        # 실험 정보 저장
        self.results['experiment_info'] = {
            'name': self.config['experiment']['name'],
            'config_path': str(self.config_path),
            'timestamp': datetime.now().isoformat(),
            'version': self.config['experiment']['version'],
            'description': self.config['experiment']['description']
        }
        
        # 모델 성능 결과 저장
        self.results['model_performance'] = self.model_results
        
        # 모델 저장
        models_dir = self.output_dir / 'models'
        for model_key, model in self.models.items():
            model_path = models_dir / f'{model_key.lower()}_model.joblib'
            joblib.dump(model, model_path)
            self.logger.info(f"{model_key} 모델 저장: {model_path}")
        
        # JSON 직렬화 가능한 형태로 변환
        def convert_to_serializable(obj):
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
        
        # 결과 저장
        results_path = self.output_dir / 'results' / 'modeling_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"전체 결과 저장: {results_path}")
        
        # 설정 저장
        config_path = self.output_dir / 'modeling_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        self.logger.info(f"설정 저장: {config_path}")
        
        # 요약 테이블 생성
        self.create_summary_table()
    
    def create_summary_table(self):
        """요약 테이블 생성"""
        summary_data = []
        
        for model_key, result in self.model_results.items():
            model_name = result['model_name']
            data_type = result['data_type']
            
            summary_data.append({
                'Model': model_name,
                'Data_Type': data_type.upper(),
                'Optimal_Threshold': result.get('optimal_threshold', 0.5),
                'CV_AUC': result['cv_score'],
                'Val_AUC': result['val_metrics']['auc'],
                'Val_F1': result['val_metrics']['f1'],
                'Test_AUC': result['test_metrics']['auc'],
                'Test_Precision': result['test_metrics']['precision'],
                'Test_Recall': result['test_metrics']['recall'],
                'Test_F1': result['test_metrics']['f1'],
                'Test_Balanced_Acc': result['test_metrics'].get('balanced_accuracy', 0),
                'Average_Precision': result['test_metrics'].get('average_precision', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 저장
        summary_path = self.output_dir / 'results' / 'summary_table.csv'
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        self.logger.info("실행 결과 요약:")
        self.logger.info(f"\n{summary_df.round(4).to_string()}")
        self.logger.info(f"요약 테이블 저장: {summary_path}")
        
        return summary_df
    
    def create_visualizations(self):
        """시각화 생성"""
        self.logger.info("시각화 생성 시작")
        
        viz_dir = self.output_dir / 'visualizations'
        
        try:
            # 1. ROC 곡선 비교
            self._plot_roc_curves(viz_dir)
            
            # 2. Precision-Recall 곡선 비교
            self._plot_pr_curves(viz_dir)
            
            # 3. 성능 비교 차트
            self._plot_performance_comparison(viz_dir)
            
            # 4. 특성 중요도 (tree 기반 모델)
            self._plot_feature_importance(viz_dir)
            
            # 5. 임계값 분석
            self._plot_threshold_analysis(viz_dir)
            
            # 6. Train vs Test 성능 비교 (Overfitting 분석)
            self._plot_train_vs_test_comparison(viz_dir)
            
            self.logger.info("시각화 생성 완료")
            
        except Exception as e:
            self.logger.error(f"시각화 생성 중 오류: {e}")
    
    def _plot_roc_curves(self, viz_dir: Path):
        """ROC 곡선 비교"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (model_key, result) in enumerate(self.model_results.items()):
            model_name = result['model_name']
            data_type = result['data_type']
            
            # 앙상블 모델 처리
            if model_name == 'ensemble':
                X_test = self.data['normal']['X_test']
                y_test = self.data['normal']['y_test']
                model = self.models[model_key]
                y_pred_proba = model.predict_proba(X_test)  # 앙상블은 이미 확률값 반환
            else:
                # 일반 모델 처리
                X_test = self.data[data_type]['X_test']
                y_test = self.data[data_type]['y_test']
                model = self.models[model_key]
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # ROC 곡선 계산
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = auc(fpr, tpr)
            
            # 플롯
            color = colors[i % len(colors)]
            linewidth = 3 if model_name == 'ensemble' else 2
            linestyle = '-' if model_name != 'ensemble' else '--'
            
            plt.plot(fpr, tpr, color=color, lw=linewidth, linestyle=linestyle,
                    label=f'{model_name}_{data_type} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curves(self, viz_dir: Path):
        """Precision-Recall 곡선 비교"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (model_key, result) in enumerate(self.model_results.items()):
            model_name = result['model_name']
            data_type = result['data_type']
            
            # 앙상블 모델 처리
            if model_name == 'ensemble':
                X_test = self.data['normal']['X_test']
                y_test = self.data['normal']['y_test']
                model = self.models[model_key]
                y_pred_proba = model.predict_proba(X_test)  # 앙상블은 이미 확률값 반환
            else:
                # 일반 모델 처리
                X_test = self.data[data_type]['X_test']
                y_test = self.data[data_type]['y_test']
                model = self.models[model_key]
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # PR 곡선 계산
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap_score = average_precision_score(y_test, y_pred_proba)
            
            # 플롯
            color = colors[i % len(colors)]
            plt.plot(recall, precision, color=color, lw=2,
                    label=f'{model_name}_{data_type} (AP = {ap_score:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'pr_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, viz_dir: Path):
        """성능 지표 비교"""
        metrics = ['test_auc', 'test_f1', 'test_precision', 'test_recall']
        metric_names = ['AUC', 'F1-Score', 'Precision', 'Recall']
        
        model_names = []
        metric_values = {metric: [] for metric in metrics}
        
        for model_key, result in self.model_results.items():
            model_name = f"{result['model_name']}_{result['data_type']}"
            model_names.append(model_name)
            
            metric_values['test_auc'].append(result['test_metrics']['auc'])
            metric_values['test_f1'].append(result['test_metrics']['f1'])
            metric_values['test_precision'].append(result['test_metrics']['precision'])
            metric_values['test_recall'].append(result['test_metrics']['recall'])
        
        # 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'plum', 'wheat']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            bars = ax.bar(model_names, metric_values[metric], 
                         color=colors[:len(model_names)], alpha=0.7)
            
            ax.set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(name)
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
            
            # 값 표시
            for bar, value in zip(bars, metric_values[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            # x축 레이블 회전
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, viz_dir: Path):
        """특성 중요도 시각화"""
        tree_models = {}
        
        # tree 기반 모델만 선택
        for model_key, result in self.model_results.items():
            if 'feature_importances' in result:
                tree_models[model_key] = result
        
        if not tree_models:
            self.logger.info("특성 중요도를 시각화할 tree 기반 모델이 없습니다.")
            return
        
        # 특성 중요도 비교
        fig, axes = plt.subplots(len(tree_models), 1, figsize=(12, 6*len(tree_models)))
        if len(tree_models) == 1:
            axes = [axes]
        
        for i, (model_key, result) in enumerate(tree_models.items()):
            model_name = f"{result['model_name']}_{result['data_type']}"
            importance = result['feature_importances']
            
            # 중요도 순으로 정렬
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            features, values = zip(*sorted_importance)
            
            ax = axes[i]
            bars = ax.barh(range(len(features)), values, color='skyblue', alpha=0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Feature Importance - {model_name}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 값 표시
            for j, (bar, value) in enumerate(zip(bars, values)):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                       f'{value:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_analysis(self, viz_dir: Path):
        """임계값 분석 시각화"""
        # 앙상블 모델 제외하고 threshold_analysis가 있는 모델만 선택
        valid_models = {}
        for model_key, result in self.model_results.items():
            if (result['model_name'] != 'ensemble' and 
                'threshold_analysis' in result and 
                'thresholds' in result['threshold_analysis']):
                valid_models[model_key] = result
        
        if not valid_models:
            self.logger.info("Threshold 분석을 위한 유효한 모델이 없습니다.")
            return
        
        fig, axes = plt.subplots(len(valid_models), 1, figsize=(12, 6*len(valid_models)))
        if len(valid_models) == 1:
            axes = [axes]
        
        for i, (model_key, result) in enumerate(valid_models.items()):
            model_name = f"{result['model_name']}_{result['data_type']}"
            
            threshold_data = result['threshold_analysis']
            
            # 데이터 존재 여부 확인
            if not all(key in threshold_data for key in ['thresholds', 'f1_scores', 'precisions', 'recalls']):
                self.logger.warning(f"{model_name}의 threshold 데이터가 불완전합니다.")
                continue
            
            thresholds = threshold_data['thresholds']
            f1_scores = threshold_data['f1_scores']
            precisions = threshold_data['precisions']
            recalls = threshold_data['recalls']
            
            ax = axes[i] if len(valid_models) > 1 else axes
            
            # 각 지표 플롯
            ax.plot(thresholds, f1_scores, 'b-', label='F1-Score', linewidth=2)
            ax.plot(thresholds, precisions, 'r-', label='Precision', linewidth=2)
            ax.plot(thresholds, recalls, 'g-', label='Recall', linewidth=2)
            
            # 최적 임계값 표시
            optimal_threshold = result.get('optimal_threshold', 0.5)
            ax.axvline(x=optimal_threshold, color='orange', linestyle='--', 
                      label=f'Optimal Threshold ({optimal_threshold:.3f})', linewidth=2)
            
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.set_title(f'Threshold Analysis - {model_name}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_train_vs_test_comparison(self, viz_dir: Path):
        """Train vs Test 성능 비교 (Overfitting 분석)"""
        # 앙상블 모델 제외
        model_results = {k: v for k, v in self.model_results.items() 
                        if v['model_name'] != 'ensemble' and 'train_metrics' in v}
        
        if not model_results:
            self.logger.info("Train vs Test 비교를 위한 모델이 없습니다.")
            return
        
        metrics = ['auc', 'f1', 'precision', 'recall']
        metric_names = ['AUC', 'F1-Score', 'Precision', 'Recall']
        
        # 데이터 준비
        model_names = []
        train_values = {metric: [] for metric in metrics}
        test_values = {metric: [] for metric in metrics}
        overfit_scores = {metric: [] for metric in metrics}
        
        for model_key, result in model_results.items():
            model_name = f"{result['model_name']}_{result['data_type']}"
            model_names.append(model_name)
            
            for metric in metrics:
                train_val = result['train_metrics'][metric]
                test_val = result['test_metrics'][metric]
                
                train_values[metric].append(train_val)
                test_values[metric].append(test_val)
                
                # Overfitting Score (Train - Test)
                # 높을수록 overfitting이 심함
                overfit_scores[metric].append(train_val - test_val)
        
        # 1. Train vs Test 성능 비교 (4개 서브플롯)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        x_positions = np.arange(len(model_names))
        width = 0.35
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            
            # Train과 Test 막대 그래프
            bars1 = ax.bar(x_positions - width/2, train_values[metric], width,
                          label='Train', alpha=0.8, color='skyblue')
            bars2 = ax.bar(x_positions + width/2, test_values[metric], width,
                          label='Test', alpha=0.8, color='lightcoral')
            
            ax.set_xlabel('Models')
            ax.set_ylabel(name)
            ax.set_title(f'{name}: Train vs Test Performance')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)
            
            # 값 표시
            for bar, value in zip(bars1, train_values[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                
            for bar, value in zip(bars2, test_values[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'train_vs_test_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Overfitting 점수 히트맵
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 데이터 매트릭스 생성
        overfit_matrix = np.array([overfit_scores[metric] for metric in metrics])
        
        # 히트맵 생성
        im = ax.imshow(overfit_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # 축 설정
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(metric_names)
        
        # 값 표시
        for i in range(len(metrics)):
            for j in range(len(model_names)):
                value = overfit_matrix[i, j]
                color = 'white' if abs(value) > 0.3 else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                       color=color, fontweight='bold')
        
        ax.set_title('Overfitting Score (Train - Test)\n높을수록 Overfitting 심함', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 컬러바 추가
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Overfitting Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'overfitting_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 모델별 전체 성능 요약 (Train/Val/Test)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 검증 데이터 값도 추가
        val_values = {metric: [] for metric in metrics}
        for model_key, result in model_results.items():
            for metric in metrics:
                val_values[metric].append(result['val_metrics'][metric])
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            
            x_pos = np.arange(len(model_names))
            width = 0.25
            
            # Train/Val/Test 막대 그래프
            bars1 = ax.bar(x_pos - width, train_values[metric], width, 
                          label='Train', alpha=0.8, color='lightblue')
            bars2 = ax.bar(x_pos, val_values[metric], width,
                          label='Validation', alpha=0.8, color='orange')
            bars3 = ax.bar(x_pos + width, test_values[metric], width,
                          label='Test', alpha=0.8, color='lightcoral')
            
            ax.set_xlabel('Models')
            ax.set_ylabel(name)
            ax.set_title(f'{name}: Train/Validation/Test Performance')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)
            
            # 간단한 값 표시 (Test만)
            for bar, value in zip(bars3, test_values[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'train_val_test_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Train vs Test 비교 시각화 완료")

    def run_ensemble(self):
        """앙상블 모델 실행"""
        if not self.config['ensemble']['enabled']:
            self.logger.info("앙상블이 비활성화되어 있습니다.")
            return
        
        self.logger.info("앙상블 모델 실행 시작")
        
        from .ensemble_pipeline import EnsemblePipeline
        
        # 앙상블에 사용할 모델들 선택
        ensemble_models = {}
        ensemble_config = self.config['ensemble']
        
        # 설정에서 지정된 모델과 데이터타입 조합만 선택
        target_models = ensemble_config.get('models', [])
        target_data_types = ensemble_config.get('data_types', ['normal'])
        
        for model_name in target_models:
            for data_type in target_data_types:
                model_key = f"{model_name}_{data_type}"
                if model_key in self.models:
                    ensemble_models[model_key] = self.models[model_key]
                    self.logger.info(f"앙상블에 추가: {model_key}")
        
        if len(ensemble_models) < 2:
            self.logger.warning("앙상블에 충분한 모델이 없습니다 (최소 2개 필요)")
            return
        
        # 앙상블 파이프라인 실행
        ensemble_pipeline = EnsemblePipeline(self.config, ensemble_models)
        
        # 앙상블 데이터 준비 (normal 데이터 사용)
        X_val = self.data['normal']['X_val']
        y_val = self.data['normal']['y_val']
        X_test = self.data['normal']['X_test']
        y_test = self.data['normal']['y_test']
        
        # 최적 임계값 찾기
        threshold_metric = ensemble_config.get('threshold_optimization', {}).get('metric_priority', 'f1')
        optimal_threshold, threshold_analysis = ensemble_pipeline.find_optimal_threshold(
            X_val, y_val, metric=threshold_metric
        )
        
        # 최종 평가
        test_metrics = ensemble_pipeline.evaluate_ensemble(X_test, y_test, optimal_threshold)
        
        # 앙상블 결과 저장
        ensemble_key = "ensemble_model"
        self.models[ensemble_key] = ensemble_pipeline
        self.model_results[ensemble_key] = {
            'model_name': 'ensemble',
            'data_type': 'combined_models',
            'cv_score': max([self.model_results[k]['cv_score'] for k in ensemble_models.keys()]),  # 최고 CV 점수 사용
            'optimal_threshold': optimal_threshold,
            'threshold_analysis': threshold_analysis,
            'val_metrics': {
                'auc': test_metrics['auc'],  # 앙상블의 검증 성능
                'f1': threshold_analysis['optimal_value'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'balanced_accuracy': test_metrics['balanced_accuracy'],
                'average_precision': test_metrics['average_precision']
            },
            'test_metrics': test_metrics,
            'ensemble_weights': ensemble_pipeline.weights,
            'component_models': list(ensemble_models.keys())
        }
        
        self.logger.info(f"앙상블 모델 완료 - Test AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        # 앙상블 모델 저장
        models_dir = self.output_dir / 'models'
        ensemble_path = models_dir / f'{ensemble_key}_model.joblib'
        joblib.dump(ensemble_pipeline, ensemble_path)
        self.logger.info(f"앙상블 모델 저장: {ensemble_path}")

    def run_pipeline(self) -> str:
        """전체 파이프라인 실행"""
        self.logger.info("=== 모델링 파이프라인 시작 ===")
        
        try:
            # 1. 데이터 로드
            self.load_data()
            
            # 2. 모든 모델 실행
            self.run_all_models()
            
            # 3. 앙상블 모델 실행
            self.run_ensemble()
            
            # 4. 시각화 생성
            self.create_visualizations()
            
            # 5. 결과 저장
            self.save_results()
            
            self.logger.info("=== 모델링 파이프라인 완료 ===")
            
            return str(self.output_dir)
            
        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 오류 발생: {str(e)}")
            raise


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='모델링 파이프라인')
    parser.add_argument('--config', '-c', type=str, 
                       default='config/modeling_config.yaml',
                       help='Config 파일 경로')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    pipeline = ModelingPipeline(args.config)
    experiment_dir = pipeline.run_pipeline()
    
    print(f"\n✅ 모델링 완료!")
    print(f"📁 결과 저장 위치: {experiment_dir}")


if __name__ == "__main__":
    main()