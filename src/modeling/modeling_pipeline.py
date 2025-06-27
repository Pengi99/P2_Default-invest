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
from sklearn.linear_model import LogisticRegression, LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
# Permutation Importance와 SHAP는 각 메서드 내에서 import

# 샘플링 라이브러리
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
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
            # 양성 샘플 수 확인
            n_minority = (y == 1).sum()
            k_neighbors = config.get('k_neighbors', 5)
            
            # 양성 샘플이 k_neighbors보다 적으면 조정
            if n_minority <= k_neighbors:
                if n_minority <= 1:
                    self.logger.warning(f"SMOTE 적용 불가: 양성 샘플이 {n_minority}개뿐. 원본 데이터 반환")
                    return X, y
                k_neighbors = max(1, n_minority - 1)
                self.logger.warning(f"k_neighbors를 {k_neighbors}로 조정 (양성 샘플: {n_minority}개)")
            
            try:
                # BorderlineSMOTE 시도
            smote = BorderlineSMOTE(
                sampling_strategy=config['sampling_strategy'],
                    random_state=config.get('random_state', self.config['random_state']),
                    k_neighbors=k_neighbors,
                    m_neighbors=min(config.get('m_neighbors', 10), k_neighbors),
                    kind=config.get('kind', 'borderline-1')
                )
                X_resampled, y_resampled = smote.fit_resample(X, y)
                return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
                
            except Exception as e:
                self.logger.warning(f"BorderlineSMOTE 실패 ({e}), 일반 SMOTE로 대체")
                try:
                    # 일반 SMOTE로 대체
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(
                        sampling_strategy=config['sampling_strategy'],
                        random_state=config.get('random_state', self.config['random_state']),
                        k_neighbors=k_neighbors
                    )
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
                    
                except Exception as e2:
                    self.logger.warning(f"일반 SMOTE도 실패 ({e2}), 원본 데이터 반환")
                    return X, y
        
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
            
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        
        elif data_type == 'combined':
            # 1단계: SMOTE 적용
            smote_config = config['smote']
            
            # 양성 샘플 수 확인
            n_minority = (y == 1).sum()
            k_neighbors = smote_config.get('k_neighbors', 5)
            
            # 양성 샘플이 k_neighbors보다 적으면 조정
            if n_minority <= k_neighbors:
                if n_minority <= 1:
                    self.logger.warning(f"Combined SMOTE 적용 불가: 양성 샘플이 {n_minority}개뿐. 원본 데이터 반환")
                    return X, y
                k_neighbors = max(1, n_minority - 1)
                self.logger.warning(f"Combined k_neighbors를 {k_neighbors}로 조정 (양성 샘플: {n_minority}개)")
            
            try:
                # BorderlineSMOTE 시도
            smote = BorderlineSMOTE(
                    sampling_strategy=smote_config['sampling_strategy'],
                    random_state=smote_config.get('random_state', self.config['random_state']),
                    k_neighbors=k_neighbors,
                    m_neighbors=min(smote_config.get('m_neighbors', 10), k_neighbors),
                    kind=smote_config.get('kind', 'borderline-1')
            )
            X_smote, y_smote = smote.fit_resample(X, y)
                
            except Exception as e:
                self.logger.warning(f"Combined BorderlineSMOTE 실패 ({e}), 일반 SMOTE로 대체")
                try:
                    # 일반 SMOTE로 대체
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(
                        sampling_strategy=smote_config['sampling_strategy'],
                        random_state=smote_config.get('random_state', self.config['random_state']),
                        k_neighbors=k_neighbors
                    )
                    X_smote, y_smote = smote.fit_resample(X, y)
                    
                except Exception as e2:
                    self.logger.warning(f"Combined 일반 SMOTE도 실패 ({e2}), 원본 데이터로 언더샘플링만 적용")
                    X_smote, y_smote = X, y
            
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
            
            return pd.DataFrame(X_combined, columns=X.columns), pd.Series(y_combined)
        
        else:
            self.logger.warning(f"알 수 없는 데이터 타입: {data_type}, 원본 데이터 반환")
            return X, y
    
    def apply_scaling(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        컬럼별 스케일링 적용 (Log 변환 포함, 존재하지 않는 컬럼은 자동으로 건너뛰기)
        
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
                # 스케일러 생성 및 적용
                if scaler_type.lower() == 'log':
                    # 로그 변환 처리
                    self._apply_log_transform(X_train, X_val, X_test, existing_columns, scaler_type)
                    scalers[scaler_type] = {
                        'scaler': 'log_transform',
                        'columns': existing_columns,
                        'missing_columns': missing_columns
                    }
                else:
                    # 기존 스케일러들
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
    
    def _apply_log_transform(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, columns: List[str], scaler_type: str):
        """
        로그 변환 적용 (음수/0 값 처리 포함)
        
        Args:
            X_train, X_val, X_test: 데이터프레임들
            columns: 변환할 컬럼 리스트
            scaler_type: 스케일러 타입 (로깅용)
        """
        for col in columns:
            for df_name, df in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
                # 음수 또는 0 값 확인
                min_val = df[col].min()
                
                if min_val <= 0:
                    # 음수나 0이 있는 경우 shift 적용 (최소값을 1로 만들기)
                    shift_value = 1 - min_val
                    self.logger.info(f"{scaler_type} - {col} ({df_name}): 최소값 {min_val:.4f}, shift {shift_value:.4f} 적용")
                    df[col] = np.log1p(df[col] + shift_value)  # log1p = log(1+x)
                else:
                    # 양수만 있는 경우 직접 로그 변환
                    df[col] = np.log1p(df[col])  # log1p는 수치적으로 더 안정적
    
    def apply_feature_selection(self, data_type: str = 'normal'):
        """이진 분류용 특성 선택 적용"""
        if not self.config['feature_selection']['enabled']:
            self.logger.info("특성 선택이 비활성화되어 있습니다.")
            return
        
        method = self.config['feature_selection'].get('method', 'logistic_regression_cv')
        self.logger.info(f"특성 선택 방법: {method.upper()} ({data_type.upper()})")
        
        X_train = self.data[data_type]['X_train']
        y_train = self.data[data_type]['y_train']
        
        if method == 'logistic_regression_cv':
            selected_features, selection_results = self._apply_logistic_regression_cv_selection(X_train, y_train)
        elif method == 'lasso_cv':
            # 기존 LassoCV 방법 (연속 회귀용이지만 참고용으로 유지)
            selected_features, selection_results = self._apply_lasso_cv_selection(X_train, y_train)
        elif method == 'permutation_importance':
            selected_features, selection_results = self._apply_permutation_importance_selection(X_train, y_train)
        elif method == 'shap':
            selected_features, selection_results = self._apply_shap_selection(X_train, y_train)
        else:
            self.logger.warning(f"지원하지 않는 특성 선택 방법: {method}, LogisticRegressionCV 사용")
            selected_features, selection_results = self._apply_logistic_regression_cv_selection(X_train, y_train)
        
        # 모든 데이터 타입에 동일한 특성 적용
        enabled_data_types = [dt for dt, config in self.config['sampling']['data_types'].items() if config['enabled']]
        for dt in enabled_data_types:
            for split in ['X_train', 'X_val', 'X_test']:
                self.data[dt][split] = self.data[dt][split][selected_features]
        
        self.logger.info(f"선택된 특성: {len(selected_features)}/{len(X_train.columns)}")
        
        # 결과 저장
        self.results['modeling_steps']['feature_selection'] = selection_results
        
        # 파일로 저장
        with open(self.output_dir / 'results' / f'{method}_selection.json', 'w', encoding='utf-8') as f:
            json.dump(selection_results, f, indent=2, ensure_ascii=False)
    
    def _apply_logistic_regression_cv_selection(self, X_train: pd.DataFrame, y_train: pd.Series):
        """LogisticRegressionCV를 이용한 특성 선택 (이진 분류 전용)"""
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler
        
        config = self.config['feature_selection']['logistic_regression_cv']
        
        # 스케일링 (L1 정규화를 위해 필수)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # LogisticRegressionCV
        logistic_cv = LogisticRegressionCV(
            Cs=config['Cs'],
            cv=config['cv_folds'],
            penalty=config['penalty'],
            solver=config['solver'],
            max_iter=config['max_iter'],
            scoring=config.get('scoring', 'roc_auc'),
            random_state=self.config['random_state'],
            n_jobs=self.config.get('performance', {}).get('n_jobs', 1)
        )
        
        logistic_cv.fit(X_train_scaled, y_train)
        
        # 특성 선택
        threshold = config['threshold']
        coefficients = logistic_cv.coef_[0]  # 이진 분류의 경우 (1, n_features)
        
        if threshold == 'median':
            threshold_value = np.median(np.abs(coefficients[coefficients != 0]))
        elif threshold == 'mean':
            threshold_value = np.mean(np.abs(coefficients[coefficients != 0]))
        else:
            threshold_value = float(threshold)
        
        selected_mask = np.abs(coefficients) >= threshold_value
        selected_features = X_train.columns[selected_mask].tolist()
        
        self.logger.info(f"최적 C: {logistic_cv.C_[0]:.6f}")
        self.logger.info(f"CV 점수: {logistic_cv.scores_[1].mean():.4f} (±{logistic_cv.scores_[1].std():.4f})")
        
        # 결과 저장
        selection_results = {
            'method': 'logistic_regression_cv',
            'optimal_C': float(logistic_cv.C_[0]),
            'cv_scores_mean': float(logistic_cv.scores_[1].mean()),
            'cv_scores_std': float(logistic_cv.scores_[1].std()),
            'threshold_value': float(threshold_value),
            'original_features': len(X_train.columns),
            'selected_features': len(selected_features),
            'selected_feature_names': selected_features,
            'feature_coefficients': dict(zip(X_train.columns, coefficients)),
            'penalty': config['penalty'],
            'solver': config['solver']
        }
        
        return selected_features, selection_results
    
    def _apply_lasso_cv_selection(self, X_train: pd.DataFrame, y_train: pd.Series):
        """기존 LassoCV 방법 (연속 회귀용)"""
        from sklearn.preprocessing import StandardScaler
        
        config = self.config['feature_selection']['lasso_cv']
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Lasso CV
        lasso_cv = LassoCV(
            alphas=config['alphas'],
            cv=config['cv_folds'],
            random_state=self.config['random_state'],
            n_jobs=self.config.get('performance', {}).get('n_jobs', 1)
        )
        lasso_cv.fit(X_train_scaled, y_train)
        
        # 특성 선택
        threshold = config['threshold']
        if threshold == 'median':
            threshold_value = np.median(np.abs(lasso_cv.coef_[lasso_cv.coef_ != 0]))
        elif threshold == 'mean':
            threshold_value = np.mean(np.abs(lasso_cv.coef_[lasso_cv.coef_ != 0]))
        else:
            threshold_value = float(threshold)
        
        selected_mask = np.abs(lasso_cv.coef_) >= threshold_value
        selected_features = X_train.columns[selected_mask].tolist()
        
        self.logger.info(f"최적 alpha: {lasso_cv.alpha_:.6f}")
        
        # 결과 저장
        selection_results = {
            'method': 'lasso_cv',
            'optimal_alpha': float(lasso_cv.alpha_),
            'threshold_value': float(threshold_value),
            'original_features': len(X_train.columns),
            'selected_features': len(selected_features),
            'selected_feature_names': selected_features,
            'feature_coefficients': dict(zip(X_train.columns, lasso_cv.coef_))
        }
        
        return selected_features, selection_results
    
    def _apply_permutation_importance_selection(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Permutation Importance를 이용한 특성 선택"""
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb
        
        config = self.config['feature_selection']['permutation_importance']
        estimator_name = config.get('base_estimator', 'random_forest')
        
        self.logger.info(f"Permutation Importance 기본 모델: {estimator_name}")
        
        # 기본 추정기 생성
        estimator_params = config.get('estimator_params', {}).get(estimator_name, {})
        
        if estimator_name == 'random_forest':
            estimator = RandomForestClassifier(
                random_state=self.config['random_state'],
                n_jobs=-1,
                **estimator_params
            )
            X_train_processed = X_train.values
        elif estimator_name == 'logistic_regression':
            # 로지스틱 회귀는 스케일링 필요
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train)
            estimator = LogisticRegression(
                random_state=self.config['random_state'],
                **estimator_params
            )
        elif estimator_name == 'xgboost':
            estimator = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=self.config['random_state'],
                n_jobs=-1,
                verbosity=0,
                **estimator_params
            )
            X_train_processed = X_train.values
        else:
            raise ValueError(f"지원하지 않는 estimator: {estimator_name}")
        
        # 모델 훈련
        self.logger.info("기본 모델 훈련 중...")
        estimator.fit(X_train_processed, y_train)
        
        # Permutation Importance 계산
        self.logger.info("Permutation Importance 계산 중...")
        perm_importance = permutation_importance(
            estimator, X_train_processed, y_train,
            n_repeats=config.get('n_repeats', 10),
            random_state=config.get('random_state', self.config['random_state']),
            scoring=config.get('scoring', 'roc_auc'),
            n_jobs=self.config.get('performance', {}).get('n_jobs', 1)
        )
        
        # 특성 선택
        threshold = config['threshold']
        importance_scores = perm_importance.importances_mean
        
        if threshold == 'median':
            threshold_value = np.median(importance_scores[importance_scores > 0])
        elif threshold == 'mean':
            threshold_value = np.mean(importance_scores[importance_scores > 0])
        else:
            threshold_value = float(threshold)
        
        selected_mask = importance_scores >= threshold_value
        selected_features = X_train.columns[selected_mask].tolist()
        
        # max_features 제한 적용
        max_features = config.get('max_features')
        if max_features and len(selected_features) > max_features:
            # 중요도 순으로 정렬하여 상위 max_features개만 선택
            feature_importance_pairs = list(zip(X_train.columns, importance_scores))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            selected_features = [pair[0] for pair in feature_importance_pairs[:max_features]]
        
        self.logger.info(f"임계값: {threshold_value:.6f}")
        self.logger.info(f"평균 중요도: {importance_scores.mean():.6f} (±{perm_importance.importances_std.mean():.6f})")
        
        # 결과 저장
        selection_results = {
            'method': 'permutation_importance',
            'base_estimator': estimator_name,
            'n_repeats': config.get('n_repeats', 10),
            'scoring': config.get('scoring', 'roc_auc'),
            'threshold_value': float(threshold_value),
            'original_features': len(X_train.columns),
            'selected_features': len(selected_features),
            'selected_feature_names': selected_features,
            'feature_importances': dict(zip(X_train.columns, importance_scores)),
            'feature_importances_std': dict(zip(X_train.columns, perm_importance.importances_std)),
            'estimator_params': estimator_params
        }
        
        return selected_features, selection_results
    
    def _apply_shap_selection(self, X_train: pd.DataFrame, y_train: pd.Series):
        """SHAP를 이용한 특성 선택"""
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP 패키지가 설치되지 않았습니다. 'pip install shap' 명령으로 설치해주세요.")
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb
        
        config = self.config['feature_selection']['shap']
        estimator_name = config.get('base_estimator', 'random_forest')
        
        self.logger.info(f"SHAP 기본 모델: {estimator_name}")
        
        # 기본 추정기 생성
        estimator_params = config.get('estimator_params', {}).get(estimator_name, {})
        
        if estimator_name == 'random_forest':
            estimator = RandomForestClassifier(
                random_state=self.config['random_state'],
                n_jobs=-1,
                **estimator_params
            )
            X_train_processed = X_train
        elif estimator_name == 'logistic_regression':
            # 로지스틱 회귀는 스케일링 필요
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_train_processed = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            estimator = LogisticRegression(
                random_state=self.config['random_state'],
                **estimator_params
            )
        elif estimator_name == 'xgboost':
            estimator = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=self.config['random_state'],
                n_jobs=-1,
                verbosity=0,
                **estimator_params
            )
            X_train_processed = X_train
        else:
            raise ValueError(f"지원하지 않는 estimator: {estimator_name}")
        
        # 모델 훈련
        self.logger.info("기본 모델 훈련 중...")
        estimator.fit(X_train_processed, y_train)
        
        # 샘플 크기 조정 (SHAP 계산 속도 개선)
        sample_size = config.get('sample_size', 1000)
        if len(X_train_processed) > sample_size:
            sample_indices = np.random.choice(len(X_train_processed), sample_size, replace=False)
            X_sample = X_train_processed.iloc[sample_indices]
        else:
            X_sample = X_train_processed
        
        # SHAP Explainer 생성
        self.logger.info("SHAP Explainer 생성 중...")
        explainer_type = config.get('explainer_type', 'auto')
        
        try:
            if explainer_type == 'auto':
                # 모델 타입에 따라 자동 선택
                if estimator_name == 'random_forest':
                    explainer = shap.TreeExplainer(estimator)
                elif estimator_name == 'xgboost':
                    explainer = shap.TreeExplainer(estimator)
                else:  # logistic_regression
                    explainer = shap.LinearExplainer(estimator, X_sample)
            elif explainer_type == 'tree':
                explainer = shap.TreeExplainer(estimator)
            elif explainer_type == 'linear':
                explainer = shap.LinearExplainer(estimator, X_sample)
            elif explainer_type == 'kernel':
                explainer = shap.KernelExplainer(estimator.predict_proba, X_sample.iloc[:100])  # 작은 배경 샘플
            elif explainer_type == 'permutation':
                explainer = shap.PermutationExplainer(estimator.predict_proba, X_sample.iloc[:100])
            else:
                raise ValueError(f"지원하지 않는 explainer_type: {explainer_type}")
        except Exception as e:
            self.logger.warning(f"지정된 explainer 생성 실패 ({e}), Permutation Explainer로 대체")
            explainer = shap.PermutationExplainer(estimator.predict_proba, X_sample.iloc[:100])
        
        # SHAP 값 계산
        self.logger.info("SHAP 값 계산 중...")
        try:
            if explainer_type in ['kernel', 'permutation']:
                shap_values = explainer(X_sample)
                if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
                    # 이진 분류의 경우 양성 클래스 SHAP 값 사용
                    shap_values_array = shap_values.values[:, :, 1]
                else:
                    shap_values_array = shap_values.values
            else:
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    # 이진 분류의 경우 양성 클래스 SHAP 값 사용
                    shap_values_array = shap_values[1]
                else:
                    shap_values_array = shap_values
        except Exception as e:
            self.logger.warning(f"SHAP 값 계산 실패 ({e}), 기본 feature importance 사용")
            if hasattr(estimator, 'feature_importances_'):
                shap_values_array = estimator.feature_importances_.reshape(1, -1)
            else:
                # 로지스틱 회귀의 경우 계수의 절댓값 사용
                shap_values_array = np.abs(estimator.coef_).reshape(1, -1)
        
        # 특성별 평균 절댓값 SHAP 값 계산
        mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)
        
        # 특성 선택
        threshold = config['threshold']
        
        if threshold == 'median':
            threshold_value = np.median(mean_abs_shap[mean_abs_shap > 0])
        elif threshold == 'mean':
            threshold_value = np.mean(mean_abs_shap[mean_abs_shap > 0])
        else:
            threshold_value = float(threshold)
        
        selected_mask = mean_abs_shap >= threshold_value
        selected_features = X_train.columns[selected_mask].tolist()
        
        # max_features 제한 적용
        max_features = config.get('max_features')
        if max_features and len(selected_features) > max_features:
            # SHAP 값 순으로 정렬하여 상위 max_features개만 선택
            feature_shap_pairs = list(zip(X_train.columns, mean_abs_shap))
            feature_shap_pairs.sort(key=lambda x: x[1], reverse=True)
            selected_features = [pair[0] for pair in feature_shap_pairs[:max_features]]
        
        self.logger.info(f"임계값: {threshold_value:.6f}")
        self.logger.info(f"평균 |SHAP|: {mean_abs_shap.mean():.6f} (±{mean_abs_shap.std():.6f})")
        
        # 결과 저장
        selection_results = {
            'method': 'shap',
            'base_estimator': estimator_name,
            'explainer_type': explainer_type,
            'sample_size': len(X_sample),
            'threshold_value': float(threshold_value),
            'original_features': len(X_train.columns),
            'selected_features': len(selected_features),
            'selected_feature_names': selected_features,
            'mean_abs_shap_values': dict(zip(X_train.columns, mean_abs_shap)),
            'estimator_params': estimator_params
        }
        
        return selected_features, selection_results
    
    # 기존 메서드명 호환성을 위한 별칭
    def apply_lasso_feature_selection(self, data_type: str = 'normal'):
        """기존 메서드명 호환성을 위한 별칭"""
        return self.apply_feature_selection(data_type)
    
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
        optimization_config = self.config['models'].get('optimization', {})
        primary_metric = optimization_config.get('primary_metric', 'roc_auc')
        
        # Config에서 class_weight 설정 가져오기
        class_weight = config.get('class_weight', 'balanced')
        self.logger.info(f"Logistic Regression 클래스 불균형 가중치: class_weight={class_weight}")
        self.logger.info(f"최적화 메트릭: {primary_metric.upper()}")
        
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
            max_iter = trial.suggest_int('max_iter', *config['max_iter_range'])
            
            params = {
                'penalty': penalty,
                'C': C,
                'solver': solver,
                'max_iter': max_iter,
                'class_weight': class_weight,  # Config에서 가져온 클래스 가중치 적용
                'random_state': self.config['random_state'],
                'tol': 1e-4  # 수렴 기준 완화
            }
            
            if penalty == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', *config['l1_ratio_range'])
            
            model = LogisticRegression(**params)
            
            try:
                # Cross validation with proper sampling (데이터 누수 방지)
                scores = self._proper_cv_with_sampling(model, X_train, y_train, data_type, cv_folds=5, scoring=primary_metric)
            
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
        best_params['class_weight'] = class_weight  # Config에서 가져온 클래스 가중치 적용
        
        model = LogisticRegression(**best_params)
        
        # 최종 모델 훈련: 전체 훈련 데이터에 스케일링 → 샘플링 적용
        X_train_scaled, _, _, scalers = self.apply_scaling(
            X_train.copy(),
            X_train.copy(),  # 더미 데이터
            X_train.copy(),  # 더미 데이터
            data_type
        )
        
        X_train_final, y_train_final = self.apply_sampling_strategy(
            X_train_scaled, y_train, data_type
        )
        
        model.fit(X_train_final, y_train_final)
        
        self.logger.info(f"최적 AUC: {study.best_value:.4f}")
        self.logger.info(f"최적 파라미터: {best_params}")
        
        return model, best_params, study.best_value
    
    def _optimize_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, data_type: str):
        """랜덤 포레스트 최적화"""
        config = self.config['models']['random_forest']
        optimization_config = self.config['models'].get('optimization', {})
        primary_metric = optimization_config.get('primary_metric', 'roc_auc')
        
        # Config에서 class_weight 설정 가져오기
        class_weight = config.get('class_weight', 'balanced')
        self.logger.info(f"Random Forest 클래스 불균형 가중치: class_weight={class_weight}")
        self.logger.info(f"최적화 메트릭: {primary_metric.upper()}")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *config['n_estimators_range']),
                'max_depth': trial.suggest_int('max_depth', *config['max_depth_range']),
                'min_samples_split': trial.suggest_int('min_samples_split', *config['min_samples_split_range']),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', *config['min_samples_leaf_range']),
                'max_features': trial.suggest_float('max_features', *config['max_features_range']),
                'class_weight': class_weight,  # Config에서 가져온 클래스 가중치 적용
                'random_state': self.config['random_state'],
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            
            # Cross validation with proper sampling (데이터 누수 방지)
            scores = self._proper_cv_with_sampling(model, X_train, y_train, data_type, cv_folds=5, scoring=primary_metric)
            
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
        best_params['class_weight'] = class_weight  # Config에서 가져온 클래스 가중치 적용
        model = RandomForestClassifier(**best_params)
        
        # 최종 모델 훈련: 전체 훈련 데이터에 스케일링 → 샘플링 적용
        X_train_scaled, _, _, scalers = self.apply_scaling(
            X_train.copy(),
            X_train.copy(),  # 더미 데이터
            X_train.copy(),  # 더미 데이터
            data_type
        )
        
        X_train_final, y_train_final = self.apply_sampling_strategy(
            X_train_scaled, y_train, data_type
        )
        
        model.fit(X_train_final, y_train_final)
        
        self.logger.info(f"최적 AUC: {study.best_value:.4f}")
        self.logger.info(f"최적 파라미터: {best_params}")
        
        return model, best_params, study.best_value
    
    def _optimize_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, data_type: str):
        """XGBoost 최적화"""
        config = self.config['models']['xgboost']
        optimization_config = self.config['models'].get('optimization', {})
        primary_metric = optimization_config.get('primary_metric', 'roc_auc')
        
        # Config에서 클래스 불균형 처리 방식 가져오기
        class_weight_mode = config.get('class_weight_mode', 'scale_pos_weight')
        scale_pos_weight_setting = config.get('scale_pos_weight', 'auto')
        
        # scale_pos_weight 계산
        if scale_pos_weight_setting == 'auto':
            # 자동 계산: negative / positive
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        elif scale_pos_weight_setting == 'balanced':
            # balanced와 동일한 방식
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        else:
            # 직접 지정된 숫자값
            scale_pos_weight = float(scale_pos_weight_setting)
        
        self.logger.info(f"XGBoost 클래스 불균형 처리: {class_weight_mode}")
        self.logger.info(f"scale_pos_weight = {scale_pos_weight:.2f} (설정: {scale_pos_weight_setting})")
        self.logger.info(f"최적화 메트릭: {primary_metric.upper()}")
        
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
                'scale_pos_weight': scale_pos_weight,  # Config에서 계산된 불균형 가중치
                'random_state': self.config['random_state'],
                'n_jobs': -1,
                'verbosity': 0
            }
            
            model = xgb.XGBClassifier(**params)
            
            # Cross validation with proper sampling (데이터 누수 방지)
            scores = self._proper_cv_with_sampling(model, X_train, y_train, data_type, cv_folds=5, scoring=primary_metric)
            
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
        best_params['scale_pos_weight'] = scale_pos_weight  # Config에서 계산된 클래스 가중치 적용
        model = xgb.XGBClassifier(**best_params)
        
        # 최종 모델 훈련: 전체 훈련 데이터에 스케일링 → 샘플링 적용
        X_train_scaled, _, _, scalers = self.apply_scaling(
            X_train.copy(),
            X_train.copy(),  # 더미 데이터
            X_train.copy(),  # 더미 데이터
            data_type
        )
        
        X_train_final, y_train_final = self.apply_sampling_strategy(
            X_train_scaled, y_train, data_type
        )
        
        model.fit(X_train_final, y_train_final)
        
        self.logger.info(f"최적 AUC: {study.best_value:.4f}")
        self.logger.info(f"최적 파라미터: {best_params}")
        
        return model, best_params, study.best_value
    
    def _proper_cv_with_sampling(self, model, X: pd.DataFrame, y: pd.Series, data_type: str, cv_folds: int = 5, scoring='roc_auc'):
        """
        샘플링 Data Leakage를 방지하는 올바른 Cross Validation
        각 CV fold마다 스케일링 → 샘플링을 순서대로 적용
        """
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config['random_state'])
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # 각 fold마다 별도로 분할
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                # 1단계: 스케일링 적용 (훈련 fold에서 fit, 검증 fold에서 transform)
                X_fold_train_scaled, X_fold_val_scaled, _, scalers = self.apply_scaling(
                    X_fold_train.copy(),
                    X_fold_val.copy(), 
                    X_fold_val.copy(),  # 더미 테스트 데이터 (사용 안 함)
                    data_type
                )
                
                # 2단계: 샘플링 적용 (스케일링된 훈련 데이터에만)
                X_fold_train_resampled, y_fold_train_resampled = self.apply_sampling_strategy(
                    X_fold_train_scaled, y_fold_train, data_type
                )
                
                # 모델 복사 및 훈련
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_fold_train_resampled, y_fold_train_resampled)
                
                # 검증 fold에서 평가 (스케일링된 검증 데이터 사용)
                y_pred_proba = model_copy.predict_proba(X_fold_val_scaled)[:, 1]
                
                # 다양한 평가 메트릭 계산
                if scoring == 'roc_auc':
                score = roc_auc_score(y_fold_val, y_pred_proba)
                elif scoring == 'average_precision':
                    score = average_precision_score(y_fold_val, y_pred_proba)
                elif scoring == 'f1':
                    # F1의 경우 임계값 0.5 사용
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    score = f1_score(y_fold_val, y_pred, zero_division=0)
                else:
                    score = roc_auc_score(y_fold_val, y_pred_proba)  # 기본값
                
                scores.append(score)
                
            except Exception as e:
                self.logger.warning(f"Fold {fold+1} {data_type.upper()} 처리 실패: {e}")
                # 처리 실패 시 원본 데이터로 훈련
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_fold_train, y_fold_train)
                y_pred_proba = model_copy.predict_proba(X_fold_val)[:, 1]
                
                if scoring == 'roc_auc':
                score = roc_auc_score(y_fold_val, y_pred_proba)
                elif scoring == 'average_precision':
                    score = average_precision_score(y_fold_val, y_pred_proba)
                elif scoring == 'f1':
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    score = f1_score(y_fold_val, y_pred, zero_division=0)
                else:
                    score = roc_auc_score(y_fold_val, y_pred_proba)
                
                scores.append(score)
        
        return np.array(scores)
    
    def _calculate_cv_metrics(self, model, X_train: pd.DataFrame, y_train: pd.Series, data_type: str) -> Dict[str, float]:
        """
        다양한 CV 메트릭 계산 (데이터 누수 방지)
        """
        metrics = {}
        
        # ROC AUC
        auc_scores = self._proper_cv_with_sampling(model, X_train, y_train, data_type, cv_folds=5, scoring='roc_auc')
        metrics['cv_auc_mean'] = float(auc_scores.mean())
        metrics['cv_auc_std'] = float(auc_scores.std())
        
        # Average Precision
        ap_scores = self._proper_cv_with_sampling(model, X_train, y_train, data_type, cv_folds=5, scoring='average_precision')
        metrics['cv_average_precision_mean'] = float(ap_scores.mean())
        metrics['cv_average_precision_std'] = float(ap_scores.std())
        
        # F1 Score
        f1_scores = self._proper_cv_with_sampling(model, X_train, y_train, data_type, cv_folds=5, scoring='f1')
        metrics['cv_f1_mean'] = float(f1_scores.mean())
        metrics['cv_f1_std'] = float(f1_scores.std())
        
        self.logger.info(f"CV 메트릭 - AUC: {metrics['cv_auc_mean']:.4f} (±{metrics['cv_auc_std']:.4f})")
        self.logger.info(f"CV 메트릭 - AP: {metrics['cv_average_precision_mean']:.4f} (±{metrics['cv_average_precision_std']:.4f})")
        self.logger.info(f"CV 메트릭 - F1: {metrics['cv_f1_mean']:.4f} (±{metrics['cv_f1_std']:.4f})")
        
        return metrics
    

    
    def evaluate_model(self, model_key: str):
        """모델 평가 (최적 threshold 사용)"""
        model = self.models[model_key]
        data_type = self.model_results[model_key]['data_type']
        
        # 원본 데이터 가져오기
        X_train = self.data[data_type]['X_train']
        y_train = self.data[data_type]['y_train']
        X_val = self.data[data_type]['X_val']
        y_val = self.data[data_type]['y_val']
        X_test = self.data[data_type]['X_test']
        y_test = self.data[data_type]['y_test']
        
        # 평가용 데이터 전처리 (모델 훈련과 동일한 방식)
        # 1단계: 스케일링 적용 (훈련 데이터로 fit, 검증/테스트 데이터로 transform)
        X_train_scaled, X_val_scaled, X_test_scaled, scalers = self.apply_scaling(
            X_train.copy(),
            X_val.copy(),
            X_test.copy(),
            data_type
        )
        
        # 2단계: 훈련 데이터에만 샘플링 적용 (평가 데이터는 원본 유지)
        X_train_final, y_train_final = self.apply_sampling_strategy(
            X_train_scaled, y_train, data_type
        )
        
        # 최적 threshold 찾기 (스케일링된 검증 데이터 사용)
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        # 임계값 범위 설정 (config에서 가져오기)
        threshold_config = self.config.get('threshold_optimization', {}).get('search_range', {})
        pos_rate = y_val.mean()
        
        # 동적 임계값 범위 계산
        low_threshold = max(threshold_config.get('low', 0.0005), pos_rate/5)
        high_threshold = threshold_config.get('high', 0.30)
        n_grid = threshold_config.get('n_grid', 300)
        
        # 임계값 배열 생성
        thresholds = np.linspace(low_threshold, high_threshold, n_grid)
        threshold_results = []
        
        self.logger.info(f"{model_key} 최적 Threshold 탐색")
        self.logger.info(f"임계값 범위: {low_threshold:.4f} ~ {high_threshold:.4f} ({n_grid}개 점)")
        
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
            optimal_threshold = 0.5
            threshold_analysis = {}
        else:
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
                optimal_threshold = optimal_thresholds[metric_priority]['threshold']
            final_value = optimal_thresholds[metric_priority]['value']
        else:
                optimal_threshold = optimal_thresholds['f1']['threshold']
            final_value = optimal_thresholds['f1']['value']
        
            self.logger.info(f"최종 선택: {optimal_threshold:.3f} ({metric_priority.upper()}: {final_value:.4f})")
        
        # Precision-Recall 곡선 데이터 저장
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_val, y_val_proba)
        
            # 시각화를 위한 임계값별 성능 데이터 추가
        threshold_analysis = {
            'all_thresholds': threshold_results,
            'optimal_by_metric': optimal_thresholds,
                'final_threshold': optimal_threshold,
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
        
        # 예측 (최적 threshold 사용, 스케일링된 데이터 사용)
        y_train_proba = model.predict_proba(X_train_final)[:, 1]
        y_train_pred = (y_train_proba >= optimal_threshold).astype(int)
        
        y_val_pred = (y_val_proba >= optimal_threshold).astype(int)
        
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # 훈련 성능 (샘플링된 데이터 기준)
        train_metrics = {
            'auc': roc_auc_score(y_train_final, y_train_proba),
            'precision': precision_score(y_train_final, y_train_pred, zero_division=0),
            'recall': recall_score(y_train_final, y_train_pred, zero_division=0),
            'f1': f1_score(y_train_final, y_train_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_train_final, y_train_pred),
            'average_precision': average_precision_score(y_train_final, y_train_proba)
        }
        
        # 검증 성능 (원본 라벨 기준)
        val_metrics = {
            'auc': roc_auc_score(y_val, y_val_proba),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, zero_division=0),
            'f1': f1_score(y_val, y_val_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_val, y_val_pred),
            'average_precision': average_precision_score(y_val, y_val_proba)
        }
        
        # 테스트 성능 (원본 라벨 기준)
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
        
        # 특성 선택 (이진 분류용)
        if self.config['feature_selection']['enabled']:
            self.apply_feature_selection('normal')
        
        # 활성화된 데이터 타입 확인
        enabled_data_types = [dt for dt, config in self.config['sampling']['data_types'].items() if config['enabled']]
        enabled_models = [model for model, config in self.config['models'].items() 
                         if isinstance(config, dict) and config.get('enabled', False)]
        
        self.logger.info(f"활성화된 데이터 타입: {enabled_data_types}")
        self.logger.info(f"활성화된 모델: {enabled_models}")
        
        # 데이터 타입별로 모델 실행
        for data_type in enabled_data_types:
            self.logger.info(f"{data_type.upper()} 데이터 처리")
            
            # 원본 데이터 사용 (스케일링과 샘플링은 CV 내부에서 처리)
            # 각 데이터 타입은 동일한 원본 데이터를 사용하되, CV 과정에서 다르게 처리됨
            self.data[data_type] = {
                'X_train': self.data['normal']['X_train'].copy(),
                'X_val': self.data['normal']['X_val'].copy(), 
                'X_test': self.data['normal']['X_test'].copy(),
                'y_train': self.data['normal']['y_train'].copy(),
                'y_val': self.data['normal']['y_val'].copy(),
                'y_test': self.data['normal']['y_test'].copy()
                }
            
            for model_name in enabled_models:
                model_key = f"{model_name}_{data_type}"
                
                # 모델 최적화 (CV 내부에서 스케일링과 샘플링 처리)
                model, best_params, cv_score = self.optimize_model(model_name, data_type)
                
                # 추가 CV 메트릭 계산
                cv_metrics = self._calculate_cv_metrics(model, self.data[data_type]['X_train'], self.data[data_type]['y_train'], data_type)
                
                # 모델 및 결과 저장
                self.models[model_key] = model
                self.model_results[model_key] = {
                    'model_name': model_name,
                    'data_type': data_type,
                    'best_params': best_params,
                    'cv_score': cv_score,
                    'cv_metrics': cv_metrics
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
            
            cv_metrics = result.get('cv_metrics', {})
            summary_data.append({
                'Model': model_name,
                'Data_Type': data_type.upper(),
                'Optimal_Threshold': result.get('optimal_threshold', 0.5),
                'CV_AUC': result['cv_score'],
                'CV_AUC_Mean': cv_metrics.get('cv_auc_mean', result['cv_score']),
                'CV_AP_Mean': cv_metrics.get('cv_average_precision_mean', 0),
                'CV_F1_Mean': cv_metrics.get('cv_f1_mean', 0),
                'Val_AUC': result['val_metrics']['auc'],
                'Val_F1': result['val_metrics']['f1'],
                'Test_AUC': result['test_metrics']['auc'],
                'Test_Precision': result['test_metrics']['precision'],
                'Test_Recall': result['test_metrics']['recall'],
                'Test_F1': result['test_metrics']['f1'],
                'Test_Balanced_Acc': result['test_metrics'].get('balanced_accuracy', 0),
                'Test_Average_Precision': result['test_metrics'].get('average_precision', 0)
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