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
import matplotlib
matplotlib.use('Agg')  # 헤드리스 환경에서 사용
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
    classification_report, confusion_matrix, roc_curve,
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
    
    def __init__(self, config_path: str, data_path_override: str = None):
        """
        파이프라인 초기화
        
        Args:
            config_path: config YAML/JSON 파일 경로
            data_path_override: 데이터 경로 오버라이드 (전처리 결과 경로)
        """
        self.config_path = config_path
        self.data_path_override = data_path_override
        self.config = self._load_config()
        
        # 프로젝트 루트 디렉토리 설정
        self.project_root = Path(__file__).parent.parent.parent
        
        # 실행 정보 설정
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.config.get('experiment', {}).get('name', 'default_modeling_run')
        self.run_name = f"{experiment_name}_{self.timestamp}"
        
        # 출력 디렉토리 설정 (로거 설정 전에 해야 함)
        output_base_dir = self.config.get('output', {}).get('base_dir', 'outputs/modeling_runs')
        self.output_dir = self.project_root / output_base_dir / self.run_name
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
        
        # 로깅 레벨 설정 (안전하게)
        logging_config = self.config.get('logging', {})
        log_level = logging_config.get('level', 'INFO')
        try:
            logger.setLevel(getattr(logging, log_level))
        except AttributeError:
            logger.setLevel(logging.INFO)
        
        # 핸들러 제거 (중복 방지)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러
        if logging_config.get('save_to_file', True):
            log_dir = self.output_dir / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"modeling_{self.timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.propagate = False
        
        return logger
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """전처리된 데이터 로드"""
        self.logger.info("데이터 로드 시작")
        
        # 오버라이드된 경로가 있으면 사용, 없으면 config에서 가져오기
        if self.data_path_override:
            data_path = self.project_root / self.data_path_override
            self.logger.info(f"오버라이드된 데이터 경로 사용: {self.data_path_override}")
        else:
            default_path = self.config.get('data', {}).get('input_path', 'data/final')
            data_path = self.project_root / default_path
            self.logger.info(f"기본 데이터 경로 사용: {default_path}")
        
        # 파일 이름 설정 (기본값 제공)
        data_files = self.config.get('data', {}).get('files', {})
        file_names = {
            'X_train': data_files.get('X_train', 'X_train.csv'),
            'X_val': data_files.get('X_val', 'X_val.csv'),
            'X_test': data_files.get('X_test', 'X_test.csv'),
            'y_train': data_files.get('y_train', 'y_train.csv'),
            'y_val': data_files.get('y_val', 'y_val.csv'),
            'y_test': data_files.get('y_test', 'y_test.csv')
        }
        
        # 데이터 로드 (안전하게)
        try:
            self.data['normal'] = {
                'X_train': pd.read_csv(data_path / file_names['X_train']),
                'X_val': pd.read_csv(data_path / file_names['X_val']),
                'X_test': pd.read_csv(data_path / file_names['X_test']),
                'y_train': pd.read_csv(data_path / file_names['y_train']).iloc[:, 0],
                'y_val': pd.read_csv(data_path / file_names['y_val']).iloc[:, 0],
                'y_test': pd.read_csv(data_path / file_names['y_test']).iloc[:, 0]
            }
        except Exception as e:
            self.logger.error(f"데이터 로드 중 오류: {e}")
            raise
        
        # 활성화된 데이터 타입별로 복사 (동적 샘플링을 위해)
        sampling_config = self.config.get('sampling', {}).get('data_types', {})
        enabled_data_types = [dt for dt, config in sampling_config.items() if config.get('enabled', False)]
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
            data_type: 데이터 타입 ('normal', 'smote', 'undersampling', 'combined', 'ctgan')
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        if data_type == 'normal':
            return X, y
        
        # config가 None이거나 존재하지 않을 경우 처리
        try:
            config = self.config['sampling']['data_types'][data_type]
            if config is None:
                self.logger.warning(f"{data_type} 샘플링 설정이 None입니다. 원본 데이터를 반환합니다.")
                return X, y
        except (KeyError, TypeError) as e:
            self.logger.warning(f"{data_type} 샘플링 설정을 찾을 수 없습니다: {e}. 원본 데이터를 반환합니다.")
            return X, y
        
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
                        random_state=config.get('random_state', self.config.get('random_state', 42)),
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
                        random_state=config.get('random_state', self.config.get('random_state', 42)),
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
                    random_state=config.get('random_state', self.config.get('random_state', 42))
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
                    random_state=config.get('random_state', self.config.get('random_state', 42))
                )
            
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        
        elif data_type == 'ctgan':
            try:
                from ctgan import CTGAN
                from sklearn.model_selection import train_test_split
                import numpy as np
                
                # 원본 데이터 크기와 클래스 분포 확인
                n_minority = (y == 1).sum()
                n_majority = (y == 0).sum()
                total_samples = len(y)
                
                self.logger.info(f"CTGAN 시작 - 클래스 분포: 0={n_majority}, 1={n_minority}")
                
                # 소수 클래스가 너무 적으면 건너뛰기
                if n_minority < 10:
                    self.logger.warning(f"소수 클래스 샘플이 {n_minority}개로 너무 적어 CTGAN 건너뛰기")
                    return X, y
                
                # CTGAN 학습용 데이터 준비
                ctgan_data = X.copy()
                ctgan_data['target'] = y
                
                # 소수 클래스 데이터만 추출해서 CTGAN 학습
                minority_data = ctgan_data[ctgan_data['target'] == 1].copy()
                
                # CTGAN 모델 생성 및 학습
                ctgan_config = config.get('ctgan', {})
                ctgan = CTGAN(
                    epochs=ctgan_config.get('epochs', 300),
                    batch_size=min(ctgan_config.get('batch_size', 500), len(minority_data)),
                    generator_dim=ctgan_config.get('generator_dim', (256, 256)),
                    discriminator_dim=ctgan_config.get('discriminator_dim', (256, 256)),
                    verbose=False
                )
                
                ctgan.fit(minority_data, discrete_columns=['target'])
                
                # 필요한 synthetic 샘플 수 계산
                target_ratio = config.get('sampling_strategy', 0.5)
                if isinstance(target_ratio, dict):
                    target_count = target_ratio.get(1, n_majority)
                else:
                    target_count = int(n_majority * target_ratio)
                
                n_synthetic = max(0, target_count - n_minority)
                
                if n_synthetic > 0:
                    # Synthetic 샘플 생성
                    synthetic_samples = ctgan.sample(n_synthetic)
                    synthetic_samples['target'] = 1  # 모두 양성 클래스로 설정
                    
                    # 기존 데이터와 결합
                    combined_data = pd.concat([ctgan_data, synthetic_samples], ignore_index=True)
                    
                    X_resampled = combined_data.drop('target', axis=1)
                    y_resampled = combined_data['target']
                    
                    self.logger.info(f"CTGAN 완료 - {n_synthetic}개 synthetic 샘플 생성")
                    return X_resampled, y_resampled
                else:
                    self.logger.info("CTGAN - 추가 샘플링 불필요")
                    return X, y
                
            except ImportError:
                self.logger.warning("CTGAN 패키지가 설치되지 않음. 'pip install ctgan' 실행 후 재시도")
                return X, y
            except Exception as e:
                self.logger.warning(f"CTGAN 실패: {e}")
                return X, y
        
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
                        random_state=smote_config.get('random_state', self.config.get('random_state', 42)),
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
                        random_state=smote_config.get('random_state', self.config.get('random_state', 42)),
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
                        random_state=undersampling_config.get('random_state', self.config.get('random_state', 42))
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
                    random_state=undersampling_config.get('random_state', self.config.get('random_state', 42))
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
            # 첫 번째 호출에서만 로그 출력 (중복 방지)
            if not hasattr(self, '_scaling_disabled_logged'):
                self.logger.info(f"스케일링이 비활성화되어 있습니다")
                self._scaling_disabled_logged = True
            return X_train, X_val, X_test, {}
        
        # self.logger.info(f"스케일링 적용 시작 ({data_type.upper()})")  # 로그 제거
        
        scaling_config = self.config['scaling']
        scalers = {}
        total_available_columns = set(X_train.columns)
        total_processed_columns = set()
        
        # 각 스케일링 방법별로 컬럼 처리
        column_groups = scaling_config.get('column_groups', {})
        if column_groups is None:
            column_groups = {}
        
        for scaler_type, columns in column_groups.items():
            # columns가 None인 경우 건너뛰기
            if columns is None:
                # self.logger.warning(f"{scaler_type} 그룹의 컬럼 목록이 None입니다. 건너뜁니다.")
                continue
            
            # 실제 존재하는 컬럼과 존재하지 않는 컬럼 구분
            existing_columns = [col for col in columns if col in total_available_columns]
            missing_columns = [col for col in columns if col not in total_available_columns]
            
            # 존재하지 않는 컬럼 정보 출력 (경고가 아닌 정보 레벨) - 로그 제거
            # if missing_columns:
            #     self.logger.info(f"{scaler_type} 그룹 - 데이터에 없는 컬럼 ({len(missing_columns)}개): {missing_columns[:5]}{'...' if len(missing_columns) > 5 else ''}")
            
            # 존재하는 컬럼이 없으면 건너뛰기 - 로그 제거
            if not existing_columns:
                # self.logger.info(f"{scaler_type} 그룹 - 적용 가능한 컬럼이 없어 건너뜁니다")
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
                # 로그 변환은 별도의 apply_log_transform 함수에서 처리하므로 여기서는 건너뛰기
                if scaler_type.lower() == 'log':
                    # self.logger.info(f"로그 변환은 apply_log_transform에서 처리됩니다. {scaler_type} 그룹 건너뜀")
                    continue
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
                
                # self.logger.info(f"{scaler_type} 스케일링 적용 완료: {len(existing_columns)}개 컬럼 처리")  # 로그 제거
                
            except Exception as e:
                self.logger.error(f"{scaler_type} 스케일링 적용 중 오류 발생: {e}")
                continue
        
        # 전체 스케일링 결과 요약
        total_scaled_columns = len(total_processed_columns)
        unscaled_columns = total_available_columns - total_processed_columns
        
        # 스케일링 요약 로그 제거
        # self.logger.info(f"스케일링 완료 요약 ({data_type.upper()}):")
        # self.logger.info(f"  - 총 컬럼 수: {len(total_available_columns)}")
        # self.logger.info(f"  - 스케일링 적용: {total_scaled_columns}개 컬럼")
        # self.logger.info(f"  - 스케일링 미적용: {len(unscaled_columns)}개 컬럼")
        
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
    
    def apply_log_transform(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        로그 변환을 스케일링 전에 별도로 적용
        
        Args:
            X_train: 훈련 데이터
            X_val: 검증 데이터  
            X_test: 테스트 데이터
            data_type: 데이터 타입
            
        Returns:
            tuple: (X_train_transformed, X_val_transformed, X_test_transformed, transform_info)
        """
        if not self.config.get('scaling', {}).get('enabled', False):
            # 첫 번째 호출에서만 로그 출력 (중복 방지)  
            if not hasattr(self, '_log_transform_disabled_logged'):
                self.logger.info(f"스케일링이 비활성화되어 있어 로그 변환도 건너뜁니다")
                self._log_transform_disabled_logged = True
            return X_train, X_val, X_test, {}
        
        scaling_config = self.config['scaling']
        column_groups = scaling_config.get('column_groups', {})
        
        # 로그 변환할 컬럼 찾기
        log_columns = column_groups.get('log', [])
        if not log_columns:
            # self.logger.info(f"로그 변환할 컬럼이 없습니다 ({data_type.upper()})")
            return X_train, X_val, X_test, {}
        
        # 실제 존재하는 컬럼만 필터링
        total_available_columns = set(X_train.columns)
        existing_log_columns = [col for col in log_columns if col in total_available_columns]
        missing_log_columns = [col for col in log_columns if col not in total_available_columns]
        
        # if missing_log_columns:
            # self.logger.info(f"로그 변환 - 데이터에 없는 컬럼 ({len(missing_log_columns)}개): {missing_log_columns[:5]}{'...' if len(missing_log_columns) > 5 else ''}")
        
        if not existing_log_columns:
            # self.logger.info(f"로그 변환 - 적용 가능한 컬럼이 없습니다 ({data_type.upper()})")
            return X_train, X_val, X_test, {}
        
        # self.logger.info(f"로그 변환 적용 ({data_type.upper()}): {len(existing_log_columns)}개 컬럼")
        
        # 로그 변환 적용
        transform_info = {}
        for col in existing_log_columns:
            transform_info[col] = {}
            for df_name, df in [('train', X_train), ('val', X_val), ('test', X_test)]:
                # 음수 또는 0 값 확인
                min_val = df[col].min()
                
                if min_val <= 0:
                    # 음수나 0이 있는 경우 shift 적용 (최소값을 1로 만들기)
                    shift_value = 1 - min_val
                    df[col] = np.log1p(df[col] + shift_value)  # log1p = log(1+x)
                    transform_info[col][df_name] = f"shift({shift_value:.4f}) + log1p"
                else:
                    # 양수만 있는 경우 직접 로그 변환
                    df[col] = np.log1p(df[col])  # log1p는 수치적으로 더 안정적
                    transform_info[col][df_name] = "log1p"
        
        log_transform_summary = {
            'enabled': True,
            'applied_columns': existing_log_columns,
            'applied_count': len(existing_log_columns),
            'missing_columns': missing_log_columns,
            'missing_count': len(missing_log_columns),
            'transform_details': transform_info
        }
        
        return X_train, X_val, X_test, log_transform_summary

    def _apply_log_transform(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, columns: List[str], scaler_type: str):
        """
        로그 변환 적용 (기존 함수 - 하위 호환성 유지)
        
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
                    # self.logger.info(f"{scaler_type} - {col} ({df_name}): 최소값 {min_val:.4f}, shift {shift_value:.4f} 적용")  # 로그 제거
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
        from sklearn.pipeline import Pipeline
        
        config = self.config['feature_selection']['logistic_regression_cv']
        cv_folds = self.config.get('cv_folds', 3)  # 공통 config cv_folds 사용
        
        # Pipeline을 사용하여 CV 내부에서 스케일링 적용
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegressionCV(
                Cs=config['Cs'],
                cv=cv_folds,
                penalty=config['penalty'],
                solver=config['solver'],
                max_iter=config['max_iter'],
                scoring=config.get('scoring', 'roc_auc'),
                random_state=self.config.get('random_state', 42),
                n_jobs=self.config.get('performance', {}).get('n_jobs', 1)
            ))
        ])
        
        # Pipeline 훈련 (CV 내부에서 각 fold마다 스케일링 적용)
        pipeline.fit(X_train, y_train)
        
        # 훈련된 LogisticRegressionCV 모델 추출
        logistic_cv = pipeline.named_steps['logistic']
        
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
        cv_folds = self.config.get('cv_folds', 3)  # 공통 config cv_folds 사용
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Lasso CV
        lasso_cv = LassoCV(
            alphas=config['alphas'],
            cv=cv_folds,
            random_state=self.config.get('random_state', 42),
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
                random_state=self.config.get('random_state', 42),
                n_jobs=-1,
                **estimator_params
            )
            X_train_processed = X_train.values
        elif estimator_name == 'logistic_regression':
            # 로지스틱 회귀는 스케일링 필요
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train)
            estimator = LogisticRegression(
                random_state=self.config.get('random_state', 42),
                **estimator_params
            )
        elif estimator_name == 'xgboost':
            estimator = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=self.config.get('random_state', 42),
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
            random_state=config.get('random_state', self.config.get('random_state', 42)),
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
                random_state=self.config.get('random_state', 42),
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
                random_state=self.config.get('random_state', 42),
                **estimator_params
            )
        elif estimator_name == 'xgboost':
            estimator = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=self.config.get('random_state', 42),
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
                if hasattr(shap_values, 'values'):
                    shap_values_raw = shap_values.values
                else:
                    shap_values_raw = shap_values
            else:
                shap_values_raw = explainer.shap_values(X_sample)
            
            # SHAP 값 배열 처리
            if isinstance(shap_values_raw, list):
                # 이진 분류의 경우 양성 클래스 SHAP 값 사용
                shap_values_array = shap_values_raw[1]
            elif shap_values_raw.ndim == 3:
                # 3차원 배열의 경우 (n_samples, n_features, n_classes)
                # 이진 분류에서 양성 클래스(클래스 1) SHAP 값 사용
                shap_values_array = shap_values_raw[:, :, 1]
            else:
                shap_values_array = shap_values_raw
                
        except Exception as e:
            self.logger.warning(f"SHAP 값 계산 실패 ({e}), 기본 feature importance 사용")
            if hasattr(estimator, 'feature_importances_'):
                shap_values_array = estimator.feature_importances_.reshape(1, -1)
            else:
                # 로지스틱 회귀의 경우 계수의 절댓값 사용
                coef = estimator.coef_
                if coef.ndim == 2:
                    # 이진 분류의 경우 (1, n_features) 형태
                    shap_values_array = np.abs(coef)
                else:
                    # 1차원인 경우
                    shap_values_array = np.abs(coef).reshape(1, -1)
        
        # 특성별 평균 절댓값 SHAP 값 계산
        if shap_values_array.ndim == 2:
            mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)
        elif shap_values_array.ndim == 1:
            mean_abs_shap = np.abs(shap_values_array)
        else:
            # 예상치 못한 차원의 경우
            self.logger.warning(f"예상치 못한 SHAP 값 차원: {shap_values_array.shape}")
            mean_abs_shap = np.abs(shap_values_array).flatten()[:len(X_train.columns)]
        
        # 차원 확인 및 디버깅 로그
        self.logger.info(f"SHAP 값 배열 형태: {shap_values_array.shape}")
        self.logger.info(f"평균 절댓값 SHAP 형태: {mean_abs_shap.shape}")
        self.logger.info(f"특성 개수: {len(X_train.columns)}")
        
        # 차원이 맞지 않는 경우 처리
        if len(mean_abs_shap) != len(X_train.columns):
            self.logger.error(f"차원 불일치: SHAP 값 {len(mean_abs_shap)}개, 특성 {len(X_train.columns)}개")
            # 최소 길이로 맞춤
            min_len = min(len(mean_abs_shap), len(X_train.columns))
            mean_abs_shap = mean_abs_shap[:min_len]
            feature_names = X_train.columns[:min_len]
        else:
            feature_names = X_train.columns
        
        # 특성 선택
        threshold = config['threshold']
        
        if threshold == 'median':
            threshold_value = np.median(mean_abs_shap[mean_abs_shap > 0])
        elif threshold == 'mean':
            threshold_value = np.mean(mean_abs_shap[mean_abs_shap > 0])
        else:
            threshold_value = float(threshold)
        
        selected_mask = mean_abs_shap >= threshold_value
        selected_features = feature_names[selected_mask].tolist()
        
        # max_features 제한 적용
        max_features = config.get('max_features')
        if max_features and len(selected_features) > max_features:
            # SHAP 값 순으로 정렬하여 상위 max_features개만 선택
            feature_shap_pairs = list(zip(feature_names, mean_abs_shap))
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
            'mean_abs_shap_values': dict(zip(feature_names, mean_abs_shap)),
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
        cv_folds = self.config.get('cv_folds', 3)  # 공통 config cv_folds 사용
        
        # Config에서 class_weight 설정 가져오기
        class_weight = config.get('class_weight', None)
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
                'random_state': self.config.get('random_state', 42),
                'tol': 1e-4  # 수렴 기준 완화
            }
            
            if penalty == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', *config['l1_ratio_range'])
            
            model = LogisticRegression(**params)
            
            try:
                # config에서 validation 방법 결정
                validation_method = self.config.get('validation', {}).get('method', 'logistic_holdout')
                
                if validation_method == 'nested_cv':
                    # Nested CV 사용 - 단일 trial에서는 inner CV만 수행
                    scores = self._proper_cv_with_sampling(model, X_train, y_train, data_type, cv_folds=cv_folds, scoring=primary_metric)
                elif validation_method == 'k_fold':
                    # 기존 K-fold CV 사용
                    scores = self._proper_cv_with_sampling(model, X_train, y_train, data_type, cv_folds=cv_folds, scoring=primary_metric)
                else:  # logistic_holdout
                    # 로지스틱 홀드아웃 + 반복 샘플링 사용
                    holdout_config = self.config.get('validation', {}).get('logistic_holdout', {})
                    n_iterations = holdout_config.get('n_iterations', 10)
                    test_size = holdout_config.get('test_size', 0.2)
                    scores = self._logistic_holdout_repeated_sampling(model, X_train, y_train, data_type, n_iterations=n_iterations, test_size=test_size, scoring=primary_metric)
            
                return scores.mean()
                
            except (Warning, Exception) as e:
                # 모든 예외에 대해 낮은 점수 반환
                return 0.5
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config.get('random_state', 42)))
        
        # optuna 기본 로그만 억제 (trial finished 메시지 등)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # 콜백 함수로 진행상황 표시 (더 간결하게)
        def progress_callback(study, trial):
            # 진행상황을 더 간결하게 출력
            if trial.number == 0 or trial.number == config['n_trials'] // 2 or trial.number == config['n_trials'] - 1:
                best_value = study.best_value if study.best_value else 0
                self.logger.info(f"  Trial {trial.number + 1}/{config['n_trials']} - Current: {trial.value:.4f}, Best: {best_value:.4f}")
        
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
        
        # 최종 모델 훈련: 전체 훈련 데이터에 로그 변환 → 스케일링 → 샘플링 적용
        X_train_log, _, _, log_info = self.apply_log_transform(
            X_train.copy(),
            X_train.copy(),  # 더미 데이터
            X_train.copy(),  # 더미 데이터
            data_type
        )
        
        X_train_scaled, _, _, scalers = self.apply_scaling(
            X_train_log,
            X_train_log,  # 더미 데이터
            X_train_log,  # 더미 데이터
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
        cv_folds = self.config.get('cv_folds', 3)  # 공통 config cv_folds 사용
        
        # Config에서 class_weight 설정 가져오기
        class_weight = config.get('class_weight', None)
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
                'random_state': self.config.get('random_state', 42),
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            
            # config에서 validation 방법 결정
            validation_method = self.config.get('validation', {}).get('method', 'logistic_holdout')
            
            if validation_method == 'nested_cv':
                # Nested CV 사용 - 단일 trial에서는 inner CV만 수행
                scores = self._proper_cv_with_sampling(model, X_train, y_train, data_type, cv_folds=cv_folds, scoring=primary_metric)
            elif validation_method == 'k_fold':
                # 기존 K-fold CV 사용
                scores = self._proper_cv_with_sampling(model, X_train, y_train, data_type, cv_folds=cv_folds, scoring=primary_metric)
            else:  # logistic_holdout
                # 로지스틱 홀드아웃 + 반복 샘플링 사용
                holdout_config = self.config.get('validation', {}).get('logistic_holdout', {})
                n_iterations = holdout_config.get('n_iterations', 10)
                test_size = holdout_config.get('test_size', 0.2)
                scores = self._logistic_holdout_repeated_sampling(model, X_train, y_train, data_type, n_iterations=n_iterations, test_size=test_size, scoring=primary_metric)
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config.get('random_state', 42)))
        
        # 진행상황 콜백 함수 (더 간결하게)
        def progress_callback(study, trial):
            if trial.number == 0 or trial.number == config['n_trials'] // 2 or trial.number == config['n_trials'] - 1:
                best_value = study.best_value if study.best_value else 0
                self.logger.info(f"  Trial {trial.number + 1}/{config['n_trials']} - Current: {trial.value:.4f}, Best: {best_value:.4f}")
        
        study.optimize(objective, n_trials=config['n_trials'], callbacks=[progress_callback])
        
        # 최적 모델 훈련
        best_params = study.best_params
        best_params['class_weight'] = class_weight  # Config에서 가져온 클래스 가중치 적용
        model = RandomForestClassifier(**best_params)
        
        # 최종 모델 훈련: 전체 훈련 데이터에 로그 변환 → 스케일링 → 샘플링 적용
        X_train_log, _, _, log_info = self.apply_log_transform(
            X_train.copy(),
            X_train.copy(),  # 더미 데이터
            X_train.copy(),  # 더미 데이터
            data_type
        )
        
        X_train_scaled, _, _, scalers = self.apply_scaling(
            X_train_log,
            X_train_log,  # 더미 데이터
            X_train_log,  # 더미 데이터
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
        cv_folds = self.config.get('cv_folds', 3)  # 공통 config cv_folds 사용
        
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
                'random_state': self.config.get('random_state', 42),
                'n_jobs': -1,
                'verbosity': 0
            }
            
            model = xgb.XGBClassifier(**params)
            
            # config에서 validation 방법 결정
            validation_method = self.config.get('validation', {}).get('method', 'logistic_holdout')
            
            if validation_method == 'nested_cv':
                # Nested CV 사용 - 단일 trial에서는 inner CV만 수행
                scores = self._proper_cv_with_sampling(model, X_train, y_train, data_type, cv_folds=cv_folds, scoring=primary_metric)
            elif validation_method == 'k_fold':
                # 기존 K-fold CV 사용
                scores = self._proper_cv_with_sampling(model, X_train, y_train, data_type, cv_folds=cv_folds, scoring=primary_metric)
            else:  # logistic_holdout
                # 로지스틱 홀드아웃 + 반복 샘플링 사용
                holdout_config = self.config.get('validation', {}).get('logistic_holdout', {})
                n_iterations = holdout_config.get('n_iterations', 10)
                test_size = holdout_config.get('test_size', 0.2)
                scores = self._logistic_holdout_repeated_sampling(model, X_train, y_train, data_type, n_iterations=n_iterations, test_size=test_size, scoring=primary_metric)
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config.get('random_state', 42)))
        
        # 진행상황 콜백 함수 (더 간결하게)
        def progress_callback(study, trial):
            if trial.number == 0 or trial.number == config['n_trials'] // 2 or trial.number == config['n_trials'] - 1:
                best_value = study.best_value if study.best_value else 0
                self.logger.info(f"  Trial {trial.number + 1}/{config['n_trials']} - Current: {trial.value:.4f}, Best: {best_value:.4f}")
        
        study.optimize(objective, n_trials=config['n_trials'], callbacks=[progress_callback])
        
        # 최적 모델 훈련
        best_params = study.best_params
        best_params['scale_pos_weight'] = scale_pos_weight  # Config에서 계산된 클래스 가중치 적용
        model = xgb.XGBClassifier(**best_params)
        
        # 최종 모델 훈련: 전체 훈련 데이터에 로그 변환 → 스케일링 → 샘플링 적용
        X_train_log, _, _, log_info = self.apply_log_transform(
            X_train.copy(),
            X_train.copy(),  # 더미 데이터
            X_train.copy(),  # 더미 데이터
            data_type
        )
        
        X_train_scaled, _, _, scalers = self.apply_scaling(
            X_train_log,
            X_train_log,  # 더미 데이터
            X_train_log,  # 더미 데이터
            data_type
        )
        
        X_train_final, y_train_final = self.apply_sampling_strategy(
            X_train_scaled, y_train, data_type
        )
        
        # Version-agnostic XGBoost fitting with robust error handling (from baseline)
        try:
            import inspect
            # Check if early_stopping_rounds parameter is supported
            sig = inspect.signature(model.fit).parameters
            if "early_stopping_rounds" in sig:
                # XGBoost ≤ 1.7.x
                self.logger.info("Using XGBoost ≤ 1.7.x early stopping")
                eval_set = [(X_train_final[:int(0.8*len(X_train_final))], y_train_final[:int(0.8*len(y_train_final))]),
                           (X_train_final[int(0.8*len(X_train_final)):], y_train_final[int(0.8*len(y_train_final)):])]
                model.fit(
                    X_train_final, y_train_final,
                    eval_set=eval_set,
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                # XGBoost ≥ 2.0
                self.logger.info("Using XGBoost ≥ 2.0 callbacks")
                from xgboost.callback import EarlyStopping
                eval_set = [(X_train_final[:int(0.8*len(X_train_final))], y_train_final[:int(0.8*len(y_train_final))]),
                           (X_train_final[int(0.8*len(X_train_final)):], y_train_final[int(0.8*len(y_train_final)):])]
                model.fit(
                    X_train_final, y_train_final,
                    eval_set=eval_set,
                    callbacks=[EarlyStopping(rounds=50, save_best=True)],
                    verbose=False
                )
        except Exception as e:
            self.logger.warning(f"Early stopping failed, using fallback: {e}")
            # Fallback: train without early stopping
            model.fit(X_train_final, y_train_final)
        
        self.logger.info(f"최적 AUC: {study.best_value:.4f}")
        self.logger.info(f"최적 파라미터: {best_params}")
        
        return model, best_params, study.best_value
    
    def _nested_cv_with_sampling(self, model_class, param_space: dict, X: pd.DataFrame, y: pd.Series, data_type: str, outer_cv_folds: int = 5, inner_cv_folds: int = 3, n_trials: int = 50, scoring='roc_auc'):
        """
        Nested Cross Validation with proper sampling
        Outer CV: 모델 성능 평가
        Inner CV: 하이퍼파라미터 튜닝
        """
        outer_skf = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=self.config.get('random_state', 42))
        nested_scores = []
        
        self.logger.info(f"Nested CV 시작: Outer={outer_cv_folds} folds, Inner={inner_cv_folds} folds, Trials={n_trials}")
        
        for outer_fold, (train_idx, test_idx) in enumerate(outer_skf.split(X, y)):
            self.logger.info(f"Outer fold {outer_fold + 1}/{outer_cv_folds} 진행 중...")
            
            # Outer loop data split
            X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
            y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
            
            # Inner CV for hyperparameter optimization
            def objective(trial):
                # Generate hyperparameters based on param_space
                params = {}
                for param_name, param_config in param_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                
                # Create model with suggested parameters
                model = model_class(**params)
                
                # Inner CV with current train data
                scores = self._proper_cv_with_sampling(
                    model, X_train_outer, y_train_outer, data_type, 
                    cv_folds=inner_cv_folds, scoring=scoring
                )
                
                return scores.mean()
            
            # Inner optimization
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config.get('random_state', 42) + outer_fold))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            # Train best model on full outer train set
            best_params = study.best_params
            best_model = model_class(**best_params)
            
            # Apply preprocessing to outer train data
            try:
                X_train_log, X_test_log, _, log_info = self.apply_log_transform(
                    X_train_outer.copy(), X_test_outer.copy(), X_test_outer.copy(), data_type
                )
                
                X_train_scaled, X_test_scaled, _, scalers = self.apply_scaling(
                    X_train_log, X_test_log, X_test_log, data_type
                )
                
                X_train_final, y_train_final = self.apply_sampling_strategy(
                    X_train_scaled, y_train_outer, data_type
                )
                
                # Train on processed data
                best_model.fit(X_train_final, y_train_final)
                
                # Evaluate on outer test set (scaled but not sampled)
                y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
                
                if scoring == 'roc_auc':
                    outer_score = roc_auc_score(y_test_outer, y_pred_proba)
                elif scoring == 'average_precision':
                    outer_score = average_precision_score(y_test_outer, y_pred_proba)
                elif scoring == 'f1':
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    outer_score = f1_score(y_test_outer, y_pred, zero_division=0)
                else:
                    outer_score = roc_auc_score(y_test_outer, y_pred_proba)
                
                nested_scores.append(outer_score)
                self.logger.info(f"Outer fold {outer_fold + 1} score: {outer_score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Outer fold {outer_fold + 1} failed: {e}")
                nested_scores.append(0.5 if scoring == 'roc_auc' else 0.0)
        
        nested_scores_array = np.array(nested_scores)
        self.logger.info(f"Nested CV 완료: {nested_scores_array.mean():.4f} ± {nested_scores_array.std():.4f}")
        
        return nested_scores_array

    def _logistic_holdout_repeated_sampling(self, model, X: pd.DataFrame, y: pd.Series, data_type: str, n_iterations: int = 10, test_size: float = 0.2, scoring='roc_auc'):
        """
        로지스틱 홀드아웃 + 반복 샘플링 검증
        매번 다른 train/validation split으로 여러 번 반복하여 robust한 성능 추정
        """
        from sklearn.model_selection import train_test_split
        
        scores = []
        # 간결한 시작 로그
        # self.logger.info(f"  Holdout validation: {n_iterations} iterations")
        
        for iteration in range(n_iterations):
            try:
                # Random train/validation split with stratification
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    stratify=y, 
                    random_state=self.config.get('random_state', 42) + iteration
                )
                
                # Apply preprocessing pipeline to current split
                X_train_log, X_val_log, _, log_info = self.apply_log_transform(
                    X_train_split.copy(), X_val_split.copy(), X_val_split.copy(), data_type
                )
                
                X_train_scaled, X_val_scaled, _, scalers = self.apply_scaling(
                    X_train_log, X_val_log, X_val_log, data_type
                )
                
                # Apply sampling only to training data
                X_train_resampled, y_train_resampled = self.apply_sampling_strategy(
                    X_train_scaled, y_train_split, data_type
                )
                
                # Train model on resampled training data
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train_resampled, y_train_resampled)
                
                # Evaluate on validation set (scaled but not resampled)
                y_pred_proba = model_copy.predict_proba(X_val_scaled)[:, 1]
                
                # Calculate score based on specified metric
                if scoring == 'roc_auc':
                    score = roc_auc_score(y_val_split, y_pred_proba)
                elif scoring == 'average_precision':
                    score = average_precision_score(y_val_split, y_pred_proba)
                elif scoring == 'f1':
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    score = f1_score(y_val_split, y_pred, zero_division=0)
                else:
                    score = roc_auc_score(y_val_split, y_pred_proba)
                
                scores.append(score)
                
                # # 진행상황을 더 간결하게 출력 (처음, 중간, 마지막만)
                # if iteration == 0 or iteration == n_iterations // 2 or iteration == n_iterations - 1:
                #     self.logger.info(f"  Iteration {iteration + 1}/{n_iterations}: {score:.4f}")
                    
            except Exception as e:
                # self.logger.warning(f"Iteration {iteration + 1} failed: {e}")
                # Add default score for failed iterations
                default_score = 0.5 if scoring == 'roc_auc' else 0.0
                scores.append(default_score)
        
        scores_array = np.array(scores)
        # self.logger.info(f"  Holdout complete: {scores_array.mean():.4f} ± {scores_array.std():.4f}")
        
        return scores_array

    def _proper_cv_with_sampling(self, model, X: pd.DataFrame, y: pd.Series, data_type: str, cv_folds: int = None, scoring='roc_auc'):
        """
        샘플링 Data Leakage를 방지하는 올바른 Cross Validation
        각 CV fold마다 스케일링 → 샘플링을 순서대로 적용
        """
        if cv_folds is None:
            cv_folds = self.config.get('cv_folds', 3)  # 공통 config cv_folds 사용
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.get('random_state', 42))
        scores = []
                
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # 각 fold마다 별도로 분할
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                # 1단계: 로그 변환 적용
                log_result = self.apply_log_transform(
                    X_fold_train.copy(),
                    X_fold_val.copy(), 
                    X_fold_val.copy(),  # 더미 테스트 데이터 (사용 안 함)
                    data_type
                )
                
                if log_result is None or len(log_result) != 4:
                    raise ValueError("apply_log_transform 함수가 None 또는 잘못된 형태를 반환했습니다")
                
                X_fold_train_log, X_fold_val_log, _, log_info = log_result
                
                # 2단계: 스케일링 적용 (로그 변환된 데이터에)
                scaling_result = self.apply_scaling(
                    X_fold_train_log,
                    X_fold_val_log, 
                    X_fold_val_log,  # 더미 테스트 데이터 (사용 안 함)
                    data_type
                )
                
                # scaling 결과 None 체크
                if scaling_result is None or len(scaling_result) != 4:
                    raise ValueError("apply_scaling 함수가 None 또는 잘못된 형태를 반환했습니다")
                
                X_fold_train_scaled, X_fold_val_scaled, _, scalers = scaling_result
                
                # 3단계: 샘플링 적용 (스케일링된 훈련 데이터에만)  
                sampling_result = self.apply_sampling_strategy(
                    X_fold_train_scaled, y_fold_train, data_type
                )
                
                # sampling 결과 None 체크
                if sampling_result is None or len(sampling_result) != 2:
                    raise ValueError("apply_sampling_strategy 함수가 None 또는 잘못된 형태를 반환했습니다")
                
                X_fold_train_resampled, y_fold_train_resampled = sampling_result
                
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
                
                try:
                    # 처리 실패 시 원본 데이터로 훈련 (최소한의 백업)
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
                    self.logger.info(f"Fold {fold+1} 백업 처리 완료: {score:.4f}")
                    
                except Exception as e2:
                    # 완전히 실패한 경우 기본값 사용
                    self.logger.error(f"Fold {fold+1} 완전 실패: {e2}")
                    default_score = 0.5 if scoring == 'roc_auc' else 0.0
                    scores.append(default_score)
        
        if not scores:
            self.logger.error(f"모든 CV fold 실패, 기본값 반환")
            default_score = 0.5 if scoring == 'roc_auc' else 0.0
            scores = [default_score] * cv_folds
        
        scores_array = np.array(scores)        
        return scores_array
    
    def _proper_cv_with_all_metrics(self, model, X: pd.DataFrame, y: pd.Series, data_type: str, cv_folds: int = None) -> Dict[str, float]:
        """
        최적화된 CV: 한 번의 CV로 모든 메트릭을 동시에 계산
        시간을 3배 단축 (15 fold → 5 fold)
        """
        if cv_folds is None:
            cv_folds = self.config.get('cv_folds', 3)  # 공통 config cv_folds 사용
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.get('random_state', 42))
        
        # 각 메트릭별 점수 저장
        auc_scores = []
        ap_scores = []
        f1_scores = []
                
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):            
            # 각 fold마다 별도로 분할
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                # 1단계: 로그 변환 적용
                log_result = self.apply_log_transform(
                    X_fold_train.copy(),
                    X_fold_val.copy(), 
                    X_fold_val.copy(),  # 더미 테스트 데이터
                    data_type
                )
                
                if log_result is None or len(log_result) != 4:
                    raise ValueError("apply_log_transform 함수가 None 또는 잘못된 형태를 반환했습니다")
                
                X_fold_train_log, X_fold_val_log, _, log_info = log_result
                
                # 2단계: 스케일링 적용 (로그 변환된 데이터에)
                scaling_result = self.apply_scaling(
                    X_fold_train_log,
                    X_fold_val_log, 
                    X_fold_val_log,  # 더미 테스트 데이터
                    data_type
                )
                
                # scaling 결과 None 체크
                if scaling_result is None or len(scaling_result) != 4:
                    raise ValueError("apply_scaling 함수가 None 또는 잘못된 형태를 반환했습니다")
                
                X_fold_train_scaled, X_fold_val_scaled, _, scalers = scaling_result
                
                # 3단계: 샘플링 적용 (훈련 데이터에만)
                sampling_result = self.apply_sampling_strategy(
                    X_fold_train_scaled, y_fold_train, data_type
                )
                
                # sampling 결과 None 체크
                if sampling_result is None or len(sampling_result) != 2:
                    raise ValueError("apply_sampling_strategy 함수가 None 또는 잘못된 형태를 반환했습니다")
                
                X_fold_train_resampled, y_fold_train_resampled = sampling_result
                
                # 3단계: 모델 훈련
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_fold_train_resampled, y_fold_train_resampled)
                
                # 4단계: 예측 (한 번만 수행)
                y_pred_proba = model_copy.predict_proba(X_fold_val_scaled)[:, 1]
                y_pred_binary = (y_pred_proba >= 0.5).astype(int)
                
                # 5단계: 모든 메트릭 동시 계산
                auc_score = roc_auc_score(y_fold_val, y_pred_proba)
                ap_score = average_precision_score(y_fold_val, y_pred_proba)
                f1_score_val = f1_score(y_fold_val, y_pred_binary, zero_division=0)
                
                auc_scores.append(auc_score)
                ap_scores.append(ap_score)
                f1_scores.append(f1_score_val)
                
            except Exception as e:
                self.logger.warning(f"Fold {fold+1} {data_type.upper()} 처리 실패: {e}")
                
                try:
                    # 백업 처리: 원본 데이터로 훈련
                    model_copy = model.__class__(**model.get_params())
                    model_copy.fit(X_fold_train, y_fold_train)
                    y_pred_proba = model_copy.predict_proba(X_fold_val)[:, 1]
                    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
                    
                    auc_score = roc_auc_score(y_fold_val, y_pred_proba)
                    ap_score = average_precision_score(y_fold_val, y_pred_proba)
                    f1_score_val = f1_score(y_fold_val, y_pred_binary, zero_division=0)
                    
                    auc_scores.append(auc_score)
                    ap_scores.append(ap_score)
                    f1_scores.append(f1_score_val)
                    
                    self.logger.info(f"Fold {fold+1} 백업 처리 완료: AUC={auc_score:.3f}, AP={ap_score:.3f}, F1={f1_score_val:.3f}")
                    
                except Exception as e2:
                    # 완전 실패 시 기본값
                    self.logger.error(f"Fold {fold+1} 완전 실패: {e2}")
                    auc_scores.append(0.5)
                    ap_scores.append(0.0)
                    f1_scores.append(0.0)
        
        # 최종 메트릭 계산
        if not auc_scores:
            self.logger.error(f"모든 CV fold 실패, 기본값 반환")
            return {
                'cv_auc_mean': 0.5, 'cv_auc_std': 0.0,
                'cv_average_precision_mean': 0.0, 'cv_average_precision_std': 0.0,
                'cv_f1_mean': 0.0, 'cv_f1_std': 0.0
            }
        
        # 결과 정리
        auc_array = np.array(auc_scores)
        ap_array = np.array(ap_scores)
        f1_array = np.array(f1_scores)
        
        metrics = {
            'cv_auc_mean': float(auc_array.mean()),
            'cv_auc_std': float(auc_array.std()),
            'cv_average_precision_mean': float(ap_array.mean()),
            'cv_average_precision_std': float(ap_array.std()),
            'cv_f1_mean': float(f1_array.mean()),
            'cv_f1_std': float(f1_array.std())
        }
        
        self.logger.info(f"최적화된 CV 완료: AUC={metrics['cv_auc_mean']:.4f}(±{metrics['cv_auc_std']:.4f}), "
                        f"AP={metrics['cv_average_precision_mean']:.4f}(±{metrics['cv_average_precision_std']:.4f}), "
                        f"F1={metrics['cv_f1_mean']:.4f}(±{metrics['cv_f1_std']:.4f})")
        
        return metrics
    
    def _calculate_cv_metrics(self, model, X_train: pd.DataFrame, y_train: pd.Series, data_type: str) -> Dict[str, float]:
        """
        다양한 CV 메트릭 계산 (데이터 누수 방지) - 최적화된 버전
        한 번의 CV로 모든 메트릭을 동시에 계산
        """
        self.logger.info(f"CV 메트릭 계산 시작 ({data_type.upper()})")
        
        try:
            # 한 번의 CV로 모든 메트릭 계산
            all_metrics = self._proper_cv_with_all_metrics(model, X_train, y_train, data_type, cv_folds=5)
            
            self.logger.info(f"CV 메트릭 계산 완료 - AUC: {all_metrics['cv_auc_mean']:.4f}, AP: {all_metrics['cv_average_precision_mean']:.4f}, F1: {all_metrics['cv_f1_mean']:.4f}")
            
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"CV 메트릭 계산 실패: {e}")
            # 기본값 반환
            return {
                'cv_auc_mean': 0.5,
                'cv_auc_std': 0.0,
                'cv_average_precision_mean': 0.0,
                'cv_average_precision_std': 0.0,
                'cv_f1_mean': 0.0,
                'cv_f1_std': 0.0
            }
    

    
    def evaluate_model(self, model_key: str):
        """
        단일 모델의 성능을 평가하고 결과를 저장

        - Validation set에서 최적의 threshold를 찾음
        - Test set에 최적 threshold를 적용하여 최종 성능 평가
        """
        self.logger.info(f"모델 평가 시작: {model_key}")
        data_type, model_name = model_key.split('__')
        
        # 앙상블 모델의 경우 별도 처리
        if model_name == 'ensemble':
            self.logger.info(f"앙상블 모델은 이미 평가됨: {model_key}")
            return
        
        # 모델 및 데이터 로드 (매번 원본에서 새로 복사하여 상태 오염 방지)
        X_train_orig, y_train_orig = self.data[data_type]['X_train'].copy(), self.data[data_type]['y_train'].copy()
        X_val_orig, y_val_orig = self.data[data_type]['X_val'].copy(), self.data[data_type]['y_val'].copy()
        X_test_orig, y_test_orig = self.data[data_type]['X_test'].copy(), self.data[data_type]['y_test'].copy()
        model = self.models[model_key]

        self.model_results[model_key] = {
            'model_name': model_name,
            'data_type': data_type
        }

        # --- CV 평가 (Train set) ---
        cv_scores = self._calculate_cv_metrics(model, X_train_orig.copy(), y_train_orig.copy(), data_type)
        self.model_results[model_key]['cv_scores'] = cv_scores
        # 주요 메트릭을 cv_score로 저장 (앙상블에서 사용)
        self.model_results[model_key]['cv_score'] = cv_scores.get('cv_f1_mean', 0)
        self.logger.info(f"[{model_key}] Cross-Validation (Train): AUC={cv_scores.get('cv_auc_mean', 0):.4f}, F1={cv_scores.get('cv_f1_mean', 0):.4f}")
        
        # === 평가를 위한 전처리 적용 ===
        # 훈련 시와 동일한 전처리 순서: 로그 변환 → 스케일링
        try:
            # 1단계: 로그 변환 적용
            X_train_log, X_val_log, X_test_log, log_info = self.apply_log_transform(
                X_train_orig, X_val_orig, X_test_orig, data_type
            )
            
            # 2단계: 스케일링 적용 (로그 변환된 데이터에)
            X_train_scaled, X_val_scaled, X_test_scaled, scalers = self.apply_scaling(
                X_train_log, X_val_log, X_test_log, data_type
            )
            
            self.logger.info(f"[{model_key}] 평가용 전처리 완료")
            
        except Exception as e:
            self.logger.error(f"[{model_key}] 평가용 전처리 실패: {e}")
            # 전처리 실패 시 원본 데이터 사용 (백업)
            X_val_scaled, X_test_scaled = X_val_orig, X_test_orig

        # --- Validation 세트 평가 (전처리된 데이터 사용) ---
        y_proba_val = model.predict_proba(X_val_scaled)[:, 1]

        # Validation 세트에서 최적 Threshold 찾기 (F1 score 기준)
        f1_optimal_threshold, optimal_f1_score = self.find_optimal_threshold(model, X_val_scaled, y_val_orig)
        optimal_thresholds = {
            'f1': {
                'threshold': f1_optimal_threshold,
                'score': optimal_f1_score
            }
        }
        self.model_results[model_key]['optimal_thresholds'] = optimal_thresholds
        
        # Validation 세트 평가 (최적 Threshold 적용)
        y_pred_val_optimal = (y_proba_val >= f1_optimal_threshold).astype(int)
        
        val_metrics_optimal = {
            'roc_auc': roc_auc_score(y_val_orig, y_proba_val),
            'f1': f1_score(y_val_orig, y_pred_val_optimal, zero_division=0),
            'precision': precision_score(y_val_orig, y_pred_val_optimal, zero_division=0),
            'recall': recall_score(y_val_orig, y_pred_val_optimal, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_val_orig, y_pred_val_optimal),
            'threshold': f1_optimal_threshold
        }
        self.model_results[model_key]['val_metrics_optimal'] = val_metrics_optimal
        self.logger.info(f"[{model_key}] Validation (Optimal Threshold): F1={val_metrics_optimal['f1']:.4f}, Recall={val_metrics_optimal['recall']:.4f}")

        # --- 테스트 세트 평가 (전처리된 데이터 사용) ---
        y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
        
        # 최적 임계값 적용
        y_pred_test_optimal = (y_proba_test >= f1_optimal_threshold).astype(int)

        # 기본 임계값(0.5) 적용
        y_pred_test_default = (y_proba_test >= 0.5).astype(int)

        test_metrics = {
            'roc_auc': roc_auc_score(y_test_orig, y_proba_test),
            'optimal_threshold': f1_optimal_threshold,
            'average_precision': average_precision_score(y_test_orig, y_proba_test),
            
            # 최적 임계값 기준 성능
            'f1_optimal': f1_score(y_test_orig, y_pred_test_optimal, zero_division=0),
            'precision_optimal': precision_score(y_test_orig, y_pred_test_optimal, zero_division=0),
            'recall_optimal': recall_score(y_test_orig, y_pred_test_optimal, zero_division=0),
            'balanced_accuracy_optimal': balanced_accuracy_score(y_test_orig, y_pred_test_optimal),
            
            # 기본 임계값 기준 성능
            'f1_default': f1_score(y_test_orig, y_pred_test_default, zero_division=0),
            'precision_default': precision_score(y_test_orig, y_pred_test_default, zero_division=0),
            'recall_default': recall_score(y_test_orig, y_pred_test_default, zero_division=0),
        }
        self.model_results[model_key]['test_metrics'] = test_metrics
        self.logger.info(f"[{model_key}] Test (Optimal Threshold={f1_optimal_threshold:.4f}): F1={test_metrics['f1_optimal']:.4f}, Recall={test_metrics['recall_optimal']:.4f}")
        self.logger.info(f"[{model_key}] Test (Default Threshold=0.5): F1={test_metrics['f1_default']:.4f}, Recall={test_metrics['recall_default']:.4f}")

        # 혼동 행렬 시각화 (최적 Threshold 기준)
        cm = confusion_matrix(y_test_orig, y_pred_test_optimal)
        self.model_results[model_key]['confusion_matrix_test'] = cm.tolist()
        
        # 예측 결과 저장 (proba)
        self.model_results[model_key]['predictions'] = {
            'y_proba_val': y_proba_val.tolist(),
            'y_proba_test': y_proba_test.tolist()
        }

        self.logger.info(f"모델 평가 완료: {model_key}")
    
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
                model_key = f"{data_type}__{model_name}"
                
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
            # 안전한 키 접근
            model_name = result.get('model_name', 'unknown')
            data_type = result.get('data_type', 'unknown')
            
            cv_metrics = result.get('cv_metrics', {})
            val_metrics = result.get('val_metrics_optimal', {})
            test_metrics = result.get('test_metrics', {})
            
            # 앙상블과 개별 모델의 저장 구조가 다르므로 분기 처리
            if model_name == 'ensemble':
                optimal_threshold = result.get('optimal_threshold', 0.5)
            else:
                optimal_threshold = test_metrics.get('optimal_threshold', 0.5)

            summary_data.append({
                'Model': model_name,
                'Data_Type': data_type.upper() if data_type != 'unknown' else 'UNKNOWN',
                'Optimal_Threshold': optimal_threshold,
                'CV_AUC': result.get('cv_score', 0),
                'CV_AUC_Mean': cv_metrics.get('cv_auc_mean', result.get('cv_score', 0)),
                'CV_AP_Mean': cv_metrics.get('cv_average_precision_mean', 0),
                'CV_F1_Mean': cv_metrics.get('cv_f1_mean', 0),
                'Val_AUC': val_metrics.get('roc_auc', 0),
                'Val_F1': val_metrics.get('f1', 0),
                'Test_AUC': test_metrics.get('roc_auc', 0),
                'Test_Precision': test_metrics.get('precision_optimal', test_metrics.get('precision_default', 0)),
                'Test_Recall': test_metrics.get('recall_optimal', test_metrics.get('recall_default', 0)),
                'Test_F1': test_metrics.get('f1_optimal', test_metrics.get('f1_default', 0)),
                'Test_Balanced_Acc': test_metrics.get('balanced_accuracy_optimal', 0),
                'Test_Average_Precision': test_metrics.get('average_precision', 0)
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
        """시각화 생성 - 개별 모델 시각화 + 통합 시각화"""
        self.logger.info("시각화 생성 시작")
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 개별 모델 시각화를 위한 하위 디렉토리 생성
        individual_dir = viz_dir / 'individual_models'
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        combined_dir = viz_dir / 'combined'
        combined_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 개별 모델 시각화 생성
            self.logger.info("개별 모델 시각화 생성 중...")
            for model_key, result in self.model_results.items():
                try:
                    self._create_individual_model_visualization(model_key, result, individual_dir)
                except Exception as e:
                    self.logger.error(f"{model_key} 개별 시각화 실패: {e}")
            
            # 2. 통합 시각화 생성 (기존 기능)
            self.logger.info("통합 시각화 생성 중...")
            
            # ROC 곡선 비교
            try:
                self._plot_roc_curves(combined_dir)
            except Exception as e:
                self.logger.error(f"ROC 곡선 시각화 실패: {e}")
            
            # Precision-Recall 곡선 비교
            try:
                self._plot_pr_curves(combined_dir)
            except Exception as e:
                self.logger.error(f"PR 곡선 시각화 실패: {e}")
            
            # 성능 비교 차트
            try:
                self._plot_performance_comparison(combined_dir)
            except Exception as e:
                self.logger.error(f"성능 비교 차트 시각화 실패: {e}")
            
            # 특성 중요도 (tree 기반 모델)
            try:
                self._plot_feature_importance(combined_dir)
            except Exception as e:
                self.logger.error(f"특성 중요도 시각화 실패: {e}")
            
            # 임계값 분석
            try:
                self._plot_threshold_analysis(combined_dir)
            except Exception as e:
                self.logger.error(f"임계값 분석 시각화 실패: {e}")
            
            # Train vs Test 성능 비교 (Overfitting 분석)
            try:
                self._plot_train_vs_test_comparison(combined_dir)
            except Exception as e:
                self.logger.error(f"Train vs Test 비교 시각화 실패: {e}")
            
            # 앙상블 가중치 시각화 (모델 성능 기반)
            try:
                self._plot_ensemble_weights(combined_dir)
            except Exception as e:
                self.logger.error(f"앙상블 가중치 시각화 실패: {e}")
            
            # 앙상블 모델을 포함한 성능 비교 차트
            try:
                self._plot_ensemble_performance_comparison(combined_dir)
            except Exception as e:
                self.logger.error(f"앙상블 성능 비교 차트 생성 실패: {e}")
            
            # 앙상블 모델이 있다면 추가 리포트 생성
            if 'ensemble_model' in self.models:
                try:
                    ensemble_pipeline = self.models['ensemble_model']
                    ensemble_pipeline.create_ensemble_report(combined_dir)
                except Exception as e:
                    self.logger.error(f"앙상블 시각화 리포트 생성 중 오류: {e}")
            
            self.logger.info("시각화 생성 완료")
            
        except Exception as e:
            self.logger.error(f"시각화 생성 중 전반적 오류: {e}")
    
    def _create_individual_model_visualization(self, model_key: str, result: dict, individual_dir: Path):
        """개별 모델 시각화 생성"""
        model_name = result['model_name']
        data_type = result['data_type']
        
        # 모델별 하위 디렉토리 생성
        model_dir = individual_dir / f"{model_name}_{data_type}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"개별 시각화 생성: {model_name}_{data_type}")
        
        # 1. 개별 ROC 곡선
        try:
            self._plot_individual_roc_curve(model_key, result, model_dir)
        except Exception as e:
            self.logger.error(f"{model_key} ROC 곡선 생성 실패: {e}")
        
        # 2. 개별 PR 곡선
        try:
            self._plot_individual_pr_curve(model_key, result, model_dir)
        except Exception as e:
            self.logger.error(f"{model_key} PR 곡선 생성 실패: {e}")
        
        # 3. 개별 성능 메트릭 차트
        try:
            self._plot_individual_performance_metrics(model_key, result, model_dir)
        except Exception as e:
            self.logger.error(f"{model_key} 성능 메트릭 차트 생성 실패: {e}")
        
        # 4. 개별 혼동행렬
        try:
            self._plot_individual_confusion_matrix(model_key, result, model_dir)
        except Exception as e:
            self.logger.error(f"{model_key} 혼동행렬 생성 실패: {e}")
        
        # 5. 개별 임계값 분석 (해당 모델만)
        try:
            self._plot_individual_threshold_analysis(model_key, result, model_dir)
        except Exception as e:
            self.logger.error(f"{model_key} 임계값 분석 생성 실패: {e}")
    
    def _plot_individual_roc_curve(self, model_key: str, result: dict, model_dir: Path):
        """개별 모델의 ROC 곡선"""
        plt.figure(figsize=(8, 6))
        
        model_name = result['model_name']
        data_type = result['data_type']
        
        # 예측 데이터 가져오기
        y_test, y_pred_proba = self._get_model_predictions(model_key, result)
        
        if y_test is None or y_pred_proba is None:
            self.logger.warning(f"{model_key}: 예측 데이터를 가져올 수 없음")
            return
        
        # ROC 곡선 계산
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # 플롯
        plt.plot(fpr, tpr, color='blue', lw=3, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve: {model_name.upper()} ({data_type.upper()})', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 저장
        save_path = model_dir / f'roc_curve_{model_name}_{data_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"개별 ROC 곡선 저장: {save_path}")
    
    def _plot_individual_pr_curve(self, model_key: str, result: dict, model_dir: Path):
        """개별 모델의 Precision-Recall 곡선"""
        plt.figure(figsize=(8, 6))
        
        model_name = result['model_name']
        data_type = result['data_type']
        
        # 예측 데이터 가져오기
        y_test, y_pred_proba = self._get_model_predictions(model_key, result)
        
        if y_test is None or y_pred_proba is None:
            return
        
        # PR 곡선 계산
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ap_score = average_precision_score(y_test, y_pred_proba)
        
        # 기준선 (양성 클래스 비율)
        baseline = np.mean(y_test)
        
        # 플롯
        plt.plot(recall, precision, color='red', lw=3, label=f'PR Curve (AP = {ap_score:.3f})')
        plt.axhline(y=baseline, color='k', linestyle='--', lw=2, label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve: {model_name.upper()} ({data_type.upper()})', fontsize=14, fontweight='bold')
        plt.legend(loc="upper right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 저장
        save_path = model_dir / f'pr_curve_{model_name}_{data_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"개별 PR 곡선 저장: {save_path}")
    
    def _plot_individual_performance_metrics(self, model_key: str, result: dict, model_dir: Path):
        """개별 모델의 성능 메트릭 바차트"""
        plt.figure(figsize=(10, 6))
        
        model_name = result['model_name']
        data_type = result['data_type']
        
        # 성능 메트릭 가져오기
        metrics = result.get('test_metrics', {})
        
        # 주요 메트릭 선택
        metric_names = ['roc_auc', 'f1_optimal', 'precision_optimal', 'recall_optimal', 'balanced_accuracy_optimal']
        metric_values = []
        display_names = []
        
        for metric in metric_names:
            if metric in metrics:
                metric_values.append(metrics[metric])
                display_names.append(metric.replace('_optimal', '').replace('_', ' ').title())
        
        if not metric_values:
            self.logger.warning(f"{model_key}: 성능 메트릭을 찾을 수 없음")
            return
        
        # 바차트 생성
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = plt.bar(display_names, metric_values, color=colors[:len(metric_values)])
        
        # 값 표시
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, 1.1)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'Performance Metrics: {model_name.upper()} ({data_type.upper()})', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # 저장
        save_path = model_dir / f'performance_metrics_{model_name}_{data_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"개별 성능 메트릭 저장: {save_path}")
    
    def _plot_individual_confusion_matrix(self, model_key: str, result: dict, model_dir: Path):
        """개별 모델의 혼동행렬"""
        plt.figure(figsize=(8, 6))
        
        model_name = result['model_name']
        data_type = result['data_type']
        
        # 예측 데이터 가져오기
        y_test, y_pred_proba = self._get_model_predictions(model_key, result)
        
        if y_test is None or y_pred_proba is None:
            return
        
        # 최적 임계값 사용 (저장된 것이 있다면)
        optimal_threshold = result.get('optimal_threshold', 0.5)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # 혼동행렬 계산
        cm = confusion_matrix(y_test, y_pred)
        
        # 히트맵 생성
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
                    xticklabels=['Normal', 'Default'], yticklabels=['Normal', 'Default'],
                    cbar_kws={'label': 'Count'})
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix: {model_name.upper()} ({data_type.upper()})\nThreshold: {optimal_threshold:.3f}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 저장
        save_path = model_dir / f'confusion_matrix_{model_name}_{data_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"개별 혼동행렬 저장: {save_path}")
    
    def _plot_individual_threshold_analysis(self, model_key: str, result: dict, model_dir: Path):
        """개별 모델의 임계값 분석"""
        plt.figure(figsize=(10, 6))
        
        model_name = result['model_name']
        data_type = result['data_type']
        
        # 예측 데이터 가져오기
        y_test, y_pred_proba = self._get_model_predictions(model_key, result)
        
        if y_test is None or y_pred_proba is None:
            return
        
        # 다양한 임계값에서 성능 계산
        thresholds = np.arange(0.1, 0.9, 0.02)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if len(np.unique(y_pred)) > 1:
                precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
                recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
                f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
            else:
                precision_scores.append(0)
                recall_scores.append(0)
                f1_scores.append(0)
        
        # 플롯
        plt.plot(thresholds, precision_scores, 'b-', label='Precision', linewidth=2)
        plt.plot(thresholds, recall_scores, 'r-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, 'g-', label='F1-Score', linewidth=2)
        
        # 최적 임계값 표시
        optimal_threshold = result.get('optimal_threshold', 0.5)
        plt.axvline(x=optimal_threshold, color='orange', linestyle='--', linewidth=2, 
                   label=f'Optimal Threshold: {optimal_threshold:.3f}')
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'Threshold Analysis: {model_name.upper()} ({data_type.upper()})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(0.1, 0.9)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # 저장
        save_path = model_dir / f'threshold_analysis_{model_name}_{data_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"개별 임계값 분석 저장: {save_path}")
    
    def _get_model_predictions(self, model_key: str, result: dict):
        """모델의 예측 결과 가져오기"""
        model_name = result['model_name']
        data_type = result['data_type']
        
        # 앙상블 모델 처리
        if model_name == 'ensemble':
            X_test = self.data['normal']['X_test']
            y_test = self.data['normal']['y_test']
            model = self.models[model_key]
            y_pred_proba = model.predict_proba(X_test)
            return y_test, y_pred_proba
        else:
            # 일반 모델의 경우 저장된 예측 결과 사용
            y_test = self.data[data_type]['y_test']
            predictions = result.get('predictions', {})
            y_pred_proba = predictions.get('y_proba_test', [])
            
            if not y_pred_proba:
                self.logger.warning(f"{model_key}: 저장된 예측 결과 없음")
                return None, None
            
            return y_test, np.array(y_pred_proba)
    
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
                # 앙상블은 전처리 내장하므로 원본 데이터 사용
                y_pred_proba = model.predict_proba(X_test)  # 앙상블은 이미 확률값 반환
            else:
                # 일반 모델의 경우 evaluate_model에서 저장된 예측 결과 사용
                y_test = self.data[data_type]['y_test']
                predictions = result.get('predictions', {})
                y_pred_proba = predictions.get('y_proba_test', [])
                
                # 저장된 예측이 없는 경우에만 직접 예측 (백업)
                if not y_pred_proba:
                    self.logger.warning(f"{model_key}: 저장된 예측 결과 없음, 직접 예측 수행")
                    # 전처리 적용해서 예측
                    try:
                        X_test = self.data[data_type]['X_test']
                        X_train = self.data[data_type]['X_train']
                        X_val = self.data[data_type]['X_val']
                        
                        # 전처리 적용
                        X_train_log, X_val_log, X_test_log, _ = self.apply_log_transform(
                            X_train.copy(), X_val.copy(), X_test.copy(), data_type
                        )
                        X_train_scaled, X_val_scaled, X_test_scaled, _ = self.apply_scaling(
                            X_train_log, X_val_log, X_test_log, data_type
                        )
                        
                        model = self.models[model_key]
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    except Exception as e:
                        self.logger.error(f"{model_key} ROC 곡선 예측 실패: {e}")
                        continue
                else:
                    y_pred_proba = np.array(y_pred_proba)
            
            # ROC 곡선 계산
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # 플롯
            color = colors[i % len(colors)]
            linewidth = 3 if model_name == 'ensemble' else 2
            linestyle = '-' if model_name != 'ensemble' else '--'
            
            # 모델명을 더 명확하게 표시
            display_name = f'{model_name.upper()}_{data_type.upper()}'
            plt.plot(fpr, tpr, color=color, lw=linewidth, linestyle=linestyle,
                    label=f'{display_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison - All Models', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 저장 경로 확인 및 저장
        save_path = viz_dir / 'roc_curves_comparison.png'
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC 곡선 저장 완료: {save_path}")
        except Exception as e:
            self.logger.error(f"ROC 곡선 저장 실패: {e}")
        finally:
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
                # 앙상블은 전처리 내장하므로 원본 데이터 사용
                y_pred_proba = model.predict_proba(X_test)  # 앙상블은 이미 확률값 반환
            else:
                # 일반 모델의 경우 evaluate_model에서 저장된 예측 결과 사용
                y_test = self.data[data_type]['y_test']
                predictions = result.get('predictions', {})
                y_pred_proba = predictions.get('y_proba_test', [])
                
                # 저장된 예측이 없는 경우에만 직접 예측 (백업)
                if not y_pred_proba:
                    self.logger.warning(f"{model_key}: 저장된 예측 결과 없음, 직접 예측 수행")
                    # 전처리 적용해서 예측
                    try:
                        X_test = self.data[data_type]['X_test']
                        X_train = self.data[data_type]['X_train']
                        X_val = self.data[data_type]['X_val']
                        
                        # 전처리 적용
                        X_train_log, X_val_log, X_test_log, _ = self.apply_log_transform(
                            X_train.copy(), X_val.copy(), X_test.copy(), data_type
                        )
                        X_train_scaled, X_val_scaled, X_test_scaled, _ = self.apply_scaling(
                            X_train_log, X_val_log, X_test_log, data_type
                        )
                        
                        model = self.models[model_key]
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    except Exception as e:
                        self.logger.error(f"{model_key} PR 곡선 예측 실패: {e}")
                        continue
                else:
                    y_pred_proba = np.array(y_pred_proba)
            
            # PR 곡선 계산
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap_score = average_precision_score(y_test, y_pred_proba)
            
            # 플롯
            color = colors[i % len(colors)]
            linewidth = 3 if model_name == 'ensemble' else 2
            linestyle = '-' if model_name != 'ensemble' else '--'
            
            # 모델명을 더 명확하게 표시
            display_name = f'{model_name.upper()}_{data_type.upper()}'
            plt.plot(recall, precision, color=color, lw=linewidth, linestyle=linestyle,
                    label=f'{display_name} (AP = {ap_score:.3f})')
        
        # 기준선 추가 (양성 클래스 비율)
        if len(self.model_results) > 0:
            # 첫 번째 모델의 타겟 데이터를 사용하여 기준선 계산
            first_result = list(self.model_results.values())[0]
            first_data_type = first_result['data_type']
            y_test_baseline = self.data[first_data_type]['y_test']
            baseline = np.mean(y_test_baseline)
            plt.axhline(y=baseline, color='k', linestyle=':', lw=2, label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison - All Models', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 저장 경로 확인 및 저장
        save_path = viz_dir / 'pr_curves_comparison.png'
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"PR 곡선 저장 완료: {save_path}")
        except Exception as e:
            self.logger.error(f"PR 곡선 저장 실패: {e}")
        finally:
            plt.close()
    
    def _plot_performance_comparison(self, viz_dir: Path):
        """성능 지표 비교"""
        metrics = ['test_auc', 'test_f1', 'test_precision', 'test_recall']
        metric_names = ['AUC', 'F1-Score', 'Precision', 'Recall']
        
        model_names = []
        metric_values = {metric: [] for metric in metrics}
        
        for model_key, result in self.model_results.items():
            model_name = f"{result['model_name'].upper()}_{result['data_type'].upper()}"
            model_names.append(model_name)
            
            # test_metrics 구조 확인 후 안전하게 접근
            test_metrics = result.get('test_metrics', {})
            metric_values['test_auc'].append(test_metrics.get('auc', test_metrics.get('roc_auc', 0)))
            metric_values['test_f1'].append(test_metrics.get('f1_optimal', test_metrics.get('f1_default', 0)))
            metric_values['test_precision'].append(test_metrics.get('precision_optimal', test_metrics.get('precision_default', 0)))
            metric_values['test_recall'].append(test_metrics.get('recall_optimal', test_metrics.get('recall_default', 0)))
        
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
        
        # 전체 제목 추가
        fig.suptitle('Performance Metrics Comparison - All Models', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # 제목 공간 확보
        
        # 저장 경로 확인 및 저장
        save_path = viz_dir / 'performance_comparison.png'
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"성능 비교 차트 저장 완료: {save_path}")
        except Exception as e:
            self.logger.error(f"성능 비교 차트 저장 실패: {e}")
        finally:
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
        
        # 저장 경로 확인 및 저장
        save_path = viz_dir / 'feature_importance.png'
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"특성 중요도 차트 저장 완료: {save_path}")
        except Exception as e:
            self.logger.error(f"특성 중요도 차트 저장 실패: {e}")
        finally:
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
        
        # 저장 경로 확인 및 저장
        save_path = viz_dir / 'threshold_analysis.png'
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"임계값 분석 차트 저장 완료: {save_path}")
        except Exception as e:
            self.logger.error(f"임계값 분석 차트 저장 실패: {e}")
        finally:
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
            
            train_metrics = result.get('train_metrics', {})
            test_metrics = result.get('test_metrics', {})
            
            for metric in metrics:
                # 다양한 키 이름 시도 (auc vs roc_auc 등)
                if metric == 'auc':
                    train_val = train_metrics.get('auc', train_metrics.get('roc_auc', 0))
                    test_val = test_metrics.get('auc', test_metrics.get('roc_auc', 0))
                else:
                    train_val = train_metrics.get(metric, 0)
                    test_val = test_metrics.get(metric, 0)
                
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
        
        # 저장 경로 확인 및 저장
        save_path = viz_dir / 'train_vs_test_comparison.png'
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Train vs Test 비교 차트 저장 완료: {save_path}")
        except Exception as e:
            self.logger.error(f"Train vs Test 비교 차트 저장 실패: {e}")
        finally:
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
        
        # 저장 경로 확인 및 저장
        save_path = viz_dir / 'overfitting_heatmap.png'
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Overfitting 히트맵 저장 완료: {save_path}")
        except Exception as e:
            self.logger.error(f"Overfitting 히트맵 저장 실패: {e}")
        finally:
            plt.close()
        
        # 3. 모델별 전체 성능 요약 (Train/Val/Test)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 검증 데이터 값도 추가
        val_values = {metric: [] for metric in metrics}
        for model_key, result in model_results.items():
            val_metrics = result.get('val_metrics', {})
            for metric in metrics:
                # 다양한 키 이름 시도 (auc vs roc_auc 등)
                if metric == 'auc':
                    val_val = val_metrics.get('auc', val_metrics.get('roc_auc', 0))
                else:
                    val_val = val_metrics.get(metric, 0)
                val_values[metric].append(val_val)
        
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
        
        # 저장 경로 확인 및 저장
        save_path = viz_dir / 'train_val_test_summary.png'
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Train/Val/Test 요약 차트 저장 완료: {save_path}")
        except Exception as e:
            self.logger.error(f"Train/Val/Test 요약 차트 저장 실패: {e}")
        finally:
            plt.close()
        
        self.logger.info("Train vs Test 비교 시각화 완료")
    
    def _plot_ensemble_weights(self, viz_dir: Path):
        """앙상블 가중치 시각화 (모델 성능 기반)"""
        try:
            # 앙상블 모델 제외
            model_results = {k: v for k, v in self.model_results.items() 
                           if v['model_name'] != 'ensemble'}
            
            if not model_results:
                self.logger.info("앙상블 가중치 시각화를 위한 모델이 없습니다.")
                return
            
            # 메트릭별 성능 기반 가중치 계산
            metrics = ['test_auc', 'test_f1', 'test_precision', 'test_recall']
            metric_names = ['AUC', 'F1-Score', 'Precision', 'Recall']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                ax = axes[i]
                
                # 모델별 성능 수집
                model_names = []
                performance_scores = []
                
                for model_key, result in model_results.items():
                    model_name = f"{result['model_name']}_{result['data_type']}"
                    model_names.append(model_name)
                    
                    # test_metrics에서 안전하게 값 가져오기
                    test_metrics = result.get('test_metrics', {})
                    if metric == 'test_auc':
                        score = test_metrics.get('auc', test_metrics.get('roc_auc', 0))
                    elif metric == 'test_f1':
                        score = test_metrics.get('f1_optimal', test_metrics.get('f1_default', 0))
                    elif metric == 'test_precision':
                        score = test_metrics.get('precision_optimal', test_metrics.get('precision_default', 0))
                    elif metric == 'test_recall':
                        score = test_metrics.get('recall_optimal', test_metrics.get('recall_default', 0))
                    else:
                        score = result.get(metric, 0)
                    
                    performance_scores.append(score)
                
                # 성능 기반 가중치 계산 (소프트맥스)
                if sum(performance_scores) > 0:
                    exp_scores = np.exp(np.array(performance_scores))
                    weights = exp_scores / np.sum(exp_scores)
                else:
                    weights = np.ones(len(performance_scores)) / len(performance_scores)
                
                # 파이차트
                colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
                wedges, texts, autotexts = ax.pie(weights, labels=model_names, autopct='%1.1f%%',
                                                 colors=colors, startangle=90)
                
                # 텍스트 크기 조정
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(10)
                    autotext.set_weight('bold')
                
                ax.set_title(f'{metric_name} 기반 앙상블 가중치', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 저장 경로 확인 및 저장
            save_path = viz_dir / 'ensemble_weights.png'
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"앙상블 가중치 차트 저장 완료: {save_path}")
            except Exception as e:
                self.logger.error(f"앙상블 가중치 차트 저장 실패: {e}")
            finally:
                plt.close()
                
        except Exception as e:
            self.logger.error(f"앙상블 가중치 시각화 중 오류: {e}")
    
    def _plot_ensemble_performance_comparison(self, viz_dir: Path):
        """앙상블 모델을 포함한 성능 비교 차트 (ensemble.png)"""
        try:
            # 모든 모델 (앙상블 포함) 성능 수집
            all_results = {}
            
            # 개별 모델 결과 수집
            for model_key, result in self.model_results.items():
                if result['model_name'] != 'ensemble':  # 앙상블 제외
                    all_results[model_key] = result
            
            # 앙상블 모델 결과 추가 (앙상블이 실행된 경우)
            if hasattr(self, 'ensemble_results') and self.ensemble_results:
                # 앙상블 결과가 있는 경우 추가
                ensemble_result = {
                    'model_name': 'ensemble',
                    'data_type': 'combined_models',
                    'test_metrics': self.ensemble_results.get('test_metrics', {})
                }
                all_results['ensemble_combined_models'] = ensemble_result
            elif 'ensemble_model' in self.models:
                # 앙상블 모델 객체가 있는 경우
                ensemble_pipeline = self.models['ensemble_model']
                ensemble_result = {
                    'model_name': 'ensemble',
                    'data_type': 'combined_models',
                    'test_metrics': {}
                }
                
                # 앙상블 파이프라인에서 결과 가져오기
                if hasattr(ensemble_pipeline, 'results') and ensemble_pipeline.results:
                    ensemble_metrics = ensemble_pipeline.results.get('test_metrics', {})
                    ensemble_result['test_metrics'] = ensemble_metrics
                
                all_results['ensemble_combined_models'] = ensemble_result
            
            if not all_results:
                self.logger.info("앙상블 성능 비교를 위한 모델이 없습니다.")
                return
            
            # 성능 메트릭 정의
            metrics = ['test_auc', 'test_f1', 'test_precision', 'test_recall']
            metric_names = ['AUC', 'F1-Score', 'Precision', 'Recall']
            
            model_names = []
            metric_values = {metric: [] for metric in metrics}
            
            # 데이터 수집 및 정렬 (모델 타입별로 그룹화)
            model_order = ['logistic_regression', 'random_forest', 'xgboost', 'ensemble']
            data_type_order = ['normal', 'smote', 'undersampling', 'combined', 'combined_models']
            
            # 정렬된 결과 생성
            sorted_results = []
            for model_type in model_order:
                for data_type in data_type_order:
                    for model_key, result in all_results.items():
                        if (result['model_name'] == model_type and 
                            result['data_type'] == data_type):
                            sorted_results.append((model_key, result))
            
            # 정렬된 데이터로 메트릭 수집
            for model_key, result in sorted_results:
                if result['model_name'] == 'ensemble':
                    model_name = "ENSEMBLE"
                else:
                    model_name = f"{result['model_name'].upper()}"
                    if result['data_type'] != 'normal':
                        model_name += f"_{result['data_type'].upper()}"
                
                model_names.append(model_name)
                
                # test_metrics에서 값 추출
                test_metrics = result.get('test_metrics', {})
                
                # AUC 값
                auc_value = test_metrics.get('auc', test_metrics.get('roc_auc', 0))
                metric_values['test_auc'].append(auc_value)
                
                # F1 값 (optimal > default > f1)
                f1_value = test_metrics.get('f1_optimal', 
                          test_metrics.get('f1_default', 
                          test_metrics.get('f1', 0)))
                metric_values['test_f1'].append(f1_value)
                
                # Precision 값
                precision_value = test_metrics.get('precision_optimal',
                                test_metrics.get('precision_default',
                                test_metrics.get('precision', 0)))
                metric_values['test_precision'].append(precision_value)
                
                # Recall 값
                recall_value = test_metrics.get('recall_optimal',
                             test_metrics.get('recall_default',
                             test_metrics.get('recall', 0)))
                metric_values['test_recall'].append(recall_value)
            
            if not model_names:
                self.logger.info("표시할 모델 결과가 없습니다.")
                return
            
            # 서브플롯 생성
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            axes = axes.flatten()
            
            # 색상 설정
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'plum', 'wheat', 'lightgray', 'cyan', 'yellow']
            ensemble_color = 'darkred'  # 앙상블 강조 색상
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                ax = axes[i]
                
                # 색상 배정
                bar_colors = []
                for j, model_name in enumerate(model_names):
                    if model_name == "ENSEMBLE":
                        bar_colors.append(ensemble_color)
                    else:
                        bar_colors.append(colors[j % len(colors)])
                
                bars = ax.bar(range(len(model_names)), metric_values[metric], 
                             color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
                
                ax.set_title(f'{name} Comparison (Including Ensemble)', fontsize=14, fontweight='bold')
                ax.set_ylabel(name, fontsize=12)
                ax.set_ylim(0, 1.0)
                ax.grid(True, alpha=0.3, axis='y')
                
                # 값 표시
                for bar, value in zip(bars, metric_values[metric]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
                
                # x축 설정
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
            
            # 전체 제목
            fig.suptitle('Performance Comparison: All Models + Ensemble', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            # 범례 추가
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor=ensemble_color, alpha=0.8, label='Ensemble Model'),
                plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.8, label='Individual Models')
            ]
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.94))
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.88, bottom=0.15)  # 제목과 x축 레이블 공간 확보
            
            # ensemble.png로 저장
            save_path = viz_dir / 'ensemble.png'
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"앙상블 성능 비교 차트 저장 완료: {save_path}")
                
                # 결과 요약 로그
                self.logger.info("앙상블 성능 비교 결과:")
                for name, values in zip(model_names, zip(*[metric_values[m] for m in metrics])):
                    self.logger.info(f"  {name}: AUC={values[0]:.3f}, F1={values[1]:.3f}, "
                                   f"Precision={values[2]:.3f}, Recall={values[3]:.3f}")
                    
            except Exception as e:
                self.logger.error(f"앙상블 성능 비교 차트 저장 실패: {e}")
            finally:
                plt.close()
                
        except Exception as e:
            self.logger.error(f"앙상블 성능 비교 시각화 중 오류: {e}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
    
    def find_optimal_threshold(self, model, X_val, y_val):
        """
        Find optimal threshold using F1 score maximization with config-based search range
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            tuple: (optimal_threshold, optimal_f1_score)
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        import numpy as np
        
        # Get validation predictions
        if hasattr(model, 'predict_proba'):
            y_val_proba = model.predict_proba(X_val)[:, 1]
        else:
            self.logger.warning("Model does not support predict_proba, using decision_function")
            y_val_proba = model.decision_function(X_val)
        
        # 디버깅 정보 출력
        self.logger.info(f"Threshold 최적화 디버깅:")
        self.logger.info(f"  - 검증 데이터 크기: {len(y_val)}")
        self.logger.info(f"  - 양성 클래스 비율: {y_val.mean():.4f} ({y_val.sum()}/{len(y_val)})")
        self.logger.info(f"  - 예측 확률 범위: [{y_val_proba.min():.4f}, {y_val_proba.max():.4f}]")
        self.logger.info(f"  - 예측 확률 평균: {y_val_proba.mean():.4f}")
        
        # 예측 확률이 모두 동일한 경우 체크
        if np.std(y_val_proba) < 1e-6:
            self.logger.warning("모든 예측 확률이 거의 동일합니다. 모델이 제대로 학습되지 않았을 수 있습니다.")
            return 0.5, 0.0
        
        # Config에서 임계값 탐색 범위 가져오기
        threshold_config = self.config.get('threshold_optimization', {})
        search_range = threshold_config.get('search_range', {})
        
        # 기본값 설정
        low = search_range.get('low', 0.0005)
        high = search_range.get('high', 0.5) 
        n_grid = search_range.get('n_grid', 500)
        
        self.logger.info(f"  - 설정된 임계값 탐색 범위: [{low:.4f}, {high:.4f}] ({n_grid}개 그리드)")
        
        # 그리드 기반 임계값 후보 생성
        threshold_candidates = np.linspace(low, high, n_grid)
        
        # 각 임계값에 대해 F1 점수 계산
        best_f1 = 0
        best_threshold = 0.5
        f1_scores = []
        
        for threshold in threshold_candidates:
            y_pred = (y_val_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            f1_scores.append(f1)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        f1_scores = np.array(f1_scores)
        
        # 디버깅: F1 점수 분포 확인
        self.logger.info(f"  - F1 점수 범위: [{f1_scores.min():.4f}, {f1_scores.max():.4f}]")
        self.logger.info(f"  - 0이 아닌 F1 점수 개수: {np.sum(f1_scores > 0)}/{len(f1_scores)}")
        
        # F1 점수가 모두 0인 경우 대안 방법 사용
        if best_f1 == 0:
            self.logger.warning("모든 임계값에서 F1 점수가 0입니다.")
            
            # 대안 1: Balanced Accuracy로 임계값 선택
            self.logger.info("대안 1: Balanced Accuracy 기준으로 임계값 선택 시도")
            try:
                from sklearn.metrics import balanced_accuracy_score
                best_balanced_acc = 0
                best_threshold_ba = 0.5
                
                # Config 범위 내에서 Balanced Accuracy 계산
                for thresh in threshold_candidates[::10]:  # 샘플링으로 계산 속도 향상
                    y_pred_thresh = (y_val_proba >= thresh).astype(int)
                    if len(np.unique(y_pred_thresh)) > 1:  # 두 클래스가 모두 예측되는 경우만
                        ba_score = balanced_accuracy_score(y_val, y_pred_thresh)
                        if ba_score > best_balanced_acc:
                            best_balanced_acc = ba_score
                            best_threshold_ba = thresh
                
                if best_balanced_acc > 0.5:
                    self.logger.info(f"Balanced Accuracy 기준 최적 임계값: {best_threshold_ba:.3f} (BA: {best_balanced_acc:.4f})")
                    return best_threshold_ba, 0.0
            except Exception as e:
                self.logger.warning(f"Balanced Accuracy 기준 임계값 선택 실패: {e}")
            
            # 대안 2: Youden's J statistic (Sensitivity + Specificity - 1) 기준
            self.logger.info("대안 2: Youden's J statistic 기준으로 임계값 선택 시도")
            try:
                from sklearn.metrics import roc_curve
                fpr, tpr, roc_thresholds = roc_curve(y_val, y_val_proba)
                
                # Config 범위 내의 임계값만 고려
                mask = (roc_thresholds >= low) & (roc_thresholds <= high)
                if np.any(mask):
                    fpr_filtered = fpr[mask]
                    tpr_filtered = tpr[mask]
                    thresholds_filtered = roc_thresholds[mask]
                    
                    # Youden's J statistic 계산
                    j_scores = tpr_filtered - fpr_filtered
                    best_j_idx = np.argmax(j_scores)
                    best_threshold_j = thresholds_filtered[best_j_idx]
                    best_j_score = j_scores[best_j_idx]
                    
                    if best_j_score > 0:
                        self.logger.info(f"Youden's J 기준 최적 임계값: {best_threshold_j:.3f} (J: {best_j_score:.4f})")
                        return best_threshold_j, 0.0
            except Exception as e:
                self.logger.warning(f"Youden's J 기준 임계값 선택 실패: {e}")
            
            # 대안 3: 탐색 범위의 중간값 사용
            self.logger.info("대안 3: 탐색 범위의 중간값을 임계값으로 사용")
            middle_threshold = (low + high) / 2
            self.logger.info(f"탐색 범위 중간값 임계값: {middle_threshold:.3f}")
            return middle_threshold, 0.0
        
        # 정상적인 경우: 최적 임계값으로 실제 성능 재확인
        y_pred_final = (y_val_proba >= best_threshold).astype(int)
        
        final_precision = precision_score(y_val, y_pred_final, zero_division=0)
        final_recall = recall_score(y_val, y_pred_final, zero_division=0)
        final_f1 = f1_score(y_val, y_pred_final, zero_division=0)
        
        self.logger.info(f"최적 임계값 검증: {best_threshold:.4f}")
        self.logger.info(f"  - F1: {final_f1:.4f}, Precision: {final_precision:.4f}, Recall: {final_recall:.4f}")
        self.logger.info(f"  - 양성 예측 개수: {y_pred_final.sum()}/{len(y_pred_final)}")
        
        return best_threshold, best_f1

    def run_ensemble(self):
        """앙상블 모델 실행"""
        if not self.config['ensemble']['enabled']:
            self.logger.info("앙상블이 비활성화되어 있습니다.")
            return
        
        self.logger.info("앙상블 모델 실행 시작")
        
        try:
            from .ensemble_pipeline import EnsemblePipeline
        except ImportError:
            # 앙상블 파이프라인이 없는 경우 간단한 대체 구현
            self.logger.warning("EnsemblePipeline을 찾을 수 없습니다. 앙상블 건너뜀")
            return
        
        # 앙상블에 사용할 모델들 선택
        ensemble_models = {}
        ensemble_config = self.config['ensemble']
        
        # 설정에서 지정된 모델과 데이터타입 조합만 선택
        target_models = ensemble_config.get('models', [])
        target_data_types = ensemble_config.get('data_types', ['normal'])
        
        for model_name in target_models:
            for data_type in target_data_types:
                model_key = f"{data_type}__{model_name}"
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
        ensemble_key = "combined_models__ensemble"
        self.models[ensemble_key] = ensemble_pipeline
        self.model_results[ensemble_key] = {
            'model_name': 'ensemble',
            'data_type': 'combined_models',
            'cv_score': max([self.model_results[k].get('cv_score', 0) for k in ensemble_models.keys()]) if ensemble_models else 0,  # 최고 CV 점수 사용
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