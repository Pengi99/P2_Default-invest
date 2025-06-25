"""
데이터 전처리 파이프라인
==============================

기능:
1. 데이터 로드 및 5:3:2 분할
2. 결측치 처리 (50% 이상 결측 행 삭제 + median 대체)
3. 윈저라이징 (양 옆 0.05%)
4. 라소 회귀 피처 선택

Config를 통한 커스터마이징 지원
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 필요한 라이브러리들
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessingPipeline:
    """
    데이터 전처리 파이프라인 클래스
    
    Config 파일을 통해 모든 설정을 관리하며,
    데이터 로드부터 피처 선택까지 전체 과정을 수행 (스케일링 제외)
    """
    
    def __init__(self, config_path: str):
        """
        파이프라인 초기화
        
        Args:
            config_path: config YAML 파일 경로
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # 프로젝트 루트 디렉토리 설정
        self.project_root = Path(__file__).parent.parent.parent
        
        # 결과 저장용 딕셔너리
        self.results = {
            'experiment_info': {},
            'data_info': {},
            'preprocessing_steps': {},
            'model_performance': {},
            'selected_features': []
        }
        
        # 피처 선택 모델 저장용
        self.feature_selector = None
        
        self.logger.info("데이터 전처리 파이프라인이 초기화되었습니다 (스케일링 제외).")
    
    def _load_config(self) -> Dict:
        """Config 파일 로드"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('DataPipeline')
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
            log_dir = Path(self.config['logging']['log_file']).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(self.config['logging']['log_file'])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        self.logger.info("데이터 로드 시작")
        
        data_path = self.project_root / self.config['data']['input_path']
        df = pd.read_csv(data_path, dtype={'거래소코드': str})
        
        self.logger.info(f"데이터 로드 완료: {df.shape}")
        
        # 기본 정보 저장
        self.results['data_info'] = {
            'original_shape': df.shape,
            'columns': list(df.columns),
            'target_distribution': df[self.config['feature_engineering']['target_column']].value_counts().to_dict(),
            'missing_data_summary': df.isnull().sum().to_dict()
        }
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터를 5:3:2로 분할"""
        self.logger.info("데이터 분할 시작 (5:3:2)")
        
        target_col = self.config['feature_engineering']['target_column']
        
        # 먼저 train과 temp로 분할 (5:5)
        if self.config['data_split']['stratify']:
            train_df, temp_df = train_test_split(
                df, 
                test_size=0.5,  # 50%를 temp로
                random_state=self.config['data_split']['random_state'],
                stratify=df[target_col]
            )
        else:
            train_df, temp_df = train_test_split(
                df,
                test_size=0.5,
                random_state=self.config['data_split']['random_state']
            )
        
        # temp를 val과 test로 분할 (3:2)
        val_ratio = self.config['data_split']['val_ratio']
        test_ratio = self.config['data_split']['test_ratio']
        val_size = val_ratio / (val_ratio + test_ratio)  # temp 내에서의 val 비율
        
        if self.config['data_split']['stratify']:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1-val_size),
                random_state=self.config['data_split']['random_state'],
                stratify=temp_df[target_col]
            )
        else:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1-val_size),
                random_state=self.config['data_split']['random_state']
            )
        
        self.logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        
        # 분할 정보 저장
        self.results['preprocessing_steps']['data_split'] = {
            'train_shape': train_df.shape,
            'val_shape': val_df.shape,
            'test_shape': test_df.shape,
            'train_target_dist': train_df[target_col].value_counts().to_dict(),
            'val_target_dist': val_df[target_col].value_counts().to_dict(),
            'test_target_dist': test_df[target_col].value_counts().to_dict()
        }
        
        return train_df, val_df, test_df
    
    def handle_missing_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """결측치 처리"""
        self.logger.info("결측치 처리 시작")
        
        # 피처 컬럼들만 추출
        exclude_cols = self.config['feature_engineering']['exclude_columns'] + [self.config['feature_engineering']['target_column']]
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        original_train_shape = train_df.shape
        original_val_shape = val_df.shape
        original_test_shape = test_df.shape
        
        # 1. 50% 이상 결측치인 행 삭제
        threshold = self.config['missing_data']['row_missing_threshold']
        
        # 각 행의 결측치 비율 계산
        train_missing_rate = train_df[feature_cols].isnull().sum(axis=1) / len(feature_cols)
        val_missing_rate = val_df[feature_cols].isnull().sum(axis=1) / len(feature_cols)
        test_missing_rate = test_df[feature_cols].isnull().sum(axis=1) / len(feature_cols)
        
        # 임계값 이하인 행만 유지
        train_df = train_df[train_missing_rate <= threshold].copy()
        val_df = val_df[val_missing_rate <= threshold].copy()
        test_df = test_df[test_missing_rate <= threshold].copy()
        
        self.logger.info(f"행 삭제 후 - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        
        # 2. 결측값 대체
        imputation_method = self.config['missing_data']['imputation_method']
        
        if imputation_method == "median":
            imputer = SimpleImputer(strategy='median')
        elif imputation_method == "mean":
            imputer = SimpleImputer(strategy='mean')
        elif imputation_method == "mode":
            imputer = SimpleImputer(strategy='most_frequent')
        elif imputation_method == "knn":
            imputer = KNNImputer(n_neighbors=5)
        else:
            raise ValueError(f"지원하지 않는 결측값 대체 방법: {imputation_method}")
        
        # Train 데이터로 imputer 학습
        imputer.fit(train_df[feature_cols])
        
        # 모든 데이터셋에 적용
        train_df[feature_cols] = imputer.transform(train_df[feature_cols])
        val_df[feature_cols] = imputer.transform(val_df[feature_cols])
        test_df[feature_cols] = imputer.transform(test_df[feature_cols])
        
        self.logger.info(f"결측값 대체 완료 ({imputation_method})")
        
        # 결측치 처리 정보 저장
        self.results['preprocessing_steps']['missing_data'] = {
            'method': imputation_method,
            'threshold': threshold,
            'before_shape': {
                'train': original_train_shape,
                'val': original_val_shape,
                'test': original_test_shape
            },
            'after_shape': {
                'train': train_df.shape,
                'val': val_df.shape,
                'test': test_df.shape
            },
            'rows_removed': {
                'train': original_train_shape[0] - train_df.shape[0],
                'val': original_val_shape[0] - val_df.shape[0],
                'test': original_test_shape[0] - test_df.shape[0]
            }
        }
        
        return train_df, val_df, test_df
    
    def apply_winsorization(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """윈저라이징 적용"""
        if not self.config['outlier_treatment']['enabled']:
            self.logger.info("윈저라이징이 비활성화되어 있습니다.")
            return train_df, val_df, test_df
        
        self.logger.info("윈저라이징 적용 시작")
        
        # 피처 컬럼들만 추출
        exclude_cols = self.config['feature_engineering']['exclude_columns'] + [self.config['feature_engineering']['target_column']]
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        lower_pct = self.config['outlier_treatment']['winsorization']['lower_percentile']
        upper_pct = self.config['outlier_treatment']['winsorization']['upper_percentile']
        
        # Train 데이터를 기준으로 임계값 계산
        winsor_limits = {}
        for col in feature_cols:
            lower_limit = train_df[col].quantile(lower_pct)
            upper_limit = train_df[col].quantile(upper_pct)
            winsor_limits[col] = (lower_limit, upper_limit)
        
        # 윈저라이징 적용
        for df_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            for col in feature_cols:
                lower_limit, upper_limit = winsor_limits[col]
                df[col] = np.clip(df[col], lower_limit, upper_limit)
        
        self.logger.info(f"윈저라이징 완료 (하위 {lower_pct*100}%, 상위 {upper_pct*100}%)")
        
        # 윈저라이징 정보 저장
        self.results['preprocessing_steps']['winsorization'] = {
            'enabled': True,
            'lower_percentile': lower_pct,
            'upper_percentile': upper_pct,
            'limits': {col: limits for col, limits in winsor_limits.items()}
        }
        
        return train_df, val_df, test_df
    
    def select_features_with_lasso(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """라소 회귀를 이용한 피처 선택"""
        if not self.config['feature_selection']['enabled']:
            self.logger.info("피처 선택이 비활성화되어 있습니다.")
            return train_df, val_df, test_df
        
        self.logger.info("라소 회귀 피처 선택 시작")
        
        # 피처와 타겟 분리
        exclude_cols = self.config['feature_engineering']['exclude_columns'] + [self.config['feature_engineering']['target_column']]
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        target_col = self.config['feature_engineering']['target_column']
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        # 라소 회귀 설정
        lasso_config = self.config['feature_selection']['lasso']
        alphas = lasso_config['alpha_range']
        cv_folds = lasso_config['cv_folds']
        max_iter = lasso_config['max_iter']
        random_state = lasso_config['random_state']
        
        # LassoCV로 최적 alpha 찾기
        lasso_cv = LassoCV(
            alphas=alphas,
            cv=cv_folds,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=self.config['performance'].get('n_jobs', 1)
        )
        
        lasso_cv.fit(X_train, y_train)
        
        # 1se rule 적용할지 결정
        if lasso_config['alpha_selection'] == "1se":
            # 1-standard-error rule
            mean_scores = lasso_cv.mse_path_.mean(axis=1)
            std_scores = lasso_cv.mse_path_.std(axis=1)
            
            best_idx = np.argmin(mean_scores)
            best_score = mean_scores[best_idx]
            best_std = std_scores[best_idx]
            
            # 1se 임계값 이하인 alpha 중 가장 큰 값 선택
            threshold = best_score + best_std
            valid_indices = np.where(mean_scores <= threshold)[0]
            selected_alpha_idx = valid_indices[0]  # 가장 큰 alpha (첫 번째 인덱스)
            selected_alpha = alphas[selected_alpha_idx]
        else:
            selected_alpha = lasso_cv.alpha_
        
        # 선택된 alpha로 최종 모델 학습
        lasso = Lasso(alpha=selected_alpha, max_iter=max_iter, random_state=random_state)
        lasso.fit(X_train, y_train)
        
        # 선택된 피처들 (계수가 0이 아닌 피처들)
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if lasso.coef_[i] != 0]
        
        self.logger.info(f"선택된 피처 수: {len(selected_features)} / {len(feature_cols)}")
        
        # 피처 선택 적용
        all_selected_cols = selected_features + [target_col] + self.config['feature_engineering']['exclude_columns']
        all_selected_cols = [col for col in all_selected_cols if col in train_df.columns]
        
        train_selected = train_df[all_selected_cols]
        val_selected = val_df[all_selected_cols]
        test_selected = test_df[all_selected_cols]
        
        # 모델 성능 평가
        train_pred = lasso.predict(X_train[selected_features])
        val_pred = lasso.predict(X_val[selected_features])
        test_pred = lasso.predict(X_test[selected_features])
        
        # 결과 저장
        self.feature_selector = lasso
        self.results['selected_features'] = selected_features
        
        self.results['preprocessing_steps']['feature_selection'] = {
            'method': 'lasso',
            'selected_alpha': selected_alpha,
            'original_features': len(feature_cols),
            'selected_features': len(selected_features),
            'feature_names': selected_features
        }
        
        self.results['model_performance'] = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'val_mse': mean_squared_error(y_val, val_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        return train_selected, val_selected, test_selected
    
    def save_results(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """결과 저장"""
        self.logger.info("결과 저장 시작")
        
        # 출력 디렉토리 설정
        output_dir = self.project_root / self.config['data']['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 서브디렉토리 생성 여부 확인
        create_subdirectory = self.config['experiment'].get('create_subdirectory', True)
        
        if create_subdirectory:
            # 실험 이름 생성
            experiment_name = self.config['experiment']['name']
            if experiment_name is None:
                experiment_name = f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            experiment_dir = output_dir / experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
        else:
            # data/final에 직접 저장
            experiment_dir = output_dir
        
        # 1. 전처리된 데이터 저장
        if self.config['output']['save_processed_data']:
            self._save_processed_data(train_df, val_df, test_df, experiment_dir)
        
        # 2. 피처 선택 모델 저장
        if self.config['output']['save_feature_selector'] and self.feature_selector is not None:
            with open(experiment_dir / "feature_selector.pkl", 'wb') as f:
                pickle.dump(self.feature_selector, f)
        
        # 3. 실험 결과 저장
        experiment_name = "preprocessing" if not create_subdirectory else self.config['experiment']['name']
        self.results['experiment_info'] = {
            'name': experiment_name,
            'config_path': str(self.config_path),
            'timestamp': datetime.now().isoformat(),
            'version': self.config['experiment']['version'],
            'description': self.config['experiment']['description']
        }
        
        # 결과를 JSON 형태로 저장 (서브디렉토리 생성 시에만)
        if create_subdirectory:
            import json
            with open(experiment_dir / "experiment_results.json", 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        # 4. Config 파일 복사 (서브디렉토리 생성 시에만)
        if self.config['output']['save_config_log'] and create_subdirectory:
            import shutil
            shutil.copy2(self.config_path, experiment_dir / "config.yaml")
        
        self.logger.info(f"결과 저장 완료: {experiment_dir}")
        
        return experiment_dir
    
    def _save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, experiment_dir: Path):
        """전처리된 데이터 저장"""
        file_naming = self.config['output']['file_naming']
        separate_features_target = file_naming.get('separate_features_target', False)
        prefix = file_naming.get('prefix', "")
        
        # 피처와 타겟 컬럼 분리
        exclude_cols = self.config['feature_engineering']['exclude_columns'] + [self.config['feature_engineering']['target_column']]
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        target_col = self.config['feature_engineering']['target_column']
        
        datasets = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        for split_name, df in datasets.items():
            if separate_features_target:
                # X, y 분리 저장
                feature_filename = prefix + file_naming['feature_format'].format(split=split_name)
                target_filename = prefix + file_naming['target_format'].format(split=split_name)
                
                # X 저장 (피처만)
                X = df[feature_cols]
                X.to_csv(experiment_dir / feature_filename, index=False)
                
                # y 저장 (타겟만)
                y = df[target_col]
                y.to_csv(experiment_dir / target_filename, index=False)
                
                self.logger.info(f"저장 완료: {feature_filename}, {target_filename}")
            else:
                # 통합 파일로 저장
                combined_filename = prefix + file_naming['combined_format'].format(split=split_name)
                df.to_csv(experiment_dir / combined_filename, index=False)
                
                self.logger.info(f"저장 완료: {combined_filename}")
    
    def generate_report(self, experiment_dir: Path):
        """결과 리포트 생성"""
        if not self.config['output']['generate_report']:
            return
        
        self.logger.info("결과 리포트 생성 중...")
        
        report_content = self._create_report_content()
        
        formats = self.config['output']['report_format']
        
        if 'txt' in formats:
            with open(experiment_dir / "preprocessing_report.txt", 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        if 'html' in formats:
            html_content = self._convert_to_html(report_content)
            with open(experiment_dir / "preprocessing_report.html", 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        self.logger.info("리포트 생성 완료")
    
    def _create_report_content(self) -> str:
        """리포트 내용 생성"""
        
        # 피처 선택 정보
        feature_selection_enabled = 'feature_selection' in self.results['preprocessing_steps']
        
        report = f"""
데이터 전처리 파이프라인 실행 결과 리포트 (스케일링 제외)
=========================================================

실험 정보
--------
- 실험명: {self.results['experiment_info']['name']}
- 실행 시간: {self.results['experiment_info']['timestamp']}
- 버전: {self.results['experiment_info']['version']}
- 설명: {self.results['experiment_info']['description']}

원본 데이터 정보
--------------
- 데이터 크기: {self.results['data_info']['original_shape']}
- 타겟 분포: {self.results['data_info']['target_distribution']}

전처리 단계별 결과
----------------

1. 데이터 분할 (5:3:2)
   - Train: {self.results['preprocessing_steps']['data_split']['train_shape']}
   - Validation: {self.results['preprocessing_steps']['data_split']['val_shape']}
   - Test: {self.results['preprocessing_steps']['data_split']['test_shape']}

2. 결측치 처리
   - 방법: {self.results['preprocessing_steps']['missing_data']['method']}
   - 임계값: {self.results['preprocessing_steps']['missing_data']['threshold']}
   - 제거된 행수: {self.results['preprocessing_steps']['missing_data']['rows_removed']}

3. 윈저라이징
   - 적용 여부: {self.results['preprocessing_steps']['winsorization']['enabled']}
   - 하위 임계값: {self.results['preprocessing_steps']['winsorization']['lower_percentile']}
   - 상위 임계값: {self.results['preprocessing_steps']['winsorization']['upper_percentile']}
"""

        # 피처 선택 정보 (활성화된 경우에만)
        if feature_selection_enabled:
            report += f"""
4. 피처 선택 (라소 회귀)
   - 원본 피처 수: {self.results['preprocessing_steps']['feature_selection']['original_features']}
   - 선택된 피처 수: {self.results['preprocessing_steps']['feature_selection']['selected_features']}
   - 선택된 Alpha: {self.results['preprocessing_steps']['feature_selection']['selected_alpha']}

모델 성능
--------
- Train MSE: {self.results['model_performance']['train_mse']:.6f}
- Validation MSE: {self.results['model_performance']['val_mse']:.6f}
- Test MSE: {self.results['model_performance']['test_mse']:.6f}
- Train R²: {self.results['model_performance']['train_r2']:.6f}
- Validation R²: {self.results['model_performance']['val_r2']:.6f}
- Test R²: {self.results['model_performance']['test_r2']:.6f}

선택된 피처 목록
--------------
{chr(10).join(f"- {feature}" for feature in self.results['selected_features'])}
"""
        else:
            report += f"""
4. 피처 선택
   - 상태: 비활성화됨
   - 모든 피처가 유지됨
"""

        return report
    
    def _convert_to_html(self, text_content: str) -> str:
        """텍스트를 HTML로 변환"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>데이터 전처리 리포트</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
        pre {{ background-color: #f5f5f5; padding: 10px; }}
    </style>
</head>
<body>
    <pre>{text_content}</pre>
</body>
</html>
"""
        return html_content
    
    def run_pipeline(self) -> str:
        """전체 파이프라인 실행"""
        self.logger.info("=== 데이터 전처리 파이프라인 시작 ===")
        
        try:
            # 1. 데이터 로드
            df = self.load_data()
            
            # 2. 데이터 분할
            train_df, val_df, test_df = self.split_data(df)
            
            # 3. 결측치 처리
            train_df, val_df, test_df = self.handle_missing_data(train_df, val_df, test_df)
            
            # 4. 윈저라이징
            train_df, val_df, test_df = self.apply_winsorization(train_df, val_df, test_df)
            
            # 5. 피처 선택
            train_df, val_df, test_df = self.select_features_with_lasso(train_df, val_df, test_df)
            
            # 6. 결과 저장
            experiment_dir = self.save_results(train_df, val_df, test_df)
            
            # 7. 리포트 생성
            self.generate_report(experiment_dir)
            
            self.logger.info("=== 데이터 전처리 파이프라인 완료 ===")
            
            return str(experiment_dir)
            
        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 오류 발생: {str(e)}")
            raise


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='데이터 전처리 파이프라인')
    parser.add_argument('--config', '-c', type=str, 
                       default='config/preprocessing_config.yaml',
                       help='Config 파일 경로')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    pipeline = DataPreprocessingPipeline(args.config)
    experiment_dir = pipeline.run_pipeline()
    
    print(f"\n✅ 전처리 완료! (스케일링 제외)")
    print(f"📁 결과 저장 위치: {experiment_dir}")
    
    # 피처 선택이 활성화된 경우에만 관련 정보 출력
    if pipeline.results['selected_features']:
        print(f"📊 선택된 피처 수: {len(pipeline.results['selected_features'])}")
        print(f"🎯 검증 R²: {pipeline.results['model_performance']['val_r2']:.4f}")
    else:
        print(f"📊 피처 선택: 비활성화됨 (모든 피처 유지)")
        print(f"🎯 데이터 처리: 완료")


if __name__ == "__main__":
    main()