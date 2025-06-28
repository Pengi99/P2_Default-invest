"""
데이터 전처리 파이프라인
==============================

기능:
1. 데이터 로드 및 5:3:2 분할
2. 결측치 처리 (20% 이상 결측 행 삭제 + median 대체)
3. 윈저라이징 (양 옆 0.5%)

Config를 통한 커스터마이징 지원
(스케일링 및 피처 선택 제외)
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
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessingPipeline:
    """
    데이터 전처리 파이프라인 클래스
    
    Config 파일을 통해 모든 설정을 관리하며,
    데이터 로드부터 윈저라이징까지 전체 과정을 수행 (스케일링 및 피처 선택 제외)
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
            'preprocessing_steps': {}
        }
        
        self.logger.info("데이터 전처리 파이프라인이 초기화되었습니다 (스케일링 및 피처 선택 제외).")
    
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
        """데이터 분할 (시계열 또는 랜덤)"""
        split_method = self.config['data_split']['split_method']
        
        if split_method == 'timeseries':
            return self._split_data_timeseries(df)
        elif split_method == 'random':
            return self._split_data_random(df)
        else:
            raise ValueError(f"지원하지 않는 분할 방식: {split_method}")
    
    def _split_data_timeseries(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """시계열 기반 데이터 분할"""
        self.logger.info("시계열 기반 데이터 분할 시작")
        
        time_col = self.config['data_split']['timeseries']['time_column']
        train_end_year = self.config['data_split']['timeseries']['train_end_year']
        val_end_year = self.config['data_split']['timeseries']['val_end_year']
        target_col = self.config['feature_engineering']['target_column']
        
        # 시간 컬럼이 존재하는지 확인
        if time_col not in df.columns:
            raise ValueError(f"시간 컬럼 '{time_col}'이 데이터에 존재하지 않습니다.")
        
        # 년도별 분할
        train_df = df[df[time_col] <= train_end_year].copy()
        val_df = df[(df[time_col] > train_end_year) & (df[time_col] <= val_end_year)].copy()
        test_df = df[df[time_col] > val_end_year].copy()
        
        # 분할 결과 확인
        if len(train_df) == 0:
            raise ValueError(f"Train 데이터가 비어있습니다. train_end_year({train_end_year})를 확인하세요.")
        if len(val_df) == 0:
            raise ValueError(f"Validation 데이터가 비어있습니다. val_end_year({val_end_year})를 확인하세요.")
        if len(test_df) == 0:
            raise ValueError(f"Test 데이터가 비어있습니다. val_end_year({val_end_year}) 이후 데이터를 확인하세요.")
        
        # 시간 범위 로깅
        train_years = sorted(train_df[time_col].unique())
        val_years = sorted(val_df[time_col].unique())
        test_years = sorted(test_df[time_col].unique())
        
        self.logger.info(f"Train: {train_df.shape} ({train_years[0]}-{train_years[-1]}년)")
        self.logger.info(f"Val: {val_df.shape} ({val_years[0]}-{val_years[-1]}년)")
        self.logger.info(f"Test: {test_df.shape} ({test_years[0]}-{test_years[-1]}년)")
        
        # 분할 정보 저장
        self.results['preprocessing_steps']['data_split'] = {
            'method': 'timeseries',
            'train_shape': train_df.shape,
            'val_shape': val_df.shape,
            'test_shape': test_df.shape,
            'train_years': train_years,
            'val_years': val_years,
            'test_years': test_years,
            'train_target_dist': train_df[target_col].value_counts().to_dict(),
            'val_target_dist': val_df[target_col].value_counts().to_dict(),
            'test_target_dist': test_df[target_col].value_counts().to_dict()
        }
        
        return train_df, val_df, test_df
    
    def _split_data_random(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """랜덤 기반 데이터 분할 (기존 방식)"""
        self.logger.info("랜덤 기반 데이터 분할 시작 (5:3:2)")
        
        target_col = self.config['feature_engineering']['target_column']
        random_config = self.config['data_split']['random']
        
        # 먼저 train과 temp로 분할 (5:5)
        if random_config['stratify']:
            train_df, temp_df = train_test_split(
                df, 
                test_size=0.5,  # 50%를 temp로
                random_state=random_config['random_state'],
                stratify=df[target_col]
            )
        else:
            train_df, temp_df = train_test_split(
                df,
                test_size=0.5,
                random_state=random_config['random_state']
            )
        
        # temp를 val과 test로 분할 (3:2)
        val_ratio = random_config['val_ratio']
        test_ratio = random_config['test_ratio']
        val_size = val_ratio / (val_ratio + test_ratio)  # temp 내에서의 val 비율
        
        if random_config['stratify']:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1-val_size),
                random_state=random_config['random_state'],
                stratify=temp_df[target_col]
            )
        else:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1-val_size),
                random_state=random_config['random_state']
            )
        
        self.logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        
        # 분할 정보 저장
        self.results['preprocessing_steps']['data_split'] = {
            'method': 'random',
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
        

        
        # 2. 실험 결과 저장
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
        
        # 3. Config 파일 복사 (서브디렉토리 생성 시에만)
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
        
        split_info = self.results['preprocessing_steps']['data_split']
        split_method = split_info['method']
        
        # 데이터 분할 정보 생성
        if split_method == 'timeseries':
            split_details = f"""1. 데이터 분할 (시계열 기반)
   - 방법: 시계열 분할
   - Train: {split_info['train_shape']} ({split_info['train_years'][0]}-{split_info['train_years'][-1]}년)
   - Validation: {split_info['val_shape']} ({split_info['val_years'][0]}-{split_info['val_years'][-1]}년)
   - Test: {split_info['test_shape']} ({split_info['test_years'][0]}-{split_info['test_years'][-1]}년)"""
        else:
            split_details = f"""1. 데이터 분할 (랜덤 기반)
   - 방법: 랜덤 분할 (5:3:2)
   - Train: {split_info['train_shape']}
   - Validation: {split_info['val_shape']}
   - Test: {split_info['test_shape']}"""
        
        report = f"""
데이터 전처리 파이프라인 실행 결과 리포트 (스케일링 및 피처 선택 제외)
=================================================================

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

{split_details}

2. 결측치 처리
   - 방법: {self.results['preprocessing_steps']['missing_data']['method']}
   - 임계값: {self.results['preprocessing_steps']['missing_data']['threshold']}
   - 제거된 행수: {self.results['preprocessing_steps']['missing_data']['rows_removed']}

3. 윈저라이징
   - 적용 여부: {self.results['preprocessing_steps']['winsorization']['enabled']}
   - 하위 임계값: {self.results['preprocessing_steps']['winsorization']['lower_percentile']}
   - 상위 임계값: {self.results['preprocessing_steps']['winsorization']['upper_percentile']}

처리 완료
--------
- 모든 피처가 유지됨 (피처 선택 없음)
- 데이터가 모델링 준비 완료
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
            
            # 5. 결과 저장
            experiment_dir = self.save_results(train_df, val_df, test_df)
            
            # 6. 리포트 생성
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
    
    print(f"\n✅ 전처리 완료! (스케일링 및 피처 선택 제외)")
    print(f"📁 결과 저장 위치: {experiment_dir}")
    print(f"📊 모든 피처 유지됨")
    print(f"🎯 데이터 처리: 완료")


if __name__ == "__main__":
    main()