#!/usr/bin/env python3
"""
Validation 방법 테스트 스크립트
=============================

새로 구현한 validation 방법들이 제대로 동작하는지 테스트
- Nested CV
- Logistic Holdout + Repeated Sampling  
- 기존 K-fold CV
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.modeling.modeling_pipeline import ModelingPipeline
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def create_test_data():
    """테스트용 데이터 생성"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.95, 0.05],  # 불균형 데이터
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i:02d}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series

def test_validation_methods():
    """다양한 validation 방법 테스트"""
    
    print("🧪 Validation 방법 테스트 시작")
    print("="*50)
    
    # 테스트 데이터 생성
    X, y = create_test_data()
    print(f"📊 테스트 데이터: {X.shape[0]:,}개 샘플, {X.shape[1]}개 특성")
    print(f"📊 클래스 분포: {y.value_counts().to_dict()}")
    
    # 간단한 모델로 테스트
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # 임시 config 생성 (최소한의 설정)
    temp_config = {
        'random_state': 42,
        'cv_folds': 3,
        'validation': {
            'method': 'logistic_holdout',
            'logistic_holdout': {
                'n_iterations': 5,
                'test_size': 0.2
            },
            'nested_cv': {
                'outer_folds': 3,
                'inner_folds': 2,
                'n_trials': 10
            }
        }
    }
    
    # Mock pipeline 객체 생성 (validation 메서드만 테스트)
    class MockPipeline:
        def __init__(self, config):
            self.config = config
            
        def apply_log_transform(self, X_train, X_val, X_test, data_type):
            return X_train, X_val, X_test, {}
            
        def apply_scaling(self, X_train, X_val, X_test, data_type):
            return X_train, X_val, X_test, {}
            
        def apply_sampling_strategy(self, X_train, y_train, data_type):
            return X_train, y_train
    
    # ModelingPipeline의 validation 메서드들을 mock pipeline에 추가
    mock_pipeline = MockPipeline(temp_config)
    
    # 메서드 바인딩
    from types import MethodType
    mock_pipeline._proper_cv_with_sampling = MethodType(ModelingPipeline._proper_cv_with_sampling, mock_pipeline)
    mock_pipeline._logistic_holdout_repeated_sampling = MethodType(ModelingPipeline._logistic_holdout_repeated_sampling, mock_pipeline)
    mock_pipeline._nested_cv_with_sampling = MethodType(ModelingPipeline._nested_cv_with_sampling, mock_pipeline)
    
    # Logger 모킹
    class MockLogger:
        def info(self, msg): print(f"ℹ️  {msg}")
        def warning(self, msg): print(f"⚠️  {msg}")
        def error(self, msg): print(f"❌ {msg}")
    
    mock_pipeline.logger = MockLogger()
    
    # 1. K-fold CV 테스트
    print("\n1️⃣ K-fold Cross Validation 테스트")
    print("-" * 30)
    try:
        scores_kfold = mock_pipeline._proper_cv_with_sampling(
            model, X, y, 'normal', cv_folds=3, scoring='roc_auc'
        )
        print(f"✅ K-fold CV 성공: {scores_kfold.mean():.4f} ± {scores_kfold.std():.4f}")
    except Exception as e:
        print(f"❌ K-fold CV 실패: {e}")
    
    # 2. Logistic Holdout + Repeated Sampling 테스트
    print("\n2️⃣ Logistic Holdout + Repeated Sampling 테스트")
    print("-" * 30)
    try:
        scores_holdout = mock_pipeline._logistic_holdout_repeated_sampling(
            model, X, y, 'normal', n_iterations=5, test_size=0.2, scoring='roc_auc'
        )
        print(f"✅ Logistic Holdout 성공: {scores_holdout.mean():.4f} ± {scores_holdout.std():.4f}")
    except Exception as e:
        print(f"❌ Logistic Holdout 실패: {e}")
    
    # 3. Nested CV 테스트 (간단한 버전)
    print("\n3️⃣ Nested Cross Validation 테스트")
    print("-" * 30)
    try:
        # 간단한 param space 정의
        param_space = {
            'C': {'type': 'float', 'low': 0.01, 'high': 10.0, 'log': True},
            'max_iter': {'type': 'int', 'low': 100, 'high': 1000}
        }
        
        scores_nested = mock_pipeline._nested_cv_with_sampling(
            LogisticRegression, param_space, X, y, 'normal', 
            outer_cv_folds=3, inner_cv_folds=2, n_trials=5, scoring='roc_auc'
        )
        print(f"✅ Nested CV 성공: {scores_nested.mean():.4f} ± {scores_nested.std():.4f}")
    except Exception as e:
        print(f"❌ Nested CV 실패: {e}")
    
    print("\n🎉 모든 테스트 완료!")

if __name__ == "__main__":
    test_validation_methods()