"""
스케일링 유연성 테스트
===================

존재하지 않는 컬럼이 있어도 파이프라인이 정상 작동하는지 테스트
"""

import pandas as pd
import numpy as np
import yaml
import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append('.')
sys.path.append('..')

from modeling_pipeline import ModelingPipeline

def create_test_data():
    """실제 컬럼 중 일부만 포함된 테스트 데이터 생성"""
    test_columns = [
        # Standard 그룹 중 일부
        '총자산', '총부채', '매출액', '자본금',
        # Robust 그룹 중 일부  
        '매출액증가율', '부채비율', '이자보상배율',
        # MinMax 그룹 중 일부
        '매출액총이익률', '유동비율', '총자산수익률',
        # 스케일링 설정에 없는 컬럼들
        '기타변수1', '기타변수2', '새로운변수'
    ]
    
    np.random.seed(42)
    n_samples = 200
    
    data = {}
    for col in test_columns:
        if '비율' in col or '률' in col:
            # 비율 변수는 0-100 범위
            data[col] = np.random.uniform(0, 100, n_samples)
        elif '증가율' in col:
            # 증가율은 -50 ~ 50 범위
            data[col] = np.random.normal(0, 20, n_samples)
        else:
            # 절댓값 변수는 큰 숫자
            data[col] = np.random.lognormal(10, 2, n_samples)
    
    df = pd.DataFrame(data)
    
    # 타겟 변수 생성 (부실 여부)
    y = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    
    return df, pd.Series(y)

def create_test_config():
    """테스트용 간단한 config 생성"""
    # 기본 config 로드
    with open('../../config/modeling_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 테스트용으로 간소화
    config['experiment']['name'] = "scaling_flexibility_test"
    config['feature_selection']['enabled'] = False
    config['ensemble']['enabled'] = False
    
    # 모델 trial 수 감소
    for model_name in config['models']:
        if 'n_trials' in config['models'][model_name]:
            config['models'][model_name]['n_trials'] = 5
    
    # 일부 모델만 활성화
    for model_name in config['models']:
        config['models'][model_name]['enabled'] = False
    config['models']['logistic_regression']['enabled'] = True
    
    # 샘플링 간소화
    for data_type in config['sampling']['data_types']:
        config['sampling']['data_types'][data_type]['enabled'] = False
    config['sampling']['data_types']['normal']['enabled'] = True
    
    return config

def run_test():
    """테스트 실행"""
    print("🧪 스케일링 유연성 테스트 시작")
    print("=" * 60)
    
    # 테스트 데이터 생성
    X, y = create_test_data()
    print(f"📊 테스트 데이터 생성: {X.shape}")
    print(f"컬럼: {list(X.columns)}")
    
    # 테스트 데이터 저장
    os.makedirs('../../data/final', exist_ok=True)
    
    # 간단한 train/val/test 분할
    n = len(X)
    train_idx = int(n * 0.6)
    val_idx = int(n * 0.8)
    
    X_train, X_val, X_test = X[:train_idx], X[val_idx:train_idx], X[val_idx:]
    y_train, y_val, y_test = y[:train_idx], y[val_idx:train_idx], y[val_idx:]
    
    # 파일 저장
    X_train.to_csv('../../data/final/X_train.csv', index=False)
    X_val.to_csv('../../data/final/X_val.csv', index=False) 
    X_test.to_csv('../../data/final/X_test.csv', index=False)
    y_train.to_csv('../../data/final/y_train.csv', index=False)
    y_val.to_csv('../../data/final/y_val.csv', index=False)
    y_test.to_csv('../../data/final/y_test.csv', index=False)
    
    print("✅ 테스트 데이터 저장 완료")
    
    # Config 생성
    config = create_test_config()
    config_path = 'test_scaling_config.yaml'
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ 테스트 Config 생성 완료")
    
    try:
        # 파이프라인 실행
        print("\n🚀 ModelingPipeline 시작...")
        pipeline = ModelingPipeline(config_path)
        
        # 데이터 로드
        pipeline.load_data()
        print("✅ 데이터 로드 완료")
        
        # 스케일링 테스트
        data_type = 'normal'
        X_train_scaled, X_val_scaled, X_test_scaled, scalers = pipeline.apply_scaling(
            pipeline.data[data_type]['X_train'].copy(),
            pipeline.data[data_type]['X_val'].copy(),
            pipeline.data[data_type]['X_test'].copy(),
            data_type
        )
        
        print("\n🎉 스케일링 테스트 성공!")
        print(f"✅ 적용된 스케일러: {len(scalers)}개")
        
        for scaler_name, scaler_info in scalers.items():
            applied_cols = len(scaler_info['columns'])
            missing_cols = len(scaler_info['missing_columns'])
            print(f"  - {scaler_name}: {applied_cols}개 적용, {missing_cols}개 누락")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 정리
        if os.path.exists(config_path):
            os.remove(config_path)
        print("🗑️ 임시 파일 정리 완료")

if __name__ == "__main__":
    success = run_test()
    if success:
        print("\n🎊 모든 테스트 성공! 스케일링 유연성이 확인되었습니다.")
    else:
        print("\n💥 테스트 실패!") 