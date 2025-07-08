"""
Optuna 최적화 하이퍼파라미터 간단 추출기
=====================================

final_models에서 Optuna로 최적화된 핵심 하이퍼파라미터만 
간단하게 추출하여 정리합니다.
"""

import joblib
import json
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def extract_optuna_params(model_path):
    """Optuna로 최적화된 핵심 하이퍼파라미터만 추출"""
    try:
        model = joblib.load(model_path)
        
        # 앙상블 모델은 제외
        if hasattr(model, 'models') and hasattr(model, 'weights'):
            return {'model_type': 'ensemble', 'params': 'ensemble_model'}
        
        model_type = str(type(model).__name__).lower()
        
        # Logistic Regression
        if 'logistic' in model_type:
            return {
                'model_type': 'logistic_regression',
                'params': {
                    'C': round(model.C, 4),
                    'penalty': model.penalty,
                    'solver': model.solver,
                    'max_iter': model.max_iter
                }
            }
        
        # Random Forest
        elif 'forest' in model_type:
            return {
                'model_type': 'random_forest',
                'params': {
                    'n_estimators': model.n_estimators,
                    'max_depth': model.max_depth,
                    'min_samples_split': model.min_samples_split,
                    'min_samples_leaf': model.min_samples_leaf,
                    'max_features': round(model.max_features, 3) if isinstance(model.max_features, float) else model.max_features
                }
            }
        
        # XGBoost
        elif 'xgb' in model_type:
            return {
                'model_type': 'xgboost',
                'params': {
                    'n_estimators': model.n_estimators,
                    'max_depth': model.max_depth,
                    'learning_rate': round(model.learning_rate, 4),
                    'subsample': round(model.subsample, 3),
                    'colsample_bytree': round(model.colsample_bytree, 3),
                    'reg_alpha': round(model.reg_alpha, 2),
                    'reg_lambda': round(model.reg_lambda, 2),
                    'scale_pos_weight': round(model.scale_pos_weight, 1) if model.scale_pos_weight else None
                }
            }
        
        else:
            return {'model_type': 'unknown', 'params': {}}
            
    except Exception as e:
        return {'model_type': 'error', 'params': str(e)}


def main():
    """메인 실행 함수"""
    print("🎯 Optuna 최적화 하이퍼파라미터 간단 추출")
    
    models_dir = Path("/Users/jojongho/KDT/P2_Default-invest/final_models")
    model_files = list(models_dir.glob("*.joblib"))
    
    # 결과 저장
    optuna_params = {}
    
    print("\n📋 모델별 최적화된 하이퍼파라미터:")
    print("="*60)
    
    for model_file in sorted(model_files):
        # 파일명에서 정보 추출
        filename = model_file.stem
        if "__" in filename:
            data_type, model_name = filename.split("__", 1)
            model_name = model_name.replace("_model", "")
        else:
            data_type = "unknown"
            model_name = filename.replace("_model", "")
        
        # 하이퍼파라미터 추출
        result = extract_optuna_params(model_file)
        
        # 키 생성
        model_key = f"{data_type}__{model_name}"
        optuna_params[model_key] = {
            'data_type': data_type,
            'model_name': model_name,
            'model_type': result['model_type'],
            'optimized_params': result['params']
        }
        
        # 출력
        print(f"\n🔹 {model_key}")
        print(f"   타입: {result['model_type']}")
        
        if isinstance(result['params'], dict):
            for param, value in result['params'].items():
                print(f"   {param}: {value}")
        else:
            print(f"   {result['params']}")
    
    # JSON 파일로 저장
    output_file = "optuna_hyperparameters_simple.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(optuna_params, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 저장 완료: {output_file}")
    
    # 모델 타입별 요약
    print(f"\n📊 모델 타입별 개수:")
    type_counts = {}
    for info in optuna_params.values():
        model_type = info['model_type']
        type_counts[model_type] = type_counts.get(model_type, 0) + 1
    
    for model_type, count in type_counts.items():
        print(f"   {model_type}: {count}개")
    
    # CSV로도 저장 (더 보기 쉽게)
    rows = []
    for key, info in optuna_params.items():
        row = {
            'Model_Key': key,
            'Data_Type': info['data_type'],
            'Model_Name': info['model_name'],
            'Model_Type': info['model_type']
        }
        
        if isinstance(info['optimized_params'], dict):
            row.update(info['optimized_params'])
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_file = "optuna_hyperparameters_simple.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✅ CSV 저장 완료: {csv_file}")
    
    return optuna_params


if __name__ == "__main__":
    main()