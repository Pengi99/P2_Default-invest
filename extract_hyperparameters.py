"""
Final Models 하이퍼파라미터 추출기
=====================================

final_models 디렉토리의 joblib 모델 파일들에서 
하이퍼파라미터를 추출하여 JSON 파일로 정리합니다.
"""

import joblib
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def extract_model_hyperparameters(model_path):
    """모델 파일에서 하이퍼파라미터 추출"""
    try:
        # 모델 로드
        model = joblib.load(model_path)
        
        # 앙상블 모델인 경우
        if hasattr(model, 'models') and hasattr(model, 'weights'):
            return {
                'model_type': 'ensemble',
                'ensemble_info': {
                    'num_models': len(model.models) if hasattr(model, 'models') else 0,
                    'weights': model.weights.tolist() if hasattr(model, 'weights') else [],
                    'weight_metric': getattr(model, 'weight_metric', 'unknown'),
                    'data_types': getattr(model, 'data_types', []),
                    'model_types': getattr(model, 'model_types', [])
                }
            }
        
        # 개별 모델인 경우
        model_params = {}
        
        # XGBoost 모델
        if hasattr(model, 'get_params') and 'xgb' in str(type(model)).lower():
            params = model.get_params()
            model_params = {
                'model_type': 'xgboost',
                'hyperparameters': {k: v for k, v in params.items() if v is not None}
            }
        
        # RandomForest 모델
        elif hasattr(model, 'n_estimators') and 'forest' in str(type(model)).lower():
            params = model.get_params()
            model_params = {
                'model_type': 'random_forest',
                'hyperparameters': {k: v for k, v in params.items() if v is not None}
            }
        
        # LogisticRegression 모델
        elif hasattr(model, 'C') and 'logistic' in str(type(model)).lower():
            params = model.get_params()
            model_params = {
                'model_type': 'logistic_regression',
                'hyperparameters': {k: v for k, v in params.items() if v is not None}
            }
        
        # 일반적인 sklearn 모델
        elif hasattr(model, 'get_params'):
            params = model.get_params()
            model_params = {
                'model_type': str(type(model).__name__),
                'hyperparameters': {k: v for k, v in params.items() if v is not None}
            }
        
        else:
            model_params = {
                'model_type': str(type(model).__name__),
                'hyperparameters': {}
            }
        
        # 추가 정보 수집
        if hasattr(model, 'feature_importances_'):
            model_params['has_feature_importance'] = True
            model_params['n_features'] = len(model.feature_importances_)
        
        if hasattr(model, 'classes_'):
            model_params['n_classes'] = len(model.classes_)
            
        return model_params
        
    except Exception as e:
        return {
            'model_type': 'unknown',
            'error': str(e),
            'hyperparameters': {}
        }


def clean_hyperparameters(params):
    """JSON 직렬화 가능하도록 파라미터 정리"""
    cleaned = {}
    
    for key, value in params.items():
        if value is None:
            cleaned[key] = None
        elif isinstance(value, (int, float, str, bool)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)):
            try:
                cleaned[key] = list(value)
            except:
                cleaned[key] = str(value)
        elif hasattr(value, 'tolist'):  # numpy arrays
            try:
                cleaned[key] = value.tolist()
            except:
                cleaned[key] = str(value)
        else:
            cleaned[key] = str(value)
    
    return cleaned


def main():
    """메인 실행 함수"""
    print("🔍 Final Models 하이퍼파라미터 추출 시작")
    
    # final_models 디렉토리 경로
    models_dir = Path("/Users/jojongho/KDT/P2_Default-invest/final_models")
    
    if not models_dir.exists():
        print(f"❌ 모델 디렉토리를 찾을 수 없습니다: {models_dir}")
        return
    
    # 모델 파일 목록
    model_files = list(models_dir.glob("*.joblib"))
    print(f"📁 발견된 모델 파일: {len(model_files)}개")
    
    # 하이퍼파라미터 추출
    all_hyperparameters = {}
    model_summary = defaultdict(list)
    
    for model_file in model_files:
        print(f"🔍 처리중: {model_file.name}")
        
        # 파일명에서 정보 추출
        filename = model_file.stem
        if "__" in filename:
            data_type, model_info = filename.split("__", 1)
            model_type = model_info.replace("_model", "")
        else:
            data_type = "unknown"
            model_type = filename.replace("_model", "")
        
        # 하이퍼파라미터 추출
        params = extract_model_hyperparameters(model_file)
        
        # JSON 직렬화 가능하도록 정리
        if 'hyperparameters' in params:
            params['hyperparameters'] = clean_hyperparameters(params['hyperparameters'])
        
        # 메타데이터 추가
        params['file_info'] = {
            'filename': model_file.name,
            'data_type': data_type,
            'model_name': model_type,
            'file_size_mb': round(model_file.stat().st_size / (1024 * 1024), 2)
        }
        
        # 결과 저장
        model_key = f"{data_type}__{model_type}"
        all_hyperparameters[model_key] = params
        
        # 요약 정보 수집
        model_summary[params['model_type']].append({
            'data_type': data_type,
            'model_name': model_type,
            'file_size_mb': params['file_info']['file_size_mb']
        })
        
        print(f"   ✅ {params['model_type']} 완료")
    
    # JSON 파일로 저장
    output_file = "final_models_hyperparameters.json"
    
    # 메타데이터 추가
    final_output = {
        'metadata': {
            'total_models': len(model_files),
            'extraction_date': pd.Timestamp.now().isoformat(),
            'models_directory': str(models_dir),
            'model_types': list(model_summary.keys())
        },
        'model_summary': dict(model_summary),
        'detailed_hyperparameters': all_hyperparameters
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 하이퍼파라미터 추출 완료!")
    print(f"📄 저장 파일: {output_file}")
    
    # 요약 출력
    print(f"\n📊 모델 요약:")
    for model_type, models in model_summary.items():
        print(f"  {model_type}: {len(models)}개")
        for model in models:
            print(f"    - {model['data_type']}__{model['model_name']} ({model['file_size_mb']}MB)")
    
    # 간단한 통계
    total_size = sum(model['file_size_mb'] for models in model_summary.values() for model in models)
    print(f"\n💾 총 모델 파일 크기: {total_size:.2f}MB")
    
    return output_file


if __name__ == "__main__":
    main()