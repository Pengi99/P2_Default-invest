"""
하이퍼파라미터 요약 정리기
=========================

JSON 파일을 읽어서 더 보기 좋게 요약 정리하고,
모델별 주요 하이퍼파라미터를 비교 테이블로 생성합니다.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def clean_nan_values(obj):
    """NaN 값들을 None으로 변경"""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj


def create_model_comparison_table(hyperparams_data):
    """모델별 주요 하이퍼파라미터 비교 테이블 생성"""
    
    # 모델별 데이터 수집
    models_data = []
    
    for model_key, model_info in hyperparams_data['detailed_hyperparameters'].items():
        data_type = model_info['file_info']['data_type']
        model_type = model_info['model_type']
        hyperparams = model_info['hyperparameters']
        
        row = {
            'Model_Key': model_key,
            'Model_Type': model_type,
            'Data_Type': data_type,
            'File_Size_MB': model_info['file_info']['file_size_mb']
        }
        
        # 모델별 주요 하이퍼파라미터 추출
        if model_type == 'logistic_regression':
            row.update({
                'C': hyperparams.get('C'),
                'penalty': hyperparams.get('penalty'),
                'solver': hyperparams.get('solver'),
                'max_iter': hyperparams.get('max_iter'),
                'class_weight': hyperparams.get('class_weight')
            })
            
        elif model_type == 'random_forest':
            row.update({
                'n_estimators': hyperparams.get('n_estimators'),
                'max_depth': hyperparams.get('max_depth'),
                'min_samples_split': hyperparams.get('min_samples_split'),
                'min_samples_leaf': hyperparams.get('min_samples_leaf'),
                'max_features': hyperparams.get('max_features'),
                'class_weight': hyperparams.get('class_weight')
            })
            
        elif model_type == 'xgboost':
            row.update({
                'n_estimators': hyperparams.get('n_estimators'),
                'max_depth': hyperparams.get('max_depth'),
                'learning_rate': hyperparams.get('learning_rate'),
                'subsample': hyperparams.get('subsample'),
                'colsample_bytree': hyperparams.get('colsample_bytree'),
                'reg_alpha': hyperparams.get('reg_alpha'),
                'reg_lambda': hyperparams.get('reg_lambda'),
                'scale_pos_weight': hyperparams.get('scale_pos_weight')
            })
        
        # 특성 개수 정보
        if 'n_features' in model_info:
            row['n_features'] = model_info['n_features']
            
        models_data.append(row)
    
    return pd.DataFrame(models_data)


def create_summary_by_model_type(df):
    """모델 타입별 요약 통계"""
    summary = {}
    
    for model_type in df['Model_Type'].unique():
        if model_type == 'unknown':  # 앙상블 모델 제외
            continue
            
        model_df = df[df['Model_Type'] == model_type]
        
        if model_type == 'logistic_regression':
            summary[model_type] = {
                'count': len(model_df),
                'avg_file_size_mb': model_df['File_Size_MB'].mean(),
                'C_range': f"{model_df['C'].min():.4f} - {model_df['C'].max():.4f}",
                'penalties': model_df['penalty'].unique().tolist(),
                'solvers': model_df['solver'].unique().tolist(),
                'max_iter_range': f"{model_df['max_iter'].min()} - {model_df['max_iter'].max()}"
            }
            
        elif model_type == 'random_forest':
            summary[model_type] = {
                'count': len(model_df),
                'avg_file_size_mb': model_df['File_Size_MB'].mean(),
                'n_estimators_range': f"{model_df['n_estimators'].min()} - {model_df['n_estimators'].max()}",
                'max_depth_range': f"{model_df['max_depth'].min()} - {model_df['max_depth'].max()}",
                'max_features_range': f"{model_df['max_features'].min():.3f} - {model_df['max_features'].max():.3f}",
                'min_samples_split_range': f"{model_df['min_samples_split'].min()} - {model_df['min_samples_split'].max()}",
                'min_samples_leaf_range': f"{model_df['min_samples_leaf'].min()} - {model_df['min_samples_leaf'].max()}"
            }
            
        elif model_type == 'xgboost':
            summary[model_type] = {
                'count': len(model_df),
                'avg_file_size_mb': model_df['File_Size_MB'].mean(),
                'n_estimators_range': f"{model_df['n_estimators'].min()} - {model_df['n_estimators'].max()}",
                'max_depth_range': f"{model_df['max_depth'].min()} - {model_df['max_depth'].max()}",
                'learning_rate_range': f"{model_df['learning_rate'].min():.4f} - {model_df['learning_rate'].max():.4f}",
                'scale_pos_weight_range': f"{model_df['scale_pos_weight'].min():.1f} - {model_df['scale_pos_weight'].max():.1f}",
                'reg_alpha_range': f"{model_df['reg_alpha'].min():.2f} - {model_df['reg_alpha'].max():.2f}",
                'reg_lambda_range': f"{model_df['reg_lambda'].min():.2f} - {model_df['reg_lambda'].max():.2f}"
            }
    
    return summary


def main():
    """메인 실행 함수"""
    print("📊 하이퍼파라미터 요약 정리 시작")
    
    # JSON 파일 읽기
    json_file = "final_models_hyperparameters.json"
    
    if not Path(json_file).exists():
        print(f"❌ JSON 파일을 찾을 수 없습니다: {json_file}")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        hyperparams_data = json.load(f)
    
    # NaN 값 정리
    hyperparams_data = clean_nan_values(hyperparams_data)
    
    # 정리된 JSON 파일 다시 저장
    with open("final_models_hyperparameters_cleaned.json", 'w', encoding='utf-8') as f:
        json.dump(hyperparams_data, f, indent=2, ensure_ascii=False)
    
    print("✅ NaN 값 정리 완료: final_models_hyperparameters_cleaned.json")
    
    # 비교 테이블 생성
    comparison_df = create_model_comparison_table(hyperparams_data)
    
    # CSV 파일로 저장
    comparison_df.to_csv("model_hyperparameters_comparison.csv", index=False, encoding='utf-8-sig')
    print("✅ 비교 테이블 저장: model_hyperparameters_comparison.csv")
    
    # 모델 타입별 요약
    summary = create_summary_by_model_type(comparison_df)
    
    # 요약 JSON 저장
    with open("model_hyperparameters_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("✅ 요약 통계 저장: model_hyperparameters_summary.json")
    
    # 콘솔 출력
    print("\n📋 모델별 하이퍼파라미터 요약:")
    print("="*60)
    
    for model_type, stats in summary.items():
        print(f"\n🔹 {model_type.upper()} ({stats['count']}개 모델)")
        print(f"   평균 파일 크기: {stats['avg_file_size_mb']:.2f}MB")
        
        for key, value in stats.items():
            if key not in ['count', 'avg_file_size_mb']:
                print(f"   {key}: {value}")
    
    # 데이터 타입별 분포
    print(f"\n📊 데이터 타입별 모델 분포:")
    data_type_counts = comparison_df['Data_Type'].value_counts()
    for data_type, count in data_type_counts.items():
        print(f"   {data_type}: {count}개")
    
    # 특성 개수 분포
    print(f"\n🔍 사용된 특성 개수:")
    feature_counts = comparison_df['n_features'].value_counts().sort_index()
    for n_features, count in feature_counts.items():
        print(f"   {n_features}개 특성: {count}개 모델")
    
    print(f"\n💾 총 모델 파일 크기: {comparison_df['File_Size_MB'].sum():.2f}MB")
    
    return comparison_df, summary


if __name__ == "__main__":
    main()