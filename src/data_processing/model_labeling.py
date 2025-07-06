import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Optional, Tuple

def load_features_data(features_path: str) -> pd.DataFrame:
    """
    피처 데이터를 로드합니다.
    
    Args:
        features_path: FS2_features.csv 파일 경로
        
    Returns:
        로드된 피처 데이터프레임
    """
    try:
        df = pd.read_csv(features_path, encoding='utf-8')
        print(f"✅ 피처 데이터 로드 완료: {df.shape[0]:,} 행 × {df.shape[1]:,} 열")
        return df
    except Exception as e:
        print(f"❌ 피처 데이터 로드 실패: {e}")
        raise

def load_selected_features(selection_path: str) -> List[str]:
    """
    특성 선택 결과를 로드합니다.
    
    Args:
        selection_path: logistic_regression_cv_selection.json 파일 경로
        
    Returns:
        선택된 특성 리스트
    """
    try:
        with open(selection_path, 'r', encoding='utf-8') as f:
            selection_data = json.load(f)

        # 'selected_feature_names' 키가 존재하면 해당 리스트를 우선 사용합니다.
        if 'selected_feature_names' in selection_data and isinstance(selection_data['selected_feature_names'], list):
            selected_features = selection_data['selected_feature_names']
        else:
            # 기존 호환성을 위해 'selected_features' 키도 확인합니다.
            cand = selection_data.get('selected_features', [])

            # 'selected_features'가 정수(특성 개수)로만 저장된 경우 예외 처리
            if isinstance(cand, list):
                selected_features = cand
            else:
                raise ValueError(
                    "'selected_feature_names' 리스트를 찾을 수 없으며, 'selected_features' 또한 리스트 형식이 아닙니다.\n"
                    f"선택된 특성 정보를 확인하세요: {cand}"
                )

        print(f"✅ 선택된 특성 로드 완료: {selected_features}")

        return selected_features
    except Exception as e:
        print(f"❌ 특성 선택 결과 로드 실패: {e}")
        raise

def find_best_model(summary_path: str) -> Tuple[str, float]:
    """
    F1 스코어가 가장 높은 모델을 찾습니다.
    
    Args:
        summary_path: summary_table.csv 파일 경로
        
    Returns:
        (최고 모델명, F1 스코어) 튜플
    """
    try:
        summary_df = pd.read_csv(summary_path, encoding='utf-8')
        
        # F1 스코어 컬럼 찾기 (다양한 가능한 컬럼명 고려)
        f1_columns = [col for col in summary_df.columns if 'f1' in col.lower()]
        if not f1_columns:
            raise ValueError("F1 스코어 컬럼을 찾을 수 없습니다.")
        
        # Test_F1 컬럼을 우선적으로 찾기
        test_f1_col = None
        for col in f1_columns:
            if 'test_f1' in col.lower():
                test_f1_col = col
                break
        
        f1_col = test_f1_col
        
        # F1 스코어가 가장 높은 행 찾기
        best_idx = summary_df[f1_col].idxmax()
        
        # Model과 Data_Type 컬럼에서 정보 추출
        best_model = summary_df.loc[best_idx, 'Model']
        best_data_type = summary_df.loc[best_idx, 'Data_Type']
        best_f1 = summary_df.loc[best_idx, f1_col]
        
        # 모델 파일명 형식: Data_Type__Model_model.joblib
        model_filename = f"{best_data_type}__{best_model}_model"
        
        print(f"✅ 최고 성능 모델: {best_model} ({best_data_type}) (F1 Score: {best_f1:.4f})")
        print(f"   모델 파일명: {model_filename}.joblib")
        
        return str(model_filename), float(best_f1)
    except Exception as e:
        print(f"❌ 최고 모델 찾기 실패: {e}")
        raise

def load_best_model(models_dir: str, model_name: str) -> object:
    """
    최고 성능 모델을 로드합니다.
    
    Args:
        models_dir: 모델이 저장된 디렉토리 경로
        model_name: 모델명
        
    Returns:
        로드된 모델 객체
    """
    try:
        models_path = Path(models_dir)
        
        # 가능한 모델 파일명 패턴들
        possible_patterns = [
            f"{model_name}.joblib",
            f"{model_name}_model.joblib",
            f"best_{model_name}.joblib",
            f"{model_name}_final.joblib"
        ]
        
        model_file = None
        for pattern in possible_patterns:
            candidate = models_path / pattern
            if candidate.exists():
                model_file = candidate
                break
        
        # 패턴으로 찾지 못한 경우 디렉토리 내 모든 joblib 파일 확인
        if model_file is None:
            joblib_files = list(models_path.glob("*.joblib"))
            for file in joblib_files:
                if model_name.lower() in file.name.lower():
                    model_file = file
                    break
        
        if model_file is None:
            available_files = list(models_path.glob("*.joblib"))
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {model_name}\n"
                f"사용 가능한 파일들: {[f.name for f in available_files]}"
            )
        
        model = joblib.load(model_file)
        print(f"✅ 모델 로드 완료: {model_file.name}")
        
        return model
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        raise

def predict_default(df: pd.DataFrame, selected_features: List[str], model: object) -> pd.DataFrame:
    """
    부도 예측을 수행하고 결과를 데이터프레임에 추가합니다.
    
    Args:
        df: 원본 데이터프레임
        selected_features: 선택된 특성 리스트
        model: 학습된 모델
        
    Returns:
        'default' 컬럼이 추가된 데이터프레임
    """
    try:
        # 선택된 특성들이 데이터에 존재하는지 확인
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            print(f"⚠️ 누락된 특성들: {missing_features}")
            # 누락된 특성들을 제외하고 진행
            available_features = [f for f in selected_features if f in df.columns]
            print(f"사용 가능한 특성: {len(available_features)}/{len(selected_features)}개")
        else:
            available_features = selected_features
        
        # 예측용 데이터 준비
        X_pred = df[available_features].copy()
        
        # 결측치 처리 (간단한 전진 채우기)
        X_pred = X_pred.fillna(method='ffill').fillna(0)
        
        # 예측 수행
        predictions = model.predict(X_pred)
        prediction_proba = model.predict_proba(X_pred)[:, 1]  # 부도 확률
        
        # 결과를 원본 데이터프레임에 추가
        result_df = df.copy()
        result_df['default'] = predictions
        result_df['default_probability'] = prediction_proba
        
        # 예측 결과 요약
        default_count = (predictions == 1).sum()
        default_rate = default_count / len(predictions) * 100
        
        print(f"✅ 예측 완료:")
        print(f"   - 총 예측 건수: {len(predictions):,}건")
        print(f"   - 부도 예측: {default_count:,}건 ({default_rate:.2f}%)")
        print(f"   - 정상 예측: {len(predictions) - default_count:,}건 ({100-default_rate:.2f}%)")
        print(f"   - 평균 부도 확률: {prediction_proba.mean():.4f}")
        
        return result_df
        
    except Exception as e:
        print(f"❌ 예측 수행 실패: {e}")
        raise

def save_labeled_data(df: pd.DataFrame, output_path: str) -> None:
    """
    라벨링된 데이터를 저장합니다.
    
    Args:
        df: 라벨링된 데이터프레임
        output_path: 출력 파일 경로
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ 라벨링된 데이터 저장 완료: {output_path}")
        
    except Exception as e:
        print(f"❌ 데이터 저장 실패: {e}")
        raise

def main():
    """
    메인 실행 함수
    """
    parser = argparse.ArgumentParser(description='모델을 사용한 부도 예측 라벨링')
    parser.add_argument('--features_path', type=str, 
                       default='data/processed/FS2_features.csv',
                       help='피처 데이터 파일 경로')
    parser.add_argument('--model_run_dir', type=str,
                        default='outputs/modeling_runs/default_modeling_run_20250706_131209',
                       help='모델링 실행 결과 디렉토리 경로')
    parser.add_argument('--output_path', type=str,
                       default='data/processed/FS2_ourmodel_labeled.csv',
                       help='라벨링된 데이터 출력 경로')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("부도 예측 모델 라벨링 파이프라인")
    print("=" * 60)
    
    try:
        # 1. 피처 데이터 로드
        df = load_features_data(args.features_path)
        
        # 2. 모델링 결과 디렉토리 경로 설정
        model_run_path = Path(args.model_run_dir)
        selection_path = model_run_path / "results" / "logistic_regression_cv_selection.json"
        summary_path = model_run_path / "results" / "summary_table.csv"
        models_dir = model_run_path / "models"
        
        # 경로 존재 확인
        for path in [selection_path, summary_path, models_dir]:
            if not path.exists():
                raise FileNotFoundError(f"필요한 파일/디렉토리가 존재하지 않습니다: {path}")
        
        # 3. 선택된 특성 로드
        selected_features = load_selected_features(str(selection_path))
        
        # 4. 최고 성능 모델 찾기
        best_model_name, best_f1 = find_best_model(str(summary_path))
        
        # 5. 최고 성능 모델 로드
        model = load_best_model(str(models_dir), best_model_name)
        
        # 6. 부도 예측 수행
        labeled_df = predict_default(df, selected_features, model)
        
        # 7. 결과 저장
        save_labeled_data(labeled_df, args.output_path)
        
        print("=" * 60)
        print("부도 예측 라벨링 완료!")
        print(f"사용된 모델: {best_model_name} (F1: {best_f1:.4f})")
        print(f"출력 파일: {args.output_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 파이프라인 실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
