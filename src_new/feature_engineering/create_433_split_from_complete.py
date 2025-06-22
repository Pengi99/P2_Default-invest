#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FS_100_complete.csv를 기준으로 4:3:3 비율 데이터 분할
기존 파일들을 덮어쓰기
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🏢 FS_100_complete.csv 기준 4:3:3 분할 및 덮어쓰기")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 데이터 로드
    print("📂 FS_100_complete.csv 로드")
    print("="*60)
    
    df = pd.read_csv('data_new/final/FS_100_complete.csv')
    print(f"✅ 데이터 로드 완료: {df.shape}")
    print(f"   총 샘플: {len(df):,}개")
    print(f"   부실기업: {df['default'].sum()}개 ({df['default'].mean():.4f})")
    print(f"   정상기업: {(~df['default'].astype(bool)).sum()}개")
    
    # 2. 특성과 타겟 분리
    print("\n🔍 특성과 타겟 분리")
    print("="*60)
    
    # 메타데이터 컬럼
    meta_columns = ['회사명', '거래소코드', '회계년도']
    
    # 특성 컬럼 (K2_Score_Original 제외)
    feature_columns = [col for col in df.columns 
                      if col not in meta_columns + ['default', 'K2_Score_Original']]
    
    print(f"특성 컬럼: {len(feature_columns)}개")
    print(f"특성 목록: {feature_columns}")
    
    X = df[feature_columns].copy()
    y = df['default'].copy()
    meta_data = df[meta_columns].copy()
    
    # 결측값 확인
    missing_check = X.isnull().sum()
    if missing_check.sum() > 0:
        print(f"\n⚠️ 결측값 발견:")
        for col, count in missing_check[missing_check > 0].items():
            print(f"   {col}: {count}개")
        print("❌ FS_100_complete.csv에는 결측값이 없어야 합니다!")
        return
    else:
        print("✅ 결측값 없음 확인")
    
    # 3. 데이터 분할 (4:3:3)
    print("\n📊 데이터 분할 (Train:Valid:Test = 4:3:3)")
    print("="*60)
    
    # 1차 분할: Train(40%) vs Temp(60%)
    X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(
        X, y, meta_data,
        test_size=0.6,  # 60% for Valid+Test
        random_state=42,
        stratify=y
    )
    
    # 2차 분할: Temp(60%) -> Valid(30%) + Test(30%)
    X_valid, X_test, y_valid, y_test, meta_valid, meta_test = train_test_split(
        X_temp, y_temp, meta_temp,
        test_size=0.5,  # Valid와 Test를 1:1로 분할
        random_state=42,
        stratify=y_temp
    )
    
    print(f"훈련 데이터: {len(X_train):,}개 ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  - 부실: {y_train.sum()}개 ({y_train.mean():.4f})")
    print(f"  - 정상: {(~y_train.astype(bool)).sum()}개")
    
    print(f"검증 데이터: {len(X_valid):,}개 ({len(X_valid)/len(df)*100:.1f}%)")
    print(f"  - 부실: {y_valid.sum()}개 ({y_valid.mean():.4f})")
    print(f"  - 정상: {(~y_valid.astype(bool)).sum()}개")
    
    print(f"테스트 데이터: {len(X_test):,}개 ({len(X_test)/len(df)*100:.1f}%)")
    print(f"  - 부실: {y_test.sum()}개 ({y_test.mean():.4f})")
    print(f"  - 정상: {(~y_test.astype(bool)).sum()}개")
    
    # 4. 스케일링 적용
    print("\n⚖️ 스케일링 적용")
    print("="*60)
    
    # StandardScaler 적용
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_valid_scaled = pd.DataFrame(
        scaler.transform(X_valid),
        columns=X_valid.columns,
        index=X_valid.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("✅ StandardScaler 적용 완료")
    
    # 5. SMOTE 적용 (훈련 데이터에만)
    print("\n🔄 SMOTE 적용")
    print("="*60)
    
    # BorderlineSMOTE 설정
    smote = BorderlineSMOTE(
        sampling_strategy=0.1,  # 부실:정상 = 1:10 비율
        random_state=42,
        k_neighbors=5,
        m_neighbors=10
    )
    
    print("SMOTE 적용 전:")
    print(f"  - 부실: {y_train.sum()}개")
    print(f"  - 정상: {(~y_train.astype(bool)).sum()}개")
    print(f"  - 비율: {y_train.sum()/(~y_train.astype(bool)).sum():.4f}")
    
    # SMOTE 적용
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    print("\nSMOTE 적용 후:")
    print(f"  - 부실: {y_train_smote.sum()}개")
    print(f"  - 정상: {(~y_train_smote.astype(bool)).sum()}개")
    print(f"  - 비율: {y_train_smote.sum()/(~y_train_smote.astype(bool)).sum():.4f}")
    print(f"  - 총 증가: {len(X_train_smote) - len(X_train_scaled):,}개 샘플")
    
    # 6. 파일 저장 (기존 파일 덮어쓰기)
    print("\n💾 파일 저장 (기존 파일 덮어쓰기)")
    print("="*60)
    
    # Normal 버전 저장
    print("Normal 버전 저장:")
    X_train_scaled.to_csv('data_new/final/X_train_100_normal.csv', index=False)
    X_valid_scaled.to_csv('data_new/final/X_valid_100_normal.csv', index=False)
    X_test_scaled.to_csv('data_new/final/X_test_100_normal.csv', index=False)
    
    y_train.to_csv('data_new/final/y_train_100_normal.csv', index=False)
    y_valid.to_csv('data_new/final/y_valid_100_normal.csv', index=False)
    y_test.to_csv('data_new/final/y_test_100_normal.csv', index=False)
    print("  ✅ Normal 버전 저장 완료")
    
    # SMOTE 버전 저장
    print("SMOTE 버전 저장:")
    pd.DataFrame(X_train_smote, columns=feature_columns).to_csv('data_new/final/X_train_100_smote.csv', index=False)
    X_valid_scaled.to_csv('data_new/final/X_valid_100_smote.csv', index=False)
    X_test_scaled.to_csv('data_new/final/X_test_100_smote.csv', index=False)
    
    pd.Series(y_train_smote).to_csv('data_new/final/y_train_100_smote.csv', index=False)
    y_valid.to_csv('data_new/final/y_valid_100_smote.csv', index=False)
    y_test.to_csv('data_new/final/y_test_100_smote.csv', index=False)
    print("  ✅ SMOTE 버전 저장 완료")
    
    # 메타데이터 저장
    print("메타데이터 저장:")
    meta_train.to_csv('data_new/final/meta_train_100.csv', index=False)
    meta_valid.to_csv('data_new/final/meta_valid_100.csv', index=False)
    meta_test.to_csv('data_new/final/meta_test_100.csv', index=False)
    print("  ✅ 메타데이터 저장 완료")
    
    # 7. 정보 파일 업데이트
    print("\n📋 정보 파일 업데이트")
    print("="*60)
    
    dataset_info = {
        "dataset_name": "100% Complete Financial Data (4:3:3 Split)",
        "creation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "original_data": {
            "total_samples": len(df),
            "features": len(feature_columns),
            "default_count": int(df['default'].sum()),
            "default_ratio": float(df['default'].mean())
        },
        "data_split": {
            "train_samples": len(X_train),
            "valid_samples": len(X_valid),
            "test_samples": len(X_test),
            "train_ratio": 0.4,
            "valid_ratio": 0.3,
            "test_ratio": 0.3
        },
        "smote_applied": {
            "original_train_samples": len(X_train),
            "smote_train_samples": len(X_train_smote),
            "target_ratio": 0.1,
            "smote_type": "BorderlineSMOTE"
        },
        "features": feature_columns,
        "scaling": "StandardScaler",
        "files": {
            "normal": {
                "X_train": "X_train_100_normal.csv",
                "X_valid": "X_valid_100_normal.csv", 
                "X_test": "X_test_100_normal.csv",
                "y_train": "y_train_100_normal.csv",
                "y_valid": "y_valid_100_normal.csv",
                "y_test": "y_test_100_normal.csv"
            },
            "smote": {
                "X_train": "X_train_100_smote.csv",
                "X_valid": "X_valid_100_smote.csv",
                "X_test": "X_test_100_smote.csv", 
                "y_train": "y_train_100_smote.csv",
                "y_valid": "y_valid_100_smote.csv",
                "y_test": "y_test_100_smote.csv"
            },
            "meta": {
                "meta_train": "meta_train_100.csv",
                "meta_valid": "meta_valid_100.csv",
                "meta_test": "meta_test_100.csv"
            }
        }
    }
    
    with open('data_new/final/dataset_info_100_complete.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print("✅ dataset_info_100_complete.json 업데이트 완료")
    
    # 8. 최종 요약
    print("\n🎉 작업 완료!")
    print("="*80)
    print()
    print("📊 새로운 4:3:3 분할 결과:")
    print(f"  - Train: {len(X_train):,}개 (40%) | 부실: {y_train.sum()}개")
    print(f"  - Valid: {len(X_valid):,}개 (30%) | 부실: {y_valid.sum()}개") 
    print(f"  - Test:  {len(X_test):,}개  (30%) | 부실: {y_test.sum()}개")
    print()
    print("🔄 SMOTE 적용:")
    print(f"  - Train: {len(X_train_smote):,}개 | 부실: {y_train_smote.sum()}개 (10%)")
    print()
    print("✅ 모든 기존 파일이 새로운 4:3:3 분할 데이터로 덮어써졌습니다!")
    print("🚀 이제 기존 모델링 코드를 그대로 사용할 수 있습니다.")

if __name__ == "__main__":
    main() 