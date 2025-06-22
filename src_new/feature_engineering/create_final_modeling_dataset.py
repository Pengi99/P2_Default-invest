import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
import warnings
warnings.filterwarnings('ignore')

print("=== 최종 모델링용 데이터셋 생성 (SMOTE 포함) ===")

# 1. 데이터 로드
print("\n1️⃣ 데이터 로드")
print("="*50)

# 라벨링된 데이터 로드
df = pd.read_csv('data_new/final/FS_ratio_flow_labeled.csv', dtype={'거래소코드': str})
print(f"라벨링된 데이터: {df.shape}")
print(f"부실(default=1): {(df['default']==1).sum():,}개 ({(df['default']==1).sum()/len(df)*100:.2f}%)")
print(f"정상(default=0): {(df['default']==0).sum():,}개 ({(df['default']==0).sum()/len(df)*100:.2f}%)")

# 2. 특성 및 타겟 분리
print("\n2️⃣ 특성 및 타겟 분리")
print("="*50)

# 메타데이터 컬럼
meta_columns = ['회사명', '거래소코드', '회계년도']

# 재무비율 컬럼 (특성)
feature_columns = [col for col in df.columns 
                  if col not in meta_columns + ['default']]

print(f"특성 컬럼: {len(feature_columns)}개")
print(f"특성 목록: {feature_columns}")

# 결측값 확인 및 처리
print(f"\n결측값 현황:")
missing_info = []
for col in feature_columns:
    missing_count = df[col].isna().sum()
    missing_pct = missing_count / len(df) * 100
    if missing_count > 0:
        missing_info.append((col, missing_count, missing_pct))
        print(f"  {col}: {missing_count:,}개 ({missing_pct:.2f}%)")
        # 중앙값으로 결측값 대체
        df[col] = df[col].fillna(df[col].median())

if not missing_info:
    print("  결측값 없음 ✅")
else:
    print("  ✅ 결측값을 중앙값으로 대체 완료")

# 3. 데이터 분할 (4:3:3 비율)
print("\n3️⃣ 데이터 분할 (Train:Valid:Test = 4:3:3)")
print("="*50)

# 특성과 타겟 분리
X = df[feature_columns].copy()
y = df['default'].copy()
meta_data = df[meta_columns].copy()

# 1차 분할: Train(40%) vs Temp(60%)
X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(
    X, y, meta_data, 
    test_size=0.6,  # 60% for Valid+Test
    random_state=42, 
    stratify=y
)

# 2차 분할: Temp(60%) -> Valid(30%) + Test(30%)
# Temp 중에서 Valid:Test = 1:1 비율이므로 test_size = 0.5
X_valid, X_test, y_valid, y_test, meta_valid, meta_test = train_test_split(
    X_temp, y_temp, meta_temp,
    test_size=0.5,  # Valid와 Test를 1:1로 분할
    random_state=42,
    stratify=y_temp
)

print(f"훈련 데이터: {len(X_train):,}개 ({len(X_train)/len(df)*100:.1f}%)")
print(f"  - 부실: {(y_train==1).sum():,}개 ({(y_train==1).sum()/len(y_train)*100:.2f}%)")
print(f"  - 정상: {(y_train==0).sum():,}개")

print(f"검증 데이터: {len(X_valid):,}개 ({len(X_valid)/len(df)*100:.1f}%)")
print(f"  - 부실: {(y_valid==1).sum():,}개 ({(y_valid==1).sum()/len(y_valid)*100:.2f}%)")
print(f"  - 정상: {(y_valid==0).sum():,}개")

print(f"테스트 데이터: {len(X_test):,}개 ({len(X_test)/len(df)*100:.1f}%)")
print(f"  - 부실: {(y_test==1).sum():,}개 ({(y_test==1).sum()/len(y_test)*100:.2f}%)")
print(f"  - 정상: {(y_test==0).sum():,}개")

# 4. SMOTE 적용 (훈련 데이터에만)
print("\n4️⃣ BorderlineSMOTE 적용")
print("="*50)

# BorderlineSMOTE 설정 (1:10 비율로 조정)
smote = BorderlineSMOTE(
    sampling_strategy=0.1,  # 부실:정상 = 1:10 비율
    random_state=42,
    k_neighbors=5,
    m_neighbors=10
)

print("SMOTE 적용 전 훈련 데이터:")
print(f"  - 부실: {(y_train==1).sum():,}개")
print(f"  - 정상: {(y_train==0).sum():,}개")
print(f"  - 비율: {(y_train==1).sum()/(y_train==0).sum():.4f}")

# SMOTE 적용
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nSMOTE 적용 후 훈련 데이터:")
print(f"  - 부실: {(y_train_smote==1).sum():,}개")
print(f"  - 정상: {(y_train_smote==0).sum():,}개")
print(f"  - 비율: {(y_train_smote==1).sum()/(y_train_smote==0).sum():.4f}")
print(f"  - 총 증가: {len(X_train_smote) - len(X_train):,}개 샘플")

# 5. 스케일링 적용
print("\n5️⃣ 스케일링 적용")
print("="*50)

# 스케일링 방법별 컬럼 분류 (이전 분석 결과 기반)
robust_scaler_columns = [
    'ROA', 'CFO_TD', 'RE_TA', 'EBIT_TA', 'MVE_TL', 'S_TA', 
    'CLCA', 'OENEG', 'CR', 'CFO_TA', 'RET_3M', 'RET_9M', 'MB'
]

standard_scaler_columns = [
    'TLTA', 'WC_TA', 'SIGMA', 'TLMTA'
]

# 실제 존재하는 컬럼만 필터링
robust_cols_available = [col for col in robust_scaler_columns if col in feature_columns]
standard_cols_available = [col for col in standard_scaler_columns if col in feature_columns]

print(f"RobustScaler 적용: {len(robust_cols_available)}개 컬럼")
print(f"StandardScaler 적용: {len(standard_cols_available)}개 컬럼")

def apply_scaling(X_train_data, X_valid_data, X_test_data, dataset_name=""):
    """스케일링을 적용하는 함수"""
    print(f"\n{dataset_name} 스케일링 적용 중...")
    
    X_train_scaled = X_train_data.copy()
    X_valid_scaled = X_valid_data.copy()
    X_test_scaled = X_test_data.copy()
    
    # RobustScaler 적용
    if robust_cols_available:
        robust_scaler = RobustScaler()
        robust_scaler.fit(X_train_data[robust_cols_available])
        
        X_train_scaled[robust_cols_available] = robust_scaler.transform(X_train_data[robust_cols_available])
        X_valid_scaled[robust_cols_available] = robust_scaler.transform(X_valid_data[robust_cols_available])
        X_test_scaled[robust_cols_available] = robust_scaler.transform(X_test_data[robust_cols_available])
        
        print(f"  ✅ RobustScaler 적용 완료 ({len(robust_cols_available)}개 컬럼)")
    
    # StandardScaler 적용
    if standard_cols_available:
        standard_scaler = StandardScaler()
        standard_scaler.fit(X_train_data[standard_cols_available])
        
        X_train_scaled[standard_cols_available] = standard_scaler.transform(X_train_data[standard_cols_available])
        X_valid_scaled[standard_cols_available] = standard_scaler.transform(X_valid_data[standard_cols_available])
        X_test_scaled[standard_cols_available] = standard_scaler.transform(X_test_data[standard_cols_available])
        
        print(f"  ✅ StandardScaler 적용 완료 ({len(standard_cols_available)}개 컬럼)")
    
    return X_train_scaled, X_valid_scaled, X_test_scaled

# 일반 버전 스케일링
X_train_scaled, X_valid_scaled, X_test_scaled = apply_scaling(
    X_train, X_valid, X_test, "일반 버전"
)

# SMOTE 버전 스케일링 (SMOTE 적용된 훈련 데이터 사용)
X_train_smote_scaled, X_valid_scaled_smote, X_test_scaled_smote = apply_scaling(
    X_train_smote, X_valid, X_test, "SMOTE 버전"
)

# 6. 데이터 저장
print("\n6️⃣ 최종 데이터 저장")
print("="*50)

# 일반 버전 저장
print("일반 버전 데이터 저장:")
X_train_scaled.to_csv('data_new/final/X_train_normal.csv', index=False, encoding='utf-8-sig')
X_valid_scaled.to_csv('data_new/final/X_valid_normal.csv', index=False, encoding='utf-8-sig')  
X_test_scaled.to_csv('data_new/final/X_test_normal.csv', index=False, encoding='utf-8-sig')

y_train.to_csv('data_new/final/y_train_normal.csv', index=False, encoding='utf-8-sig')
y_valid.to_csv('data_new/final/y_valid_normal.csv', index=False, encoding='utf-8-sig')
y_test.to_csv('data_new/final/y_test_normal.csv', index=False, encoding='utf-8-sig')

print("  ✅ 일반 버전 저장 완료")

# SMOTE 버전 저장
print("SMOTE 버전 데이터 저장:")
X_train_smote_scaled.to_csv('data_new/final/X_train_smote.csv', index=False, encoding='utf-8-sig')
X_valid_scaled_smote.to_csv('data_new/final/X_valid_smote.csv', index=False, encoding='utf-8-sig')
X_test_scaled_smote.to_csv('data_new/final/X_test_smote.csv', index=False, encoding='utf-8-sig')

pd.Series(y_train_smote).to_csv('data_new/final/y_train_smote.csv', index=False, encoding='utf-8-sig')
y_valid.to_csv('data_new/final/y_valid_smote.csv', index=False, encoding='utf-8-sig')
y_test.to_csv('data_new/final/y_test_smote.csv', index=False, encoding='utf-8-sig')

print("  ✅ SMOTE 버전 저장 완료")

# 메타데이터도 저장
meta_train.to_csv('data_new/final/meta_train.csv', index=False, encoding='utf-8-sig')
meta_valid.to_csv('data_new/final/meta_valid.csv', index=False, encoding='utf-8-sig')
meta_test.to_csv('data_new/final/meta_test.csv', index=False, encoding='utf-8-sig')

print("  ✅ 메타데이터 저장 완료")

# 7. 데이터셋 요약 정보 저장
print("\n7️⃣ 데이터셋 요약 정보 생성")
print("="*50)

summary_info = {
    'dataset_info': {
        'original_samples': len(df),
        'total_features': len(feature_columns),
        'original_default_rate': (df['default']==1).sum() / len(df),
        'split_ratio': [4, 3, 3],  # train:valid:test
        'split_method': 'stratified_random'
    },
    'normal_version': {
        'train_samples': len(X_train),
        'valid_samples': len(X_valid),
        'test_samples': len(X_test),
        'train_default_rate': (y_train==1).sum() / len(y_train),
        'valid_default_rate': (y_valid==1).sum() / len(y_valid),
        'test_default_rate': (y_test==1).sum() / len(y_test)
    },
    'smote_version': {
        'train_samples': len(X_train_smote),
        'valid_samples': len(X_valid),  # 검증/테스트는 동일
        'test_samples': len(X_test),
        'train_default_rate': (y_train_smote==1).sum() / len(y_train_smote),
        'smote_method': 'BorderlineSMOTE',
        'samples_added': len(X_train_smote) - len(X_train)
    },
    'feature_info': {
        'feature_columns': feature_columns,
        'robust_scaled': robust_cols_available,
        'standard_scaled': standard_cols_available,
        'meta_columns': meta_columns
    },
    'scaling_info': {
        'robust_scaler_features': len(robust_cols_available),
        'standard_scaler_features': len(standard_cols_available),
        'scaling_fit_on': 'train_data_only'
    }
}

import json
with open('data_new/final/dataset_info_final.json', 'w', encoding='utf-8') as f:
    json.dump(summary_info, f, ensure_ascii=False, indent=2)

print(f"✅ 데이터셋 정보: data_new/final/dataset_info_final.json")

# 8. 최종 요약
print("\n8️⃣ 최종 요약")
print("="*50)

print(f"📊 생성된 파일들:")
print(f"\n🔹 일반 버전 (Original):")
print(f"  - X_train_normal.csv: {len(X_train):,}개 샘플")
print(f"  - X_valid_normal.csv: {len(X_valid):,}개 샘플")
print(f"  - X_test_normal.csv: {len(X_test):,}개 샘플")
print(f"  - y_train_normal.csv, y_valid_normal.csv, y_test_normal.csv")

print(f"\n🔹 SMOTE 버전 (Balanced):")
print(f"  - X_train_smote.csv: {len(X_train_smote):,}개 샘플 (+{len(X_train_smote)-len(X_train):,})")
print(f"  - X_valid_smote.csv: {len(X_valid):,}개 샘플 (동일)")
print(f"  - X_test_smote.csv: {len(X_test):,}개 샘플 (동일)")
print(f"  - y_train_smote.csv, y_valid_smote.csv, y_test_smote.csv")

print(f"\n🔹 메타데이터:")
print(f"  - meta_train.csv, meta_valid.csv, meta_test.csv")
print(f"  - dataset_info_final.json")

print(f"\n🎯 데이터 특성:")
print(f"  - 총 특성: {len(feature_columns)}개")
print(f"  - 분할 비율: Train {len(X_train)/len(df)*100:.1f}% : Valid {len(X_valid)/len(df)*100:.1f}% : Test {len(X_test)/len(df)*100:.1f}%")
print(f"  - 원본 부실 비율: {(df['default']==1).sum()/len(df)*100:.2f}%")
print(f"  - SMOTE 후 훈련 부실 비율: {(y_train_smote==1).sum()/len(y_train_smote)*100:.2f}%")
print(f"  - 스케일링: RobustScaler({len(robust_cols_available)}개), StandardScaler({len(standard_cols_available)}개)")

print(f"\n🚀 다음 단계:")
print(f"  1. 일반 버전으로 베이스라인 모델 훈련")
print(f"  2. SMOTE 버전으로 성능 향상 모델 훈련")
print(f"  3. 모델별 성능 비교 (Logistic Regression, Random Forest, XGBoost)")
print(f"  4. 하이퍼파라미터 튜닝")
print(f"  5. 최종 모델 선택 및 해석")

print(f"\n✅ 최종 모델링용 데이터셋 생성 완료!")
print(f"💡 SMOTE는 훈련 데이터에만 적용되어 데이터 누수를 방지했습니다.") 