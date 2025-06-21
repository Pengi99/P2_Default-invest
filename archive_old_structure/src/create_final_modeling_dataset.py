import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=== 최종 모델링용 데이터셋 생성 ===")

# 1. 데이터 로드
print("\n1️⃣ 데이터 로드")
print("="*50)

# 라벨링된 데이터 로드
df = pd.read_csv('data/processed/FS_ratio_flow_labeled.csv', dtype={'거래소코드': str})
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

# 결측값 확인
print(f"\n결측값 현황:")
missing_info = []
for col in feature_columns:
    missing_count = df[col].isna().sum()
    missing_pct = missing_count / len(df) * 100
    if missing_count > 0:
        missing_info.append((col, missing_count, missing_pct))
        print(f"  {col}: {missing_count:,}개 ({missing_pct:.2f}%)")

if not missing_info:
    print("  결측값 없음 ✅")

# 3. 스케일링 적용
print("\n3️⃣ 스케일링 적용")
print("="*50)

# 스케일링 방법별 컬럼 분류 (분석 결과 기반)
robust_scaler_columns = [
    'ROA', 'CFO_TD', 'RE_TA', 'EBIT_TA', 'MVE_TL', 'S_TA', 
    'CLCA', 'OENEG', 'CR', 'CFO_TA', 'RET_3M', 'RET_9M', 'MB'
]

standard_scaler_columns = [
    'TLTA', 'WC_TA', 'SIGMA', 'TLMTA'
]

print(f"RobustScaler 적용: {len(robust_scaler_columns)}개 컬럼")
print(f"StandardScaler 적용: {len(standard_scaler_columns)}개 컬럼")

# 스케일링 전 데이터 복사
df_scaled = df.copy()

# RobustScaler 적용
robust_scaler = RobustScaler()
robust_cols_available = [col for col in robust_scaler_columns if col in df.columns]

if robust_cols_available:
    print(f"\nRobustScaler 적용 중... ({len(robust_cols_available)}개 컬럼)")
    
    # 결측값이 없는 데이터만으로 스케일러 훈련
    robust_data = df[robust_cols_available].dropna()
    robust_scaler.fit(robust_data)
    
    # 전체 데이터에 적용 (결측값은 그대로 유지)
    for col in robust_cols_available:
        non_null_mask = df[col].notna()
        if non_null_mask.sum() > 0:
            df_scaled.loc[non_null_mask, col] = robust_scaler.fit_transform(
                df.loc[non_null_mask, [col]]
            ).flatten()
    
    print(f"  ✅ RobustScaler 적용 완료")

# StandardScaler 적용
standard_scaler = StandardScaler()
standard_cols_available = [col for col in standard_scaler_columns if col in df.columns]

if standard_cols_available:
    print(f"\nStandardScaler 적용 중... ({len(standard_cols_available)}개 컬럼)")
    
    # 결측값이 없는 데이터만으로 스케일러 훈련
    standard_data = df[standard_cols_available].dropna()
    standard_scaler.fit(standard_data)
    
    # 전체 데이터에 적용
    for col in standard_cols_available:
        non_null_mask = df[col].notna()
        if non_null_mask.sum() > 0:
            df_scaled.loc[non_null_mask, col] = standard_scaler.fit_transform(
                df.loc[non_null_mask, [col]]
            ).flatten()
    
    print(f"  ✅ StandardScaler 적용 완료")

# 4. 데이터 분할 (시계열 특성 고려)
print("\n4️⃣ 데이터 분할 (시계열 고려)")
print("="*50)

# 회계년도에서 연도 추출
df_scaled['year'] = pd.to_datetime(df_scaled['회계년도'], format='%Y/%m').dt.year

# 시계열 기반 분할 (Look-ahead Bias 방지)
# 2012-2019: 훈련용 (8년)
# 2020-2021: 검증용 (2년) 
# 2022-2023: 테스트용 (2년)

train_years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
val_years = [2020, 2021]
test_years = [2022, 2023]

train_mask = df_scaled['year'].isin(train_years)
val_mask = df_scaled['year'].isin(val_years)
test_mask = df_scaled['year'].isin(test_years)

train_df = df_scaled[train_mask].copy()
val_df = df_scaled[val_mask].copy()
test_df = df_scaled[test_mask].copy()

print(f"훈련 데이터 ({min(train_years)}-{max(train_years)}): {len(train_df):,}개")
print(f"  - 부실: {(train_df['default']==1).sum():,}개 ({(train_df['default']==1).sum()/len(train_df)*100:.2f}%)")
print(f"  - 정상: {(train_df['default']==0).sum():,}개")

print(f"검증 데이터 ({min(val_years)}-{max(val_years)}): {len(val_df):,}개")
print(f"  - 부실: {(val_df['default']==1).sum():,}개 ({(val_df['default']==1).sum()/len(val_df)*100:.2f}%)")
print(f"  - 정상: {(val_df['default']==0).sum():,}개")

print(f"테스트 데이터 ({min(test_years)}-{max(test_years)}): {len(test_df):,}개")
print(f"  - 부실: {(test_df['default']==1).sum():,}개 ({(test_df['default']==1).sum()/len(test_df)*100:.2f}%)")
print(f"  - 정상: {(test_df['default']==0).sum():,}개")

# 5. 최종 특성 행렬 및 타겟 벡터 생성
print("\n5️⃣ 최종 특성 행렬 및 타겟 벡터 생성")
print("="*50)

def create_feature_target(data, feature_cols):
    """특성 행렬과 타겟 벡터를 생성하고 결측값을 처리합니다."""
    X = data[feature_cols].copy()
    y = data['default'].copy()
    
    # 결측값 처리 (중앙값으로 대체)
    for col in feature_cols:
        if X[col].isna().sum() > 0:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"  {col}: 결측값 {X[col].isna().sum()}개를 중앙값({median_val:.4f})으로 대체")
    
    return X, y

# 훈련 데이터
X_train, y_train = create_feature_target(train_df, feature_columns)
print(f"훈련 특성 행렬: {X_train.shape}")

# 검증 데이터  
X_val, y_val = create_feature_target(val_df, feature_columns)
print(f"검증 특성 행렬: {X_val.shape}")

# 테스트 데이터
X_test, y_test = create_feature_target(test_df, feature_columns)
print(f"테스트 특성 행렬: {X_test.shape}")

# 6. 데이터 저장
print("\n6️⃣ 최종 데이터 저장")
print("="*50)

# 전체 스케일링된 데이터 저장
df_scaled_final = df_scaled.drop(columns=['year'])
df_scaled_final.to_csv('data/processed/FS_ratio_flow_scaled.csv', index=False, encoding='utf-8-sig')
print(f"✅ 스케일링된 전체 데이터: data/processed/FS_ratio_flow_scaled.csv")

# 분할된 데이터 저장
datasets = {
    'train': (train_df, X_train, y_train),
    'val': (val_df, X_val, y_val), 
    'test': (test_df, X_test, y_test)
}

for split_name, (df_split, X_split, y_split) in datasets.items():
    # 메타데이터 포함 전체 데이터
    df_split_final = df_split.drop(columns=['year'])
    df_split_final.to_csv(f'data/processed/{split_name}_data.csv', index=False, encoding='utf-8-sig')
    
    # 특성 행렬만
    X_split.to_csv(f'data/processed/X_{split_name}.csv', index=False, encoding='utf-8-sig')
    
    # 타겟 벡터만
    y_split.to_csv(f'data/processed/y_{split_name}.csv', index=False, encoding='utf-8-sig')
    
    print(f"✅ {split_name.upper()} 데이터셋 저장 완료")

# 7. 데이터셋 요약 정보 저장
print("\n7️⃣ 데이터셋 요약 정보 생성")
print("="*50)

summary_info = {
    'dataset_info': {
        'total_samples': len(df_scaled_final),
        'total_features': len(feature_columns),
        'default_rate': (df_scaled_final['default']==1).sum() / len(df_scaled_final),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df)
    },
    'feature_info': {
        'feature_columns': feature_columns,
        'robust_scaled': robust_cols_available,
        'standard_scaled': standard_cols_available,
        'meta_columns': meta_columns
    },
    'split_info': {
        'train_years': train_years,
        'val_years': val_years,
        'test_years': test_years,
        'split_method': 'time_based'
    }
}

import json
with open('data/processed/dataset_info.json', 'w', encoding='utf-8') as f:
    json.dump(summary_info, f, ensure_ascii=False, indent=2)

print(f"✅ 데이터셋 정보: data/processed/dataset_info.json")

# 8. 최종 요약
print("\n8️⃣ 최종 요약")
print("="*50)

print(f"📊 생성된 파일들:")
print(f"  1. FS_ratio_flow_scaled.csv - 전체 스케일링된 데이터 ({len(df_scaled_final):,}개)")
print(f"  2. train_data.csv - 훈련 데이터 ({len(train_df):,}개)")
print(f"  3. val_data.csv - 검증 데이터 ({len(val_df):,}개)")
print(f"  4. test_data.csv - 테스트 데이터 ({len(test_df):,}개)")
print(f"  5. X_train.csv, y_train.csv - 훈련용 특성/타겟")
print(f"  6. X_val.csv, y_val.csv - 검증용 특성/타겟")
print(f"  7. X_test.csv, y_test.csv - 테스트용 특성/타겟")
print(f"  8. dataset_info.json - 데이터셋 메타정보")

print(f"\n🎯 데이터 특성:")
print(f"  - 총 특성: {len(feature_columns)}개")
print(f"  - 부실 비율: {(df_scaled_final['default']==1).sum()/len(df_scaled_final)*100:.2f}%")
print(f"  - 시계열 분할: Look-ahead Bias 방지")
print(f"  - 스케일링: RobustScaler({len(robust_cols_available)}개), StandardScaler({len(standard_cols_available)}개)")

print(f"\n🚀 다음 단계:")
print(f"  1. 클래스 불균형 처리 (SMOTE, 가중치 조정)")
print(f"  2. 모델 훈련 (XGBoost, Random Forest, Logistic Regression)")
print(f"  3. 하이퍼파라미터 튜닝")
print(f"  4. 모델 성능 평가 (AUC, Precision, Recall, F1)")
print(f"  5. 특성 중요도 분석")

print(f"\n✅ 최종 모델링용 데이터셋 생성 완료!") 