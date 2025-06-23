import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=== 부실 라벨링 적용 및 스케일링 분석 ===")

# 1. 데이터 로드
print("\n1️⃣ 데이터 로드")
print("="*50)

# FS_ratio_flow.csv 로드
fs_ratio = pd.read_csv('data/final/FS_ratio_flow.csv', dtype={'거래소코드': str})
print(f"FS_ratio_flow.csv: {fs_ratio.shape}")

# 부실 기업 정보 로드
value_fail = pd.read_csv('data/raw/value_fail.csv', dtype={'종목코드': str})
print(f"value_fail.csv: {value_fail.shape}")

# 2. 부실 라벨링 적용
print("\n2️⃣ 부실 라벨링 적용")
print("="*50)

# 종목코드를 6자리로 정규화
value_fail['종목코드'] = value_fail['종목코드'].astype(str).str.zfill(6)

# 폐지일자를 datetime으로 변환
value_fail['폐지일자'] = pd.to_datetime(value_fail['폐지일자'], errors='coerce')
value_fail = value_fail.dropna(subset=['폐지일자'])
value_fail['폐지년도'] = value_fail['폐지일자'].dt.year

print(f"부실 기업 정보: {len(value_fail)}개")
print(f"폐지년도 범위: {value_fail['폐지년도'].min()} ~ {value_fail['폐지년도'].max()}")

# FS_ratio_flow에서 회계년도에서 연도 추출
fs_ratio['회계년도_year'] = pd.to_datetime(fs_ratio['회계년도'], format='%Y/%m', errors='coerce').dt.year

# default 컬럼 초기화
fs_ratio['default'] = 0

# 부실 기업 라벨링 (부실 전년도에 default=1)
labeled_count = 0
for _, row in value_fail.iterrows():
    company_code = row['종목코드']
    target_year = row['폐지년도'] - 1  # 부실 전년도
    
    condition = (fs_ratio['거래소코드'] == company_code) & (fs_ratio['회계년도_year'] == target_year)
    matches = fs_ratio.loc[condition]
    
    if len(matches) > 0:
        fs_ratio.loc[condition, 'default'] = 1
        labeled_count += len(matches)

print(f"부실 라벨링 완료: {labeled_count}개 데이터에 default=1 적용")

# 부실 기업 통계
default_1_count = (fs_ratio['default'] == 1).sum()
default_0_count = (fs_ratio['default'] == 0).sum()
print(f"default=1 (부실): {default_1_count:,}개 ({default_1_count/len(fs_ratio)*100:.2f}%)")
print(f"default=0 (정상): {default_0_count:,}개 ({default_0_count/len(fs_ratio)*100:.2f}%)")

# 부실 기업의 다른 연도 데이터 제거 (부실예측 모델링을 위해)
default_companies = fs_ratio[fs_ratio['default'] == 1]['거래소코드'].unique()
print(f"부실 기업 수: {len(default_companies)}개")

# 부실 기업은 default=1인 행만 남기고, 정상 기업은 모든 행 유지
condition_to_keep = ~fs_ratio['거래소코드'].isin(default_companies) | (fs_ratio['default'] == 1)
fs_ratio_final = fs_ratio[condition_to_keep].copy()

print(f"최종 데이터: {len(fs_ratio_final):,}개 (부실 기업의 다른 연도 데이터 제거 후)")

# 3. 재무비율 스케일링 분석
print("\n3️⃣ 재무비율 스케일링 필요성 분석")
print("="*50)

# 재무비율 컬럼 추출
ratio_columns = [col for col in fs_ratio_final.columns 
                if col not in ['회사명', '거래소코드', '회계년도', '회계년도_year', 'default']]

print(f"분석할 재무비율: {len(ratio_columns)}개")

# 각 비율의 기초 통계량 계산
stats_results = []

for col in ratio_columns:
    data = fs_ratio_final[col].dropna()
    
    if len(data) > 0:
        stats = {
            '비율명': col,
            '데이터수': len(data),
            '평균': data.mean(),
            '표준편차': data.std(),
            '최솟값': data.min(),
            '25%': data.quantile(0.25),
            '중앙값': data.median(),
            '75%': data.quantile(0.75),
            '최댓값': data.max(),
            '왜도': data.skew(),
            '첨도': data.kurtosis(),
            '범위': data.max() - data.min(),
            '변동계수': data.std() / abs(data.mean()) if data.mean() != 0 else np.inf
        }
        stats_results.append(stats)

stats_df = pd.DataFrame(stats_results)

print("\n📊 재무비율별 기초 통계량:")
print("="*80)
for _, row in stats_df.iterrows():
    print(f"{row['비율명']:8} | 평균: {row['평균']:8.4f} | 표준편차: {row['표준편차']:8.4f} | 범위: {row['범위']:12.2e}")

# 4. 스케일링 필요성 판단
print("\n4️⃣ 스케일링 필요성 판단")
print("="*50)

# 스케일링 필요성 기준
scaling_needed = []

for _, row in stats_df.iterrows():
    col = row['비율명']
    needs_scaling = False
    reasons = []
    
    # 1. 범위가 매우 큰 경우 (1000배 이상 차이)
    if row['범위'] > 1000:
        needs_scaling = True
        reasons.append(f"큰 범위({row['범위']:.2e})")
    
    # 2. 표준편차가 평균보다 훨씬 큰 경우 (변동계수 > 2)
    if row['변동계수'] > 2:
        needs_scaling = True
        reasons.append(f"높은 변동성(CV={row['변동계수']:.2f})")
    
    # 3. 왜도가 매우 큰 경우 (절댓값 > 3)
    if abs(row['왜도']) > 3:
        needs_scaling = True
        reasons.append(f"높은 왜도({row['왜도']:.2f})")
    
    # 4. 첨도가 매우 큰 경우 (절댓값 > 10)
    if abs(row['첨도']) > 10:
        needs_scaling = True
        reasons.append(f"높은 첨도({row['첨도']:.2f})")
    
    # 5. 평균과 표준편차의 스케일이 다른 비율들과 크게 다른 경우
    mean_scale = abs(row['평균'])
    std_scale = row['표준편차']
    
    if mean_scale > 100 or std_scale > 100:
        needs_scaling = True
        reasons.append("큰 스케일")
    elif mean_scale < 0.001 or std_scale < 0.001:
        needs_scaling = True
        reasons.append("작은 스케일")
    
    scaling_needed.append({
        '비율명': col,
        '스케일링_필요': needs_scaling,
        '이유': ', '.join(reasons) if reasons else '정상',
        '우선순위': len(reasons)
    })

scaling_df = pd.DataFrame(scaling_needed)

# 스케일링 필요한 비율들
need_scaling = scaling_df[scaling_df['스케일링_필요'] == True].sort_values('우선순위', ascending=False)
no_scaling = scaling_df[scaling_df['스케일링_필요'] == False]

print(f"✅ 스케일링 필요: {len(need_scaling)}개 비율")
print(f"✅ 스케일링 불필요: {len(no_scaling)}개 비율")

if len(need_scaling) > 0:
    print(f"\n🔥 스케일링 필요한 비율들 (우선순위 순):")
    for _, row in need_scaling.iterrows():
        print(f"  {row['비율명']:8} | 이유: {row['이유']}")

if len(no_scaling) > 0:
    print(f"\n✨ 스케일링 불필요한 비율들:")
    for _, row in no_scaling.iterrows():
        print(f"  {row['비율명']:8} | {row['이유']}")

# 5. 스케일링 방법 추천
print("\n5️⃣ 스케일링 방법 추천")
print("="*50)

scaling_recommendations = []

for _, row in stats_df.iterrows():
    col = row['비율명']
    
    # 이상치가 많은 경우 (높은 첨도, 왜도) -> RobustScaler
    if abs(row['왜도']) > 2 or abs(row['첨도']) > 5:
        recommended = "RobustScaler"
        reason = "이상치 많음"
    
    # 범위가 매우 큰 경우 -> MinMaxScaler 또는 RobustScaler
    elif row['범위'] > 1000:
        recommended = "MinMaxScaler 또는 RobustScaler"
        reason = "매우 큰 범위"
    
    # 정규분포에 가까운 경우 -> StandardScaler
    elif abs(row['왜도']) < 1 and abs(row['첨도']) < 3:
        recommended = "StandardScaler"
        reason = "정규분포에 가까움"
    
    # 기본적으로 StandardScaler
    else:
        recommended = "StandardScaler"
        reason = "일반적인 경우"
    
    scaling_recommendations.append({
        '비율명': col,
        '추천방법': recommended,
        '이유': reason
    })

recommend_df = pd.DataFrame(scaling_recommendations)

# 추천 방법별 그룹화
method_groups = recommend_df.groupby('추천방법')['비율명'].apply(list).to_dict()

print("📋 스케일링 방법별 추천:")
for method, ratios in method_groups.items():
    print(f"\n🔧 {method}:")
    for ratio in ratios:
        reason = recommend_df[recommend_df['비율명'] == ratio]['이유'].iloc[0]
        print(f"  - {ratio:8} ({reason})")

# 6. 최종 데이터 저장
print("\n6️⃣ 최종 데이터 저장")
print("="*50)

# 불필요한 임시 컬럼 제거
fs_ratio_final = fs_ratio_final.drop(columns=['회계년도_year'])

# 라벨링된 데이터 저장
fs_ratio_final.to_csv('data/final/FS_ratio_flow_labeled.csv', index=False, encoding='utf-8-sig')
print(f"✅ 라벨링된 데이터 저장: data/final/FS_ratio_flow_labeled.csv")
print(f"   - 총 데이터: {len(fs_ratio_final):,}개")
print(f"   - 부실(default=1): {(fs_ratio_final['default']==1).sum():,}개")
print(f"   - 정상(default=0): {(fs_ratio_final['default']==0).sum():,}개")

# 스케일링 분석 결과 저장
scaling_analysis = {
    '기초통계량': stats_df,
    '스케일링필요성': scaling_df,
    '스케일링추천': recommend_df
}

with pd.ExcelWriter('outputs/reports/scaling_analysis.xlsx', engine='openpyxl') as writer:
    stats_df.to_excel(writer, sheet_name='기초통계량', index=False)
    scaling_df.to_excel(writer, sheet_name='스케일링필요성', index=False)
    recommend_df.to_excel(writer, sheet_name='스케일링추천', index=False)

print(f"✅ 스케일링 분석 결과 저장: outputs/reports/scaling_analysis.xlsx")

# 7. 요약 및 권장사항
print("\n7️⃣ 최종 요약 및 권장사항")
print("="*50)

print(f"📊 데이터 요약:")
print(f"  - 최종 데이터: {len(fs_ratio_final):,}개")
print(f"  - 부실 비율: {(fs_ratio_final['default']==1).sum()/len(fs_ratio_final)*100:.2f}%")
print(f"  - 분석 재무비율: {len(ratio_columns)}개")

print(f"\n🔧 스케일링 권장사항:")
print(f"  - 스케일링 필요: {len(need_scaling)}개 비율")
print(f"  - 주요 스케일링 방법: {', '.join(method_groups.keys())}")

print(f"\n🚀 다음 단계:")
print(f"  1. 스케일링 적용 (추천 방법별)")
print(f"  2. 훈련/검증/테스트 데이터 분할")
print(f"  3. 모델 훈련 (XGBoost, Random Forest, Logistic Regression)")
print(f"  4. 모델 성능 평가 (AUC, Precision, Recall)")

print(f"\n💡 모델링 팁:")
print(f"  - 클래스 불균형 해결: SMOTE, 가중치 조정")
print(f"  - 시계열 특성 고려: 시간 기반 분할")
print(f"  - 특성 선택: 상관관계 분석, 특성 중요도")

print(f"\n✅ 부실 라벨링 및 스케일링 분석 완료!") 