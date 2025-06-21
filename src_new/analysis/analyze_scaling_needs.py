import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import platform
from matplotlib import font_manager
import matplotlib as mpl

# 시스템별 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    font_list = ['AppleGothic', 'Apple SD Gothic Neo', 'Nanum Gothic', 'Arial Unicode MS']
elif platform.system() == 'Windows':
    font_list = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
else:  # Linux
    font_list = ['Nanum Gothic', 'DejaVu Sans', 'Liberation Sans']

# 사용 가능한 폰트 찾기
available_fonts = [f.name for f in font_manager.fontManager.ttflist]
selected_font = 'DejaVu Sans'  # 기본값

for font in font_list:
    if font in available_fonts:
        selected_font = font
        break

plt.rcParams['font.family'] = selected_font
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

# 마이너스 기호 문제 해결
mpl.rcParams['axes.unicode_minus'] = False

print(f"사용된 폰트: {selected_font}")

print("=== 재무비율 스케일링 필요성 시각적 분석 ===")

# 1. 데이터 로드
print("\n1️⃣ 데이터 로드")
print("="*50)

# FS_ratio_flow.csv 로드 (라벨링 전 원본 데이터)
fs_ratio = pd.read_csv('data_new/processed/FS_ratio_flow.csv', dtype={'거래소코드': str})
print(f"FS_ratio_flow.csv: {fs_ratio.shape}")

# 재무비율 컬럼만 추출
ratio_columns = [col for col in fs_ratio.columns 
                if col not in ['회사명', '거래소코드', '회계년도']]

print(f"분석할 재무비율: {len(ratio_columns)}개")
print(f"비율 목록: {ratio_columns}")

# 2. 기초 통계량 계산
print("\n2️⃣ 기초 통계량 계산")
print("="*50)

stats_results = []
for col in ratio_columns:
    data = fs_ratio[col].dropna()
    
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
            '변동계수': data.std() / abs(data.mean()) if data.mean() != 0 else np.inf,
            'IQR': data.quantile(0.75) - data.quantile(0.25)
        }
        stats_results.append(stats)

stats_df = pd.DataFrame(stats_results)
print("기초 통계량 계산 완료")

# 3. 분포 시각화 (히스토그램 + 박스플롯)
print("\n3️⃣ 분포 시각화")
print("="*50)

# 3-1. 모든 비율의 히스토그램
n_cols = 4
n_rows = (len(ratio_columns) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(ratio_columns):
    data = fs_ratio[col].dropna()
    
    if len(data) > 0:
        axes[i].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'{col}\n평균: {data.mean():.4f}, 표준편차: {data.std():.4f}', fontsize=10)
        axes[i].set_xlabel('값')
        axes[i].set_ylabel('빈도')
        axes[i].grid(True, alpha=0.3)
        
        # 이상치 표시 (평균 ± 3*표준편차 벗어나는 값)
        mean_val = data.mean()
        std_val = data.std()
        outlier_threshold = 3
        
        if std_val > 0:
            lower_bound = mean_val - outlier_threshold * std_val
            upper_bound = mean_val + outlier_threshold * std_val
            axes[i].axvline(lower_bound, color='red', linestyle='--', alpha=0.5, label='±3σ')
            axes[i].axvline(upper_bound, color='red', linestyle='--', alpha=0.5)

# 빈 subplot 제거
for i in range(len(ratio_columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('outputs/visualizations/ratio_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 3-2. 박스플롯 (스케일링 전)
fig, ax = plt.subplots(figsize=(16, 10))

# 각 비율을 정규화해서 같은 스케일로 표시
normalized_data = []
labels = []

for col in ratio_columns:
    data = fs_ratio[col].dropna()
    if len(data) > 0:
        # 각 비율을 0-1 범위로 정규화 (시각화 목적)
        min_val = data.min()
        max_val = data.max()
        if max_val != min_val:
            normalized = (data - min_val) / (max_val - min_val)
        else:
            normalized = data
        normalized_data.append(normalized)
        labels.append(col)

ax.boxplot(normalized_data, labels=labels)
ax.set_title('재무비율별 박스플롯 (정규화된 값)', fontsize=14, fontweight='bold')
ax.set_ylabel('정규화된 값 (0-1)')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/ratio_boxplots_normalized.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 스케일링 필요성 지표 시각화
print("\n4️⃣ 스케일링 필요성 지표 시각화")
print("="*50)

# 4-1. 변동계수 vs 왜도 산점도
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 변동계수 vs 왜도
cv_values = stats_df['변동계수'].replace([np.inf, -np.inf], np.nan).dropna()
skew_values = stats_df.loc[stats_df['변동계수'].replace([np.inf, -np.inf], np.nan).notna(), '왜도']

ax1.scatter(cv_values, skew_values, alpha=0.7, s=60)
ax1.set_xlabel('변동계수 (CV)')
ax1.set_ylabel('왜도 (Skewness)')
ax1.set_title('변동계수 vs 왜도')
ax1.grid(True, alpha=0.3)

# 스케일링 필요 영역 표시
ax1.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='왜도 임계값 (±3)')
ax1.axhline(y=-3, color='red', linestyle='--', alpha=0.5)
ax1.axvline(x=2, color='orange', linestyle='--', alpha=0.5, label='CV 임계값 (2)')
ax1.legend()

# 각 점에 비율명 표시 (CV > 5 또는 |왜도| > 5인 경우만)
for i, row in stats_df.iterrows():
    cv = row['변동계수']
    skew = row['왜도']
    if not np.isinf(cv) and (cv > 5 or abs(skew) > 5):
        ax1.annotate(row['비율명'], (cv, skew), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, alpha=0.8)

# 4-2. 범위 vs 첨도
ax2.scatter(stats_df['범위'], stats_df['첨도'], alpha=0.7, s=60, color='green')
ax2.set_xlabel('범위 (Range)')
ax2.set_ylabel('첨도 (Kurtosis)')
ax2.set_title('범위 vs 첨도')
ax2.set_xscale('log')  # 로그 스케일로 표시
ax2.grid(True, alpha=0.3)

# 스케일링 필요 영역 표시
ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='첨도 임계값 (10)')
ax2.axvline(x=1000, color='orange', linestyle='--', alpha=0.5, label='범위 임계값 (1000)')
ax2.legend()

# 4-3. 평균의 절댓값 분포
mean_abs = stats_df['평균'].abs()
ax3.bar(range(len(stats_df)), mean_abs, alpha=0.7, color='purple')
ax3.set_xlabel('재무비율 인덱스')
ax3.set_ylabel('평균의 절댓값')
ax3.set_title('재무비율별 평균의 절댓값')
ax3.set_yscale('log')  # 로그 스케일로 표시
ax3.grid(True, alpha=0.3)

# x축에 비율명 표시
ax3.set_xticks(range(len(stats_df)))
ax3.set_xticklabels(stats_df['비율명'], rotation=45, ha='right')

# 4-4. 표준편차 분포
ax4.bar(range(len(stats_df)), stats_df['표준편차'], alpha=0.7, color='orange')
ax4.set_xlabel('재무비율 인덱스')
ax4.set_ylabel('표준편차')
ax4.set_title('재무비율별 표준편차')
ax4.set_yscale('log')  # 로그 스케일로 표시
ax4.grid(True, alpha=0.3)

# x축에 비율명 표시
ax4.set_xticks(range(len(stats_df)))
ax4.set_xticklabels(stats_df['비율명'], rotation=45, ha='right')

plt.tight_layout()
plt.savefig('outputs/visualizations/scaling_need_indicators.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 스케일링 필요성 종합 점수 계산
print("\n5️⃣ 스케일링 필요성 종합 점수 계산")
print("="*50)

scaling_scores = []

for _, row in stats_df.iterrows():
    score = 0
    reasons = []
    
    # 1. 범위 점수 (0-3점)
    if row['범위'] > 10000:
        score += 3
        reasons.append("매우 큰 범위")
    elif row['범위'] > 1000:
        score += 2
        reasons.append("큰 범위")
    elif row['범위'] > 100:
        score += 1
        reasons.append("중간 범위")
    
    # 2. 변동계수 점수 (0-3점)
    cv = row['변동계수']
    if not np.isinf(cv):
        if cv > 10:
            score += 3
            reasons.append("매우 높은 변동성")
        elif cv > 5:
            score += 2
            reasons.append("높은 변동성")
        elif cv > 2:
            score += 1
            reasons.append("중간 변동성")
    
    # 3. 왜도 점수 (0-2점)
    if abs(row['왜도']) > 5:
        score += 2
        reasons.append("매우 높은 왜도")
    elif abs(row['왜도']) > 3:
        score += 1
        reasons.append("높은 왜도")
    
    # 4. 첨도 점수 (0-2점)
    if abs(row['첨도']) > 20:
        score += 2
        reasons.append("매우 높은 첨도")
    elif abs(row['첨도']) > 10:
        score += 1
        reasons.append("높은 첨도")
    
    # 5. 스케일 점수 (0-2점)
    mean_abs = abs(row['평균'])
    std_val = row['표준편차']
    
    if mean_abs > 1000 or std_val > 1000:
        score += 2
        reasons.append("매우 큰 스케일")
    elif mean_abs > 100 or std_val > 100:
        score += 1
        reasons.append("큰 스케일")
    elif mean_abs < 0.001 or std_val < 0.001:
        score += 1
        reasons.append("매우 작은 스케일")
    
    scaling_scores.append({
        '비율명': row['비율명'],
        '스케일링점수': score,
        '이유': ', '.join(reasons) if reasons else '정상',
        '우선순위': 'High' if score >= 7 else 'Medium' if score >= 4 else 'Low'
    })

scaling_score_df = pd.DataFrame(scaling_scores)

# 6. 스케일링 필요성 종합 시각화
print("\n6️⃣ 스케일링 필요성 종합 시각화")
print("="*50)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 6-1. 스케일링 점수 막대그래프
colors = ['red' if score >= 7 else 'orange' if score >= 4 else 'green' 
          for score in scaling_score_df['스케일링점수']]

bars = ax1.bar(range(len(scaling_score_df)), scaling_score_df['스케일링점수'], 
               color=colors, alpha=0.7)
ax1.set_xlabel('재무비율')
ax1.set_ylabel('스케일링 필요성 점수')
ax1.set_title('재무비율별 스케일링 필요성 점수\n(빨강: High≥7, 주황: Medium≥4, 초록: Low<4)')
ax1.set_xticks(range(len(scaling_score_df)))
ax1.set_xticklabels(scaling_score_df['비율명'], rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# 점수 표시
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{int(height)}', ha='center', va='bottom', fontsize=8)

# 6-2. 우선순위별 파이차트
priority_counts = scaling_score_df['우선순위'].value_counts()
colors_pie = ['red', 'orange', 'green']
ax2.pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%',
        colors=colors_pie, startangle=90)
ax2.set_title('스케일링 우선순위 분포')

# 6-3. 상관관계 히트맵 (스케일링 전)
correlation_data = fs_ratio[ratio_columns].corr()
mask = np.triu(np.ones_like(correlation_data, dtype=bool))

sns.heatmap(correlation_data, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, ax=ax3, cbar_kws={"shrink": .8})
ax3.set_title('재무비율 간 상관관계 (스케일링 전)')

# 6-4. 이상치 개수 막대그래프
outlier_counts = []
for col in ratio_columns:
    data = fs_ratio[col].dropna()
    if len(data) > 0:
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        outlier_counts.append(outliers)
    else:
        outlier_counts.append(0)

ax4.bar(range(len(ratio_columns)), outlier_counts, alpha=0.7, color='red')
ax4.set_xlabel('재무비율')
ax4.set_ylabel('이상치 개수')
ax4.set_title('재무비율별 이상치 개수 (IQR 방법)')
ax4.set_xticks(range(len(ratio_columns)))
ax4.set_xticklabels(ratio_columns, rotation=45, ha='right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/scaling_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 스케일링 방법 추천
print("\n7️⃣ 스케일링 방법 추천")
print("="*50)

scaling_recommendations = []

for _, row in stats_df.iterrows():
    col = row['비율명']
    score = scaling_score_df[scaling_score_df['비율명'] == col]['스케일링점수'].iloc[0]
    
    # 이상치가 많은 경우 (높은 첨도, 왜도) -> RobustScaler
    if abs(row['왜도']) > 3 or abs(row['첨도']) > 10:
        recommended = "RobustScaler"
        reason = "이상치 많음 (높은 왜도/첨도)"
    
    # 범위가 매우 큰 경우 -> MinMaxScaler
    elif row['범위'] > 1000:
        recommended = "MinMaxScaler"
        reason = "매우 큰 범위"
    
    # 변동성이 매우 높은 경우 -> RobustScaler
    elif not np.isinf(row['변동계수']) and row['변동계수'] > 5:
        recommended = "RobustScaler"
        reason = "매우 높은 변동성"
    
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
        '스케일링점수': score,
        '추천방법': recommended,
        '이유': reason,
        '우선순위': scaling_score_df[scaling_score_df['비율명'] == col]['우선순위'].iloc[0]
    })

recommend_df = pd.DataFrame(scaling_recommendations)

# 8. 결과 출력 및 저장
print("\n8️⃣ 분석 결과 출력")
print("="*50)

print("\n📊 기초 통계량 요약:")
print(stats_df.round(4))

print(f"\n🔥 스케일링 고우선순위 (점수 ≥ 7):")
high_priority = recommend_df[recommend_df['우선순위'] == 'High'].sort_values('스케일링점수', ascending=False)
for _, row in high_priority.iterrows():
    print(f"  {row['비율명']:8} | 점수: {row['스케일링점수']:2d} | 추천: {row['추천방법']:15} | {row['이유']}")

print(f"\n⚠️ 스케일링 중우선순위 (점수 4-6):")
medium_priority = recommend_df[recommend_df['우선순위'] == 'Medium'].sort_values('스케일링점수', ascending=False)
for _, row in medium_priority.iterrows():
    print(f"  {row['비율명']:8} | 점수: {row['스케일링점수']:2d} | 추천: {row['추천방법']:15} | {row['이유']}")

print(f"\n✅ 스케일링 저우선순위 (점수 < 4):")
low_priority = recommend_df[recommend_df['우선순위'] == 'Low'].sort_values('스케일링점수', ascending=False)
for _, row in low_priority.iterrows():
    print(f"  {row['비율명']:8} | 점수: {row['스케일링점수']:2d} | 추천: {row['추천방법']:15} | {row['이유']}")

# 추천 방법별 그룹화
method_groups = recommend_df.groupby('추천방법')['비율명'].apply(list).to_dict()

print(f"\n📋 스케일링 방법별 그룹:")
for method, ratios in method_groups.items():
    print(f"\n🔧 {method} ({len(ratios)}개):")
    for ratio in ratios:
        score = recommend_df[recommend_df['비율명'] == ratio]['스케일링점수'].iloc[0]
        print(f"  - {ratio:8} (점수: {score})")

# 9. 결과 저장
print("\n9️⃣ 결과 저장")
print("="*50)

# Excel 파일로 저장
with pd.ExcelWriter('outputs/reports/scaling_analysis_detailed.xlsx', engine='openpyxl') as writer:
    stats_df.to_excel(writer, sheet_name='기초통계량', index=False)
    scaling_score_df.to_excel(writer, sheet_name='스케일링점수', index=False)
    recommend_df.to_excel(writer, sheet_name='스케일링추천', index=False)

print(f"✅ 상세 분석 결과 저장: outputs/reports/scaling_analysis_detailed.xlsx")
print(f"✅ 시각화 파일 저장:")
print(f"   - outputs/visualizations/ratio_distributions.png")
print(f"   - outputs/visualizations/ratio_boxplots_normalized.png")
print(f"   - outputs/visualizations/scaling_need_indicators.png")
print(f"   - outputs/visualizations/scaling_comprehensive_analysis.png")

print(f"\n🎯 분석 완료! 시각화 파일들을 확인하여 스케일링 필요성을 판단하세요.")
print(f"📈 고우선순위: {len(high_priority)}개, 중우선순위: {len(medium_priority)}개, 저우선순위: {len(low_priority)}개") 