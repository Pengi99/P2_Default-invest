import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
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

print("=== Default 그룹별 재무비율 통계 분석 ===")

# 1. 프로젝트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
data_path = project_root / 'data' / 'final' / 'FS_ratio_flow_labeled.csv'
output_base = project_root / 'outputs'

# 출력 디렉토리 생성
reports_dir = output_base / 'reports'
viz_dir = output_base / 'visualizations' / 'default_group_analysis'
reports_dir.mkdir(parents=True, exist_ok=True)
viz_dir.mkdir(parents=True, exist_ok=True)

# 2. 데이터 로드
print("\n1️⃣ 데이터 로드")
print("="*50)

df = pd.read_csv(data_path, dtype={'거래소코드': str})
print(f"원본 데이터 크기: {df.shape}")

# Default 분포 확인
print(f"\nDefault 분포:")
print(df['default'].value_counts())
print(f"Default 비율: {df['default'].mean():.4f}")

# 재무비율 컬럼만 추출 (타겟 변수 제외)
feature_columns = [col for col in df.columns 
                  if col not in ['회사명', '거래소코드', '회계년도', 'default']]
print(f"\n재무비율 변수 수: {len(feature_columns)}개")

# 3. Default별 데이터프레임 분리
print("\n2️⃣ Default별 데이터프레임 분리")
print("="*50)

# Default=0 (정상 기업)
normal_df = df[df['default'] == 0].copy()
print(f"정상 기업 (Default=0): {len(normal_df):,}개")

# Default=1 (부실 기업)
default_df = df[df['default'] == 1].copy()
print(f"부실 기업 (Default=1): {len(default_df):,}개")

# 4. 각 그룹별 기초 통계량 계산
print("\n3️⃣ 각 그룹별 기초 통계량 계산")
print("="*50)

# 정상 기업 통계량
print("📊 정상 기업 (Default=0) 통계량:")
normal_stats = normal_df[feature_columns].describe()
print(normal_stats.round(4))

print("\n📊 부실 기업 (Default=1) 통계량:")
default_stats = default_df[feature_columns].describe()
print(default_stats.round(4))

# 5. 통계량 비교 분석
print("\n4️⃣ 통계량 비교 분석")
print("="*50)

# 평균값 비교
mean_comparison = pd.DataFrame({
    '정상기업_평균': normal_stats.loc['mean'],
    '부실기업_평균': default_stats.loc['mean']
})
mean_comparison['차이'] = mean_comparison['부실기업_평균'] - mean_comparison['정상기업_평균']
mean_comparison['차이_비율(%)'] = (mean_comparison['차이'] / mean_comparison['정상기업_평균'].abs()) * 100

# 표준편차 비교
std_comparison = pd.DataFrame({
    '정상기업_표준편차': normal_stats.loc['std'],
    '부실기업_표준편차': default_stats.loc['std']
})
std_comparison['차이'] = std_comparison['부실기업_표준편차'] - std_comparison['정상기업_표준편차']
std_comparison['차이_비율(%)'] = (std_comparison['차이'] / std_comparison['정상기업_표준편차'].abs()) * 100

print("📈 평균값 차이가 큰 상위 10개 변수:")
top_mean_diff = mean_comparison.reindex(mean_comparison['차이'].abs().nlargest(10).index)
print(top_mean_diff.round(4))

print("\n📈 표준편차 차이가 큰 상위 10개 변수:")
top_std_diff = std_comparison.reindex(std_comparison['차이'].abs().nlargest(10).index)
print(top_std_diff.round(4))

# 6. 시각화
print("\n5️⃣ 시각화")
print("="*50)

# 6-1. 평균값 비교 막대그래프 (상위 15개 변수, 발생액 제외)
print("평균값 비교 차트 생성 중...")
# 발생액은 단위가 너무 커서 제외하고 상위 15개 선택
mean_comparison_filtered = mean_comparison[mean_comparison.index != '발생액']
top_15_mean = mean_comparison_filtered.reindex(mean_comparison_filtered['차이'].abs().nlargest(15).index)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# 상단: 그룹별 평균값 비교
x_pos = np.arange(len(top_15_mean))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, top_15_mean['정상기업_평균'], width, 
               label='정상기업 (Default=0)', color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x_pos + width/2, top_15_mean['부실기업_평균'], width,
               label='부실기업 (Default=1)', color='red', alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_xlabel('재무비율 변수', fontsize=12)
ax1.set_ylabel('평균값', fontsize=12)
ax1.set_title('Default 그룹별 재무비율 평균값 비교 (차이 상위 15개, 발생액 제외)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(top_15_mean.index, rotation=45, ha='right', fontsize=10)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# 평균값 표시 (절댓값이 큰 경우만)
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    val1 = bar1.get_height()
    val2 = bar2.get_height()
    if abs(val1) > 0.01:  # 절댓값이 0.01보다 큰 경우만 표시
        ax1.text(bar1.get_x() + bar1.get_width()/2., val1 + (abs(val1)*0.02 if val1 >= 0 else -abs(val1)*0.02),
               f'{val1:.3f}', ha='center', va='bottom' if val1 >= 0 else 'top', fontsize=8)
    if abs(val2) > 0.01:
        ax1.text(bar2.get_x() + bar2.get_width()/2., val2 + (abs(val2)*0.02 if val2 >= 0 else -abs(val2)*0.02),
               f'{val2:.3f}', ha='center', va='bottom' if val2 >= 0 else 'top', fontsize=8)

# 하단: 차이값 막대그래프
colors = ['red' if x > 0 else 'blue' for x in top_15_mean['차이']]
bars3 = ax2.bar(x_pos, top_15_mean['차이'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('재무비율 변수', fontsize=12)
ax2.set_ylabel('평균값 차이 (부실기업 - 정상기업)', fontsize=12)
ax2.set_title('평균값 차이 (음수: 부실기업이 더 낮음, 양수: 부실기업이 더 높음)', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(top_15_mean.index, rotation=45, ha='right', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)

# 차이값 표시
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
           f'{height:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
           fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(viz_dir / '01_mean_comparison_top15.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-2. 표준편차 비교 막대그래프 (상위 15개 변수, 발생액 제외)
print("표준편차 비교 차트 생성 중...")
# 발생액은 단위가 너무 커서 제외하고 상위 15개 선택
std_comparison_filtered = std_comparison[std_comparison.index != '발생액']
top_15_std = std_comparison_filtered.reindex(std_comparison_filtered['차이'].abs().nlargest(15).index)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# 상단: 그룹별 표준편차 비교
x_pos = np.arange(len(top_15_std))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, top_15_std['정상기업_표준편차'], width, 
               label='정상기업 (Default=0)', color='lightgreen', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x_pos + width/2, top_15_std['부실기업_표준편차'], width,
               label='부실기업 (Default=1)', color='orange', alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_xlabel('재무비율 변수', fontsize=12)
ax1.set_ylabel('표준편차', fontsize=12)
ax1.set_title('Default 그룹별 재무비율 표준편차 비교 (차이 상위 15개, 발생액 제외)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(top_15_std.index, rotation=45, ha='right', fontsize=10)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# 표준편차값 표시 (절댓값이 큰 경우만)
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    val1 = bar1.get_height()
    val2 = bar2.get_height()
    if abs(val1) > 0.01:  # 절댓값이 0.01보다 큰 경우만 표시
        ax1.text(bar1.get_x() + bar1.get_width()/2., val1 + abs(val1)*0.02,
               f'{val1:.3f}', ha='center', va='bottom', fontsize=8)
    if abs(val2) > 0.01:
        ax1.text(bar2.get_x() + bar2.get_width()/2., val2 + abs(val2)*0.02,
               f'{val2:.3f}', ha='center', va='bottom', fontsize=8)

# 하단: 표준편차 차이값 막대그래프
colors = ['red' if x > 0 else 'blue' for x in top_15_std['차이']]
bars3 = ax2.bar(x_pos, top_15_std['차이'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('재무비율 변수', fontsize=12)
ax2.set_ylabel('표준편차 차이 (부실기업 - 정상기업)', fontsize=12)
ax2.set_title('표준편차 차이 (음수: 부실기업이 더 안정적, 양수: 부실기업이 더 변동적)', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(top_15_std.index, rotation=45, ha='right', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)

# 차이값 표시
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
           f'{height:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
           fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(viz_dir / '02_std_comparison_top15.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-3. 박스플롯 비교 (상위 12개 변수, 발생액 제외)
print("박스플롯 비교 차트 생성 중...")
top_12_vars = mean_comparison_filtered['차이'].abs().nlargest(12).index

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for i, var in enumerate(top_12_vars):
    data_normal = normal_df[var].dropna()
    data_default = default_df[var].dropna()
    
    axes[i].boxplot([data_normal, data_default], labels=['정상기업', '부실기업'])
    axes[i].set_title(f'{var}', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].tick_params(axis='both', which='major', labelsize=8)

plt.suptitle('Default 그룹별 재무비율 분포 비교 (박스플롯)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(viz_dir / '03_boxplot_comparison_top12.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-4. 히스토그램 비교 (상위 6개 변수, 발생액 제외)
print("히스토그램 비교 차트 생성 중...")
top_6_vars = mean_comparison_filtered['차이'].abs().nlargest(6).index

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, var in enumerate(top_6_vars):
    data_normal = normal_df[var].dropna()
    data_default = default_df[var].dropna()
    
    axes[i].hist(data_normal, bins=50, alpha=0.7, label='정상기업', color='skyblue', density=True)
    axes[i].hist(data_default, bins=50, alpha=0.7, label='부실기업', color='red', density=True)
    axes[i].set_title(f'{var}', fontsize=12)
    axes[i].set_xlabel('값')
    axes[i].set_ylabel('밀도')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Default 그룹별 재무비율 분포 비교 (히스토그램)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(viz_dir / '04_histogram_comparison_top6.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-5. 통계량 히트맵
print("통계량 히트맵 생성 중...")
# 주요 통계량 추출 (평균, 표준편차, 중앙값)
stats_heatmap_data = []

for stat in ['mean', 'std', '50%']:  # 50%는 중앙값
    for group, stats_df in [('정상기업', normal_stats), ('부실기업', default_stats)]:
        for var in feature_columns[:20]:  # 상위 20개 변수만
            stats_heatmap_data.append({
                '그룹': group,
                '통계량': stat,
                '변수명': var,
                '값': stats_df.loc[stat, var]
            })

heatmap_df = pd.DataFrame(stats_heatmap_data)
heatmap_pivot = heatmap_df.pivot_table(index='변수명', columns=['그룹', '통계량'], values='값')

fig, ax = plt.subplots(figsize=(12, 16))
sns.heatmap(heatmap_pivot, annot=False, cmap='RdBu_r', center=0, ax=ax,
            cbar_kws={'label': '값'})
ax.set_title('Default 그룹별 재무비율 통계량 비교 히트맵', fontsize=14, fontweight='bold')
ax.set_xlabel('그룹 및 통계량')
ax.set_ylabel('재무비율 변수')

plt.tight_layout()
plt.savefig(viz_dir / '05_statistics_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-6. 차이 분석 종합 대시보드
print("종합 대시보드 생성 중...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. 평균값 차이 (절댓값 기준 상위 10개, 발생액 제외)
top_10_mean_abs = mean_comparison_filtered.reindex(mean_comparison_filtered['차이'].abs().nlargest(10).index)
colors = ['red' if x > 0 else 'blue' for x in top_10_mean_abs['차이']]
bars = ax1.barh(range(len(top_10_mean_abs)), top_10_mean_abs['차이'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(top_10_mean_abs)))
ax1.set_yticklabels(top_10_mean_abs.index, fontsize=9)
ax1.set_xlabel('평균값 차이 (부실기업 - 정상기업)')
ax1.set_title('평균값 차이 상위 10개 변수')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)

# 2. 표준편차 차이 (절댓값 기준 상위 10개, 발생액 제외)
top_10_std_abs = std_comparison_filtered.reindex(std_comparison_filtered['차이'].abs().nlargest(10).index)
colors = ['red' if x > 0 else 'blue' for x in top_10_std_abs['차이']]
bars = ax2.barh(range(len(top_10_std_abs)), top_10_std_abs['차이'], color=colors, alpha=0.7)
ax2.set_yticks(range(len(top_10_std_abs)))
ax2.set_yticklabels(top_10_std_abs.index, fontsize=9)
ax2.set_xlabel('표준편차 차이 (부실기업 - 정상기업)')
ax2.set_title('표준편차 차이 상위 10개 변수')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)

# 3. 변동계수 비교
cv_normal = (normal_stats.loc['std'] / normal_stats.loc['mean'].abs()).replace([np.inf, -np.inf], np.nan)
cv_default = (default_stats.loc['std'] / default_stats.loc['mean'].abs()).replace([np.inf, -np.inf], np.nan)
cv_diff = (cv_default - cv_normal).dropna()
top_10_cv = cv_diff.abs().nlargest(10)

colors = ['red' if x > 0 else 'blue' for x in cv_diff[top_10_cv.index]]
bars = ax3.barh(range(len(top_10_cv)), cv_diff[top_10_cv.index], color=colors, alpha=0.7)
ax3.set_yticks(range(len(top_10_cv)))
ax3.set_yticklabels(top_10_cv.index, fontsize=9)
ax3.set_xlabel('변동계수 차이 (부실기업 - 정상기업)')
ax3.set_title('변동계수 차이 상위 10개 변수')
ax3.grid(True, alpha=0.3)
ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)

# 4. 그룹별 데이터 분포 요약
group_summary = pd.DataFrame({
    '정상기업': [len(normal_df), normal_df[feature_columns].isnull().sum().sum()],
    '부실기업': [len(default_df), default_df[feature_columns].isnull().sum().sum()]
}, index=['관측치수', '결측치수'])

x_pos = np.arange(len(group_summary.index))
width = 0.35
bars1 = ax4.bar(x_pos - width/2, group_summary['정상기업'], width, 
               label='정상기업', color='skyblue', alpha=0.8)
bars2 = ax4.bar(x_pos + width/2, group_summary['부실기업'], width,
               label='부실기업', color='red', alpha=0.8)

ax4.set_xlabel('항목')
ax4.set_ylabel('개수')
ax4.set_title('그룹별 데이터 현황')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(group_summary.index)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 값 표시
for bar1, bar2 in zip(bars1, bars2):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + max(group_summary.values.flatten())*0.01,
           f'{int(height1):,}', ha='center', va='bottom', fontsize=10)
    ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + max(group_summary.values.flatten())*0.01,
           f'{int(height2):,}', ha='center', va='bottom', fontsize=10)

plt.suptitle('Default 그룹별 재무비율 통계 분석 종합 대시보드 (발생액 제외)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(viz_dir / '06_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. 결과 저장
print("\n6️⃣ 결과 저장")
print("="*50)

# 기초 통계량 저장
normal_stats.to_csv(reports_dir / 'normal_companies_statistics.csv', encoding='utf-8-sig')
default_stats.to_csv(reports_dir / 'default_companies_statistics.csv', encoding='utf-8-sig')

# 비교 분석 결과 저장
mean_comparison.to_csv(reports_dir / 'mean_comparison_analysis.csv', encoding='utf-8-sig')
std_comparison.to_csv(reports_dir / 'std_comparison_analysis.csv', encoding='utf-8-sig')

# 종합 비교 분석 결과
comprehensive_comparison = pd.DataFrame({
    '정상기업_평균': normal_stats.loc['mean'],
    '정상기업_표준편차': normal_stats.loc['std'],
    '정상기업_중앙값': normal_stats.loc['50%'],
    '부실기업_평균': default_stats.loc['mean'],
    '부실기업_표준편차': default_stats.loc['std'],
    '부실기업_중앙값': default_stats.loc['50%'],
    '평균차이': mean_comparison['차이'],
    '표준편차차이': std_comparison['차이']
})

comprehensive_comparison.to_csv(reports_dir / 'comprehensive_group_comparison.csv', encoding='utf-8-sig')

# 상세 분석 리포트 생성
report_content = f"""
# Default 그룹별 재무비율 통계 분석 리포트

## 1. 분석 개요
- 원본 데이터: {len(df):,}개 관측치
- 정상 기업 (Default=0): {len(normal_df):,}개 ({len(normal_df)/len(df)*100:.1f}%)
- 부실 기업 (Default=1): {len(default_df):,}개 ({len(default_df)/len(df)*100:.1f}%)
- 분석 변수: {len(feature_columns)}개 재무비율

## 2. 주요 발견사항

### 2.1 평균값 차이가 가장 큰 변수들
"""

for i, (var, row) in enumerate(top_mean_diff.head(5).iterrows(), 1):
    report_content += f"{i}. {var}\n"
    report_content += f"   정상기업: {row['정상기업_평균']:.4f}, 부실기업: {row['부실기업_평균']:.4f}\n"
    report_content += f"   차이: {row['차이']:+.4f} ({row['차이_비율(%)']:+.1f}%)\n\n"

report_content += f"""
### 2.2 표준편차 차이가 가장 큰 변수들
"""

for i, (var, row) in enumerate(top_std_diff.head(5).iterrows(), 1):
    report_content += f"{i}. {var}\n"
    report_content += f"   정상기업: {row['정상기업_표준편차']:.4f}, 부실기업: {row['부실기업_표준편차']:.4f}\n"
    report_content += f"   차이: {row['차이']:+.4f} ({row['차이_비율(%)']:+.1f}%)\n\n"

report_content += f"""
## 3. 그룹별 데이터 품질
- 정상기업 결측치: {normal_df[feature_columns].isnull().sum().sum():,}개
- 부실기업 결측치: {default_df[feature_columns].isnull().sum().sum():,}개

## 4. 결론 및 시사점
1. 부실기업과 정상기업 간 재무비율에서 유의미한 차이 존재
2. 특정 재무비율들이 부실 예측에 중요한 지표로 활용 가능
3. 변동성 차이도 고려하여 모델링 시 활용 필요

## 5. 생성된 파일
- 시각화: {viz_dir}/에 6개 차트 파일
- 통계 데이터: {reports_dir}/에 4개 CSV 파일
"""

# 리포트 저장
with open(reports_dir / 'default_group_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"✅ 분석 완료!")
print(f"📊 정상기업 통계: {reports_dir / 'normal_companies_statistics.csv'}")
print(f"📊 부실기업 통계: {reports_dir / 'default_companies_statistics.csv'}")
print(f"📊 평균 비교 분석: {reports_dir / 'mean_comparison_analysis.csv'}")
print(f"📊 표준편차 비교 분석: {reports_dir / 'std_comparison_analysis.csv'}")
print(f"📊 종합 비교 분석: {reports_dir / 'comprehensive_group_comparison.csv'}")
print(f"📄 상세 리포트: {reports_dir / 'default_group_analysis_report.txt'}")
print(f"📈 시각화 파일: {viz_dir}/ (6개 차트)")

print(f"\n🎯 주요 분석 결과:")
print(f"- 정상기업: {len(normal_df):,}개 ({len(normal_df)/len(df)*100:.1f}%)")
print(f"- 부실기업: {len(default_df):,}개 ({len(default_df)/len(df)*100:.1f}%)")
print(f"- 평균값 차이가 가장 큰 변수: {mean_comparison['차이'].abs().idxmax()}")
print(f"- 표준편차 차이가 가장 큰 변수: {std_comparison['차이'].abs().idxmax()}")

print(f"\n📈 생성된 시각화:")
print(f"   01_mean_comparison_top15.png : 평균값 비교 (상위 15개)")
print(f"   02_std_comparison_top15.png : 표준편차 비교 (상위 15개)")
print(f"   03_boxplot_comparison_top12.png : 박스플롯 비교 (상위 12개)")
print(f"   04_histogram_comparison_top6.png : 히스토그램 비교 (상위 6개)")
print(f"   05_statistics_heatmap.png : 통계량 히트맵")
print(f"   06_comprehensive_dashboard.png : 종합 대시보드")