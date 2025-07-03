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

print("=== 결측치 비율별 행 제거 시 Default 분포 분석 ===")

# 1. 프로젝트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
data_path = project_root / 'data' / 'processed' / 'FS2_filtered.csv'
output_base = project_root / 'outputs' / 'analysis' / 'missing_data_default_analysis'

# 출력 디렉토리 생성
reports_dir = output_base / 'reports'
viz_dir = output_base / 'visualizations'
reports_dir.mkdir(parents=True, exist_ok=True)
viz_dir.mkdir(parents=True, exist_ok=True)

# 2. 데이터 로드
print("\n1️⃣ 데이터 로드")
print("="*50)

df = pd.read_csv(data_path, dtype={'거래소코드': str})
print(f"원본 데이터 크기: {df.shape}")
print(f"컬럼: {list(df.columns)}")

# 타겟 변수 확인
print(f"\nDefault 분포:")
print(df['default'].value_counts())
print(f"Default 비율: {df['default'].mean():.4f}")

# 재무비율 컬럼만 추출 (타겟 변수 제외)
feature_columns = [col for col in df.columns 
                  if col not in ['회사명', '거래소코드', '회계년도', 'default']]
print(f"\n재무비율 변수 수: {len(feature_columns)}개")

# 3. 결측치 분석
print("\n2️⃣ 결측치 분석")
print("="*50)

# 각 변수별 결측치 비율 계산
missing_stats = []
for col in feature_columns:
    missing_count = df[col].isnull().sum()
    missing_rate = (missing_count / len(df)) * 100
    missing_stats.append({
        '변수명': col,
        '결측치수': missing_count,
        '결측치비율(%)': missing_rate
    })

missing_df = pd.DataFrame(missing_stats)
missing_df = missing_df.sort_values('결측치비율(%)', ascending=False)

print("상위 10개 결측치 비율이 높은 변수:")
print(missing_df.head(10))

# 4. 결측치 비율별 행 제거 시나리오
print("\n3️⃣ 결측치 비율별 행 제거 시나리오")
print("="*50)

# 결측치 비율 임계값 설정
threshold_scenarios = [0, 10, 20, 30, 40, 50, 60, 70, 80]
scenario_results = []
missing_pattern_results = []

for threshold in threshold_scenarios:
    print(f"\n📊 결측치 {threshold}% 이상인 행 제거 시나리오")
    
    # 각 행의 결측치 비율 계산
    row_missing_counts = df[feature_columns].isnull().sum(axis=1)
    row_missing_rates = (row_missing_counts / len(feature_columns)) * 100
    
    # 임계값 이하인 행만 선택
    valid_mask = row_missing_rates <= threshold
    filtered_df = df[valid_mask].copy()
    
    # 결과 통계
    original_count = len(df)
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    retention_rate = (filtered_count / original_count) * 100
    
    # Default 분포 분석
    original_default_count = df['default'].sum()
    original_default_rate = df['default'].mean()
    
    if filtered_count > 0:
        filtered_default_count = filtered_df['default'].sum()
        filtered_default_rate = filtered_df['default'].mean()
        default_retention_rate = (filtered_default_count / original_default_count) * 100 if original_default_count > 0 else 0
        
        # 필터링 후 남은 데이터의 결측치 분석
        remaining_missing_total = filtered_df[feature_columns].isnull().sum().sum()
        remaining_missing_rate = (remaining_missing_total / (filtered_count * len(feature_columns))) * 100
        
        # 컬럼별 결측치 분석
        column_missing_stats = []
        for col in feature_columns:
            original_missing = df[col].isnull().sum()
            remaining_missing = filtered_df[col].isnull().sum()
            original_missing_rate = (original_missing / len(df)) * 100
            remaining_missing_rate_col = (remaining_missing / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            
            column_missing_stats.append({
                '컬럼명': col,
                '원본결측치수': original_missing,
                '필터링후결측치수': remaining_missing,
                '원본결측치비율(%)': original_missing_rate,
                '필터링후결측치비율(%)': remaining_missing_rate_col,
                '결측치변화(%)': remaining_missing_rate_col - original_missing_rate
            })
        
    else:
        filtered_default_count = 0
        filtered_default_rate = 0
        default_retention_rate = 0
        remaining_missing_total = 0
        remaining_missing_rate = 0
        column_missing_stats = []
    
    # 시나리오별 결측치 패턴 저장
    missing_pattern_results.append({
        '임계값(%)': threshold,
        '필터링후행수': filtered_count,
        '남은결측치총개수': remaining_missing_total,
        '남은결측치비율(%)': remaining_missing_rate,
        '컬럼별결측치통계': column_missing_stats
    })
    
    result = {
        '임계값(%)': threshold,
        '원본행수': original_count,
        '필터링후행수': filtered_count,
        '제거된행수': removed_count,
        '데이터보존율(%)': retention_rate,
        '원본Default수': original_default_count,
        '필터링후Default수': filtered_default_count,
        'Default보존율(%)': default_retention_rate,
        '원본Default비율(%)': original_default_rate * 100,
        '필터링후Default비율(%)': filtered_default_rate * 100,
        'Default비율변화(%)': (filtered_default_rate - original_default_rate) * 100,
        '남은결측치총개수': remaining_missing_total,
        '남은결측치비율(%)': remaining_missing_rate
    }
    
    scenario_results.append(result)
    
    print(f"  원본 데이터: {original_count:,}행")
    print(f"  필터링 후: {filtered_count:,}행 ({retention_rate:.1f}% 보존)")
    print(f"  제거된 행: {removed_count:,}행")
    print(f"  원본 Default: {original_default_count:,}개 ({original_default_rate:.4f})")
    print(f"  필터링 후 Default: {filtered_default_count:,}개 ({filtered_default_rate:.4f})")
    print(f"  Default 보존율: {default_retention_rate:.1f}%")
    print(f"  Default 비율 변화: {(filtered_default_rate - original_default_rate) * 100:+.4f}%p")
    print(f"  남은 결측치: {remaining_missing_total:,}개 ({remaining_missing_rate:.2f}%)")

# 5. 결과 DataFrame 생성
results_df = pd.DataFrame(scenario_results)
print("\n📋 시나리오별 결과 요약:")
print(results_df.round(2))

# 6. 시각화
print("\n4️⃣ 시각화")
print("="*50)

# 6-1. 데이터 보존율 vs Default 보존율
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 데이터 보존율
ax1.plot(results_df['임계값(%)'], results_df['데이터보존율(%)'], 
         marker='o', linewidth=2, markersize=8, color='blue', label='데이터 보존율')
ax1.plot(results_df['임계값(%)'], results_df['Default보존율(%)'], 
         marker='s', linewidth=2, markersize=8, color='red', label='Default 보존율')
ax1.set_xlabel('결측치 비율 임계값 (%)')
ax1.set_ylabel('보존율 (%)')
ax1.set_title('결측치 임계값별 데이터 및 Default 보존율')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(0, 105)

# Default 비율 변화
ax2.plot(results_df['임계값(%)'], results_df['필터링후Default비율(%)'], 
         marker='o', linewidth=2, markersize=8, color='green', label='필터링 후 Default 비율')
ax2.axhline(y=results_df['원본Default비율(%)'].iloc[0], 
           color='orange', linestyle='--', linewidth=2, label='원본 Default 비율')
ax2.set_xlabel('결측치 비율 임계값 (%)')
ax2.set_ylabel('Default 비율 (%)')
ax2.set_title('결측치 임계값별 Default 비율 변화')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(viz_dir / '01_missing_threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-2. 데이터 크기 변화
fig, ax = plt.subplots(figsize=(12, 8))

x_pos = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x_pos - width/2, results_df['필터링후행수'], width, 
               label='전체 데이터', color='skyblue', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, results_df['필터링후Default수'], width,
               label='Default 데이터', color='red', alpha=0.8)

ax.set_xlabel('결측치 비율 임계값 (%)')
ax.set_ylabel('데이터 수')
ax.set_title('결측치 임계값별 데이터 수 변화')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['임계값(%)'])
ax.legend()
ax.grid(True, alpha=0.3)

# 값 표시
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax.text(bar1.get_x() + bar1.get_width()/2., height1 + max(results_df['필터링후행수'])*0.01,
           f'{int(height1):,}', ha='center', va='bottom', fontsize=9, rotation=45)
    ax.text(bar2.get_x() + bar2.get_width()/2., height2 + max(results_df['필터링후행수'])*0.01,
           f'{int(height2):,}', ha='center', va='bottom', fontsize=9, rotation=45)

plt.tight_layout()
plt.savefig(viz_dir / '02_data_count_changes.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-3. Default 비율 변화 상세 분석
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(results_df['임계값(%)'], results_df['Default비율변화(%)'], 
        marker='o', linewidth=3, markersize=10, color='purple')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax.fill_between(results_df['임계값(%)'], results_df['Default비율변화(%)'], 0, 
                alpha=0.3, color='purple')

ax.set_xlabel('결측치 비율 임계값 (%)')
ax.set_ylabel('Default 비율 변화 (%p)')
ax.set_title('결측치 임계값별 Default 비율 변화량')
ax.grid(True, alpha=0.3)

# 값 표시
for i, row in results_df.iterrows():
    ax.annotate(f'{row["Default비율변화(%)"]:.3f}%p', 
               (row['임계값(%)'], row['Default비율변화(%)']),
               textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(viz_dir / '03_default_rate_changes.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-4. 남은 결측치 분석
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. 남은 결측치 총량
ax1.plot(results_df['임계값(%)'], results_df['남은결측치총개수'], 
         marker='o', linewidth=2, markersize=8, color='purple', label='남은 결측치 총개수')
ax1_twin = ax1.twinx()
ax1_twin.plot(results_df['임계값(%)'], results_df['남은결측치비율(%)'], 
              marker='s', linewidth=2, markersize=8, color='orange', label='남은 결측치 비율')
ax1.set_xlabel('결측치 비율 임계값 (%)')
ax1.set_ylabel('남은 결측치 총개수', color='purple')
ax1_twin.set_ylabel('남은 결측치 비율 (%)', color='orange')
ax1.set_title('임계값별 남은 결측치 총량')
ax1.grid(True, alpha=0.3)

# 2. 데이터 효율성 (데이터 보존율 대비 결측치 감소율)
missing_reduction_rate = 100 - results_df['남은결측치비율(%)']
ax2.scatter(results_df['데이터보존율(%)'], missing_reduction_rate, 
           s=100, alpha=0.7, color='green')
for i, threshold in enumerate(results_df['임계값(%)']):
    ax2.annotate(f'{threshold}%', 
                (results_df['데이터보존율(%)'].iloc[i], missing_reduction_rate.iloc[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax2.set_xlabel('데이터 보존율 (%)')
ax2.set_ylabel('결측치 제거율 (%)')
ax2.set_title('데이터 보존율 vs 결측치 제거율')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / '05_remaining_missing_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 6-5. 컬럼별 결측치 변화 히트맵 (주요 임계값만)
key_thresholds = [0, 20, 40, 60, 80]
column_changes_data = []

for threshold in key_thresholds:
    pattern_data = next((item for item in missing_pattern_results if item['임계값(%)'] == threshold), None)
    if pattern_data and pattern_data['컬럼별결측치통계']:
        for col_stat in pattern_data['컬럼별결측치통계']:
            column_changes_data.append({
                '임계값': f"{threshold}%",
                '컬럼명': col_stat['컬럼명'],
                '결측치변화(%)': col_stat['결측치변화(%)']
            })

if column_changes_data:
    col_changes_df = pd.DataFrame(column_changes_data)
    pivot_data = col_changes_df.pivot(index='컬럼명', columns='임계값', values='결측치변화(%)')
    
    fig, ax = plt.subplots(figsize=(12, 16))
    sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, 
                fmt='.2f', ax=ax, cbar_kws={'label': '결측치 비율 변화 (%)'})
    ax.set_title('임계값별 컬럼별 결측치 비율 변화', fontsize=14, fontweight='bold')
    ax.set_xlabel('결측치 임계값')
    ax.set_ylabel('재무비율 컬럼')
    
    plt.tight_layout()
    plt.savefig(viz_dir / '06_column_missing_changes_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6-6. 종합 대시보드
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. 보존율 비교
ax1.plot(results_df['임계값(%)'], results_df['데이터보존율(%)'], 
         marker='o', linewidth=2, markersize=6, color='blue', label='전체 데이터')
ax1.plot(results_df['임계값(%)'], results_df['Default보존율(%)'], 
         marker='s', linewidth=2, markersize=6, color='red', label='Default 데이터')
ax1.set_title('데이터 보존율 비교')
ax1.set_xlabel('임계값 (%)')
ax1.set_ylabel('보존율 (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 절대 수량 변화
ax2.plot(results_df['임계값(%)'], results_df['필터링후행수'], 
         marker='o', linewidth=2, markersize=6, color='green', label='전체 행수')
ax2_twin = ax2.twinx()
ax2_twin.plot(results_df['임계값(%)'], results_df['필터링후Default수'], 
              marker='s', linewidth=2, markersize=6, color='red', label='Default 수')
ax2.set_title('절대 데이터 수 변화')
ax2.set_xlabel('임계값 (%)')
ax2.set_ylabel('전체 행수', color='green')
ax2_twin.set_ylabel('Default 수', color='red')
ax2.grid(True, alpha=0.3)

# 3. 남은 결측치 비율
ax3.plot(results_df['임계값(%)'], results_df['남은결측치비율(%)'], 
         marker='o', linewidth=2, markersize=6, color='purple', label='남은 결측치 비율')
ax3.set_title('남은 결측치 비율 변화')
ax3.set_xlabel('임계값 (%)')
ax3.set_ylabel('결측치 비율 (%)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Default 비율 변화량
colors = ['red' if x > 0 else 'blue' for x in results_df['Default비율변화(%)']]
ax4.bar(results_df['임계값(%)'], results_df['Default비율변화(%)'], 
        alpha=0.7, color=colors)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax4.set_title('Default 비율 변화량')
ax4.set_xlabel('임계값 (%)')
ax4.set_ylabel('변화량 (%p)')
ax4.grid(True, alpha=0.3)

plt.suptitle('결측치 임계값별 종합 분석 대시보드', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(viz_dir / '04_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. 결과 저장
print("\n5️⃣ 결과 저장")
print("="*50)

# CSV 파일로 저장
results_df.to_csv(reports_dir / 'missing_threshold_default_analysis.csv', 
                 index=False, encoding='utf-8-sig')

# 컬럼별 결측치 변화 상세 데이터 저장
all_column_changes = []
for pattern_data in missing_pattern_results:
    threshold = pattern_data['임계값(%)']
    if pattern_data['컬럼별결측치통계']:
        for col_stat in pattern_data['컬럼별결측치통계']:
            all_column_changes.append({
                '임계값(%)': threshold,
                **col_stat
            })

if all_column_changes:
    column_changes_df = pd.DataFrame(all_column_changes)
    column_changes_df.to_csv(reports_dir / 'column_missing_changes_by_threshold.csv', 
                           index=False, encoding='utf-8-sig')

# 상세 분석 리포트 생성
report_content = f"""
# 결측치 비율별 행 제거 시 Default 분포 분석 리포트

## 1. 분석 개요
- 원본 데이터: {len(df):,}행, {len(feature_columns)}개 재무비율 변수
- 원본 Default 비율: {df['default'].mean():.4f} ({df['default'].sum():,}개)

## 2. 주요 발견사항

### 2.1 최적 임계값 추천
"""

# 최적 임계값 찾기 (데이터 보존율 70% 이상, Default 보존율 최대)
optimal_candidates = results_df[results_df['데이터보존율(%)'] >= 70]
if len(optimal_candidates) > 0:
    optimal_idx = optimal_candidates['Default보존율(%)'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, '임계값(%)']
    report_content += f"- 추천 임계값: {optimal_threshold}%\n"
    report_content += f"- 데이터 보존율: {results_df.loc[optimal_idx, '데이터보존율(%)']:.1f}%\n"
    report_content += f"- Default 보존율: {results_df.loc[optimal_idx, 'Default보존율(%)']:.1f}%\n"
    report_content += f"- Default 비율 변화: {results_df.loc[optimal_idx, 'Default비율변화(%)']:+.3f}%p\n\n"

report_content += f"""
### 2.2 극단적 시나리오 분석
- 0% 임계값 (결측치 없는 행만): {results_df.iloc[0]['데이터보존율(%)']:.1f}% 데이터 보존
- 50% 임계값 (절반 이상 결측치 허용): {results_df.iloc[-1]['데이터보존율(%)']:.1f}% 데이터 보존

### 2.3 Default 비율 변화 패턴
"""

# Default 비율 변화 패턴 분석
increasing_thresholds = results_df[results_df['Default비율변화(%)'] > 0]
decreasing_thresholds = results_df[results_df['Default비율변화(%)'] < 0]

if len(increasing_thresholds) > 0:
    report_content += f"- Default 비율 증가 구간: {increasing_thresholds['임계값(%)'].min()}-{increasing_thresholds['임계값(%)'].max()}%\n"
if len(decreasing_thresholds) > 0:
    report_content += f"- Default 비율 감소 구간: {decreasing_thresholds['임계값(%)'].min()}-{decreasing_thresholds['임계값(%)'].max()}%\n"

report_content += f"""

## 3. 세부 결과표

{results_df.to_string(index=False)}

## 4. 권장사항
1. 데이터 품질과 모델 성능의 균형을 고려하여 임계값 선택
2. Default 비율 변화가 적은 구간에서 적절한 임계값 선택
3. 도메인 전문가와 협의하여 최종 임계값 결정
"""

# 리포트 저장
with open(reports_dir / 'missing_threshold_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"✅ 분석 완료!")
print(f"📊 주요 결과 CSV: {reports_dir / 'missing_threshold_default_analysis.csv'}")
print(f"📊 컬럼별 결측치 변화: {reports_dir / 'column_missing_changes_by_threshold.csv'}")
print(f"📄 상세 리포트: {reports_dir / 'missing_threshold_analysis_report.txt'}")
print(f"📈 시각화 파일: {viz_dir}/ (6개 차트)")
print(f"\n🎯 주요 발견사항:")
print(f"- 원본 데이터: {len(df):,}행, Default 비율: {df['default'].mean():.4f}")

# 주요 임계값들의 결과 출력 (결측치 정보 포함)
key_thresholds = [0, 20, 40, 60]
for threshold in key_thresholds:
    if threshold in results_df['임계값(%)'].values:
        row = results_df[results_df['임계값(%)'] == threshold].iloc[0]
        print(f"- {threshold}% 임계값:")
        print(f"  데이터 보존: {row['데이터보존율(%)']:.1f}%, Default 보존: {row['Default보존율(%)']:.1f}%")
        print(f"  Default 비율 변화: {row['Default비율변화(%)']:+.3f}%p")
        print(f"  남은 결측치: {row['남은결측치총개수']:,}개 ({row['남은결측치비율(%)']:.2f}%)")

print(f"\n📈 추가 생성된 시각화:")
print(f"   05_remaining_missing_analysis.png : 남은 결측치 분석")
print(f"   06_column_missing_changes_heatmap.png : 컬럼별 결측치 변화 히트맵")