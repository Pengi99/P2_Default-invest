import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import os
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

# 프로젝트 루트 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# 데이터 파일 경로 설정 (기본값: FS_filtered.csv, 필요시 수정 가능)
# 다른 파일 분석하려면 아래 경로를 변경하세요
# 예: 'FS.csv', 'FS_filtered.csv', 'custom_data.csv' 등
data_filename = 'FS2_filtered.csv'  # 👈 여기서 파일명 변경 가능
data_path = os.path.join(project_root, 'data', 'processed', data_filename)
fs_ratio = pd.read_csv(data_path, dtype={'거래소코드': str})
print(f"분석 데이터 ({data_filename}): {fs_ratio.shape}")

# 재무비율 컬럼만 추출
ratio_columns = [col for col in fs_ratio.columns 
                if col not in ['회사명', '거래소코드', '회계년도']]

print(f"분석할 재무비율: {len(ratio_columns)}개")
print(f"비율 목록: {ratio_columns}")

# 2. 결측치 분석
print("\n2️⃣ 결측치 분석")
print("="*50)

# 출력 폴더 생성
output_base = os.path.join(project_root, 'outputs/analysis/scaling_needs')
viz_base = os.path.join(output_base, 'visualizations')
reports_base = os.path.join(output_base, 'reports')

# 하위 디렉토리 생성
missing_dir = os.path.join(viz_base, 'missing_analysis')
dist_dir = os.path.join(viz_base, 'distributions')
box_dir = os.path.join(viz_base, 'boxplots')
scaling_dir = os.path.join(viz_base, 'scaling_indicators')
comp_dir = os.path.join(viz_base, 'comprehensive')

os.makedirs(missing_dir, exist_ok=True)
os.makedirs(dist_dir, exist_ok=True)
os.makedirs(box_dir, exist_ok=True)
os.makedirs(scaling_dir, exist_ok=True)
os.makedirs(comp_dir, exist_ok=True)
os.makedirs(reports_base, exist_ok=True)

# 2-1. 결측치 기본 통계
total_rows = len(fs_ratio)
missing_stats = []

for col in ratio_columns:
    missing_count = fs_ratio[col].isnull().sum()
    missing_rate = (missing_count / total_rows) * 100
    valid_count = total_rows - missing_count
    valid_rate = (valid_count / total_rows) * 100
    
    missing_stats.append({
        '변수명': col,
        '전체행수': total_rows,
        '결측치수': missing_count,
        '결측치비율(%)': missing_rate,
        '유효데이터수': valid_count,
        '유효데이터비율(%)': valid_rate
    })

missing_df = pd.DataFrame(missing_stats)
missing_df = missing_df.sort_values('결측치비율(%)', ascending=False)

print("📊 결측치 현황 요약:")
print(f"전체 변수 수: {len(ratio_columns)}개")
print(f"전체 관측치 수: {total_rows:,}개")
print(f"결측치 없는 변수: {len(missing_df[missing_df['결측치비율(%)'] == 0])}개")
print(f"결측치 있는 변수: {len(missing_df[missing_df['결측치비율(%)'] > 0])}개")

# 2-2. 결측치 비율별 변수 분류
no_missing = missing_df[missing_df['결측치비율(%)'] == 0]
low_missing = missing_df[(missing_df['결측치비율(%)'] > 0) & (missing_df['결측치비율(%)'] <= 5)]
medium_missing = missing_df[(missing_df['결측치비율(%)'] > 5) & (missing_df['결측치비율(%)'] <= 20)]
high_missing = missing_df[missing_df['결측치비율(%)'] > 20]

print(f"\n📈 결측치 비율별 분류:")
print(f"🟢 결측치 없음 (0%): {len(no_missing)}개")
for var in no_missing['변수명'].tolist():
    print(f"   - {var}")

print(f"\n🟡 낮은 결측치 (0-5%): {len(low_missing)}개")
for _, row in low_missing.iterrows():
    print(f"   - {row['변수명']:25} : {row['결측치비율(%)']:5.2f}%")

print(f"\n🟠 중간 결측치 (5-20%): {len(medium_missing)}개")
for _, row in medium_missing.iterrows():
    print(f"   - {row['변수명']:25} : {row['결측치비율(%)']:5.2f}%")

print(f"\n🔴 높은 결측치 (>20%): {len(high_missing)}개")
for _, row in high_missing.iterrows():
    print(f"   - {row['변수명']:25} : {row['결측치비율(%)']:5.2f}%")

# 2-3. 결측치 패턴 분석
print(f"\n🔍 결측치 패턴 분석:")

# 완전한 관측치 (모든 변수에 값이 있는 행)
complete_cases = fs_ratio[ratio_columns].dropna()
complete_rate = (len(complete_cases) / total_rows) * 100

print(f"완전한 관측치 (모든 변수 유효): {len(complete_cases):,}개 ({complete_rate:.2f}%)")
print(f"불완전한 관측치 (일부 변수 결측): {total_rows - len(complete_cases):,}개 ({100-complete_rate:.2f}%)")

# 결측치 조합 패턴 분석 (상위 10개)
missing_pattern = fs_ratio[ratio_columns].isnull()
pattern_counts = missing_pattern.value_counts().head(10)

print(f"\n🔢 주요 결측치 패턴 (상위 10개):")
for i, (pattern, count) in enumerate(pattern_counts.items(), 1):
    rate = (count / total_rows) * 100
    missing_vars = [col for col, is_missing in zip(ratio_columns, pattern) if is_missing]
    print(f"{i:2d}. {count:,}개 ({rate:.2f}%) - 결측변수: {len(missing_vars)}개")
    if len(missing_vars) <= 5:
        print(f"     {missing_vars}")
    else:
        print(f"     {missing_vars[:3]} ... (총 {len(missing_vars)}개)")

# 2-4. 결측치 시각화
print(f"\n📊 결측치 시각화 생성 중...")

# 2-4-1. 결측치 비율 막대그래프
fig, ax = plt.subplots(figsize=(16, 8))

colors = ['red' if rate > 20 else 'orange' if rate > 5 else 'yellow' if rate > 0 else 'green' 
          for rate in missing_df['결측치비율(%)']]

bars = ax.bar(range(len(missing_df)), missing_df['결측치비율(%)'], 
              color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_xlabel('재무비율 변수', fontsize=12)
ax.set_ylabel('결측치 비율 (%)', fontsize=12)
ax.set_title('재무비율별 결측치 비율\n(빨강: >20%, 주황: 5-20%, 노랑: 0-5%, 초록: 0%)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(range(len(missing_df)))
ax.set_xticklabels(missing_df['변수명'], rotation=45, ha='right')
ax.grid(True, alpha=0.3)

# 임계선 표시
ax.axhline(y=20, color='red', linestyle=':', alpha=0.7, linewidth=2, label='높은 결측치 (20%)')
ax.axhline(y=5, color='orange', linestyle=':', alpha=0.7, linewidth=2, label='중간 결측치 (5%)')
ax.legend()

# 비율 표시 (결측치가 있는 변수만)
for i, bar in enumerate(bars):
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(missing_dir, '01_missing_rates_by_variable.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

# 2-4-2. 결측치 히트맵 (샘플링)
print("결측치 패턴 히트맵 생성 중...")
fig, ax = plt.subplots(figsize=(16, 10))

# 데이터가 너무 크므로 샘플링 (최대 1000행)
sample_size = min(1000, len(fs_ratio))
sample_indices = np.random.choice(len(fs_ratio), sample_size, replace=False)
sample_data = fs_ratio.iloc[sample_indices][ratio_columns]

# 결측치를 1, 유효값을 0으로 변환
missing_matrix = sample_data.isnull().astype(int)

# 결측치 비율이 높은 순으로 변수 정렬
sorted_cols = missing_df['변수명'].tolist()
missing_matrix_sorted = missing_matrix[sorted_cols]

sns.heatmap(missing_matrix_sorted.T, cmap='RdYlGn_r', cbar_kws={'label': '결측치 (1=결측, 0=유효)'}, 
            ax=ax, xticklabels=False, yticklabels=True)
ax.set_title(f'결측치 패턴 히트맵 (샘플 {sample_size}개 관측치)', fontsize=14, fontweight='bold')
ax.set_xlabel('관측치 (샘플)', fontsize=12)
ax.set_ylabel('재무비율 변수', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(missing_dir, '02_missing_pattern_heatmap.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

# 2-4-3. 결측치 비율 분포 파이차트
print("결측치 분포 파이차트 생성 중...")
fig, ax = plt.subplots(figsize=(10, 8))

category_counts = {
    '결측치 없음 (0%)': len(no_missing),
    '낮은 결측치 (0-5%)': len(low_missing),
    '중간 결측치 (5-20%)': len(medium_missing),
    '높은 결측치 (>20%)': len(high_missing)
}

colors_pie = ['green', 'yellow', 'orange', 'red']
wedges, texts, autotexts = ax.pie(category_counts.values(), labels=category_counts.keys(),
                                  autopct='%1.1f%%', colors=colors_pie, startangle=90,
                                  textprops={'fontsize': 11}, explode=(0.05, 0.02, 0.02, 0.1))

ax.set_title('재무비율 변수의 결측치 수준 분포', fontsize=14, fontweight='bold')

# 개수 정보 추가
for i, (label, count) in enumerate(category_counts.items()):
    autotexts[i].set_text(f'{count}개\n({count/len(ratio_columns)*100:.1f}%)')
    autotexts[i].set_fontweight('bold')

plt.tight_layout()
plt.savefig(os.path.join(missing_dir, '03_missing_level_distribution.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

# 2-4-4. 완전성 분석 (변수별 유효 데이터 수)
print("데이터 완전성 분석 차트 생성 중...")
fig, ax = plt.subplots(figsize=(16, 8))

bars = ax.bar(range(len(missing_df)), missing_df['유효데이터수'], 
              color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_xlabel('재무비율 변수', fontsize=12)
ax.set_ylabel('유효 데이터 수', fontsize=12)
ax.set_title('재무비율별 유효 데이터 수', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(missing_df)))
ax.set_xticklabels(missing_df['변수명'], rotation=45, ha='right')
ax.grid(True, alpha=0.3)

# 평균선 표시
mean_valid = missing_df['유효데이터수'].mean()
ax.axhline(y=mean_valid, color='red', linestyle='--', alpha=0.7, linewidth=2, 
          label=f'평균 유효데이터 수: {mean_valid:,.0f}')
ax.legend()

# 유효 데이터 수 표시 (하위 10개만)
bottom_10_indices = missing_df['유효데이터수'].nsmallest(10).index
for i in range(len(missing_df)):
    if i in bottom_10_indices:
        height = bars[i].get_height()
        ax.text(bars[i].get_x() + bars[i].get_width()/2., height + max(missing_df['유효데이터수'])*0.01,
               f'{int(height):,}', ha='center', va='bottom', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(missing_dir, '04_valid_data_counts.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

print("✅ 결측치 분석 및 시각화 완료")

# 3. 기초 통계량 계산
print("\n3️⃣ 기초 통계량 계산")
print("="*50)

stats_results = []
for col in ratio_columns:
    data = fs_ratio[col].dropna()
    
    if len(data) > 0:
        stats = {
            '비율명': col,
            '데이터수': len(data),
            '결측치수': fs_ratio[col].isnull().sum(),
            '결측치비율(%)': (fs_ratio[col].isnull().sum() / len(fs_ratio)) * 100,
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

# 4. 분포 시각화 (히스토그램 + 박스플롯)
print("\n4️⃣ 분포 시각화")
print("="*50)

# 3-1. 각 비율별 개별 히스토그램
print("개별 히스토그램 생성 중...")
for i, col in enumerate(ratio_columns):
    data = fs_ratio[col].dropna()
    
    if len(data) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'{col}\n평균: {data.mean():.4f}, 표준편차: {data.std():.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('값')
        ax.set_ylabel('빈도')
        ax.grid(True, alpha=0.3)
        
        # 분위수 선 추가 (0.5%, 2.5%, 97.5%, 99.5%)
        q05_pct = data.quantile(0.005)
        q25_pct = data.quantile(0.025)
        q975_pct = data.quantile(0.975)
        q995_pct = data.quantile(0.995)
        
        ax.axvline(q05_pct, color='red', linestyle=':', alpha=0.8, linewidth=2, label='0.5%/99.5% 분위수')
        ax.axvline(q995_pct, color='red', linestyle=':', alpha=0.8, linewidth=2)
        ax.axvline(q25_pct, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='2.5%/97.5% 분위수')
        ax.axvline(q975_pct, color='orange', linestyle='--', alpha=0.8, linewidth=2)
        
        # 중앙값 선 추가
        median_val = data.median()
        ax.axvline(median_val, color='green', linestyle='-', alpha=0.8, linewidth=2, label='중앙값')
        
        ax.legend()
        
        # 통계 정보 텍스트 박스
        stats_text = f'데이터 수: {len(data):,}\n중앙값: {median_val:.4f}\n0.5%: {q05_pct:.4f}\n2.5%: {q25_pct:.4f}\n97.5%: {q975_pct:.4f}\n99.5%: {q995_pct:.4f}\n왜도: {data.skew():.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        safe_filename = col.replace('/', '_').replace('%', 'pct').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(dist_dir, f'{i+1:02d}_{safe_filename}_hist.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

print(f"✅ {len(ratio_columns)}개 히스토그램 저장 완료")

# 3-2. 각 비율별 개별 박스플롯
print("개별 박스플롯 생성 중...")
for i, col in enumerate(ratio_columns):
    data = fs_ratio[col].dropna()
    
    if len(data) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 원본 데이터 박스플롯
        ax1.boxplot([data], labels=[col])
        ax1.set_title(f'{col} - 원본 데이터', fontsize=12, fontweight='bold')
        ax1.set_ylabel('값')
        ax1.grid(True, alpha=0.3)
        
        # 정규화된 데이터 박스플롯
        min_val = data.min()
        max_val = data.max()
        if max_val != min_val:
            normalized = (data - min_val) / (max_val - min_val)
        else:
            normalized = data
            
        ax2.boxplot([normalized], labels=[col])
        ax2.set_title(f'{col} - 정규화된 데이터', fontsize=12, fontweight='bold')
        ax2.set_ylabel('정규화된 값 (0-1)')
        ax2.grid(True, alpha=0.3)
        
        # 통계 정보
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum()
        
        stats_text = f'Q1: {q1:.4f}\nQ3: {q3:.4f}\nIQR: {iqr:.4f}\n이상치: {outliers}개'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        safe_filename = col.replace('/', '_').replace('%', 'pct').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(box_dir, f'{i+1:02d}_{safe_filename}_box.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

print(f"✅ {len(ratio_columns)}개 박스플롯 저장 완료")

# 3-3. 전체 요약 히스토그램 (4x4 그리드)
print("전체 요약 히스토그램 생성 중...")
n_cols = 4
n_rows = (len(ratio_columns) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(ratio_columns):
    data = fs_ratio[col].dropna()
    
    if len(data) > 0:
        axes[i].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'{col}', fontsize=9)
        axes[i].tick_params(axis='both', which='major', labelsize=8)
        axes[i].grid(True, alpha=0.3)

# 빈 subplot 제거
for i in range(len(ratio_columns), len(axes)):
    fig.delaxes(axes[i])

plt.suptitle('전체 재무비율 분포 요약', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(viz_base, '00_ratio_distributions_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3-4. 전체 요약 박스플롯
print("전체 요약 박스플롯 생성 중...")
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
plt.savefig(os.path.join(viz_base, '00_ratio_boxplots_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✅ 요약 차트 저장 완료")

# 5. 스케일링 필요성 지표 시각화
print("\n5️⃣ 스케일링 필요성 지표 시각화")
print("="*50)

# 4-1. 변동계수 vs 왜도 산점도
print("변동계수 vs 왜도 산점도 생성 중...")
fig, ax = plt.subplots(figsize=(12, 8))

cv_values = stats_df['변동계수'].replace([np.inf, -np.inf], np.nan).dropna()
skew_values = stats_df.loc[stats_df['변동계수'].replace([np.inf, -np.inf], np.nan).notna(), '왜도']

ax.scatter(cv_values, skew_values, alpha=0.7, s=80, color='steelblue', edgecolors='black', linewidth=0.5)
ax.set_xlabel('변동계수 (CV)', fontsize=12)
ax.set_ylabel('왜도 (Skewness)', fontsize=12)
ax.set_title('변동계수 vs 왜도 - 스케일링 필요성 분석', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 스케일링 필요 영역 표시
ax.axhline(y=3, color='red', linestyle='--', alpha=0.7, linewidth=2, label='왜도 임계값 (±3)')
ax.axhline(y=-3, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax.axvline(x=2, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='CV 임계값 (2)')
ax.legend(fontsize=11)

# 각 점에 비율명 표시 (CV > 5 또는 |왜도| > 5인 경우만)
for i, row in stats_df.iterrows():
    cv = row['변동계수']
    skew = row['왜도']
    if not np.isinf(cv) and (cv > 5 or abs(skew) > 5):
        ax.annotate(row['비율명'], (cv, skew), xytext=(8, 8), 
                   textcoords='offset points', fontsize=9, alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(scaling_dir, '01_cv_vs_skewness.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4-2. 범위 vs 첨도
print("범위 vs 첨도 산점도 생성 중...")
fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(stats_df['범위'], stats_df['첨도'], alpha=0.7, s=80, color='forestgreen', edgecolors='black', linewidth=0.5)
ax.set_xlabel('범위 (Range)', fontsize=12)
ax.set_ylabel('첨도 (Kurtosis)', fontsize=12)
ax.set_title('범위 vs 첨도 - 스케일링 필요성 분석', fontsize=14, fontweight='bold')
ax.set_xscale('log')  # 로그 스케일로 표시
ax.grid(True, alpha=0.3)

# 스케일링 필요 영역 표시
ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='첨도 임계값 (10)')
ax.axvline(x=1000, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='범위 임계값 (1000)')
ax.legend(fontsize=11)

# 문제 변수 표시
for i, row in stats_df.iterrows():
    if row['범위'] > 1000 or abs(row['첨도']) > 10:
        ax.annotate(row['비율명'], (row['범위'], row['첨도']), xytext=(8, 8),
                   textcoords='offset points', fontsize=9, alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(scaling_dir, '02_range_vs_kurtosis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4-3. 평균의 절댓값 분포
print("평균의 절댓값 분포 생성 중...")
fig, ax = plt.subplots(figsize=(14, 8))

mean_abs = stats_df['평균'].abs()
bars = ax.bar(range(len(stats_df)), mean_abs, alpha=0.7, color='purple', edgecolor='black', linewidth=0.5)
ax.set_xlabel('재무비율', fontsize=12)
ax.set_ylabel('평균의 절댓값', fontsize=12)
ax.set_title('재무비율별 평균의 절댓값 분포', fontsize=14, fontweight='bold')
ax.set_yscale('log')  # 로그 스케일로 표시
ax.grid(True, alpha=0.3)

# x축에 비율명 표시
ax.set_xticks(range(len(stats_df)))
ax.set_xticklabels(stats_df['비율명'], rotation=45, ha='right')

# 값 표시 (상위 5개만)
top_5_indices = mean_abs.nlargest(5).index
for i in top_5_indices:
    height = bars[i].get_height()
    ax.text(bars[i].get_x() + bars[i].get_width()/2., height*1.1,
           f'{height:.2e}', ha='center', va='bottom', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(scaling_dir, '03_mean_abs_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4-4. 표준편차 분포
print("표준편차 분포 생성 중...")
fig, ax = plt.subplots(figsize=(14, 8))

bars = ax.bar(range(len(stats_df)), stats_df['표준편차'], alpha=0.7, color='darkorange', edgecolor='black', linewidth=0.5)
ax.set_xlabel('재무비율', fontsize=12)
ax.set_ylabel('표준편차', fontsize=12)
ax.set_title('재무비율별 표준편차 분포', fontsize=14, fontweight='bold')
ax.set_yscale('log')  # 로그 스케일로 표시
ax.grid(True, alpha=0.3)

# x축에 비율명 표시
ax.set_xticks(range(len(stats_df)))
ax.set_xticklabels(stats_df['비율명'], rotation=45, ha='right')

# 값 표시 (상위 5개만)
top_5_indices = stats_df['표준편차'].nlargest(5).index
for i in top_5_indices:
    height = bars[i].get_height()
    ax.text(bars[i].get_x() + bars[i].get_width()/2., height*1.1,
           f'{height:.2e}', ha='center', va='bottom', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(scaling_dir, '04_std_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✅ 스케일링 지표 차트 4개 저장 완료")

# 6. 스케일링 필요성 종합 점수 계산
print("\n6️⃣ 스케일링 필요성 종합 점수 계산")
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

# 7. 스케일링 필요성 종합 시각화
print("\n7️⃣ 스케일링 필요성 종합 시각화")
print("="*50)

# 폴더 생성
os.makedirs(comp_dir, exist_ok=True)

# 6-1. 스케일링 점수 막대그래프
print("스케일링 점수 막대그래프 생성 중...")
fig, ax = plt.subplots(figsize=(16, 8))

colors = ['red' if score >= 7 else 'orange' if score >= 4 else 'green' 
          for score in scaling_score_df['스케일링점수']]

bars = ax.bar(range(len(scaling_score_df)), scaling_score_df['스케일링점수'], 
               color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xlabel('재무비율', fontsize=12)
ax.set_ylabel('스케일링 필요성 점수', fontsize=12)
ax.set_title('재무비율별 스케일링 필요성 점수\n(빨강: High≥7, 주황: Medium≥4, 초록: Low<4)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(range(len(scaling_score_df)))
ax.set_xticklabels(scaling_score_df['비율명'], rotation=45, ha='right')
ax.grid(True, alpha=0.3)

# 점수 표시
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 임계선 표시
ax.axhline(y=7, color='red', linestyle=':', alpha=0.7, linewidth=2, label='High 임계값 (7)')
ax.axhline(y=4, color='orange', linestyle=':', alpha=0.7, linewidth=2, label='Medium 임계값 (4)')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(comp_dir, '01_scaling_scores.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6-2. 우선순위별 분포 파이차트
print("우선순위별 파이차트 생성 중...")
priority_counts = pd.Series(scaling_score_df['우선순위'].value_counts())

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['red', 'orange', 'lightgreen']
ax.pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%', 
       colors=colors, startangle=90, textprops={'fontsize': 12})
ax.set_title('스케일링 우선순위별 변수 분포', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(comp_dir, '02_priority_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6-3. 상관관계 히트맵
print("상관관계 히트맵 생성 중...")
corr_matrix = fs_ratio[ratio_columns].corr()

fig, ax = plt.subplots(figsize=(20, 16))
mask = np.triu(np.ones_like(corr_matrix))  # 상삼각행렬 마스킹
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title('재무비율 간 상관관계 히트맵', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(comp_dir, '03_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6-3-1. VIF 다중공선성 분석
print(f"VIF 다중공선성 분석 중 ({data_filename})...")
vif_data = []

# 완전한 관측치만 사용
complete_data = fs_ratio[ratio_columns].dropna()
print(f"VIF 분석용 완전한 관측치 ({data_filename}): {len(complete_data):,}개")

if len(complete_data) > 0:
    # 데이터 표준화 (VIF 계산 안정성을 위해)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(complete_data)
    scaled_df = pd.DataFrame(scaled_data, columns=ratio_columns)
    
    # VIF 계산
    for i, col in enumerate(ratio_columns):
        try:
            vif_value = variance_inflation_factor(scaled_df.values, i)
            vif_data.append({
                '변수명': col,
                'VIF': vif_value,
                '다중공선성': 'High' if vif_value > 10 else 'Medium' if vif_value > 5 else 'Low'
            })
        except Exception as e:
            print(f"VIF 계산 오류 ({col}): {e}")
            vif_data.append({
                '변수명': col,
                'VIF': np.nan,
                '다중공선성': 'Error'
            })
    
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('VIF', ascending=False)
    
    # VIF 결과 출력
    print(f"\n📊 VIF 다중공선성 분석 결과 ({data_filename}):")
    print("VIF > 10: 높은 다중공선성")
    print("VIF 5-10: 중간 다중공선성")
    print("VIF < 5: 낮은 다중공선성")
    print("\n상위 20개 VIF 값:")
    print(vif_df.head(20))
    
    # VIF 시각화
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = ['red' if vif > 10 else 'orange' if vif > 5 else 'green' 
              for vif in vif_df['VIF']]
    
    bars = ax.bar(range(len(vif_df)), vif_df['VIF'], 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('재무비율 변수', fontsize=12)
    ax.set_ylabel('VIF (Variance Inflation Factor)', fontsize=12)
    ax.set_title(f'VIF 다중공선성 분석 ({data_filename})\n(빨강: >10, 주황: 5-10, 초록: <5)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(vif_df)))
    ax.set_xticklabels(vif_df['변수명'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # VIF 임계선 표시
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='높은 다중공선성 (VIF=10)')
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='중간 다중공선성 (VIF=5)')
    ax.legend()
    
    # VIF 값 표시 (상위 10개만)
    top_10_vif = vif_df.head(10)
    for i in range(min(10, len(vif_df))):
        if not np.isnan(vif_df.iloc[i]['VIF']):
            height = bars[i].get_height()
            if height > 5:  # VIF > 5인 경우만 표시
                ax.text(bars[i].get_x() + bars[i].get_width()/2., height + max(vif_df['VIF'])*0.01,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, '05_vif_multicollinearity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 높은 VIF 변수들 분석
    high_vif = vif_df[vif_df['VIF'] > 10]
    medium_vif = vif_df[(vif_df['VIF'] > 5) & (vif_df['VIF'] <= 10)]
    low_vif = vif_df[vif_df['VIF'] <= 5]
    
    print(f"\n🔴 높은 다중공선성 (VIF > 10): {len(high_vif)}개")
    for _, row in high_vif.iterrows():
        print(f"   - {row['변수명']:25} : VIF = {row['VIF']:.2f}")
    
    print(f"\n🟠 중간 다중공선성 (VIF 5-10): {len(medium_vif)}개")
    for _, row in medium_vif.iterrows():
        print(f"   - {row['변수명']:25} : VIF = {row['VIF']:.2f}")
    
    print(f"\n🟢 낮은 다중공선성 (VIF < 5): {len(low_vif)}개")
    
    # VIF와 상관관계 결합 분석
    print(f"\n📈 VIF-상관관계 결합 분석 ({data_filename}):")
    high_corr_pairs = []
    
    # 높은 상관관계 찾기 (|r| > 0.8)
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.8:
                var1 = corr_matrix.index[i]
                var2 = corr_matrix.columns[j]
                vif1 = vif_df[vif_df['변수명'] == var1]['VIF'].iloc[0] if len(vif_df[vif_df['변수명'] == var1]) > 0 else np.nan
                vif2 = vif_df[vif_df['변수명'] == var2]['VIF'].iloc[0] if len(vif_df[vif_df['변수명'] == var2]) > 0 else np.nan
                
                high_corr_pairs.append({
                    '변수1': var1,
                    '변수2': var2,
                    '상관계수': corr_val,
                    'VIF1': vif1,
                    'VIF2': vif2
                })
    
    if high_corr_pairs:
        corr_pairs_df = pd.DataFrame(high_corr_pairs)
        corr_pairs_df = corr_pairs_df.sort_values('상관계수', ascending=False)
        
        print(f"높은 상관관계 쌍 (|r| > 0.8): {len(corr_pairs_df)}개")
        for _, row in corr_pairs_df.iterrows():
            print(f"   {row['변수1']} ↔ {row['변수2']}")
            print(f"      상관계수: {row['상관계수']:.3f}, VIF: {row['VIF1']:.2f} / {row['VIF2']:.2f}")
    
    # VIF 결과를 CSV로 저장
    vif_filename = f'vif_analysis_{data_filename.replace(".csv", "")}.csv'
    corr_filename = f'high_correlation_pairs_{data_filename.replace(".csv", "")}.csv'
    vif_df.to_csv(os.path.join(reports_base, vif_filename), index=False, encoding='utf-8-sig')
    if high_corr_pairs:
        corr_pairs_df.to_csv(os.path.join(reports_base, corr_filename), index=False, encoding='utf-8-sig')
    
else:
    print(f"⚠️ 완전한 관측치가 없어 VIF 분석을 수행할 수 없습니다 ({data_filename}).")
    vif_df = pd.DataFrame(columns=['변수명', 'VIF', '다중공선성'])


# 6-4. 이상치 개수 막대그래프
print("이상치 개수 막대그래프 생성 중...")
fig, ax = plt.subplots(figsize=(16, 8))

outlier_counts = []
outlier_rates = []
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
        outlier_rates.append(outliers / len(data) * 100)
    else:
        outlier_counts.append(0)
        outlier_rates.append(0)

# 이상치 비율에 따른 색상 설정
colors = ['red' if rate > 15 else 'orange' if rate > 10 else 'green' for rate in outlier_rates]

bars = ax.bar(range(len(ratio_columns)), outlier_counts, alpha=0.8, color=colors, 
              edgecolor='black', linewidth=0.5)
ax.set_xlabel('재무비율', fontsize=12)
ax.set_ylabel('이상치 개수', fontsize=12)
ax.set_title('재무비율별 이상치 개수 (IQR 방법)\n(빨강: >15%, 주황: >10%, 초록: ≤10%)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(range(len(ratio_columns)))
ax.set_xticklabels(ratio_columns, rotation=45, ha='right')
ax.grid(True, alpha=0.3)

# 이상치 비율 표시 (상위 10개만)
top_10_indices = pd.Series(outlier_rates).nlargest(10).index
for i in top_10_indices:
    height = bars[i].get_height()
    ax.text(bars[i].get_x() + bars[i].get_width()/2., height + max(outlier_counts)*0.01,
           f'{outlier_rates[i]:.1f}%', ha='center', va='bottom', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(comp_dir, '04_outlier_counts.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✅ 종합 분석 차트 5개 저장 완료 (VIF 1개 포함)")

# 8. 스케일링 방법 추천
print("\n8️⃣ 스케일링 방법 추천")
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

# CSV 파일 저장
missing_df.to_csv(os.path.join(reports_base, 'missing_analysis.csv'), index=False, encoding='utf-8-sig')
stats_df.to_csv(os.path.join(reports_base, 'basic_statistics.csv'), index=False, encoding='utf-8-sig')
scaling_score_df.to_csv(os.path.join(reports_base, 'scaling_scores.csv'), index=False, encoding='utf-8-sig')
recommend_df.to_csv(os.path.join(reports_base, 'scaling_recommendations.csv'), index=False, encoding='utf-8-sig')
# VIF 결과 저장 - 이미 위에서 저장됨

print(f"✅ 상세 분석 결과 저장:")
print(f"   📄 outputs/reports/missing_analysis.csv : 결측치 분석 결과")
print(f"   📄 outputs/reports/basic_statistics.csv : 기초 통계량")
print(f"   📄 outputs/reports/scaling_scores.csv : 스케일링 점수")
print(f"   📄 outputs/reports/scaling_recommendations.csv : 스케일링 추천")

# VIF 결과 파일 안내
if 'vif_df' in locals() and len(vif_df) > 0:
    print(f"   📄 outputs/reports/vif_analysis_FS.csv : VIF 분석 (FS.csv)")
    if 'high_corr_pairs' in locals() and high_corr_pairs:
        print(f"   📄 outputs/reports/high_correlation_pairs_FS.csv : 높은 상관관계 (FS.csv)")

print(f"✅ 시각화 파일 저장:")
print(f"   📁 missing_analysis/ : 4개 결측치 분석 차트")
print(f"   📁 distributions/ : {len(ratio_columns)}개 개별 히스토그램")
print(f"   📁 boxplots/ : {len(ratio_columns)}개 개별 박스플롯")
print(f"   📁 scaling_indicators/ : 4개 스케일링 지표")
print(f"   📁 comprehensive/ : 6개 종합 분석 (VIF 2개 포함)")
print(f"   📄 00_ratio_distributions_summary.png : 전체 히스토그램 요약")
print(f"   📄 00_ratio_boxplots_summary.png : 전체 박스플롯 요약")

total_charts = len(ratio_columns)*2 + 16  # VIF 차트 2개 추가
print(f"\n🎯 분석 완료! 총 {total_charts}개의 시각화 파일이 생성되었습니다.")

# FS.csv VIF 현황
if 'vif_df' in locals() and len(vif_df) > 0:
    high_vif_count = len(vif_df[vif_df['VIF'] > 10])
    medium_vif_count = len(vif_df[(vif_df['VIF'] > 5) & (vif_df['VIF'] <= 10)])
    low_vif_count = len(vif_df[vif_df['VIF'] <= 5])
    print(f"🔗 다중공선성 현황 (FS.csv): 높음 {high_vif_count}개, 중간 {medium_vif_count}개, 낮음 {low_vif_count}개")

print(f"📈 고우선순위: {len(high_priority)}개, 중우선순위: {len(medium_priority)}개, 저우선순위: {len(low_priority)}개")
print(f"📊 결측치 현황: 결측치 없음 {len(no_missing)}개, 낮은 결측치 {len(low_missing)}개, 중간 결측치 {len(medium_missing)}개, 높은 결측치 {len(high_missing)}개")
print(f"📂 파일 구조:")
print(f"   outputs/visualizations/")
print(f"   ├── missing_analysis/  : 결측치 분석")
print(f"   ├── distributions/     : 개별 히스토그램")
print(f"   ├── boxplots/          : 개별 박스플롯")
print(f"   ├── scaling_indicators/ : 스케일링 지표")
print(f"   ├── comprehensive/     : 종합 분석 (VIF 2개 포함)")
print(f"   ├── 00_ratio_distributions_summary.png")
print(f"   └── 00_ratio_boxplots_summary.png") 