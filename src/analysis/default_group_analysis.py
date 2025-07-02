import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
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

print("=== Default 그룹별 재무비율 통계 분석 (스케일링 포함) ===")

# 1. 프로젝트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
data_path = project_root / 'data' / 'processed' / 'FS2_filtered.csv'
output_base = project_root / 'outputs' / 'analysis' / 'default_group_analysis'

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

# Default 분포 확인
print(f"\nDefault 분포:")
print(df['default'].value_counts())
print(f"Default 비율: {df['default'].mean():.4f}")

# 재무비율 컬럼만 추출 (타겟 변수 제외)
feature_columns = [col for col in df.columns 
                  if col not in ['회사명', '거래소코드', '회계년도', 'default']]
print(f"\n재무비율 변수 수: {len(feature_columns)}개")

# 2-1. 변수 타입 분류
print("\n📊 변수 타입 분류")
print("="*30)

# 절댓값이 큰 변수들 (원화 단위, 주식 수, 종업원 수 등)
absolute_value_vars = [
    '총자산', '총부채', '총자본', '발행주식총수', '유동자산', '유동부채', 
    '매출액', '자본금', '이익잉여금', '영업이익', '당기순이익', '영업현금흐름',
    '법인세비용차감전손익', '인건비', '금융비용', '임차료', '세금과공과', 
    '감가상각비', '종업원수', '기업가치', 'EBITDA', '부가가치',
    '종업원당부가가치', '종업원당매출액', '종업원당영업이익', '종업원당순이익', 
    '종업원당인건비', '주당매출액', '주당순이익', '주당현금흐름', '주당순자산', 
    '주당영업이익', '주당EBITDA'
]

# 실제 데이터에 존재하는 절댓값 변수들만 필터링
absolute_value_vars = [var for var in absolute_value_vars if var in feature_columns]

# 비율/배수 변수들
ratio_vars = [var for var in feature_columns if var not in absolute_value_vars]

print(f"절댓값 변수: {len(absolute_value_vars)}개")
print(f"비율/배수 변수: {len(ratio_vars)}개")

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

# 전체 기업 통계량
print("📊 전체 기업 통계량:")
all_stats = df[feature_columns].describe()
print(all_stats.round(4))

# 정상 기업 통계량
print("\n📊 정상 기업 (Default=0) 통계량:")
normal_stats = normal_df[feature_columns].describe()
print(normal_stats.round(4))

print("\n📊 부실 기업 (Default=1) 통계량:")
default_stats = default_df[feature_columns].describe()
print(default_stats.round(4))

# 5. 스케일링을 통한 통합 분석
print("\n4️⃣ 스케일링을 통한 통합 분석")
print("="*50)

def create_scaled_comparison(normal_data, default_data, scaler_type='minmax'):
    """
    스케일링을 통해 정상기업과 부실기업 데이터를 비교 가능하게 변환
    """
    # 결측치 처리
    normal_clean = normal_data.fillna(normal_data.median())
    default_clean = default_data.fillna(default_data.median())
    
    # 스케일러 선택
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    
    # 전체 데이터로 스케일러 훈련
    all_data = pd.concat([normal_clean, default_clean])
    scaler.fit(all_data)
    
    # 각 그룹별 스케일링
    normal_scaled = pd.DataFrame(
        scaler.transform(normal_clean),
        columns=normal_clean.columns,
        index=normal_clean.index
    )
    
    default_scaled = pd.DataFrame(
        scaler.transform(default_clean),
        columns=default_clean.columns,
        index=default_clean.index
    )
    
    return normal_scaled, default_scaled, scaler

# 5-1. MinMax 스케일링 비교
print("🔄 MinMax 스케일링 수행 중...")
normal_minmax, default_minmax, minmax_scaler = create_scaled_comparison(
    normal_df[feature_columns], default_df[feature_columns], 'minmax'
)

# 5-2. 스케일링된 데이터 통계량
normal_minmax_stats = normal_minmax.describe()
default_minmax_stats = default_minmax.describe()

# 5-3. 스케일링된 데이터 비교 분석
minmax_mean_comparison = pd.DataFrame({
    '정상기업_평균': normal_minmax_stats.loc['mean'],
    '부실기업_평균': default_minmax_stats.loc['mean']
})
minmax_mean_comparison['차이'] = minmax_mean_comparison['부실기업_평균'] - minmax_mean_comparison['정상기업_평균']
minmax_mean_comparison['차이_절댓값'] = minmax_mean_comparison['차이'].abs()

# 5-4. 표준화된 데이터 비교
print("🔄 표준화(Standard Scaling) 수행 중...")
normal_standard, default_standard, standard_scaler = create_scaled_comparison(
    normal_df[feature_columns], default_df[feature_columns], 'standard'
)

standard_mean_comparison = pd.DataFrame({
    '정상기업_평균': normal_standard.describe().loc['mean'],
    '부실기업_평균': default_standard.describe().loc['mean']
})
standard_mean_comparison['차이'] = standard_mean_comparison['부실기업_평균'] - standard_mean_comparison['정상기업_평균']
standard_mean_comparison['차이_절댓값'] = standard_mean_comparison['차이'].abs()

# 6. 원본 데이터 통계량 비교 분석
print("\n5️⃣ 원본 데이터 통계량 비교 분석")
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

print("📈 원본 데이터 - 평균값 차이가 큰 상위 10개 변수:")
top_mean_diff = mean_comparison.reindex(mean_comparison['차이'].abs().nlargest(10).index)
print(top_mean_diff.round(4))

print("\n📈 MinMax 스케일링 - 평균값 차이가 큰 상위 10개 변수:")
top_minmax_diff = minmax_mean_comparison.reindex(minmax_mean_comparison['차이_절댓값'].nlargest(10).index)
print(top_minmax_diff.round(4))

print("\n📈 표준화 - 평균값 차이가 큰 상위 10개 변수:")
top_standard_diff = standard_mean_comparison.reindex(standard_mean_comparison['차이_절댓값'].nlargest(10).index)
print(top_standard_diff.round(4))

# 7. 향상된 시각화
print("\n6️⃣ 향상된 시각화")
print("="*50)

# 개별 폴더 생성
individual_viz_dir = viz_dir / 'individual_charts'
boxplot_viz_dir = viz_dir / 'boxplots'
histogram_viz_dir = viz_dir / 'histograms'
scatter_viz_dir = viz_dir / 'scatter_plots'

for folder in [individual_viz_dir, boxplot_viz_dir, histogram_viz_dir, scatter_viz_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# 7-1. 개별 변수별 비교 차트 생성 (모든 변수)
print("개별 변수별 비교 차트 생성 중...")

# 변수들을 스케일링 차이 기준으로 정렬
all_vars_sorted = minmax_mean_comparison.sort_values('차이_절댓값', ascending=False).index

for i, var in enumerate(all_vars_sorted, 1):
    print(f"  {i:3d}/{len(all_vars_sorted)}: {var}")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 변수 타입 확인
    var_type = '절댓값' if var in absolute_value_vars else '비율/배수'
    
    # 1. 원본 데이터 비교 (비율 변수) 또는 MinMax 스케일링 (절댓값 변수)
    if var in ratio_vars:
        # 비율 변수는 원본 데이터 사용
        normal_val = mean_comparison.loc[var, '정상기업_평균']
        default_val = mean_comparison.loc[var, '부실기업_평균']
        diff_val = mean_comparison.loc[var, '차이']
        title_suffix = "원본 데이터"
    else:
        # 절댓값 변수는 MinMax 스케일링 사용
        normal_val = minmax_mean_comparison.loc[var, '정상기업_평균']
        default_val = minmax_mean_comparison.loc[var, '부실기업_평균']
        diff_val = minmax_mean_comparison.loc[var, '차이']
        title_suffix = "MinMax 스케일링"
    
    # 막대 그래프
    bars = ax1.bar(['정상기업', '부실기업'], [normal_val, default_val], 
                   color=['skyblue', 'red'], alpha=0.8, edgecolor='black')
    ax1.set_title(f'{var}\n평균값 비교 ({title_suffix})', fontweight='bold', fontsize=11)
    ax1.set_ylabel('평균값')
    ax1.grid(True, alpha=0.3)

    # 값 표시
    for bar, val in zip(bars, [normal_val, default_val]):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.02,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. 차이값 표시
    color = 'red' if diff_val > 0 else 'blue'
    bar = ax2.bar(['차이 (부실-정상)'], [diff_val], color=color, alpha=0.8, edgecolor='black')
    ax2.set_title(f'평균값 차이: {diff_val:+.4f}', fontweight='bold', fontsize=11)
    ax2.set_ylabel('차이값')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # 차이값 표시
    height = bar[0].get_height()
    ax2.text(bar[0].get_x() + bar[0].get_width()/2., height + (abs(height)*0.02 if height >= 0 else -abs(height)*0.02),
           f'{height:+.4f}', ha='center', va='bottom' if height >= 0 else 'top', 
           fontsize=12, fontweight='bold')
    
    # 3. 박스플롯
    data_normal = normal_df[var].dropna()
    data_default = default_df[var].dropna()
    
    if len(data_normal) > 0 and len(data_default) > 0:
        box_data = [data_normal, data_default]
        bp = ax3.boxplot(box_data, labels=['정상기업', '부실기업'], patch_artist=True)
        bp['boxes'][0].set_facecolor('skyblue')
        bp['boxes'][1].set_facecolor('red')
        ax3.set_title(f'분포 비교 (박스플롯)', fontweight='bold', fontsize=11)
        ax3.set_ylabel('값')
        ax3.grid(True, alpha=0.3)

    # 4. 히스토그램
    if len(data_normal) > 0 and len(data_default) > 0:
        # 이상치 제거를 위한 범위 설정
        q1_normal, q3_normal = data_normal.quantile([0.25, 0.75])
        q1_default, q3_default = data_default.quantile([0.25, 0.75])
        iqr_normal = q3_normal - q1_normal
        iqr_default = q3_default - q1_default
        
        lower_bound = min(q1_normal - 1.5*iqr_normal, q1_default - 1.5*iqr_default)
        upper_bound = max(q3_normal + 1.5*iqr_normal, q3_default + 1.5*iqr_default)
        
        # 범위 내 데이터만 사용
        data_normal_filtered = data_normal[(data_normal >= lower_bound) & (data_normal <= upper_bound)]
        data_default_filtered = data_default[(data_default >= lower_bound) & (data_default <= upper_bound)]
        
        if len(data_normal_filtered) > 0 and len(data_default_filtered) > 0:
            bins = min(30, max(10, len(data_normal_filtered)//10))
            ax4.hist(data_normal_filtered, bins=bins, alpha=0.7, label='정상기업', 
                    color='skyblue', density=True, edgecolor='black', linewidth=0.5)
            ax4.hist(data_default_filtered, bins=bins, alpha=0.7, label='부실기업', 
                    color='red', density=True, edgecolor='black', linewidth=0.5)
            ax4.set_title(f'분포 비교 (히스토그램)', fontweight='bold', fontsize=11)
            ax4.set_xlabel('값')
            ax4.set_ylabel('밀도')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

    # 전체 제목 및 정보
    fig.suptitle(f'{var} ({var_type} 변수)\n순위: {i}/{len(all_vars_sorted)} (MinMax 차이 기준)', 
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    
    # 파일명 안전하게 처리
    safe_filename = var.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    plt.savefig(individual_viz_dir / f'{i:03d}_{safe_filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7-2. 박스플롯 모음 생성 (상위 50개 변수)
print("박스플롯 모음 생성 중...")
top_50_vars = all_vars_sorted[:50]

# 10개씩 5개 그룹으로 나누어 생성
for group_idx in range(0, len(top_50_vars), 10):
    group_vars = top_50_vars[group_idx:group_idx+10]

    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    axes = axes.flatten()

    for i, var in enumerate(group_vars):
        data_normal = normal_df[var].dropna()
        data_default = default_df[var].dropna()
        
        if len(data_normal) > 0 and len(data_default) > 0:
            box_data = [data_normal, data_default]
            bp = axes[i].boxplot(box_data, labels=['정상', '부실'], patch_artist=True)
            bp['boxes'][0].set_facecolor('skyblue')
            bp['boxes'][1].set_facecolor('red')
            
            var_type = '절댓값' if var in absolute_value_vars else '비율'
            axes[i].set_title(f'{var}\n({var_type})', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', which='major', labelsize=8)

    # 빈 서브플롯 숨기기
    for i in range(len(group_vars), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'박스플롯 비교 - 순위 {group_idx+1}~{min(group_idx+10, len(top_50_vars))}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(boxplot_viz_dir / f'boxplots_rank_{group_idx+1:02d}_{min(group_idx+10, len(top_50_vars)):02d}.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

# 7-3. 히스토그램 모음 생성 (상위 30개 변수)
print("히스토그램 모음 생성 중...")
top_30_vars = all_vars_sorted[:30]

# 6개씩 5개 그룹으로 나누어 생성
for group_idx in range(0, len(top_30_vars), 6):
    group_vars = top_30_vars[group_idx:group_idx+6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, var in enumerate(group_vars):
        data_normal = normal_df[var].dropna()
        data_default = default_df[var].dropna()
        
        if len(data_normal) > 0 and len(data_default) > 0:
            # 이상치 제거
            q1_normal, q3_normal = data_normal.quantile([0.25, 0.75])
            q1_default, q3_default = data_default.quantile([0.25, 0.75])
            iqr_normal = q3_normal - q1_normal
            iqr_default = q3_default - q1_default
            
            lower_bound = min(q1_normal - 1.5*iqr_normal, q1_default - 1.5*iqr_default)
            upper_bound = max(q3_normal + 1.5*iqr_normal, q3_default + 1.5*iqr_default)
            
            data_normal_filtered = data_normal[(data_normal >= lower_bound) & (data_normal <= upper_bound)]
            data_default_filtered = data_default[(data_default >= lower_bound) & (data_default <= upper_bound)]
            
            if len(data_normal_filtered) > 0 and len(data_default_filtered) > 0:
                bins = min(30, max(10, len(data_normal_filtered)//20))
                axes[i].hist(data_normal_filtered, bins=bins, alpha=0.7, label='정상기업', 
                           color='skyblue', density=True, edgecolor='black', linewidth=0.5)
                axes[i].hist(data_default_filtered, bins=bins, alpha=0.7, label='부실기업', 
                           color='red', density=True, edgecolor='black', linewidth=0.5)
                
                var_type = '절댓값' if var in absolute_value_vars else '비율'
                axes[i].set_title(f'{var}\n({var_type})', fontsize=10)
                axes[i].set_xlabel('값', fontsize=9)
                axes[i].set_ylabel('밀도', fontsize=9)
                axes[i].legend(fontsize=8)
                axes[i].grid(True, alpha=0.3)
    
    # 빈 서브플롯 숨기기
    for i in range(len(group_vars), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'히스토그램 비교 - 순위 {group_idx+1}~{min(group_idx+6, len(top_30_vars))}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(histogram_viz_dir / f'histograms_rank_{group_idx+1:02d}_{min(group_idx+6, len(top_30_vars)):02d}.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

# 7-4. 산점도 생성 (상위 20개 변수)
print("산점도 생성 중...")
top_20_vars_scatter = all_vars_sorted[:20]

for i, var in enumerate(top_20_vars_scatter, 1):
    print(f"  산점도 {i:2d}/{len(top_20_vars_scatter)}: {var}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 데이터 준비
    data_normal = normal_df[var].dropna()
    data_default = default_df[var].dropna()
    
    if len(data_normal) > 0 and len(data_default) > 0:
        # 1. 인덱스 vs 값 산점도
        ax1.scatter(range(len(data_normal)), data_normal, alpha=0.6, s=1, 
                   color='skyblue', label=f'정상기업 ({len(data_normal):,}개)')
        ax1.scatter(range(len(data_default)), data_default, alpha=0.8, s=3, 
                   color='red', label=f'부실기업 ({len(data_default):,}개)')
        
        var_type = '절댓값' if var in absolute_value_vars else '비율'
        ax1.set_title(f'{var} ({var_type})\n데이터 분포 산점도', fontweight='bold')
        ax1.set_xlabel('데이터 인덱스')
        ax1.set_ylabel('값')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 확률밀도 추정 (KDE)
        try:
            from scipy import stats
            
            # 이상치 제거
            q1_normal, q3_normal = data_normal.quantile([0.25, 0.75])
            q1_default, q3_default = data_default.quantile([0.25, 0.75])
            iqr_normal = q3_normal - q1_normal
            iqr_default = q3_default - q1_default
            
            lower_bound = min(q1_normal - 1.5*iqr_normal, q1_default - 1.5*iqr_default)
            upper_bound = max(q3_normal + 1.5*iqr_normal, q3_default + 1.5*iqr_default)
            
            data_normal_clean = data_normal[(data_normal >= lower_bound) & (data_normal <= upper_bound)]
            data_default_clean = data_default[(data_default >= lower_bound) & (data_default <= upper_bound)]
            
            if len(data_normal_clean) > 10 and len(data_default_clean) > 10:
                try:
                    # KDE 계산
                    kde_normal = stats.gaussian_kde(data_normal_clean)
                    kde_default = stats.gaussian_kde(data_default_clean)

                    # x 범위 설정
                    x_min = min(data_normal_clean.min(), data_default_clean.min())
                    x_max = max(data_normal_clean.max(), data_default_clean.max())
                    x_range = np.linspace(x_min, x_max, 200)
                    
                    # KDE 플롯
                    ax2.plot(x_range, kde_normal(x_range), color='skyblue', linewidth=2, label='정상기업')
                    ax2.fill_between(x_range, kde_normal(x_range), alpha=0.3, color='skyblue')
                    ax2.plot(x_range, kde_default(x_range), color='red', linewidth=2, label='부실기업')
                    ax2.fill_between(x_range, kde_default(x_range), alpha=0.3, color='red')
                    
                    ax2.set_title(f'확률밀도 분포 (KDE)', fontweight='bold')
                    ax2.set_xlabel('값')
                    ax2.set_ylabel('확률밀도')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                except (np.linalg.LinAlgError, ValueError) as e:
                    ax2.text(0.5, 0.5, f'KDE 생성 불가\n(데이터 특이성)\n{str(e)[:50]}...', 
                            ha='center', va='center', transform=ax2.transAxes, fontsize=10)
            else:
                ax2.text(0.5, 0.5, 'KDE 생성 불가\n(데이터 부족)', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                
        except ImportError:
            ax2.text(0.5, 0.5, 'scipy 모듈 필요', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    plt.tight_layout()
    
    # 파일명 안전하게 처리
    safe_filename = var.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    plt.savefig(scatter_viz_dir / f'{i:02d}_{safe_filename}_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n📈 시각화 완료!")
print(f"   📁 individual_charts/: 모든 변수별 개별 차트 ({len(all_vars_sorted)}개)")
print(f"   📁 boxplots/: 박스플롯 모음 ({len(range(0, 50, 10))}개)")
print(f"   📁 histograms/: 히스토그램 모음 ({len(range(0, 30, 6))}개)")
print(f"   📁 scatter_plots/: 산점도 모음 (20개)")

# 8. 결과 저장
print("\n7️⃣ 결과 저장")
print("="*50)

# 기초 통계량 저장
all_stats.to_csv(reports_dir / 'all_companies_statistics.csv', encoding='utf-8-sig')
normal_stats.to_csv(reports_dir / 'normal_companies_statistics.csv', encoding='utf-8-sig')
default_stats.to_csv(reports_dir / 'default_companies_statistics.csv', encoding='utf-8-sig')

# 원본 데이터 비교 분석 결과 저장
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
# Default 그룹별 재무비율 통계 분석 리포트 (스케일링 포함)

## 1. 분석 개요
- 원본 데이터: {len(df):,}개 관측치
- 정상 기업 (Default=0): {len(normal_df):,}개 ({len(normal_df)/len(df)*100:.1f}%)
- 부실 기업 (Default=1): {len(default_df):,}개 ({len(default_df)/len(df)*100:.1f}%)
- 분석 변수: {len(feature_columns)}개 재무비율
- 절댓값 변수: {len(absolute_value_vars)}개 (총자산, 매출액 등)
- 비율/배수 변수: {len(ratio_vars)}개 (각종 비율, 증가율 등)

## 2. 스케일링 방법별 주요 발견사항

### 2.1 원본 데이터 - 평균값 차이가 가장 큰 변수들 (비율 변수 중심)
"""

ratio_mean_diff = mean_comparison.loc[ratio_vars].reindex(mean_comparison.loc[ratio_vars]['차이'].abs().nlargest(5).index)
for i, (var, row) in enumerate(ratio_mean_diff.iterrows(), 1):
    report_content += f"{i}. {var}\n"
    report_content += f"   정상기업: {row['정상기업_평균']:.4f}, 부실기업: {row['부실기업_평균']:.4f}\n"
    report_content += f"   차이: {row['차이']:+.4f} ({row['차이_비율(%)']:+.1f}%)\n\n"

report_content += f"""
### 2.2 MinMax 스케일링 - 차이가 가장 큰 변수들 (전체 변수)
"""

for i, (var, row) in enumerate(top_minmax_diff.head(5).iterrows(), 1):
    report_content += f"{i}. {var} ({'절댓값' if var in absolute_value_vars else '비율/배수'})\n"
    report_content += f"   정상기업: {row['정상기업_평균']:.4f}, 부실기업: {row['부실기업_평균']:.4f}\n"
    report_content += f"   차이: {row['차이']:+.4f}\n\n"

report_content += f"""
### 2.3 표준화 - 차이가 가장 큰 변수들 (전체 변수)
"""

for i, (var, row) in enumerate(top_standard_diff.head(5).iterrows(), 1):
    report_content += f"{i}. {var} ({'절댓값' if var in absolute_value_vars else '비율/배수'})\n"
    report_content += f"   정상기업: {row['정상기업_평균']:.4f}, 부실기업: {row['부실기업_평균']:.4f}\n"
    report_content += f"   차이: {row['차이']:+.4f}\n\n"

report_content += f"""
## 3. 변수 타입별 분석 결과
- 절댓값 변수 중 차이가 큰 변수: {[var for var in all_vars_sorted[:5] if var in absolute_value_vars]}
- 비율/배수 변수 중 차이가 큰 변수: {[var for var in all_vars_sorted[:5] if var in ratio_vars]}

## 4. 스케일링 방법 비교
1. **원본 데이터**: 비율 변수만 직접 비교 가능, 절댓값 변수는 단위 차이로 비교 어려움
2. **MinMax 스케일링**: 모든 변수를 0-1 범위로 정규화하여 동일한 척도로 비교 가능
3. **표준화**: 평균 0, 표준편차 1로 변환하여 분포 특성 고려한 비교 가능

## 5. 그룹별 데이터 품질
- 정상기업 결측치: {normal_df[feature_columns].isnull().sum().sum():,}개
- 부실기업 결측치: {default_df[feature_columns].isnull().sum().sum():,}개

## 6. 결론 및 시사점
1. **스케일링의 필요성**: 절댓값 변수와 비율 변수를 함께 분석하려면 스케일링 필수
2. **변수 중요도**: MinMax/표준화 기준으로 {all_vars_sorted[0]}가 가장 큰 차이 보임
3. **모델링 전략**: 절댓값 변수는 스케일링 후 사용, 비율 변수는 원본 또는 스케일링 선택적 적용
4. **특징 공학**: 변수 타입별로 다른 전처리 전략 필요

## 7. 생성된 파일
- 시각화: {viz_dir}/에 4개 폴더
- 통계 데이터: {reports_dir}/에 6개 CSV 파일
  - all_companies_statistics.csv: 전체 기업 기초 통계량
  - normal_companies_statistics.csv: 정상 기업 기초 통계량
  - default_companies_statistics.csv: 부실 기업 기초 통계량
  - mean_comparison_analysis.csv: 원본 데이터 평균값 비교 분석
  - std_comparison_analysis.csv: 원본 데이터 표준편차 비교 분석
  - minmax_scaling_comparison.csv: MinMax 스케일링 비교 분석
  - standard_scaling_comparison.csv: 표준화 비교 분석
  - variable_classification.csv: 변수 타입별 분류
  - comprehensive_group_comparison.csv: 종합 비교 분석
"""

# 리포트 저장
with open(reports_dir / 'default_group_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"✅ 분석 완료!")
print(f"📊 전체기업 통계: {reports_dir / 'all_companies_statistics.csv'}")
print(f"📊 정상기업 통계: {reports_dir / 'normal_companies_statistics.csv'}")
print(f"📊 부실기업 통계: {reports_dir / 'default_companies_statistics.csv'}")
print(f"📊 평균 비교 분석: {reports_dir / 'mean_comparison_analysis.csv'}")
print(f"📊 표준편차 비교 분석: {reports_dir / 'std_comparison_analysis.csv'}")
print(f"📊 종합 비교 분석: {reports_dir / 'comprehensive_group_comparison.csv'}")
print(f"📄 상세 리포트: {reports_dir / 'default_group_analysis_report.txt'}")
print(f"📈 시각화 파일: {viz_dir}/ (8개 차트)")

print(f"\n🎯 주요 분석 결과:")
print(f"- 정상기업: {len(normal_df):,}개 ({len(normal_df)/len(df)*100:.1f}%)")
print(f"- 부실기업: {len(default_df):,}개 ({len(default_df)/len(df)*100:.1f}%)")
print(f"- 절댓값 변수: {len(absolute_value_vars)}개")
print(f"- 비율/배수 변수: {len(ratio_vars)}개")
print(f"- MinMax 기준 차이가 가장 큰 변수: {minmax_mean_comparison['차이_절댓값'].idxmax()}")
print(f"- 표준화 기준 차이가 가장 큰 변수: {standard_mean_comparison['차이_절댓값'].idxmax()}")

print(f"\n📈 폴더별 시각화 완료:")
print(f"   📁 individual_charts/: 모든 변수별 개별 차트 ({len(feature_columns)}개)")
print(f"   📁 boxplots/: 박스플롯 모음 (5개)")
print(f"   📁 histograms/: 히스토그램 모음 (5개)")
print(f"   📁 scatter_plots/: 산점도 모음 (20개)")
print(f"\n💡 개별 차트 특징:")
print(f"   - 각 변수마다 4가지 시각화 (막대그래프, 차이값, 박스플롯, 히스토그램)")
print(f"   - 변수 타입에 따라 적절한 스케일링 적용")
print(f"   - 순위별로 파일명 정렬하여 중요도 확인 가능")
print(f"   - 모든 {len(feature_columns)}개 변수의 상세 분석 결과 제공")