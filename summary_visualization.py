"""
Summary Table 기반 모델 성능 시각화
=====================================

기존 modeling run의 summary_table.csv를 읽어서 
앙상블을 포함한 모델별 성능 비교 시각화를 생성합니다.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 한글 폰트 설정
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':  # Windows
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False


def create_ensemble_visualization(summary_csv_path, output_path="ensemble.png"):
    """Summary table을 기반으로 앙상블 성능 비교 차트 생성"""
    
    # CSV 파일 읽기
    df = pd.read_csv(summary_csv_path)
    
    # 모델명 정리
    model_names = []
    for _, row in df.iterrows():
        if row['Model'] == 'ensemble':
            model_names.append('ENSEMBLE')
        else:
            model_name = row['Model'].upper()
            if row['Data_Type'] != 'NORMAL':
                model_name += f"_{row['Data_Type']}"
            model_names.append(model_name)
    
    # 성능 메트릭 추출
    metrics = {
        'AUC': df['Test_AUC'].values,
        'F1-Score': df['Test_F1'].values,
        'Precision': df['Test_Precision'].values,
        'Recall': df['Test_Recall'].values
    }
    
    # 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 색상 설정
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'plum', 'wheat', 'lightgray', 'cyan']
    ensemble_color = 'darkred'
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]
        
        # 색상 배정 (앙상블은 특별한 색상)
        bar_colors = []
        for model_name in model_names:
            if model_name == 'ENSEMBLE':
                bar_colors.append(ensemble_color)
            else:
                bar_colors.append(colors[len(bar_colors) % len(colors)])
        
        # 바 차트 생성
        bars = ax.bar(range(len(model_names)), values, 
                     color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # 차트 설정
        ax.set_title(f'{metric_name} Comparison (Including Ensemble)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 값 표시
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # x축 설정
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    
    # 전체 제목
    fig.suptitle('Performance Comparison: All Models + Ensemble', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # 범례 추가
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=ensemble_color, alpha=0.8, label='Ensemble Model'),
        plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.8, label='Individual Models')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.94))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)
    
    # 저장
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 앙상블 성능 비교 차트 저장 완료: {output_path}")
    
    # 결과 요약 출력
    print("\n📊 성능 요약:")
    for name, auc, f1, precision, recall in zip(model_names, 
                                               metrics['AUC'], 
                                               metrics['F1-Score'],
                                               metrics['Precision'], 
                                               metrics['Recall']):
        print(f"  {name}: AUC={auc:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")


def create_model_category_visualizations(summary_csv_path, output_dir="."):
    """모델 카테고리별 개별 시각화 생성 (logistic.png, RF.png, XGboost.png)"""
    
    df = pd.read_csv(summary_csv_path)
    output_dir = Path(output_dir)
    
    # 모델 카테고리 정의 (앙상블 제외 - 전체 비교는 별도 함수에서 처리)
    model_categories = {
        'logistic': ['logistic_regression'],
        'RF': ['random_forest'], 
        'XGboost': ['xgboost']
    }
    
    for category, model_types in model_categories.items():
        # 해당 카테고리의 모델들 필터링
        category_df = df[df['Model'].isin(model_types)]
        
        if category_df.empty:
            print(f"⚠️ {category} 카테고리에 해당하는 모델이 없습니다.")
            continue
        
        # 모델명 생성
        model_names = []
        for _, row in category_df.iterrows():
            if row['Model'] == 'ensemble':
                model_names.append('ENSEMBLE')
            else:
                model_name = f"{row['Model']}_{row['Data_Type']}"
                model_names.append(model_name)
        
        # 성능 메트릭 추출
        metrics_data = {
            'AUC': category_df['Test_AUC'].values,
            'F1-Score': category_df['Test_F1'].values,
            'Precision': category_df['Test_Precision'].values,
            'Recall': category_df['Test_Recall'].values
        }
        
        # 시각화 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 색상 설정 (예시 이미지와 유사하게)
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'plum', 'wheat', 'lightgray']
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[i]
            
            # 바 차트 생성
            bars = ax.bar(range(len(model_names)), values, 
                         color=colors[:len(model_names)], alpha=0.8)
            
            # 값 표시
            for j, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=10)
            
            # 축 설정
            ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            
            # x축 레이블 설정
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # 전체 제목
        fig.suptitle(f'{category} Models Performance Comparison', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # 저장
        output_path = output_dir / f"{category}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {category} 시각화 저장: {output_path}")


def main():
    """메인 실행 함수"""
    print("📊 Summary Table 기반 모델 성능 시각화 시작")
    
    # 최신 modeling run 결과 경로
    summary_path = "/Users/jojongho/KDT/P2_Default-invest/summary_table.csv"
    
    # 1. 앙상블 포함 전체 성능 비교
    print("\n🎨 앙상블 포함 전체 성능 비교 차트 생성...")
    create_ensemble_visualization(summary_path, "ensemble.png")
    
    # 2. 모델 카테고리별 개별 시각화
    print("\n🎨 모델 카테고리별 개별 시각화 생성...")
    create_model_category_visualizations(summary_path, ".")
    
    print("\n🎉 모든 시각화 완료!")


if __name__ == "__main__":
    main()