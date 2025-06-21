import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="부실예측 모델링 프로젝트 코드리뷰",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 메뉴
st.sidebar.title("📊 Navigation")
menu = st.sidebar.selectbox(
    "메뉴 선택",
    ["🏠 프로젝트 개요", "🏗️ 코드베이스 구조", "📁 데이터 파이프라인", "🔧 핵심 기능", "📈 데이터 현황", "🎯 모델링 준비"]
)

# 메인 타이틀
st.title("🏢 한국 기업 부실예측 모델링 프로젝트")
st.markdown("---")

if menu == "🏠 프로젝트 개요":
    st.header("📋 프로젝트 개요")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 프로젝트 목표")
        st.markdown("""
        **한국 기업의 재무데이터와 주가데이터를 활용한 부실예측 모델 개발**
        
        - 📊 **데이터 기간**: 2012~2023년 (12년간)
        - 🏢 **대상 기업**: 한국 상장기업 약 2,630개
        - 🎯 **예측 목표**: 기업 부실 (상장폐지) 1년 전 예측
        - 🤖 **모델링**: Machine Learning 기반 분류 모델
        """)
        
        st.subheader("🔄 프로젝트 워크플로우")
        st.markdown("""
        1. **데이터 수집 및 정제** → 재무제표 + 주가데이터
        2. **데이터 매칭 및 통합** → 거래소코드 정규화
        3. **재무비율 계산** → 17개 핵심 재무지표
        4. **부실 라벨링** → 상장폐지 전년도 = 부실
        5. **특성 엔지니어링** → SMOTE + 스케일링
        6. **모델 개발** → 다중 알고리즘 비교
        """)
    
    with col2:
        st.subheader("📊 프로젝트 지표")
        
        # 메트릭 카드
        st.metric("총 데이터", "22,780개", "기업-연도 조합")
        st.metric("고유 기업", "2,630개", "12년간")
        st.metric("재무비율", "17개", "핵심 지표")
        st.metric("부실 기업", "132개", "0.58%")
        
        st.subheader("🛠️ 기술 스택")
        st.markdown("""
        - **언어**: Python 🐍
        - **데이터**: Pandas, NumPy
        - **시각화**: Matplotlib, Seaborn, Plotly
        - **ML**: Scikit-learn, XGBoost, Imbalanced-learn
        - **대시보드**: Streamlit
        """)

elif menu == "🏗️ 코드베이스 구조":
    st.header("🏗️ 코드베이스 구조")
    
    # 프로젝트 구조 시각화
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 디렉토리 구조")
        st.code("""
P2_Default-invest/
├── 📁 data_new/
│   ├── raw/           # 원본 데이터
│   ├── processed/     # 전처리된 데이터  
│   └── final/         # 최종 모델링 데이터
├── 📁 src_new/
│   ├── data_processing/    # 데이터 처리
│   ├── feature_engineering/ # 특성 엔지니어링
│   ├── analysis/          # 데이터 분석
│   ├── modeling/          # 모델링
│   └── utils/             # 유틸리티
├── 📁 outputs/
│   ├── visualizations/    # 시각화 결과
│   ├── reports/          # 분석 보고서
│   └── models/           # 훈련된 모델
├── 📁 notebooks/         # Jupyter 노트북
└── 📁 dashboard/         # 대시보드
        """, language="text")
    
    with col2:
        st.subheader("🔧 핵심 모듈")
        
        modules = {
            "data_processing": {
                "files": ["create_financial_ratios_master.py"],
                "description": "4단계 재무비율 계산 파이프라인",
                "color": "#FF6B6B"
            },
            "feature_engineering": {
                "files": ["add_financial_variables.py", "create_final_modeling_dataset.py"],
                "description": "특성 생성, SMOTE, 스케일링",
                "color": "#4ECDC4"
            },
            "analysis": {
                "files": ["analyze_scaling_needs.py", "apply_default_labeling_and_scaling.py"],
                "description": "데이터 분석 및 라벨링",
                "color": "#45B7D1"
            },
            "modeling": {
                "files": ["logistic_regression.py", "RF.py", "xgboost.py"],
                "description": "ML 모델 구현",
                "color": "#96CEB4"
            }
        }
        
        for module, info in modules.items():
            with st.expander(f"📦 {module}"):
                st.markdown(f"**설명**: {info['description']}")
                st.markdown("**파일들**:")
                for file in info['files']:
                    st.markdown(f"- `{file}`")

    # 데이터 플로우 다이어그램
    st.subheader("🔄 데이터 플로우")
    
    # Plotly로 플로우차트 생성
    fig = go.Figure()
    
    # 노드 정의
    nodes = [
        {"name": "원본 데이터", "x": 0, "y": 4, "color": "#FF6B6B"},
        {"name": "데이터 정제", "x": 1, "y": 4, "color": "#FF6B6B"},
        {"name": "재무비율 계산", "x": 2, "y": 4, "color": "#4ECDC4"},
        {"name": "부실 라벨링", "x": 3, "y": 4, "color": "#45B7D1"},
        {"name": "SMOTE 적용", "x": 4, "y": 5, "color": "#96CEB4"},
        {"name": "스케일링", "x": 4, "y": 3, "color": "#96CEB4"},
        {"name": "모델 훈련", "x": 5, "y": 4, "color": "#FFEAA7"}
    ]
    
    # 노드 그리기
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node["x"]], y=[node["y"]],
            mode='markers+text',
            marker=dict(size=60, color=node["color"]),
            text=node["name"],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            name=node["name"],
            showlegend=False
        ))
    
    # 화살표 연결
    arrows = [
        (0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)
    ]
    
    for start, end in arrows:
        fig.add_annotation(
            x=nodes[end]["x"], y=nodes[end]["y"],
            ax=nodes[start]["x"], ay=nodes[start]["y"],
            xref="x", yref="y",
            axref="x", ayref="y",
            arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor="gray"
        )
    
    fig.update_layout(
        title="데이터 처리 파이프라인",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        plot_bgcolor="white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif menu == "📁 데이터 파이프라인":
    st.header("📁 데이터 파이프라인")
    
    # 탭으로 구분
    tab1, tab2, tab3, tab4 = st.tabs(["📊 데이터 수집", "🔧 전처리", "📈 재무비율", "🎯 최종 준비"])
    
    with tab1:
        st.subheader("📊 원본 데이터")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**재무제표 데이터**")
            st.code("""
# 파일: FS.csv, cFS.csv
- 기간: 2012~2023년
- 기업: 2,632개 고유 기업
- 내용: 대차대조표, 손익계산서, 현금흐름표
- 크기: 약 50MB
            """)
            
        with col2:
            st.markdown("**주가 데이터**")
            st.code("""
# 파일: 2012.csv ~ 2023.csv (연도별)
- 기간: 2012~2023년  
- 기업: 1,997개 고유 기업
- 내용: 일별 주가, 거래량, 시가총액
- 크기: 약 200MB
            """)
        
        st.subheader("🔍 주요 이슈 및 해결")
        
        issues = [
            {
                "문제": "거래소코드 형식 불일치",
                "원인": "FS데이터: 정수형(5380), 주가데이터: 문자열(005380)",
                "해결": "모든 거래소코드를 6자리 문자열로 정규화",
                "결과": "매칭률 67.2% → 73.4% 향상"
            },
            {
                "문제": "회사명 표기 차이",
                "원인": "FS: 현대자동차(주), 주가: 현대자동차보통주",
                "해결": "거래소코드 기준 매칭으로 변경",
                "결과": "회사명 의존성 제거"
            }
        ]
        
        for i, issue in enumerate(issues):
            with st.expander(f"이슈 {i+1}: {issue['문제']}"):
                st.markdown(f"**원인**: {issue['원인']}")
                st.markdown(f"**해결**: {issue['해결']}")
                st.markdown(f"**결과**: {issue['결과']}")
    
    with tab2:
        st.subheader("🔧 데이터 전처리")
        
        st.markdown("**핵심 전처리 단계**")
        
        preprocessing_steps = [
            {"step": "1. 거래소코드 정규화", "code": "df['거래소코드'] = df['거래소코드'].astype(str).str.zfill(6)"},
            {"step": "2. 중복 데이터 제거", "code": "df = df.drop_duplicates(subset=['거래소코드', '회계년도'])"},
            {"step": "3. 결측값 처리", "code": "df[col] = df[col].fillna(df[col].median())"},
            {"step": "4. 데이터 타입 통일", "code": "df = df.astype({'거래소코드': str, '회계년도': str})"}
        ]
        
        for step in preprocessing_steps:
            st.markdown(f"**{step['step']}**")
            st.code(step['code'], language="python")
    
    with tab3:
        st.subheader("📈 재무비율 계산")
        
        st.markdown("**17개 핵심 재무비율**")
        
        ratios = {
            "수익성 지표": ["ROA", "RE_TA", "EBIT_TA", "CFO_TA"],
            "안정성 지표": ["TLTA", "CR", "CLCA", "WC_TA"],
            "시장 지표": ["MVE_TL", "TLMTA", "MB"],
            "활동성 지표": ["S_TA"],
            "기타 지표": ["CFO_TD", "SIGMA", "RET_3M", "RET_9M", "OENEG"]
        }
        
        for category, ratio_list in ratios.items():
            with st.expander(f"📊 {category} ({len(ratio_list)}개)"):
                cols = st.columns(2)
                for i, ratio in enumerate(ratio_list):
                    with cols[i % 2]:
                        st.markdown(f"- **{ratio}**")
        
        st.markdown("**계산 파이프라인**")
        st.code("""
# 4단계 계산 프로세스
def calculate_financial_ratios():
    # 1단계: 기본 재무비율 (FS_flow 활용)
    calculate_basic_ratios()
    
    # 2단계: 시장기반 비율 (주가데이터 활용)  
    calculate_market_ratios()
    
    # 3단계: 변동성과 수익률
    calculate_volatility_returns()
    
    # 4단계: 최종 통합 및 저장
    finalize_ratios()
        """, language="python")
    
    with tab4:
        st.subheader("🎯 최종 모델링 데이터 준비")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**부실 라벨링**")
            st.code("""
# 상장폐지 전년도 = 부실(1)
for company in failed_companies:
    target_year = delisting_year - 1
    df.loc[condition, 'default'] = 1

# 결과: 132개 부실 기업 (0.58%)
            """, language="python")
        
        with col2:
            st.markdown("**SMOTE 적용**")
            st.code("""
# BorderlineSMOTE로 클래스 불균형 해결 (1:10 비율)
smote = BorderlineSMOTE(
    sampling_strategy=0.1,  # 부실:정상 = 1:10
    random_state=42
)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 결과: 부실 비율 0.58% → 10%
            """, language="python")

elif menu == "🔧 핵심 기능":
    st.header("🔧 핵심 기능 코드 리뷰")
    
    # 핵심 함수들 소개
    function_tabs = st.tabs(["📊 재무비율 계산", "🎯 SMOTE 적용", "📈 스케일링", "🔍 데이터 검증"])
    
    with function_tabs[0]:
        st.subheader("📊 재무비율 계산 마스터")
        
        st.code("""
def calculate_financial_ratios_master():
    \"\"\"4단계 재무비율 계산 프로세스 관리자\"\"\"
    
    steps = [
        ("step1_basic_financial_ratios.py", "기본 재무비율"),
        ("step2_market_based_ratios.py", "시장기반 비율"),
        ("step3_volatility_returns.py", "변동성과 수익률"),
        ("step4_finalize_ratios.py", "최종 통합")
    ]
    
    for step_file, step_name in steps:
        success = run_step(step_file, step_name)
        if not success:
            break
    
    return success

def run_step(step_file, step_name):
    \"\"\"단계별 스크립트 실행 및 모니터링\"\"\"
    start_time = time.time()
    
    result = subprocess.run([sys.executable, f'archive_old_structure/src/{step_file}'])
    
    duration = time.time() - start_time
    print(f"✅ {step_name} 완료 (소요시간: {duration:.1f}초)")
    
    return result.returncode == 0
        """, language="python")
        
        st.markdown("**특징:**")
        st.markdown("- 각 단계별 독립 실행 및 오류 처리")
        st.markdown("- 실행 시간 측정 및 성공률 통계")
        st.markdown("- 파이프라인 중단 시 상세 디버깅 정보")
    
    with function_tabs[1]:
        st.subheader("🎯 SMOTE 적용")
        
        st.code("""
def apply_borderline_smote(X_train, y_train):
    \"\"\"BorderlineSMOTE로 클래스 불균형 해결 (1:10 비율)\"\"\"
    
    # SMOTE 설정 (1:10 비율)
    smote = BorderlineSMOTE(
        sampling_strategy=0.1,  # 부실:정상 = 1:10 비율
        random_state=42,
        k_neighbors=5,
        m_neighbors=10
    )
    
    print(f"SMOTE 적용 전: 부실 {(y_train==1).sum()}개, 정상 {(y_train==0).sum()}개")
    
    # SMOTE 적용 (훈련 데이터에만!)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"SMOTE 적용 후: 부실 {(y_train_smote==1).sum()}개, 정상 {(y_train_smote==0).sum()}개")
    print(f"총 증가: {len(X_train_smote) - len(X_train)}개 샘플")
    print(f"최종 비율: 1:{(y_train_smote==0).sum()/(y_train_smote==1).sum():.0f}")
    
    return X_train_smote, y_train_smote
        """, language="python")
        
        st.markdown("**핵심 포인트:**")
        st.markdown("- 훈련 데이터에만 적용하여 데이터 누수 방지")
        st.markdown("- BorderlineSMOTE로 경계선 근처 샘플 생성")
        st.markdown("- 부실 비율 0.58% → 10% (1:10 비율)로 조정")
    
    with function_tabs[2]:
        st.subheader("📈 스케일링")
        
        st.code("""
def apply_scaling(X_train, X_valid, X_test):
    \"\"\"재무비율 특성에 맞는 스케일링 적용\"\"\"
    
    # 스케일링 방법별 컬럼 분류 (분석 결과 기반)
    robust_scaler_columns = [
        'ROA', 'CFO_TD', 'RE_TA', 'EBIT_TA', 'MVE_TL', 'S_TA', 
        'CLCA', 'OENEG', 'CR', 'CFO_TA', 'RET_3M', 'RET_9M', 'MB'
    ]
    
    standard_scaler_columns = ['TLTA', 'WC_TA', 'SIGMA', 'TLMTA']
    
    # RobustScaler: 이상치가 많은 재무비율
    robust_scaler = RobustScaler()
    robust_scaler.fit(X_train[robust_scaler_columns])
    
    X_train_scaled[robust_scaler_columns] = robust_scaler.transform(X_train[robust_scaler_columns])
    X_valid_scaled[robust_scaler_columns] = robust_scaler.transform(X_valid[robust_scaler_columns])
    X_test_scaled[robust_scaler_columns] = robust_scaler.transform(X_test[robust_scaler_columns])
    
    # StandardScaler: 정규분포에 가까운 재무비율
    standard_scaler = StandardScaler()
    standard_scaler.fit(X_train[standard_scaler_columns])
    
    # ... 동일한 방식으로 적용
    
    return X_train_scaled, X_valid_scaled, X_test_scaled
        """, language="python")
        
        st.markdown("**스케일링 전략:**")
        st.markdown("- **RobustScaler**: 이상치 많은 13개 비율")
        st.markdown("- **StandardScaler**: 정규분포 가까운 4개 비율")
        st.markdown("- 훈련 데이터로만 fit하여 데이터 누수 방지")
    
    with function_tabs[3]:
        st.subheader("🔍 데이터 검증")
        
        st.code("""
def validate_data_quality(df):
    \"\"\"데이터 품질 검증 및 리포트\"\"\"
    
    validation_results = {}
    
    # 1. 결측값 검사
    missing_info = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) * 100
        if missing_count > 0:
            missing_info.append({
                'column': col,
                'missing_count': missing_count,
                'missing_pct': missing_pct
            })
    
    validation_results['missing_values'] = missing_info
    
    # 2. 이상치 검사
    outlier_info = []
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        outlier_info.append({
            'column': col,
            'outlier_count': outliers,
            'outlier_pct': outliers / len(df) * 100
        })
    
    validation_results['outliers'] = outlier_info
    
    # 3. 데이터 분포 검사
    distribution_info = []
    for col in numeric_columns:
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()
        distribution_info.append({
            'column': col,
            'skewness': skewness,
            'kurtosis': kurtosis
        })
    
    validation_results['distributions'] = distribution_info
    
    return validation_results
        """, language="python")

elif menu == "📈 데이터 현황":
    st.header("📈 데이터 현황 분석")
    
    # 데이터 정보 로드 시도
    try:
        with open('data_new/final/dataset_info_final.json', 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        
        # 메트릭 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "총 샘플", 
                f"{dataset_info['dataset_info']['original_samples']:,}개",
                "기업-연도 조합"
            )
        
        with col2:
            st.metric(
                "총 특성", 
                f"{dataset_info['dataset_info']['total_features']}개",
                "재무비율"
            )
        
        with col3:
            st.metric(
                "부실 비율", 
                f"{dataset_info['dataset_info']['original_default_rate']*100:.2f}%",
                "원본 데이터"
            )
        
        with col4:
            st.metric(
                "SMOTE 후", 
                f"{dataset_info['smote_version']['train_default_rate']*100:.0f}%",
                "훈련 데이터"
            )
        
        # 데이터 분할 현황
        st.subheader("📊 데이터 분할 현황")
        
        # 분할 비율 차트
        split_data = {
            'Dataset': ['Train', 'Valid', 'Test'],
            'Normal': [
                dataset_info['normal_version']['train_samples'],
                dataset_info['normal_version']['valid_samples'],
                dataset_info['normal_version']['test_samples']
            ],
            'SMOTE': [
                dataset_info['smote_version']['train_samples'],
                dataset_info['smote_version']['valid_samples'],
                dataset_info['smote_version']['test_samples']
            ]
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Normal Version',
            x=split_data['Dataset'],
            y=split_data['Normal'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='SMOTE Version',
            x=split_data['Dataset'],
            y=split_data['SMOTE'],
            marker_color='orange'
        ))
        
        fig.update_layout(
            title='데이터셋 크기 비교 (Normal vs SMOTE)',
            xaxis_title='Dataset',
            yaxis_title='Sample Count',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 특성 정보
        st.subheader("🔧 특성 엔지니어링")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**RobustScaler 적용 (13개)**")
            for feature in dataset_info['feature_info']['robust_scaled']:
                st.markdown(f"- {feature}")
        
        with col2:
            st.markdown("**StandardScaler 적용 (4개)**")
            for feature in dataset_info['feature_info']['standard_scaled']:
                st.markdown(f"- {feature}")
        
    except FileNotFoundError:
        st.warning("데이터셋 정보 파일을 찾을 수 없습니다. 먼저 데이터 생성 스크립트를 실행해주세요.")
        
        # 예시 데이터로 시각화
        st.subheader("📊 예시 데이터 현황")
        
        # 가상의 데이터 분포
        sample_data = pd.DataFrame({
            'ROA': pd.np.random.normal(0.05, 0.1, 1000),
            'TLTA': pd.np.random.normal(0.6, 0.2, 1000),
            'MVE_TL': pd.np.random.lognormal(0, 1, 1000)
        })
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=['ROA', 'TLTA', 'MVE_TL'])
        
        for i, col in enumerate(sample_data.columns):
            fig.add_trace(
                go.Histogram(x=sample_data[col], name=col),
                row=1, col=i+1
            )
        
        fig.update_layout(title='재무비율 분포 예시', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

elif menu == "🎯 모델링 준비":
    st.header("🎯 모델링 준비")
    
    st.subheader("🤖 구현된 모델들")
    
    models = [
        {
            "name": "Logistic Regression",
            "description": "선형 분류 모델, 해석 가능성 우수",
            "pros": ["빠른 훈련", "계수 해석", "확률 출력"],
            "cons": ["선형 관계 가정", "복잡한 패턴 학습 제한"],
            "use_case": "베이스라인 모델, 해석이 중요한 경우"
        },
        {
            "name": "Random Forest",
            "description": "앙상블 기반 결정 트리 모델",
            "pros": ["높은 정확도", "특성 중요도", "오버피팅 방지"],
            "cons": ["해석성 제한", "메모리 사용량 높음"],
            "use_case": "안정적인 성능, 특성 중요도 분석"
        },
        {
            "name": "XGBoost",
            "description": "그래디언트 부스팅 기반 고성능 모델",
            "pros": ["최고 수준 성능", "불균형 데이터 처리", "빠른 훈련"],
            "cons": ["하이퍼파라미터 튜닝 복잡"],
            "use_case": "최고 성능 추구, 불균형 데이터"
        }
    ]
    
    for model in models:
        with st.expander(f"🔧 {model['name']}"):
            st.markdown(f"**설명**: {model['description']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**장점**")
                for pro in model['pros']:
                    st.markdown(f"✅ {pro}")
            
            with col2:
                st.markdown("**단점**")
                for con in model['cons']:
                    st.markdown(f"⚠️ {con}")
            
            with col3:
                st.markdown("**사용 사례**")
                st.markdown(f"💡 {model['use_case']}")
    
    st.subheader("📊 평가 지표")
    
    metrics_info = {
        "AUC-ROC": "불균형 데이터의 주요 평가 지표",
        "Precision": "부실 예측 정확도 (False Positive 최소화)",
        "Recall": "부실 탐지율 (False Negative 최소화)",
        "F1-Score": "Precision과 Recall의 조화평균"
    }
    
    for metric, description in metrics_info.items():
        st.markdown(f"**{metric}**: {description}")
    
    st.subheader("🚀 다음 단계")
    
    next_steps = [
        "일반 버전으로 베이스라인 모델 훈련",
        "SMOTE 버전으로 성능 향상 모델 훈련",
        "모델별 성능 비교 및 분석",
        "하이퍼파라미터 튜닝",
        "최종 모델 선택 및 해석"
    ]
    
    for i, step in enumerate(next_steps, 1):
        st.markdown(f"{i}. {step}")

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🏢 한국 기업 부실예측 모델링 프로젝트 | 📊 Code Review Dashboard</p>
        <p>Made with ❤️ using Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
) 