# Reports - CSV 및 텍스트 분석 자료

본 디렉토리는 프로젝트 전반에 걸쳐 생성된 **CSV 분석 결과**와 **텍스트 리포트**를 저장합니다.

## 📂 주요 파일 분류

### 1. 스케일링 & 데이터 품질 분석
- `basic_statistics.csv` : 33개 재무변수 기초 통계량
- `missing_analysis.csv` : 변수별 결측치 비율 및 패턴
- `scaling_scores.csv` : 스케일링 필요성 점수
- `scaling_recommendations.csv` : 변수별 추천 스케일링 방법
- `scaling_analysis.xlsx` : 상세 스케일링 분석 (엑셀)

### 2. 결측치 임계값 분석 (missing_data_default_analysis)
- `missing_threshold_default_analysis.csv` : 임계값(0,10,20,30,50%)별 데이터·Default·결측치 지표
- `column_missing_changes_by_threshold.csv` : 임계값별 컬럼별 결측치 비율 변화
- `missing_threshold_analysis_report.txt` : 상세 분석 리포트

### 3. Default 그룹 통계 분석 (default_group_analysis)
- `normal_companies_statistics.csv` : 정상기업 describe 통계
- `default_companies_statistics.csv` : 부실기업 describe 통계
- `mean_comparison_analysis.csv` : 평균값 차이 분석
- `std_comparison_analysis.csv` : 표준편차 차이 분석
- `comprehensive_group_comparison.csv` : 종합 비교 분석
- `default_group_analysis_report.txt` : Default 그룹 비교 리포트

---
*최종 업데이트: 2025-06-23*
