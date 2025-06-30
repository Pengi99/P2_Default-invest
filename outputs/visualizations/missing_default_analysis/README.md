# Missing Threshold Default Analysis - 시각화 자료

본 디렉토리는 **결측치 비율 임계값별(Default 분포 & 남은 결측치)** 분석 결과로 생성된 6개 시각화 파일을 포함합니다.

## 📁 파일 설명

| 파일명 | 설명 |
|--------|------|
| **01_missing_threshold_analysis.png** | 임계값(0,10,20,30,50%)별 데이터·Default 보존율 비교 |
| **02_data_count_changes.png** | 임계값별 남은 데이터 행 수 변화 |
| **03_default_rate_changes.png** | 임계값별 Default 비율 변화 |
| **05_remaining_missing_analysis.png** | 필터링 후 남은 결측치 총량·비율 분석 |
| **06_column_missing_changes_heatmap.png** | 컬럼별 결측치 비율 변화 히트맵 |
| **04_comprehensive_dashboard.png** | 데이터·Default·결측치 지표 종합 대시보드 |

## 🔍 활용 가이드
1. **01** : 임계값에 따른 데이터 손실과 Default 보존율 간 트레이드오프 시각화.
2. **02 / 03** : 모델 학습 데이터 크기와 Default 분포 안정성 평가.
3. **05 / 06** : 데이터 품질(남은 결측치)과 변수별 영향도 분석.
4. **04** : 핵심 지표를 한 화면에 요약하여 최적 임계값 결정에 도움.

---
*생성 스크립트: `src/analysis/missing_data_default_analysis.py`* 