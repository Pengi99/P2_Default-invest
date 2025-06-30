
# label_default_iterative.py
# value_fail.csv와 final.csv를 사용하여 Default 라벨링 및 필터링 수행 (병합 없이 개별 처리)

import pandas as pd

# 1. 데이터 로드
# value_fail.csv에는 ['company', 'company_code', 'year'] 컬럼이 있다고 가정
fail_df = pd.read_csv('C:\Users\82106\Desktop\유비온 데이터분석\2차 프로젝트\P2_Default-invest\data\processed\value_fail.csv')
# final.csv에는 ['company', 'company_code', 'year', ...] 컬럼이 있다고 가정
final_df = pd.read_csv('C:\Users\82106\Desktop\유비온 데이터분석\2차 프로젝트\P2_Default-invest\data\processed\final.csv')

# 2. 기본 default 컬럼 생성 (모두 0으로 초기화)
final_df['default'] = 0

# 3. value_fail.csv를 하나씩 확인하며 처리
for _, fail in fail_df.iterrows():
    comp = fail['company']
    code = fail.get('company_code', None)
    fail_year = int(fail['year'])
    prev_year = fail_year - 1

    # 동일 기업에 대해 직전 연도(default=1) 라벨링
    mask_prev = (
        (final_df['company'] == comp) &
        ((code is None) | (final_df['company_code'] == code)) &
        (final_df['year'] == prev_year)
    )
    final_df.loc[mask_prev, 'default'] = 1

    # 동일 기업의 직전년도 이전(year < prev_year) 데이터는 제거
    mask_drop = (
        (final_df['company'] == comp) &
        ((code is None) | (final_df['company_code'] == code)) &
        (final_df['year'] < prev_year)
    )
    final_df = final_df[~mask_drop]

# 4. value_fail.csv에 없는 기업들은 default=0으로 유지 (이미 초기화됨)

# 5. 결과 저장
final_df.to_csv('final_labeled.csv', index=False)
print("Default 라벨링 및 필터링 완료: 'final_labeled.csv' 생성")
