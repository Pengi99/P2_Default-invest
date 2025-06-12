# fill_statements.py
# -*- coding: utf-8 -*-
"""
연결 재무제표(IFRS) NaN → 개별 재무제표(IFRS) 값으로 채우기
2025-06-12  by ChatGPT
"""

import pandas as pd
from pathlib import Path

# ────────────────────────── 1. 경로 ────────────────────────── #
base_path = Path("/Users/jojongho/KDT/P2_Default-invest/data/raw/연결재무비율.csv")
sub_path  = Path("/Users/jojongho/KDT/P2_Default-invest/data/raw/개별재무비율.csv")
out_path  = Path("/Users/jojongho/KDT/P2_Default-invest/data/processed/ratio.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)   # 폴더 없으면 생성

# ────────────────────────── 2. 읽기 ────────────────────────── #
read_kw = dict(encoding="utf-8-sig", low_memory=False)
base_df = pd.read_csv(base_path, **read_kw)
sub_df  = pd.read_csv(sub_path,  **read_kw)

# ────────────────────────── 3. 열 이름 맞추기 ────────────────────────── #
# (IFRS) → (IFRS연결)  로 치환
sub_df.columns = [c.replace("(IFRS)", "(IFRS연결)") if "(IFRS)" in c else c
                  for c in sub_df.columns]

# ────────────────────────── 4. 중복 제거 ────────────────────────── #
idx_cols = ["회사명", "거래소코드", "회계년도"]  # 행 식별 컬럼
base_df = base_df.drop_duplicates(subset=idx_cols, keep="first")
sub_df  = sub_df.drop_duplicates(subset=idx_cols, keep="first")

# ────────────────────────── 5. 인덱스 세팅 ────────────────────────── #
base_df = base_df.set_index(idx_cols)
sub_df  = sub_df.set_index(idx_cols)

# ────────────────────────── 6. NaN 채우기 ────────────────────────── #
# 1) 공통 열 값이 NaN일 때만 개별 재무제표 값으로 보충
base_df.update(sub_df, overwrite=False)

# 2) 개별 재무제표에만 있는 열도 이어붙임
extra_cols = sub_df.columns.difference(base_df.columns)
if extra_cols.any():
    base_df = pd.concat([base_df, sub_df[extra_cols]], axis=1)

# ────────────────────────── 7. 저장 ────────────────────────── #
base_df.reset_index().to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"[INFO] 결측치 보충 및 병합 완료  ➜  {out_path}")
