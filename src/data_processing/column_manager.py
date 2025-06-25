"""
컬럼 관리 유틸리티
================

데이터프레임의 컬럼을 편리하게 추가, 제거, 선택, 변경할 수 있는 유틸리티 클래스

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Callable
import logging
from pathlib import Path
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColumnManager:
    """
    데이터프레임 컬럼 관리 클래스
    
    컬럼 추가, 제거, 선택, 이름 변경 등의 작업을 편리하게 수행할 수 있습니다.
    """
    
    def __init__(self, df: pd.DataFrame, backup: bool = True):
        """
        ColumnManager 초기화
        
        Args:
            df: 관리할 데이터프레임
            backup: 백업 생성 여부
        """
        self.df = df.copy()
        self.original_df = df.copy() if backup else None
        self.backup_df = None
        self.column_history = []  # 변경 이력
        
        logger.info(f"ColumnManager 초기화 완료. 컬럼 수: {len(self.df.columns)}")
    
    def backup(self) -> 'ColumnManager':
        """현재 상태 백업"""
        self.backup_df = self.df.copy()
        logger.info("데이터프레임 백업 완료")
        return self
    
    def restore_backup(self) -> 'ColumnManager':
        """백업 상태로 복원"""
        if self.backup_df is not None:
            self.df = self.backup_df.copy()
            logger.info("백업으로 복원 완료")
        else:
            logger.warning("백업이 없습니다")
        return self
    
    def restore_original(self) -> 'ColumnManager':
        """원본 상태로 복원"""
        if self.original_df is not None:
            self.df = self.original_df.copy()
            logger.info("원본으로 복원 완료")
        else:
            logger.warning("원본 백업이 없습니다")
        return self
    
    # ===========================================
    # 컬럼 조회 및 탐색
    # ===========================================
    
    def list_columns(self, pattern: Optional[str] = None) -> List[str]:
        """
        컬럼 목록 조회
        
        Args:
            pattern: 정규식 패턴 (옵션)
            
        Returns:
            컬럼명 리스트
        """
        columns = list(self.df.columns)
        
        if pattern:
            columns = [col for col in columns if re.search(pattern, col, re.IGNORECASE)]
            
        return columns
    
    def find_columns(self, keyword: str, exact: bool = False) -> List[str]:
        """
        키워드로 컬럼 검색
        
        Args:
            keyword: 검색 키워드
            exact: 정확히 일치하는지 여부
            
        Returns:
            일치하는 컬럼명 리스트
        """
        if exact:
            return [col for col in self.df.columns if col == keyword]
        else:
            return [col for col in self.df.columns if keyword in col]
    
    def info(self) -> None:
        """데이터프레임 정보 출력"""
        print(f"\n{'='*50}")
        print(f"데이터프레임 정보")
        print(f"{'='*50}")
        print(f"행 수: {len(self.df):,}")
        print(f"컬럼 수: {len(self.df.columns)}")
        print(f"메모리 사용량: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"\n컬럼 타입별 개수:")
        print(self.df.dtypes.value_counts())
        
    # ===========================================
    # 컬럼 선택 및 제거
    # ===========================================
    
    def select_columns(self, columns: Union[List[str], str]) -> 'ColumnManager':
        """
        특정 컬럼들만 선택
        
        Args:
            columns: 선택할 컬럼명 또는 컬럼명 리스트
            
        Returns:
            ColumnManager 인스턴스
        """
        if isinstance(columns, str):
            columns = [columns]
            
        # 존재하지 않는 컬럼 확인
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            logger.warning(f"존재하지 않는 컬럼: {missing_cols}")
            columns = [col for col in columns if col in self.df.columns]
        
        self.df = self.df[columns]
        self.column_history.append(f"컬럼 선택: {len(columns)}개")
        logger.info(f"{len(columns)}개 컬럼 선택 완료")
        
        return self
    
    def drop_columns(self, columns: Union[List[str], str]) -> 'ColumnManager':
        """
        컬럼 제거
        
        Args:
            columns: 제거할 컬럼명 또는 컬럼명 리스트
            
        Returns:
            ColumnManager 인스턴스
        """
        if isinstance(columns, str):
            columns = [columns]
            
        # 존재하는 컬럼만 제거
        existing_cols = [col for col in columns if col in self.df.columns]
        missing_cols = [col for col in columns if col not in self.df.columns]
        
        if missing_cols:
            logger.warning(f"존재하지 않는 컬럼 (무시됨): {missing_cols}")
        
        if existing_cols:
            self.df = self.df.drop(columns=existing_cols)
            self.column_history.append(f"컬럼 제거: {existing_cols}")
            logger.info(f"{len(existing_cols)}개 컬럼 제거 완료")
        
        return self
    
    def keep_only(self, pattern: str) -> 'ColumnManager':
        """
        패턴에 매칭되는 컬럼만 유지
        
        Args:
            pattern: 정규식 패턴
            
        Returns:
            ColumnManager 인스턴스
        """
        matching_cols = [col for col in self.df.columns 
                        if re.search(pattern, col, re.IGNORECASE)]
        
        return self.select_columns(matching_cols)
    
    def drop_by_pattern(self, pattern: str) -> 'ColumnManager':
        """
        패턴에 매칭되는 컬럼 제거
        
        Args:
            pattern: 정규식 패턴
            
        Returns:
            ColumnManager 인스턴스
        """
        matching_cols = [col for col in self.df.columns 
                        if re.search(pattern, col, re.IGNORECASE)]
        
        return self.drop_columns(matching_cols)
    
    # ===========================================
    # 컬럼 추가 및 변경
    # ===========================================
    
    def add_column(self, name: str, data: Union[pd.Series, List, np.ndarray, Callable], 
                   position: Optional[int] = None) -> 'ColumnManager':
        """
        새 컬럼 추가
        
        Args:
            name: 컬럼명
            data: 데이터 (Series, List, Array 또는 함수)
            position: 삽입 위치 (None이면 마지막에 추가)
            
        Returns:
            ColumnManager 인스턴스
        """
        # 함수인 경우 실행
        if callable(data):
            data = data(self.df)
        
        if position is None:
            self.df[name] = data
        else:
            # 특정 위치에 삽입
            self.df.insert(position, name, data)
        
        self.column_history.append(f"컬럼 추가: {name}")
        logger.info(f"컬럼 '{name}' 추가 완료")
        
        return self
    
    def rename_columns(self, mapping: Dict[str, str]) -> 'ColumnManager':
        """
        컬럼명 변경
        
        Args:
            mapping: {기존명: 새이름} 딕셔너리
            
        Returns:
            ColumnManager 인스턴스
        """
        # 존재하는 컬럼만 변경
        existing_mapping = {old: new for old, new in mapping.items() 
                           if old in self.df.columns}
        missing_cols = [old for old in mapping.keys() if old not in self.df.columns]
        
        if missing_cols:
            logger.warning(f"존재하지 않는 컬럼 (무시됨): {missing_cols}")
        
        if existing_mapping:
            self.df = self.df.rename(columns=existing_mapping)
            self.column_history.append(f"컬럼명 변경: {list(existing_mapping.keys())}")
            logger.info(f"{len(existing_mapping)}개 컬럼명 변경 완료")
        
        return self
    
    def add_prefix(self, prefix: str, columns: Optional[List[str]] = None) -> 'ColumnManager':
        """
        컬럼에 접두사 추가
        
        Args:
            prefix: 접두사
            columns: 대상 컬럼 (None이면 모든 컬럼)
            
        Returns:
            ColumnManager 인스턴스
        """
        if columns is None:
            columns = list(self.df.columns)
        
        mapping = {col: f"{prefix}{col}" for col in columns if col in self.df.columns}
        return self.rename_columns(mapping)
    
    def add_suffix(self, suffix: str, columns: Optional[List[str]] = None) -> 'ColumnManager':
        """
        컬럼에 접미사 추가
        
        Args:
            suffix: 접미사
            columns: 대상 컬럼 (None이면 모든 컬럼)
            
        Returns:
            ColumnManager 인스턴스
        """
        if columns is None:
            columns = list(self.df.columns)
        
        mapping = {col: f"{col}{suffix}" for col in columns if col in self.df.columns}
        return self.rename_columns(mapping)
    
    # ===========================================
    # 금융 데이터 특화 기능
    # ===========================================
    
    def get_financial_ratios(self) -> List[str]:
        """금융 비율 컬럼들 반환"""
        ratio_patterns = ['비율', '률$', '율$', '배수', '배율']
        ratio_cols = []
        
        for pattern in ratio_patterns:
            ratio_cols.extend([col for col in self.df.columns 
                             if re.search(pattern, col)])
        
        return list(set(ratio_cols))
    
    def get_growth_rates(self) -> List[str]:
        """성장률 컬럼들 반환"""
        return [col for col in self.df.columns if '증가율' in col]
    
    def get_absolute_values(self) -> List[str]:
        """절대값 컬럼들 반환 (총자산, 매출액 등)"""
        absolute_patterns = ['총자산', '총부채', '총자본', '매출액', '영업이익', 
                           '당기순이익', '자본금', '유동자산', '유동부채']
        
        absolute_cols = []
        for pattern in absolute_patterns:
            absolute_cols.extend([col for col in self.df.columns 
                                if pattern in col and '비율' not in col and '률' not in col])
        
        return list(set(absolute_cols))
    
    def separate_by_type(self) -> Dict[str, List[str]]:
        """컬럼을 유형별로 분류"""
        return {
            'absolute_values': self.get_absolute_values(),
            'ratios': self.get_financial_ratios(),
            'growth_rates': self.get_growth_rates(),
            'others': [col for col in self.df.columns 
                      if col not in self.get_absolute_values() + 
                                 self.get_financial_ratios() + 
                                 self.get_growth_rates()]
        }
    
    # ===========================================
    # 컬럼 순서 및 정렬
    # ===========================================
    
    def reorder_columns(self, new_order: List[str]) -> 'ColumnManager':
        """
        컬럼 순서 변경
        
        Args:
            new_order: 새로운 컬럼 순서
            
        Returns:
            ColumnManager 인스턴스
        """
        # 존재하는 컬럼만 사용
        existing_cols = [col for col in new_order if col in self.df.columns]
        remaining_cols = [col for col in self.df.columns if col not in existing_cols]
        
        final_order = existing_cols + remaining_cols
        self.df = self.df[final_order]
        
        self.column_history.append("컬럼 순서 변경")
        logger.info("컬럼 순서 변경 완료")
        
        return self
    
    def sort_columns(self, reverse: bool = False) -> 'ColumnManager':
        """
        컬럼명 알파벳 순 정렬
        
        Args:
            reverse: 역순 정렬 여부
            
        Returns:
            ColumnManager 인스턴스
        """
        sorted_cols = sorted(self.df.columns, reverse=reverse)
        self.df = self.df[sorted_cols]
        
        self.column_history.append(f"컬럼 정렬 (역순={reverse})")
        logger.info("컬럼 정렬 완료")
        
        return self
    
    # ===========================================
    # 조건부 작업
    # ===========================================
    
    def filter_by_missing_rate(self, max_missing_rate: float = 0.5) -> 'ColumnManager':
        """
        결측치 비율에 따른 컬럼 필터링
        
        Args:
            max_missing_rate: 최대 허용 결측치 비율
            
        Returns:
            ColumnManager 인스턴스
        """
        missing_rates = self.df.isnull().mean()
        keep_cols = missing_rates[missing_rates <= max_missing_rate].index.tolist()
        
        removed_count = len(self.df.columns) - len(keep_cols)
        self.df = self.df[keep_cols]
        
        self.column_history.append(f"결측치 필터링: {removed_count}개 제거")
        logger.info(f"결측치 비율 {max_missing_rate} 초과 컬럼 {removed_count}개 제거")
        
        return self
    
    def filter_by_variance(self, min_variance: float = 1e-10) -> 'ColumnManager':
        """
        분산에 따른 컬럼 필터링
        
        Args:
            min_variance: 최소 분산
            
        Returns:
            ColumnManager 인스턴스
        """
        # 숫자 컬럼만 대상
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        variances = self.df[numeric_cols].var()
        
        keep_cols = variances[variances >= min_variance].index.tolist()
        non_numeric_cols = [col for col in self.df.columns if col not in numeric_cols]
        
        final_cols = keep_cols + non_numeric_cols
        removed_count = len(numeric_cols) - len(keep_cols)
        
        self.df = self.df[final_cols]
        
        self.column_history.append(f"분산 필터링: {removed_count}개 제거")
        logger.info(f"분산 {min_variance} 미만 컬럼 {removed_count}개 제거")
        
        return self
    
    # ===========================================
    # 저장 및 내보내기
    # ===========================================
    
    def save(self, filepath: str, **kwargs) -> 'ColumnManager':
        """
        데이터프레임 저장
        
        Args:
            filepath: 저장 경로
            **kwargs: pandas.to_csv 추가 옵션
            
        Returns:
            ColumnManager 인스턴스
        """
        self.df.to_csv(filepath, index=False, **kwargs)
        logger.info(f"데이터프레임 저장 완료: {filepath}")
        return self
    
    def get_dataframe(self) -> pd.DataFrame:
        """현재 데이터프레임 반환"""
        return self.df.copy()
    
    def show_history(self) -> None:
        """변경 이력 출력"""
        print("\n변경 이력:")
        print("-" * 30)
        for i, change in enumerate(self.column_history, 1):
            print(f"{i}. {change}")


# ===========================================
# 편의 함수들
# ===========================================

def quick_column_selector(df: pd.DataFrame, 
                         include_patterns: Optional[List[str]] = None,
                         exclude_patterns: Optional[List[str]] = None) -> List[str]:
    """
    패턴 기반 빠른 컬럼 선택
    
    Args:
        df: 데이터프레임
        include_patterns: 포함할 패턴들
        exclude_patterns: 제외할 패턴들
        
    Returns:
        선택된 컬럼명 리스트
    """
    columns = list(df.columns)
    
    # 포함 패턴 적용
    if include_patterns:
        included = []
        for pattern in include_patterns:
            included.extend([col for col in columns 
                           if re.search(pattern, col, re.IGNORECASE)])
        columns = list(set(included))
    
    # 제외 패턴 적용
    if exclude_patterns:
        for pattern in exclude_patterns:
            columns = [col for col in columns 
                      if not re.search(pattern, col, re.IGNORECASE)]
    
    return columns


def load_and_manage(filepath: str) -> ColumnManager:
    """
    파일 로드 후 ColumnManager 생성
    
    Args:
        filepath: 파일 경로
        
    Returns:
        ColumnManager 인스턴스
    """
    df = pd.read_csv(filepath)
    return ColumnManager(df)


# ===========================================
# 사용 예시
# ===========================================

if __name__ == "__main__":
    # 사용 예시
    print("ColumnManager 사용 예시")
    print("=" * 50)
    
    # 샘플 데이터 생성
    sample_data = {
        '총자산': [1000, 2000, 1500],
        '매출액': [500, 800, 600],
        '매출액증가율': [0.1, 0.2, -0.05],
        '부채비율': [0.6, 0.4, 0.7],
        '회사명': ['A', 'B', 'C'],
        '기타데이터': [1, 2, 3]
    }
    
    df = pd.DataFrame(sample_data)
    
    # ColumnManager 사용
    cm = ColumnManager(df)
    
    print("1. 초기 상태:")
    cm.info()
    
    print("\n2. 금융 비율 컬럼 조회:")
    print(cm.get_financial_ratios())
    
    print("\n3. 성장률 컬럼 조회:")
    print(cm.get_growth_rates())
    
    print("\n4. 절대값 컬럼 조회:")
    print(cm.get_absolute_values())
    
    # 체인 방식으로 여러 작업 수행
    result = (cm
              .backup()  # 백업
              .add_column('새컬럼', lambda x: x['총자산'] * 2)  # 컬럼 추가
              .rename_columns({'기타데이터': '기타_데이터'})  # 이름 변경
              .add_prefix('FIN_', ['총자산', '매출액'])  # 접두사 추가
              .sort_columns()  # 정렬
              )
    
    print("\n5. 작업 후 컬럼:")
    print(result.list_columns())
    
    print("\n6. 변경 이력:")
    result.show_history() 