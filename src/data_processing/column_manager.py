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
import yaml

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
    

    
    # ===========================================
    # 조건부 작업
    # ===========================================
    

    
    # ===========================================
    # YAML 기반 컬럼 관리
    # ===========================================
    
    def load_column_config(self, yaml_path: str) -> 'ColumnManager':
        """
        YAML 파일에서 컬럼 설정 로드 및 적용
        
        Args:
            yaml_path: YAML 설정 파일 경로
            
        Returns:
            ColumnManager 인스턴스
        """
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"YAML 설정 파일 로드: {yaml_path}")
            
            # 백업 생성
            self.backup()
            
            # 설정 적용
            self._apply_yaml_config(config)
            
            return self
            
        except Exception as e:
            logger.error(f"YAML 설정 로드 실패: {e}")
            raise
    
    def _apply_yaml_config(self, config: Dict) -> None:
        """YAML 설정 적용 (간단 버전)"""
        
        # 1. 명시적 컬럼 제거
        if 'drop_columns' in config and config['drop_columns']:
            self.drop_columns(config['drop_columns'])
        
        # 2. 명시적 컬럼 유지
        if 'keep_columns' in config and config['keep_columns']:
            self.select_columns(config['keep_columns'])
    

    
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


def create_column_config_template(output_path: str = "column_config.yaml") -> None:
    """
    컬럼 관리 YAML 설정 파일 템플릿 생성 (간단 버전)
    
    Args:
        output_path: 출력 파일 경로
    """
    template = {
        'config': {
            'description': '컬럼 관리 설정 파일',
            'version': '1.0',
            'author': 'ColumnManager'
        },
        
        # 살릴 컬럼들 (명시적 지정)
        'keep_columns': [
            '총자산',
            '매출액', 
            '당기순이익',
            '영업이익',
            '부채비율',
            '유동비율'
        ],
        
        # 죽일 컬럼들 (명시적 지정)
        'drop_columns': [
            '회사명',
            '거래소코드',
            '회계년도'
        ]
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(template, f, allow_unicode=True, default_flow_style=False, indent=2)
        
        print(f"✅ 컬럼 설정 템플릿 생성 완료: {output_path}")
        print(f"📝 파일을 편집하여 원하는 컬럼 설정을 만드세요!")
        
    except Exception as e:
        print(f"❌ 템플릿 생성 실패: {e}")


def apply_column_config_from_yaml(df: pd.DataFrame, yaml_path: str) -> pd.DataFrame:
    """
    YAML 설정을 데이터프레임에 바로 적용
    
    Args:
        df: 원본 데이터프레임
        yaml_path: YAML 설정 파일 경로
        
    Returns:
        처리된 데이터프레임
    """
    cm = ColumnManager(df)
    return cm.load_column_config(yaml_path).get_dataframe()


# ===========================================
# 실행 함수
# ===========================================


def process_fs_data():
    """FS.csv 파일을 처리하여 FS_filtered.csv로 저장"""
    print("📊 FS.csv 데이터 처리 시작")
    print("=" * 60)
    
    # 파일 경로 설정 (프로젝트 루트 기준으로 절대 경로 사용)
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / 'data' / 'processed' / 'FS2_default.csv'
    output_file = project_root / 'data' / 'processed' / 'FS2_filtered.csv'
    config_file = project_root / 'config' / 'column_config.yaml'
    
    try:
        # 1. 데이터 로드
        print(f"\n🔄 데이터 로드 중: {input_file}")
        df = pd.read_csv(input_file)
        print(f"✅ 원본 데이터 로드 완료: {df.shape}")
        print(f"   총 컬럼 수: {len(df.columns)}")
        
        # 2. ColumnManager 생성
        cm = ColumnManager(df, backup=True)
        
        # 3. 기본 정보 출력
        print(f"\n📋 데이터 기본 정보:")
        cm.info()
        
        # 4. 금융 변수 분류
        print(f"\n🎯 금융 변수 자동 분류:")
        classification = cm.separate_by_type()
        for category, columns in classification.items():
            print(f"  {category}: {len(columns)}개 컬럼")
            if len(columns) <= 5:
                print(f"    {columns}")
            else:
                print(f"    {columns[:5]}... (외 {len(columns)-5}개)")
        
        # 5. 컬럼 설정 파일 확인 및 적용
        if Path(config_file).exists():
            print(f"\n📋 YAML 설정 파일 적용: {config_file}")
            cm.load_column_config(config_file)
            print(f"✅ 설정 적용 완료: {len(cm.list_columns())}개 컬럼 유지")
        else:
            print(f"\n⚠️ 설정 파일이 없습니다: {config_file}")
            print("📝 기본 필터링을 적용합니다...")
            
            # 기본 필터링: 메타 정보 제거
            meta_columns = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['회사명', '코드', '년도', '날짜', 'id']):
                    meta_columns.append(col)
            
            if meta_columns:
                print(f"   메타 컬럼 제거: {meta_columns}")
                cm.drop_columns(meta_columns)
            
            # 결측치가 너무 많은 컬럼 제거 (90% 이상)
            high_missing_cols = []
            for col in cm.list_columns():
                missing_rate = cm.df[col].isnull().sum() / len(cm.df)
                if missing_rate > 0.9:
                    high_missing_cols.append(col)
            
            if high_missing_cols:
                print(f"   고결측 컬럼 제거 (90%+): {len(high_missing_cols)}개")
                cm.drop_columns(high_missing_cols)
        
        # 6. 최종 결과 정보
        final_columns = cm.list_columns()
        print(f"\n📊 최종 결과:")
        print(f"   원본 컬럼 수: {len(df.columns)}")
        print(f"   필터링 후: {len(final_columns)}")
        print(f"   제거된 컬럼: {len(df.columns) - len(final_columns)}")
        
        # 7. 변경 이력 출력
        print(f"\n📈 변경 이력:")
        cm.show_history()
        
        # 8. 결과 저장
        print(f"\n💾 결과 저장 중: {output_file}")
        cm.save(output_file, encoding='utf-8-sig')
        
        # 9. 저장된 파일 확인
        saved_df = pd.read_csv(output_file)
        print(f"✅ 저장 완료: {saved_df.shape}")
        
        # 10. 최종 컬럼 목록 출력 (처음 20개만)
        print(f"\n📝 최종 컬럼 목록 (처음 20개):")
        for i, col in enumerate(final_columns[:20], 1):
            print(f"   {i:2d}. {col}")
        if len(final_columns) > 20:
            print(f"   ... 외 {len(final_columns)-20}개 컬럼")
        
        print(f"\n🎉 FS.csv 처리 완료!")
        print(f"📁 출력 파일: {output_file}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        print(f"💡 다음 위치에 FS.csv 파일이 있는지 확인하세요: {Path(input_file).absolute()}")
        return False
        
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 ColumnManager - FS.csv 처리 도구")
    print("=" * 60)
    
    try:
        success = process_fs_data()
        if success:
            print("\n✨ 작업이 성공적으로 완료되었습니다!")
        else:
            print("\n💥 작업이 실패했습니다.")
            
    except KeyboardInterrupt:
        print("\n\n👋 종료합니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("💡 FS.csv 처리를 시도합니다.")
        process_fs_data() 