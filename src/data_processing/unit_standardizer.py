"""
재무데이터 단위 표준화
====================

기능:
1. FS_temp.csv에서 컬럼명 표준화 (단위 통일, 이름 간략화)
2. 모든 금액 단위를 원(元) 기준으로 통일 (천원×1000, 백만원×1000000)
3. 결과를 data/processed/FS_standardized.csv로 저장

사용법:
    python unit_standardizer.py
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime
import re

class UnitStandardizer:
    """재무데이터 단위 표준화 클래스"""
    
    def __init__(self, input_file: str = "data/processed/FS_temp.csv"):
        """
        초기화
        
        Args:
            input_file: FS_temp.csv 파일 경로
        """
        self.input_file = input_file
        self.project_root = Path(__file__).parent.parent.parent
        
        # 로깅 설정
        self.logger = self._setup_logging()
        
        # 컬럼명 표준화 매핑 (전체 176개 컬럼 매핑)
        self.column_standardization = {
            # 기본 정보 (키 컬럼들은 그대로 유지)
            '회사명': '회사명',
            '거래소코드': '거래소코드', 
            '회계년도': '회계년도',
            
            # === 재무상태표 항목들 ===
            '자산(*)(IFRS연결)(천원)': '총자산',
            '부채(*)(IFRS연결)(천원)': '총부채',
            '자본(*)(IFRS연결)(천원)': '총자본',
            '* 발행한 주식총수(*)(IFRS연결)(천원)': '발행주식총수',
            '유동자산(*)(IFRS연결)(천원)': '유동자산',
            '유동부채(*)(IFRS연결)(천원)': '유동부채',
            '자본금(*)(IFRS연결)(천원)': '자본금',
            '이익잉여금(결손금)(*)(IFRS연결)(천원)': '이익잉여금',
            # c_fs/i_fs 기반 재무상태표 항목 추가
            '유형자산(*)(IFRS연결)(천원)': '유형자산',
            '무형자산(*)(IFRS연결)(천원)': '무형자산',
            '재고자산(*)(IFRS연결)(천원)': '재고자산',
            '현금및현금성자산(*)(IFRS연결)(천원)': '현금및현금성자산',
            '비유동자산(*)(IFRS연결)(천원)': '비유동자산',
            '비유동부채 (*)(IFRS연결)(천원)': '비유동부채',
            '장기차입금(*)(IFRS연결)(천원)': '장기차입금',
            '단기차입금(*)(IFRS연결)(천원)': '단기차입금',
            '매출채권(IFRS연결)(천원)': '매출채권',
            '매입채무(IFRS연결)(천원)': '매입채무',
            '선수수익(IFRS연결)(천원)': '선수수익',
            '기타유동부채(IFRS연결)(천원)': '기타유동부채',
            '기타유동자산(IFRS연결)(천원)': '기타유동자산',
            '(금융)리스부채(IFRS연결)(천원)': '(금융)리스부채',
            '단기금융상품(금융기관예치금)(IFRS연결)(천원)': '단기금융상품(금융기관예치금)',
            '기타포괄손익누계액(*)(IFRS연결)(천원)': '기타포괄손익누계액',
            
            # === 손익계산서 항목들 ===
            '매출액(수익)(*)(IFRS연결)(천원)': '매출액',
            '* (정상)영업손익(보고서기재)(IFRS연결)(천원)': '영업이익',
            '당기순이익(손실)(IFRS연결)(천원)': '당기순이익',
            # c_fs/i_fs 기반 손익계산서 항목 추가
            '매출원가(*)(IFRS연결)(천원)': '매출원가',
            '매출총이익(손실)(IFRS연결)(천원)': '매출총이익',
            '계속영업이익(손실)(IFRS연결)(천원)': '계속영업이익',
            '이자비용(IFRS연결)(천원)': '이자비용',
            '납입자본금(IFRS연결)(천원)': '납입자본금',
            
            # === 현금흐름표 항목들 ===
            '영업활동으로 인한 현금흐름(간접법)(*)(IFRS연결)(천원)': '영업현금흐름',
            
            # === 수익성 지표들 ===
            '매출액증가율(IFRS연결)': '매출액증가율',
            '매출액총이익률(IFRS연결)': '매출액총이익률',
            '매출액정상영업이익률(IFRS연결)': '매출액영업이익률',
            '매출액순이익률(IFRS연결)': '매출액순이익률',
            '총자본순이익률(IFRS연결)': '총자산수익률',
            '자기자본순이익률(IFRS연결)': '자기자본순이익률',
            '총자본사업이익률(IFRS연결)': '총자본사업이익률',
            '총자본정상영업이익률(IFRS연결)': '총자본영업이익률',
            '자기자본정상영업이익률(IFRS연결)': '자기자본영업이익률',
            '경영자본정상영업이익률(IFRS연결)': '경영자본영업이익률',
            '경영자본순이익률(IFRS연결)': '경영자본순이익률',
            '자본금정상영업이익률(IFRS연결)': '자본금영업이익률',
            '자본금순이익률(IFRS연결)': '자본금순이익률',
            
            # === 안정성 지표들 ===
            '유동비율(IFRS연결)': '유동비율',
            '부채비율(IFRS연결)': '부채비율',
            '이자보상배율(이자비용)(IFRS연결)': '이자보상배율',
            '이자보상배율(순금융비용)(IFRS연결)': '순금융비용이자보상배율',
            '정상영업이익대비이자보상배율(IFRS연결)': '영업이익이자보상배율',
            '당좌비율(IFRS연결)': '당좌비율',
            '현금비율(IFRS연결)': '현금비율',
            '자기자본구성비율(IFRS연결)': '자기자본구성비율',
            '타인자본구성비율(IFRS연결)': '타인자본구성비율',
            '자기자본배율(IFRS연결)': '자기자본배율',
            '차입금의존도(IFRS연결)': '차입금의존도',
            '차입금비율(IFRS연결)': '차입금비율',
            
            # === 활동성 지표들 ===
            '총자본회전률(IFRS연결)': '총자본회전률',
            '경영자본회전률(IFRS연결)': '경영자본회전률',
            '자기자본회전률(IFRS연결)': '자기자본회전률',
            '자본금회전률(IFRS연결)': '자본금회전률',
            '타인자본회전률(IFRS연결)': '타인자본회전률',
            '유동자산회전률(IFRS연결)': '유동자산회전률',
            '당좌자산회전률(IFRS연결)': '당좌자산회전률',
            '재고자산회전률(IFRS연결)': '재고자산회전률',
            '재고자산회전기간(IFRS연결)': '재고자산회전기간',
            '매출채권회전률(IFRS연결)': '매출채권회전률',
            '매출채권회전기간(IFRS연결)': '매출채권회전기간',
            '매입채무회전률(IFRS연결)': '매입채무회전률',
            '매입채무회전기간(IFRS연결)': '매입채무회전기간',
            '비유동자산회전률(IFRS연결)': '비유동자산회전률',
            '유형자산회전율(IFRS연결)': '유형자산회전률',
            '순운전자본회전률(IFRS연결)': '순운전자본회전률',
            '운전자본회전률(IFRS연결)': '운전자본회전률',
            '1회전기간(IFRS연결)': '회전기간',
            
            # === 성장성 지표들 ===
            '총자본증가율(IFRS연결)': '총자본증가율',
            '유형자산증가율(IFRS연결)': '유형자산증가율',
            '비유동생물자산증가율(IFRS연결)': '비유동생물자산증가율',
            '투자부동산증가율(IFRS연결)': '투자부동산증가율',
            '비유동자산증가율(IFRS연결)': '비유동자산증가율',
            '유동자산증가율(IFRS연결)': '유동자산증가율',
            '재고자산증가율(IFRS연결)': '재고자산증가율',
            '자기자본증가율(IFRS연결)': '자기자본증가율',
            '정상영업이익증가율(IFRS연결)': '영업이익증가율',
            '순이익증가율(IFRS연결)': '순이익증가율',
            '총포괄이익증가율(IFRS연결)': '총포괄이익증가율',
            
            # === 비용구조 지표들 ===
            '매출원가 대 매출액비율(IFRS연결)': '매출원가율',
            '영업비용 대 영업수익비율(IFRS연결)': '영업비용률',
            '기타손익비률(IFRS연결)': '기타손익비율',
            '금융손익비율(IFRS연결)': '금융손익비율',
            '금융비용부담률(IFRS연결)': '금융비용부담률',
            '외환이익 대 매출액비율(IFRS연결)': '외환이익비율',
            '광고선전비 대 매출액비율(IFRS연결)': '광고선전비비율',
            '세금과공과 대 세금과공과 차감전순이익률(IFRS연결)': '세금과공과비율',
            '기업순이익률(IFRS연결)': '기업순이익률',
            '수지비율(관계기업투자손익 제외)(IFRS연결)': '수지비율',
            '인건비 대 총비용비율(IFRS연결)': '인건비비율',
            'R & D 투자효율(IFRS연결)': 'R&D투자효율',
            '세금과공과 대 총비용비율(IFRS연결)': '세금과공과총비용비율',
            '금융비용 대 총비용비율(IFRS연결)': '금융비용총비용비율',
            '감가상각비 대 총비용비율(IFRS연결)': '감가상각비총비용비율',
            
            # === 기타 재무지표들 ===
            '감가상각률(IFRS연결)': '감가상각률',
            '누적감가상각률(IFRS연결)': '누적감가상각률',
            '이자부담률(IFRS연결)': '이자부담률',
            '지급이자율(IFRS연결)': '지급이자율',
            '차입금평균이자율(IFRS연결)': '차입금평균이자율',
            '유보율(IFRS연결)': '유보율',
            '사내유보율(IFRS연결)': '사내유보율',
            '사내유보 대 자기자본비율(IFRS연결)': '사내유보자기자본비율',
            '적립금비율(재정비율)(IFRS연결)': '적립금비율',
            '평균배당률(IFRS연결)': '평균배당률',
            '자기자본배당률(IFRS연결)': '자기자본배당률',
            '배당성향(IFRS연결)': '배당성향',
            
            # === 주당 지표들 ===
            '1주당매출액(IFRS연결)(원)': '주당매출액',
            '1주당순이익(IFRS연결)(원)': '주당순이익',
            '1주당 CASH FLOW(IFRS연결)(원)': '주당현금흐름',
            '1주당순자산(IFRS연결)(원)': '주당순자산',
            '1주당정상영업이익(IFRS연결)(원)': '주당영업이익',
            
            # === 구성비율 지표들 ===
            '유동자산구성비율(IFRS연결)': '유동자산구성비율',
            '재고자산 대 유동자산비율(IFRS연결)': '재고자산유동자산비율',
            '유동자산 대 비유동자산비율(IFRS연결)': '유동비유동자산비율',
            '당좌자산구성비율(IFRS연결)': '당좌자산구성비율',
            '비유동자산구성비율(IFRS연결)': '비유동자산구성비율',
            '비유동비율(IFRS연결)': '비유동비율',
            '비유동장기적합률(IFRS연결)': '비유동장기적합률',
            '매출채권비율(IFRS연결)': '매출채권비율',
            '재고자산 대 순운전자본비율(IFRS연결)': '재고자산순운전자본비율',
            '매출채권 대 매입채무비율(IFRS연결)': '매출채권매입채무비율',
            '매출채권 대 상,제품비율(IFRS연결)': '매출채권상제품비율',
            '매입채무 대 재고자산비율(IFRS연결)': '매입채무재고자산비율',
            '유동부채비율(IFRS연결)': '유동부채비율',
            '단기차입금 대 총차입금비율(IFRS연결)': '단기차입금총차입금비율',
            '비유동부채비율(IFRS연결)': '비유동부채비율',
            '비유동부채 대 순운전자본비율(IFRS연결)': '비유동부채순운전자본비율',
            '순운전자본비율(IFRS연결)': '순운전자본비율',
            '유보액대비율(IFRS연결)': '유보액대비율',
            '유보액 대 납입자본배율(IFRS연결)': '유보액납입자본비율',
            '유동자산집중도(IFRS연결)': '유동자산집중도',
            '비유동자산집중도(IFRS연결)': '비유동자산집중도',
            '투자집중도(IFRS연결)': '투자집중도',
            
            # === 현금흐름 지표들 ===
            'CASH FLOW 대 부채비율(IFRS연결)': '현금흐름부채비율',
            'CASH FLOW 대 차입금비율(IFRS연결)': '현금흐름차입금비율',
            'CASH FLOW 대 총자본비율(IFRS연결)': '현금흐름총자본비율',
            'CASH FLOW 대 매출액비율(IFRS연결)': '현금흐름매출액비율',
            
            # === 회전률 특수 지표들 ===
            '상품,제품회전률(IFRS연결)': '상품제품회전률',
            '원,부재료회전률(IFRS연결)': '원부재료회전률',
            '재공품회전률(IFRS연결)': '재공품회전률',
            
            # === 생산성 지표들 (백만원 단위) ===
            '부가가치(IFRS연결)(백만원)': '부가가치',
            '종업원1인당 부가가치(IFRS연결)(백만원)': '종업원당부가가치',
            '종업원1인당 매출액(IFRS연결)(백만원)': '종업원당매출액',
            '종업원1인당 정상영업이익(IFRS연결)(백만원)': '종업원당영업이익',
            '종업원1인당 순이익(IFRS연결)(백만원)': '종업원당순이익',
            '종업원1인당 인건비(IFRS연결)(백만원)': '종업원당인건비',
            '법인세비용차감전(계속사업)손익(IFRS연결)(백만원)': '법인세비용차감전손익',
            '인건비(IFRS연결)(백만원)': '인건비',
            '금융비용(IFRS연결)(백만원)': '금융비용',
            '임차료(IFRS연결)(백만원)': '임차료',
            '세금과공과(IFRS연결)(백만원)': '세금과공과',
            '감가상각비(IFRS연결)(백만원)': '감가상각비',
            
            # === 투자효율 지표들 ===
            '노동장비율(IFRS연결)': '노동장비율',
            '기계장비율(IFRS연결)': '기계장비율',
            '자본집약도(IFRS연결)': '자본집약도',
            '총자본투자효율(IFRS연결)': '총자본투자효율',
            '설비투자효율(IFRS연결)': '설비투자효율',
            '기계투자효율(IFRS연결)': '기계투자효율',
            '부가가치율(IFRS연결)': '부가가치율',
            '노동소득분배율(IFRS연결)': '노동소득분배율',
            '자본분배율(IFRS연결)': '자본분배율',
            '이윤분배율(IFRS연결)': '이윤분배율',
            
            # === 종업원 관련 성장성 지표들 ===
            '종업원1인당 부가가치증가율(IFRS연결)': '종업원당부가가치증가율',
            '종업원수증가율(IFRS연결)': '종업원수증가율',
            '종업원1인당 매출액증가율(IFRS연결)': '종업원당매출액증가율',
            '종업원1인당 인건비증가율(IFRS연결)': '종업원당인건비증가율',
            '종업원수(IFRS연결)': '종업원수',

            
            # === 개별재무제표 고유 지표들 (밸류에이션) ===
            '기업가치(EV)(IFRS)(백만원)': '기업가치',
            'EBITDA(IFRS)(백만원)': 'EBITDA',
            'EV/EBITDA(IFRS)(배)': 'EV_EBITDA배수',
            'PER(최고)(IFRS)': 'PER최고',
            'PER(최저)(IFRS)': 'PER최저',
            'PBR(최고)(IFRS)': 'PBR최고',
            'PBR(최저)(IFRS)': 'PBR최저',
            'PCR(최고)(IFRS)': 'PCR최고',
            'PCR(최저)(IFRS)': 'PCR최저',
            'PSR(최고)(IFRS)': 'PSR최고',
            'PSR(최저)(IFRS)': 'PSR최저',
            'EBITDA/매출액(IFRS)(%)': 'EBITDA매출액비율',
            'EBITDA/금융비용(IFRS)(배)': 'EBITDA금융비용배수',
            'EBITDA/평균발행주식수(IFRS)(백만원)': '주당EBITDA',
        }
        
        # 단위 환산 비율 (원 기준)
        self.unit_conversion = {
            '원': 1,         # 기준 단위
            '천원': 1000,    # 원 기준으로 1000배
            '백만원': 1000000,  # 원 기준으로 1000000배
            '십억원': 1000000000,  # 원 기준으로 1000000000배
        }
        
        self.logger.info("UnitStandardizer 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('UnitStandardizer')
        logger.setLevel(logging.INFO)
        
        # 핸들러 제거 (중복 방지)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        self.logger.info("데이터 로드 시작")
        
        # FS_temp.csv 로드
        input_path = self.project_root / self.input_file
        if not input_path.exists():
            raise FileNotFoundError(f"FS_temp.csv 파일을 찾을 수 없습니다: {input_path}")
        
        df = pd.read_csv(input_path, dtype={'거래소코드': str})
        self.logger.info(f"FS_temp.csv 로드: {df.shape}")
        
        return df
    
    def extract_unit_from_column(self, col_name: str) -> str:
        """컬럼명에서 단위 추출"""
        # 괄호 안의 단위 추출 (긴 단위부터 체크)
        unit_patterns = ['십억원', '백만원', '천원', '원']
        for unit in unit_patterns:
            if unit in col_name:
                return unit
        return '원'  # 기본값
    
    def standardize_columns_and_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """컬럼명 표준화 및 단위 통일"""
        self.logger.info("컬럼명 표준화 및 단위 통일 시작")
        
        result_df = df.copy()
        
        # 컬럼별 처리
        for original_col, standard_col in self.column_standardization.items():
            if original_col not in result_df.columns:
                self.logger.warning(f"매핑된 컬럼이 데이터에 없음: {original_col}")
                continue
            
            # 단위 추출 및 변환
            unit = self.extract_unit_from_column(original_col)
            conversion_factor = self.unit_conversion.get(unit, 1)
            
            # 수치 데이터인 경우에만 단위 변환
            if pd.api.types.is_numeric_dtype(result_df[original_col]) and conversion_factor != 1:
                result_df[original_col] = result_df[original_col] * conversion_factor
                self.logger.info(f"단위 변환 적용: {original_col} (×{conversion_factor})")
            
            # 컬럼명 변경
            if original_col != standard_col:
                result_df = result_df.rename(columns={original_col: standard_col})
        
        # 매핑되지 않은 컬럼 확인
        unmapped_cols = [col for col in df.columns if col not in self.column_standardization]
        if unmapped_cols:
            self.logger.warning(f"매핑되지 않은 컬럼들: {unmapped_cols}")
        
        # 중복 컬럼 처리 (같은 표준화 이름으로 매핑된 경우)
        duplicate_cols = result_df.columns[result_df.columns.duplicated()].unique()
        if len(duplicate_cols) > 0:
            self.logger.warning(f"중복 컬럼 발견: {duplicate_cols.tolist()}")
            # 첫 번째 컬럼만 유지
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        
        self.logger.info(f"컬럼명 표준화 완료: {len(df.columns)} → {len(result_df.columns)}개 컬럼")
        
        return result_df
    
    def save_result(self, df: pd.DataFrame):
        """결과 저장"""
        output_dir = self.project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "FS2.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"결과 저장 완료: {output_path}")
        self.logger.info(f"최종 데이터 크기: {df.shape}")
        
        # 컬럼 정보 출력
        self.logger.info(f"최종 컬럼 목록: {df.columns.tolist()}")
        
        return output_path
    
    def run_pipeline(self) -> str:
        """전체 파이프라인 실행"""
        self.logger.info("=== 재무데이터 단위 표준화 파이프라인 시작 ===")
        
        try:
            # 1. 데이터 로드
            df = self.load_data()
            
            # 2. 컬럼명 표준화 및 단위 통일
            standardized_df = self.standardize_columns_and_units(df)
            
            # 3. 결과 저장
            output_path = self.save_result(standardized_df)
            
            self.logger.info("=== 재무데이터 단위 표준화 파이프라인 완료 ===")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 오류 발생: {str(e)}")
            raise


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='재무데이터 단위 표준화')
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/processed/FS_temp.csv',
        help='입력 파일 경로 (기본값: data/processed/FS_temp.csv)'
    )
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    standardizer = UnitStandardizer(args.input)
    output_path = standardizer.run_pipeline()
    
    print(f"\n✅ 단위 표준화 완료!")
    print(f"📁 결과 파일: {output_path}")
    
    # 추가 정보
    print(f"\n💡 참고사항:")
    print(f"   - 컬럼 표준화가 부정확하다면 코드 내 column_standardization 딕셔너리를 수정하세요")
    print(f"   - 단위 변환이 필요한 새로운 단위가 있다면 unit_conversion 딕셔너리에 추가하세요")


if __name__ == "__main__":
    main()