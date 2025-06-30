"""
재무데이터 병합 및 결측치 채우기
===============================

기능:
1. 두 개의 CSV 파일을 ['거래소코드', '회계년도'] 기준으로 병합
2. 첫 번째 CSV의 결측치를 두 번째 CSV 데이터로 채움
3. 컬럼명 매칭 (예: 자산(IFRS연결)(천원) ↔ 자산(IFRS)(천원))
4. 두 번째 파일에만 있는 컬럼 추가
5. 회계년도 필터링 (12월만 유지)
6. 결과를 data/processed/FS_temp.csv로 저장

사용법:
    python step6_merge_fill_financial_data.py
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime
import re

class FinancialDataMerger:
    """재무데이터 병합 및 결측치 채우기 클래스"""
    
    def __init__(self, primary_file: str, secondary_file: str):
        """
        초기화
        
        Args:
            primary_file: 첫 번째 CSV 파일 경로 (메인 데이터)
            secondary_file: 두 번째 CSV 파일 경로 (보완 데이터)
        """
        self.primary_file = primary_file
        self.secondary_file = secondary_file
        self.project_root = Path(__file__).parent.parent.parent
        
        # 로깅 설정
        self.logger = self._setup_logging()
        
        # 컬럼 매칭 딕셔너리 (연결.csv → 개별.csv 전체 수동 매핑)
        self.column_mapping = {
            # 기본 키 컬럼들 (동일)
            '회사명': '회사명',
            '거래소코드': '거래소코드', 
            '회계년도': '회계년도',
            
            # 기본 재무제표 항목들
            '자산(*)(IFRS연결)(천원)': '자산(*)(IFRS)(천원)',
            '부채(*)(IFRS연결)(천원)': '부채(*)(IFRS)(천원)',
            '자본(*)(IFRS연결)(천원)': '자본(*)(IFRS)(천원)',
            '* 발행한 주식총수(*)(IFRS연결)(천원)': '* 발행한 주식총수(*)(IFRS)(주)',  # 단위 다름
            '유동자산(*)(IFRS연결)(천원)': '유동자산(*)(IFRS)(천원)',
            '유동부채(*)(IFRS연결)(천원)': '유동부채(*)(IFRS)(천원)',
            '매출액(수익)(*)(IFRS연결)(천원)': '매출액(수익)(*)(IFRS)(천원)',
            '자본금(*)(IFRS연결)(천원)': '자본금(*)(IFRS)(천원)',
            '이익잉여금(결손금)(*)(IFRS연결)(천원)': '이익잉여금(결손금)(*)(IFRS)(천원)',
            '* (정상)영업손익(보고서기재)(IFRS연결)(천원)': '* (정상)영업손익(보고서기재)(IFRS)(천원)',
            '당기순이익(손실)(IFRS연결)(천원)': '당기순이익(손실)(IFRS)(천원)',
            '영업활동으로 인한 현금흐름(간접법)(*)(IFRS연결)(천원)': '영업활동으로 인한 현금흐름(간접법)(*)(IFRS)(천원)',
            
            # 재무비율들
            '매출액증가율(IFRS연결)': '매출액증가율(IFRS)',
            '매출액총이익률(IFRS연결)': '매출액총이익률(IFRS)',
            '매출액정상영업이익률(IFRS연결)': '매출액정상영업이익률(IFRS)',
            '매출액순이익률(IFRS연결)': '매출액순이익률(IFRS)',
            '총자본순이익률(IFRS연결)': '총자본순이익률(IFRS)',
            '자기자본순이익률(IFRS연결)': '자기자본순이익률(IFRS)',
            '유동비율(IFRS연결)': '유동비율(IFRS)',
            '부채비율(IFRS연결)': '부채비율(IFRS)',
            '이자보상배율(이자비용)(IFRS연결)': '이자보상배율(이자비용)(IFRS)',
            '총자본회전률(IFRS연결)': '총자본회전률(IFRS)',
            
            # 증가율 지표들
            '총자본증가율(IFRS연결)': '총자본증가율(IFRS)',
            '유형자산증가율(IFRS연결)': '유형자산증가율(IFRS)',
            '비유동생물자산증가율(IFRS연결)': '비유동생물자산증가율(IFRS)',
            '투자부동산증가율(IFRS연결)': '투자부동산증가율(IFRS)',
            '비유동자산증가율(IFRS연결)': '비유동자산증가율(IFRS)',
            '유동자산증가율(IFRS연결)': '유동자산증가율(IFRS)',
            '재고자산증가율(IFRS연결)': '재고자산증가율(IFRS)',
            '자기자본증가율(IFRS연결)': '자기자본증가율(IFRS)',
            '정상영업이익증가율(IFRS연결)': '정상영업이익증가율(IFRS)',
            '순이익증가율(IFRS연결)': '순이익증가율(IFRS)',
            '총포괄이익증가율(IFRS연결)': '총포괄이익증가율(IFRS)',
            '종업원1인당 부가가치증가율(IFRS연결)': '종업원1인당 부가가치증가율(IFRS)',
            '종업원수증가율(IFRS연결)': '종업원수증가율(IFRS)',
            '종업원1인당 매출액증가율(IFRS연결)': '종업원1인당 매출액증가율(IFRS)',
            '종업원1인당 인건비증가율(IFRS연결)': '종업원1인당 인건비증가율(IFRS)',
            
            # 수익성 지표들
            '총자본사업이익률(IFRS연결)': '총자본사업이익률(IFRS)',
            '총자본정상영업이익률(IFRS연결)': '총자본정상영업이익률(IFRS)',
            '자기자본정상영업이익률(IFRS연결)': '자기자본정상영업이익률(IFRS)',
            '경영자본정상영업이익률(IFRS연결)': '경영자본정상영업이익률(IFRS)',
            '경영자본순이익률(IFRS연결)': '경영자본순이익률(IFRS)',
            '자본금정상영업이익률(IFRS연결)': '자본금정상영업이익률(IFRS)',
            '자본금순이익률(IFRS연결)': '자본금순이익률(IFRS)',
            
            # 비용구조 지표들
            '매출원가 대 매출액비율(IFRS연결)': '매출원가 대 매출액비율(IFRS)',
            '영업비용 대 영업수익비율(IFRS연결)': '영업비용 대 영업수익비율(IFRS)',
            '기타손익비률(IFRS연결)': '기타손익비률(IFRS)',
            '금융손익비율(IFRS연결)': '금융손익비율(IFRS)',
            '금융비용부담률(IFRS연결)': '금융비용부담률(IFRS)',
            '외환이익 대 매출액비율(IFRS연결)': '외환이익 대 매출액비율(IFRS)',
            '광고선전비 대 매출액비율(IFRS연결)': '광고선전비 대 매출액비율(IFRS)',
            '세금과공과 대 세금과공과 차감전순이익률(IFRS연결)': '세금과공과 대 세금과공과 차감전순이익률(IFRS)',
            '기업순이익률(IFRS연결)': '기업순이익률(IFRS)',
            '수지비율(관계기업투자손익 제외)(IFRS연결)': '수지비율(관계기업투자손익 제외)(IFRS)',
            '인건비 대 총비용비율(IFRS연결)': '인건비 대 총비용비율(IFRS)',
            'R & D 투자효율(IFRS연결)': 'R & D 투자효율(IFRS)',
            '세금과공과 대 총비용비율(IFRS연결)': '세금과공과 대 총비용비율(IFRS)',
            '금융비용 대 총비용비율(IFRS연결)': '금융비용 대 총비용비율(IFRS)',
            '감가상각비 대 총비용비율(IFRS연결)': '감가상각비 대 총비용비율(IFRS)',
            
            # 기본 지표들
            '감가상각률(IFRS연결)': '감가상각률(IFRS)',
            '누적감가상각률(IFRS연결)': '누적감가상각률(IFRS)',
            '이자부담률(IFRS연결)': '이자부담률(IFRS)',
            '지급이자율(IFRS연결)': '지급이자율(IFRS)',
            '차입금평균이자율(IFRS연결)': '차입금평균이자율(IFRS)',
            '유보율(IFRS연결)': '유보율(IFRS)',
            '사내유보율(IFRS연결)': '사내유보율(IFRS)',
            '사내유보 대 자기자본비율(IFRS연결)': '사내유보 대 자기자본비율(IFRS)',
            '적립금비율(재정비율)(IFRS연결)': '적립금비율(재정비율)(IFRS)',
            '평균배당률(IFRS연결)': '평균배당률(IFRS)',
            '자기자본배당률(IFRS연결)': '자기자본배당률(IFRS)',
            '배당성향(IFRS연결)': '배당성향(IFRS)',
            
            # 주당 지표들
            '1주당매출액(IFRS연결)(원)': '1주당매출액(IFRS)(원)',
            '1주당순이익(IFRS연결)(원)': '1주당순이익(IFRS)(원)',
            '1주당 CASH FLOW(IFRS연결)(원)': '1주당 CASH FLOW(IFRS)(원)',
            '1주당순자산(IFRS연결)(원)': '1주당순자산(IFRS)(원)',
            '1주당정상영업이익(IFRS연결)(원)': '1주당정상영업이익(IFRS)(원)',
            
            # 구성비율 및 안정성 지표들
            '유동자산구성비율(IFRS연결)': '유동자산구성비율(IFRS)',
            '재고자산 대 유동자산비율(IFRS연결)': '재고자산 대 유동자산비율(IFRS)',
            '유동자산 대 비유동자산비율(IFRS연결)': '유동자산 대 비유동자산비율(IFRS)',
            '당좌자산구성비율(IFRS연결)': '당좌자산구성비율(IFRS)',
            '비유동자산구성비율(IFRS연결)': '비유동자산구성비율(IFRS)',
            '자기자본구성비율(IFRS연결)': '자기자본구성비율(IFRS)',
            '타인자본구성비율(IFRS연결)': '타인자본구성비율(IFRS)',
            '자기자본배율(IFRS연결)': '자기자본배율(IFRS)',
            '비유동비율(IFRS연결)': '비유동비율(IFRS)',
            '비유동장기적합률(IFRS연결)': '비유동장기적합률(IFRS)',
            '당좌비율(IFRS연결)': '당좌비율(IFRS)',
            '현금비율(IFRS연결)': '현금비율(IFRS)',
            '매출채권비율(IFRS연결)': '매출채권비율(IFRS)',
            '재고자산 대 순운전자본비율(IFRS연결)': '재고자산 대 순운전자본비율(IFRS)',
            '매출채권 대 매입채무비율(IFRS연결)': '매출채권 대 매입채무비율(IFRS)',
            '매출채권 대 상,제품비율(IFRS연결)': '매출채권 대 상,제품비율(IFRS)',
            '매입채무 대 재고자산비율(IFRS연결)': '매입채무 대 재고자산비율(IFRS)',
            '유동부채비율(IFRS연결)': '유동부채비율(IFRS)',
            '단기차입금 대 총차입금비율(IFRS연결)': '단기차입금 대 총차입금비율(IFRS)',
            '비유동부채비율(IFRS연결)': '비유동부채비율(IFRS)',
            '비유동부채 대 순운전자본비율(IFRS연결)': '비유동부채 대 순운전자본비율(IFRS)',
            '순운전자본비율(IFRS연결)': '순운전자본비율(IFRS)',
            '차입금의존도(IFRS연결)': '차입금의존도(IFRS)',
            '차입금비율(IFRS연결)': '차입금비율(IFRS)',
            '이자보상배율(순금융비용)(IFRS연결)': '이자보상배율(순금융비용)(IFRS)',
            '유보액대비율(IFRS연결)': '유보액대비율(IFRS)',
            '유보액 대 납입자본배율(IFRS연결)': '유보액 대 납입자본배율(IFRS)',
            '유동자산집중도(IFRS연결)': '유동자산집중도(IFRS)',
            '비유동자산집중도(IFRS연결)': '비유동자산집중도(IFRS)',
            '투자집중도(IFRS연결)': '투자집중도(IFRS)',
            'CASH FLOW 대 부채비율(IFRS연결)': 'CASH FLOW 대 부채비율(IFRS)',
            'CASH FLOW 대 차입금비율(IFRS연결)': 'CASH FLOW 대 차입금비율(IFRS)',
            'CASH FLOW 대 총자본비율(IFRS연결)': 'CASH FLOW 대 총자본비율(IFRS)',
            'CASH FLOW 대 매출액비율(IFRS연결)': 'CASH FLOW 대 매출액비율(IFRS)',
            '정상영업이익대비이자보상배율(IFRS연결)': '정상영업이익대비이자보상배율(IFRS)',
            
            # 회전율 지표들
            '경영자본회전률(IFRS연결)': '경영자본회전률(IFRS)',
            '자기자본회전률(IFRS연결)': '자기자본회전률(IFRS)',
            '자본금회전률(IFRS연결)': '자본금회전률(IFRS)',
            '타인자본회전률(IFRS연결)': '타인자본회전률(IFRS)',
            '매입채무회전률(IFRS연결)': '매입채무회전률(IFRS)',
            '매입채무회전기간(IFRS연결)': '매입채무회전기간(IFRS)',
            '유동자산회전률(IFRS연결)': '유동자산회전률(IFRS)',
            '당좌자산회전률(IFRS연결)': '당좌자산회전률(IFRS)',
            '재고자산회전률(IFRS연결)': '재고자산회전률(IFRS)',
            '재고자산회전기간(IFRS연결)': '재고자산회전기간(IFRS)',
            '상품,제품회전률(IFRS연결)': '상품,제품회전률(IFRS)',
            '원,부재료회전률(IFRS연결)': '원,부재료회전률(IFRS)',
            '재공품회전률(IFRS연결)': '재공품회전률(IFRS)',
            '매출채권회전률(IFRS연결)': '매출채권회전률(IFRS)',
            '매출채권회전기간(IFRS연결)': '매출채권회전기간(IFRS)',
            '비유동자산회전률(IFRS연결)': '비유동자산회전률(IFRS)',
            '유형자산회전율(IFRS연결)': '유형자산회전율(IFRS)',
            '순운전자본회전률(IFRS연결)': '순운전자본회전률(IFRS)',
            '운전자본회전률(IFRS연결)': '운전자본회전률(IFRS)',
            '1회전기간(IFRS연결)': '1회전기간(IFRS)',
            
            # 부가가치 및 생산성 지표들
            '부가가치(IFRS연결)(백만원)': '부가가치(IFRS)(백만원)',
            '종업원1인당 부가가치(IFRS연결)(백만원)': '종업원1인당 부가가치(IFRS)(백만원)',
            '종업원1인당 매출액(IFRS연결)(백만원)': '종업원1인당 매출액(IFRS)(백만원)',
            '종업원1인당 정상영업이익(IFRS연결)(백만원)': '종업원1인당 정상영업이익(IFRS)(백만원)',
            '종업원1인당 순이익(IFRS연결)(백만원)': '종업원1인당 순이익(IFRS)(백만원)',
            '종업원1인당 인건비(IFRS연결)(백만원)': '종업원1인당 인건비(IFRS)(백만원)',
            '노동장비율(IFRS연결)': '노동장비율(IFRS)',
            '기계장비율(IFRS연결)': '기계장비율(IFRS)',
            '자본집약도(IFRS연결)': '자본집약도(IFRS)',
            '총자본투자효율(IFRS연결)': '총자본투자효율(IFRS)',
            '설비투자효율(IFRS연결)': '설비투자효율(IFRS)',
            '기계투자효율(IFRS연결)': '기계투자효율(IFRS)',
            '부가가치율(IFRS연결)': '부가가치율(IFRS)',
            '노동소득분배율(IFRS연결)': '노동소득분배율(IFRS)',
            '자본분배율(IFRS연결)': '자본분배율(IFRS)',
            '이윤분배율(IFRS연결)': '이윤분배율(IFRS)',
            
            # 추가 손익 및 비용 지표들
            '법인세비용차감전(계속사업)손익(IFRS연결)(백만원)': '법인세비용차감전(계속사업)손익(IFRS)(백만원)',
            '인건비(IFRS연결)(백만원)': '인건비(IFRS)(백만원)',
            '금융비용(IFRS연결)(백만원)': '금융비용(IFRS)(백만원)',
            '임차료(IFRS연결)(백만원)': '임차료(IFRS)(백만원)',
            '세금과공과(IFRS연결)(백만원)': '세금과공과(IFRS)(백만원)',
            '감가상각비(IFRS연결)(백만원)': '감가상각비(IFRS)(백만원)',
            '종업원수(IFRS연결)': '종업원수(IFRS)',
            
            # 개별재무제표에만 있는 컬럼들 (매칭 불가)
            # '기업가치(EV)(IFRS)(백만원)': None,  # 연결재무제표에 없음
            # 'EBITDA(IFRS)(백만원)': None,  # 연결재무제표에 없음
            # 'EV/EBITDA(IFRS)(배)': None,  # 연결재무제표에 없음
            # 'PER(최고)(IFRS)': None,  # 연결재무제표에 없음
            # 'PER(최저)(IFRS)': None,  # 연결재무제표에 없음
            # 'PBR(최고)(IFRS)': None,  # 연결재무제표에 없음
            # 'PBR(최저)(IFRS)': None,  # 연결재무제표에 없음
            # 'PCR(최고)(IFRS)': None,  # 연결재무제표에 없음
            # 'PCR(최저)(IFRS)': None,  # 연결재무제표에 없음
            # 'PSR(최고)(IFRS)': None,  # 연결재무제표에 없음
            # 'PSR(최저)(IFRS)': None,  # 연결재무제표에 없음
            # 'EBITDA/매출액(IFRS)(%)': None,  # 연결재무제표에 없음
            # 'EBITDA/금융비용(IFRS)(배)': None,  # 연결재무제표에 없음
            # 'EBITDA/평균발행주식수(IFRS)(백만원)': None,  # 연결재무제표에 없음
        }
        
        self.logger.info("FinancialDataMerger 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('FinancialDataMerger')
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
    
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 로드"""
        self.logger.info("데이터 로드 시작")
        
        # 첫 번째 파일 (메인 데이터)
        primary_path = self.project_root / self.primary_file
        if not primary_path.exists():
            raise FileNotFoundError(f"첫 번째 파일을 찾을 수 없습니다: {primary_path}")
        
        df1 = pd.read_csv(primary_path, dtype={'거래소코드': str})
        self.logger.info(f"첫 번째 파일 로드: {df1.shape}")
        
        # 두 번째 파일 (보완 데이터)
        secondary_path = self.project_root / self.secondary_file
        if not secondary_path.exists():
            raise FileNotFoundError(f"두 번째 파일을 찾을 수 없습니다: {secondary_path}")
        
        df2 = pd.read_csv(secondary_path, dtype={'거래소코드': str})
        self.logger.info(f"두 번째 파일 로드: {df2.shape}")
        
        return df1, df2
    
    def filter_december_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """12월 회계년도만 유지"""
        self.logger.info("회계년도 필터링 (12월만 유지)")
        
        original_count = len(df)
        
        # 회계년도 컬럼에서 월 추출
        if '회계년도' in df.columns:
            # YYYY/mm 형태에서 mm 추출
            df['month'] = df['회계년도'].astype(str).str.split('/').str[1]
            
            # 12월만 유지
            df_filtered = df[df['month'] == '12'].copy()
            df_filtered = df_filtered.drop('month', axis=1)
            
            # 회계년도를 YYYY 형태로 변환
            df_filtered['회계년도'] = df_filtered['회계년도'].astype(str).str.split('/').str[0]
            
            filtered_count = len(df_filtered)
            removed_count = original_count - filtered_count
            
            self.logger.info(f"필터링 완료: {original_count:,}개 → {filtered_count:,}개 ({removed_count:,}개 제거)")
            
            return df_filtered
        else:
            self.logger.warning("회계년도 컬럼을 찾을 수 없습니다.")
            return df
    
    
    def merge_and_fill_data(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """데이터 병합 및 결측치 채우기 (수동 매핑만 사용)"""
        self.logger.info("데이터 병합 및 결측치 채우기 시작")
        
        # 키 컬럼 확인
        key_cols = ['거래소코드', '회계년도']
        for col in key_cols:
            if col not in df1.columns or col not in df2.columns:
                raise ValueError(f"키 컬럼 '{col}'이 두 데이터프레임 중 하나에 없습니다.")
        
        # 수동 매핑만 사용
        final_mapping = self.column_mapping
        
        self.logger.info(f"수동 컬럼 매칭: {len(final_mapping)}개")
        
        # 매칭되지 않은 컬럼들 출력
        unmatched_cols = []
        for col in df1.columns:
            if col not in key_cols and col not in final_mapping:
                unmatched_cols.append(col)
        
        if unmatched_cols:
            self.logger.warning(f"매칭되지 않은 컬럼들: {unmatched_cols}")
            self.logger.warning("필요시 column_mapping 딕셔너리에 수동으로 추가하세요.")
        
        # 결과 데이터프레임 초기화 (df1 기반)
        result_df = df1.copy()
        original_missing = result_df.isnull().sum().sum()
        
        # 두 번째 데이터프레임에만 있는 컬럼들 추가
        unique_cols_df2 = []
        for col in df2.columns:
            if col not in key_cols and col not in final_mapping.values():
                unique_cols_df2.append(col)
                result_df[col] = np.nan
        
        if unique_cols_df2:
            self.logger.info(f"두 번째 파일의 고유 컬럼 추가: {unique_cols_df2}")
        
        # 병합을 통한 결측치 채우기
        fill_count = 0
        total_rows = len(result_df)
        
        self.logger.info(f"총 {total_rows:,}개 행 처리 시작...")
        
        for i, (idx, row) in enumerate(result_df.iterrows()):
            # 진행률 출력 (1000개 행마다)
            if (i + 1) % 1000 == 0 or i == 0:
                progress = (i + 1) / total_rows * 100
                self.logger.info(f"진행률: {i+1:,}/{total_rows:,} ({progress:.1f}%)")
            
            # 매칭되는 행 찾기
            matching_rows = df2[
                (df2['거래소코드'] == row['거래소코드']) & 
                (df2['회계년도'] == row['회계년도'])
            ]
            
            if not matching_rows.empty:
                match_row = matching_rows.iloc[0]
                
                # 매핑된 컬럼들의 결측치 채우기
                for col1, col2 in final_mapping.items():
                    if col1 in result_df.columns and col2 in match_row.index:
                        if pd.isna(result_df.loc[idx, col1]) and not pd.isna(match_row[col2]):
                            result_df.loc[idx, col1] = match_row[col2]
                            fill_count += 1
                
                # 두 번째 파일의 고유 컬럼들 채우기
                for col in unique_cols_df2:
                    if col in match_row.index and not pd.isna(match_row[col]):
                        result_df.loc[idx, col] = match_row[col]
                        fill_count += 1
        
        final_missing = result_df.isnull().sum().sum()
        filled_values = original_missing - final_missing
        
        self.logger.info(f"결측치 채우기 완료:")
        self.logger.info(f"  - 원본 결측치: {original_missing:,}개")
        self.logger.info(f"  - 채워진 값: {filled_values:,}개")
        self.logger.info(f"  - 남은 결측치: {final_missing:,}개")
        self.logger.info(f"  - 개선율: {filled_values/original_missing*100:.1f}%" if original_missing > 0 else "  - 개선율: 0%")
        
        return result_df
    
    def save_result(self, df: pd.DataFrame):
        """결과 저장"""
        output_dir = self.project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "FS_temp.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"결과 저장 완료: {output_path}")
        self.logger.info(f"최종 데이터 크기: {df.shape}")
        
        return output_path
    
    def run_pipeline(self) -> str:
        """전체 파이프라인 실행"""
        self.logger.info("=== 재무데이터 병합 파이프라인 시작 ===")
        
        try:
            # 1. 데이터 로드
            df1, df2 = self.load_data()
            
            # 2. 12월 데이터만 필터링
            df1_filtered = self.filter_december_only(df1)
            df2_filtered = self.filter_december_only(df2)
            
            # 3. 데이터 병합 및 결측치 채우기
            result_df = self.merge_and_fill_data(df1_filtered, df2_filtered)
            
            # 4. 결과 저장
            output_path = self.save_result(result_df)
            
            self.logger.info("=== 재무데이터 병합 파이프라인 완료 ===")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 오류 발생: {str(e)}")
            raise


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='재무데이터 병합 및 결측치 채우기')
    parser.add_argument(
        '--primary', '-p',
        type=str,
        default='data/raw/연결.csv',  # 실제 파일명으로 변경하세요
        help='첫 번째 CSV 파일 경로 (메인 데이터)'
    )
    parser.add_argument(
        '--secondary', '-s',
        type=str,
        default='data/raw/개별.csv',  # 실제 파일명으로 변경하세요
        help='두 번째 CSV 파일 경로 (보완 데이터)'
    )
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    merger = FinancialDataMerger(args.primary, args.secondary)
    output_path = merger.run_pipeline()
    
    print(f"\n✅ 파이프라인 완료!")
    print(f"📁 결과 파일: {output_path}")
    
    # 컬럼 매칭 확인 메시지
    print(f"\n💡 참고사항:")
    print(f"   - 컬럼 매칭이 부정확하다면 코드 내 column_mapping 딕셔너리를 수정하세요")
    print(f"   - 매칭되지 않은 컬럼들은 로그에서 확인할 수 있습니다")


if __name__ == "__main__":
    main()