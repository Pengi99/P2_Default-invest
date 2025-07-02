"""
부실 라벨링
==========

기능:
1. FS_standardized.csv와 value_fail.csv를 매칭하여 부실 라벨링
2. 폐지일자 기준으로 t-1년에 부실 라벨 부여
3. 부실 기업의 다른 년도 데이터 제거
4. 결과를 data/processed/FS2.csv로 저장

사용법:
    python default_labeler.py
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime

class DefaultLabeler:
    """부실 라벨링 클래스"""
    
    def __init__(self, 
                 fs_file: str = "data/processed/FS2_features.csv",
                 value_fail_file: str = "data/raw/value_fail.csv"):
        """
        초기화
        
        Args:
            fs_file: FS2_features.csv 파일 경로
            value_fail_file: value_fail.csv 파일 경로
        """
        self.fs_file = fs_file
        self.value_fail_file = value_fail_file
        self.project_root = Path(__file__).parent.parent.parent
        
        # 로깅 설정
        self.logger = self._setup_logging()
        
        self.logger.info("DefaultLabeler 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('DefaultLabeler')
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
        
        # FS_standardized.csv 로드
        fs_path = self.project_root / self.fs_file
        if not fs_path.exists():
            raise FileNotFoundError(f"FS_standardized.csv 파일을 찾을 수 없습니다: {fs_path}")
        
        fs_df = pd.read_csv(fs_path, dtype={'거래소코드': str})
        self.logger.info(f"FS_standardized.csv 로드: {fs_df.shape}")
        
        # value_fail.csv 로드
        value_fail_path = self.project_root / self.value_fail_file
        if not value_fail_path.exists():
            raise FileNotFoundError(f"value_fail.csv 파일을 찾을 수 없습니다: {value_fail_path}")
        
        fail_df = pd.read_csv(value_fail_path, dtype={'종목코드': str})
        self.logger.info(f"value_fail.csv 로드: {fail_df.shape}")
        
        return fs_df, fail_df
    
    def create_default_labels(self, fs_df: pd.DataFrame, fail_df: pd.DataFrame) -> pd.DataFrame:
        """부실 라벨 생성"""
        self.logger.info("부실 라벨 생성 시작")
        
        # 기본적으로 모든 기업은 정상(0)
        result_df = fs_df.copy()
        result_df['default'] = 0
        
        # 폐지일자에서 연도 추출
        fail_df['폐지년도'] = pd.to_datetime(fail_df['폐지일자'], errors='coerce').dt.year
        
        # 부실 기업들만 먼저 처리
        fail_companies = set()
        labeled_rows = []
        
        for _, fail_row in fail_df.iterrows():
            company_code = fail_row['종목코드']
            delisting_year = fail_row['폐지년도']
            
            if pd.isna(delisting_year):
                self.logger.warning(f"기업 {company_code}의 폐지일자가 유효하지 않습니다.")
                continue
            
            delisting_year = int(delisting_year)
            
            # 해당 기업의 모든 데이터 가져오기
            company_data = result_df[result_df['거래소코드'] == company_code].copy()
            
            if company_data.empty:
                self.logger.warning(f"기업 {company_code}의 재무데이터가 없습니다 (폐지: {delisting_year}년)")
                continue
            
            # t-1, t-2, t-3년 순서로 데이터 찾기
            target_years = [delisting_year - i for i in range(1, 4)]
            labeled = False
            
            for target_year in target_years:
                # 정수 타입으로 비교 (수정된 부분)
                target_data = company_data[company_data['연도'] == target_year]
                
                if not target_data.empty:
                    # 해당 년도에 부실 라벨 부여
                    target_idx = target_data.index[0]
                    result_df.loc[target_idx, 'default'] = 1
                    labeled_rows.append(target_idx)
                    labeled = True
                    
                    self.logger.info(f"부실 라벨 부여: 기업 {company_code}, {target_year}년 (폐지: {delisting_year}년)")
                    break
            
            if labeled:
                fail_companies.add(company_code)
            else:
                self.logger.warning(f"기업 {company_code}의 부실 라벨을 부여할 데이터가 없습니다 (폐지: {delisting_year}년)")
        
        # 정상 기업들의 모든 데이터 추가
        normal_companies = set(result_df['거래소코드'].unique()) - fail_companies
        normal_rows = []
        
        for company_code in normal_companies:
            company_data = result_df[result_df['거래소코드'] == company_code]
            normal_rows.extend(company_data.index.tolist())
        
        # 최종 데이터: 부실 기업의 라벨된 행 + 정상 기업의 모든 행
        final_rows = labeled_rows + normal_rows
        final_df = result_df.loc[final_rows].copy().reset_index(drop=True)
        
        # 결과 통계
        total_companies = final_df['거래소코드'].nunique()
        default_companies = final_df[final_df['default'] == 1]['거래소코드'].nunique()
        normal_companies = total_companies - default_companies
        
        total_records = len(final_df)
        default_records = len(final_df[final_df['default'] == 1])
        normal_records = total_records - default_records
        
        self.logger.info(f"라벨링 완료:")
        self.logger.info(f"  - 총 기업 수: {total_companies:,}개")
        self.logger.info(f"  - 부실 기업: {default_companies:,}개 ({default_companies/total_companies*100:.1f}%)")
        self.logger.info(f"  - 정상 기업: {normal_companies:,}개 ({normal_companies/total_companies*100:.1f}%)")
        self.logger.info(f"  - 총 레코드: {total_records:,}개")
        self.logger.info(f"  - 부실 레코드: {default_records:,}개 ({default_records/total_records*100:.1f}%)")
        self.logger.info(f"  - 정상 레코드: {normal_records:,}개 ({normal_records/total_records*100:.1f}%)")
        
        return final_df
    
    def save_result(self, df: pd.DataFrame):
        """결과 저장"""
        output_dir = self.project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "FS2_default.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"결과 저장 완료: {output_path}")
        self.logger.info(f"최종 데이터 크기: {df.shape}")
        
        # 컬럼 정보 출력
        self.logger.info(f"최종 컬럼 목록: {df.columns.tolist()}")
        
        return output_path
    
    def run_pipeline(self) -> str:
        """전체 파이프라인 실행"""
        self.logger.info("=== 부실 라벨링 파이프라인 시작 ===")
        
        try:
            # 1. 데이터 로드
            fs_df, fail_df = self.load_data()
            
            # 2. 부실 라벨 생성
            labeled_df = self.create_default_labels(fs_df, fail_df)
            
            # 3. 결과 저장
            output_path = self.save_result(labeled_df)
            
            self.logger.info("=== 부실 라벨링 파이프라인 완료 ===")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 오류 발생: {str(e)}")
            raise


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='부실 라벨링')
    parser.add_argument(
        '--fs-file', '-f',
        type=str,
        default='data/processed/FS2_features.csv',
        help='FS2_features.csv 파일 경로'
    )
    parser.add_argument(
        '--value-fail', '-v',
        type=str,
        default='data/raw/value_fail.csv',
        help='value_fail.csv 파일 경로'
    )
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    labeler = DefaultLabeler(args.fs_file, args.value_fail)
    output_path = labeler.run_pipeline()
    
    print(f"\n✅ 부실 라벨링 완료!")
    print(f"📁 결과 파일: {output_path}")
    
    # 추가 정보
    print(f"\n💡 참고사항:")
    print(f"   - 부실 라벨링 결과를 로그에서 확인하세요")
    print(f"   - 폐지일자가 없는 기업들은 자동으로 제외됩니다")


if __name__ == "__main__":
    main()