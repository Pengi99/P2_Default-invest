import json
import subprocess
import pandas as pd
import os

def extract_hyperparameters(file_path):
    # jq를 사용하여 model_performance 섹션 전체를 추출
    command = f"jq '.model_performance' {file_path}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error extracting data from {file_path}: {result.stderr}")
        return None
    
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return None
        
    extracted_data = []
    for model_key, model_data in data.items():
        # model_data가 딕셔너리이고 'best_hyperparameters' 키를 가지고 있는지 확인
        if isinstance(model_data, dict) and 'best_hyperparameters' in model_data:
            run_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            
            # 'ensemble' 모델은 키 구조가 다를 수 있으므로 분기 처리
            if "ensemble" in model_key:
                sampling_type = model_key.split('__')[0]
                model_name = 'ensemble'
            else:
                try:
                    sampling_type, model_name = model_key.split('__', 1)
                except ValueError:
                    print(f"Warning: Could not parse model key '{model_key}'. Skipping.")
                    continue

            params = model_data['best_hyperparameters']
            if not isinstance(params, dict):
                params = {'value': params}
            
            row = {
                'run_id': run_id,
                'sampling_type': sampling_type,
                'model_name': model_name,
                **params
            }
            extracted_data.append(row)
            
    return extracted_data

def main():
    files_to_process = [
        "/Users/jojongho/KDT/P2_Default-invest/outputs/modeling_runs/default_modeling_run_20250707_103859/results/modeling_results.json",
        "/Users/jojongho/KDT/P2_Default-invest/outputs/modeling_runs/default_modeling_run_20250708_004131/results/modeling_results.json"
    ]
    
    all_hyperparameters = []
    for file_path in files_to_process:
        hyperparams = extract_hyperparameters(file_path)
        if hyperparams:
            all_hyperparameters.extend(hyperparams)
            
    if not all_hyperparameters:
        print("No hyperparameters were extracted.")
        return

    # DataFrame 생성 및 CSV 저장
    df = pd.DataFrame(all_hyperparameters)
    
    # 컬럼 순서 재배치 (공통 컬럼을 앞으로)
    core_columns = ['run_id', 'sampling_type', 'model_name']
    param_columns = sorted([col for col in df.columns if col not in core_columns])
    df = df[core_columns + param_columns]
    
    output_path = "/Users/jojongho/KDT/P2_Default-invest/outputs/hyperparameter_summary.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Successfully saved hyperparameter summary to {output_path}")
    print("\nCSV Content Preview:")
    print(df.head().to_string())

if __name__ == "__main__":
    main() 