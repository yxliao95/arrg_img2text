import os
import re
import json
from pathlib import Path

def list_first_level_dirs_scandir(path):
    """
    使用 os.scandir() 读取指定路径下的第一层子文件夹。
    
    Args:
        path (str): 要读取的目录路径。
    
    Returns:
        list: 包含所有第一层子文件夹名称的列表。
    """
    dir_names = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir():
                    dir_names.append(entry.name)
    except FileNotFoundError:
        print(f"Error: The path '{path}' was not found.")
    except NotADirectoryError:
        print(f"Error: The path '{path}' is not a directory.")
    
    return dir_names

def get_sort_key(folder_name):
    """
    内部辅助函数，用于提取排序所需的数字 N。
    """
    pattern = re.compile(r'8_3_text_(\d+)_(\S+)')
    match = pattern.match(folder_name)
    if match:
        # 如果匹配成功，返回提取到的数字 N 的整数形式
        return int(match.group(1))
    # 如果不符合命名规则，则返回一个极大的值，将其排在最后
    return float('inf')

def get_sort_key2(folder_name):
    """
    内部辅助函数，用于提取排序所需的数字 N。
    """
    pattern = re.compile(r'7_3_text__(\S+)_epoch(\S+)')
    match = pattern.match(folder_name)
    if match:
        # 如果匹配成功，返回提取到的数字 N 的整数形式
        return int(match.group(2))
    # 如果不符合命名规则，则返回一个极大的值，将其排在最后
    return float('inf')

def find_eval_log_files_first_level(path):
    """
    只在指定路径的第一层文件夹下查找以 'eval_finetuned.log' 结尾的文件。
    """
    p = Path(path)
    if not p.is_dir():
        print(f"错误: 路径 '{path}' 不存在或不是一个目录。")
        return []
    
    # glob 方法只在当前层级搜索
    log_files = list(p.glob('*_eval_finetuned.log'))
    return log_files


def extract_all_metrics_from_log(file_path):
    """
    从日志文件中提取所有 TextGen 的 JSON 格式的性能指标，并返回一个列表。

    Args:
        file_path (str): 日志文件的完整路径。

    Returns:
        list: 包含所有性能指标字典的列表。如果文件不存在或未找到任何匹配，返回空列表。
    """
    all_metrics = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在。")
        return []
    
    # 定义 JSON 数据的起始和结束标记
    start_marker = "[TextGen]: {"
    end_marker = "}"
    
    # 从日志内容的开头开始搜索
    current_search_index = 0
    
    while True:
        # 查找起始标记，从上一次找到的索引位置开始
        start_index = log_content.find(start_marker, current_search_index)
        
        if start_index == -1:
            # 如果找不到起始标记了，就结束循环
            break
            
        # 查找结束标记，确保它在起始标记的后面
        end_index = log_content.find(end_marker, start_index)
        
        if end_index == -1:
            # 如果找不到结束标记，说明日志文件格式有问题，直接跳出
            print(f"警告: 在索引 {start_index} 处找到起始标记，但未找到匹配的结束标记。")
            break

        # 提取 JSON 字符串
        json_string = log_content[start_index + len("[TextGen]: "): end_index + 1]
        
        try:
            # 解析 JSON 并添加到列表中
            metrics_dict = json.loads(json_string)
            all_metrics.append(metrics_dict)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}，跳过此段数据。")
        
        # 更新搜索起始索引，从本次找到的结束标记后开始
        current_search_index = end_index + 1
        
    return all_metrics

    
# 示例用法
target_path = "/scratch/c.c21051562/workspace/arrg_img2text/outputs_7_3_mix/results"
sub_directories = list_first_level_dirs_scandir(target_path)
# sorted_folders = sorted(sub_directories, key=get_sort_key)
sorted_folders = sorted(sub_directories)
results = {}

for d in sorted_folders:
    log_file_path = os.path.join(target_path,d)
    if "inj" not in log_file_path:
        continue
    print(log_file_path)
    found_files = find_eval_log_files_first_level(log_file_path)
    assert len(found_files) == 1
    metrics_list = extract_all_metrics_from_log(found_files[0])
    results[log_file_path] = {}
    for i, metrics_data in enumerate(metrics_list):
        keys_to_extract = ["BLEU", "ROUGEL", "radgraph_partial", "chexbert-all_micro avg_f1-score", "bertscore"]
        values = [metrics_data[key] for key in keys_to_extract]
        results[log_file_path][i] = ",".join(map(str, values))
        
print("test")
print("\n".join([result[1] for name, result in results.items()]))
print("validation")
print("\n".join([result[0] for name, result in results.items()]))
