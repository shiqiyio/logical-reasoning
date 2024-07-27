import json

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def write_json_file(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 指定输入和输出文件路径
    input_file_path = 'output_test.jsonl'
    output_file_path = 'output_test.json'

    # 读取原始数据
    original_data = read_jsonl_file(input_file_path)

    # 写入新文件
    write_json_file(original_data, output_file_path)

    print(f"Data has been successfully written to {output_file_path}")