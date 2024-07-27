import json


def transform_data(data):
    transformed_data = []
    for problem in data:
        for question in problem["questions"]:
            options_with_letters = []
            for i, option in enumerate(question['options']):
                # 添加选项标签
                letter = chr(65 + i)  # 65 is ASCII for 'A'
                options_with_letters.append(f"{letter}: {option}")

            input_str = f"{question['question']} {'; '.join(options_with_letters)}"
            transformed_item = {
                "instruction": problem["problem"],
                "input": input_str,
                # "output": question["answer"]  #trian
                "output": "",
            }
            transformed_data.append(transformed_item)
    return transformed_data


def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]
    return data


def write_jsonl_file(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')


if __name__ == "__main__":
    # 指定输入和输出文件路径
    input_file_path = 'round1_test_data.jsonl'
    output_file_path = 'output_test.jsonl'

    # 读取原始数据
    original_data = read_jsonl_file(input_file_path)

    # 转换数据
    transformed_data = transform_data(original_data)

    # 写入新文件
    write_jsonl_file(transformed_data, output_file_path)

    print(f"Data has been successfully written to {output_file_path}")