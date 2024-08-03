import json


def process_json(json_file_path, answer_list, output_file_path):
    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # 初始化输出数据结构
    output_data = []

    # 初始化 round_id
    round_id = -1
    current_instruction = ""

    # 处理每一项数据
    for item in data:
        # 如果 instruction 发生变化，增加 round_id
        if current_instruction != item['instruction']:
            round_id += 1
            current_instruction = item['instruction']
            current_item = {
                'id': f'round1_test_data_{round_id:03d}',
                'questions': []
            }
            output_data.append(current_item)
        else:
            current_item = output_data[-1]

        # 解析 input 字段中的选择题编号
        question_number = int(item['input'].split('选择题 ')[1].split(':')[0])

        # 获取对应答案
        answer = answer_list.pop(0)  # 使用并移除第一个答案

        # 添加问题及答案到 questions 列表
        current_item['questions'].append({'answer': answer})

    # 写入 JSONL 文件
    with open(output_file_path, 'w', encoding='utf-8') as jsonl_file:
        for item in output_data:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')




if __name__ == '__main__':
    # 指定输入文件路径
    input_file_path = './output_test.json'

    # 指定答案列表
    answers = ["A","A","B","C","D","A","C",'D','A']

    # 指定输出文件路径
    output_file_path = './output.jsonl'

    # 处理 JSON 文件并输出到 JSONL 文件
    process_json(input_file_path, answers, output_file_path)
#
# def extract_options(answers):
#     options = []
#     for item in answers:
#         if item in "ABCD":
#             options.append(item)
#     return options
#
# # 示例列表
# answers = ['A', 'B', 'C', '答案是:A']
#
# # 提取选项
# options = extract_options(answers)
# print(options)
