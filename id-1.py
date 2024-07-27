import json

def decrement_id_in_jsonl(input_file, output_file):
    # 打开输入文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # 处理每一行，并减少"id"字段的数字
    new_lines = []
    for line in lines:
        data = json.loads(line)
        old_id = data['id']
        # 提取数字部分并减一
        num = int(old_id.split('_')[-1]) - 1
        new_id = f"round1_test_data_{str(num).zfill(3)}"
        data['id'] = new_id
        new_lines.append(json.dumps(data, ensure_ascii=False))

    # 将处理后的内容写入输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for new_line in new_lines:
            outfile.write(new_line + '\n')

# 使用例子
input_file = 'submit.jsonl'  # 输入文件名
output_file = 'submit1.jsonl'  # 输出文件名
decrement_id_in_jsonl(input_file, output_file)
