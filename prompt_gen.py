import json

def get_prompt(instruction, input):
    # 提取选项部分
    options = []
    for line in input.split('; '):
        if line.startswith('A:') or line.startswith('B:') or line.startswith('C:') or line.startswith('D:'):
            options.append(line.split(':', 1)[1])

    # 构建选项字符串
    options_str = '\n'.join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(options))

    # 构建prompt
    prompt = f"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为"A"。题目如下：

### 题目:
{instruction}

### 问题:
{input}
{options_str}
"""

    return prompt


def read_and_process_json_file(file_path):
    prompts = []

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        instruction = item["instruction"]
        input = item["input"]
        prompt = get_prompt(instruction, input)
        prompts.append(prompt)

    return prompts


# if __name__ == "__main__":
#     # 指定输入文件路径
#     input_file_path = './dataset/output_test.json'

#     # 读取并处理JSON文件
#     prompts = read_and_process_json_file(input_file_path)
#     print(prompts[:10])

#     # # 打印生成的prompts
#     # for i, prompt in enumerate(prompts, start=1):
#     #     print(f"Prompt {i}:{prompt}")
#     #     if i==3:
#     #         break
#     print(len(prompts))