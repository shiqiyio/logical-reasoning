# logical-reasoning
第二届世界科学智能大赛逻辑推理赛道：复杂推理能力评估

# 训练数据以及测试数据的处理
数据有一部分的":"为"："，得注意，导致推理过程中，无法通过“:”切分.

## 将jsonl文件通过脚本 —> 转换为大模型训练的三段式结构，保存为jsonl格式
## 切分器数据处理为json格式，则通过 [脚本]("https://github.com/shiqiyio/logical-reasoning/blob/main/jsonl2json.py")转换为json文件，
