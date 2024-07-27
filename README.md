# logical-reasoning
第二届世界科学智能大赛逻辑推理赛道：复杂推理能力评估

## 训练数据以及测试数据的处理
### 数据有一部分的":"为"："，得注意，导致推理过程中，无法通过“:”切分.

* ### 将jsonl文件通过[脚本1](https://github.com/shiqiyio/logical-reasoning/blob/main/handle_data.py) —> 转换为大模型训练的三段式结构，保存为jsonl格式
`{
    "instruction":"回答以下用户问题，仅输出答案。",
    "input":"1+1等于几?",
    "output":"2"
}`
* ### 切分器数据处理为json格式，则通过 [脚本2](https://github.com/shiqiyio/logical-reasoning/blob/main/jsonl2json.py)转换为json文件
* ### 模型进行推理时，通过[脚本3](https://github.com/shiqiyio/logical-reasoning/blob/main/prompt_gen.py)将测试集的三段式结构进行处理，为prompt，并输出列表，方便循环推理。

## 保存答案，将答案写入为固定格式，[脚本4](https://github.com/shiqiyio/logical-reasoning/blob/main/trans.py)
`{'id': 'round1_test_data_000',
 'questions': [{'answer': 'A'}, {'answer': 'D'}, ...], # 顺序与子问题对应
}`

* ### 最后提交发现id序号不对，从1开始的，故用[脚本5](https://github.com/shiqiyio/logical-reasoning/blob/main/id-1.py)



## lmdepoly报错（Python）asyncio使用异常：`This event loop is already running`解决方式
* 问题解决
`引入nest_asyncio模块
pip install nest_asyncio -i https://pypi.douban.com/simple`
* 添加一下代码即可
  `import nest_asyncio`
  `nest_asyncio.apply()`
