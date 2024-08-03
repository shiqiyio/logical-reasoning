# 第二届世界科学智能大赛逻辑推理赛道：复杂推理能力评估 [Datawhale AI夏令营](https://linklearner.com/activity/12) 微调思路

## 赛事解读
### 比赛任务
本次比赛提供基于自然语言的逻辑推理问题，涉及多样的场景，包括关系预测、数值计算、谜题等，期待选手通过分析推理数据，利用机器学习、深度学习算法或者大语言模型，建立预测模型。

* 初赛
  任务：构建一个能够完成推理任务的选择模型
  运用机器学习模型或者深度学习模型解决推理问题。或者利用训练集数据对开源大语言模型进行微调。
* 复赛
任务：构建一个能够完成推理任务的选择模型

运用机器学习模型或者深度学习模型解决推理问题。或者利用训练集数据对开源大语言模型进行微调。

-----

## 模型微调  
本节我们简要介绍如何基于 transformers、peft 等框架，对 Qwen2-7B-Instruct 模型进行 Lora 微调。

### 模型下载  

使用 modelscope 中的 snapshot_download 函数下载模型，第一个参数为模型名称，参数 cache_dir 为模型的下载路径。
在 /root/model 路径下新建 model_download.py 文件并在其中输入以下内容，粘贴代码后请及时保存文件，如下图所示。并运行 `python /root/model/model_download.py` 执行下载，模型大小为 15GB，下载模型大概需要 5 分钟。
本文拟采用`qwen/Qwen2-7B-Instruct`
```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir='/root/model', revision='master')
```

### 环境配置

在完成基本环境配置和本地模型部署的情况下，你还需要安装一些第三方库，可以使用以下命令：

```bash
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.9.5
pip install "transformers>=4.39.0"
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.27
pip install transformers_stream_generator==0.0.4
pip install datasets==2.18.0
pip install peft==0.10.0
```
### 指令集构建

LLM 的微调一般指指令微调过程。所谓指令微调，是说我们使用的微调数据形如：

```json
{
    "instruction":"回答以下用户问题，仅输出答案。",
    "input":"1+1等于几?",
    "output":"2"
}
```

其中，`instruction` 是用户指令，告知模型其需要完成的任务；`input` 是用户输入，是完成用户指令所必须的输入内容；`output` 是模型应该给出的输出。

即我们的核心训练目标是让模型具有理解并遵循用户指令的能力。因此，在指令集构建时，我们应针对我们的目标任务，针对性构建任务指令集。例如，在本节我们使用比赛数据集作为示例，我们的目标是构建一个能够具有推理能力的个性化 LLM，因此我们构造的指令形如：

```json
{
  "instruction": "下面是一道关于计算阶乘的选择题：\n\n已知如下规则：\n\n1. 数字 0 的阶乘是 1。\n2. 数字 N 的阶乘等于 N 乘以 N-1 的阶乘。\n\n请根据此规则回答以下问题：",
  "input": "选择题 1：\n7 的阶乘是多少？ A: 720; B: 40320; C: 5040; D: 362880",
  "output": "C"
    }
```

我们所构造的全部指令数据集在目录`/dataset`下。数据集的相关处理及问题见[笔记](https://github.com/shiqiyio/logical-reasoning/blob/main/README.md)

### prompt提取

从数据集中提取大模型所需的prompt

```python
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
```
### 定义LoraConfig

`LoraConfig`这个类中可以设置很多参数，但主要的参数没多少，简单讲一讲，感兴趣的同学可以直接看源码。

- `task_type`：模型类型
- `target_modules`：需要训练的模型层的名字，主要就是`attention`部分的层，不同的模型对应的层的名字不同，可以传入数组，也可以字符串，也可以正则表达式。
- `r`：`lora`的秩，具体可以看`Lora`原理
- `lora_alpha`：`Lora alaph`，具体作用参见 `Lora` 原理 

`Lora`的缩放是啥嘞？当然不是`r`（秩），这个缩放就是`lora_alpha/r`, 在这个`LoraConfig`中缩放就是4倍。

```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
```

### 自定义 TrainingArguments 参数

`TrainingArguments`这个类的源码也介绍了每个参数的具体作用，当然大家可以来自行探索，这里就简单说几个常用的。

- `output_dir`：模型的输出路径
- `per_device_train_batch_size`：顾名思义 `batch_size`
- `gradient_accumulation_steps`: 梯度累加，如果你的显存比较小，那可以把 `batch_size` 设置小一点，梯度累加增大一些。
- `logging_steps`：多少步，输出一次`log`
- `num_train_epochs`：顾名思义 `epoch`
- `gradient_checkpointing`：梯度检查，这个一旦开启，模型就必须执行`model.enable_input_require_grads()`，这个原理大家可以自行探索，这里就不细说了。

```python
args = TrainingArguments(
    output_dir="./output/Qwen2_instruct_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)
```

### 使用 Trainer 训练

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
```

### 加载 lora 权重推理

训练好了之后可以使用如下方式加载`lora`权重进行推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = '/root/model/qwen/Qwen2-7B-Instruct/'
lora_path = 'lora_path'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

prompt = "题目如下：有一个列表，找出该列表的最后一个元素。\n\n下列选项中哪个是列表 `[a, b, c, d]` 的最后一个元素？A: a; B: b; C: c; D: d"
       
inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True).to('cuda')

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
```
----
## 遇到的问题，尝试用 lmdepoly报错（Python）asyncio使用异常：`This event loop is already running`解决方式

* 问题解决
引入nest_asyncio模块 <br/> 
`pip install nest_asyncio -i https://pypi.douban.com/simple`
* 添加一下代码即可 <br/> 
  `import nest_asyncio` <br/> 
  `nest_asyncio.apply()`
