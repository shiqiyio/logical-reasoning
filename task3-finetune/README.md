# Task3 微调+vllm加速+多路投票 策略
第二届世界科学智能大赛逻辑推理赛道：复杂推理能力评估

## 1 大语言模型微调

大模型微调（Fine-tuning）是一种技术，通过在预训练的大型语言模型上使用特定数据集进行进一步训练，使模型能够更好地适应特定任务或领域。

其核心原理在于，机器学习模型只能代表其训练数据的逻辑和理解。对于未见过的数据样本，模型可能无法准确识别或理解。对于大型模型而言，它们虽然能够处理广泛的语言信息并进行流畅的对话，但在特定场景下可能无法提供准确的答案。

将官网`test.jsonl`文件进行处理！总计1328道题。

处理完的示例：

```
{"instruction": "有一组数字，分别为：2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16。请根据以下条件回答问题：", "input": "选择题 1：**\n**最大的数字是什么？ A: 10; B: 14; C: 16; D: 12", "output": ""}
{"instruction": "有一组数字，分别为：2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16。请根据以下条件回答问题：", "input": "选择题 2：**\n**次大的数字是什么？ A: 10; B: 14; C: 16; D: 12", "output": ""}
{"instruction": "有一组数字，分别为：2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16。请根据以下条件回答问题：", "input": "选择题 3：**\n**数字10是最大的数字吗？ A: 是; B: 否", "output": ""}
{"instruction": "有一组数字，分别为：2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16。请根据以下条件回答问题：", "input": "选择题 4：**\n**数字6是最大的数字吗？ A: 是; B: 否", "output": ""}```
```


并进行prompt的构建：

```prompt = f"""### Context:
你是一位逻辑推理专家，擅长解决各种逻辑推理问题。

### Observation:
以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都基于闭世界假设，即未观测事实都为假。

### Skills:
你需要运用你的逻辑推理能力，仔细分析并得出正确答案。

### Task:
请逐步分析问题并得出结论。

### Action:
根据你的分析，只返回最终的答案。确保回复的格式是 "A" "B" "C" "D" 四个选项中的一个。不要提供任何解释或多余信息。如果无法推理出结果或问题模糊不清，请直接返回 "A"。

### Result:
最终答案应该只包括一个字母选项，确保回答简洁明了。

### 题目:
题目如下：有一个列表，找出该列表的最后一个元素。

### 问题:
{question}
```

## 2 LoRA微调

### LoRA介绍

LoRA（Low-Rank Adaptation）微调是一种高效的模型微调技术，特别适用于大型预训练语言模型的适应性调整。LoRA的核心思想是通过引入低秩矩阵来调整模型的权重，从而在不显著增加模型参数数量的情况下，实现对模型的微调。

#### LoRA 的优势

\- 可以针对不同的下游任务构建小型 LoRA 模块，从而在共享预训练模型参数基础上有效地切换下游任务。

\- LoRA 使用自适应优化器（Adaptive Optimizer），不需要计算梯度或维护大多数参数的优化器状态，训练更有效、硬件门槛更低。

\- LoRA 使用简单的线性设计，在部署时将可训练矩阵与冻结权重合并，不存在推理延迟。

\- LoRA 与其他方法正交，可以组合。

#### LoRA 的原理

![img](https://datawhaler.feishu.cn/space/api/box/stream/download/asynccode/?code=NmM3NjcxOTkwMTU0Y2E4YzNmNTAzMjJjZGE4ZjQwOTVfaHFjOGVmcG9tTjJ5Y052RVBLNkpOODc3aWtDdmQwZ0RfVG9rZW46S2VBb2Jqd21Zb3g0NXZ4WW0xQmNUNjgybmZiXzE3MjI2OTU1NDE6MTcyMjY5OTE0MV9WNA)



## 使用本地qwen2_7B_instruct进行lora微调



## 3 vllm加速

所谓的“多路召回策略”就是指采用不同的策略、特征或者简单模型，分别召回一部分候选集，然后再把这些候选集混合在一起后供后续排序模型使用的策略。

`python -m vllm.entrypoints.openai.api_server --model ./merge  --served-model-name Qwen2-7B-Instruct-lora --max-model-len=4096 --enforce-eager` 启动服务，方便后续模型的调用。

模型api：

```
def call_qwen_api(MODEL_NAME, prompt):


  \# 这里采用dashscope的api调用模型推理，通过http传输的json封装返回结果

  openai_api_key = "EMPTY"

  openai_api_base = "http://localhost:8000/v1"

  client = OpenAI(

​    api_key=openai_api_key,

​    base_url=openai_api_base,

  )

  completion = client.chat.completions.create(

   model=MODEL_NAME,

   messages=[

​        \# {'role':'system','content':'你是一个解决推理任务的专家，你需要分析出问题中的每个实体以及响应关系。然后根据问题一步步推理出结果。并且给出正确的结论。'},



​    {"role": "user", "content": prompt}

   ]

  )

  return completion.choices[0].message.content
```

## 4 多路投票

在三路投票的基础上，增加至5路，当然推理时间有所延长。

```
def get_answer(prompts,MODEL_NAME):
    answer_list = []

    # 送入多线程任务
​    for prompt in tqdm(prompts, desc="Submitting tasks", total=len(prompts)):
        # 统一使用llm 三次调用
        # res,res1,res2 = call_qwen_api(MODEL_NAME, prompt),call_qwen_api(MODEL_NAME, prompt),call_qwen_api(MODEL_NAME, prompt)
        # # 通过投票函数获取最终结果并返回
        # ans = most_frequent_char(res,res1,res2)
​        res,res1,res2,res3,res4 = call_qwen_api(MODEL_NAME, prompt),call_qwen_api(MODEL_NAME, prompt),call_qwen_api(MODEL_NAME, prompt),call_qwen_api(MODEL_NAME, prompt),call_qwen_api(MODEL_NAME, prompt)
        # 通过投票函数获取最终结果并返回
​        ans = most_frequent_char(res,res1,res2,res3,res4)
​       
​        
​        
​        answer_list.append(ans)
​     
​    return answer_list
```

## 5 推理服务结果

通过自定义答案提取以及自动化脚本，完成文件的写入，最终提交。

```
提示词获取成功
Submitting tasks: 100%|██████████| 1328/1328 [3:15:47<00:00,  8.85s/it]  
1328
正在写入到./answer/answer.txt文件...
100%|██████████| 1328/1328 [00:00<00:00, 1769388.73it/s]
写入完成！！！
2024-07-29 20:01:15
执行结束！
```

得分截图![image-20240803224315634](./score.png)
