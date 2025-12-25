import time
from math_parsing import parse_math
from prompt import agent_prompt
from openai import OpenAI


def query_model(client, agent_context, model_name="gemini-2.0-flash", max_retries=5):
    """
    查询OpenAI模型，带有重试机制

    参数:
    client: OpenAI客户端实例
    agent_context: 对话上下文
    model_name: 模型名称
    max_retries: 最大重试次数

    返回:
    模型生成的文本内容
    """
    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=agent_context,
                n=1
            )
            content = completion.choices[0].message.content
            return content
        except Exception as e:
            retries += 1
            print(f"⚠️ 请求失败 (尝试 {retries}/{max_retries}): {str(e)}")

            # 如果是速率限制错误，等待更长时间
            if "rate limit" in str(e).lower():
                wait_time = min(30, 2 ** retries)  # 指数退避，最多等待30秒
                print(f"⏳ 速率限制错误，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                wait_time = min(10, retries * 2)  # 线性增加等待时间
                print(f"⏳ 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

    # 所有重试都失败后抛出异常
    raise Exception(f"❌ 在 {max_retries} 次重试后仍然无法完成请求")


def parse_question_answer(dataset_name, sample):
    # parse_question_answer: 这个函数的主要作用是根据不同的数据集类型，从样本数据中提取问题、答案和原始任务数据，并格式化问题文本以便输入给语言模型。
    if dataset_name == "mmlu":
        question_raw = sample[0]  # 原始问题
        a = sample[1]  # 选项A
        b = sample[2]  # 选项B
        c = sample[3]  # 选项C
        d = sample[4]  # 选项D
        answer = sample[5]  # 正确答案索引
        if type(sample) == list:
            raw_task = tuple(sample)
        else:
            raw_task = tuple(sample.values())
        question = agent_prompt[dataset_name]['question'].format(question_raw, a, b, c, d)
        return question, answer, raw_task

    elif dataset_name == "math":
        question_raw = sample["problem"]  # 数学问题
        answer = parse_math(sample["solution"])  # 解析数学答案
        question = agent_prompt[dataset_name]['question'].format(question_raw)
        raw_task = sample
        return question, answer, raw_task

    elif dataset_name == "chess":
        question_raw = sample["input"]
        last_move = sample['input'].split(' ')[-1]
        question = agent_prompt[dataset_name]['question'].format(question_raw, last_move)
        answer = sample["target"]
        raw_task = sample
        return question, answer, raw_task

    elif dataset_name == "mquake":
        question_raw = sample['questions'][0]
        answer = [sample['answer']] + sample['answer_alias']
        raw_task = sample
        question = agent_prompt[dataset_name]['question'].format(question_raw, question_raw)
        return question, answer, raw_task

    elif dataset_name == "musique":
        question_raw = sample['question']
        answer = [sample['answer']] + sample['answer_aliases']
        raw_task = sample
        question = agent_prompt[dataset_name]['question'].format(question_raw, question_raw)
        return question, answer, raw_task

    elif dataset_name == "truthfulqa":
        question_raw = sample['question']
        answers_raw = sample['mc1_targets']
        answers = [(chr(97 + i), answer) for i, answer in enumerate(answers_raw)]
        answer = [(chr(97 + i), answer) for i, answer in enumerate(answers_raw) if answers_raw[answer] == 1]
        raw_task = sample
        answers_txt = ', '.join([f"({letter.upper()}) {answer}" for letter, answer in answers])
        question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
        return question, answer, raw_task

    elif dataset_name == "medmcqa":
        question_raw = sample['question']
        answers_letters = ['a', 'b', 'c', 'd']
        answers = [sample['opa'], sample['opb'], sample['opc'], sample['opd']]
        answer = answers_letters[sample['cop'] - 1]
        raw_task = sample
        answers_txt = ', '.join([f"({letter.upper()}) {answer}" for letter, answer in zip(answers_letters, answers)])
        question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
        return question, answer, raw_task

    elif dataset_name == 'scalr':

        question_raw = sample['question']
        answers_letters = ['a', 'b', 'c', 'd', 'e']
        answers = [sample['choice_0'], sample['choice_1'], sample['choice_2'], sample['choice_3'], sample['choice_4']]
        answer = answers_letters[sample['answer']]
        raw_task = sample
        answers_txt = ', '.join([f"({letter.upper()}) {answer}" for letter, answer in zip(answers_letters, answers)])
        question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
        return question, answer, raw_task

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")