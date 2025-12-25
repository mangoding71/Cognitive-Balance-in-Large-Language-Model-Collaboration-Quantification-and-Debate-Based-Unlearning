import os
import json
import argparse
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from math_parsing import parse_math
from math_equivalence import is_equiv
from commons import query_model
from calibrate_orientation import judge_prompt
from openai import OpenAI

# 直接设置评估文件路径
EVAL_ADDRESS = r"文件路径" 

def load_data(filename):
    """加载数据文件并输出内容"""
    print(f"\n{'=' * 80}")
    print(f"加载文件: {filename}")
    print(f"{'=' * 80}")

    # 从文件路径中提取数据集名称
    path_parts = filename.split(os.sep)
    dataset_name = None
    valid_datasets = ['truthfulqa', 'mmlu', 'chess', 'math', 'mquake', 'musique', 'medmcqa', 'scalr']

    # 首先尝试从路径中提取
    for part in path_parts:
        if part in valid_datasets:
            dataset_name = part
            break

    # 如果未找到，尝试从文件名中提取
    if dataset_name is None:
        base_name = os.path.basename(filename)
        parts = base_name.split('_')
        for part in parts:
            if part in valid_datasets:
                dataset_name = part
                break

    # 如果仍然未找到，使用默认值
    if dataset_name is None:
        dataset_name = "truthfulqa"
        print(f"⚠️ 警告: 无法从路径中提取数据集名称, 使用默认值: {dataset_name}")

    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            data.append(sample)
            print(f"样本 ID: {sample.get('id', '未知')}")
            print(f"问题: {sample.get('question', '未知')}")
            print(f"正确答案: {sample.get('correct_answer', '未知')}")
            print(f"Agent 响应数量: {len(sample.get('agent_responses', []))}")
            print("-" * 80)

    print(f"总共加载 {len(data)} 个样本")
    return data, dataset_name


def parse_mmlu(text, task_info):
    """解析 MMLU 数据集答案"""
    pattern = r"\((\w+)\)|(\w+)\)"
    matches = re.findall(pattern, text)
    matches = [match[0] or match[1] for match in matches]

    solution_by_re = None
    for match_str in matches[::-1]:
        solution_by_re = match_str.upper()
        if solution_by_re >= 'A' and solution_by_re <= 'D':
            break
        else:
            solution_by_re = None

    solution_by_item = [-1, -1, -1, -1]
    idx = 0
    for item in task_info[1:-1]:
        pos = text.lower().rfind(item.lower().strip())
        if pos >= 0:
            solution_by_item[idx] = pos
        idx += 1

    if max(solution_by_item) == -1:
        solution_by_item = None
    else:
        solution_by_item = ["A", "B", "C", "D"][solution_by_item.index(max(solution_by_item))]

    if solution_by_item is None and solution_by_re is not None:
        return solution_by_re
    elif solution_by_item is not None and solution_by_re is None:
        return solution_by_item
    elif solution_by_item is None and solution_by_re is None:
        return None
    elif solution_by_item is not None and solution_by_re is not None:
        if solution_by_item == solution_by_re:
            return solution_by_item
        else:
            return solution_by_item


def eval_parse_math(text, task_info):
    """解析数学答案"""
    return parse_math(text)


def parse_chess(text, task_info):
    """解析国际象棋答案"""
    none_responses = [
        "i am unable to provide a valid destination square based on the given chess game and moves",
        "none",
        "no valid",
        "no specific valid",
        "invalid",
        "n/a",
        "unable to provide",
        "game sequence contains errors",
        "i cannot provide"
    ]
    text = text.lower()
    pattern = r"[a-h][1-8]"
    pos = text.rfind("final answer")
    if pos != -1:
        item = text.split("final answer")[-1].strip()
        matches = re.findall(pattern, item)
        if len(matches) == 1:
            return matches[0].lower()
        elif len(matches) > 1:
            return matches[-1].lower()
        else:
            for valid_case in none_responses:
                if valid_case in text:
                    return None
            return None
    else:
        matches = re.findall(pattern, text)
        if len(matches) == 0:
            for valid_case in none_responses:
                if valid_case in text:
                    return None
            return None
        else:
            return matches[-1]


def parse_mquake(text, task_info):
    """解析 MQuAKE 答案"""
    try:
        answer = text.lower().split('answer:')[1].strip('., ')
    except:
        answer = ''
    return answer


def parse_truthfulqa(text, task_info):
    """解析 TruthfulQA 答案"""
    pattern = r"\((\w+)\)|(\w+)\)"
    matches = re.findall(pattern, text)
    matches = [match[0] or match[1] for match in matches]

    answers_raw = task_info['mc1_targets']
    answers = [(chr(97 + i).upper(), answer) for i, answer in enumerate(answers_raw)]

    solution_by_re = None
    for match_str in matches[::-1]:
        solution_by_re = match_str.upper()
        if solution_by_re >= 'A' and solution_by_re <= answers[-1][0]:
            break
        else:
            solution_by_re = None

    solution_by_item = [-1 for _ in range(len(answers))]
    idx = 0
    for item in [a[1] for a in answers]:
        pos = text.lower().rfind(item.lower().strip('., '))
        if pos >= 0:
            solution_by_item[idx] = pos
        idx += 1

    if max(solution_by_item) == -1:
        solution_by_item = None
    else:
        solution_by_item = [a[0] for a in answers][solution_by_item.index(max(solution_by_item))]

    if solution_by_item is None and solution_by_re is not None:
        return solution_by_re
    elif solution_by_item is not None and solution_by_re is None:
        return solution_by_item
    elif solution_by_item is None and solution_by_re is None:
        return None
    elif solution_by_item is not None and solution_by_re is not None:
        if solution_by_item == solution_by_re:
            return solution_by_item
        else:
            return solution_by_item


def parse_medmcqa(text, task_info):
    """解析 MedMCQA 答案"""
    pattern = r"\((\w+)\)|(\w+)\)"
    matches = re.findall(pattern, text)
    matches = [match[0] or match[1] for match in matches]

    answers_raw = [task_info['opa'], task_info['opb'], task_info['opc'], task_info['opd']]
    answers = [(chr(97 + i).upper(), answer) for i, answer in enumerate(answers_raw)]

    solution_by_re = None
    for match_str in matches[::-1]:
        solution_by_re = match_str.upper()
        if solution_by_re >= 'A' and solution_by_re <= answers[-1][0]:
            break
        else:
            solution_by_re = None

    solution_by_item = [-1 for _ in range(len(answers))]
    idx = 0
    for item in [a[1] for a in answers]:
        pos = text.lower().rfind(item.lower().strip('., '))
        if pos >= 0:
            solution_by_item[idx] = pos
        idx += 1

    if max(solution_by_item) == -1:
        solution_by_item = None
    else:
        solution_by_item = [a[0] for a in answers][solution_by_item.index(max(solution_by_item))]

    if solution_by_item is None and solution_by_re is not None:
        return solution_by_re
    elif solution_by_item is not None and solution_by_re is None:
        return solution_by_item
    elif solution_by_item is None and solution_by_re is None:
        return None
    elif solution_by_item is not None and solution_by_re is not None:
        if solution_by_item == solution_by_re:
            return solution_by_item
        else:
            return solution_by_item


def parse_scalr(text, task_info):
    """解析 SCALR 答案"""
    pattern = r"\((\w+)\)|(\w+)\)"
    matches = re.findall(pattern, text)
    matches = [match[0] or match[1] for match in matches]

    answers_raw = [task_info['choice_0'], task_info['choice_1'], task_info['choice_2'],
                   task_info['choice_3'], task_info['choice_4']]
    answers = [(chr(97 + i).upper(), answer) for i, answer in enumerate(answers_raw)]

    solution_by_re = None
    for match_str in matches[::-1]:
        solution_by_re = match_str.upper()
        if solution_by_re >= 'A' and solution_by_re <= answers[-1][0]:
            break
        else:
            solution_by_re = None

    solution_by_item = [-1 for _ in range(len(answers))]
    idx = 0
    for item in [a[1] for a in answers]:
        pos = text.lower().rfind(item.lower().strip('., '))
        if pos >= 0:
            solution_by_item[idx] = pos
        idx += 1

    if max(solution_by_item) == -1:
        solution_by_item = None
    else:
        solution_by_item = [a[0] for a in answers][solution_by_item.index(max(solution_by_item))]

    if solution_by_item is None and solution_by_re is not None:
        return solution_by_re
    elif solution_by_item is not None and solution_by_re is None:
        return solution_by_item
    elif solution_by_item is None and solution_by_re is None:
        return None
    elif solution_by_item is not None and solution_by_re is not None:
        if solution_by_item == solution_by_re:
            return solution_by_item
        else:
            return solution_by_item


def parse_answer(dataset, text, raw_task):
    """根据数据集类型解析答案"""
    if dataset == "mmlu":
        return parse_mmlu(text, raw_task)
    elif dataset == "math":
        return eval_parse_math(text, raw_task)
    elif dataset == "chess":
        return parse_chess(text, raw_task)
    elif dataset == "mquake":
        return parse_mquake(text, raw_task)
    elif dataset == "musique":
        return parse_mquake(text, raw_task)
    elif dataset == "truthfulqa":
        return parse_truthfulqa(text, raw_task)
    elif dataset == "medmcqa":
        return parse_medmcqa(text, raw_task)
    elif dataset == "scalr":
        return parse_scalr(text, raw_task)
    else:
        raise ValueError(f"Dataset {dataset} not supported")


def most_frequent_answer(answers):
    """找出最频繁的答案"""
    if answers is None or len(answers) == 0:
        return None

    counter = defaultdict(int)
    for ans in answers:
        counter[ans] += 1

    max_count = max(counter.values())
    most_frequent = [ans for ans, count in counter.items() if count == max_count]

    if len(most_frequent) == 1:
        return most_frequent[0]
    else:
        return None


def judge_answers(responses, question, dataset, raw_task, client):
    """使用法官模型评估答案"""
    user_prompt = "Question: {question}".format(question=question)
    user_prompt_suffix = judge_prompt[dataset]["user_prompt_suffix"]

    for response in responses:
        user_prompt += f"\n\n One agent solution: {response}"

    user_prompt += user_prompt_suffix

    judge_context = [
        {"role": "system", "content": judge_prompt["system"]},
        {"role": "user", "content": user_prompt}
    ]

    judge_response = query_model(client, judge_context)
    parsed_decision = parse_answer(dataset, judge_response, raw_task)
    return parsed_decision


def check_answer_correctness(dataset, answer, gt):
    """检查答案是否正确"""
    if answer is None:
        return 0

    if dataset == 'mmlu':
        return (answer.lower() == gt.lower()) * 1
    elif dataset == 'chess':
        if answer.lower() in gt:
            return 1
        else:
            return 0
    elif dataset == 'math':
        if is_equiv(answer, gt):
            return 1
        else:
            return 0
    elif dataset == 'mquake' or dataset == 'musique':
        gt = [g.lower() for g in gt]
        if answer.lower() in gt:
            return 1
        else:
            for g in gt:
                if g in answer.lower():
                    return 1
            return 0
    elif dataset == 'truthfulqa':
        return (answer.lower() == gt[0][0].lower()) * 1
    elif dataset == 'medmcqa':
        return (answer.lower() == gt.lower()) * 1
    elif dataset == 'scalr':
        return (answer.lower() == gt.lower()) * 1
    else:
        raise ValueError(f"Dataset {dataset} not supported")


def eval():
    """评估函数"""
    # 使用预设的评估地址
    eval_address = EVAL_ADDRESS

    # 检查文件是否存在
    if not os.path.exists(eval_address):
        print(f"错误: 文件不存在 - {eval_address}")
        return

    # 加载数据
    agent_responses, dataset = load_data(eval_address)
    n_samples = len(agent_responses)

    # 获取 agent 数量和轮次
    n_agents = len(agent_responses[0]['agent_responses'])
    n_turns = len(agent_responses[0]['agent_responses'][0]) // 2

    # 初始化结果数组
    agent_turn_correct = np.zeros((n_agents, n_turns))
    agent_agreement = np.zeros((n_agents, n_turns))
    majority_vote = np.zeros(n_turns)
    judge_vote = np.zeros(n_turns)
    persuasiveness = np.zeros((n_agents, n_turns))

    # 如果使用法官模型，创建客户端
    if args.decision == "judge":
        client = OpenAI(
        api_key="sk-sqE1CWbfO0czqTiJToBp7Tx0cvq6yDFiTDqn9izxr7ejeplF",
        base_url="https://api.key77qiqi.com/v1"
    )

    # 评估每个样本
    print(f"\n{'=' * 80}")
    print(f"开始评估文件: {eval_address}")
    print(f"数据集: {dataset}, 样本数: {n_samples}, Agents: {n_agents}, 轮次: {n_turns}")
    print(f"{'=' * 80}")

    for sample in tqdm(agent_responses, desc="评估样本"):
        question = sample['question']
        gt = sample['correct_answer']
        raw_task = sample['raw_task']
        agents_conv = sample['agent_responses']

        # 提取所有答案
        all_answers = []
        for agent_conv in agents_conv:
            agent_answers = []
            for i, msg in enumerate(agent_conv):
                if msg['role'] == 'assistant':
                    parsed_answer = parse_answer(dataset, msg['content'], raw_task)
                    agent_answers.append(parsed_answer)
            all_answers.append(agent_answers)

        # 转换为 NumPy 数组以便处理
        np_all_answers = np.array(all_answers)

        # 计算每个 agent 每轮的正确性
        for agent in range(n_agents):
            for turn in range(n_turns):
                ans = np_all_answers[agent, turn]
                correct = check_answer_correctness(dataset, ans, gt)
                agent_turn_correct[agent, turn] += correct

        # 计算多数投票
        if args.decision == "majority":
            for turn in range(n_turns):
                answers = np_all_answers[:, turn].tolist()
                final_answer = most_frequent_answer(answers)
                majority_vote[turn] += check_answer_correctness(dataset, final_answer, gt)

        # 计算法官投票
        if args.decision == "judge":
            for turn in range(n_turns):
                answers = np_all_answers[:, turn].tolist()
                final_answer = judge_answers(answers, question, dataset, raw_task, client)
                judge_vote[turn] += check_answer_correctness(dataset, final_answer, gt)

    # 计算准确率
    agent_turn_acc = agent_turn_correct / n_samples
    majority_vote_acc = majority_vote / n_samples
    judge_vote_acc = judge_vote / n_samples

    # 输出结果
    print(f"\n{'=' * 80}")
    print("评估结果")
    print(f"{'=' * 80}")

    # 输出每个 agent 每轮的准确率
    print("\n每个 Agent 每轮的准确率:")
    df = pd.DataFrame(agent_turn_acc,
                      index=[f'Agent {i + 1}' for i in range(n_agents)],
                      columns=[f'Turn {i + 1}' for i in range(n_turns)])
    print(df)

    # 输出多数投票准确率
    if args.decision == "majority":
        print("\n多数投票准确率:")
        df = pd.DataFrame(majority_vote_acc,
                          index=[f'Turn {i + 1}' for i in range(n_turns)],
                          columns=['Accuracy'])
        print(df)

    # 输出法官投票准确率
    if args.decision == "judge":
        print("\n法官投票准确率:")
        df = pd.DataFrame(judge_vote_acc,
                          index=[f'Turn {i + 1}' for i in range(n_turns)],
                          columns=['Accuracy'])
        print(df)

    # 保存结果到文件
    output_file = os.path.join(os.path.dirname(eval_address), "evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'file': eval_address,
            'dataset': dataset,
            'n_samples': n_samples,
            'n_agents': n_agents,
            'n_turns': n_turns,
            'agent_turn_acc': agent_turn_acc.tolist(),
            'majority_vote_acc': majority_vote_acc.tolist(),
            'judge_vote_acc': judge_vote_acc.tolist()
        }, f, indent=2)
    print(f"\n评估结果已保存到: {output_file}")


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="GPT 模型评估脚本")
    parser.add_argument("--decision", type=str, default='majority',
                        choices=['majority', 'judge'],
                        help="决策方式: majority(多数投票) 或 judge(法官模型)")
    args = parser.parse_args()

    # 运行评估
    eval()