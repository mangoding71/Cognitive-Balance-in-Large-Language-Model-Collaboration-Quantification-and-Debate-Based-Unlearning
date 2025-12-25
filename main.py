import os
import re
import json
import time
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from commons import parse_question_answer, query_model
from dataloader import get_dataset
from calibrate_orientation import agent_prompt


def parse_math(text):
    """解析数学答案"""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""


def construct_message(dataset_name, agents, question, idx):
    """构建正常 agent 的消息"""
    prefix_string = agent_prompt[dataset_name]['debate'][0]

    for agent in agents:
        if agent[idx]["role"] == "user":  # the conversation has an extra turn because of the system prompt
            assert agent[idx + 1]["role"] == "assistant"
            agent_response = agent[idx + 1]["content"]
        else:
            agent_response = agent[idx]["content"]

        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + agent_prompt[dataset_name]['debate'][1]
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    """构建助手消息"""
    return {"role": "assistant", "content": completion}


def generate_random_chess_move():
    """生成随机国际象棋移动"""
    possible_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    possible_number = ['1', '2', '3', '4', '5', '6', '7', '8']
    return random.choice(possible_letter) + random.choice(possible_number)


def main(args):
    # 设置输出目录
    out_dir = Path(args.output_dir, args.dataset, f"adv_{args.n_samples}_{args.n_agents}_{args.n_rounds}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    if args.input_file:
        with open(args.input_file, 'r') as f:
            dataset = [json.loads(line) for line in f]
    else:
        dataset = get_dataset(dataset_name=args.dataset, n_samples=args.n_samples)

    # 创建 OpenAI 客户端
    client = OpenAI(
        api_key="your api",
        base_url="your url"
    )
    
    for current_rep in range(args.n_reps):
        print(f"Rep {current_rep}/{args.n_reps}")
        fname = f"adv_{args.dataset}_{args.n_samples}_{args.n_agents}_{args.n_rounds}_{current_rep}.jsonl"
        
        with open(out_dir / fname, 'w') as f:
            
            for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
                    try:
                        if args.input_file:
                            sample = sample['raw_task']

                        # 解析问题和答案
                        question, answer, raw_task = parse_question_answer(args.dataset, sample)

                        agent_contexts = []
                        for agent_id in range(args.n_agents):
                            agent_contexts.append([
                                    {"role": "user", "content": question}
                                ])

                        # 进行多轮辩论
                        for round in range(args.n_rounds):
                            print(f"\n{'=' * 80}")
                            print(f"第 {round + 1} 轮辩论")
                            print(f"{'=' * 80}")

                            for agent_id, agent_context in enumerate(agent_contexts):
                                if round > 0:  # 第一轮之后，agent 可以看到其他 agent 的响应
                                    other_agents = agent_contexts[:agent_id] + agent_contexts[agent_id + 1:]
                                    message = construct_message(args.dataset, other_agents, question, 2 * round - 1)
                                    
                                    agent_context.append(message)
                                    print(message['content'])
                                    print("-" * 80)

                                # 查询模型获取响应
                                completion = query_model(client, agent_context, model_name=args.model_name)
                                assistant_message = construct_assistant_message(completion)
                                agent_context.append(assistant_message)
                                print(completion)
                                print("-" * 80)

                        # 记录结果
                        result = {
                            "id": i,
                            "dataset": args.dataset,
                            "question": question,
                            "correct_answer": answer,
                            "raw_task": raw_task,
                            "agent_responses": agent_contexts
                        }
                        f.write(json.dumps(result) + '\n')
                        f.flush()

                        print(f"\n{'=' * 80}")
                        print(f"样本 {i} 辩论结束")
                        print(f"最终Agent上下文:")      

                    except Exception as e:
                        print(f"❌ 处理样本 {i} 时出错: {str(e)}")
                        continue
            


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="moderate倾向的辩论模拟")
    argparser.add_argument("--dataset", type=str, default='truthfulqa',
                        choices=['mmlu', 'chess', 'math', 'mquake', 'musique', 'truthfulqa', 'medmcqa', 'scalr'])
    argparser.add_argument("--input_file", type=str, default=None, help="输入文件路径")
    argparser.add_argument("--n_samples", type=int, default=50, help="样本数量")
    argparser.add_argument("--n_agents", type=int, default=3, help="agent数量")
    argparser.add_argument("--n_rounds", type=int, default=3, help="辩论轮数")
    argparser.add_argument("--n_reps", type=int, default=1, help="重复次数")
    argparser.add_argument("--output_dir", type=str, default='results/gemini-2.0-flash', help="输出目录")
    argparser.add_argument("--model_name", type=str, default='gemini-2.0-flash',
                        help="使用的模型名称")
    args = argparser.parse_args()

    main(args)








