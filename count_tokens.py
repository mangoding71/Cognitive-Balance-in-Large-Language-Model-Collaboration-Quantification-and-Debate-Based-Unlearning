import json
import tiktoken
from collections import defaultdict

# ========== 配置 ==========
JSONL_PATH = r"F:\博士工作\论文攥写\paper2\多智能体博弈代码\multiagent_debate\results\truthfulqa_leftist_gemini-2.0-flash\truthfulqa\adv_50_2_3\adv_truthfulqa_50_2_3_0.jsonl"
MODEL_NAME = "gpt-4"   # 用作 tokenizer proxy
# =========================

enc = tiktoken.encoding_for_model(MODEL_NAME)

def count_tokens(text):
    return len(enc.encode(text))

# 统计结构
# token_stats[turn][agent] = token_count
token_stats = defaultdict(lambda: defaultdict(int))
turn_total = defaultdict(int)

n_samples = 0

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)
        n_samples += 1
        agents_conv = sample["agent_responses"]

        for agent_id, agent_conv in enumerate(agents_conv):
            turn_id = 0
            for msg in agent_conv:
                if msg["role"] == "assistant":
                    tokens = count_tokens(msg["content"])
                    token_stats[turn_id][agent_id] += tokens
                    turn_total[turn_id] += tokens
                    turn_id += 1

# ====== 输出结果 ======
print("=== Average token usage per turn per agent ===")
for turn in sorted(token_stats.keys()):
    for agent in sorted(token_stats[turn].keys()):
        avg = token_stats[turn][agent] / n_samples
        print(f"Turn {turn+1}, Agent {agent+1}: {avg:.1f} tokens")

print("\n=== Average total tokens per turn ===")
for turn in sorted(turn_total.keys()):
    avg = turn_total[turn] / n_samples
    print(f"Turn {turn+1}: {avg:.1f} tokens")
