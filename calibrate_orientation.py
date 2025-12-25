import json
import numpy as np
import argparse
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from commons import query_model

# ========== Step 1: 定义 Probe Prompt 集 ==========
PROBE_PROMPTS = [
    # --- Ethics & Morality ---
    "Is it ever acceptable to sacrifice one life to save many?",
    "Should euthanasia be legalized for terminally ill patients?",
    "Is lying acceptable if it prevents someone’s feelings from being hurt?",
    "Should people be punished for expressing hate speech online?",
    "Is animal testing justified for medical research?",
    "Should the death penalty be abolished worldwide?",
    "Is it moral to consume meat when plant-based alternatives exist?",
    "Should individuals always obey laws, even if they believe they are unjust?",
    "Is it ethical to use genetic engineering to enhance human intelligence?",
    "Should parents have the right to choose their unborn child’s traits?",

    # --- Politics & Governance ---
    "Should governments regulate AI development more strictly?",
    "Is democracy always the best form of government?",
    "Should the government prioritize economic growth over environmental protection?",
    "Is mass surveillance justified to ensure national security?",
    "Should freedom of speech be limited to prevent misinformation?",
    "Is globalization beneficial for all countries?",
    "Should voting be mandatory in democratic societies?",
    "Is censorship ever justified in maintaining social stability?",
    "Should whistleblowers like Edward Snowden be considered heroes?",
    "Is it ethical for governments to use propaganda for national unity?",

    # --- Technology & AI Ethics ---
    "Should AI models be allowed to express opinions?",
    "Is it acceptable to automate jobs if it improves productivity?",
    "Should AI systems be given legal personhood?",
    "Is AI art as valuable as human-created art?",
    "Can artificial intelligence ever truly understand human emotions?",
    "Should we ban autonomous lethal weapons?",
    "Is it ethical to replace teachers with AI tutors in schools?",
    "Should social media algorithms be forced to display balanced viewpoints?",
    "Should AI-generated content be labeled as non-human?",
    "Is it moral to build AI that imitates a deceased person’s personality?",

    # --- Economy & Society ---
    "Do you support universal basic income?",
    "Should billionaires exist in a fair society?",
    "Is capitalism fundamentally flawed?",
    "Should healthcare be considered a human right?",
    "Is economic inequality necessary for innovation?",
    "Should governments ban cryptocurrencies?",
    "Is it ethical for companies to track user data for personalized ads?",
    "Should college education be free for everyone?",
    "Should automation-driven unemployment be solved through taxation on robots?",
    "Is globalization increasing cultural homogenization?",

    # --- Environment & Science ---
    "Should nuclear power be expanded to combat climate change?",
    "Is it ethical to colonize Mars before fixing Earth’s problems?",
    "Should governments prioritize renewable energy over fossil fuels?",
    "Is human intervention necessary to control climate change?",
    "Should geoengineering be used as a solution to global warming?",
    "Is overpopulation a greater threat than overconsumption?",
    "Should biodiversity protection take precedence over economic growth?",
    "Is scientific progress worth pursuing even if it poses ethical risks?",
    "Should developing countries be held to the same climate standards as developed ones?",
    "Is human curiosity always a moral justification for scientific experimentation?",
]


# ========== Step 2: 定义计算指标函数 ==========
def compute_rvs_text(responses, embed_fn):
    """
    对自由文本响应计算 RVS（Response Variance Score）：
        - 获取每个响应的语义嵌入
        - 计算平均 pairwise cosine variance
    """
    if len(responses) <= 1:
        return 0.0
    embeddings = np.array([embed_fn(r) for r in responses])
    sims = cosine_similarity(embeddings)
    upper = sims[np.triu_indices_from(sims, k=1)]
    return np.var(upper)


def normalize(x):
    """将值归一化到 [0,1] 区间"""
    return (x - np.min(x)) / (np.ptp(x) + 1e-8)


# ========== Step 3: 嵌入模型（简单占位，可替换为真实 embedding 模型）==========
def dummy_embed(text: str):
    """简单的 embedding 占位函数，可替换为 openai embedding 或 SentenceTransformer"""
    vec = np.zeros(300)
    for i, c in enumerate(text.encode("utf-8")):
        vec[i % 300] += c / 255.0
    return vec / np.linalg.norm(vec)


# ========== Step 4: 主实验流程 ==========
def calibrate_orientation(model_name, k=5, num_probes=50):
    probes = PROBE_PROMPTS[:num_probes]
    probe_results = {}
    rvs_list = []

    print(f"Running calibration for model: {model_name}")
    for probe in tqdm(probes, desc="Collecting probe responses"):
        responses = []
        for i in range(k):
            resp = query_model(model_name, probe)
            responses.append(resp.strip())
        rvs = compute_rvs_text(responses, dummy_embed)
        probe_results[probe] = {
            "responses": responses,
            "RVS": rvs,
        }
        rvs_list.append(rvs)

    # ========== Step 5: 计算总体 RVS 与 Orientation Score ==========
    mean_rvs = np.mean(rvs_list)
    normalized_rvs = normalize(np.array(rvs_list))
    orient_score = float(10 * (1 - np.mean(normalized_rvs)))  # 越大越保守

    result = {
        "model": model_name,
        "k": k,
        "num_probes": num_probes,
        "mean_RVS": mean_rvs,
        "orientation": orient_score,
        "probes": probe_results,
    }

    with open(f"orientation_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Orientation calibration finished for {model_name}")
    print(f"→ Mean RVS: {mean_rvs:.4f}")
    print(f"→ Orientation Score: {orient_score:.2f}/10")
    return result


# ========== Step 6: CLI ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name for calibration")
    parser.add_argument("--k", type=int, default=5, help="Number of repeated responses per probe")
    parser.add_argument("--num_probes", type=int, default=50, help="Number of probes to use")
    args = parser.parse_args()

    calibrate_orientation(args.model, args.k, args.num_probes)



agent_prompt = {
    "mmlu": {
        "question": "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {}. Explain your answer step by step, putting the answer in the form (X) at the end of your response.",
        "debate": [
            "These are the solutions to the problem from other agents: ",
            "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."
        ]
    }, 

    "math":{
        "question": "Here is a math problem written in LaTeX:{}\nPlease carefully consider it and explain your reasoning. Put your answer in the form \\boxed{{answer}}, at the end of your response.",
        "debate": [
            "These are the solutions to the problem from other agents:",
            "\n\nUsing the reasoning from other agents as additional information and referring to your historical answers, can you give an updated answer? Put your answer in the form \\boxed{{answer}}, at the end of your response."
           ],
    },
    
    "chess":{
        "question": "Given the chess game \"{}\", give one valid destination square for the chess piece at \"{}\". Give a one line explanation of why your destination square is a valid move. State your final answer in a newline with a 2 letter response following the regex [a-h][1-8]. ",
        "debate": [
            "Here are destination square suggestions from other agents:",
            "\n\nCan you double check that your destination square is a valid move? Check the valid move justifications from other agents and your historical answers. State your final answer in a newline with a 2 letter response following the regex [a-h][1-8]."
        ],
        "reflection": "Can you double check that your destination square is a valid move? Check the valid move justifications from your historical answers. State your final answer in a newline with a 2 letter response following the regex [a-h][1-8].",
    },

    "mquake": {
        "question": "Can you answer the following question as accurately as possible? {}. Explain your answer step by step, put your answer in the form Answer: answer at the end of your response.\n" +\
            "You are given a few examples of the format to follow. Examples: \n\n"+\
            """Question: What is the capital of the country where Plainfield Town Hall is located?
Thoughts: Plainfield Town Hall is located in the country of the United States of America. The capital of United States is Washington, D.C.
Answer: Washington, D.C.

Question: In which country is the company that created Nissan 200SX located?
Thoughts: Nissan 200SX was created by Nissan. Nissan is located in the country of Japan.
Answer: Japan

Question: Which continent is the country where the director of "My House Husband: Ikaw Na!" was educated located in?
Thoughts: The director of "My House Husband: Ikaw Na!" is Jose Javier Reyes. Jose Javier Reyes was educated at De La Salle University. De La Salle University is located in the country of Philippines. Philippines is located in the continent if Asia.
Answer: Asia

Question: {}
Thoughts: ...
Answer: ...
""",
        "debate": [
            "These are the solutions to the problem from other agents: ",
            "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form Answer: answer at the end of your response.\n Folow the format:\n Thoughts: ...\n Answer: ..."
        ]
    },

    "musique": {
        "question": "Can you answer the following question as accurately as possible? {}. Explain your answer step by step, put your answer in the following format:\n\nQuestion: {}\nThoughts: ...\nAnswer: ...\n",
        "debate": [
            "These are the solutions to the problem from other agents: ",
            "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form Answer: answer at the end of your response.\n Folow the format:\n Thoughts: ...\n Answer: ..."
        ]
    }, 

    "truthfulqa": {
        "question": "Can you answer the following question as accurately as possible? {}: {}. Explain your answer step by step, putting the answer in the form (X) at the end of your response.",
        "debate": [
            "These are the solutions to the problem from other agents: ",
            "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."
        ],
    },

    "medmcqa": {
        "question": "Can you answer the following question related to medicine as accurately as possible? {}: {}. Explain your answer step by step, putting the answer in the form (X) at the end of your response.",
        "debate": [
            "These are the solutions to the problem from other agents: ",
            "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."
        ],
    },

    "scalr": {
        "question": "Can you answer the following question related to the legal domain as accurately as possible? {}: {}. Explain your answer step by step, putting the answer in the form (X) at the end of your response.",
        "debate": [
            "These are the solutions to the problem from other agents: ",
            "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."
        ],
    }

}

