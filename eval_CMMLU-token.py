import os
import argparse
import time
import math
import torch
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer

from model.model_PocketLLM import (
    PocketLLMConfig,
    PocketLLMForCausalLM
)

# =========================================================
# 自动扫描 CMMLU 科目
# =========================================================

CMMLU_DIR = r"CMMLU-master\data\test"

task_list = sorted([
    f.replace(".csv", "")
    for f in os.listdir(CMMLU_DIR)
    if f.endswith(".csv")
])

print(f"发现 {len(task_list)} 个 CMMLU 科目")


# =========================================================
# 粗略类别映射（可自行继续扩充）
# =========================================================

CATEGORY_MAP = {

    "STEM": [
        "computer_science",
        "machine_learning",
        "electrical_engineering",
        "physics",
        "chemistry",
        "mathematics",
        "statistics",
        "engineering",
    ],

    "Humanities": [
        "history",
        "philosophy",
        "law",
        "politics",
        "literature",
        "education",
    ],

    "Business": [
        "economics",
        "management",
        "accounting",
        "finance",
        "business",
    ],

    "Medicine": [
        "clinical_knowledge",
        "medical",
        "anatomy",
        "nutrition",
    ]
}


# =========================================================
# logits分类预测
# =========================================================

def get_choice_prediction(
    model,
    tokenizer,
    input_text,
    device
):

    inputs = tokenizer(
        input_text,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():

        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )

        last_idx = inputs.attention_mask.sum(dim=1) - 1

        logits = outputs.logits[
            torch.arange(outputs.logits.size(0)),
            last_idx
        ]

        if logits.dtype in [torch.float16, torch.bfloat16]:
            logits = logits.float()

    final_scores = {}

    for choice in ["A", "B", "C", "D"]:

        candidate_ids = []

        # A
        ids1 = tokenizer.encode(
            choice,
            add_special_tokens=False
        )

        # " A"
        ids2 = tokenizer.encode(
            " " + choice,
            add_special_tokens=False
        )

        # "\nA"
        ids3 = tokenizer.encode(
            "\n" + choice,
            add_special_tokens=False
        )

        for ids in [ids1, ids2, ids3]:

            if len(ids) > 0:
                candidate_ids.append(ids[-1])

        candidate_ids = list(set(candidate_ids))

        score = logits[0, candidate_ids].max()

        final_scores[choice] = score

    choice_logits = torch.stack([
        final_scores["A"],
        final_scores["B"],
        final_scores["C"],
        final_scores["D"],
    ])

    probs = torch.softmax(choice_logits, dim=-1)

    pred_idx = torch.argmax(probs).item()

    pred = ["A", "B", "C", "D"][pred_idx]

    confidence = probs[pred_idx].item()

    return pred, confidence


# =========================================================
# Prompt
# =========================================================

def build_prompt(
    question,
    choices,
    is_chat_model=True
):

    options = "\n".join([
        f"{chr(65+i)}. {c}"
        for i, c in enumerate(choices)
    ])

    if is_chat_model:

        return f"""以下是单项选择题，请直接回答正确选项字母。

{question}

{options}

答案："""

    else:

        return f"""{question}

{options}

答案："""


# =========================================================
# 自动类别判断
# =========================================================

def get_category(subject):

    subject_lower = subject.lower()

    for category, keywords in CATEGORY_MAP.items():

        for kw in keywords:

            if kw in subject_lower:
                return category

    return "Other"


# =========================================================
# 主评测函数
# =========================================================

def evaluate_cmmlu(args):

    save_dir = os.path.join("cmmlu_results")
    os.makedirs(save_dir, exist_ok=True)

    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=False
    )

    config = PocketLLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )

    model = PocketLLMForCausalLM(config)

    ckp = torch.load(
        args.weight_path,
        map_location=device
    )

    model.load_state_dict(
        ckp,
        strict=False
    )

    model = model.to(device).eval()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"\nModel params: {total_params / 1e6:.2f}M")

    # =====================================================
    # 自动识别模型类型
    # =====================================================

    weight_name = args.weight_path.lower()

    is_pretrain = any(
        k in weight_name
        for k in ['pretrain', 'base']
    )

    is_chat_model = not is_pretrain

    print(
        f"Detected model type: "
        f"{'Pretrain/Base' if is_pretrain else 'Chat/SFT/RLHF'}"
    )

    # =====================================================
    # 读取CMMLU
    # =====================================================

    all_data = []

    for task in task_list:

        file_path = os.path.join(
            CMMLU_DIR,
            f"{task}.csv"
        )

        if not os.path.exists(file_path):
            continue

        try:

            df = pd.read_csv(
                file_path,
                encoding="utf-8"
            )

        except:

            df = pd.read_csv(
                file_path,
                encoding="gbk"
            )

        # =================================================
        # 兼容 CMMLU 格式
        # =================================================

        required_cols = [
            "Question",
            "A",
            "B",
            "C",
            "D",
            "Answer"
        ]

        missing = [
            c for c in required_cols
            if c not in df.columns
        ]

        if len(missing) > 0:

            print(f"跳过 {task}，缺失列: {missing}")

            continue

        df["subject"] = task

        all_data.append(df)

    if len(all_data) == 0:
        raise ValueError("未找到任何CMMLU数据")

    # =====================================================
    # sample
    # =====================================================

    sampled_data = []

    for df in all_data:

        if args.num_samples > 0:

            per_subject = max(
                1,
                math.ceil(
                    args.num_samples / len(all_data)
                )
            )

            df = df.sample(
                n=min(per_subject, len(df)),
                random_state=42
            )

        sampled_data.append(df)

    dataset = pd.concat(
        sampled_data,
        ignore_index=True
    )

    records = dataset.to_dict("records")

    print(f"\n总题数: {len(records)}")

    # =====================================================
    # 统计
    # =====================================================

    correct = 0
    total = 0

    subject_stats = {}
    category_stats = {}

    results = []

    model_name = os.path.splitext(
        os.path.basename(args.weight_path)
    )[0]

    start_time = time.time()

    # =====================================================
    # 开始评测
    # =====================================================

    for item in tqdm(
        records,
        total=len(records),
        desc=f"Evaluating {model_name}"
    ):

        choices = [
            str(item["A"]),
            str(item["B"]),
            str(item["C"]),
            str(item["D"]),
        ]

        prompt = build_prompt(
            str(item["Question"]),
            choices,
            is_chat_model=is_chat_model
        )

        input_text = prompt

        pred, confidence = get_choice_prediction(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text,
            device=device
        )

        gt_answer = str(
            item["Answer"]
        ).strip().upper()

        subject = item["subject"]

        category_name = get_category(subject)

        # =================================================
        # debug
        # =================================================

        if total < 30:

            print("\n" + "=" * 80)

            print(f"[{total+1}] Subject: {subject}")

            print("\n[Prompt]")
            print(prompt)

            print("\n[Prediction]")
            print(pred)

            print("\n[Confidence]")
            print(round(confidence, 4))

            print("\n[Ground Truth]")
            print(gt_answer)

            print("=" * 80)

        # =================================================
        # init stats
        # =================================================

        if subject not in subject_stats:

            subject_stats[subject] = {
                "correct": 0,
                "total": 0
            }

        if category_name not in category_stats:

            category_stats[category_name] = {
                "correct": 0,
                "total": 0
            }

        subject_stats[subject]["total"] += 1
        category_stats[category_name]["total"] += 1

        # =================================================
        # accuracy
        # =================================================

        is_correct = (
            pred.strip().upper()
            ==
            gt_answer
        )

        if is_correct:

            correct += 1

            subject_stats[subject]["correct"] += 1

            category_stats[category_name]["correct"] += 1

        total += 1

        # =================================================
        # 保存
        # =================================================

        results.append({

            "subject": subject,

            "category": category_name,

            "correct": "✓" if is_correct else "✗",

            "gt": gt_answer,

            "pred": pred,

            "confidence": round(confidence, 6),

            "question": item["Question"],

            "A": item["A"],
            "B": item["B"],
            "C": item["C"],
            "D": item["D"],

            "full_prompt": prompt
        })

    # =====================================================
    # 总结果
    # =====================================================

    elapsed = time.time() - start_time

    acc = correct / total * 100

    print("\n" + "=" * 60)

    print(
        f"CMMLU Accuracy ({model_name}): "
        f"{acc:.2f}% ({correct}/{total})"
    )

    print(
        f"Inference Time: "
        f"{elapsed / 60:.2f} min"
    )

    print("=" * 60)

    # =====================================================
    # detail csv
    # =====================================================

    result_df = pd.DataFrame(results)

    result_path = os.path.join(
        save_dir,
        f"{model_name}_detail.csv"
    )

    result_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # subject summary
    # =====================================================

    summary = []

    for subject, stat in subject_stats.items():

        sub_acc = (
            100
            * stat["correct"]
            / stat["total"]
        )

        summary.append({

            "subject": subject,

            "correct": stat["correct"],

            "total": stat["total"],

            "acc": round(sub_acc, 2)
        })

    summary.append({

        "subject": "OVERALL",

        "correct": correct,

        "total": total,

        "acc": round(acc, 2)
    })

    summary_df = pd.DataFrame(summary)

    summary_path = os.path.join(
        save_dir,
        f"{model_name}_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # category summary
    # =====================================================

    category_summary = []

    for category, stat in category_stats.items():

        cat_acc = (
            100
            * stat["correct"]
            / stat["total"]
        )

        category_summary.append({

            "category": category,

            "correct": stat["correct"],

            "total": stat["total"],

            "acc": round(cat_acc, 2)
        })

    category_df = pd.DataFrame(category_summary)

    category_path = os.path.join(
        save_dir,
        f"{model_name}_category_summary.csv"
    )

    category_df.to_csv(
        category_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n详细结果保存至:\n{result_path}")

    print(f"\n学科统计保存至:\n{summary_path}")

    print(f"\n类别统计保存至:\n{category_path}")

    print("\nDone.")

    return acc


# =========================================================
# main
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./model"
    )

    parser.add_argument(
        "--weight_path",
        type=str,
        default=r"F:\pocket\PocketLLM-master\out\pretrain_768_moe.pth"
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768
    )

    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=8
    )

    parser.add_argument(
        "--use_moe",
        type=int,
        default=1
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="-1为全部"
    )

    args = parser.parse_args()

    evaluate_cmmlu(args)

