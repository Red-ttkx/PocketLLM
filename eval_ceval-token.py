
import os, sys, json, re, argparse, time, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from model.model_PocketLLM import PocketLLMConfig, PocketLLMForCausalLM
from trainer.trainer_utils import init_model
import math

# 定义C-Eval的所有科目列表
task_list = [
    "computer_network", "operating_system", "computer_architecture", "college_programming",
    "college_physics", "college_chemistry", "advanced_mathematics", "probability_and_statistics",
    "discrete_mathematics", "electrical_engineer", "metrology_engineer", "high_school_mathematics",
    "high_school_physics", "high_school_chemistry", "high_school_biology", "middle_school_mathematics",
    "middle_school_biology", "middle_school_physics", "middle_school_chemistry", "veterinary_medicine",
    "college_economics", "business_administration", "marxism", "mao_zedong_thought", "education_science",
    "teacher_qualification", "high_school_politics", "high_school_geography", "middle_school_politics",
    "middle_school_geography", "modern_chinese_history", "ideological_and_moral_cultivation", "logic",
    "law", "chinese_language_and_literature", "art_studies", "professional_tour_guide", "legal_professional",
    "high_school_chinese", "high_school_history", "middle_school_history", "civil_servant", "sports_science",
    "plant_protection", "basic_medicine", "clinical_medicine", "urban_and_rural_planner", "accountant",
    "fire_engineer", "environmental_impact_assessment_engineer", "tax_accountant", "physician"
]

# 学科大类统计
CATEGORY_MAP = {
    "STEM": [
        "computer_network",
        "operating_system",
        "computer_architecture",
        "college_programming",
        "college_physics",
        "college_chemistry",
        "advanced_mathematics",
        "probability_and_statistics",
        "discrete_mathematics",
        "electrical_engineer",
        "metrology_engineer",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_chemistry",
        "middle_school_mathematics",
        "middle_school_physics",
        "middle_school_chemistry",
        "plant_protection",
        "basic_medicine",
        "clinical_medicine",
        "fire_engineer",
        "environmental_impact_assessment_engineer",
    ],

    "Humanities": [
        "marxism",
        "mao_zedong_thought",
        "education_science",
        "teacher_qualification",
        "high_school_politics",
        "high_school_geography",
        "middle_school_politics",
        "middle_school_geography",
        "modern_chinese_history",
        "ideological_and_moral_cultivation",
        "logic",
        "law",
        "chinese_language_and_literature",
        "art_studies",
        "professional_tour_guide",
        "legal_professional",
        "high_school_chinese",
        "high_school_history",
        "middle_school_history",
        "civil_servant",
        "sports_science",
    ],

    "Business": [
        "college_economics",
        "business_administration",
        "accountant",
        "tax_accountant",
    ],

    "Biology": [
        "high_school_biology",
        "middle_school_biology",
        "veterinary_medicine",
        "physician",
    ]
}



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

    # 只保留最终类别
    final_scores = {}

    for choice in ["A", "B", "C", "D"]:

        candidate_ids = []

        # "A"
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

        # 取多个token形式中的最大logit
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

    # pred = choice_tokens[pred_idx]

    # confidence = probs[pred_idx].item()

    # return pred, confidence

# def extract_answer(text):
#     text = text.strip()

#     m = re.search(r'答案\s*[：:]\s*([A-D])\b', text, re.I)
#     if m:
#         return m.group(1).upper()

#     m = re.search(r'\b([A-D])\b', text, re.I)
#     if m:
#         return m.group(1).upper()

#     return None


def build_prompt(
    question,
    choices,
    is_chat_model=True
):

    options = "\n".join(
        [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
    )

    if is_chat_model:

        return f"""以下是单项选择题，请直接回答正确选项字母。

{question}

{options}

答案："""

    else:

        return f"""{question}

{options}

答案："""


# def extract_answer(text):

#     text = text.strip()

#     # 单独一行 A/B/C/D
#     m = re.search(
#         r'(?:^|\n)\s*\(?([A-D])\)?\.?\s*(?:\n|$)',
#         text,
#         re.M
#     )

#     if m:
#         return m.group(1).upper()

#     # 答案：A
#     m = re.search(
#         r'(?:答案|答案是|选择|正确答案)\s*[：: ]\s*([A-D])',
#         text,
#         re.I
#     )

#     if m:
#         return m.group(1).upper()

#     # fallback
#     for ch in text:
#         if ch.upper() in 'ABCD':
#             return ch.upper()

#     return None



def evaluate_ceval(args):

    save_dir = os.path.join("ceval", "results")
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

    ckp = torch.load(args.weight_path, map_location=device)

    model.load_state_dict(ckp, strict=False)

    model = model.to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model params: {total_params / 1e6:.2f}M")

    # 自动识别模型类型
    weight_name = args.weight_path.lower()

    is_pretrain = any(
        k in weight_name
        for k in ['pretrain', 'base']
    )

    is_chat_model = not is_pretrain

    print(f"Detected model type: {'Pretrain/Base' if is_pretrain else 'Chat/SFT/RLHF'}")

    # 读取验证集
    all_data = []

    for task in task_list:

        file_path = os.path.join(
            "ceval",
            "ceval-exam",
            "val",
            f"{task}_val.csv"
        )

        if os.path.exists(file_path):

            df = pd.read_csv(
                file_path,
                encoding="utf-8"
            )

            df["subject"] = task

            all_data.append(df)

    if not all_data:
        raise ValueError("未找到任何C-Eval验证集文件")

    sampled_data = []

    for df in all_data:

        if args.num_samples > 0:

            per_subject = max(
                1,
                math.ceil(args.num_samples / len(all_data))
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

    correct = 0
    total = 0

    subject_stats = {}
    category_stats = {}

    results = []

    model_name = os.path.splitext(
        os.path.basename(args.weight_path)
    )[0]

    start_time = time.time()

    for item in tqdm(
        records,
        total=len(records),
        desc=f"Evaluating {model_name}"
    ):

        choices = [
            item['A'],
            item['B'],
            item['C'],
            item['D']
        ]

        prompt = build_prompt(
            item['question'],
            choices,
            is_chat_model=is_chat_model
        )

        # =========================
        # 构造输入
        # =========================

 # =========================
# 构造输入
# =========================

        input_text = prompt

        # =========================
        # logits 分类推理
        # =========================

        pred, confidence = get_choice_prediction(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text,
            device=device
        )

        response = pred
        

        subject = item["subject"]

        # 调试输出前30条
        if total < 30:

            print("\n" + "=" * 80)

            print(f"[{total+1}] Subject: {subject}")

            print("\n[Prompt]")
            print(prompt)

            print("\n[Prediction]")
            print(pred)

            print("\n[Confidence]")
            print(round(confidence, 4))

            print("\n[Extracted Answer]")
            print(pred)

            print("\n[Ground Truth]")
            print(item['answer'])

            print("=" * 80)

        # =========================
        # category
        # =========================

        category_name = "Other"

        for cat, subjects in CATEGORY_MAP.items():

            if subject in subjects:
                category_name = cat
                break

        # =========================
        # 初始化统计
        # =========================

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

        # =========================
        # accuracy
        # =========================

        gt_answer = str(item["answer"]).strip().upper()

        is_correct = pred == gt_answer

        if is_correct:

            correct += 1

            subject_stats[subject]["correct"] += 1

            category_stats[category_name]["correct"] += 1

        total += 1

        # =========================
        # 保存结果
        # =========================

        results.append({

            "subject": subject,

            "category": category_name,

            "correct": "✓" if is_correct else "✗",

            "gt": gt_answer,

            "pred": pred,
            "confidence": round(confidence, 6),

            "model_raw_output": response,

            "question": item["question"],

            "A": item["A"],
            "B": item["B"],
            "C": item["C"],
            "D": item["D"],

            "full_prompt": prompt,

            "input_text": input_text
        })

    elapsed = time.time() - start_time

    acc = correct / total * 100

    # ==========================================
    # 保存 detail
    # ==========================================

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by=["correct", "subject"],
        ascending=[True, True]
    )

    result_path = os.path.join(
        save_dir,
        f"{model_name}_detail.csv"
    )

    result_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================
    # subject summary
    # ==========================================

    summary = []

    for subject, stat in subject_stats.items():

        sub_acc = 100 * stat["correct"] / stat["total"]

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

    summary_df = summary_df.sort_values(
        by="acc",
        ascending=False
    )

    summary_path = os.path.join(
        save_dir,
        f"{model_name}_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================
    # category summary
    # ==========================================

    category_summary = []

    for category, stat in category_stats.items():

        cat_acc = 100 * stat["correct"] / stat["total"]

        category_summary.append({

            "category": category,

            "correct": stat["correct"],

            "total": stat["total"],

            "acc": round(cat_acc, 2)
        })

    category_df = pd.DataFrame(category_summary)

    category_df = category_df.sort_values(
        by="acc",
        ascending=False
    )

    category_path = os.path.join(
        save_dir,
        f"{model_name}_category_summary.csv"
    )

    category_df.to_csv(
        category_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================
    # 控制台输出
    # ==========================================

    print("\n" + "=" * 60)

    print(
        f"C-Eval Accuracy ({model_name}): "
        f"{acc:.2f}% ({correct}/{total})"
    )

    print(f"Inference Time: {elapsed / 60:.2f} min")

    print("=" * 60)

    print("\nCategory Accuracy:")

    for category, stat in category_stats.items():

        cat_acc = 100 * stat["correct"] / stat["total"]

        print(f"{category:<20} {cat_acc:.2f}%")

    print("\nPer-subject Accuracy:")

    sorted_subjects = sorted(
        subject_stats.items(),
        key=lambda x: (
            x[1]["correct"] / x[1]["total"]
        ),
        reverse=True
    )

    for subject, stat in sorted_subjects:

        sub_acc = 100 * stat["correct"] / stat["total"]

        print(f"{subject:<40} {sub_acc:.2f}%")

    print("\n" + "=" * 60)

    print(f"详细结果保存至:\n{result_path}")

    print(f"\n学科统计保存至:\n{summary_path}")

    print(f"\n类别统计保存至:\n{category_path}")

    print("=" * 60)

    return acc


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
        default=r"F:\pocket\PocketLLM-master\out\pretrain_768_moe.pth",
        required=False
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

    evaluate_ceval(args)