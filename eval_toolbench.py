import os
import re
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from model.model_PocketLLM import PocketLLMConfig, PocketLLMForCausalLM


# =========================================================
# 工具定义（与 eval_toolcall.py 完全一致）
# =========================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如123+456、2**10、sqrt(144)"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间，支持指定时区",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "时区名称，如Asia/Shanghai、America/New_York",
                        "default": "Asia/Shanghai"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "random_number",
            "description": "生成指定范围内的随机数",
            "parameters": {
                "type": "object",
                "properties": {
                    "min": {
                        "type": "integer",
                        "description": "最小值",
                        "default": 0
                    },
                    "max": {
                        "type": "integer",
                        "description": "最大值",
                        "default": 100
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "text_length",
            "description": "计算文本的字符数和单词数",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要统计的文本"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "进行单位换算，支持长度、重量、温度等",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "要转换的数值"
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "源单位，如km、miles、kg、pounds、celsius、fahrenheit"
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "目标单位"
                    }
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，如北京、上海、New York"
                    },
                    "unit": {
                        "type": "string",
                        "description": "温度单位，celsius或fahrenheit",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "查询两种货币之间的实时汇率",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {
                        "type": "string",
                        "description": "源货币代码，如USD、CNY、EUR"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "目标货币代码，如USD、CNY、EUR"
                    }
                },
                "required": ["from_currency", "to_currency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "将文本翻译成目标语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要翻译的文本"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "目标语言，如english、chinese、japanese、french"
                    }
                },
                "required": ["text", "target_language"]
            }
        }
    }
]

# =========================================================
# 解析 tool call
# =========================================================

def parse_tool_call(text):
    """
    从模型输出中解析:
    <tool_call>
    {
        ...
    }
    </tool_call>
    """
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None, False

    raw = matches[0].strip()
    try:
        data = json.loads(raw)
        return data, True
    except Exception:
        return raw, False


# =========================================================
# 参数比较（只要求 target_args 中的键值全部匹配）
# =========================================================

def compare_args(pred_args, gt_args):
    """
    检查 target_args 里的每个必填参数是否在 pred_args 中出现且值一致。
    数值比较容忍 int/float 差异，字符串忽略大小写。
    """
    if not isinstance(pred_args, dict):
        return False

    for k, v in gt_args.items():
        if k not in pred_args:
            return False

        pred_val = pred_args[k]
        gt_val = v

        # 尝试按数值比较
        try:
            if abs(float(pred_val) - float(gt_val)) > 1e-6:
                return False
        except (ValueError, TypeError):
            # 不是数值，按字符串比较
            if str(pred_val).strip().lower() != str(gt_val).strip().lower():
                return False
    return True


# =========================================================
# 推理函数
# =========================================================

@torch.no_grad()
def generate_response(model, tokenizer, messages, tools, device="cuda", max_new_tokens=256):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,          # temperature=0 在某些模型上会报错，用 1.0 + do_sample=False 等效贪心
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return response


# =========================================================
# 主评测逻辑
# =========================================================

def evaluate(args):
    device = args.device

    print("=" * 60)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model config...")
    config = PocketLLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )

    print("Loading model weights...")
    model = PocketLLMForCausalLM(config)
    ckpt = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()
    print("Model loaded.\n" + "=" * 60)

    # 读取测试数据
    samples = []
    with open(args.test_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} test samples.")

    # 统计变量
    total = 0
    format_correct = 0
    tool_correct = 0
    args_correct = 0
    failed_cases = []
    success_cases = []

    for sample in tqdm(samples, desc="Evaluating"):
        total += 1
        messages = sample["messages"]
        gt_tool = sample["target_tool"]
        gt_args = sample["target_args"]

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            tools=TOOLS,
            device=device,
            max_new_tokens=args.max_new_tokens
        )

        parsed, format_ok = parse_tool_call(response)
        if format_ok:
            format_correct += 1

        pred_tool = None
        pred_args = None
        if format_ok and isinstance(parsed, dict):
            pred_tool = parsed.get("name", None)
            pred_args = parsed.get("arguments", {})

        tool_ok = False
        if pred_tool == gt_tool:
            tool_correct += 1
            tool_ok = True

        args_ok = False
        if tool_ok and compare_args(pred_args, gt_args):
            args_correct += 1
            args_ok = True

# ... existing code ...
        if not (format_ok and tool_ok and args_ok):
            failed_cases.append({
                "query": messages[-1]["content"],
                "ground_truth_tool": gt_tool,
                "pred_tool": pred_tool,
                "ground_truth_args": gt_args,
                "pred_args": pred_args,
                "response": response
            })
        else:
            success_cases.append({
                "query": messages[-1]["content"],
                "ground_truth_tool": gt_tool,
                "pred_tool": pred_tool,
                "ground_truth_args": gt_args,
                "pred_args": pred_args,
                "response": response
            })

    # 计算指标
    format_acc = 100.0 * format_correct / total
    tool_acc = 100.0 * tool_correct / total
    args_acc = 100.0 * args_correct / total

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total Samples           : {total}")
    print(f"Format Accuracy         : {format_acc:.2f}%")
    print(f"Tool Selection Accuracy : {tool_acc:.2f}%")
    print(f"Argument Accuracy       : {args_acc:.2f}%")
    print("=" * 60)

    # 保存失败案例
    os.makedirs(args.output_dir, exist_ok=True)
    failed_path = os.path.join(args.output_dir, "failed_cases.json")
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed_cases, f, ensure_ascii=False, indent=2)
    print(f"Failed cases saved to: {failed_path}")

    # 保存成功案例
    success_path = os.path.join(args.output_dir, "success_cases.json")
    with open(success_path, "w", encoding="utf-8") as f:
        json.dump(success_cases, f, ensure_ascii=False, indent=2)
    print(f"Success cases saved to: {success_path}")

    # 保存整体结果
    results = {
        "model_weight": args.weight_path,
        "total_samples": total,
        "format_accuracy": format_acc,
        "tool_accuracy": tool_acc,
        "argument_accuracy": args_acc,
        "failed_count": len(failed_cases)
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Summary results saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PocketLLM ToolCall Benchmark")
    # 模型
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument("--weight_path", type=str, default=r"F:\pocket\PocketLLM-master\out\merge_lora_toolcall_768_moe.pth", required=False)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=1)

    # 数据
    parser.add_argument("--test_path", type=str, default=r"F:\pocket\PocketLLM-master\test.jsonl")
    parser.add_argument("--output_dir", type=str, default="./eval_results")

    # 推理
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()
    evaluate(args)