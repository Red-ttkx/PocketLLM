"""
generate_toolcall_sft.py
生成 ToolCall LoRA 训练数据（适配 train_lora.py conversations 格式）
单工具调用 + 少量普通负例，总量 400+ 条
"""

import json
import random
import math

# =========================================================
# Tool Definitions (与 eval_toolcall.py 保持一致)
# =========================================================
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    {"type": "function", "function": {"name": "text_length", "description": "统计文本长度", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "获取汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "文本翻译", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}}
]

# =========================================================
# Mock Data (大幅扩充)
# =========================================================
WEATHER_DATA = {
    "北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"),
    "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"),
    "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"),
    "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"),
    "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗"),
    "Dubai": ("38°C", "炎热"), "Moscow": ("-5°C", "雪"), "Toronto": ("-2°C", "阴"),
    "Seoul": ("18°C", "晴"), "Bangkok": ("33°C", "多云")
}

TIME_DATA = {
    "Asia/Shanghai": "2025-03-07 14:30:00", "Asia/Tokyo": "2025-03-07 15:30:00",
    "Europe/London": "2025-03-07 06:30:00", "America/New_York": "2025-03-07 01:30:00",
    "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00",
    "Asia/Dubai": "2025-03-07 10:30:00", "America/Chicago": "2025-03-07 00:30:00",
    "Asia/Seoul": "2025-03-07 15:30:00"
}

EXCHANGE_DATA = {
    ("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12,
    ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79,
    ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72, ("CAD", "CNY"): 5.30,
    ("KRW", "CNY"): 0.0054, ("SGD", "CNY"): 5.30, ("USD", "JPY"): 145.0,
    ("EUR", "USD"): 1.09, ("GBP", "USD"): 1.27
}

TRANSLATE_DATA = {
    ("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好",
    ("今天天气真好", "english"): "The weather is nice today",
    ("I love programming", "chinese"): "我喜欢编程",
    ("机器学习很有趣", "english"): "Machine learning is interesting",
    ("Happy birthday", "chinese"): "生日快乐", ("再见", "english"): "Goodbye",
    ("谢谢", "english"): "Thank you", ("对不起", "english"): "Sorry",
    ("今天吃什么", "english"): "What to eat today", ("明天见", "english"): "See you tomorrow",
    ("我来自中国", "english"): "I come from China",
    ("深度学习", "english"): "Deep Learning",
    ("自然语言处理", "english"): "Natural Language Processing"
}

UNIT_DATA = {
    "km_miles": 0.621371, "miles_km": 1.60934,
    "kg_pounds": 2.20462, "pounds_kg": 0.453592,
    "meters_feet": 3.28084, "feet_meters": 0.3048,
    "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556,
    "cm_inches": 0.393701, "inches_cm": 2.54,
    "km_meters": 1000, "meters_km": 0.001
}

# =========================================================
# Helpers
# =========================================================

# =========================================================
# 新增：生成与推理时一致的 system 提示
# =========================================================
def get_system_content():
    """
    返回与推理时 apply_chat_template(tools=TOOLS) 完全一致的 system 消息。
    内容来自实际抓取结果，必须原样保留。
    """
    return """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}}
{"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间，支持指定时区", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区名称，如Asia/Shanghai、America/New_York", "default": "Asia/Shanghai"}}, "required": []}}}
{"type": "function", "function": {"name": "random_number", "description": "生成指定范围内的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值", "default": 0}, "max": {"type": "integer", "description": "最大值", "default": 100}}, "required": []}}}
{"type": "function", "function": {"name": "text_length", "description": "计算文本的字符数和单词数", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要统计的文本"}}, "required": ["text"]}}}
{"type": "function", "function": {"name": "unit_converter", "description": "进行单位换算，支持长度、重量、温度等", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位，如km、miles、kg、pounds、celsius、fahrenheit"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}}
{"type": "function", "function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海、New York"}, "unit": {"type": "string", "description": "温度单位，celsius或fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}}
{"type": "function", "function": {"name": "get_exchange_rate", "description": "查询两种货币之间的实时汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码，如USD、CNY、EUR"}}, "required": ["from_currency", "to_currency"]}}}
{"type": "function", "function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如english、chinese、japanese、french"}}, "required": ["text", "target_language"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def build_toolcall(name, args):
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        + "\n</tool_call>"
    )

def generate_samples():
    samples = []
    system_msg = {"role": "system", "content": get_system_content()}

    # ---------- 天气 ----------
    weather_templates = [
        "{}今天温度多少", "{}天气怎么样", "帮我查一下{}天气", "{}现在天气如何",
        "{}的实时天气", "我想知道{}的天气", "{}外面热吗", "{}会不会下雨"
    ]
    cities = list(WEATHER_DATA.keys())
    for city in cities:
        for _ in range(4):
            q = random.choice(weather_templates).format(city)
            args = {"location": city}
            samples.append({
                "conversations": [
                    system_msg,
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": build_toolcall("get_current_weather", args)}
                ]
            })

    # ---------- 时间 ----------
    time_templates = [
        "{}现在几点", "查询{}时间", "{}当前时间",
        "告诉我{}的现在时间", "{}的时间", "{}现在是什么时候"
    ]
    tzones = list(TIME_DATA.keys())
    for tz in tzones:
        for _ in range(4):
            q = random.choice(time_templates).format(tz)
            args = {"timezone": tz}
            samples.append({
                "conversations": [
                    system_msg,
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": build_toolcall("get_current_time", args)}
                ]
            })

    # ---------- 数学 ----------
    math_exps = [
        "25*48", "128+256", "999-321", "12×8", "144÷12",
        "(15+25)*2", "3^4", "100/5", "7*9+12", "256-128+64",
        "100+200", "30*15", "200-55", "91/7", "16*16",
        "0.5*8", "529/23", "1.2^3", "sqrt(144)", "log10(100)",
        "55+77", "345+128", "800-333", "222*3", "64/4"
    ]
    math_templates = [
        "计算 {}", "{} 等于多少", "帮我算一下 {}", "数学题：{}",
        "{} 的结果", "请问 {}", "求 {} 的值", "{} 得数是多少"
    ]
    for exp in math_exps:
        for _ in range(3):
            q = random.choice(math_templates).format(exp)
            args = {"expression": exp}
            samples.append({
                "conversations": [
                    system_msg,
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": build_toolcall("calculate_math", args)}
                ]
            })

    # ---------- 随机数 ----------
    random_templates = [
        "生成一个{}到{}的随机数", "给我一个{}到{}的数字",
        "从{}到{}随机选一个", "{}~{}随机数",
        "在{}和{}之间随便给我个数", "{}到{}之间的随机整数"
    ]
    random_ranges = [
        (0, 100), (1, 50), (100, 200), (1, 1000), (50, 100),
        (20, 80), (500, 600), (1, 10), (200, 300)
    ]
    for mn, mx in random_ranges:
        for _ in range(3):
            q = random.choice(random_templates).format(mn, mx)
            args = {"min": mn, "max": mx}
            samples.append({
                "conversations": [
                    system_msg,
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": build_toolcall("random_number", args)}
                ]
            })

    # ---------- 文本长度 ----------
    text_len_templates = [
        "统计一下“{}”的字符数", "计算文本长度：{}",
        "帮我数一下“{}”的长度", "“{}”有多少个字和单词",
        "这段话有多少字符：{}"
    ]
    texts = [
        "hello world", "今天天气真好", "I love programming",
        "你好世界", "机器学习很有趣", "Good morning sunshine",
        "The quick brown fox jumps over the lazy dog",
        "深度学习是人工智能的一个重要分支",
        "人生苦短，我用Python", "I have a dream",
        "自然语言处理", "Transformer is great"
    ]
    for txt in texts:
        for _ in range(3):
            q = random.choice(text_len_templates).format(txt)
            args = {"text": txt}
            samples.append({
                "conversations": [
                    system_msg,
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": build_toolcall("text_length", args)}
                ]
            })

    # ---------- 单位换算 ----------
    unit_templates = [
        "{} {} 等于多少 {}", "把 {} {} 转换成 {}",
        "{} {} 换算成 {}", "{} {} 相当于多少 {}",
        "计算 {} {} 到 {}"
    ]
    unit_cases = [
        (3, "km", "miles"), (5, "miles", "km"), (10, "kg", "pounds"),
        (20, "pounds", "kg"), (100, "meters", "feet"), (300, "feet", "meters"),
        (30, "celsius", "fahrenheit"), (86, "fahrenheit", "celsius"),
        (1, "km", "miles"), (2, "kg", "pounds"),
        (50, "cm", "inches"), (12, "inches", "cm"),
        (15, "meters", "feet"), (1000, "meters", "km"),
        (0.5, "km", "miles"), (40, "miles", "km"),
        (7, "kg", "pounds"), (25, "pounds", "kg")
    ]
    for val, fu, tu in unit_cases:
        for _ in range(2):
            q = random.choice(unit_templates).format(val, fu, tu)
            args = {"value": val, "from_unit": fu, "to_unit": tu}
            samples.append({
                "conversations": [
                    system_msg,
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": build_toolcall("unit_converter", args)}
                ]
            })

    # ---------- 汇率 ----------
    exchange_templates = [
        "{}兑{}汇率是多少", "查询{}和{}汇率", "{}转换{}汇率",
        "{}对{}的汇率", "{}换{}是多少", "{}兑{}"
    ]
    for f, t in EXCHANGE_DATA.keys():
        for _ in range(3):
            q = random.choice(exchange_templates).format(f, t)
            args = {"from_currency": f, "to_currency": t}
            samples.append({
                "conversations": [
                    system_msg,
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": build_toolcall("get_exchange_rate", args)}
                ]
            })

    # ---------- 翻译 ----------
    translate_templates = [
        "把“{}”翻译成{}", "翻译：{} 到 {}", "{} 用{}怎么说",
        "将“{}”翻译为{}", "“{}”的{}翻译", "请将“{}”译成{}"
    ]
    for (text, lang) in TRANSLATE_DATA.keys():
        for _ in range(3):
            q = random.choice(translate_templates).format(text, lang)
            args = {"text": text, "target_language": lang}
            samples.append({
                "conversations": [
                    system_msg,
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": build_toolcall("translate_text", args)}
                ]
            })

    # ---------- 负例（同样带上 system 消息）----------
    chat_samples = [
        ("你好呀", "你好！有什么可以帮你的吗？"),
        ("讲个笑话", "为什么数学书总是很忧伤？因为它有太多的问题。"),
        ("介绍一下北京", "北京是中国的首都，拥有悠久历史。"),
        ("什么是AI", "AI是人工智能。"),
        ("谢谢", "不客气。"),
        ("再见", "再见，祝你开心。"),
        ("今天心情不好", "很抱歉听到这个，想聊聊发生了什么吗？"),
        ("推荐一道菜", "你可以试试宫保鸡丁，非常经典。"),
        ("What is AI?", "AI, or Artificial Intelligence, is the simulation of human intelligence in machines."),
        ("世界最高峰是什么", "世界最高峰是珠穆朗玛峰。"),
        ("python是什么", "Python是一种广泛使用的高级编程语言。"),
        ("给我讲个故事", "从前有座山，山上有座庙。"),
        ("现在是什么季节", "现在是春天。"),
        ("你喜欢什么颜色", "我没有偏好，但可以帮你选。")
    ]
    for q, ans in chat_samples:
        for _ in range(2):
            samples.append({
                "conversations": [
                    system_msg,
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": ans}
                ]
            })

    return samples


if __name__ == "__main__":
    random.seed(42)
    data = generate_samples()
    print(f"生成完成，共 {len(data)} 条训练数据。")
    output_path = "toolcall_sft.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已保存至 {output_path}")