import random
import re
import json
import os
import base64
from threading import Thread
from datetime import datetime

import torch
import numpy as np
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

st.set_page_config(page_title="PocketLLM", initial_sidebar_state="expanded", layout="wide")

# ================= 本地图片转 Base64 =================
def get_image_base64(filename):
    img_dir = r"F:\pocket\PocketLLM-master\images"
    path = os.path.join(img_dir, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    # 占位透明图
    return "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

logo_b64 = get_image_base64("logo.png")
banner_b64 = get_image_base64("顶部横幅.png")
avatar_b64 = get_image_base64("助手头像.png")
# =================================================

st.markdown("""
    <style>
        /* 全局字体 (Gemini 风格) */
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Google Sans', 'Noto Sans SC', sans-serif !important;
            color: #1f1f1f;
        }
        .stMainBlockContainer {
            padding-top: 1rem !important;
            padding-bottom: 5rem !important;
        }
        /* 历史对话按钮模拟 */
        .history-btn {
            background: transparent;
            border: none;
            color: #444;
            padding: 10px 15px;
            width: 100%;
            text-align: left;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            margin-bottom: 5px;
            transition: background 0.2s;
        }
        .history-btn:hover {
            background: #f0f4f9;
        }
        /* 滚动条 */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
        ::-webkit-scrollbar-track { background: transparent; }
    </style>
""", unsafe_allow_html=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 多语言文本 =================
LANG_TEXTS = {
    'zh': {
        'settings': '模型设定调整',
        'history_rounds': '历史对话轮次',
        'max_length': '最大生成长度',
        'temperature': '温度',
        'thinking': '✨ 开启深度思考',
        'tools': '工具选择 (最多4个)',
        'language': '语言',
        'send': '给 PocketLLM 发送消息...',
        'disclaimer': 'AI 生成内容可能存在错误，请仔细核实',
        'think_tip': '自适应思考，多轮对话或Tool Call时可能不稳定',
    },
    'en': {
        'settings': 'Model Settings',
        'history_rounds': 'History Rounds',
        'max_length': 'Max Length',
        'temperature': 'Temperature',
        'thinking': '✨ Enable Deep Thinking',
        'tools': 'Tool Selection (max 4)',
        'language': 'Language',
        'send': 'Message PocketLLM...',
        'disclaimer': 'AI-generated content may be inaccurate, please verify',
        'think_tip': 'Adaptive thinking; may be unstable with multi-turn or Tool Call',
    }
}

def get_text(key):
    lang = st.session_state.get('lang', 'zh')
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS['zh'].get(key, key))

# ================= 工具定义 & 执行 (完全保持原版) =================
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本长度", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位转换", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "获取汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
]

TOOL_SHORT_NAMES = {
    'calculate_math': '数学', 'get_current_time': '时间', 'random_number': '随机',
    'text_length': '字数', 'unit_converter': '单位', 'get_current_weather': '天气',
    'get_exchange_rate': '汇率', 'translate_text': '翻译'
}

def execute_tool(tool_name, args):
    import datetime
    try:
        if tool_name == 'calculate_math':
            return {"result": eval(args.get('expression', '0'))}
        elif tool_name == 'get_current_time':
            return {"result": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        elif tool_name == 'random_number':
            return {"result": random.randint(args.get('min', 0), args.get('max', 100))}
        elif tool_name == 'text_length':
            return {"result": len(args.get('text', ''))}
        elif tool_name == 'unit_converter':
            return {"result": f"{args.get('value', 0)} {args.get('from_unit', '')} = ? {args.get('to_unit', '')}"}
        elif tool_name == 'get_current_weather':
            return {"result": f"{args.get('city', 'Unknown')}: 晴, 7~10°C"}
        elif tool_name == 'get_exchange_rate':
            return {"result": f"1 {args.get('from_currency', 'USD')} = 7.2 {args.get('to_currency', 'CNY')}"}
        elif tool_name == 'translate_text':
            return {"result": f"[翻译结果]: hello world"}
        return {"result": "Unknown tool"}
    except Exception as e:
        return {"error": str(e)}

# ================= 原版 process_assistant_content (完全保持副本逻辑) =================
def process_assistant_content(content, is_streaming=False):
    # 处理tool_call标签，格式化显示
    if '<tool_call>' in content:
        def format_tool_call(match):
            try:
                tc = json.loads(match.group(1))
                name = tc.get('name', 'unknown')
                args = tc.get('arguments', {})
                return f'<div style="background: rgba(80, 110, 150, 0.20); border: 1px solid rgba(140, 170, 210, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalling</div><div><b>{name}</b>: {json.dumps(args, ensure_ascii=False)}</div></div>'
            except:
                return match.group(0)
        content = re.sub(r'<tool_call>(.*?)</tool_call>', format_tool_call, content, flags=re.DOTALL)
    
    # 流式生成且开启思考时，一开始就放到折叠里
    if is_streaming and st.session_state.get('enable_thinking', False) and '</think>' not in content and '<think>' not in content:
        m = re.search(r'(\n\n(?:我是|您好|你好)[^\n]*)', content)
        if m and m.start(1) > 5:
            i = m.start(1)
            think_part = content[:i]
            answer_part = content[i:]
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_part.strip()}</div></details>{answer_part}'
        elif len(content) > 5:
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{content.strip().replace(chr(10), "<br>")}</div></div></details>'

    if '<think>' in content and '</think>' in content:
        def format_think(match):
            think_content = match.group(2)
            if think_content.replace('\n', '').strip():
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(<think>)(.*?)(</think>)', format_think, content, flags=re.DOTALL)

    if '<think>' in content and '</think>' not in content:
        def format_think_in_progress(match):
            tc = match.group(1)
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{tc.strip().replace(chr(10), "<br>")}</div></div></details>'
        content = re.sub(r'<think>(.*?)$', format_think_in_progress, content, flags=re.DOTALL)

    if '<think>' not in content and '</think>' in content:
        def format_think_no_start(match):
            think_content = match.group(1)
            if think_content.replace('\n', '').strip():
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(.*?)</think>', format_think_no_start, content, flags=re.DOTALL)

    return content

# ================= 模型加载 =================
@st.cache_resource
def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.half().eval().to(device)
    return model, tokenizer

def clear_chat_messages():
    # 保存当前对话到历史（非空时）
    save_current_conversation()
    st.session_state.messages = []
    st.session_state.chat_messages = []

def save_current_conversation():
    """保存当前对话到历史记录列表中"""
    if "messages" not in st.session_state or not st.session_state.messages:
        return
    # 避免重复保存空对话
    if len(st.session_state.messages) == 0:
        return
    # 获取最后一条消息时间作为标题
    timestamp = datetime.now().strftime("%m-%d %H:%M")
    # 取用户的第一条消息作为预览
    preview = ""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            preview = msg["content"][:30]
            break
    title = f"{timestamp} - {preview}" if preview else timestamp
    
    # 保存到历史列表（最多20条）
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    # 避免与最后一条完全相同
    if st.session_state.conversation_history and st.session_state.conversation_history[-1]["messages"] == st.session_state.messages:
        return
    st.session_state.conversation_history.append({
        "title": title,
        "messages": st.session_state.messages.copy(),
        "chat_messages": st.session_state.chat_messages.copy()
    })
    # 保留最近20条
    if len(st.session_state.conversation_history) > 20:
        st.session_state.conversation_history = st.session_state.conversation_history[-20:]

def load_conversation(index):
    """加载指定索引的历史对话"""
    if 0 <= index < len(st.session_state.conversation_history):
        conv = st.session_state.conversation_history[index]
        st.session_state.messages = conv["messages"].copy()
        st.session_state.chat_messages = conv["chat_messages"].copy()
        st.rerun()

# 动态扫描模型目录
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {}
for d in sorted(os.listdir(script_dir), reverse=True):
    full_path = os.path.join(script_dir, d)
    if os.path.isdir(full_path) and not d.startswith('.') and not d.startswith('_'):
        if any(f.endswith(('.bin', '.safetensors', '.pt')) or os.path.exists(os.path.join(full_path, 'model.safetensors.index.json')) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))):
            MODEL_PATHS[d] = [d, d]
if not MODEL_PATHS:
    MODEL_PATHS = {"No models found": ["", "No models"]}

# ================= 侧边栏 UI (Logo 放大 + 深度思考独立开关) =================
with st.sidebar:
    # 侧边栏顶部：大 Logo + 标题
    st.markdown(f'<div style="text-align: center; margin-bottom: 20px;"><img src="{logo_b64}" style="width: 100px; border-radius: 12px;"></div>', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-top: -10px; margin-bottom: 20px;'>PocketLLM</h2>", unsafe_allow_html=True)
    
    # 新建对话按钮
    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
        clear_chat_messages()
        st.rerun()
    
    st.markdown("<div style='margin-top: 24px; font-size: 13px; color: #5f6368; font-weight: 500; margin-bottom: 8px;'>历史对话</div>", unsafe_allow_html=True)
    
    # 动态显示所有已保存的历史对话列表（不再显示独立的“之前的讨论...”按钮）
    if "conversation_history" in st.session_state and st.session_state.conversation_history:
        for idx, conv in enumerate(reversed(st.session_state.conversation_history)):
            # 倒序显示，最新的在上
            if st.button(f"📝 {conv['title']}", key=f"hist_{idx}", use_container_width=True):
                load_conversation(len(st.session_state.conversation_history) - 1 - idx)
    else:
        st.caption("暂无历史对话，新建对话后会自动保存")
    
    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
    
    # 深度思考开关作为独立显眼组件
    st.session_state.enable_thinking = st.toggle(get_text('thinking'), value=st.session_state.get('enable_thinking', False), help=get_text('think_tip'))
    
    st.markdown("<hr style='margin: 20px 0 12px 0;'>", unsafe_allow_html=True)
    
    # 设置收纳在 Popover
    with st.popover("⚙️ 设置", use_container_width=True):
        selected_model = st.selectbox('模型 (Model)', list(MODEL_PATHS.keys()), index=0)
        lang_options = {'中文': 'zh', 'English': 'en'}
        current_lang = st.session_state.get('lang', 'zh')
        lang_index = 0 if current_lang == 'zh' else 1
        lang_label = st.radio('语言 (Language)', list(lang_options.keys()), index=lang_index, horizontal=True)
        if lang_options[lang_label] != current_lang:
            st.session_state.lang = lang_options[lang_label]
            st.rerun()
        
        st.divider()
        st.session_state.history_chat_num = st.slider(get_text('history_rounds'), 0, 8, 0, step=2)
        st.session_state.max_new_tokens = st.slider(get_text('max_length'), 256, 8192, 8192, step=1)
        st.session_state.temperature = st.slider(get_text('temperature'), 0.6, 1.2, 0.90, step=0.01)
        
        st.divider()
        st.caption(get_text('tools'))
        st.session_state.selected_tools = []
        selected_count = sum(1 for tool in TOOLS if st.session_state.get(f"tool_{tool['function']['name']}", False))
        t_col1, t_col2 = st.columns(2)
        for i, tool in enumerate(TOOLS):
            name = tool['function']['name']
            short_name = TOOL_SHORT_NAMES.get(name, name)
            col = t_col1 if i % 2 == 0 else t_col2
            with col:
                checked = st.checkbox(short_name, key=f"tool_{name}", disabled=(selected_count >= 4 and not st.session_state.get(f"tool_{name}", False)))
                if checked and len(st.session_state.selected_tools) < 4:
                    st.session_state.selected_tools.append(name)

model_path = MODEL_PATHS[selected_model][0]
slogan = f"我是 {MODEL_PATHS[selected_model][1]}，有什么可以帮你的？" if st.session_state.get('lang', 'zh') == 'zh' else f"I am {MODEL_PATHS[selected_model][1]}, how can I help you?"

# ================= 渲染函数 (仅用于已完成消息，带左侧头像) =================
def render_user_msg(text):
    return f'<div style="display: flex; justify-content: flex-end; margin: 16px 0;"><div style="background-color: #f0f4f9; color: #1f1f1f; padding: 12px 20px; border-radius: 24px; font-size: 15px; max-width: 80%; line-height: 1.6;">{text}</div></div>'

def render_bot_msg(html_content):
    return f'<div style="display: flex; gap: 16px; margin: 20px 0;"><img src="{avatar_b64}" style="width: 34px; height: 34px; border-radius: 50%; object-fit: cover; border: 1px solid #eee;"><div style="color: #1f1f1f; font-size: 15px; padding-top: 4px; line-height: 1.7; flex: 1; overflow-x: auto;">{html_content}</div></div>'

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================= Main =================
def main():
    model, tokenizer = load_model_tokenizer(r'F:\pocket\PocketLLM-master\pout')

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    messages = st.session_state.messages

    # 主界面顶部：横幅图片（修复截断问题：使用 contain 确保完整显示）
    if banner_b64 and not banner_b64.endswith("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"):
        st.markdown(
            f'<div style="display: flex; justify-content: center; overflow: visible; padding-top: 18px;">'
            f'<img src="{banner_b64}" style="width: auto; max-width: 500px; max-height: 150px; height: auto; object-fit: contain; border-radius: 12px; margin-bottom: 20px;">'
            f'</div>',
            unsafe_allow_html=True
        )

    # 欢迎界面 (无历史消息时)
    if not messages:
        st.markdown(
            f'<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-top: 2rem;">'
            f'<h1 style="font-size: 28px; color: #202124; margin-bottom: 8px;">{slogan}</h1>'
            f'<p style="color: #5f6368; font-size: 14px;">{get_text("disclaimer")}</p>'
            '</div>',
            unsafe_allow_html=True
        )

    # 渲染历史消息 (带左侧头像)
    for message in messages:
        if message["role"] == "assistant":
            st.markdown(render_bot_msg(process_assistant_content(message["content"])), unsafe_allow_html=True)
        else:
            st.markdown(render_user_msg(message["content"]), unsafe_allow_html=True)

    # 输入框
    prompt = st.chat_input(key="input", placeholder=get_text('send'))

    if prompt:
        # 显示用户消息
        st.markdown(render_user_msg(prompt), unsafe_allow_html=True)
        messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})
        st.session_state.chat_messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})

        placeholder = st.empty()  # 用于流式输出

        random_seed = random.randint(0, 2 ** 32 - 1)
        setup_seed(random_seed)

        # 工具 & 系统提示 (与原版完全一致)
        tools = [t for t in TOOLS if t['function']['name'] in st.session_state.get('selected_tools', [])] or None
        sys_prompt = [] if tools else [{"role": "system", "content": "你是PocketLLM，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。"}]
        st.session_state.chat_messages = sys_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]

        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if st.session_state.get('enable_thinking', False):
            template_kwargs["open_thinking"] = True
        if tools:
            template_kwargs["tools"] = tools

        new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        # 流式生成 (完全原版方式，不套头像)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
            "num_return_sequences": 1,
            "do_sample": True,
            "attention_mask": inputs.attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": st.session_state.temperature,
            "top_p": 0.85,
            "streamer": streamer,
        }

        Thread(target=model.generate, kwargs=generation_kwargs).start()

        answer = ""
        for new_text in streamer:
            answer += new_text
            placeholder.markdown(process_assistant_content(answer, is_streaming=True), unsafe_allow_html=True)

        full_answer = answer

        # Tool call 循环 (与原版完全一致)
        for _ in range(16):
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', answer, re.DOTALL)
            if not tool_calls:
                break
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            tool_results = []
            for tc_str in tool_calls:
                try:
                    tc = json.loads(tc_str.strip())
                    result = execute_tool(tc.get('name', ''), tc.get('arguments', {}))
                    st.session_state.chat_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                    tool_results.append(f'<div style="background: rgba(90, 130, 110, 0.20); border: 1px solid rgba(150, 200, 170, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalled</div><div><b>{tc.get("name", "")}</b>: {json.dumps(result, ensure_ascii=False)}</div></div>')
                except:
                    pass
            full_answer += "\n" + "\n".join(tool_results) + "\n"
            placeholder.markdown(process_assistant_content(full_answer, is_streaming=True), unsafe_allow_html=True)

            new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["input_ids"] = inputs.input_ids
            generation_kwargs["attention_mask"] = inputs.attention_mask
            generation_kwargs["max_length"] = inputs.input_ids.shape[1] + st.session_state.max_new_tokens
            generation_kwargs["streamer"] = streamer

            Thread(target=model.generate, kwargs=generation_kwargs).start()
            answer = ""
            for new_text in streamer:
                answer += new_text
                placeholder.markdown(process_assistant_content(full_answer + answer, is_streaming=True), unsafe_allow_html=True)
            full_answer += answer

        answer = full_answer
        messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        
        # 注意：不再每轮对话后自动保存历史，仅在点击“新建对话”时保存

if __name__ == "__main__":
    main()
