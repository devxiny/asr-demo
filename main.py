import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import json
import requests
import re
import textwrap
import httpx
from pypinyin import pinyin, lazy_pinyin, Style

app = FastAPI()

# 创建一个全局的异步客户端
client = httpx.AsyncClient(timeout=30.0)


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()


@app.post("/ai/farming/sow_farrowing")
async def asr(audio_file: UploadFile = File(...)):
    try:
        # ASR请求
        asr_url = "http://10.1.11.201:50009/asr"
        asr_params = {"hotword": "健仔 弱仔 畸形 白仔 黑仔 窝总重"}
        audio_content = await audio_file.read()

        files = {
            "audio_file": (audio_file.filename, audio_content, audio_file.content_type)
        }
        asr_response = await client.post(
            asr_url, params=asr_params, files=files, timeout=20.0
        )
        if asr_response.status_code != 200:
            return JSONResponse(
                content={"error": f"ASR service error: {asr_response.status_code}"},
                status_code=500,
            )

        asr_text = convert_chinese_to_custom_text(asr_response.json()[0].get("text"))

        # LLM请求
        llm_url = "http://10.1.11.201:50002/v1/chat-messages"
        llm_headers = {
            "Authorization": "Bearer app-sH0fzgF8gfcHO2YzLFupxweN",
            "Content-Type": "application/json",
        }
        llm_data = {
            "inputs": {},
            "query": textwrap.dedent(
                f"""
                # 角色：你是语言专家，精通拼音与文字的互转。

                # 任务：
                1. 将提供的文字转成拼音
                2. 根据词表“健仔（jian zai）,弱仔（ruo zai）,畸形（ji xing）,白仔（bai zai）,黑仔（hei zai）,窝总重（wo zong zhong）”括号内的拼音，将步骤1中的拼音替换成词表对应的文字
                3. 将替换后的文字转成下方JSON格式返回，未提及的内容值设为-1
                JSON格式:
                {{
                "healthy_piglets": -1, // 健仔数量(整数)
                "weak_piglets": -1, // 弱仔数量(整数)
                "deformed_piglets": -1, // 畸形数量(整数)
                "white_piglets": -1, // 白仔数量(整数)
                "black_piglets": -1, // 黑仔数量(整数)
                "total_litter_weight": -1.0 // 窝总重(浮点数)
                }}

                # 例子：
                提供的文字：舰 载 三 个 若 在 五 个 急 性 四 个 我 总 中 六 十 公 斤 败 在 九 个 黑 再 十 个
                拼音：jian zai san ge ruo zai wu ge bai zai jiu ge hei zai shi ge ji xing si ge wo zong zhong liu shi gong jin
                替换规则：健仔（jian zai）,弱仔（ruo zai），白仔（bai zai），黑仔（hei zai）,畸形（ji xing）,窝总重（wo zong zhong）
                替换后：健仔三个，弱仔五个，白仔九个，黑仔十个，畸形四个，窝总重六十公斤
                转换后的JSON：
                {{
                "healthy_piglets": 3,
                "weak_piglets": 5,
                "deformed_piglets": 4,
                "white_piglets": 9,
                "black_piglets": 10,
                "total_litter_weight": 60.0
                }}

                提供的文字：建 在 一 个 弱 在 三 个 记 性 六 个 沃 宗 重 七 十 五 公 斤 摆 在 一 个 黑 仔 六 个
                拼音：jian zai yi ge ruo zai san ge ji xing liu ge wo zong zhong qi shi wu gong jin bai zai yi ge hei zai liu ge
                替换规则：健仔（jian zai）,弱仔（ruo zai）,畸形（ji xing）,窝总重（wo zong zhong），白仔（bai zai），黑仔（hei zai）
                替换后：健仔一个，弱仔三个，畸形六个，窝总重七十五公斤，白仔一个，黑仔六个
                转换后的JSON：
                {{
                "healthy_piglets": 1,
                "weak_piglets": 3,
                "deformed_piglets": 6,
                "white_piglets": 1,
                "black_piglets": 6,
                "total_litter_weight": 75.0
                }}

                让我们开始吧：
                提供的文字：{asr_text}
                """
            ),
            "response_mode": "blocking",
            "conversation_id": "",
            "user": "system",
            "files": [],
        }
        llm_response = await client.post(
            url=llm_url,
            headers=llm_headers,
            json=llm_data,
            timeout=20.0,  # 为LLM请求设置更长的超时时间
        )

        if llm_response.status_code != 200:
            return JSONResponse(
                content={"error": f"LLM service error: {llm_response.status_code}"},
                status_code=500,
            )

        answer = llm_response.json().get("answer")
        result = extract_json(answer)

        if result is None:
            return JSONResponse(
                content={"error": "Failed to parse LLM response"}, status_code=500
            )

        return JSONResponse(content=result)

    except httpx.TimeoutException:
        return JSONResponse(content={"error": "Request timeout"}, status_code=504)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Unexpected error: {str(e)}"}, status_code=500
        )


def extract_json(text):
    pattern = r"\{.*?\}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None


# 定义拼音到文本的映射
pinyin_to_text = {
    "jian zai": "健 仔",
    "ruo zai": "弱 仔",
    "ruo zi": "弱 仔",
    "wo zai": "弱 仔",
    "wo zi": "弱 仔",
    "ji xing": "畸 形",
    "bai zai": "白 仔",
    "hei zai": "黑 仔",
    "wo zong zhong": "窝 总 重",
    "yi": "一",
    "er": "二",
    "san": "三",
    "si": "四",
    "wu": "五",
    "liu": "六",
    "qi": "七",
    "ba": "八",
    "jiu": "九",
    "shi": "十",
    "ge": "个",
    "tou": "头",
    "zhi": "只",
    "gong jin": "公 斤",
    "qian ke": "千 克",
}


def convert_chinese_to_custom_text(input_string):
    # 移除特殊字符
    input_string = remove_special_characters(input_string)
    # 将中文转换为拼音
    pinyin_list = lazy_pinyin(input_string)

    # 将拼音列表转换为字符串
    pinyin_string = " ".join(pinyin_list)

    # 使用正则表达式替换匹配的拼音
    for pinyin, text in pinyin_to_text.items():
        pinyin_string = re.sub(r"\b" + pinyin + r"\b", text, pinyin_string)

    # 将未匹配的拼音替换回原中文字符
    result = []
    pinyin_words = pinyin_string.split()
    for i, word in enumerate(pinyin_words):
        if check_language(word) == "Chinese":
            result.append(word)
        else:
            result.append(input_string[i])

    return "".join(result)


import re


def check_language(text):
    # 检查是否包含中文字符
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"
    # 检查是否包含英文字符
    elif re.search(r"[a-zA-Z]", text):
        return "English"
    else:
        return "Unknown"


def remove_special_characters(text, keep_spaces=False):
    """
    移除文本中的特殊字符，保留中文、英文字母、数字。

    参数:
    text (str): 需要处理的输入文本
    keep_spaces (bool): 是否保留空格，默认为True

    返回:
    str: 处理后的文本
    """
    # 如果需要保留空格，使用第一个模式；否则使用第二个模式
    if keep_spaces:
        pattern = r"[^\u4e00-\u9fa5a-zA-Z0-9\s]"
    else:
        pattern = r"[^\u4e00-\u9fa5a-zA-Z0-9]"

    # 使用正则表达式替换特殊字符为空字符串
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=50011,
        timeout_keep_alive=20,
        limit_concurrency=10,
    )
