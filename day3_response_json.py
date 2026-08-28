"""DeepSeek 结构化输出示例：长文本 → 摘要/关键词/标签（强制 JSON 返回）。"""
import json
import os
import sys

import requests

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.2  # 结构化输出必须低温，降低格式漂移概率

SYSTEM_PROMPT = (
    "你是一个文本分析引擎。始终以 JSON 格式返回，固定三个字段："
    'summary（字符串，一句话摘要）、keywords（字符串数组，5个关键词）、tags（字符串数组，3个分类标签）。'
    "不要输出 JSON 以外的任何内容，不要使用 markdown 代码块。"
)

TEST_TEXT = (
    "文本 1：人工智能应用工程师是当下热门的岗位方向，主要职责是把大模型的能力通过 API、提示词工程、检索增强生成（RAG）和智能体（Agent）等技术，落地到具体的业务场景中，例如企业知识库、智能客服、文档摘要和工作流自动化。与传统算法岗不同，应用工程师更关注工程集成、成本控制与交付效率，而非从头训练模型。"
)


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("错误：未设置环境变量 DEEPSEEK_API_KEY")

    headers = {"Authorization": f"Bearer {api_key}"}

    resp = requests.post(
        API_URL,
        headers=headers,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请分析以下文本：\n\n{TEST_TEXT}"},
            ],
            "temperature": TEMPERATURE,
            # 关键参数：强制模型只输出合法 JSON 对象
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )

    if resp.status_code != 200:
        sys.exit(f"请求失败 [{resp.status_code}]: {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(f"[tokens] prompt={usage.get('prompt_tokens')} "
          f"completion={usage.get('completion_tokens')} "
          f"total={usage.get('total_tokens')}\n")

    # 模型仍可能偶发输出非 JSON 内容（如裹了 ```json 代码块），解析失败要给出清晰提示
    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        print("错误：模型返回内容不是合法 JSON，无法解析。")
        print(f"解析异常: {e}")
        print(f"原始返回内容:\n{content}")
        sys.exit(1)

    # 防御性取值：字段缺失时不崩，给出占位提示
    summary = result.get("summary", "(缺失)")
    keywords = result.get("keywords", [])
    tags = result.get("tags", [])

    print(f"摘要: {summary}")
    print(f"关键词: {', '.join(keywords) if keywords else '(缺失)'}")
    print(f"标签: {', '.join(tags) if tags else '(缺失)'}")


if __name__ == "__main__":
    main()
