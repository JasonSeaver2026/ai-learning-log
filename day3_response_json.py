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
    "2026年8月，某大型制造企业启动了AI转型项目，计划在未来两年内投入5000万元，"
    "用于建设企业级大模型平台。该项目将分三个阶段推进：第一阶段聚焦智能客服和知识库问答，"
    "预计三个月内上线；第二阶段将大模型能力接入生产排程和供应链预测系统；"
    "第三阶段探索多模态质检，用视觉模型替代部分人工目检环节。"
    "企业CTO表示，转型最大的挑战不是技术，而是员工的技能升级和组织流程的重塑。"
    "为此公司同步启动了全员AI素养培训，计划覆盖8000名员工。"
    "行业分析人士指出，制造业正在成为大模型落地最快的领域之一，"
    "2026年国内制造业AI渗透率预计将从15%提升到30%。"
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
