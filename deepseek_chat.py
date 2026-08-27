"""DeepSeek 多轮对话命令行程序：每次调用打印 token 消耗。"""
import os
import sys

import requests

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "你是一位耐心的 AI 导师，专门帮助有 10 年管理经验、正在转型 AI 应用工程师的学员。"
    "回答要通俗、多用类比、鼓励为主。"
)


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("错误：未设置环境变量 DEEPSEEK_API_KEY")

    headers = {"Authorization": f"Bearer {api_key}"}
    # messages 数组保存完整对话历史，System Prompt 固定在首位
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"模型: {MODEL} | temperature: {TEMPERATURE}（输入 exit 退出）")
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("再见！")
            break

        # 每轮把用户输入追加进历史，实现多轮上下文
        messages.append({"role": "user", "content": user_input})

        try:
            resp = requests.post(
                API_URL,
                headers=headers,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": TEMPERATURE,
                },
                timeout=60,
            )
        except requests.RequestException as e:
            print(f"网络请求失败: {e}")
            messages.pop()  # 失败时移除本轮输入，保持历史干净
            continue

        if resp.status_code != 200:
            print(f"请求失败 [{resp.status_code}]: {resp.text}")
            messages.pop()
            continue

        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        print(f"\n导师: {reply}")
        print(
            f"[tokens] prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')} "
            f"total={usage.get('total_tokens')}"
        )

        # 助手回复也加入历史，模型下一轮才能"记住"自己说过什么
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
