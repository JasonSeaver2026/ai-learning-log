"""Day4 批量处理：遍历 ./samples/*.txt → DeepSeek 摘要/关键词/标签 → 汇总导出 JSON + MD 报告。"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.2          # 结构化输出必须低温
MAX_CHARS = 3000           # 超长文件截断，避免单次 token 过多
SAMPLES_DIR = Path("./samples")
OUTPUT_DIR = Path("./output")  # 输出目录，与源代码分离

SYSTEM_PROMPT = (
    "你是摘要助手。始终以 JSON 格式返回，固定三个字段："
    "summary（字符串，一句话摘要）、keywords（字符串数组，5个关键词）、tags（字符串数组，3个分类标签）。"
    "不要输出 JSON 以外的任何内容，不要使用 markdown 代码块。"
)


def analyze_text(client: OpenAI, text: str) -> tuple[dict, int, int]:
    """调用 DeepSeek 分析单段文本，返回 (解析结果, prompt_tokens, completion_tokens)。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请分析以下文本：\n\n{text}"},
        ],
        temperature=TEMPERATURE,
        # 强制 JSON 模式：服务端保证语法合法，字段正确性仍需解析后校验
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    usage = resp.usage
    result = json.loads(content)  # 解析失败会抛 JSONDecodeError，由调用方捕获记录
    parsed = {
        "summary": result.get("summary", "(缺失)"),
        "keywords": result.get("keywords", []),
        "tags": result.get("tags", []),
    }
    return parsed, usage.prompt_tokens, usage.completion_tokens


def export_json(results: list) -> None:
    with open(OUTPUT_DIR / "day4_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def export_md(results: list, failures: list, elapsed: float,
              prompt_tokens: int, completion_tokens: int) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Day4 批量摘要报告",
        "",
        f"- 生成时间：{now}",
        f"- 总耗时：{elapsed:.1f} 秒",
        f"- 处理结果：成功 {len(results)} 个，失败 {len(failures)} 个",
        f"- Token 消耗：prompt={prompt_tokens}，completion={completion_tokens}，"
        f"total={prompt_tokens + completion_tokens}",
        "",
    ]
    for i, item in enumerate(results, 1):
        lines += [
            f"## {i}. {item['file']}",
            "",
            f"**摘要**：{item['summary']}",
            "",
            f"**关键词**：{', '.join(item['keywords'])}",
            "",
            f"**标签**：{', '.join(item['tags'])}",
            "",
        ]
    if failures:
        lines += ["## 失败列表", ""]
        lines += [f"- {name}：{reason}" for name, reason in failures]
    with open(OUTPUT_DIR / "day4_result.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("错误：未设置环境变量 DEEPSEEK_API_KEY")

    txt_files = sorted(SAMPLES_DIR.glob("*.txt"))
    if not txt_files:
        sys.exit(f"错误：{SAMPLES_DIR} 目录下没有 .txt 文件")
    print(f"发现 {len(txt_files)} 个待处理文件\n")

    OUTPUT_DIR.mkdir(exist_ok=True)  # 输出目录不存在时自动创建

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    results, failures = [], []
    total_prompt, total_completion = 0, 0
    t0 = time.perf_counter()

    for i, path in enumerate(txt_files, 1):
        name = path.name
        try:
            text = path.read_text(encoding="utf-8")
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS] + "\n…(超长已截断)"
            parsed, pt, ct = analyze_text(client, text)
            total_prompt += pt
            total_completion += ct
            results.append({"file": name, **parsed})
            print(f"[{i}/{len(txt_files)}] 成功 {name}  "
                  f"tokens: prompt={pt} completion={ct}")
        except json.JSONDecodeError as e:
            failures.append((name, f"JSON 解析失败: {e}"))
            print(f"[{i}/{len(txt_files)}] 失败 {name}  原因: JSON 解析失败")
        except Exception as e:  # 网络/限流/API 错误都记录后继续，不中断整体
            failures.append((name, str(e)))
            print(f"[{i}/{len(txt_files)}] 失败 {name}  原因: {e}")
        if i < len(txt_files):
            time.sleep(1)  # 文件间间隔 1 秒，避免限流

    elapsed = time.perf_counter() - t0

    export_json(results)
    export_md(results, failures, elapsed, total_prompt, total_completion)

    print(f"\n===== 汇总 =====")
    print(f"总耗时: {elapsed:.1f} 秒 | 成功 {len(results)} | 失败 {len(failures)}")
    print(f"总 token: prompt={total_prompt} completion={total_completion} "
          f"total={total_prompt + total_completion}")
    print(f"已导出: {OUTPUT_DIR}/day4_result.json / {OUTPUT_DIR}/day4_result.md")


if __name__ == "__main__":
    main()
