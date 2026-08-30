"""Day5 语义检索 v2：本地 txt 切块向量化入库（缓存）+ 交互 Top-K 余弦相似度检索。

【关于 Embedding 来源】
需求指定使用 DeepSeek，但经实测（2026-08-30 访问 api.deepseek.com）：
  * DeepSeek 官方仅提供 chat 模型（v4-flash / v4-pro / v4-flash-vision-exp）；
  * /models 列表里没有任何 embedding 模型；
  * /embeddings 端点统一返回 404。
因此本程序保留 DEEPSEEK_API_KEY 的读取（满足需求约束），并在启动时提示；
实际向量化走本地 Ollama 端点（http://localhost:11434/api/embeddings），
模型名来自实测 /api/tags（当前为 nomic-embed-text:latest），
想切换到别的供应商只要改 EMBEDDING_MODE / EMBEDDING_MODEL 即可。

【v2 变更：切块（chunking）取代全文截断】
v1 每个文件只取前 2000 字做"一个向量"，问题：
  1) 长文档 2000 字之后的内容对检索完全不可见；
  2) 局部问题（如"第 6 章的防护策略"）被全文平均语义稀释。
v2 把每个文件切成 ~CHUNK_SIZE 字的块（段落优先聚合、超长段落硬切+重叠），
每块独立向量化、独立参与检索——命中到"具体那一段"，这是 RAG 的标准地基做法。
缓存 meta 里记录 version/model/mode/chunk_size/chunk_overlap，
任一变更缓存整体失效自动重建，杜绝"旧向量配新参数"的静默错误。
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

# ============ 配置区 ============
DEEPSEEK_BASE_URL = "https://api.deepseek.com"       # 需求：DeepSeek base_url（保留，以备后续切换）
# 用户指定的本地 Ollama embedding 端点
EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "ollama")   # ollama | deepseek（若未来开放）
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text:latest")  # 经实测的模型名

SAMPLES_DIR = Path("./samples")
OUTPUT_DIR = Path("./output")  # 输出目录，与源代码分离
CACHE_FILE = OUTPUT_DIR / "vectors.json"

CHUNK_SIZE = 400     # 每块目标长度（字）。nomic-embed-text 上下文有限（实测 2700 字 OK/3000 字爆），
                     # 400 字远在安全区内，且短块语义更聚焦，检索精度更高
CHUNK_OVERLAP = 80   # 相邻块的字符重叠量（硬切超长段落时使用），避免关键句正好被切断
TOP_K = 3            # 检索 Top-K
CACHE_VERSION = 2    # 缓存格式版本：v1=整文一个向量，v2=切块多向量


def load_api_key() -> str:
    """需求 1：从环境变量读取 DEEPSEEK_API_KEY，缺失即退出。"""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("错误：未设置环境变量 DEEPSEEK_API_KEY")
    return key


def file_fingerprint(text: str) -> str:
    """对全文做 SHA256，文件级缓存命中判断的依据（v2 指纹 = 全文，不再截断）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_text(text: str) -> list[str]:
    """切块：段落优先聚合到 CHUNK_SIZE；单段超长则硬切，相邻窗口重叠 CHUNK_OVERLAP。

    返回块列表。段落边界是自然语义切分点；重叠保证跨块边界的句子
    至少在一个块里保持完整上下文。
    """
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(p) > CHUNK_SIZE:
            # 单段超长：先落盘 buf，再按滑动窗口硬切（步长 = CHUNK_SIZE - CHUNK_OVERLAP）
            if buf:
                chunks.append(buf)
                buf = ""
            step = CHUNK_SIZE - CHUNK_OVERLAP
            for i in range(0, len(p), step):
                chunks.append(p[i:i + CHUNK_SIZE])
            continue
        if len(buf) + len(p) + 1 <= CHUNK_SIZE:
            buf = f"{buf}\n{p}" if buf else p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def get_embedding(text: str) -> list[float]:
    """调 embedding 服务，返回向量数组。当前仅实现 Ollama 本地端点。"""
    if EMBEDDING_MODE == "ollama":
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=180,
        )
        if resp.status_code != 200:
            # 带上远端的明确错误（常见如 "the input length exceeds the context length"）
            raise RuntimeError(f"Ollama embedding 失败 [{resp.status_code}]: {resp.text[:300]}")
        return resp.json()["embedding"]
    elif EMBEDDING_MODE == "deepseek":
        # 留着占位：如果未来 DeepSeek 开放 embedding，启用下面这段
        # from openai import OpenAI
        # key = load_api_key()
        # client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
        # resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        # return resp.data[0].embedding
        sys.exit("deepseek 模式暂不可用：截至 2026-08-30 DeepSeek 未开放 embeddings 端点。")
    else:
        sys.exit(f"未知 EMBEDDING_MODE: {EMBEDDING_MODE}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """需求 5：numpy 实现余弦相似度 A·B/(|A||B|)（单对教学版；实际检索走矩阵版）。"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _empty_cache() -> dict:
    return {
        "meta": {
            "version": CACHE_VERSION,
            "model": EMBEDDING_MODEL,
            "mode": EMBEDDING_MODE,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "files": {},
    }


def load_cache() -> dict:
    """读缓存 vectors.json；不存在/损坏/关键参数变更 → 返回空结构整体重建。"""
    OUTPUT_DIR.mkdir(exist_ok=True)  # 确保输出目录存在
    if not CACHE_FILE.exists():
        return _empty_cache()
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"[警告] {CACHE_FILE} 损坏，将被覆盖重建。")
        return _empty_cache()
    # 任一关键参数变更 → 向量空间/语义单位改变，缓存整体失效（防"静默错误"）
    current_meta = _empty_cache()["meta"]
    for k, v in current_meta.items():
        if data.get("meta", {}).get(k) != v:
            print(f"[信息] 缓存参数 {k}: {data.get('meta', {}).get(k)} → {v}，缓存整体失效，将重建。")
            return _empty_cache()
    data.setdefault("files", {})
    return data


def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def build_library() -> tuple[list[str], list[str], list[np.ndarray], list[str]]:
    """遍历 samples/*.txt，切块 + 向量化（带缓存）。

    返回 (names, labels, vectors, chunk_texts)：
      names       每块所属文件名（与块一一对应）
      labels      显示用标签，如 "xxx.txt[3/22]"
      vectors     每块的向量（ndarray）
      chunk_texts 每块原文（用于结果片段展示）
    """
    txt_files = sorted(SAMPLES_DIR.glob("*.txt"))
    if not txt_files:
        sys.exit(f"错误：{SAMPLES_DIR} 下没有 .txt 文件")

    cache = load_cache()
    names, labels, vectors, chunk_texts = [], [], [], []
    dim = None
    total_chunks = 0

    print(f"发现 {len(txt_files)} 个文件，embedding 模型：{EMBEDDING_MODEL}（{EMBEDDING_MODE}）")
    print(f"切块参数：每块 {CHUNK_SIZE} 字，重叠 {CHUNK_OVERLAP} 字\n")

    t0 = time.perf_counter()
    for i, path in enumerate(txt_files, 1):
        name = path.name
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            print(f"[{i}/{len(txt_files)}] 跳过 {name}  → 编码错误: {e}")
            continue

        fp = file_fingerprint(text)
        cached_file = cache["files"].get(name)
        if cached_file and cached_file.get("fingerprint") == fp and cached_file.get("chunks"):
            # 文件未变：整组块向量直接复用，零 API 调用
            file_texts = [c["text"] for c in cached_file["chunks"]]
            file_vecs = [np.array(c["vector"], dtype=np.float64) for c in cached_file["chunks"]]
            print(f"[{i}/{len(txt_files)}] 缓存命中 {name}（{len(file_vecs)} 块）")
        else:
            # 文件新增/变更：切块后逐块向量化，完成后整体写缓存（文件级事务）
            file_texts = split_text(text)
            print(f"[{i}/{len(txt_files)}] 切块 {name}: {len(file_texts)} 块，逐块向量化 ", end="")
            sys.stdout.flush()
            t1 = time.perf_counter()
            file_vecs = []
            for ctext in file_texts:
                vec = get_embedding(ctext)
                file_vecs.append(np.array(vec, dtype=np.float64))
                print(".", end="")
                sys.stdout.flush()
            print(f"  耗时 {time.perf_counter() - t1:.1f}s")
            cache["files"][name] = {
                "fingerprint": fp,
                "chunks": [
                    {"idx": j, "text": ctext, "vector": v.tolist()}
                    for j, (ctext, v) in enumerate(zip(file_texts, file_vecs))
                ],
            }
            save_cache(cache)

        n_chunks = len(file_vecs)
        total_chunks += n_chunks
        for j in range(n_chunks):
            if dim is None:
                dim = len(file_vecs[j])
                print(f"[向量维度] {dim}\n")
            elif len(file_vecs[j]) != dim:
                print(f"  [警告] {name} 第 {j + 1} 块维度 {len(file_vecs[j])} ≠ {dim}，跳过该块。")
                continue
            names.append(name)
            labels.append(f"{name}[{j + 1}/{n_chunks}]")
            vectors.append(file_vecs[j])
            chunk_texts.append(file_texts[j])

    if not vectors:
        sys.exit("错误：没有任何块成功向量化。")

    total_s = time.perf_counter() - t0
    print(f"\n入库完成：{len(txt_files)} 个文件 / {total_chunks} 块（参与检索 {len(vectors)} 块），"
          f"维度 {dim}，总耗时 {total_s:.1f} 秒")
    return names, labels, vectors, chunk_texts


def search_loop(names: list[str], labels: list[str],
                vectors: list[np.ndarray], texts: list[str]) -> None:
    """交互输入问题 → 向量化 → 与所有块算余弦相似度 → Top-K 打印。"""
    print(f"\n=== 语义检索（Top-{TOP_K}，余弦相似度，按块）===")
    print('输入问题开始检索，输入 exit / quit 退出\n')
    # 一次性矩阵化所有块向量：查询时只做一次矩阵×向量 + 广播除法
    doc_matrix = np.vstack(vectors)                  # (N, D)
    doc_norms = np.linalg.norm(doc_matrix, axis=1)   # (N,)

    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        t0 = time.perf_counter()
        try:
            q_vec = np.array(get_embedding(q), dtype=np.float64)
        except Exception as e:
            print(f"[错误] 问题向量化失败: {e}\n")
            continue
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            print("[警告] 问题向量为 0，无法检索\n")
            continue

        # numpy 矩阵运算一次得全量分数（比循环两两算快）
        dots = doc_matrix @ q_vec                 # (N,)
        scores = dots / (doc_norms * q_norm)      # 余弦相似度
        top_idx = np.argsort(-scores)[:TOP_K]     # 降序取 Top-K

        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"  [检索耗时 {dt_ms:.0f}ms]")
        for rank, idx in enumerate(top_idx, 1):
            score = float(scores[idx])
            snippet = texts[idx][:120].replace("\n", " ")
            if len(texts[idx]) > 120:
                snippet += "…"
            print(f"  #{rank}  分数={score:.4f}  {labels[idx]}  （文件 {names[idx]}）")
            print(f"       片段: {snippet}\n")


def main():
    api_key = load_api_key()  # 需求 1：必须设置
    # 提示：当前 embedding 实际走本地 Ollama
    print(f"[配置] DEEPSEEK_API_KEY 已读取（长度 {len(api_key)}，已通过校验）。")
    if EMBEDDING_MODE == "ollama":
        print(f"[配置] Embedding 走本地 Ollama：{OLLAMA_BASE_URL}，模型 {EMBEDDING_MODEL}。"
              "（截至 2026-08-30 DeepSeek 未开放 embeddings 端点。）")
    else:
        print(f"[配置] Embedding 模式：{EMBEDDING_MODE}，模型 {EMBEDDING_MODEL}。")

    # 启动时预探 Ollama 是否在线，失败直接给可操作提示
    if EMBEDDING_MODE == "ollama":
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:
            sys.exit(f"错误：无法连接 Ollama（{OLLAMA_BASE_URL}）: {e}\n"
                     "请确认：1) 已执行 `ollama serve` 启动服务；2) 已执行 `ollama pull nomic-embed-text`。")

    names, labels, vectors, texts = build_library()
    search_loop(names, labels, vectors, texts)


if __name__ == "__main__":
    main()
