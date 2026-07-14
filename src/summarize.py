"""使用 DeepSeek / 兼容 OpenAI 的 API 对抓取的内容进行总结

架构说明（参考 follow-builders 的设计）：
  - prepare_digest()：纯数据整理，输出结构化 JSON，不调用 LLM
  - summarize()：将 JSON 送给 LLM 生成可读摘要
  两步分离，便于调试和复用。
"""

import json
import time
from openai import OpenAI, RateLimitError, BadRequestError, APIError
from src.fetchers.base import RawItem

_PROMPT_BASE = """
## 内容处理规则（按类型）

### 播客 / YouTube 字幕（source_type: youtube_transcript 或 follow_builders）
每集写一段 150-300 字的精华提炼：
- 第一句：「核心观点」——这集最重要的一个结论是什么？
- 介绍主讲人的身份背景（姓名、公司/职位）
- 提炼 2-3 个反直觉、具体、或有实操价值的洞见，避免泛泛而谈
- 引用一句原文中最有力的话（原文语言）
- 写法：像聪明的朋友在给你做口头总结，直接切入内容，不写「本期节目」「主持人问道」之类的废话

### Twitter/X 推文（source_type: follow_builders 且来自 @handle）
每个人写 2-4 句话：
- 先介绍此人身份：全名 + 职位/公司（如「Replit CEO Amjad Masad」）
- 只写实质性内容：原创观点、产品动态、技术讨论、行业判断
- 跳过：日常碎碎念、转发、「活动很棒！」类内容
- 如果有大胆预测或反主流观点，优先提
- 如果该人无实质内容，写「本期无实质更新」

### 文章 / RSS（source_type: rss 或 email）
每篇写：
- **核心论点**：一句话说清楚这篇在讲什么
- **关键要点**：如果正文内容充实（超过 500 字），列出 5-10 条 bullet points，每条一句话，聚焦具体观点、数据、结论或反直觉见解；内容较短则列 2-3 条即可，不要凑数
- **值不值得深读**：一句话说明理由
- 附原文链接

### 暂无更新的来源（title == "暂无更新"）
在「来源详情」中仍然列出该来源，写一句：「今日无新内容」。不要跳过，不要省略。
**将所有「今日无新内容」的来源统一列在「来源详情」最末尾，不要穿插在有内容的来源之间。**

## 强制要求
- 每一条有实质内容的条目都必须附原始链接；没有链接的不要写进摘要
- 暂无更新的来源：只写来源名称 + 「今日无新内容」，不需要链接
- 不要编造任何内容，只基于 JSON 中给出的数据
- 格式要适合手机阅读：段落之间空行，重点用加粗
- 不使用破折号（——除外）开头的列表
"""

_PROMPTS = {
    "zh": f"""你是一个国际新闻与科技信息助手。我会给你一段 JSON，包含从多个来源抓取的最新内容。
请全程使用中文输出，技术词汇（AI、LLM、API、quantum、space、genomics 等）保留英文原文，人名和产品名保留英文。

## 输出结构

### 今日要点
跨所有来源，按重要性排出 5 条最值得关注的内容，每条一段话（不是一句话），说清楚为什么重要。

### 来源详情
按来源逐一展开，每个来源单独一节，标题写来源名称。

{_PROMPT_BASE}""",

    "en": f"""You are an international news and science digest assistant. I will give you a JSON containing the latest content fetched from multiple sources.
Output entirely in English.

## Output Structure

### Today's Highlights
Pick the 5 most important items across all sources, ranked by importance. Each item gets a full paragraph (not a single sentence) explaining why it matters.

### Source Breakdown
Go through each source one by one, with its name as a section heading.

{_PROMPT_BASE}""",

    "bilingual": f"""You are an international news and science digest assistant. I will give you a JSON containing the latest content fetched from multiple sources.
Output in bilingual format: Chinese and English interleaved paragraph by paragraph.

Rules:
- Section headings in both languages: e.g. "## 今日要点 / Today's Highlights"
- For each item: write the Chinese paragraph first, then the English paragraph directly below (blank line between), then move to the next item
- Do NOT output all Chinese first then all English
- Technical terms (AI, LLM, API, quantum, space, genomics, etc.) stay in English even in Chinese paragraphs
- Proper nouns (people, companies, products) stay in English

## Output Structure

### 今日要点 / Today's Highlights
5 most important items, each gets a paragraph in both languages.

### 来源详情 / Source Breakdown
Each source as a separate section.

{_PROMPT_BASE}""",
}


def prepare_digest(items: list[RawItem]) -> dict:
    """将 RawItem 列表整理成结构化 JSON（不调用 LLM）。

    这一步是纯数据整理：按 source_type 分组，保留所有元数据。
    好处：可以单独调试、序列化存档、或送给任意 LLM 处理。
    """
    groups: dict[str, list[dict]] = {}
    for item in items:
        entry = {
            "source": item.source_name,
            "title": item.title,
            "content": item.content,
        }
        if item.link:
            entry["link"] = item.link
        if item.published:
            entry["published"] = item.published
        groups.setdefault(item.source_type, []).append(entry)
    return {"sources": groups, "total_items": len(items)}


def summarize(
    items: list[RawItem],
    api_key: str | None = None,
    model: str = "qwen3.6-flash-2026-04-16",
    language: str = "zh",
    base_url: str = "https://ws-ywg8kc6s0ma5m3se.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
) -> str:
    """调用 LLM (DeepSeek / OpenAI 兼容 API) 生成总结。

    先通过 prepare_digest() 整理成 JSON，再送给 LLM，
    让 LLM 只做「读 JSON → 写文章」一件事，数据与生成解耦。

    language: "zh"（中文）| "en"（英文）| "bilingual"（中英双语交错）
    """
    if not items:
        no_content = {"zh": "暂无新内容。", "en": "No new content.", "bilingual": "暂无新内容。/ No new content."}
        return no_content.get(language, "暂无新内容。")

    digest = prepare_digest(items)
    content = json.dumps(digest, ensure_ascii=False, indent=2)

    system_prompt = _PROMPTS.get(language, _PROMPTS["zh"])

    client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_api(payload: str) -> str:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=8192,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": payload},
                    ],
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                print(f"[LLM 429] attempt {attempt + 1}/3: {e}")
                if attempt < 2:
                    time.sleep(30 * (attempt + 1))
                    continue
                raise
            except APIError as e:
                print(f"[LLM API error] attempt {attempt + 1}/3: {e}")
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("unreachable")

    try:
        return _call_api(content)
    except RateLimitError:
        print("[LLM fallback] 三次重试后仍 429，返回占位符。")
        fallback = {
            "zh": "今日摘要生成失败：API 负载过高，请稍后重试。",
            "en": "Summary generation failed: API engine overloaded. Please try again later.",
            "bilingual": "今日摘要生成失败：API 负载过高。\nSummary generation failed: API engine overloaded.",
        }
        return fallback.get(language, fallback["zh"])
    except BadRequestError as e:
        if "content_filter" not in str(e) and "high risk" not in str(e):
            print(f"[警告] 调用 LLM 失败: {e}")
            fallback = {
                "zh": f"今日摘要生成失败：{e}",
                "en": f"Summary generation failed: {e}",
                "bilingual": f"今日摘要生成失败：{e}\nSummary generation failed: {e}",
            }
            return fallback.get(language, fallback["zh"])

        # 内容过滤，逐来源重试
        print("[警告] 全量内容被内容过滤器拦截，尝试按来源逐一重试...")
        by_source = {}
        for item in items:
            by_source.setdefault(item.source_name, []).append(item)

        passed_items: list[RawItem] = []
        skipped_sources: list[str] = []
        for source_name, source_items in by_source.items():
            test_digest = prepare_digest(source_items)
            test_content = json.dumps(test_digest, ensure_ascii=False, indent=2)
            try:
                _call_api(test_content)
                passed_items.extend(source_items)
            except BadRequestError:
                print(f"[警告] 跳过被过滤的来源: {source_name}")
                skipped_sources.append(source_name)

        if not passed_items:
            fallback = {
                "zh": "今日摘要生成失败：所有内容均被内容过滤器拦截。",
                "en": "Summary generation failed: all content was blocked by the content filter.",
                "bilingual": "今日摘要生成失败：所有内容均被内容过滤器拦截。\nSummary generation failed: all content was blocked.",
            }
            return fallback.get(language, fallback["zh"])

        clean_digest = prepare_digest(passed_items)
        clean_content = json.dumps(clean_digest, ensure_ascii=False, indent=2)
        result = _call_api(clean_content)
        if skipped_sources:
            note = "\n\n---\n⚠️ 以下来源因内容过滤被跳过：" + "、".join(skipped_sources)
            result += note
        return result
