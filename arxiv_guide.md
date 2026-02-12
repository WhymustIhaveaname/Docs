---
name: arxiv-search
description: 搜索和分析 arXiv 论文。当用户需要查找学术论文、搜索 arXiv、获取论文全文、被引反查或分析特定论文时使用此技能。
---

# arXiv 论文搜索与分析

此技能用于搜索、下载和分析 arXiv 上的学术论文。

## 使用时机

- 用户需要搜索 arXiv 论文
- 用户提到 arXiv ID（如 2401.12345）
- 用户需要获取论文全文或摘要
- 用户需要查找某领域的最新论文
- 用户想知道哪些论文引用了某篇论文（被引反查）

---

**运行方式**：Docs 下有 `.venv`，直接用：

```bash
/home/prime/Codes/Docs/.venv/bin/python /home/prime/Codes/Docs/arxiv_tool.py <子命令>
```

也可以用 `uv run`（会自动识别 `.venv`，不重复安装）：

```bash
uv run /home/prime/Codes/Docs/arxiv_tool.py <子命令>
```

### 搜索论文

```bash
uv run /home/prime/Codes/Docs/arxiv_tool.py search "关键词" --max 10
uv run /home/prime/Codes/Docs/arxiv_tool.py search "PINN" --categories cs.LG,physics.comp-ph
uv run /home/prime/Codes/Docs/arxiv_tool.py search "neural network" --sort relevance
```

参数：
- `--max`: 结果数量，默认 10
- `--sort`: 排序方式 (relevance, submitted, updated)
- `--categories`: 分类过滤，逗号分隔

### 获取论文全文

```bash
uv run /home/prime/Codes/Docs/arxiv_tool.py fetch 2401.12345
uv run /home/prime/Codes/Docs/arxiv_tool.py fetch 2401.12345 -o ./my_papers
```

- 下载 PDF 并转换为 txt（PDF 和 txt 都保留）
- 默认保存到 `/home/prime/Codes/Docs/arxiv/`
- 文件名格式: `{arxiv_id}.pdf` 和 `{arxiv_id}.txt`
- txt 包含元数据头 + 全文

### 获取论文信息（不下载）

```bash
uv run /home/prime/Codes/Docs/arxiv_tool.py info 2401.12345
```

### 生成 BibTeX 引用

```bash
uv run /home/prime/Codes/Docs/arxiv_tool.py bib 2505.08783
uv run /home/prime/Codes/Docs/arxiv_tool.py bib 2505.08783 -o references.bib  # 追加到文件
```

- 输出 arXiv 官方格式 BibTeX
- Citation key 格式: `{第一作者姓小写}{年份}{标题首实词}`（如 `li2025codepde`）
- `-o` 参数追加写入文件

### 下载 LaTeX 源文件

```bash
uv run /home/prime/Codes/Docs/arxiv_tool.py tex 2505.08783
```

- 下载 arXiv 源文件（tar.gz/gzip）并自动解压
- 保存到 `Docs/arxiv/{arxiv_id}_{title}/` 目录
- 输出目录树结构
- 保留完整 LaTeX 格式（含数学公式）和图片

### 被引反查

查看哪些论文引用了某篇论文。数据源：Semantic Scholar（首选）+ OpenAlex（备选）。

```bash
uv run /home/prime/Codes/Docs/arxiv_tool.py cited 1711.10561
uv run /home/prime/Codes/Docs/arxiv_tool.py cited 1711.10561 --max 50
uv run /home/prime/Codes/Docs/arxiv_tool.py cited 1711.10561 --offset 20        # 翻页：第 21-40 条
uv run /home/prime/Codes/Docs/arxiv_tool.py cited 1711.10561 --source openalex  # 强制用 OpenAlex
uv run /home/prime/Codes/Docs/arxiv_tool.py cited 1711.10561 --source s2        # 强制用 Semantic Scholar
```

参数：
- `--max`: 显示条数，默认 20
- `--offset`: 跳过前 N 条，用于翻页（默认 0）
- `--source`: 数据源选择 (auto, s2, openalex)，默认 auto（S2 优先，失败切 OpenAlex）

输出包含：论文标题、总被引次数、引用论文列表（标题、作者、年份、被引数）。

---

## 常用 arXiv 分类

| 分类 | 领域 |
|------|------|
| cs.AI | 人工智能 |
| cs.LG | 机器学习 |
| cs.CV | 计算机视觉 |
| cs.CL | 计算语言学/NLP |
| physics.comp-ph | 计算物理 |
| math.NA | 数值分析 |
| stat.ML | 统计机器学习 |

---

## 最佳实践

| 场景 | 命令 |
|------|------|
| 已知 arXiv ID，需要全文（纯文本） | `arxiv_tool.py fetch <ID>` |
| 已知 arXiv ID，需要 LaTeX 源码 | `arxiv_tool.py tex <ID>` |
| 已知 arXiv ID，只需摘要 | `arxiv_tool.py info <ID>` |
| 搜索某领域论文 | `arxiv_tool.py search "关键词"` |
| 生成 BibTeX 引用 | `arxiv_tool.py bib <ID>` |
| 生成引用到文件 | `arxiv_tool.py bib <ID> -o refs.bib` |
| 查看哪些论文引用了它 | `arxiv_tool.py cited <ID>` |
| 查看被引（翻页） | `arxiv_tool.py cited <ID> --offset 20` |
| 非 arXiv 论文 | 此工具不适用，用 curl 获取网页或让用户提供摘要 |

## 配置

API Key 存放在 `Docs/.env`（已 gitignore），脚本启动时自动加载：

```
S2_API_KEY=xxx         # Semantic Scholar（被引反查用）
OPENALEX_API_KEY=xxx   # OpenAlex（被引反查备选）
```

有 key 就用，没有也不影响基本功能（search/fetch/info/bib/tex 不需要 key）。

## 输出目录

全文 txt 和 PDF 保存位置: `/home/prime/Codes/Docs/arxiv/`
