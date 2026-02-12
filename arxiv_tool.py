#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "arxiv",
#     "pymupdf",
#     "python-dotenv",
#     "requests",
# ]
# ///
"""
arXiv 论文搜索与全文获取工具

功能：
1. search - 搜索论文（关键词、标题、摘要）
2. fetch - 获取论文全文（PDF 和 txt 都保留）
3. info - 获取论文信息（标题、作者、摘要等，不下载）
4. bib - 生成 BibTeX 引用（自动生成 citation key）
5. tex - 下载 LaTeX 源文件并解压
6. cited - 被引反查（Semantic Scholar 首选，OpenAlex 备选）

使用方法（通过 uv run）：
    uv run /home/prime/Codes/Docs/arxiv_tool.py search "PINN" --max 5
    uv run /home/prime/Codes/Docs/arxiv_tool.py fetch 2401.12345
    uv run /home/prime/Codes/Docs/arxiv_tool.py info 2401.12345
    uv run /home/prime/Codes/Docs/arxiv_tool.py bib 2401.12345 -o refs.bib
    uv run /home/prime/Codes/Docs/arxiv_tool.py tex 2401.12345
    uv run /home/prime/Codes/Docs/arxiv_tool.py cited 1711.10561 --max 20
    uv run /home/prime/Codes/Docs/arxiv_tool.py cited 1711.10561 --offset 20  # 翻页
    uv run /home/prime/Codes/Docs/arxiv_tool.py cited 1711.10561 --source openalex
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import arxiv
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv

if TYPE_CHECKING:
    from arxiv import Result

# 加载 .env（与脚本同目录），已有环境变量不覆盖
SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env", override=False)

# API Keys（有就用，没有也不影响基本功能）
S2_API_KEY: str | None = os.environ.get("S2_API_KEY")
OPENALEX_API_KEY: str | None = os.environ.get("OPENALEX_API_KEY")

# HTTP 请求头（arXiv 推荐设置 User-Agent）
HTTP_HEADERS = {
    "User-Agent": "arxiv-tool/1.0 (mailto:syouran0508@gmail.com)",
}

OUTPUT_DIR = SCRIPT_DIR / "arxiv"


def get_paper_info(arxiv_id: str, retries: int = 3) -> Result | None:
    """获取论文元数据，带重试和退避

    Args:
        arxiv_id: 清理后的 arXiv ID
        retries: 最大重试次数

    Returns:
        论文结果，失败返回 None（不抛异常）
    """
    clean_id = extract_arxiv_id(arxiv_id)
    for attempt in range(retries):
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[clean_id])
            results = list(client.results(search))
            if results:
                return results[0]
            print(f"未找到论文: {clean_id}", file=sys.stderr)
            return None
        except Exception as e:
            err_str = str(e)
            is_429 = "429" in err_str
            if is_429 and attempt < retries - 1:
                wait = 3 * (attempt + 1)
                print(f"API 限流 (429)，{wait}s 后重试 ({attempt + 1}/{retries})...", file=sys.stderr)
                time.sleep(wait)
            elif attempt < retries - 1:
                wait = 2 * (attempt + 1)
                print(f"API 请求失败: {e}，{wait}s 后重试 ({attempt + 1}/{retries})...", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"API 请求失败（已重试 {retries} 次）: {e}", file=sys.stderr)
    return None


def sanitize_filename(name: str, max_length: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._")
    if len(name) > max_length:
        name = name[:max_length]
    return name


def extract_arxiv_id(input_str: str) -> str:
    """从输入中提取 arXiv ID

    支持格式：
    - 2401.12345
    - arXiv:2401.12345
    - https://arxiv.org/abs/2401.12345
    - https://arxiv.org/pdf/2401.12345.pdf
    """
    patterns = [
        r"(\d{4}\.\d{4,5}(?:v\d+)?)",  # 新格式: 2401.12345 或 2401.12345v1
        r"([a-z-]+/\d{7})",  # 旧格式: cs/0401001
    ]
    for pattern in patterns:
        match = re.search(pattern, input_str)
        if match:
            return match.group(1)
    return input_str


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text().strip())
    doc.close()
    return "\n".join(text_parts)


def download_pdf(url: str, save_path: Path) -> bool:
    try:
        response = requests.get(url, headers=HTTP_HEADERS, timeout=60)
        response.raise_for_status()
        save_path.write_bytes(response.content)
        return True
    except requests.RequestException as e:
        print(f"下载失败: {e}", file=sys.stderr)
        return False


def search_papers(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    categories: list[str] | None = None,
) -> list[Result]:
    """搜索 arXiv 论文

    Args:
        query: 搜索关键词
        max_results: 最大结果数
        sort_by: 排序方式 (relevance, submitted, updated)
        categories: arXiv 分类列表，如 ["cs.AI", "cs.LG"]
    """
    search_query = query
    if categories:
        cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
        search_query = f"({query}) AND ({cat_query})"

    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submitted": arxiv.SortCriterion.SubmittedDate,
        "updated": arxiv.SortCriterion.LastUpdatedDate,
    }
    sort_criterion = sort_map[sort_by]

    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=sort_criterion,
    )

    return list(client.results(search))


def fetch_paper(arxiv_id: str, output_dir: Path) -> Path | None:
    """获取论文全文并保存为 txt

    不调用 API，直接下载 PDF 并转换。文件名使用 arXiv ID。

    Args:
        arxiv_id: arXiv ID
        output_dir: 输出目录

    Returns:
        保存的文件路径，失败返回 None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_id = extract_arxiv_id(arxiv_id)
    file_id = clean_id.replace("/", "_")
    txt_file = output_dir / f"{file_id}.txt"
    pdf_file = output_dir / f"{file_id}.pdf"

    if txt_file.exists():
        print(f"文件已存在: {txt_file}")
        return txt_file

    pdf_url = f"https://arxiv.org/pdf/{clean_id}"
    print(f"下载 PDF: {pdf_url}")

    if not download_pdf(pdf_url, pdf_file):
        return None

    print("提取文本...")
    text = extract_text_from_pdf(pdf_file)

    header = f"""\
# arXiv:{clean_id}

URL: https://arxiv.org/abs/{clean_id}

## Full Text

"""

    txt_file.write_text(header + text, encoding="utf-8")
    print(f"已保存 PDF: {pdf_file}")
    print(f"已保存 TXT: {txt_file}")
    return txt_file


def cmd_search(args):
    categories = args.categories.split(",") if args.categories else None

    results = search_papers(
        query=args.query,
        max_results=args.max,
        sort_by=args.sort,
        categories=categories,
    )

    if not results:
        print("未找到结果")
        return

    print(f"\n找到 {len(results)} 篇论文:\n")

    for i, paper in enumerate(results, 1):
        arxiv_id = paper.entry_id.split("/abs/")[-1]

        print(f"[{i}] {arxiv_id}")
        print(f"    标题: {paper.title}")
        print(f"    作者: {', '.join(a.name for a in paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"    日期: {paper.published.strftime('%Y-%m-%d')}")
        print(f"    分类: {', '.join(paper.categories[:3])}")

        print(f"    摘要: {paper.summary.replace(chr(10), ' ')}")
        print()


def cmd_fetch(args):
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    result = fetch_paper(args.arxiv_id, output_dir)
    if not result:
        sys.exit(1)


def cmd_info(args):
    clean_id = extract_arxiv_id(args.arxiv_id)

    paper = get_paper_info(clean_id)
    if not paper:
        return

    print(f"arXiv ID: {clean_id}")
    print(f"标题: {paper.title}")
    print(f"作者: {', '.join(a.name for a in paper.authors)}")
    print(f"发布日期: {paper.published.strftime('%Y-%m-%d')}")
    print(f"更新日期: {paper.updated.strftime('%Y-%m-%d')}")
    print(f"分类: {', '.join(paper.categories)}")
    print(f"PDF: {paper.pdf_url}")
    print(f"\n摘要:\n{paper.summary}")


# 停用词列表，用于生成 citation key
STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "at", "to", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need", "dare", "ought", "used",
    "via", "using", "based", "towards", "toward",
}


def generate_citation_key(paper: Result) -> str:
    """生成 BibTeX citation key

    格式：{第一作者姓小写}{年份}{标题首个实词小写}
    示例：li2025codepde, raissi2017physics
    """
    last_name = re.sub(r"[^a-z]", "", paper.authors[0].name.split()[-1].lower())
    year = paper.published.year

    title_words = re.findall(r"[a-zA-Z]+", paper.title)
    first_word = ""
    for word in title_words:
        if word.lower() not in STOPWORDS:
            first_word = word.lower()
            break

    return f"{last_name}{year}{first_word}"


def generate_bibtex(paper: Result, arxiv_id: str) -> str:
    """生成 arXiv 标准格式的 BibTeX 条目"""
    citation_key = generate_citation_key(paper)
    authors = " and ".join(a.name for a in paper.authors)
    clean_id = re.sub(r"v\d+$", "", arxiv_id)

    bibtex = f"""@misc{{{citation_key},
      title={{{paper.title}}},
      author={{{authors}}},
      year={{{paper.published.year}}},
      eprint={{{clean_id}}},
      archivePrefix={{arXiv}},
      primaryClass={{{paper.categories[0]}}},
      url={{https://arxiv.org/abs/{clean_id}}},
}}"""
    return bibtex


def cmd_bib(args):
    clean_id = extract_arxiv_id(args.arxiv_id)

    paper = get_paper_info(clean_id)
    if not paper:
        sys.exit(1)

    bibtex = generate_bibtex(paper, clean_id)

    if args.output:
        output_path = Path(args.output)
        mode = "a" if output_path.exists() else "w"
        with open(output_path, mode, encoding="utf-8") as f:
            if mode == "a" and output_path.stat().st_size > 0:
                f.write("\n\n")
            f.write(bibtex)
            f.write("\n")
        print(f"已{'追加' if mode == 'a' else '写入'}到: {output_path}")
    else:
        print(bibtex)


def print_tree(directory: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> list[str]:
    lines = []
    if current_depth >= max_depth:
        return lines

    items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name))
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{item.name}")

        if item.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(print_tree(item, prefix + extension, max_depth, current_depth + 1))

    return lines


def fetch_tex_source(arxiv_id: str, output_dir: Path) -> Path | None:
    """下载 arXiv LaTeX 源文件并解压

    不调用 API，直接从 e-print 下载源文件。目录名使用 arXiv ID，
    下载后尝试从 tex 文件中提取标题来补充目录名。

    Args:
        arxiv_id: arXiv ID
        output_dir: 输出目录

    Returns:
        解压后的目录路径，失败返回 None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_id = extract_arxiv_id(arxiv_id)
    # 移除版本号用于目录名
    dir_id = re.sub(r"v\d+$", "", clean_id).replace("/", "_")
    target_dir = output_dir / dir_id

    if target_dir.exists():
        print(f"目录已存在: {target_dir}")
        return target_dir
    existing = [p for p in output_dir.glob(f"{dir_id}_*") if p.is_dir()]
    if existing:
        print(f"目录已存在: {existing[0]}")
        return existing[0]

    source_url = f"https://arxiv.org/e-print/{clean_id}"
    print(f"下载源文件: {source_url}")

    try:
        response = requests.get(source_url, headers=HTTP_HEADERS, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"下载失败: {e}", file=sys.stderr)
        return None

    content = response.content

    target_dir.mkdir(parents=True, exist_ok=True)
    print("解压源文件...")
    try:
        _extract_source(content, target_dir)
    except Exception as e:
        print(f"解压失败: {e}", file=sys.stderr)
        shutil.rmtree(target_dir, ignore_errors=True)
        return None

    new_dir = _try_rename_with_title(target_dir, dir_id, output_dir)
    if new_dir:
        target_dir = new_dir

    print(f"已保存到: {target_dir}")
    return target_dir


def _extract_source(content: bytes, target_dir: Path) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            tar.extractall(target_dir, filter="data")
            print("解压为 tar.gz 格式")
            return
    except tarfile.ReadError:
        pass

    try:
        decompressed = gzip.decompress(content)
        # 解压后可能是 tar
        try:
            with tarfile.open(fileobj=io.BytesIO(decompressed), mode="r") as tar:
                tar.extractall(target_dir, filter="data")
                print("解压为 gzip+tar 格式")
                return
        except tarfile.ReadError:
            # 纯 gzip 压缩的单个文件
            tex_file = target_dir / "main.tex"
            tex_file.write_bytes(decompressed)
            print("解压为单个 tex 文件")
            return
    except gzip.BadGzipFile:
        pass

    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r") as tar:
            tar.extractall(target_dir, filter="data")
            print("解压为 tar 格式")
            return
    except tarfile.ReadError:
        pass

    tex_file = target_dir / "main.tex"
    tex_file.write_bytes(content)
    print("保存为单个 tex 文件（无压缩）")


def _try_rename_with_title(target_dir: Path, dir_id: str, output_dir: Path) -> Path | None:
    tex_files = list(target_dir.glob("*.tex"))
    if not tex_files:
        return None

    main_tex = next((f for f in tex_files if f.name == "main.tex"), tex_files[0])

    try:
        content = main_tex.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"\\title\s*\{([^}]+)\}", content, re.DOTALL)
        if match:
            raw_title = match.group(1).strip()
            raw_title = re.sub(r"\\[a-zA-Z]+\s*", " ", raw_title)
            raw_title = re.sub(r"[{}]", "", raw_title)
            raw_title = re.sub(r"\s+", " ", raw_title).strip()

            if raw_title:
                safe_title = sanitize_filename(raw_title, max_length=40)
                new_dir = output_dir / f"{dir_id}_{safe_title}"
                if not new_dir.exists():
                    target_dir.rename(new_dir)
                    print(f"从 tex 提取标题，目录重命名为: {new_dir.name}")
                    return new_dir
    except Exception:
        pass

    return None


S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
OPENALEX_API_BASE = "https://api.openalex.org"
CONTACT_EMAIL = "syouran0508@gmail.com"


def _s2_headers() -> dict[str, str]:
    headers = dict(HTTP_HEADERS)
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
    return headers


def _fetch_citations_s2(arxiv_id: str, max_results: int, offset: int = 0) -> tuple[list[dict], int] | None:
    """从 Semantic Scholar 获取引用该论文的论文列表

    Returns:
        (引用论文列表, 总被引次数)，失败返回 None
    """
    info_url = f"{S2_API_BASE}/paper/ArXiv:{arxiv_id}"
    try:
        resp = requests.get(
            info_url,
            params={"fields": "title,citationCount"},
            headers=_s2_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        paper_info = resp.json()
        print(f"论文: {paper_info['title']}")
        print(f"总被引次数: {paper_info['citationCount']}")
    except requests.RequestException as e:
        print(f"Semantic Scholar 查询失败: {e}", file=sys.stderr)
        return None

    citations_url = f"{S2_API_BASE}/paper/ArXiv:{arxiv_id}/citations"
    try:
        resp = requests.get(
            citations_url,
            params={
                "fields": "title,year,externalIds,citationCount,authors",
                "offset": offset,
                "limit": min(max_results, 1000),
            },
            headers=_s2_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"Semantic Scholar 引用列表获取失败: {e}", file=sys.stderr)
        return None

    results = [item["citingPaper"] for item in data["data"] if item["citingPaper"]["title"]]
    return results[:max_results], paper_info["citationCount"]


def _openalex_params(**extra) -> dict[str, str]:
    params = dict(extra)
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    else:
        params["mailto"] = CONTACT_EMAIL
    return params


def _resolve_openalex_id(arxiv_id: str) -> tuple[str, str, int] | None:
    """通过 arXiv DOI 查找 OpenAlex work ID

    Returns:
        (openalex_work_id, 论文标题, 被引次数)，失败返回 None
    """
    doi = f"10.48550/arXiv.{arxiv_id}"
    url = f"{OPENALEX_API_BASE}/works/doi:{doi}"
    try:
        resp = requests.get(url, params=_openalex_params(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        openalex_id = data["id"].split("/")[-1]  # "https://openalex.org/W123" -> "W123"
        return openalex_id, data["title"], data["cited_by_count"]
    except requests.RequestException:
        return None


def _fetch_citations_openalex(arxiv_id: str, max_results: int, offset: int = 0) -> tuple[list[dict], int] | None:
    """从 OpenAlex 获取引用该论文的论文列表

    Returns:
        (引用论文列表, 总被引次数)，失败返回 None
    """
    resolved = _resolve_openalex_id(arxiv_id)
    if not resolved:
        print("OpenAlex: 未找到该论文", file=sys.stderr)
        return None

    work_id, title, total_citations = resolved
    print(f"论文: {title}")
    print(f"总被引次数: {total_citations}")

    # OpenAlex 用 page 分页，page 从 1 开始
    per_page = min(max_results, 200)
    page = (offset // per_page) + 1

    url = f"{OPENALEX_API_BASE}/works"
    try:
        resp = requests.get(
            url,
            params=_openalex_params(
                filter=f"cites:{work_id}",
                select="id,title,authorships,publication_year,cited_by_count",
                per_page=str(per_page),
                page=str(page),
                sort="cited_by_count:desc",
            ),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"OpenAlex 引用列表获取失败: {e}", file=sys.stderr)
        return None

    return data["results"][:max_results], total_citations


def _print_citations_s2(results: list[dict], start: int = 1) -> None:
    for i, paper in enumerate(results, start):
        ext_ids = paper["externalIds"] or {}
        arxiv_ext = ext_ids.get("ArXiv")
        arxiv_str = f"  arXiv:{arxiv_ext}" if arxiv_ext else ""

        authors = paper["authors"] or []
        author_str = ", ".join(a["name"] for a in authors[:3])
        if len(authors) > 3:
            author_str += "..."

        print(f"[{i}] {paper['title']}")
        print(f"    作者: {author_str}")
        print(f"    年份: {paper['year'] or '?'}  被引: {paper['citationCount']}{arxiv_str}")
        print()


def _print_citations_openalex(results: list[dict], start: int = 1) -> None:
    for i, work in enumerate(results, start):
        authorships = work["authorships"] or []
        author_str = ", ".join(a["author"]["display_name"] for a in authorships[:3])
        if len(authorships) > 3:
            author_str += "..."

        print(f"[{i}] {work['title']}")
        print(f"    作者: {author_str}")
        print(f"    年份: {work['publication_year'] or '?'}  被引: {work['cited_by_count']}")
        print()


def cmd_cited(args):
    clean_id = extract_arxiv_id(args.arxiv_id)
    source = args.source
    offset = args.offset
    results = None
    used_source = ""

    if source in ("s2", "auto"):
        print(f"查询 Semantic Scholar: ArXiv:{clean_id}")
        ret = _fetch_citations_s2(clean_id, args.max, offset)
        if ret is not None:
            results, _total = ret
            used_source = "Semantic Scholar"

    if results is None and source in ("openalex", "auto"):
        if source == "auto":
            print("\nSemantic Scholar 失败，切换到 OpenAlex...")
        else:
            print(f"查询 OpenAlex: ArXiv:{clean_id}")
        ret = _fetch_citations_openalex(clean_id, args.max, offset)
        if ret is not None:
            results, _total = ret
            used_source = "OpenAlex"

    if not results:
        print(f"\n未找到引用 arXiv:{clean_id} 的论文")
        return

    start_num = offset + 1
    end_num = offset + len(results)
    print(f"\n数据源: {used_source}")
    print(f"显示第 {start_num}-{end_num} 篇引用论文:\n")

    if used_source == "Semantic Scholar":
        _print_citations_s2(results, start_num)
    else:
        _print_citations_openalex(results, start_num)


def cmd_tex(args):
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    result = fetch_tex_source(args.arxiv_id, output_dir)
    if result:
        print("\n目录结构:")
        print(result.name)
        tree_lines = print_tree(result)
        for line in tree_lines:
            print(line)
    else:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="arXiv 论文搜索与全文获取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
    %(prog)s search "PINN" --max 5
    %(prog)s search "physics-informed" --categories cs.LG,physics.comp-ph
    %(prog)s fetch 2401.12345
    %(prog)s fetch 2401.12345 --output ./papers
    %(prog)s info 2401.12345
    %(prog)s bib 2505.08783
    %(prog)s bib 2505.08783 -o references.bib
    %(prog)s tex 2505.08783
    %(prog)s cited 1711.10561
    %(prog)s cited 1711.10561 --max 50
    %(prog)s cited 1711.10561 --offset 20          # 第 21-40 条
    %(prog)s cited 1711.10561 --source openalex
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # search 子命令
    search_parser = subparsers.add_parser("search", help="搜索论文")
    search_parser.add_argument("query", help="搜索关键词")
    search_parser.add_argument("--max", type=int, default=10, help="最大结果数 (默认 10)")
    search_parser.add_argument(
        "--sort",
        choices=["relevance", "submitted", "updated"],
        default="relevance",
        help="排序方式 (默认 relevance)",
    )
    search_parser.add_argument("--categories", help="arXiv 分类，逗号分隔 (如 cs.AI,cs.LG)")
    search_parser.set_defaults(func=cmd_search)

    # fetch 子命令
    fetch_parser = subparsers.add_parser("fetch", help="获取论文全文并保存为 txt")
    fetch_parser.add_argument("arxiv_id", help="arXiv ID")
    fetch_parser.add_argument("--output", "-o", help=f"输出目录 (默认 {OUTPUT_DIR})")
    fetch_parser.set_defaults(func=cmd_fetch)

    # info 子命令
    info_parser = subparsers.add_parser("info", help="获取论文信息（不下载全文）")
    info_parser.add_argument("arxiv_id", help="arXiv ID")
    info_parser.set_defaults(func=cmd_info)

    # bib 子命令
    bib_parser = subparsers.add_parser("bib", help="生成 BibTeX 引用")
    bib_parser.add_argument("arxiv_id", help="arXiv ID")
    bib_parser.add_argument("--output", "-o", help="输出文件路径（追加写入）")
    bib_parser.set_defaults(func=cmd_bib)

    # cited 子命令
    cited_parser = subparsers.add_parser("cited", help="被引反查：查看哪些论文引用了它")
    cited_parser.add_argument("arxiv_id", help="arXiv ID")
    cited_parser.add_argument("--max", type=int, default=20, help="最大显示条数 (默认 20)")
    cited_parser.add_argument("--offset", type=int, default=0, help="跳过前 N 条结果，用于翻页 (默认 0)")
    cited_parser.add_argument(
        "--source",
        choices=["auto", "s2", "openalex"],
        default="auto",
        help="数据源: auto=自动(S2优先), s2=Semantic Scholar, openalex=OpenAlex (默认 auto)",
    )
    cited_parser.set_defaults(func=cmd_cited)

    # tex 子命令
    tex_parser = subparsers.add_parser("tex", help="下载 LaTeX 源文件并解压")
    tex_parser.add_argument("arxiv_id", help="arXiv ID")
    tex_parser.add_argument("--output", "-o", help=f"输出目录 (默认 {OUTPUT_DIR})")
    tex_parser.set_defaults(func=cmd_tex)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
