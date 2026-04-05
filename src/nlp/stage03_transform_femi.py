"""
stage03_transform_femi.py

Source: validated BeautifulSoup object
Sink:   analysis-ready Pandas DataFrame

NOTE: We use Pandas for consistency with Module 5.
You may use Polars or another library if you prefer.
The pipeline pattern is identical; only the DataFrame API differs.

======================================================================
THE ANALYST WORKFLOW: Transform is not a single step
======================================================================

In a real NLP project, Transform is an iterative loop, not a
linear sequence of operations.

The loop looks like this:

  inspect → clean → inspect → engineer → inspect → clean → repeat

We need to inspect the data to know what cleaning is needed.
We inspect the cleaner data and begin to engineer additional features.
We inspect again to see how the cleaning worked and do more as needed.

This loop continues until the data is analysis-ready,
meaning a model or analyst could use the data without being misled
by noise, inconsistency, or missing signal.

This transform module is the SETTLED VERSION of that loop.
It captures decisions that survived inspection.
It does not show the full iterative process.
You should run it, inspect the logged output at each substage,
and ask yourself:

  - Does this look cleaner than the previous step?
  - Is there still noise I should remove?
  - Am I losing signal I want to keep?
  - What derived features would help a model or analyst?

The answers often suggest more work and that increases value.
That's the analyst workflow in action.

======================================================================
MODERN LLM Tools
======================================================================

Modern LLM tools are powerful, but only as good as the data provided.

`Garbage in, garbage out` is not a cliche.
It's why good data analysts are still very much needed.

A valuable analyst is one who understands:
  - why text needs cleaning before analysis
  - what signal looks like vs what noise looks like
  - how to inspect, iterate, and improve data quality
  - how to document those decisions so they are reproducible

The goal in this stage is not just to produce a clean DataFrame.
The goal is to develop the judgment to know when a DataFrame
is ready for use, and to professionally document why choices were made.

Judgment is what makes an analyst irreplaceable.

======================================================================
THIS STAGE
======================================================================

This stage runs three substages, each logged separately:

  03a. Extract fields from the validated HTML into a raw DataFrame.
       Inspect: does the raw data look right?

  03b. Clean and normalize the text fields.
       Inspect: is the text cleaner? Did we lose anything we needed?

  03c. Engineer derived features (tokens, word count, frequency).
       Inspect: do the derived fields add signal for downstream analysis?

The final DataFrame is the settled result of these three passes.

======================================================================
PURPOSE AND ANALYTICAL QUESTIONS
======================================================================

Purpose

  Transform validated HTML into a clean, analysis-ready DataFrame.

Analytical Questions

  - What does the raw extracted text look like before cleaning?
  - What noise is present and how should it be removed?
  - What derived features would support NLP analysis?
  - How does the cleaned text differ from the raw text?
  - Is the DataFrame genuinely ready for downstream use?

"""

# ============================================================
# Section 1. Setup and Imports
# ============================================================

import logging
import re
import string

from bs4 import BeautifulSoup, Tag
import pandas as pd
import spacy

# Load the spaCy English model.
# Run once if needed:
#   uv run python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


# ============================================================
# Section 2. Define Helper Functions
# ============================================================


def _get_text(element: Tag | None, strip_prefix: str = "", separator: str = "") -> str:
    """Return element text or 'unknown' if element is None."""
    if element is None:
        return "unknown"
    text = element.get_text(separator=separator, strip=True)
    return text.replace(strip_prefix, "", 1).strip() if strip_prefix else text


def _clean_text(text: str, nlp_model: spacy.language.Language) -> str:
    """Clean and normalize a text string for NLP analysis."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp_model(text)
    text = " ".join(
        [token.text for token in doc if not token.is_stop and not token.is_space]
    )
    return text


# ============================================================
# Section 3. Define Run Transform Function
# ============================================================


def run_transform(
    soup: BeautifulSoup,
    LOG: logging.Logger,
) -> pd.DataFrame:
    """Transform HTML into a clean, analysis-ready DataFrame."""
    LOG.info("========================")
    LOG.info("STAGE 03: TRANSFORM starting...")
    LOG.info("========================")
    LOG.info(
        "Source sink: validated BeautifulSoup object -> analysis-ready Pandas DataFrame"
    )
    LOG.info(
        "Goal: extract arXiv metadata, clean text fields, and engineer NLP features"
    )

    # ========================================================
    # PHASE 3.1: Locate and extract raw fields
    # ========================================================
    LOG.info("========================")
    LOG.info("PHASE 3.1: Locate HTML elements and extract raw fields")
    LOG.info("========================")

    title_tag: Tag | None = soup.find("h1", class_="title")
    authors_tag: Tag | None = soup.find("div", class_="authors")
    abstract_tag: Tag | None = soup.find("blockquote", class_="abstract")
    subheader_tag: Tag | None = soup.find("div", class_="subheader")
    dateline_tag: Tag | None = soup.find("div", class_="dateline")
    canonical_tag: Tag | None = soup.find("link", rel="canonical")
    primary_subject_tag: Tag | None = soup.find("span", class_="primary-subject")
    pdf_tag: Tag | None = soup.find("a", class_="abs-button download-pdf")
    journal_ref_label: Tag | None = soup.find(
        "td", string=re.compile("Journal reference:")
    )

    LOG.info(f"title_tag found: {title_tag is not None}")
    LOG.info(f"authors_tag found: {authors_tag is not None}")
    LOG.info(f"abstract_tag found: {abstract_tag is not None}")
    LOG.info(f"subheader_tag found: {subheader_tag is not None}")
    LOG.info(f"dateline_tag found: {dateline_tag is not None}")
    LOG.info(f"canonical_tag found: {canonical_tag is not None}")
    LOG.info(f"primary_subject_tag found: {primary_subject_tag is not None}")
    LOG.info(f"pdf_tag found: {pdf_tag is not None}")
    LOG.info(f"journal_ref_label found: {journal_ref_label is not None}")

    # Title requires special handling because arXiv includes the descriptor
    # "Title:" inside the same tag as the paper title.
    title: str = _get_text(title_tag, strip_prefix="Title:")
    LOG.info(f"Extracted title: {title}")

    author_tags_list: list[Tag] = authors_tag.find_all("a") if authors_tag else []
    author_names_list: list[str] = [
        tag.get_text(strip=True) for tag in author_tags_list
    ]
    authors: str = ", ".join(author_names_list) if author_names_list else "unknown"
    first_author: str = author_names_list[0] if author_names_list else "unknown"
    LOG.info(f"Extracted authors: {authors}")
    LOG.info(f"Derived first_author: {first_author}")

    # Abstract also requires special handling because arXiv stores "Abstract:"
    # inside the blockquote before the actual abstract text.
    abstract_raw: str = _get_text(abstract_tag, strip_prefix="Abstract:")
    LOG.info(f"Extracted abstract preview: {abstract_raw[:120]}...")

    subjects: str = _get_text(subheader_tag)
    primary_subject: str = _get_text(primary_subject_tag)
    LOG.info(f"Extracted subjects heading: {subjects}")
    LOG.info(f"Extracted primary_subject: {primary_subject}")

    date_submitted_raw: str = _get_text(dateline_tag)
    date_submitted_str: str = (
        date_submitted_raw.replace("[", "")
        .replace("]", "")
        .replace("Submitted on", "", 1)
        .strip()
        if date_submitted_raw != "unknown"
        else "unknown"
    )
    LOG.info(f"Extracted raw submitted date: {date_submitted_raw}")
    LOG.info(f"Cleaned submitted date: {date_submitted_str}")

    if canonical_tag is None:
        LOG.warning("Canonical link not found, setting arxiv_id to 'unknown'")
        arxiv_id: str = "unknown"
    else:
        href: str = str(canonical_tag.get("href", ""))
        arxiv_id = href.split("/abs/")[-1] if "/abs/" in href else "unknown"
    LOG.info(f"Extracted arxiv_id: {arxiv_id}")

    if pdf_tag is None:
        LOG.warning("PDF link not found, setting pdf_url to 'unknown'")
        pdf_url: str = "unknown"
    else:
        pdf_href = str(pdf_tag.get("href", "")).strip()
        pdf_url = (
            f"https://arxiv.org{pdf_href}"
            if pdf_href.startswith("/")
            else pdf_href
            if pdf_href
            else "unknown"
        )
    LOG.info(f"Extracted pdf_url: {pdf_url}")

    category_match = re.search(r"\(([^)]+)\)", primary_subject)
    primary_category_code: str = (
        category_match.group(1) if category_match else "unknown"
    )
    LOG.info(f"Derived primary_category_code: {primary_category_code}")

    journal_reference: str = "unknown"
    if journal_ref_label is not None:
        journal_ref_value = journal_ref_label.find_next_sibling("td")
        journal_reference = _get_text(journal_ref_value)
    LOG.info(f"Extracted journal_reference: {journal_reference}")

    # ========================================================
    # PHASE 3.2: Clean and normalize text fields
    # ========================================================
    LOG.info("========================")
    LOG.info("PHASE 3.2: Clean and normalize text")
    LOG.info("========================")
    LOG.info("Comparing raw abstract with cleaned abstract to inspect tradeoffs")

    abstract_clean: str = (
        _clean_text(abstract_raw, nlp) if abstract_raw != "unknown" else "unknown"
    )

    raw_len = len(abstract_raw) if abstract_raw != "unknown" else 0
    clean_len = len(abstract_clean) if abstract_clean != "unknown" else 0
    removed_chars = raw_len - clean_len if raw_len > 0 else 0
    removed_pct = 100 * (1 - clean_len / raw_len) if raw_len > 0 else 0.0

    LOG.info(f"abstract_raw preview:   {abstract_raw[:120]}...")
    LOG.info(f"abstract_clean preview: {abstract_clean[:120]}...")
    LOG.info(f"characters removed: {removed_chars} ({removed_pct:.1f}%)")

    # ========================================================
    # PHASE 3.3: Engineer derived features
    # ========================================================
    LOG.info("========================")
    LOG.info("PHASE 3.3: Engineer derived analytical features")
    LOG.info("========================")

    abstract_raw_word_count: int = (
        len(abstract_raw.split()) if abstract_raw != "unknown" else 0
    )
    author_count: int = len(author_names_list)

    tokens: list[str] = abstract_clean.split() if abstract_clean != "unknown" else []
    token_count: int = len(tokens)
    unique_token_count: int = len(set(tokens))
    type_token_ratio: float = (
        round(unique_token_count / token_count, 4) if token_count > 0 else 0.0
    )

    # New derived field: sentence count in the raw abstract
    abstract_sentence_count: int = (
        len([s for s in re.split(r"[.!?]+", abstract_raw) if s.strip()])
        if abstract_raw != "unknown"
        else 0
    )

    title_char_count: int = len(title) if title != "unknown" else 0

    LOG.info(f"abstract_word_count: {abstract_raw_word_count}")
    LOG.info(f"abstract_sentence_count: {abstract_sentence_count}")
    LOG.info(f"author_count: {author_count}")
    LOG.info(f"token_count: {token_count}")
    LOG.info(f"unique_token_count: {unique_token_count}")
    LOG.info(f"type_token_ratio: {type_token_ratio}")
    LOG.info(f"title_char_count: {title_char_count}")
    LOG.info(f"top 10 cleaned tokens preview: {tokens[:10]}")

    # ========================================================
    # PHASE 3.4: Build final record and DataFrame
    # ========================================================
    LOG.info("========================")
    LOG.info("PHASE 3.4: Build record and create DataFrame")
    LOG.info("========================")

    record = {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "first_author": first_author,
        "subjects": subjects,
        "primary_subject": primary_subject,
        "primary_category_code": primary_category_code,
        "submitted": date_submitted_str,
        "journal_reference": journal_reference,
        "pdf_url": pdf_url,
        "abstract_raw": abstract_raw,
        "abstract_clean": abstract_clean,
        "tokens": " ".join(tokens),
        "abstract_word_count": abstract_raw_word_count,
        "abstract_sentence_count": abstract_sentence_count,
        "token_count": token_count,
        "unique_token_count": unique_token_count,
        "type_token_ratio": type_token_ratio,
        "author_count": author_count,
        "title_char_count": title_char_count,
    }

    df = pd.DataFrame([record])

    LOG.info(f"Created DataFrame with {len(df)} row and {len(df.columns)} columns")
    LOG.info(f"Columns: {list(df.columns)}")
    LOG.info("DataFrame summary preview:")
    LOG.info(
        "\n%s",
        df[
            [
                "arxiv_id",
                "title",
                "first_author",
                "primary_category_code",
                "token_count",
                "type_token_ratio",
                "author_count",
            ]
        ]
        .head()
        .to_string(index=False),
    )

    LOG.info("Special handling note:")
    LOG.info(
        "The title and abstract required cleaning because arXiv stores descriptor "
        "prefixes like 'Title:' and 'Abstract:' inside the same HTML tags as the content."
    )

    LOG.info("Sink: Pandas DataFrame created")
    LOG.info("Transformation complete.")

    return df
