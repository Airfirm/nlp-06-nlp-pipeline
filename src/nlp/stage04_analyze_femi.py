"""
stage04_analyze_femi.py

Source: analysis-ready Pandas DataFrame (from Transform stage)
Sink:   visualizations saved to data/processed/

======================================================================
THE ANALYST WORKFLOW: Analyze means look, not just calculate
======================================================================

Engineered features have no value until someone looks at them.

The Analyze stage makes the data visible:

  - frequency distributions show which words dominate the text
  - word clouds give an immediate gestalt of the content
  - bar charts allow comparison across documents or categories

In a single-document pipeline like this one, analysis is exploratory:
you are asking "what is in here?" before deciding what to do with it.

In a multi-document pipeline (Module 7 and beyond), analysis becomes
comparative: "how does this document differ from others?"

The same tools apply in both cases. The questions change.

======================================================================
PURPOSE AND ANALYTICAL QUESTIONS
======================================================================

Purpose

  Compute frequency distributions and produce visualizations
  that surface patterns in the cleaned text.

Analytical Questions

  - Which words appear most frequently in the cleaned abstract?
  - Does the frequency distribution look meaningful or noisy?
  - Does the word cloud reflect the actual topic of the paper?
  - What does the type-token ratio tell us about vocabulary richness?
  - Would a different cleaning strategy change the results?

"""

# ============================================================
# Section 1. Setup and Imports
# ============================================================

from collections import Counter
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import spacy
from wordcloud import WordCloud

# Load spaCy model for POS tagging in analysis stage.
nlp = spacy.load("en_core_web_sm")


# ============================================================
# Section 2. Define Helper Functions
# ============================================================


def _plot_top_tokens(
    tokens: list[str],
    top_n: int,
    output_path: Path,
    title: str,
    LOG: logging.Logger,
) -> None:
    counter = Counter(tokens)
    most_common = counter.most_common(top_n)

    if not most_common:
        LOG.warning("No tokens to plot.")
        return

    words, counts = zip(*most_common, strict=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(reversed(words)), list(reversed(counts)))
    ax.set_xlabel("Frequency")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    LOG.info(f"  Saved top-token bar chart to {output_path}")


def _plot_wordcloud(
    text: str,
    output_path: Path,
    title: str,
    LOG: logging.Logger,
) -> None:
    if not text or text == "unknown":
        LOG.warning("No text available for word cloud.")
        return

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=80,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    LOG.info(f"  Saved word cloud to {output_path}")


def _plot_top_bigrams(
    tokens: list[str],
    top_n: int,
    output_path: Path,
    title: str,
    LOG: logging.Logger,
) -> None:
    if len(tokens) < 2:
        LOG.warning("Not enough tokens for bigram plotting.")
        return

    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    counter = Counter(bigrams)
    most_common = counter.most_common(top_n)

    if not most_common:
        LOG.warning("No bigrams to plot.")
        return

    phrases, counts = zip(*most_common, strict=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(reversed(phrases)), list(reversed(counts)))
    ax.set_xlabel("Frequency")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    LOG.info(f"  Saved bigram bar chart to {output_path}")


def _plot_token_length_histogram(
    tokens: list[str],
    output_path: Path,
    title: str,
    LOG: logging.Logger,
) -> None:
    if not tokens:
        LOG.warning("No tokens available for token-length histogram.")
        return

    token_lengths = [len(token) for token in tokens]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(token_lengths, bins=10)
    ax.set_xlabel("Token Length")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    LOG.info(f"  Saved token-length histogram to {output_path}")


def _plot_pos_distribution(
    text: str,
    output_path: Path,
    title: str,
    LOG: logging.Logger,
) -> None:
    if not text or text == "unknown":
        LOG.warning("No text available for POS distribution plot.")
        return

    doc = nlp(text)
    pos_counts = Counter(token.pos_ for token in doc if token.is_alpha)

    if not pos_counts:
        LOG.warning("No POS tags available to plot.")
        return

    labels, counts = zip(*pos_counts.most_common(), strict=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, counts)
    ax.set_xlabel("POS Tag")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    LOG.info(f"  Saved POS distribution chart to {output_path}")


def _plot_summary_metrics(
    metrics: dict[str, float | int],
    output_path: Path,
    title: str,
    LOG: logging.Logger,
) -> None:
    labels = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values)
    ax.set_ylabel("Value")
    ax.set_title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    LOG.info(f"  Saved summary metrics chart to {output_path}")


# ============================================================
# Section 3. Define Run Analyze Function
# ============================================================


def run_analyze(
    df: pd.DataFrame,
    LOG: logging.Logger,
    output_dir: Path = Path("data/processed"),
    top_n: int = 20,
) -> None:
    """Analyze the transformed DataFrame and produce visualizations."""
    LOG.info("========================")
    LOG.info("STAGE 04: ANALYZE starting...")
    LOG.info("========================")
    LOG.info(
        "Source sink: transformed DataFrame -> charts, summaries, and saved analysis artifacts"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # PHASE 4.1: Read row and summary stats
    # ========================================================
    LOG.info("========================")
    LOG.info("PHASE 4.1: Extract tokens and summary statistics from DataFrame")
    LOG.info("========================")

    row = df.iloc[0]

    title: str = str(row.get("title", "unknown"))
    tokens_str: str = str(row.get("tokens", ""))
    abstract_clean: str = str(row.get("abstract_clean", ""))
    token_count: int = int(row.get("token_count", 0))
    unique_token_count: int = int(row.get("unique_token_count", 0))
    type_token_ratio: float = float(row.get("type_token_ratio", 0.0))
    abstract_word_count: int = int(row.get("abstract_word_count", 0))
    abstract_sentence_count: int = int(row.get("abstract_sentence_count", 0))
    author_count: int = int(row.get("author_count", 0))

    tokens: list[str] = tokens_str.split() if tokens_str else []

    LOG.info(f"Paper title: {title}")
    LOG.info(f"Abstract word count (raw): {abstract_word_count}")
    LOG.info(f"Abstract sentence count: {abstract_sentence_count}")
    LOG.info(f"Token count (clean): {token_count}")
    LOG.info(f"Unique token count: {unique_token_count}")
    LOG.info(f"Type-token ratio: {type_token_ratio}")
    LOG.info(f"Author count: {author_count}")

    # ========================================================
    # PHASE 4.2: Top token bar chart
    # ========================================================
    LOG.info("========================")
    LOG.info(f"PHASE 4.2: Top {top_n} token frequency chart")
    LOG.info("========================")

    _plot_top_tokens(
        tokens=tokens,
        top_n=top_n,
        output_path=output_dir / "femi_top_tokens.png",
        title=f"Top {top_n} Tokens: {title}",
        LOG=LOG,
    )

    # ========================================================
    # PHASE 4.3: Word cloud
    # ========================================================
    LOG.info("========================")
    LOG.info("PHASE 4.3: Word cloud")
    LOG.info("========================")

    _plot_wordcloud(
        text=tokens_str,
        output_path=output_dir / "femi_wordcloud.png",
        title=f"Word Cloud: {title}",
        LOG=LOG,
    )

    # ========================================================
    # PHASE 4.4: Top bigrams
    # ========================================================
    LOG.info("========================")
    LOG.info("PHASE 4.4: Top bigram frequency chart")
    LOG.info("========================")

    _plot_top_bigrams(
        tokens=tokens,
        top_n=10,
        output_path=output_dir / "femi_top_bigrams.png",
        title=f"Top 10 Bigrams: {title}",
        LOG=LOG,
    )

    # ========================================================
    # PHASE 4.5: Token length histogram
    # ========================================================
    LOG.info("========================")
    LOG.info("PHASE 4.5: Token length histogram")
    LOG.info("========================")

    _plot_token_length_histogram(
        tokens=tokens,
        output_path=output_dir / "femi_token_length_histogram.png",
        title=f"Token Length Distribution: {title}",
        LOG=LOG,
    )

    # ========================================================
    # PHASE 4.6: POS distribution
    # ========================================================
    LOG.info("========================")
    LOG.info("PHASE 4.6: POS tag distribution")
    LOG.info("========================")

    _plot_pos_distribution(
        text=abstract_clean,
        output_path=output_dir / "femi_pos_distribution.png",
        title=f"POS Distribution: {title}",
        LOG=LOG,
    )

    # ========================================================
    # PHASE 4.7: Summary metrics chart
    # ========================================================
    LOG.info("========================")
    LOG.info("PHASE 4.7: Summary metrics chart")
    LOG.info("========================")

    _plot_summary_metrics(
        metrics={
            "raw_words": abstract_word_count,
            "sentences": abstract_sentence_count,
            "clean_tokens": token_count,
            "unique_tokens": unique_token_count,
            "authors": author_count,
        },
        output_path=output_dir / "femi_summary_metrics.png",
        title=f"Summary Metrics: {title}",
        LOG=LOG,
    )

    # ========================================================
    # PHASE 4.8: Inline token summary
    # ========================================================
    LOG.info("========================")
    LOG.info("PHASE 4.8: Top token summary (inline log output)")
    LOG.info("========================")

    counter = Counter(tokens)
    top_tokens = counter.most_common(top_n)

    LOG.info(f"Top {top_n} tokens by frequency:")
    for rank, (word, count) in enumerate(top_tokens, start=1):
        LOG.info(f"  {rank:>2}. {word:<25} {count}")

    LOG.info("Sink: visualizations saved to data/processed/")
    LOG.info("Analysis complete.")
