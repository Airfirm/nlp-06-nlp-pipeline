# NLP Portfolio

## Overview

This portfolio summarizes my Web Mining and Applied NLP work across the later modules of the course.

My strongest work focused on building EVTL and EVTAL pipelines that extracted data from APIs and web pages, validated structure, transformed messy text into analysis-ready tables, and produced useful summaries and visuals.

A major representative project in this portfolio is my Module 6 arXiv HTML EVTAL pipeline built around this paper page:

- Source URL: [APITestGenie on arXiv](https://arxiv.org/abs/2604.02039)
- Paper title: APITestGenie: Generating Web API Tests from Requirements and API Specifications with LLMs

This project went beyond a basic scrape by adding richer metadata extraction, text cleaning, feature engineering, and multiple analysis visuals.

## 1. NLP Techniques Implemented

I implemented the following NLP and text-processing techniques.

### Tokenization

In stage03_transform_femi.py, I tokenized the cleaned abstract text into individual tokens and stored them in the tokens field. I then used those tokens in stage04_analyze_femi.py for frequency analysis, bigram analysis, and visualizations.

Evidence from my work:

- File: stage03_transform_femi.py
- Derived fields:
  - tokens
  - token_count
  - unique_token_count
  - type_token_ratio

### Frequency Analysis

I used collections.Counter in stage04_analyze_femi.py to compute top token frequencies and create a frequency bar chart. I also logged the ranked token list inline for quick inspection in the terminal.

Evidence from my work:

- File: stage04_analyze_femi.py
- Output files:
  - data/processed/femi_top_tokens.png
- Logged output:
  - top token list with rank and count

### N-gram and Bigram Analysis

I added bigram analysis in the Analyze stage by creating consecutive token pairs and plotting the most common bigrams.

Evidence from my work:

- File: stage04_analyze_femi.py
- Output file:
  - data/processed/femi_top_bigrams.png

### Text Cleaning and Normalization

In the _clean_text helper inside stage03_transform_femi.py, I applied:

- lowercasing
- punctuation removal with str.translate()
- whitespace normalization with re.sub()
- stopword removal using spaCy

I also documented the tradeoffs of each step directly in comments.

Evidence from my work:

- File: stage03_transform_femi.py
- Fields:
  - abstract_raw
  - abstract_clean

### Web Scraping and Content Extraction from HTML

I used BeautifulSoup to extract HTML fields such as:

- title
- authors
- abstract
- subjects
- submitted date
- arXiv ID
- PDF URL
- journal reference

Evidence from my work:

- File: stage03_transform_femi.py
- Extracted fields:
  - title
  - authors
  - submitted
  - pdf_url
  - primary_category_code
  - journal_reference

### API-Based Text Analysis and JSON

In earlier modules, I built an API-based EVTL pipeline using JSON data. I adapted the transform stage to match the real API schema and added analytical fields such as:

- email_domain
- body_clean
- body_preview
- body_length_words
- line_count
- is_multiline

Evidence from my work:

- File: stage03_transform_femi.py in my API project
- Observed issue:
  - KeyError for userId showed that the example schema did not match my actual API response, so I updated the transform logic to use postId, id, name, email, and body

### POS-Based Analysis

In Module 6, I extended analysis by using spaCy POS tags to create a grammatical distribution chart.

Evidence from my work:

- File: stage04_analyze_femi.py
- Output file:
  - data/processed/femi_pos_distribution.png

## 2. Systems and Data Sources

I worked with multiple text and semi-structured data sources.

### HTML Web Pages

For Module 5 and Module 6, I analyzed arXiv paper pages in HTML format.

Representative source:

- [APITestGenie on arXiv](https://arxiv.org/abs/2604.02039)

HTML-specific issues I handled:

- descriptor text such as Title: and Abstract: embedded inside the same HTML tag as the content
- submitted date formatting such as [Submitted on 2 Apr 2026]
- metadata spread across different tags and classes

### JSON API Data

In the API module, I worked with comment-style JSON data containing fields like:

- postId
- id
- name
- email
- body

JSON-specific issues I handled:

- mismatch between example schema and actual API schema
- optional field handling with .get()
- creating derived columns for text analysis

### Plain Text and Controlled Corpus

In earlier NLP modules, I worked with:

- small controlled corpora for tokenization, co-occurrence, and bigram analysis
- local text files for preprocessing and frequency analysis

### Handling Messy or Variable Data

Across all projects, I learned that source data often cannot be used directly. I handled messy or variable data by:

- inspecting HTML and JSON before coding assumptions
- validating structure before transformation
- cleaning prefixes, punctuation, and whitespace
- using safe extraction patterns
- engineering fields that better support analysis

## 3. Pipeline Structure

My work consistently followed staged pipeline design.

### Extract

I collected data from:

- APIs using HTTP requests
- arXiv webpages using requests and raw HTML capture

Evidence:

- Files:
  - stage01_extract_femi.py
  - pipeline_api_femi_json.py
  - pipeline_femi_html.py

### Validate

I validated the structure of:

- JSON responses before transformation
- HTML documents before field extraction

Evidence:

- Files:
  - stage02_validate_femi.py
- Validation focus:
  - required keys for JSON
  - required tags and classes for HTML

### Transform

I transformed raw source data into a structured DataFrame by:

- mapping extracted values to columns
- cleaning text
- creating derived features
- preparing text for NLP analysis

Evidence:

- File:
  - stage03_transform_femi.py
- Example derived fields:
  - first_author
  - abstract_sentence_count
  - type_token_ratio
  - primary_category_code
  - pdf_url

### Analyze

In Module 6, I explicitly added an Analyze stage to produce visual summaries and computed signals.

Evidence:

- File:
  - stage04_analyze_femi.py
- Output visuals:
  - femi_top_tokens.png
  - femi_wordcloud.png
  - femi_top_bigrams.png
  - femi_token_length_histogram.png
  - femi_pos_distribution.png
  - femi_summary_metrics.png

### Load

I saved structured outputs such as:

- processed CSV files
- analysis charts
- cleaned DataFrames

Evidence:

- Files:
  - stage05_load.py
  - stage04_load_femi.py in earlier projects

## 4. Signals and Analysis Methods

I computed multiple text signals across projects.

### Word Frequency

I computed token frequencies with Counter(tokens) and visualized the top N tokens.

Used in:

- Module 6 HTML EVTAL project
- earlier preprocessing and corpus exploration projects

### Bigrams

I measured the most common token pairs to capture phrase-level structure.

Used in:

- stage04_analyze_femi.py
- earlier corpus exploration module

### Vocabulary Richness

I computed:

- token_count
- unique_token_count
- type_token_ratio

This helped measure repetition versus vocabulary variety.

### Structural Signals

I extracted and engineered metadata signals such as:

- author_count
- first_author
- abstract_sentence_count
- title_char_count
- primary_category_code

### POS Distribution

I used spaCy to analyze grammatical structure in the cleaned abstract and visualize POS tags.

### Document Summary Metrics

I created a summary metrics chart that combined:

- raw word count
- sentence count
- clean token count
- unique token count
- author count

## 5. Insights

My analysis produced several useful insights.

### HTML Extraction Requires Cleaning

One of the clearest lessons from my arXiv projects was that extracted HTML text is not automatically clean. Fields like title and abstract required removing descriptor prefixes such as Title: and Abstract: before they were usable.

### Transform Is Where Value Is Added

I learned that the transform stage is where raw web or API data becomes analytically useful. Adding fields like pdf_url, first_author, email_domain, or body_length_words made the outputs much more meaningful than a direct scrape.

### More Analysis Stages Create Better Understanding

In Module 6, adding multiple visuals helped me understand the abstract from different angles:

- token frequency showed dominant terms
- bigrams showed phrase structure
- POS distribution showed grammatical composition
- summary metrics showed document-level characteristics

### Structure Matters Across Data Sources

Working with JSON and HTML showed me that pipelines must be adapted to the source. HTML requires tag-based parsing and cleaning, while JSON requires schema inspection and field validation.

### Specific Observed Results

From my Module 6 project, I observed:

- successful extraction of arXiv metadata for 2604.02039
- a cleaned abstract suitable for token analysis
- generated analysis outputs saved into data/processed
- a one-row structured DataFrame with metadata and engineered features

From my earlier API project, I observed:

- the example code failed because it expected userId but my API returned postId
- after fixing the transform stage, I produced a much richer analysis-ready comment dataset

## 6. Representative Work

### 1. Module 6 HTML EVTAL Pipeline

Representative files:

- pipeline_web_html_femi.py <https://github.com/Airfirm/nlp-06-nlp-pipeline/blob/main/src/nlp/pipeline_web_html_femi.py>
- stage03_transform_femi.py <https://github.com/Airfirm/nlp-06-nlp-pipeline/blob/main/src/nlp/stage03_transform_femi.py>
- stage04_analyze_femi.py <https://github.com/Airfirm/nlp-06-nlp-pipeline/blob/main/src/nlp/stage04_analyze_femi.py>

This project extracts, cleans, analyzes, and visualizes metadata and abstract text from an arXiv HTML page. It is representative because it combines web mining, BeautifulSoup extraction, text cleaning, feature engineering, and multiple NLP visuals in a full staged pipeline.

### 2. API JSON EVTL Pipeline

Representative files:

- pipeline_api_femi_json.py <https://github.com/Airfirm/nlp-04-api-text-data/blob/main/src/nlp/pipeline_api_femi_json.py>
- stage03_transform_femi.py <https://github.com/Airfirm/nlp-04-api-text-data/blob/main/src/nlp/stage03_transform_femi.py>

This project extracts JSON comment data from an API, validates the structure, transforms the records into a structured DataFrame, and engineers features such as email domain, body length, and multiline flags. It is representative because it shows my ability to adapt example code to a real schema and handle inconsistent or unexpected input.

### 3. Corpus Exploration and Text Preprocessing Work

Representative files:

- nlp_corpus_enhanced_femi.py <https://github.com/Airfirm/nlp-03-text-exploration/blob/main/src/nlp/nlp_corpus_enhanced_femi.py>
- text_preprocessing_modified_femi.py <https://github.com/Airfirm/nlp-02-text-preprocessing/blob/main/src/nlp/text_preprocessing_modified_femi.py>

These projects demonstrate foundational NLP skills such as tokenization, frequency analysis, bigrams, co-occurrence, vocabulary diversity, and charting. They are representative because they show the earlier building blocks that supported my later HTML and API pipelines.

## 7. Skills

Based on this work, I can clearly demonstrate the following skills.

### Python Data Processing

I can build Python pipelines that extract, validate, transform, analyze, and load data using staged workflow design.

### Working with Text Data

I can clean, normalize, tokenize, summarize, and analyze text from:

- HTML webpages
- JSON APIs
- local corpora

### Handling Messy or Inconsistent Inputs

I can inspect raw source data, identify schema or structure mismatches, and adapt parsing or transformation logic accordingly.

### Structuring Repeatable Pipelines

I can organize projects into separate stage files and create repeatable EVTL or EVTAL workflows using:

- config files
- logging
- modular stage functions
- saved outputs

### HTML and JSON Parsing

I can use:

- BeautifulSoup for HTML extraction
- Python dictionaries and lists for JSON extraction
- validation logic to confirm required fields or tags exist

### NLP Feature Engineering

I can create useful text-based features such as:

- token counts
- type-token ratio
- sentence counts
- author counts
- domain extraction
- body and abstract length metrics

### Analysis and Visualization

I can create visualizations that communicate findings clearly, including:

- token frequency charts
- word clouds
- bigram charts
- token-length histograms
- POS distribution charts
- summary metrics charts

### Professional Communication

I can document projects clearly in Markdown, explain technical choices, describe tradeoffs, and present results as portfolio-ready artifacts.

## Conclusion

My Web Mining and Applied NLP work shows that I can move from raw, messy source data to structured, professional outputs. Across corpora, JSON APIs, and HTML webpages, I built reproducible pipelines that combined extraction, validation, transformation, analysis, and communication of results.

The Module 6 EVTAL project is my strongest portfolio artifact because it integrates web scraping, text preprocessing, feature engineering, and visual analysis into one complete workflow.
