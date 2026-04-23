# 🧠 NLP Suite App
### AI-Powered Text Analysis Toolkit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co)

> A professional, dark-themed NLP web application that combines four powerful AI tools — Text Summarization, Named Entity Recognition, Keyword Extraction, and Text Statistics — into a single, easy-to-use interface.

---

## 🌐 Live Demo

👉 **[Click here to try the app](https://share.streamlit.io)** *(replace with your deployed URL)*

---

## 📸 Screenshots

| Home | Text Summarizer |
|------|----------------|
| ![Home](screenshots/home.png) | ![Summarizer](screenshots/summarizer.png) |

| NER | Keyword Extractor |
|-----|------------------|
| ![NER](screenshots/ner.png) | ![Keywords](screenshots/keywords.png) |

---

## 🚀 Features & Tools

### 🔵 1. Text Summarizer
Automatically condenses long articles, reports, or paragraphs into concise summaries.

**Model:** `T5-Small` by Google (via HuggingFace Transformers)

**About T5-Small:**
- T5 (Text-to-Text Transfer Transformer) is a transformer model developed by **Google Research**
- Pre-trained on the **C4 (Colossal Clean Crawled Corpus)** dataset — 750GB of clean English text
- Fine-tuned on **CNN/DailyMail** dataset for summarization tasks
- Uses a sequence-to-sequence architecture — treats every NLP task as text-to-text

**Model Specs:**
| Property | Value |
|----------|-------|
| Parameters | 60 Million |
| Model Size | ~240 MB |
| Architecture | Encoder-Decoder Transformer |
| Training Data | C4 Corpus (750GB) |
| Fine-tuning | CNN/DailyMail News Dataset |

**Performance (ROUGE Scores on CNN/DailyMail):**
| Metric | Score |
|--------|-------|
| ROUGE-1 | 37.4 |
| ROUGE-2 | 17.3 |
| ROUGE-L | 34.0 |

> ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures overlap between generated and reference summaries. Higher is better.

**Visualizations:**
- 📊 Original vs Summary word count comparison
- 📉 Compression ratio metric
- 📋 Side-by-side original and summary display

---

### 🟢 2. Named Entity Recognition (NER)
Identifies and classifies named entities such as people, organizations, locations, dates, and more.

**Model:** `en_core_web_sm` by spaCy

**About spaCy en_core_web_sm:**
- Trained on the **OntoNotes 5.0** corpus — a large multilingual dataset with 1.5 million words
- Uses a **Convolutional Neural Network (CNN)** architecture for token classification
- Recognizes 18 entity types including PERSON, ORG, GPE, DATE, MONEY, and more

**Model Specs:**
| Property | Value |
|----------|-------|
| Model Size | ~12 MB |
| Architecture | CNN Token Classifier |
| Training Data | OntoNotes 5.0 |
| Language | English |
| Pipeline | tok2vec, tagger, parser, ner |

**Performance (OntoNotes 5.0 Test Set):**
| Metric | Score |
|--------|-------|
| Precision | 84.6% |
| Recall | 83.9% |
| F1 Score | 84.2% |

**Entity Types Detected:**
| Label | Description | Example |
|-------|-------------|---------|
| PERSON | People, including fictional | Raheem Khan |
| ORG | Companies, agencies, institutions | FAST University |
| GPE | Countries, cities, states | Gilgit Baltistan |
| DATE | Absolute or relative dates | January 2024 |
| MONEY | Monetary values | $1 million |
| LOC | Non-GPE locations | Himalayas |
| TIME | Times smaller than a day | 3:00 PM |
| NORP | Nationalities, religious groups | Pakistani |

> 💡 **Tip:** Capitalize proper nouns (Names, Places, Organizations) for best NER results.

**Visualizations:**
- 🏷️ Color-coded entity tags
- 📊 Entity distribution bar chart
- 📋 Full entity table with descriptions

---

### 🟡 3. Keyword Extractor
Discovers the most relevant keywords and key phrases from any text using semantic similarity.

**Model:** `KeyBERT` with `all-MiniLM-L6-v2` backbone

**About KeyBERT:**
- KeyBERT uses **BERT embeddings** to find keywords most similar to the document
- Backbone model `all-MiniLM-L6-v2` is trained on **1 billion sentence pairs**
- Uses **MMR (Maximal Marginal Relevance)** to reduce redundancy and increase diversity
- No fine-tuning needed — works zero-shot on any domain

**Model Specs:**
| Property | Value |
|----------|-------|
| Backbone | all-MiniLM-L6-v2 |
| Model Size | ~90 MB |
| Architecture | BERT (distilled) |
| Training Data | 1 Billion Sentence Pairs |
| Embedding Dim | 384 |

**Performance (Inspec Dataset):**
| Metric | Score |
|--------|-------|
| F1 Score | 39.8% |
| Precision | 45.2% |
| Recall | 35.7% |

> KeyBERT outperforms TF-IDF and RAKE on semantic relevance tasks.

**Key Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| top_n | Number of keywords to extract | 5 |
| diversity | MMR diversity score (0-1) | 0.5 |
| ngram_range | Unigrams and bigrams | (1,2) |

**Visualizations:**
- 🏷️ Keyword tags with relevance scores
- 📊 Lollipop chart of keyword scores
- ☁️ Word cloud of extracted keywords

---

### 🟠 4. Text Statistics
Analyzes readability, grade level, and linguistic complexity of any text.

**Library:** `textstat`

**About textstat:**
- Rule-based library implementing all major readability formulas
- No ML model needed — pure linguistic analysis
- Used in education, publishing, and content optimization industries

**Readability Metrics Explained:**
| Metric | Description | Score Range |
|--------|-------------|-------------|
| Flesch Reading Ease | Higher = easier to read | 0-100 |
| Flesch-Kincaid Grade | US school grade level | 1-18 |
| Gunning Fog Index | Years of education needed | 1-20 |
| SMOG Index | Simple measure of gobbledygook | 1-20 |
| ARI Score | Automated Readability Index | 1-14 |

**Readability Scale:**
| Flesch Score | Level | Audience |
|-------------|-------|----------|
| 90-100 | Very Easy 😊 | 5th Grade |
| 70-89 | Easy 📖 | 6th Grade |
| 60-69 | Standard 📖 | 7th Grade |
| 50-59 | Fairly Difficult 🤔 | High School |
| 30-49 | Difficult 🧠 | College |
| 0-29 | Very Difficult 🎓 | Professional |

**Visualizations:**
- 📊 Grade level comparison bar chart
- 🕸️ Text complexity radar chart
- 📋 Full statistics table

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Frontend | Streamlit |
| Summarization | HuggingFace Transformers, T5-Small |
| NER | spaCy, en_core_web_sm |
| Keywords | KeyBERT, sentence-transformers |
| Text Stats | textstat |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Data | Pandas, NumPy |
| Language | Python 3.11 |

---

## 📁 Project Structure

```
nlp-suite-app/
│
├── app.py                          # Main Streamlit application
│
├── utils/
│   ├── summarizer.py               # T5 summarization logic
│   ├── ner.py                      # spaCy NER logic
│   ├── keywords.py                 # KeyBERT keyword extraction
│   └── text_stats.py               # textstat analysis
│
├── notebooks/
│   ├── 01_summarizer_eda.ipynb     # T5 + ROUGE evaluation
│   ├── 02_ner_spacy.ipynb          # spaCy NER analysis
│   ├── 03_keywords_keybert.ipynb   # KeyBERT exploration
│   └── 04_textstats_eda.ipynb      # Text statistics EDA
│
├── requirements.txt                # All dependencies
└── README.md                       # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.11+
- pip
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/ConnectRaheem/nlp-suite-app.git
cd nlp-suite-app

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install PyTorch (CPU version)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# 5. Install all dependencies
pip install -r requirements.txt

# 6. Download spaCy model
python -m spacy download en_core_web_sm

# 7. Run the app
streamlit run app.py
```

---

## 📊 Model Comparison Summary

| Tool | Model | Size | Speed | Accuracy |
|------|-------|------|-------|----------|
| Summarizer | T5-Small | 240MB | Medium | ROUGE-1: 37.4 |
| NER | en_core_web_sm | 12MB | ⚡ Fast | F1: 84.2% |
| Keywords | all-MiniLM-L6-v2 | 90MB | ⚡ Fast | F1: 39.8% |
| Text Stats | textstat (rules) | 0MB | ⚡ Instant | Rule-based |

---

## 👨‍💻 About the Author

**Raheem Khan**
- 🎓 BS Computer Science — FAST University
- 📍 Gilgit Baltistan, Pakistan
- 💼 Data Scientist | NLP Engineer | ML Developer
- 🔗 [GitHub](https://github.com/ConnectRaheem/nlp-suite-app)

---

## 📄 License

This project is licensed under the MIT License — feel free to use it for learning and portfolio purposes.

---

## ⭐ If you found this useful, please give it a star!
