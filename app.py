import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NLP Suite App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #0D1117;
        color: #E6EDF3;
    }
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    .card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0px;
    }
    .metric-card {
        background-color: #1C2128;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 12px;
        color: #8B949E;
        margin: 0;
    }
    .tool-title {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .tool-desc {
        font-size: 15px;
        color: #8B949E;
        margin-bottom: 20px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #238636, #2EA043);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2EA043, #3FB950);
        transform: translateY(-2px);
    }
    .stTextArea > div > div > textarea {
        background-color: #1C2128;
        color: #E6EDF3;
        border: 1px solid #30363D;
        border-radius: 8px;
        font-size: 15px;
    }
    .tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: bold;
        margin: 3px;
    }
    .divider {
        border-top: 1px solid #30363D;
        margin: 20px 0;
    }
    .header-banner {
        background: linear-gradient(135deg, #1C2128, #161B22);
        border: 1px solid #30363D;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        text-align: center;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stSelectbox > div > div {
        background-color: #1C2128;
        color: #E6EDF3;
        border: 1px solid #30363D;
    }
    .stAlert {
        background-color: #1C2128;
        border: 1px solid #30363D;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_summarizer():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("summarization",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,
                    framework="pt")

@st.cache_resource
def load_ner():
    import spacy
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_keybert():
    from keybert import KeyBERT
    return KeyBERT(model='all-MiniLM-L6-v2')


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0px;'>
        <div style='font-size:48px;'>🧠</div>
        <div style='font-size:22px; font-weight:bold; color:#E6EDF3;'>NLP Suite</div>
        <div style='font-size:12px; color:#8B949E;'>AI-Powered Text Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("<div style='color:#8B949E; font-size:12px; padding:5px 0;'>TOOLS</div>",
                unsafe_allow_html=True)

    tool = st.selectbox(
        "Select a Tool",
        options=[
            "🏠  Home",
            "🔵  Text Summarizer",
            "🟢  Named Entity Recognition",
            "🟡  Keyword Extractor",
            "🟠  Text Statistics"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='padding: 10px 0;'>
        <div style='color:#8B949E; font-size:12px;'>MODELS USED</div>
        <div style='margin-top:10px;'>
            <div style='color:#E6EDF3; font-size:13px; padding:4px 0;'>🤗 T5-Small (Summarizer)</div>
            <div style='color:#E6EDF3; font-size:13px; padding:4px 0;'>🔬 en_core_web_sm (NER)</div>
            <div style='color:#E6EDF3; font-size:13px; padding:4px 0;'>🔑 all-MiniLM-L6-v2 (Keywords)</div>
            <div style='color:#E6EDF3; font-size:13px; padding:4px 0;'>📊 textstat (Text Stats)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; padding:10px 0;'>
        <div style='color:#8B949E; font-size:12px;'>Built by</div>
        <div style='color:#E6EDF3; font-size:14px; font-weight:bold;'>Raheem</div>
        <div style='margin-top:8px;'>
            <a href='https://github.com/ConnectRaheem/nlp-suite-app'
               style='color:#58A6FF; font-size:12px;'>
               GitHub Repo
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────
if "Home" in tool:
    st.markdown("""
    <div class='header-banner'>
        <div style='font-size:52px;'>🧠</div>
        <div style='font-size:36px; font-weight:bold; color:#E6EDF3;
                    margin:10px 0;'>NLP Suite App</div>
        <div style='font-size:16px; color:#8B949E;'>
            Your AI-Powered Text Analysis Toolkit
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='card' style='border-top: 3px solid #58A6FF;'>
            <div style='font-size:28px;'>🔵</div>
            <div style='font-size:18px; font-weight:bold;
                        color:#58A6FF; margin:8px 0;'>Text Summarizer</div>
            <div style='color:#8B949E; font-size:13px;'>
                Summarize long texts into concise summaries using T5 transformer model
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card' style='border-top: 3px solid #3FB950;'>
            <div style='font-size:28px;'>🟢</div>
            <div style='font-size:18px; font-weight:bold;
                        color:#3FB950; margin:8px 0;'>Named Entity Recognition</div>
            <div style='color:#8B949E; font-size:13px;'>
                Extract people, organizations, locations and more using spaCy NER
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='card' style='border-top: 3px solid #D29922;'>
            <div style='font-size:28px;'>🟡</div>
            <div style='font-size:18px; font-weight:bold;
                        color:#D29922; margin:8px 0;'>Keyword Extractor</div>
            <div style='color:#8B949E; font-size:13px;'>
                Discover the most relevant keywords and phrases using KeyBERT
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='card' style='border-top: 3px solid #F78166;'>
            <div style='font-size:28px;'>🟠</div>
            <div style='font-size:18px; font-weight:bold;
                        color:#F78166; margin:8px 0;'>Text Statistics</div>
            <div style='color:#8B949E; font-size:13px;'>
                Analyze readability, grade level and complexity of any text
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; color:#8B949E; font-size:14px; padding:20px;'>
        👈 Select a tool from the sidebar to get started
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# TEXT SUMMARIZER
# ─────────────────────────────────────────
elif "Summarizer" in tool:
    st.markdown("""
    <div class='tool-title' style='color:#58A6FF;'>🔵 Text Summarizer</div>
    <div class='tool-desc'>
        Paste any long text and get a concise summary powered by T5 Transformer
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area(
            "📝 Enter your text here",
            height=200,
            placeholder="Paste your article, paragraph or any long text here..."
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        max_len = st.slider("Max Summary Length", 50, 300, 150)
        min_len = st.slider("Min Summary Length", 20, 100, 50)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Summarize Text")

    if analyze_btn and text_input.strip():
        with st.spinner("🤖 T5 model is summarizing your text..."):
            summarizer = load_summarizer()
            result = summarizer(text_input[:1024],
                   max_length=max_len,
                   min_length=min_len,
                   do_sample=False)
            summary = result[0]['summary_text']
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("### 📊 Results")

        col1, col2, col3, col4 = st.columns(4)
        orig_words = len(text_input.split())
        summ_words = len(summary.split())
        reduction  = round((1 - summ_words/orig_words) * 100, 1)

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value' style='color:#58A6FF;'>{orig_words}</p>
                <p class='metric-label'>Original Words</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value' style='color:#3FB950;'>{summ_words}</p>
                <p class='metric-label'>Summary Words</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value' style='color:#D29922;'>{reduction}%</p>
                <p class='metric-label'>Reduction</p>
            </div>""", unsafe_allow_html=True)
        with col4:
            ratio = round(summ_words/orig_words, 2)
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value' style='color:#F78166;'>{ratio}</p>
                <p class='metric-label'>Compression Ratio</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class='card'>
                <div style='color:#8B949E; font-size:12px;
                            margin-bottom:8px;'>ORIGINAL TEXT</div>
            """, unsafe_allow_html=True)
            st.write(text_input)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='card' style='border: 1px solid #58A6FF;'>
                <div style='color:#58A6FF; font-size:12px;
                            margin-bottom:8px;'>✨ SUMMARY</div>
            """, unsafe_allow_html=True)
            st.write(summary)
            st.markdown("</div>", unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("⚠️ Please enter some text first!")


# ─────────────────────────────────────────
# NAMED ENTITY RECOGNITION
# ─────────────────────────────────────────
elif "Entity" in tool:
    st.markdown("""
    <div class='tool-title' style='color:#3FB950;'>🟢 Named Entity Recognition</div>
    <div class='tool-desc'>
        Extract and identify people, organizations, locations, dates and more
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.info("💡 Tip: Capitalize proper nouns (Names, Places, Organizations) for best results!")

    text_input = st.text_area(
        "📝 Enter your text here",
        height=200,
        placeholder="Enter text containing names, places, organizations, dates..."
    )

    analyze_btn = st.button("🚀 Extract Entities")

    if analyze_btn and text_input.strip():
        with st.spinner("🔬 spaCy is analyzing your text..."):
            import spacy
            nlp = load_ner()
            doc = nlp(text_input)

            records = []
            for ent in doc.ents:
                records.append({
                    "Entity" : ent.text,
                    "Label"  : ent.label_,
                    "Description": spacy.explain(ent.label_)
                })

        if records:
            df_ents = pd.DataFrame(records)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### 📊 Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <p class='metric-value' style='color:#3FB950;'>{len(df_ents)}</p>
                    <p class='metric-label'>Total Entities</p>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <p class='metric-value' style='color:#58A6FF;'>
                        {df_ents['Label'].nunique()}
                    </p>
                    <p class='metric-label'>Unique Types</p>
                </div>""", unsafe_allow_html=True)
            with col3:
                top_label = df_ents['Label'].value_counts().index[0]
                st.markdown(f"""
                <div class='metric-card'>
                    <p class='metric-value' style='color:#D29922;'>{top_label}</p>
                    <p class='metric-label'>Top Entity Type</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            label_colors = {
                "PERSON": "#58A6FF", "ORG": "#3FB950",
                "GPE": "#D29922",    "DATE": "#F78166",
                "MONEY": "#A371F7",  "LOC": "#39D353",
                "TIME": "#FF7B72",   "NORP": "#FFA657"
            }

            st.markdown("#### 🏷️ Entities Found")
            tags_html = ""
            for _, row in df_ents.iterrows():
                color = label_colors.get(row["Label"], "#8B949E")
                tags_html += f"""
                <span class='tag' style='background-color:{color}22;
                      color:{color}; border:1px solid {color};'>
                    {row['Entity']} · {row['Label']}
                </span>"""
            st.markdown(tags_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📋 Entity Table")
                st.dataframe(df_ents, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("#### 📊 Entity Distribution")
                label_counts = df_ents["Label"].value_counts()
                colors_list = [label_colors.get(l, "#8B949E") for l in label_counts.index]

                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor("#161B22")
                ax.set_facecolor("#161B22")
                bars = ax.barh(label_counts.index, label_counts.values, color=colors_list)
                ax.set_title("Entity Label Distribution", color="#E6EDF3", fontsize=13)
                ax.tick_params(colors="#8B949E")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("#30363D")
                ax.spines["left"].set_color("#30363D")
                for bar, val in zip(bars, label_counts.values):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                            str(val), va="center", color="#E6EDF3", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("ℹ️ No entities found in the text.")

    elif analyze_btn:
        st.warning("⚠️ Please enter some text first!")


# ─────────────────────────────────────────
# KEYWORD EXTRACTOR
# ─────────────────────────────────────────
elif "Keyword" in tool:
    st.markdown("""
    <div class='tool-title' style='color:#D29922;'>🟡 Keyword Extractor</div>
    <div class='tool-desc'>
        Discover the most relevant keywords and keyphrases using KeyBERT
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area(
            "📝 Enter your text here",
            height=200,
            placeholder="Enter a paragraph or article to extract keywords..."
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        top_n     = st.slider("Number of Keywords", 3, 15, 5)
        diversity = st.slider("Diversity", 0.0, 1.0, 0.5)

    analyze_btn = st.button("🚀 Extract Keywords")

    if analyze_btn and text_input.strip():
        with st.spinner("🔑 KeyBERT is extracting keywords..."):
            kw_model = load_keybert()
            keywords = kw_model.extract_keywords(
                text_input,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n,
                use_mmr=True,
                diversity=diversity
            )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("### 📊 Results")

        df_kw = pd.DataFrame(keywords, columns=["Keyword", "Score"])
        df_kw["Score"] = df_kw["Score"].round(4)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value' style='color:#D29922;'>{len(df_kw)}</p>
                <p class='metric-label'>Keywords Found</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value' style='color:#3FB950;'>{df_kw['Score'].max():.3f}</p>
                <p class='metric-label'>Top Score</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value' style='color:#58A6FF;'>{df_kw['Score'].mean():.3f}</p>
                <p class='metric-label'>Avg Score</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### 🏷️ Extracted Keywords")
        tags_html = ""
        for _, row in df_kw.iterrows():
            tags_html += f"""
            <span class='tag' style='background-color:#D2992222;
                  color:#D29922; border:1px solid #D29922;'>
                🔑 {row['Keyword']} ({row['Score']})
            </span>"""
        st.markdown(tags_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📋 Keywords Table")
            st.dataframe(df_kw, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### 📊 Keyword Scores")
            df_sorted = df_kw.sort_values("Score", ascending=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#161B22")
            ax.set_facecolor("#161B22")
            colors_grad = plt.cm.YlOrBr(np.linspace(0.3, 0.9, len(df_sorted)))
            ax.hlines(y=df_sorted["Keyword"], xmin=0, xmax=df_sorted["Score"],
                      color="#30363D", linewidth=2)
            ax.scatter(df_sorted["Score"], df_sorted["Keyword"],
                       color=colors_grad, s=100, zorder=5)
            for score, keyword in zip(df_sorted["Score"], df_sorted["Keyword"]):
                ax.text(score + 0.01, keyword, f"{score:.3f}", va="center",
                        color="#E6EDF3", fontsize=9)
            ax.set_title("Keyword Relevance Scores", color="#E6EDF3", fontsize=13)
            ax.set_xlabel("Score", color="#8B949E")
            ax.tick_params(colors="#8B949E")
            ax.set_xlim(0, 1.15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#30363D")
            ax.spines["left"].set_color("#30363D")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("#### ☁️ Keyword Cloud")
        keyword_dict = dict(keywords)
        colors_wc = ["#D29922", "#F0B429", "#FFC940", "#FFE0A3", "#8B6914"]
        custom_cmap = LinearSegmentedColormap.from_list("gold", colors_wc)
        wordcloud = WordCloud(
            width=1400, height=500,
            background_color="#0D1117",
            colormap=custom_cmap,
            max_words=50,
            prefer_horizontal=0.85,
            min_font_size=14,
            max_font_size=120,
            collocations=False
        ).generate_from_frequencies(keyword_dict)

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

    elif analyze_btn:
        st.warning("⚠️ Please enter some text first!")


# ─────────────────────────────────────────
# TEXT STATISTICS
# ─────────────────────────────────────────
elif "Statistics" in tool:
    st.markdown("""
    <div class='tool-title' style='color:#F78166;'>🟠 Text Statistics</div>
    <div class='tool-desc'>
        Analyze readability, grade level and complexity of any text
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    text_input = st.text_area(
        "📝 Enter your text here",
        height=200,
        placeholder="Enter any text to analyze its readability and complexity..."
    )

    analyze_btn = st.button("🚀 Analyze Text")

    if analyze_btn and text_input.strip():
        with st.spinner("📊 Calculating text statistics..."):
            import textstat
            stats = {
                "word_count"          : textstat.lexicon_count(text_input, removepunct=True),
                "sentence_count"      : textstat.sentence_count(text_input),
                "syllable_count"      : textstat.syllable_count(text_input),
                "difficult_words"     : textstat.difficult_words(text_input),
                "avg_words_per_sent"  : round(textstat.avg_sentence_length(text_input), 2),
                "flesch_reading_ease" : round(textstat.flesch_reading_ease(text_input), 2),
                "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text_input), 2),
                "gunning_fog"         : round(textstat.gunning_fog(text_input), 2),
                "smog_index"          : round(textstat.smog_index(text_input), 2),
                "ari_score"           : round(textstat.automated_readability_index(text_input), 2)
            }

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("### 📊 Results")

        fre = stats["flesch_reading_ease"]
        if fre >= 80:
            read_label = "Very Easy 😊"
            read_color = "#3FB950"
        elif fre >= 60:
            read_label = "Standard 📖"
            read_color = "#D29922"
        elif fre >= 40:
            read_label = "Difficult 🤔"
            read_color = "#F78166"
        else:
            read_label = "Very Difficult 🧠"
            read_color = "#FF7B72"

        col1, col2, col3, col4, col5 = st.columns(5)
        metrics = [
            (stats["word_count"],          "Words",          "#58A6FF"),
            (stats["sentence_count"],      "Sentences",      "#3FB950"),
            (stats["difficult_words"],     "Difficult Words", "#D29922"),
            (stats["flesch_reading_ease"], "Reading Ease",   read_color),
            (read_label,                   "Level",          read_color),
        ]
        for col, (val, label, color) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <p class='metric-value' style='color:{color}; font-size:22px;'>{val}</p>
                    <p class='metric-label'>{label}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📋 Full Statistics Table")
            df_stats = pd.DataFrame(stats.items(), columns=["Metric", "Value"])
            st.dataframe(df_stats, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### 📊 Grade Level Comparison")
            grade_metrics = {
                "FK Grade"   : stats["flesch_kincaid_grade"],
                "Gunning Fog": stats["gunning_fog"],
                "SMOG Index" : stats["smog_index"],
                "ARI Score"  : stats["ari_score"]
            }
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#161B22")
            ax.set_facecolor("#161B22")
            colors_list = ["#58A6FF", "#3FB950", "#D29922", "#F78166"]
            bars = ax.bar(grade_metrics.keys(), grade_metrics.values(),
                          color=colors_list, edgecolor="#0D1117", width=0.5)
            for bar, val in zip(bars, grade_metrics.values()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        str(val), ha="center", color="#E6EDF3", fontsize=11, fontweight="bold")
            ax.set_title("Readability Grade Levels", color="#E6EDF3", fontsize=13)
            ax.set_ylabel("Grade Level", color="#8B949E")
            ax.tick_params(colors="#8B949E")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#30363D")
            ax.spines["left"].set_color("#30363D")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("#### 🕸️ Text Complexity Radar")
        radar_data = {
            "FK Grade"    : stats["flesch_kincaid_grade"],
            "Gunning Fog" : stats["gunning_fog"],
            "SMOG"        : stats["smog_index"],
            "ARI"         : stats["ari_score"],
            "Difficult\nWords": stats["difficult_words"]
        }
        labels  = list(radar_data.keys())
        values  = list(radar_data.values())
        max_val = max(values) if max(values) > 0 else 1
        values_norm  = [v / max_val for v in values]
        values_norm += values_norm[:1]

        N      = len(labels)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#161B22")
        ax.set_facecolor("#1C2128")
        ax.plot(angles, values_norm, "o-", linewidth=2, color="#F78166")
        ax.fill(angles, values_norm, alpha=0.2, color="#F78166")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color="#E6EDF3", fontsize=11)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="#8B949E", fontsize=8)
        ax.grid(color="#30363D", linewidth=0.5)
        ax.spines["polar"].set_color("#30363D")
        ax.set_title("Text Complexity Profile", color="#E6EDF3", fontsize=13,
                     fontweight="bold", pad=20)
        plt.tight_layout()
        st.pyplot(fig)

    elif analyze_btn:
        st.warning("⚠️ Please enter some text first!")