import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize, FreqDist, collocations
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import string

# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    return lemmatized

def process_text(text):
    sentences = nltk.sent_tokenize(text)
    data = []
    for sent in sentences:
        tokens = clean_text(sent)
        tags = pos_tag(tokens)  # POS tagging
        sentiment = sia.polarity_scores(sent)["compound"]
        data.append({"Sentence": sent, "Tokens": tokens, "POS": tags, "Sentiment": sentiment})
    return pd.DataFrame(data)

def get_analysis(df):
    all_tokens = [token for row in df["Tokens"] for token in row]
    freq_dist = FreqDist(all_tokens)
    freq_df = pd.DataFrame(freq_dist.most_common(), columns=["Token", "Count"])

    bigram_finder = collocations.BigramCollocationFinder.from_words(all_tokens)
    top_collocations = bigram_finder.nbest(collocations.BigramAssocMeasures().pmi, 10)
    colloc_strings = [" ".join(bigram) for bigram in top_collocations]

    sentiments = df["Sentiment"].tolist()

    return {
        "freq_df": freq_df,
        "collocations": colloc_strings,
        "sentiments": sentiments
    }
