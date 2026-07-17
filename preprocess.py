

import pandas as pd
import numpy as np
import re
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class Sentiment140Preprocessor:
    """
    Preprocessor built specifically for Sentiment140 tweets
    """

    def __init__(self):
        self.lemmatizer  = WordNetLemmatizer()
        self.vader       = SentimentIntensityAnalyzer()
        self.stop_words  = set(stopwords.words('english'))

        # KEEP these - important for sentiment!
        keep = {
            'not', 'no', 'never', 'neither', 'nor',
            'but', 'however', 'although', 'despite',
            'very', 'really', 'extremely', 'absolutely',
            'so', 'too', 'quite', 'rather',
        }
        self.stop_words -= keep

        # Slang dictionary
        self.slang = {
            'omg':'oh my god', 'lol':'laugh out loud',
            'lmao':'laughing', 'rofl':'laughing',
            'smh':'shaking my head', 'fml':'my life terrible',
            'wtf':'what the hell', 'wth':'what the heck',
            'ugh':'frustration', 'aw':'disappointment',
            'aww':'disappointment', 'awww':'disappointment',
            'yay':'happy excited', 'woohoo':'very happy',
            'u':'you', 'ur':'your', 'r':'are', 'y':'why',
            'b4':'before', 'gr8':'great', 'luv':'love',
            'thx':'thanks', 'thnx':'thanks', 'ty':'thank you',
            'idk':'i dont know', 'idc':'i dont care',
            'imo':'in my opinion', 'tbh':'to be honest',
            'ngl':'not going to lie', 'rn':'right now',
            'bc':'because', 'cuz':'because', 'coz':'because',
            'gonna':'going to', 'wanna':'want to',
            'gotta':'got to', 'kinda':'kind of',
            'sorta':'sort of', 'shoulda':'should have',
            'woulda':'would have', 'coulda':'could have',
            'bday':'birthday', 'bf':'boyfriend',
            'gf':'girlfriend', 'bff':'best friend',
            'brb':'be right back', 'gtg':'got to go',
            'asap':'as soon as possible',
        }

        # Compiled regex patterns
        self.re_url      = re.compile(r'http\S+|www\S+|https\S+')
        self.re_mention  = re.compile(r'@\w+')
        self.re_hashtag  = re.compile(r'#(\w+)')
        self.re_html     = re.compile(r'<[^>]+>')
        self.re_emoji    = re.compile(
            r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]'
            r'|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]'
            r'|[\U00002702-\U000027B0]|[\U000024C2-\U0001F251]'
        )
        self.re_special  = re.compile(r'[^a-zA-Z\s]')
        self.re_spaces   = re.compile(r'\s+')
        self.re_repeated = re.compile(r'(.)\1{2,}')

    def count_features(self, text):
        """Count tweet features BEFORE cleaning"""
        text = str(text)
        return {
            'has_url'    : bool(self.re_url.search(text)),
            'has_mention': bool(self.re_mention.search(text)),
            'n_hashtags' : len(self.re_hashtag.findall(text)),
            'n_exclaim'  : text.count('!'),
            'n_question' : text.count('?'),
            'caps_ratio' : sum(1 for c in text if c.isupper()) / max(len(text), 1),
        }

    def clean(self, text):
        """Full cleaning pipeline for one tweet"""
        if not isinstance(text, str):
            return ""
        text = str(text).strip()
        if not text:
            return ""

        text = text.lower()
        text = self.re_html.sub(' ', text)
        text = text.replace('&amp;', 'and').replace('&lt;', '')
        text = text.replace('&gt;', '').replace('&quot;', '')
        text = self.re_url.sub(' ', text)
        text = self.re_mention.sub(' ', text)
        text = self.re_hashtag.sub(r' \1 ', text)
        text = self.re_emoji.sub(' ', text)
        text = self.re_repeated.sub(r'\1\1', text)

        words = text.split()
        words = [self.slang.get(w, w) for w in words]
        text  = ' '.join(words)

        text  = self.re_special.sub(' ', text)
        words = [
            self.lemmatizer.lemmatize(w)
            for w in text.split()
            if w not in self.stop_words and len(w) > 2
        ]
        text = ' '.join(words)
        text = self.re_spaces.sub(' ', text).strip()
        return text

    def get_vader_scores(self, text):
        scores = self.vader.polarity_scores(str(text))
        return scores['compound'], scores['pos'], scores['neg'], scores['neu']

    def get_textblob_scores(self, text):
        blob = TextBlob(str(text))
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def process_dataframe(self, df, text_col='text'):
        """Process full dataframe"""
        print("\n" + "═"*55)
        print("🧹 RUNNING PREPROCESSING PIPELINE")
        print(f"   Tweets to process: {len(df):,}")
        print("═"*55)

        # Step 1: Raw features
        print("\n📊 Step 1/4: Extracting raw features...")
        features = df[text_col].apply(self.count_features)
        df['has_url']     = features.apply(lambda x: x['has_url'])
        df['has_mention'] = features.apply(lambda x: x['has_mention'])
        df['n_hashtags']  = features.apply(lambda x: x['n_hashtags'])
        df['n_exclaim']   = features.apply(lambda x: x['n_exclaim'])
        df['n_question']  = features.apply(lambda x: x['n_question'])
        df['caps_ratio']  = features.apply(lambda x: x['caps_ratio'])
        print("   ✅ Raw features extracted!")

        # Step 2: Clean text
        print("\n🧹 Step 2/4: Cleaning tweets...")
        total   = len(df)
        cleaned = []
        for i, text in enumerate(df[text_col]):
            cleaned.append(self.clean(text))
            if (i + 1) % 5000 == 0 or i == total - 1:
                pct  = (i + 1) / total * 100
                done = int(pct / 2)
                bar  = '█' * done + '░' * (50 - done)
                print(f"\r   [{bar}] {pct:.1f}% ({i+1:,}/{total:,})",
                      end='', flush=True)
        print()
        df['cleaned_text'] = cleaned

        before = len(df)
        df = df[df['cleaned_text'].str.len() > 3].reset_index(drop=True)
        print(f"   ✅ Cleaned! Removed {before - len(df):,} empty rows")

        # Step 3: VADER
        print("\n⚡ Step 3/4: Computing VADER scores...")
        v = df[text_col].apply(self.get_vader_scores)
        df['vader_compound'] = v.apply(lambda x: x[0])
        df['vader_pos']      = v.apply(lambda x: x[1])
        df['vader_neg']      = v.apply(lambda x: x[2])
        df['vader_neu']      = v.apply(lambda x: x[3])
        print("   ✅ VADER scores computed!")

        # Step 4: TextBlob
        print("\n📝 Step 4/4: Computing TextBlob scores...")
        tb = df[text_col].apply(self.get_textblob_scores)
        df['tb_polarity']     = tb.apply(lambda x: x[0])
        df['tb_subjectivity'] = tb.apply(lambda x: x[1])
        print("   ✅ TextBlob scores computed!")

        # Save
        import os
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/cleaned_tweets.csv', index=False)

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              ✅ PREPROCESSING COMPLETE!                       ║
╠══════════════════════════════════════════════════════════════╣
║  Processed  : {len(df):>10,} tweets
║  Avg length : {df['cleaned_text'].str.len().mean():>10.1f} chars
║  Avg words  : {df['cleaned_text'].apply(lambda x: len(x.split())).mean():>10.1f} words
║  Saved to   : data/cleaned_tweets.csv
╚══════════════════════════════════════════════════════════════╝
        """)

        return df


# ── ONLY runs when you do: python preprocess.py directly ──────
if __name__ == "__main__":
    from load_data import load_sentiment140

    df = load_sentiment140(
        'data/training.1600000.processed.noemoticon.csv',
        sample_size=50000
    )
    prep     = Sentiment140Preprocessor()
    df_clean = prep.process_dataframe(df.copy(), text_col='text')
    print(f"\n✅ Ready! {len(df_clean):,} tweets processed.")