

import time
import os

print("""
╔══════════════════════════════════════════════════════════════╗
║   🐦 SENTIMENT140 — COMPLETE SENTIMENT ANALYSIS PROJECT      ║
╚══════════════════════════════════════════════════════════════╝
""")

t0 = time.time()

# ════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ════════════════════════════════════════════════════════════
print("▶ STEP 1/4 — Loading Sentiment140 Dataset")
print("─"*50)

from load_data import load_sentiment140

df_raw = load_sentiment140(
    filepath    = 'data/training.1600000.processed.noemoticon.csv',
    sample_size = 50000
)

# ════════════════════════════════════════════════════════════
# STEP 2: PREPROCESS
# ════════════════════════════════════════════════════════════
print("\n▶ STEP 2/4 — Preprocessing Tweets")
print("─"*50)

from preprocess import Sentiment140Preprocessor

preprocessor = Sentiment140Preprocessor()
df_clean     = preprocessor.process_dataframe(
    df_raw.copy(), text_col='text'
)

print(f"\n✅ Preprocessing done! {len(df_clean):,} tweets ready.")

# ════════════════════════════════════════════════════════════
# STEP 3: VISUALIZATIONS
# ════════════════════════════════════════════════════════════
print("\n▶ STEP 3/4 — Creating Visualizations")
print("─"*50)

from visualize import create_dashboard, create_wordclouds, create_analytics

create_dashboard(df_clean)
create_wordclouds(df_clean)
create_analytics(df_clean)

# ════════════════════════════════════════════════════════════
# STEP 4: TRAIN MODELS
# ════════════════════════════════════════════════════════════
print("\n▶ STEP 4/4 — Training ML Models")
print("─"*50)

from train_models import Sentiment140Trainer      # ✅ Only imports CLASS now

trainer = Sentiment140Trainer(df_clean)           # ✅ Pass df_clean here
trainer.prepare()
trainer.train()
trainer.visualize()
trainer.print_reports()

# ════════════════════════════════════════════════════════════
# BONUS: LIVE PREDICTIONS ON YOUR DATASET TWEETS
# ════════════════════════════════════════════════════════════
print("\n🔮 BONUS — Predicting Original Dataset Tweets")
print("─"*50)

# Use actual tweets from YOUR loaded dataset
sample_tweets = df_raw['text'].sample(10, random_state=42).tolist()
trainer.predict_live(sample_tweets)

# Also predict some custom tweets
custom_tweets = [
    "@switchfoot that's a bummer you shoulda got David Carr",
    "is upset that he can't update Facebook might cry as result",
    "I dived many times for the ball managed to save 50%",
    "my whole body feels itchy and like its on fire",
    "Need a hug right now please",
    "hey long time no see! Rains a bit LOL I'm fine thanks",
    "spring break in plain city its snowing outside blah",
    "I just re-pierced my ears feeling awesome today!",
    "Hollis death scene will hurt me severely to watch on film",
    "I miss you so much cant believe you are gone",
]

print("\n📋 Custom Tweet Predictions:")
trainer.predict_live(custom_tweets)

# ════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════
elapsed = time.time() - t0
pos_pct = (df_clean['sentiment'] == 'Positive').mean() * 100
neg_pct = (df_clean['sentiment'] == 'Negative').mean() * 100
best_acc = trainer.results[trainer.best_name]['accuracy']

print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   🎊 PROJECT COMPLETE!                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ⏱️  Total Time    : {elapsed:.0f} seconds                         
║  📊 Dataset       : Sentiment140                            ║
║  📝 Tweets Used   : {len(df_clean):,}                              
║  😊 Positive      : {pos_pct:.1f}%                                 
║  😢 Negative      : {neg_pct:.1f}%                                 
║  🏆 Best Model    : {trainer.best_name}                     
║  🎯 Best Accuracy : {best_acc:.4f} ({best_acc*100:.2f}%)           
║                                                              ║
║  📁 Output Files:                                           ║
║     outputs/01_dashboard.png                                ║
║     outputs/02_wordclouds.png                               ║
║     outputs/03_analytics.png                                ║
║     outputs/04_model_results.png                            ║
║     data/cleaned_tweets.csv                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")