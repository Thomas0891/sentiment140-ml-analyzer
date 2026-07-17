

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("""
╔══════════════════════════════════════════════════════════════╗
║         📥 LOADING SENTIMENT140 DATASET                      ║
╚══════════════════════════════════════════════════════════════╝
""")

# ════════════════════════════════════════════════════════════
# LOAD DATASET - Handles your exact format
# ════════════════════════════════════════════════════════════

def load_sentiment140(filepath, sample_size=50000):
    """
    Load Sentiment140 dataset
    
    Your file columns:
    polarity | id | date | query | user | text
    
    polarity: 0 = Negative, 4 = Positive
    """
    
    print(f"📂 Loading: {filepath}")
    print(f"📊 Sample size: {sample_size:,} tweets")
    print("⏳ Please wait...\n")
    
    # ── Find your file ────────────────────────────────────────
    possible_paths = [
        filepath,
        'data/training.1600000.processed.noemoticon.csv',
        'training.1600000.processed.noemoticon.csv',
        'data/sentiment140.csv',
        'sentiment140.csv',
    ]
    
    actual_path = None
    for path in possible_paths:
        if os.path.exists(path):
            actual_path = path
            print(f"✅ Found file: {path}")
            file_size = os.path.getsize(path) / (1024**2)
            print(f"📦 File size: {file_size:.1f} MB")
            break
    
    if actual_path is None:
        print("❌ Dataset file not found!")
        print("\n📥 Download from:")
        print("   https://www.kaggle.com/datasets/kazanova/sentiment140")
        print("\n📁 Place file in: data/ folder")
        print("   OR same folder as this script")
        raise FileNotFoundError("Sentiment140 file not found!")
    
    # ── Load the CSV ──────────────────────────────────────────
    print("\n📖 Reading CSV file...")
    
    df = pd.read_csv(
        actual_path,
        encoding    = 'latin-1',      # IMPORTANT for this dataset!
        header      = None,            # No header row
        names       = ['polarity', 'id', 'date',
                       'query', 'user', 'text'],
        dtype       = {
            'polarity': int,
            'id'      : str,
            'date'    : str,
            'query'   : str,
            'user'    : str,
            'text'    : str,
        }
    )
    
    print(f"✅ Total rows loaded: {len(df):,}")
    print(f"📊 Columns: {list(df.columns)}")
    
    # ── Show exactly what your data looks like ─────────────────
    print("\n📋 YOUR DATA (first 5 rows):")
    print("─"*80)
    for i, row in df.head(5).iterrows():
        print(f"Row {i}:")
        print(f"  polarity : {row['polarity']}")
        print(f"  id       : {row['id']}")
        print(f"  date     : {row['date']}")
        print(f"  user     : {row['user']}")
        print(f"  text     : {row['text'][:60]}...")
        print()
    
    # ── Map polarity to labels ────────────────────────────────
    print("🏷️  Mapping labels...")
    print("   0 → Negative")
    print("   4 → Positive")
    
    df['sentiment'] = df['polarity'].map({
        0: 'Negative',
        4: 'Positive'
    })
    
    # Remove any rows with unmapped sentiment
    df = df[df['sentiment'].notna()]
    df = df[df['text'].notna()]
    df = df[df['text'].str.strip() != '']
    
    # ── Check original distribution ───────────────────────────
    print(f"\n📊 Original Distribution:")
    dist = df['sentiment'].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        bar = '█' * int(pct / 2)
        print(f"  {label:<12}: {count:>8,} ({pct:.1f}%) {bar}")
    
    # ── Sample balanced dataset ───────────────────────────────
    print(f"\n⚖️  Creating balanced sample of {sample_size:,} tweets...")
    
    per_class  = sample_size // 2
    
    neg_sample = df[df['sentiment'] == 'Negative'].sample(
        min(per_class, len(df[df['sentiment']=='Negative'])),
        random_state=42
    )
    pos_sample = df[df['sentiment'] == 'Positive'].sample(
        min(per_class, len(df[df['sentiment']=='Positive'])),
        random_state=42
    )
    
    df_sample = pd.concat(
        [neg_sample, pos_sample]
    ).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # ── Add useful columns ────────────────────────────────────
    df_sample['text_length']  = df_sample['text'].str.len()
    df_sample['word_count']   = df_sample['text'].apply(
        lambda x: len(str(x).split())
    )
    
    # Parse dates safely
    try:
        df_sample['parsed_date'] = pd.to_datetime(
            df_sample['date'],
            format='%a %b %d %H:%M:%S PDT %Y',
            errors='coerce'
        )
        df_sample['hour']        = df_sample['parsed_date'].dt.hour
        df_sample['month']       = df_sample['parsed_date'].dt.month
        df_sample['day_of_week'] = df_sample['parsed_date'].dt.day_name()
        print("✅ Dates parsed successfully!")
    except Exception as e:
        print(f"⚠️  Date parsing: {e}")
        df_sample['hour']        = np.random.randint(0, 24, len(df_sample))
        df_sample['month']       = np.random.randint(1, 13, len(df_sample))
        df_sample['day_of_week'] = 'Monday'
    
    # Add engagement metrics (simulated - not in original data)
    np.random.seed(42)
    df_sample['likes']    = np.random.randint(0, 5000, len(df_sample))
    df_sample['retweets'] = np.random.randint(0, 2000, len(df_sample))
    
    # Save
    os.makedirs('data', exist_ok=True)
    df_sample.to_csv('data/sentiment140_sample.csv', index=False)
    
    # ── Final Summary ─────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              ✅ DATASET LOADED SUCCESSFULLY!                  ║
╠══════════════════════════════════════════════════════════════╣
║  Total loaded   : {len(df):>10,} tweets                      
║  Sample size    : {len(df_sample):>10,} tweets (balanced)    
║  Negative       : {len(df_sample[df_sample['sentiment']=='Negative']):>10,} tweets  
║  Positive       : {len(df_sample[df_sample['sentiment']=='Positive']):>10,} tweets  
║  Avg length     : {df_sample['text_length'].mean():>10.1f} chars   
║  Avg words      : {df_sample['word_count'].mean():>10.1f} words   
║  Saved to       : data/sentiment140_sample.csv               
╚══════════════════════════════════════════════════════════════╝
    """)
    
    return df_sample


# ════════════════════════════════════════════════════════════
# LOAD YOUR DATA
# Change sample_size based on your computer speed:
# Fast PC  → 100000
# Normal   → 50000  ← recommended
# Slow PC  → 10000
# ════════════════════════════════════════════════════════════

df = load_sentiment140(
    filepath    = 'data/training.1600000.processed.noemoticon.csv',
    sample_size = 50000
)

print("Sample tweets from YOUR dataset:")
print("─"*70)
for i in range(5):
    row = df.iloc[i]
    emoji = "😊" if row['sentiment'] == 'Positive' else "😢"
    print(f"{emoji} [{row['sentiment']}] {row['text'][:65]}")
print("─"*70)