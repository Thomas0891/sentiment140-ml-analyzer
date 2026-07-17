

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re
import os
import random
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

# ════════════════════════════════════════════════════════════
# 🎨 COLOR THEME
# ════════════════════════════════════════════════════════════
BG     = '#0A0E1A'
SURF   = '#1A1F35'
POS    = '#00D4AA'
NEG    = '#FF6B6B'
GOLD   = '#FFD700'
CYAN   = '#00FFFF'
ACC    = '#7B68EE'
WHITE  = '#FFFFFF'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor'  : SURF,
    'text.color'      : WHITE,
    'axes.labelcolor' : WHITE,
    'xtick.color'     : WHITE,
    'ytick.color'     : WHITE,
    'axes.edgecolor'  : ACC,
    'grid.color'      : '#2A2F45',
    'grid.alpha'      : 0.4,
    'font.size'       : 10,
})


# ════════════════════════════════════════════════════════════
# CHART 1: MAIN DASHBOARD
# ════════════════════════════════════════════════════════════
def create_dashboard(df):
    print("\n🎨 Creating Dashboard...")
    
    fig = plt.figure(figsize=(24, 18))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.45, wspace=0.35,
        top=0.93, bottom=0.05,
        left=0.05, right=0.97
    )
    
    # ── Title ─────────────────────────────────────────────────
    fig.text(
        0.5, 0.97,
        '🐦 SENTIMENT140 ANALYSIS DASHBOARD',
        ha='center', fontsize=24, fontweight='bold',
        color=GOLD,
        path_effects=[pe.withStroke(linewidth=4, foreground='black')]
    )
    fig.text(
        0.5, 0.945,
        f'Dataset: Sentiment140 | {len(df):,} Tweets | '
        f'Binary Classification: Positive vs Negative',
        ha='center', fontsize=12, color=CYAN, alpha=0.85
    )
    
    counts = df['sentiment'].value_counts()
    colors = [POS if s == 'Positive' else NEG for s in counts.index]
    
    # ── PANEL 1: Big Donut Chart ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    
    wedges, _, autotexts = ax1.pie(
        counts.values,
        autopct    ='%1.1f%%',
        colors     = colors,
        startangle = 90,
        pctdistance= 0.75,
        wedgeprops = dict(
            width=0.6,
            edgecolor=BG,
            linewidth=4
        ),
        explode=[0.06, 0.06],
        shadow=True
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(13)
        at.set_fontweight('bold')
    
    ax1.text(0, 0.12, str(len(df)), ha='center',
             fontsize=18, fontweight='bold', color=GOLD)
    ax1.text(0, -0.18, 'Total Tweets', ha='center',
             fontsize=9, color=WHITE, alpha=0.7)
    
    ax1.set_title('📊 Sentiment\nDistribution',
                  color=GOLD, fontsize=13, fontweight='bold', pad=12)
    
    patches = [
        mpatches.Patch(color=POS, label=f'Positive ({counts.get("Positive",0):,})'),
        mpatches.Patch(color=NEG, label=f'Negative ({counts.get("Negative",0):,})'),
    ]
    ax1.legend(
        handles=patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.12),
        facecolor=SURF, edgecolor=ACC,
        labelcolor=WHITE, fontsize=10
    )
    
    # ── PANEL 2: Bar Chart ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    
    bars = ax2.bar(
        counts.index, counts.values,
        color=colors, width=0.5,
        edgecolor=BG, linewidth=3
    )
    for bar, val in zip(bars, counts.values):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + counts.values.max() * 0.02,
            f'{val:,}\n({val/len(df)*100:.1f}%)',
            ha='center', va='bottom',
            color=WHITE, fontsize=11, fontweight='bold'
        )
    
    ax2.set_title('📈 Tweet Count',
                  color=GOLD, fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(counts.values) * 1.2)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_facecolor(SURF)
    
    # ── PANEL 3: Tweet Length Histogram ───────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    
    for sentiment, color in [('Positive', POS), ('Negative', NEG)]:
        if sentiment in df['sentiment'].values:
            data = df[df['sentiment']==sentiment]['text_length']
            ax3.hist(data, bins=30, alpha=0.65,
                     color=color, label=sentiment,
                     edgecolor='none')
            ax3.axvline(
                data.mean(), color=color,
                linestyle='--', linewidth=2.5,
                label=f'{sentiment} avg: {data.mean():.0f}'
            )
    
    ax3.set_title('📏 Tweet Length\nDistribution',
                  color=GOLD, fontsize=13, fontweight='bold')
    ax3.set_xlabel('Characters', fontsize=10)
    ax3.set_ylabel('Frequency',  fontsize=10)
    ax3.legend(
        facecolor=SURF, edgecolor=ACC,
        labelcolor=WHITE, fontsize=8
    )
    ax3.grid(alpha=0.3)
    ax3.set_facecolor(SURF)
    
    # ── PANEL 4: KPI Cards ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_facecolor(BG)
    
    pos_count = counts.get('Positive', 0)
    neg_count = counts.get('Negative', 0)
    
    kpis = [
        ('🐦', 'Total Tweets',  f'{len(df):,}', ACC),
        ('😊', 'Positive',      f'{pos_count:,}', POS),
        ('😢', 'Negative',      f'{neg_count:,}', NEG),
        ('📝', 'Avg Length',
         f'{df["text_length"].mean():.0f} chars', CYAN),
    ]
    
    ax4.set_title('🎯 Key Metrics',
                  color=GOLD, fontsize=13, fontweight='bold')
    
    for i, (icon, label, val, col) in enumerate(kpis):
        y = 8.2 - i * 2.1
        rect = FancyBboxPatch(
            (0.3, y - 0.75), 9.4, 1.5,
            boxstyle='round,pad=0.2',
            facecolor=col + '20',
            edgecolor=col, linewidth=2,
            transform=ax4.transData
        )
        ax4.add_patch(rect)
        ax4.text(1.2, y + 0.1, icon,
                 fontsize=18, va='center', ha='center')
        ax4.text(2.5, y + 0.25, label,
                 fontsize=9, color=WHITE, alpha=0.7)
        ax4.text(2.5, y - 0.25, val,
                 fontsize=14, color=col, fontweight='bold')
    
    # ── PANEL 5: VADER Score Distribution ─────────────────────
    ax5 = fig.add_subplot(gs[1, 0:2])
    
    for sentiment, color in [('Positive', POS), ('Negative', NEG)]:
        mask = df['sentiment'] == sentiment
        vals = sorted(df[mask]['vader_compound'].values)
        prob = np.linspace(0, 1, len(vals))
        ax5.plot(vals, prob, color=color, lw=2.5,
                 label=f'{sentiment}')
        ax5.fill_between(vals, prob, alpha=0.1, color=color)
    
    ax5.axvline(0, color=WHITE, ls='--', alpha=0.4, lw=1.5)
    ax5.axvline(0.05,  color=POS, ls=':', alpha=0.5,
                label='Pos threshold')
    ax5.axvline(-0.05, color=NEG, ls=':', alpha=0.5,
                label='Neg threshold')
    
    ax5.set_title('📉 VADER Compound Score CDF',
                  color=GOLD, fontsize=13, fontweight='bold')
    ax5.set_xlabel('VADER Compound Score')
    ax5.set_ylabel('Cumulative Probability')
    ax5.legend(
        facecolor=SURF, edgecolor=ACC,
        labelcolor=WHITE, fontsize=9
    )
    ax5.set_xlim(-1, 1)
    ax5.grid(alpha=0.3)
    ax5.set_facecolor(SURF)
    
    # ── PANEL 6: Word Count Distribution ──────────────────────
    ax6 = fig.add_subplot(gs[1, 2:4])
    
    for sentiment, color in [('Positive', POS), ('Negative', NEG)]:
        mask = df['sentiment'] == sentiment
        data = df[mask]['word_count']
        ax6.hist(data, bins=25, alpha=0.65,
                 color=color, label=sentiment)
        ax6.axvline(
            data.mean(), color=color,
            linestyle='--', linewidth=2.5
        )
    
    ax6.set_title('📝 Word Count Distribution',
                  color=GOLD, fontsize=13, fontweight='bold')
    ax6.set_xlabel('Words per Tweet')
    ax6.set_ylabel('Frequency')
    ax6.legend(
        facecolor=SURF, edgecolor=ACC,
        labelcolor=WHITE, fontsize=10
    )
    ax6.grid(alpha=0.3)
    ax6.set_facecolor(SURF)
    
    # ── PANEL 7: TextBlob Polarity Scatter ────────────────────
    ax7 = fig.add_subplot(gs[2, 0:2])
    
    for sentiment, color in [('Positive', POS), ('Negative', NEG)]:
        mask = df['sentiment'] == sentiment
        sample = df[mask].sample(min(500, mask.sum()), random_state=42)
        ax7.scatter(
            sample['tb_polarity'],
            sample['tb_subjectivity'],
            c=color, alpha=0.4, s=15,
            label=sentiment
        )
    
    ax7.axvline(0, color=WHITE, ls='--', alpha=0.3)
    ax7.axhline(0.5, color=WHITE, ls='--', alpha=0.3)
    
    ax7.text(0.6, 0.95, 'Positive\nOpinions',
             transform=ax7.transAxes, fontsize=8,
             color=POS, ha='center', va='top', alpha=0.8)
    ax7.text(0.1, 0.95, 'Negative\nOpinions',
             transform=ax7.transAxes, fontsize=8,
             color=NEG, ha='center', va='top', alpha=0.8)
    
    ax7.set_title('🔬 Polarity vs Subjectivity (TextBlob)',
                  color=GOLD, fontsize=13, fontweight='bold')
    ax7.set_xlabel('Polarity')
    ax7.set_ylabel('Subjectivity')
    ax7.legend(
        facecolor=SURF, edgecolor=ACC,
        labelcolor=WHITE, fontsize=10
    )
    ax7.grid(alpha=0.2)
    ax7.set_facecolor(SURF)
    
    # ── PANEL 8: Top Users ─────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2:4])
    
    if 'user' in df.columns:
        top_users = df['user'].value_counts().head(12)
        user_colors = [
            POS if df[df['user']==u]['sentiment'].mode()[0]=='Positive'
            else NEG
            for u in top_users.index
        ]
        
        bars = ax8.barh(
            top_users.index[::-1],
            top_users.values[::-1],
            color=user_colors[::-1],
            edgecolor=BG, linewidth=1
        )
        for bar, val in zip(bars, top_users.values[::-1]):
            ax8.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height()/2,
                str(val), va='center',
                color=WHITE, fontsize=9
            )
        
        ax8.set_title('👤 Most Active Users',
                      color=GOLD, fontsize=13, fontweight='bold')
        ax8.set_xlabel('Tweet Count')
        ax8.grid(axis='x', alpha=0.3)
        ax8.set_facecolor(SURF)
    
    plt.savefig(
        'outputs/01_dashboard.png',
        dpi=150, bbox_inches='tight',
        facecolor=BG
    )
    plt.show()
    print("   ✅ Saved: outputs/01_dashboard.png")


# ════════════════════════════════════════════════════════════
# CHART 2: DUAL WORD CLOUDS
# ════════════════════════════════════════════════════════════
def create_wordclouds(df):
    print("\n☁️  Creating Word Clouds...")
    
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.patch.set_facecolor(BG)
    
    fig.suptitle(
        '☁️  SENTIMENT WORD CLOUDS  ☁️',
        fontsize=28, fontweight='bold',
        color=GOLD, y=1.02,
        path_effects=[pe.withStroke(linewidth=4, foreground='black')]
    )
    fig.text(
        0.5, 0.97,
        'Most Frequently Used Words in Positive vs Negative Tweets',
        ha='center', fontsize=13, color=CYAN, alpha=0.85
    )
    
    configs = [
        {
            'sentiment': 'Positive',
            'emoji'    : '😊',
            'bg'       : '#041A0F',
            'colors'   : ['#00FF7F','#00FA9A','#7FFF00',
                          '#ADFF2F','#00D4AA','#50C878',
                          '#90EE90','#00CED1','#3CB371'],
        },
        {
            'sentiment': 'Negative',
            'emoji'    : '😢',
            'bg'       : '#1A0404',
            'colors'   : ['#FF4500','#FF6347','#FF0000',
                          '#DC143C','#B22222','#FF69B4',
                          '#FF1493','#C71585','#FF8C00'],
        },
    ]
    
    for ax, cfg in zip(axes, configs):
        text_data = df[df['sentiment'] == cfg['sentiment']
                       ]['cleaned_text'].dropna()
        text = ' '.join(text_data.values)
        
        if not text.strip():
            ax.text(0.5, 0.5, 'No Data',
                    ha='center', va='center',
                    color=WHITE, transform=ax.transAxes)
            continue
        
        palette = cfg['colors']
        def make_color_func(pal):
            def color_func(*a, **kw):
                return random.choice(pal)
            return color_func
        
        wc = WordCloud(
            width           = 1400,
            height          = 800,
            background_color= cfg['bg'],
            max_words       = 150,
            color_func      = make_color_func(palette),
            prefer_horizontal=0.7,
            min_font_size   = 8,
            max_font_size   = 150,
            relative_scaling= 0.4,
            stopwords       = STOPWORDS,
            collocations    = False,
        ).generate(text)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        
        color = POS if cfg['sentiment'] == 'Positive' else NEG
        n     = len(text_data)
        
        ax.set_title(
            f"{cfg['emoji']}  {cfg['sentiment'].upper()} TWEETS",
            color=color, fontsize=22,
            fontweight='bold', pad=18,
            path_effects=[pe.withStroke(linewidth=3,
                                         foreground='black')]
        )
        ax.text(
            0.98, 0.02,
            f'n = {n:,} tweets',
            transform=ax.transAxes,
            ha='right', va='bottom',
            color=WHITE, fontsize=12, fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor=color+'33',
                edgecolor=color, linewidth=2
            )
        )
        ax.set_facecolor(cfg['bg'])
    
    plt.tight_layout(pad=2.5)
    plt.savefig(
        'outputs/02_wordclouds.png',
        dpi=150, bbox_inches='tight',
        facecolor=BG
    )
    plt.show()
    print("   ✅ Saved: outputs/02_wordclouds.png")


# ════════════════════════════════════════════════════════════
# CHART 3: DEEP ANALYTICS
# ════════════════════════════════════════════════════════════
def create_analytics(df):
    print("\n🔬 Creating Analytics Charts...")
    
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        top=0.93, bottom=0.07
    )
    
    fig.suptitle(
        '🔬 DEEP SENTIMENT ANALYTICS',
        fontsize=20, fontweight='bold',
        color=GOLD, y=0.98
    )
    
    # ── Violin Plot ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    
    pos_vc = df[df['sentiment']=='Positive']['vader_compound'].values
    neg_vc = df[df['sentiment']=='Negative']['vader_compound'].values
    
    parts = ax1.violinplot(
        [pos_vc, neg_vc],
        positions=[1, 2],
        showmeans=True,
        showmedians=True
    )
    
    colors_v = [POS, NEG]
    for pc, col in zip(parts['bodies'], colors_v):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)
        pc.set_edgecolor(WHITE)
        pc.set_linewidth(1.5)
    
    parts['cmeans'].set_color(GOLD)
    parts['cmeans'].set_linewidth(2.5)
    parts['cmedians'].set_color(WHITE)
    parts['cmedians'].set_linewidth(2)
    
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['😊 Positive', '😢 Negative'])
    ax1.set_title('🎻 VADER Score Violin',
                  color=GOLD, fontsize=13, fontweight='bold')
    ax1.set_ylabel('Compound Score')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_facecolor(SURF)
    
    # ── Box Plot ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    
    bp = ax2.boxplot(
        [pos_vc, neg_vc],
        patch_artist=True,
        notch=True,
        medianprops=dict(color=GOLD, linewidth=2.5),
        whiskerprops=dict(color=WHITE, linewidth=1.5),
        capprops=dict(color=WHITE, linewidth=2),
        flierprops=dict(
            marker='o',
            markerfacecolor=WHITE,
            markersize=3,
            alpha=0.3
        )
    )
    
    for patch, col in zip(bp['boxes'], [POS, NEG]):
        patch.set_facecolor(col + '77')
        patch.set_edgecolor(col)
        patch.set_linewidth(2.5)
    
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['😊 Positive', '😢 Negative'])
    ax2.set_title('📦 VADER Score Box Plot',
                  color=GOLD, fontsize=13, fontweight='bold')
    ax2.set_ylabel('Compound Score')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_facecolor(SURF)
    
    # ── Top Words Comparison ──────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    
    def get_top_words(sentiment, n=12):
        words = ' '.join(
            df[df['sentiment']==sentiment]['cleaned_text'].values
        ).split()
        return Counter(w for w in words if len(w)>3).most_common(n)
    
    pos_words = get_top_words('Positive', 10)
    neg_words = get_top_words('Negative', 10)
    
    y = np.arange(10)
    w = 0.4
    
    if pos_words and neg_words:
        pw, pf = zip(*pos_words)
        nw, nf = zip(*neg_words)
        
        pf_norm = np.array(pf) / max(pf)
        nf_norm = np.array(nf) / max(nf)
        
        ax3.barh(y + w/2, pf_norm, w,
                 color=POS, alpha=0.85, label='Positive')
        ax3.barh(y - w/2, nf_norm, w,
                 color=NEG, alpha=0.85, label='Negative')
        
        ax3.set_yticks(y)
        ax3.set_yticklabels([
            f'{pw[i]} / {nw[i]}' for i in range(10)
        ], fontsize=8)
    
    ax3.set_title('🔤 Top Words Comparison\n(Positive / Negative)',
                  color=GOLD, fontsize=12, fontweight='bold')
    ax3.set_xlabel('Relative Frequency')
    ax3.legend(
        facecolor=SURF, edgecolor=ACC,
        labelcolor=WHITE, fontsize=9
    )
    ax3.grid(axis='x', alpha=0.3)
    ax3.set_facecolor(SURF)
    
    # ── Correlation Heatmap ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    
    num_cols = [
        'text_length', 'word_count',
        'n_hashtags', 'n_exclaim',
        'n_question', 'caps_ratio',
        'tb_polarity', 'tb_subjectivity',
        'vader_compound', 'vader_pos', 'vader_neg',
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, mask=mask, ax=ax4,
        cmap=sns.diverging_palette(250, 15, as_cmap=True),
        center=0, vmax=1, vmin=-1,
        square=True, linewidths=0.5,
        linecolor=BG,
        annot=True, fmt='.2f',
        annot_kws={'size': 7, 'color': WHITE},
        cbar_kws={'shrink': 0.8}
    )
    ax4.set_title('🔥 Feature Correlation',
                  color=GOLD, fontsize=13, fontweight='bold')
    ax4.tick_params(labelsize=7, labelcolor=WHITE)
    ax4.set_facecolor(SURF)
    
    # ── VADER vs TextBlob Agreement ───────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    
    df['vader_label'] = df['vader_compound'].apply(
        lambda x: 'Positive' if x > 0.05
        else 'Negative' if x < -0.05
        else 'Neutral'
    )
    
    agree_pos = (
        (df['sentiment']=='Positive') &
        (df['vader_label']=='Positive')
    ).sum()
    agree_neg = (
        (df['sentiment']=='Negative') &
        (df['vader_label']=='Negative')
    ).sum()
    disagree  = len(df) - agree_pos - agree_neg
    
    wedge_data   = [agree_pos, agree_neg, disagree]
    wedge_labels = ['ML=VADER\nPositive', 'ML=VADER\nNegative', 'Disagree']
    wedge_colors = [POS, NEG, ACC]
    
    wedges, texts, autotexts = ax5.pie(
        wedge_data,
        labels    = wedge_labels,
        autopct   ='%1.1f%%',
        colors    = wedge_colors,
        startangle= 90,
        wedgeprops= dict(
            edgecolor=BG, linewidth=3
        )
    )
    for at in autotexts:
        at.set_color(WHITE)
        at.set_fontsize(10)
        at.set_fontweight('bold')
    for t in texts:
        t.set_color(WHITE)
        t.set_fontsize(9)
    
    ax5.set_title('⚖️ Label vs VADER Agreement',
                  color=GOLD, fontsize=13, fontweight='bold')
    ax5.set_facecolor(SURF)
    
    # ── Hourly Trend ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    
    if 'hour' in df.columns:
        hourly = df.groupby(['hour','sentiment']).size().unstack(fill_value=0)
        for s, color in [('Positive', POS), ('Negative', NEG)]:
            if s in hourly.columns:
                ax6.plot(
                    hourly.index, hourly[s],
                    color=color, lw=2.5, label=s,
                    marker='o', markersize=5
                )
                ax6.fill_between(
                    hourly.index, hourly[s],
                    alpha=0.15, color=color
                )
    
    ax6.set_title('🕐 Hourly Tweet Pattern',
                  color=GOLD, fontsize=13, fontweight='bold')
    ax6.set_xlabel('Hour of Day')
    ax6.set_ylabel('Tweet Count')
    ax6.legend(
        facecolor=SURF, edgecolor=ACC,
        labelcolor=WHITE, fontsize=10
    )
    ax6.grid(alpha=0.3)
    ax6.set_xticks(range(0, 24, 2))
    ax6.set_facecolor(SURF)
    
    plt.savefig(
        'outputs/03_analytics.png',
        dpi=150, bbox_inches='tight',
        facecolor=BG
    )
    plt.show()
    print("   ✅ Saved: outputs/03_analytics.png")


# ════════════════════════════════════════════════════════════
# RUN ALL VISUALIZATIONS
# ════════════════════════════════════════════════════════════
# ============================================================
# Run only when this file is executed directly
# ============================================================

if __name__ == "__main__":
    from load_data import load_sentiment140
    from preprocess import Sentiment140Preprocessor

    df = load_sentiment140(
        "data/training.1600000.processed.noemoticon.csv",
        sample_size=50000
    )

    prep = Sentiment140Preprocessor()
    df_clean = prep.process_dataframe(df)

    create_dashboard(df_clean)
    create_wordclouds(df_clean)
    create_analytics(df_clean)