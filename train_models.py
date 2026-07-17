

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import time
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report,
                              roc_curve, auc)
from sklearn.preprocessing import LabelEncoder

# ── Color Theme ───────────────────────────────────────────────
BG   = '#0A0E1A'
SURF = '#1A1F35'
POS  = '#00D4AA'
NEG  = '#FF6B6B'
GOLD = '#FFD700'
ACC  = '#7B68EE'
WHITE= '#FFFFFF'
CYAN = '#00FFFF'

plt.rcParams.update({
    'figure.facecolor': BG,   'axes.facecolor': SURF,
    'text.color': WHITE,      'axes.labelcolor': WHITE,
    'xtick.color': WHITE,     'ytick.color': WHITE,
    'axes.edgecolor': ACC,    'grid.color': '#2A2F45',
    'grid.alpha': 0.4,
})

os.makedirs('outputs', exist_ok=True)

# ════════════════════════════════════════════════════════════
class Sentiment140Trainer:
    """Complete ML training pipeline for Sentiment140"""

    MODELS = {
        '🧠 Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, solver='lbfgs'
        ),
        '📊 Naive Bayes'  : MultinomialNB(alpha=0.1),
        '⚡ Linear SVM'   : LinearSVC(C=1.0, max_iter=2000, random_state=42),
        '🌲 Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    SHORT = {
        '🧠 Logistic Regression': 'Log. Reg',
        '📊 Naive Bayes'        : 'Naive Bayes',
        '⚡ Linear SVM'         : 'Linear SVM',
        '🌲 Random Forest'      : 'Rand. Forest',
    }

    MCOLORS = ['#00D4AA', '#FF9F43', '#7B68EE', '#54A0FF']

    def __init__(self, df):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'cleaned_text' and 'sentiment' columns
        """
        self.df         = df
        self.le         = LabelEncoder()
        self.results    = {}
        self.tfidf      = None
        self.best_name  = None
        self.best_model = None

        print("""
╔══════════════════════════════════════════════════════════════╗
║         🤖 SENTIMENT140 ML TRAINER                           ║
║         Models: LogReg | NB | SVM | RandomForest            ║
╚══════════════════════════════════════════════════════════════╝
        """)

    # ── prepare ───────────────────────────────────────────────
    def prepare(self):
        """Build TF-IDF matrix and train/test split"""
        print("📊 Preparing TF-IDF features...")

        X = self.df['cleaned_text']
        y = self.le.fit_transform(self.df['sentiment'])

        self.tfidf = TfidfVectorizer(
            max_features = 15000,
            ngram_range  = (1, 2),
            min_df       = 3,
            max_df       = 0.95,
            sublinear_tf = True,
        )

        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.Xtr = self.tfidf.fit_transform(self.X_train)
        self.Xte = self.tfidf.transform(self.X_test)

        print(f"  ✅ Train shape : {self.Xtr.shape}")
        print(f"  ✅ Test  shape : {self.Xte.shape}")
        print(f"  ✅ Vocabulary  : {len(self.tfidf.vocabulary_):,}")
        print(f"  ✅ Classes     : {list(self.le.classes_)}")
        return self

    # ── train ─────────────────────────────────────────────────
    def train(self):
        """Train all models and collect metrics"""
        print("\n" + "═"*55)
        print("🚀 TRAINING ALL MODELS")
        print("═"*55)

        for name, model in self.MODELS.items():
            print(f"\n  ▶ {name}")
            t0     = time.time()

            model.fit(self.Xtr, self.y_train)
            y_pred  = model.predict(self.Xte)
            elapsed = time.time() - t0

            acc = accuracy_score(self.y_test, y_pred)
            f1  = f1_score(self.y_test, y_pred, average='weighted')
            cv  = cross_val_score(
                model, self.Xtr, self.y_train,
                cv=5, scoring='accuracy', n_jobs=-1
            )

            # ROC AUC
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(self.Xte)[:, 1]
                elif hasattr(model, 'decision_function'):
                    prob = model.decision_function(self.Xte)
                else:
                    prob = y_pred.astype(float)
                fpr, tpr, _ = roc_curve(self.y_test, prob)
                roc_auc     = auc(fpr, tpr)
            except Exception:
                fpr = tpr = None
                roc_auc   = 0.0

            self.results[name] = {
                'model'  : model,
                'y_pred' : y_pred,
                'accuracy': acc,
                'f1'      : f1,
                'cv_mean' : cv.mean(),
                'cv_std'  : cv.std(),
                'time'    : elapsed,
                'cm'      : confusion_matrix(self.y_test, y_pred),
                'report'  : classification_report(
                    self.y_test, y_pred,
                    target_names=self.le.classes_
                ),
                'fpr'     : fpr,
                'tpr'     : tpr,
                'roc_auc' : roc_auc,
            }

            print(f"    Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
            print(f"    F1 Score : {f1:.4f}")
            print(f"    CV Score : {cv.mean():.4f} ± {cv.std():.4f}")
            print(f"    ROC AUC  : {roc_auc:.4f}")
            print(f"    Time     : {elapsed:.2f}s")

        self.best_name  = max(
            self.results, key=lambda k: self.results[k]['accuracy']
        )
        self.best_model = self.results[self.best_name]['model']

        print(f"\n🏆 BEST MODEL : {self.best_name}")
        print(f"   Accuracy   : {self.results[self.best_name]['accuracy']:.4f}")
        return self

    # ── visualize ─────────────────────────────────────────────
    def visualize(self):
        """Beautiful model comparison dashboard"""
        print("\n🎨 Creating model visualizations...")

        fig = plt.figure(figsize=(24, 20))
        fig.patch.set_facecolor(BG)
        gs  = gridspec.GridSpec(
            3, 4, figure=fig,
            hspace=0.5, wspace=0.4,
            top=0.93, bottom=0.05
        )

        import matplotlib.patheffects as pe
        fig.suptitle(
            '🤖 ML MODEL PERFORMANCE — SENTIMENT140',
            fontsize=22, fontweight='bold', color=GOLD, y=0.98,
            path_effects=[pe.withStroke(linewidth=3, foreground='black')]
        )

        names  = list(self.results.keys())
        snames = [self.SHORT[n] for n in names]
        mc     = self.MCOLORS
        accs   = [self.results[n]['accuracy'] for n in names]
        f1s    = [self.results[n]['f1']        for n in names]
        cvs    = [self.results[n]['cv_mean']   for n in names]
        cvstds = [self.results[n]['cv_std']    for n in names]
        times  = [self.results[n]['time']      for n in names]
        aucs   = [self.results[n]['roc_auc']   for n in names]

        # ── Row 0: Metric bars ────────────────────────────────
        # Accuracy
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(range(len(names)), accs,
                       color=mc, width=0.6, edgecolor=BG, linewidth=2)
        for bar, acc, name in zip(bars, accs, names):
            crown = '\n👑' if name == self.best_name else ''
            ax1.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{acc:.4f}{crown}',
                ha='center', va='bottom',
                color=WHITE, fontsize=9, fontweight='bold'
            )
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(snames, rotation=15, fontsize=9)
        ax1.set_title('🎯 Accuracy', color=GOLD, fontsize=13, fontweight='bold')
        ax1.set_ylim(0, max(accs) * 1.14)
        ax1.axhline(0.5, color=WHITE, ls='--', alpha=0.4, label='Baseline 50%')
        ax1.legend(labelcolor=WHITE, facecolor=SURF, fontsize=8)
        ax1.grid(axis='y', alpha=0.3)

        # F1 vs CV
        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(len(names)); w = 0.35
        ax2.bar(x - w/2, f1s, w, color=mc, label='F1 Score', alpha=0.9)
        ax2.bar(x + w/2, cvs, w,
                color=[c + '88' for c in mc],
                edgecolor=mc, linewidth=2, label='CV Score')
        ax2.errorbar(x + w/2, cvs, yerr=cvstds,
                     fmt='none', color=WHITE, capsize=5, linewidth=1.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(snames, rotation=15, fontsize=9)
        ax2.set_title('📊 F1 vs CV Score', color=GOLD, fontsize=13, fontweight='bold')
        ax2.legend(facecolor=SURF, labelcolor=WHITE, fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

        # ROC AUC
        ax3 = fig.add_subplot(gs[0, 2])
        bars = ax3.bar(range(len(names)), aucs,
                       color=mc, width=0.6, edgecolor=BG, linewidth=2)
        for bar, a in zip(bars, aucs):
            ax3.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{a:.4f}', ha='center', va='bottom',
                color=WHITE, fontsize=10, fontweight='bold'
            )
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(snames, rotation=15, fontsize=9)
        ax3.set_title('📈 ROC AUC', color=GOLD, fontsize=13, fontweight='bold')
        ax3.set_ylim(0, max(aucs) * 1.1)
        ax3.axhline(0.5, color=WHITE, ls='--', alpha=0.4)
        ax3.grid(axis='y', alpha=0.3)

        # Speed
        ax4 = fig.add_subplot(gs[0, 3])
        bars = ax4.barh(snames, times, color=mc, edgecolor=BG)
        for bar, t in zip(bars, times):
            ax4.text(
                bar.get_width() + max(times) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{t:.2f}s', va='center', color=WHITE, fontsize=10
            )
        ax4.set_title('⏱️ Training Time', color=GOLD, fontsize=13, fontweight='bold')
        ax4.set_xlabel('Seconds')
        ax4.grid(axis='x', alpha=0.3)

        # ── Row 1: Confusion Matrices ─────────────────────────
        for col_idx, (name, col) in enumerate(zip(names, mc)):
            ax = fig.add_subplot(gs[1, col_idx])
            cm     = self.results[name]['cm']
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

            ann = np.empty_like(cm, dtype=object)
            for i in range(2):
                for j in range(2):
                    ann[i, j] = f'{cm[i,j]:,}\n{cm_pct[i,j]:.0%}'

            sns.heatmap(
                cm_pct, ax=ax, annot=ann, fmt='',
                cmap=sns.light_palette(col, as_cmap=True),
                xticklabels=self.le.classes_,
                yticklabels=self.le.classes_,
                cbar=False, linewidths=3, linecolor=BG,
                annot_kws={'size': 12, 'weight': 'bold', 'color': WHITE}
            )
            ax.set_title(
                f'{self.SHORT[name]}\nAcc: {self.results[name]["accuracy"]:.4f}',
                color=col, fontsize=11, fontweight='bold'
            )
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('True',      fontsize=9)
            ax.tick_params(labelsize=9, labelcolor=WHITE)

        # ── Row 2: ROC Curves + Radar ─────────────────────────
        ax_roc = fig.add_subplot(gs[2, 0:2])
        ax_roc.plot([0, 1], [0, 1], 'w--', alpha=0.5, label='Random (AUC=0.50)')
        for name, col in zip(names, mc):
            r = self.results[name]
            if r['fpr'] is not None:
                ax_roc.plot(
                    r['fpr'], r['tpr'], color=col, lw=2.5,
                    label=f"{self.SHORT[name]} (AUC={r['roc_auc']:.4f})"
                )
        ax_roc.set_title('📈 ROC Curves', color=GOLD, fontsize=13, fontweight='bold')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(facecolor=SURF, edgecolor=ACC, labelcolor=WHITE, fontsize=10)
        ax_roc.grid(alpha=0.3)
        ax_roc.set_facecolor(SURF)

        # Radar
        ax_r = fig.add_subplot(gs[2, 2:], projection='polar')
        ax_r.set_facecolor(SURF)
        metrics = ['Accuracy', 'F1', 'CV Score', 'ROC AUC', 'Speed']
        N       = len(metrics)
        angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        max_t   = max(times) or 1

        for name, col in zip(names, mc):
            r  = self.results[name]
            vs = [
                r['accuracy'], r['f1'], r['cv_mean'],
                r['roc_auc'], 1 - r['time'] / max_t,
            ]
            vs = [max(0, min(1, v)) for v in vs] + [max(0, min(1, vs[0]))]
            ax_r.plot(angles, vs, 'o-', lw=2.5, color=col, label=self.SHORT[name])
            ax_r.fill(angles, vs, alpha=0.12, color=col)

        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(metrics, size=10, color=WHITE)
        ax_r.set_ylim(0, 1)
        ax_r.set_title('🕷️ Radar Performance',
                        color=GOLD, fontsize=13, fontweight='bold', y=1.12)
        ax_r.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1),
                    facecolor=SURF, labelcolor=WHITE, fontsize=10)
        ax_r.grid(color=WHITE, alpha=0.2)

        plt.savefig('outputs/04_model_results.png',
                    dpi=150, bbox_inches='tight', facecolor=BG)
        plt.show()
        print("   ✅ Saved: outputs/04_model_results.png")
        return self

    # ── reports ───────────────────────────────────────────────
    def print_reports(self):
        print("\n" + "═"*60)
        print("📋 FINAL RESULTS SUMMARY")
        print("═"*60)
        print(f"\n{'MODEL':<18} {'ACC':>8} {'F1':>8} "
              f"{'CV':>8} {'AUC':>8} {'TIME':>8}")
        print("─"*60)

        for name in sorted(self.results,
                           key=lambda k: self.results[k]['accuracy'],
                           reverse=True):
            r     = self.results[name]
            crown = ' 👑' if name == self.best_name else ''
            print(f"{self.SHORT[name]:<18} "
                  f"{r['accuracy']:>8.4f} {r['f1']:>8.4f} "
                  f"{r['cv_mean']:>8.4f} {r['roc_auc']:>8.4f} "
                  f"{r['time']:>7.2f}s{crown}")

        print(f"\n🏆 Best Model Report → {self.best_name}")
        print("─"*60)
        print(self.results[self.best_name]['report'])
        return self

    # ── live predict ──────────────────────────────────────────
    def predict_live(self, tweets):
        """Predict sentiment for new tweets"""
        from preprocess import Sentiment140Preprocessor
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        print("\n" + "═"*60)
        print("🔮 LIVE TWEET PREDICTION")
        print(f"   Best Model: {self.best_name}")
        print("═"*60)

        prep    = Sentiment140Preprocessor()
        vader   = SentimentIntensityAnalyzer()
        em      = {'Positive': '😊', 'Negative': '😢'}

        cleaned  = [prep.clean(t) for t in tweets]
        features = self.tfidf.transform(cleaned)
        preds    = self.best_model.predict(features)
        labels   = self.le.inverse_transform(preds)

        print(f"\n{'#':<3} {'TWEET':<45} {'ML':>10} {'VADER':>10}")
        print("─"*72)

        for i, (tweet, label) in enumerate(zip(tweets, labels), 1):
            vs = vader.polarity_scores(tweet)['compound']
            vl = 'Positive' if vs > 0.05 else 'Negative'
            match = '✅' if label == vl else '⚠️'
            print(f"{i:<3} {tweet[:43]:<45} "
                  f"{em[label]} {label:>8} "
                  f"{em[vl]} {vl:>8}  {match}")

        return labels


# ════════════════════════════════════════════════════════════
# ONLY runs when: python train_models.py
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from load_data   import load_sentiment140
    from preprocess  import Sentiment140Preprocessor

    # Load
    df = load_sentiment140(
        'data/training.1600000.processed.noemoticon.csv',
        sample_size=50000
    )

    # Preprocess
    prep     = Sentiment140Preprocessor()
    df_clean = prep.process_dataframe(df.copy(), text_col='text')

    # Train
    trainer = Sentiment140Trainer(df_clean)
    trainer.prepare().train().visualize().print_reports()

    # Predict
    test_tweets = [
        "@switchfoot that's a bummer you shoulda got David Carr",
        "is upset that he can't update his Facebook might cry",
        "I dived many times for the ball. Managed to save 50%",
        "my whole body feels itchy and like its on fire",
        "Need a hug",
        "hey long time no see! Rains a bit LOL I'm fine thanks",
        "spring break in plain city its snowing",
        "I just re-pierced my ears feeling great!",
    ]
    trainer.predict_live(test_tweets)