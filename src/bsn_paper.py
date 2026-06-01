"""
BSN Paper v3 — Causal-ICCM: Personalized Directed-Coupling Monitor
====================================================================
Clean final version. No fallback. Hybrid coupling. Frobenius distance.
Confidence-threshold baseline. Random-abstention baseline.

Core contribution:
  Structural ambiguity is a measurable pre-inference risk factor for
  subject-specific failure in wearable stress classification. We detect
  it using personalized hybrid directed-coupling monitoring.

Method:
  - Hybrid coupling vector: correlation + max-lag cross-correlation
    + Granger-style directed coupling (-log p), 9 features total
  - Frobenius distance from subject's resting baseline (WESAD)
  - Direction-aware cosine divergence (Stress-Predict)
  - Conformal p-value, 3-zone gate: CLASSIFY / DEFER / ABSTAIN
  - No fallback classifier

Ablation:
  Corr-ICCM   | correlation only      (3 features)
  Lag-ICCM    | corr + max-lag        (6 features)
  Causal-ICCM | corr + directed       (6 features)
  Hybrid-ICCM | corr + lag + directed (9 features)  ← proposed

Baselines at matched coverage:
  - Full RF (no gating)
  - Random abstention
  - RF confidence threshold

Run:
  python bsn_paper_v3.py \
      --wesad_path  data/WESAD \
      --stress_path data/Stress-Predict-Dataset/Raw_data

  Add --skip_ablation for fast run (main method only).
"""

import os, pickle, argparse, warnings
import numpy as np
import pandas as pd
from scipy import signal as sp_signal, stats
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not found. Directed coupling uses lagged-correlation fallback.")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

EDA_FS   = 4
BVP_FS   = 64
LABEL_FS = 700
WIN_N    = 240        # 60 s × 4 Hz
STEP_N   = 120        # 30 s × 4 Hz
ALPHA    = 0.05
GC_MAXLAG = 2
EPS      = 1e-8

C_PURPLE = '#693C7E'
C_ORANGE = '#E67E22'
C_DARK   = '#2C3E50'
C_GRAY   = '#7F8C8D'
C_GRID   = '#F2F2F2'

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':        8,
    'axes.labelsize':   8.5,
    'axes.linewidth':   0.7,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize':  7.5,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.facecolor':'white',
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
})


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def sliding_hr(bvp, n):
    """Derive HR at EDA sampling rate from BVP via 5-second sliding window."""
    step = BVP_FS // EDA_FS
    w    = BVP_FS * 5
    out  = []
    for i in range(n):
        s, e = i * step, i * step + w
        if e <= len(bvp):
            pks, _ = sp_signal.find_peaks(bvp[s:e], distance=int(BVP_FS * 0.5))
            hr = (60.0 / np.mean(np.diff(pks) / BVP_FS)
                  if len(pks) >= 2 else (out[-1] if out else 70.0))
        else:
            hr = out[-1] if out else 70.0
        out.append(hr)
    return np.array(out, dtype=float)


def hr_from_bvp(seg, fs=BVP_FS):
    if len(seg) < fs: return 70.0
    pks, _ = sp_signal.find_peaks(seg, distance=int(fs * 0.5))
    return float(60.0 / np.mean(np.diff(pks) / fs)) if len(pks) >= 2 else 70.0


def hrv_rmssd(seg, fs=BVP_FS):
    if len(seg) < fs: return 0.0
    pks, _ = sp_signal.find_peaks(seg, distance=int(fs * 0.5))
    if len(pks) < 3: return 0.0
    return float(np.sqrt(np.mean(np.diff(np.diff(pks) / fs) ** 2))) * 1000


def safe_pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    k = min(len(a), len(b))
    if k < 3: return 0.0
    a, b = a[:k], b[:k]
    if np.std(a) < EPS or np.std(b) < EPS: return 0.0
    r, _ = pearsonr(a, b)
    return float(np.nan_to_num(r))


def cosine_sim(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / d) if d > EPS else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# COUPLING FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def max_lagged_crosscorr(a, b, lag_min_s, lag_max_s, fs=EDA_FS):
    """
    Maximum absolute cross-correlation over a physiological lag range.
    Avoids assuming a fixed delay — more robust than fixed-lag.
    EDA pairs: [1,10]s; HR-TEMP pair: [1,5]s.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    k = min(len(a), len(b))
    if k < 5: return abs(safe_pearson(a, b))
    a, b = a[:k], b[:k]
    lag_min = max(1, int(lag_min_s * fs))
    lag_max = min(k - 2, int(lag_max_s * fs))
    if lag_min >= lag_max: return abs(safe_pearson(a, b))
    return float(max(abs(safe_pearson(a[:-lag], b[lag:]))
                     for lag in range(lag_min, lag_max + 1)))


def granger_directed_score(cause, effect, maxlag=GC_MAXLAG):
    """
    Granger-style directed coupling from cause -> effect.
    Score = -log(min p-value across lags), clipped to [0,10], normalized to [0,1].
    Uses -log(p) — no arbitrary sigmoid scaling.
    Falls back to max-lagged correlation if statsmodels unavailable or fails.
    """
    cause  = np.asarray(cause,  dtype=float)
    effect = np.asarray(effect, dtype=float)
    k = min(len(cause), len(effect))
    if k < (maxlag + 5):
        return max_lagged_crosscorr(cause, effect, 1, 5)
    if not HAS_STATSMODELS:
        return max_lagged_crosscorr(cause, effect, 1, 5)
    try:
        data    = np.column_stack([effect[:k], cause[:k]])
        res     = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        p_vals  = [res[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)]
        p_min   = max(min(p_vals), EPS)
        return  float(min(-np.log(p_min), 10.0) / 10.0)
    except Exception:
        return max_lagged_crosscorr(cause, effect, 1, 5)


def _align(eda, bvp, temp):
    n = len(eda)
    hr = sliding_hr(bvp, n)
    k  = min(n, len(hr), len(temp))
    return eda[:k], hr[:k], temp[:k]


def coupling_vec_corr(eda, bvp, temp):
    """3 features: Pearson |correlation| for EDA-HR, EDA-TEMP, HR-TEMP."""
    eda_k, hr_k, temp_k = _align(eda, bvp, temp)
    return np.array([
        abs(safe_pearson(eda_k, hr_k)),
        abs(safe_pearson(eda_k, temp_k)),
        abs(safe_pearson(hr_k,  temp_k)),
    ], dtype=float)


def coupling_vec_lag(eda, bvp, temp):
    """
    6 features: correlation + max lagged cross-correlation.
    EDA pairs use physiological lag range [1,10]s.
    HR-TEMP pair uses [1,5]s.
    """
    eda_k, hr_k, temp_k = _align(eda, bvp, temp)
    corr = np.array([
        abs(safe_pearson(eda_k, hr_k)),
        abs(safe_pearson(eda_k, temp_k)),
        abs(safe_pearson(hr_k,  temp_k)),
    ], dtype=float)
    lag = np.array([
        max_lagged_crosscorr(eda_k, hr_k,   1, 10),
        max_lagged_crosscorr(eda_k, temp_k, 1, 10),
        max_lagged_crosscorr(hr_k,  temp_k, 1,  5),
    ], dtype=float)
    return np.concatenate([corr, lag])


def coupling_vec_causal(eda, bvp, temp):
    """
    6 features: correlation + Granger-style directed coupling (-log p).
    Directed pairs: EDA->HR, HR->EDA, TEMP->HR.
    """
    eda_k, hr_k, temp_k = _align(eda, bvp, temp)
    corr = np.array([
        abs(safe_pearson(eda_k, hr_k)),
        abs(safe_pearson(eda_k, temp_k)),
        abs(safe_pearson(hr_k,  temp_k)),
    ], dtype=float)
    gc = np.array([
        granger_directed_score(eda_k,  hr_k),
        granger_directed_score(hr_k,   eda_k),
        granger_directed_score(temp_k, hr_k),
    ], dtype=float)
    return np.concatenate([corr, gc])


def coupling_vec_hybrid(eda, bvp, temp):
    """
    9 features: correlation + max-lag cross-correlation + directed coupling.
    This is the proposed Causal-ICCM coupling vector.

    v(t) = [ρ(EDA,HR), ρ(EDA,TEMP), ρ(HR,TEMP),          <- correlation
             ℓ(EDA,HR), ℓ(EDA,TEMP), ℓ(HR,TEMP),         <- max-lag [1,10]s
             g(EDA->HR), g(HR->EDA), g(TEMP->HR)]          <- directed -log(p)
    """
    eda_k, hr_k, temp_k = _align(eda, bvp, temp)
    corr = np.array([
        abs(safe_pearson(eda_k, hr_k)),
        abs(safe_pearson(eda_k, temp_k)),
        abs(safe_pearson(hr_k,  temp_k)),
    ], dtype=float)
    lag = np.array([
        max_lagged_crosscorr(eda_k, hr_k,   1, 10),
        max_lagged_crosscorr(eda_k, temp_k, 1, 10),
        max_lagged_crosscorr(hr_k,  temp_k, 1,  5),
    ], dtype=float)
    gc = np.array([
        granger_directed_score(eda_k,  hr_k),
        granger_directed_score(hr_k,   eda_k),
        granger_directed_score(temp_k, hr_k),
    ], dtype=float)
    return np.concatenate([corr, lag, gc])


# ══════════════════════════════════════════════════════════════════════════════
# ICCM — FROBENIUS (WESAD, single-protocol)
# ══════════════════════════════════════════════════════════════════════════════

class ICCM_Frobenius:
    """
    Conformal coupling monitor using Frobenius distance from individual baseline.
    Appropriate for single-protocol datasets (WESAD).

    Calibrates on subject's resting-state windows.
    At inference time, computes D(t) = ||v(t) - v0||_2 and conformal p-value.

    3-Zone gate:
      Zone 1 (p >= alpha): CLASSIFY
      Zone 2 (alpha/2 <= p < alpha): DEFER
      Zone 3 (p < alpha/2): ABSTAIN
    """

    def __init__(self, vec_fn, alpha=ALPHA):
        self.vec_fn  = vec_fn
        self.alpha   = alpha
        self.v0      = None
        self.cal_D   = None

    def calibrate(self, baseline_windows):
        vecs      = [self.vec_fn(*w) for w in baseline_windows]
        self.v0   = np.mean(vecs, axis=0)
        self.cal_D = np.array([np.linalg.norm(v - self.v0) for v in vecs])

    def score(self, eda, bvp, temp):
        vt = self.vec_fn(eda, bvp, temp)
        Dt = np.linalg.norm(vt - self.v0)
        p  = (1 + np.sum(self.cal_D >= Dt)) / (len(self.cal_D) + 1)
        return float(p), float(Dt)

    def detect(self, eda, bvp, temp):
        p, _ = self.score(eda, bvp, temp)
        if   p >= self.alpha:          return 'CLASSIFY', p
        elif p >= self.alpha / 2:      return 'DEFER',    p
        else:                          return 'ABSTAIN',  p

    def rgraph(self, eda, bvp, temp, lbl, stress_label=1):
        """Subject-level ambiguity score: max coupling divergence during stress."""
        dists = []
        for s in range(0, len(eda) - WIN_N, STEP_N):
            if np.mean(lbl[s:s + WIN_N] == stress_label) >= 0.80:
                b_s = int(s / EDA_FS * BVP_FS)
                b_e = int((s + WIN_N) / EDA_FS * BVP_FS)
                _, D = self.score(eda[s:s+WIN_N], bvp[b_s:b_e], temp[s:s+WIN_N])
                dists.append(D)
        return float(np.max(dists)) if dists else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ICCM — DIRECTION-AWARE (Stress-Predict, multi-protocol)
# ══════════════════════════════════════════════════════════════════════════════

class ICCM_Direction:
    """
    Direction-aware coupling monitor for multi-protocol datasets.

    Uses D_dir(t) = 1 - cos(Δv(t), μ_Δ) where μ_Δ is the population mean
    coupling-change direction estimated from training subjects under LOSO.
    Separates atypical coupling direction from physiological reactivity magnitude.

    Ambiguity score R = mean D_dir during stress windows.
    """

    def __init__(self, vec_fn, alpha=ALPHA):
        self.vec_fn    = vec_fn
        self.alpha     = alpha
        self.v0        = None
        self.mu_delta  = None
        self.cal_cos   = None

    def calibrate_baseline(self, baseline_windows):
        vecs    = [self.vec_fn(*w) for w in baseline_windows]
        self.v0 = np.mean(vecs, axis=0)

    def calibrate_population(self, train_stress_vecs, train_v0s):
        deltas        = [v - v0 for v, v0 in zip(train_stress_vecs, train_v0s)]
        self.mu_delta = np.mean(deltas, axis=0)
        self.cal_cos  = np.array([cosine_sim(d, self.mu_delta) for d in deltas])

    def score(self, eda, bvp, temp):
        vt    = self.vec_fn(eda, bvp, temp)
        delta = vt - self.v0
        cos   = cosine_sim(delta, self.mu_delta)
        D_dir = 1.0 - cos
        p     = (1 + np.sum(self.cal_cos >= cos)) / (len(self.cal_cos) + 1)
        return float(p), float(D_dir)

    def detect(self, eda, bvp, temp):
        p, _ = self.score(eda, bvp, temp)
        if   p >= self.alpha:          return 'CLASSIFY', p
        elif p >= self.alpha / 2:      return 'DEFER',    p
        else:                          return 'ABSTAIN',  p

    def rgraph(self, eda, bvp, temp, lbl, stress_label=1):
        """Ambiguity score: mean direction divergence during stress."""
        vals = []
        for s in range(0, len(eda) - WIN_N, STEP_N):
            if np.mean(lbl[s:s + WIN_N] == stress_label) >= 0.80:
                b_s = int(s / EDA_FS * BVP_FS)
                b_e = int((s + WIN_N) / EDA_FS * BVP_FS)
                _, D = self.score(eda[s:s+WIN_N], bvp[b_s:b_e], temp[s:s+WIN_N])
                vals.append(D)
        return float(np.mean(vals)) if vals else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(eda_w, bvp_w, temp_w):
    """14 time-domain features (identical to original ICCM paper)."""
    f = []
    f += [np.mean(eda_w), np.std(eda_w), np.ptp(eda_w)]
    f.append(np.polyfit(np.arange(len(eda_w)), eda_w, 1)[0])
    pks, props = sp_signal.find_peaks(eda_w, prominence=0.005)
    f.append(float(len(pks)))
    f.append(float(np.mean(props['prominences'])) if len(pks) else 0.0)
    hr  = hr_from_bvp(bvp_w)
    hrv = hrv_rmssd(bvp_w)
    segs = [hr_from_bvp(bvp_w[i:i+BVP_FS*10])
            for i in range(0, max(0, len(bvp_w)-BVP_FS*10), BVP_FS*5)]
    f += [hr, hrv, float(np.std(segs)) if len(segs) > 1 else 0.0]
    f.append(np.mean(temp_w))
    f.append(np.polyfit(np.arange(len(temp_w)), temp_w, 1)[0])
    n = min(len(eda_w), len(temp_w))
    f.append(np.corrcoef(eda_w[:n], temp_w[:n])[0, 1])
    f += [np.mean(eda_w) / (np.mean(temp_w) + EPS),
          np.std(eda_w) * hr / 100.0]
    return np.nan_to_num(np.array(f, dtype=float))


def make_windows(eda, bvp, temp, lbl, stress_label=1, purity=0.80):
    X, y, starts = [], [], []
    valid = list(set(lbl[lbl >= 0]))
    for start in range(0, len(eda) - WIN_N, STEP_N):
        end = start + WIN_N
        seg = lbl[start:end]
        sv  = seg[seg >= 0]
        if not len(sv): continue
        counts = {l: np.sum(sv == l) for l in valid}
        maj    = max(counts, key=counts.get)
        if counts[maj] / len(seg) < purity: continue
        b_s = int(start / EDA_FS * BVP_FS)
        b_e = int(end   / EDA_FS * BVP_FS)
        X.append(extract_features(eda[start:end], bvp[b_s:b_e], temp[start:end]))
        y.append(1 if maj == stress_label else 0)
        starts.append(start)
    return np.array(X), np.array(y), np.array(starts)


def _baseline_windows(eda, bvp, temp, lbl, rest_label=0):
    wins = []
    for s in range(0, len(eda) - WIN_N, STEP_N):
        if np.mean(lbl[s:s+WIN_N] == rest_label) >= 0.80:
            b_s = int(s / EDA_FS * BVP_FS)
            b_e = int((s+WIN_N) / EDA_FS * BVP_FS)
            wins.append((eda[s:s+WIN_N], bvp[b_s:b_e], temp[s:s+WIN_N]))
    return wins


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_wesad(path):
    SIDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    sigs = {}
    for sid in SIDS:
        pkl = os.path.join(path, f'S{sid}', f'S{sid}.pkl')
        if not os.path.exists(pkl): continue
        print(f'  S{sid}...', end=' ', flush=True)
        with open(pkl, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        w       = data['signal']['wrist']
        eda     = w['EDA'].flatten().astype(float)
        bvp     = w['BVP'].flatten().astype(float)
        temp    = w['TEMP'].flatten().astype(float)
        lbl_raw = data['label'].flatten().astype(int)
        fac = LABEL_FS // EDA_FS
        n_l = len(lbl_raw) // fac
        lbl = np.array([stats.mode(lbl_raw[i*fac:(i+1)*fac],
                        keepdims=True).mode[0] for i in range(n_l)])
        n = min(len(eda), len(lbl), len(temp))
        lbl_bin = np.where(lbl[:n]==2, 1, np.where(lbl[:n]>=1, 0, -1))
        sigs[f'S{sid}'] = (eda[:n], bvp, temp[:n], lbl_bin)
        print('done')
    return sigs


def load_stress_predict(raw_path):
    STAGES = [('Baseline',0),('Stroop',1),('Rest1',0),
              ('Interview',1),('Rest2',0),('Hypervent',-1),('Rest3',0)]

    def read_e4(fp):
        df = pd.read_csv(fp, header=None, on_bad_lines='skip')
        ts = float(df.iloc[0,0])
        return df.iloc[2:].values.flatten().astype(float), ts

    def read_tags(fp):
        df = pd.read_csv(fp, header=None, on_bad_lines='skip')
        return np.sort(df.iloc[:,0].values.astype(float))

    def make_labels(tags, ts0, n):
        lbl = np.full(n, -1, dtype=int)
        st  = tags[:len(STAGES)]
        for i, ((nm, lab), ts_t) in enumerate(zip(STAGES, st)):
            s = max(0, int((ts_t - ts0) * EDA_FS))
            e = int((st[i+1]-ts0)*EDA_FS) if i+1<len(st) else n
            if s < n: lbl[s:min(e,n)] = lab
        return lbl

    sigs = {}
    for sid in sorted(os.listdir(raw_path)):
        d = os.path.join(raw_path, sid)
        if not os.path.isdir(d) or not sid.startswith('S'): continue
        print(f'  {sid}...', end=' ', flush=True)
        try:
            eda, ts = read_e4(os.path.join(d, 'EDA.csv'))
            bvp, _  = read_e4(os.path.join(d, 'BVP.csv'))
            tmp, _  = read_e4(os.path.join(d, 'TEMP.csv'))
            tags    = read_tags(os.path.join(d, f'tags_{sid}.csv'))
            n       = min(len(eda), len(tmp))
            lbl     = make_labels(tags, ts, n)
            sigs[sid] = (eda[:n], bvp, tmp[:n], lbl[:n])
            print('done')
        except Exception as e:
            print(f'ERROR {e}')
    return sigs


# ══════════════════════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════════════════════

def random_abstention_baseline(y_raw, y_pred, coverage_pct, seed=42):
    """Randomly keep coverage_pct% of windows. Matched-coverage baseline."""
    rng   = np.random.default_rng(seed)
    n     = len(y_raw)
    n_keep = max(6, int(round(n * coverage_pct / 100.0)))
    n_keep = min(n_keep, n)
    idx   = np.sort(rng.choice(n, size=n_keep, replace=False))
    if len(np.unique(y_raw[idx])) < 2:
        return None
    acc = accuracy_score(y_raw[idx], y_pred[idx])
    f1  = f1_score(y_raw[idx], y_pred[idx], zero_division=0)
    cm  = confusion_matrix(y_raw[idx], y_pred[idx], labels=[0,1])
    fp  = int(cm[0,1])
    return acc, f1, fp


def confidence_threshold_baseline(y_raw, y_pred, y_proba, coverage_pct):
    """Keep top-confidence predictions at matched coverage."""
    n     = len(y_raw)
    n_keep = max(6, int(round(n * coverage_pct / 100.0)))
    n_keep = min(n_keep, n)
    conf  = np.max(y_proba, axis=1)
    idx   = np.sort(np.argsort(conf)[-n_keep:])
    if len(np.unique(y_raw[idx])) < 2:
        return None
    acc = accuracy_score(y_raw[idx], y_pred[idx])
    f1  = f1_score(y_raw[idx], y_pred[idx], zero_division=0)
    cm  = confusion_matrix(y_raw[idx], y_pred[idx], labels=[0,1])
    fp  = int(cm[0,1])
    return acc, f1, fp


# ══════════════════════════════════════════════════════════════════════════════
# LOSO — FROBENIUS (WESAD)
# ══════════════════════════════════════════════════════════════════════════════

def run_loso_frobenius(all_sigs, vec_fn, variant_name,
                       stress_label=1, rest_label=0):
    """
    LOSO evaluation with Frobenius-distance ICCM.
    Used for WESAD (single-protocol).
    """
    sids = list(all_sigs.keys())
    rows = []

    for test_sid in sids:
        eda, bvp, temp, lbl = all_sigs[test_sid]
        X, y_raw, starts = make_windows(eda, bvp, temp, lbl, stress_label)
        if len(X) == 0 or len(np.unique(y_raw)) < 2: continue

        bw = _baseline_windows(eda, bvp, temp, lbl, rest_label)
        if len(bw) < 3: continue

        det = ICCM_Frobenius(vec_fn)
        det.calibrate(bw)
        R = det.rgraph(eda, bvp, temp, lbl, stress_label)

        # Train RF on all other subjects
        X_tr, y_tr = [], []
        for sid in sids:
            if sid == test_sid: continue
            Xs, ys, _ = make_windows(*all_sigs[sid][:3], all_sigs[sid][3], stress_label)
            if len(Xs) > 0: X_tr.append(Xs); y_tr.append(ys)
        if not X_tr: continue

        X_train = np.vstack(X_tr); y_train = np.concatenate(y_tr)
        sc  = StandardScaler()
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(sc.fit_transform(X_train), y_train)
        X_sc    = sc.transform(X)
        y_pred  = clf.predict(X_sc)
        y_proba = clf.predict_proba(X_sc)

        # Population model metrics
        acc = accuracy_score(y_raw, y_pred)
        f1  = f1_score(y_raw, y_pred, zero_division=0)
        cm  = confusion_matrix(y_raw, y_pred, labels=[0,1])
        fp  = int(cm[0,1])

        # ICCM gating
        decs = []
        for start in starts:
            b_s = int(start / EDA_FS * BVP_FS)
            b_e = int((start+WIN_N) / EDA_FS * BVP_FS)
            d, _ = det.detect(eda[start:start+WIN_N], bvp[b_s:b_e], temp[start:start+WIN_N])
            decs.append(d)
        decs = np.array(decs)

        classify_mask = (decs == 'CLASSIFY')
        n_classify    = classify_mask.sum()
        n_defer       = (decs == 'DEFER').sum()
        n_abstain     = (decs == 'ABSTAIN').sum()
        n_total       = len(decs)
        coverage      = round(100 * n_classify / max(n_total, 1), 1)

        if n_classify > 5 and len(np.unique(y_raw[classify_mask])) > 1:
            acc_g = accuracy_score(y_raw[classify_mask], y_pred[classify_mask])
            f1_g  = f1_score(y_raw[classify_mask], y_pred[classify_mask], zero_division=0)
            cm_g  = confusion_matrix(y_raw[classify_mask], y_pred[classify_mask], labels=[0,1])
            fp_g  = int(cm_g[0,1])
        else:
            acc_g, f1_g, fp_g = acc, f1, fp

        # Baselines at matched coverage
        rand = random_abstention_baseline(y_raw, y_pred, coverage)
        conf = confidence_threshold_baseline(y_raw, y_pred, y_proba, coverage)

        worse_strict = acc_g < acc
        worse_2pct   = acc_g < acc - 0.02
        pct_abst     = round(100 * n_abstain / max(n_total,1), 1)
        pct_defer    = round(100 * n_defer   / max(n_total,1), 1)

        print(f'  [{variant_name}] {str(test_sid):>5}: '
              f'pop={acc:.3f}(FP={fp})  '
              f'gated={acc_g:.3f}(FP={fp_g},abst={pct_abst}%,cov={coverage}%)  '
              f'R={R:.3f}{"  WORSE!" if worse_2pct else ""}')

        rows.append({
            'subject':      test_sid,
            'R_graph':      round(R, 4),
            'acc':          acc,
            'f1':           f1,
            'fp':           fp,
            'acc_g':        acc_g,
            'f1_g':         f1_g,
            'fp_g':         fp_g,
            'coverage':     coverage,
            'pct_abst':     pct_abst,
            'pct_defer':    pct_defer,
            'worse_strict': worse_strict,
            'worse_2pct':   worse_2pct,
            'rand_acc':     rand[0] if rand else np.nan,
            'rand_f1':      rand[1] if rand else np.nan,
            'rand_fp':      rand[2] if rand else np.nan,
            'conf_acc':     conf[0] if conf else np.nan,
            'conf_f1':      conf[1] if conf else np.nan,
            'conf_fp':      conf[2] if conf else np.nan,
            'variant':      variant_name,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# LOSO — DIRECTION-AWARE (Stress-Predict)
# ══════════════════════════════════════════════════════════════════════════════

def run_loso_direction(all_sigs, vec_fn, variant_name,
                       stress_label=1, rest_label=0):
    """
    LOSO evaluation with direction-aware ICCM.
    Used for Stress-Predict (multi-protocol).
    """
    sids = list(all_sigs.keys())
    rows = []

    for test_sid in sids:
        eda, bvp, temp, lbl = all_sigs[test_sid]
        X, y_raw, starts = make_windows(eda, bvp, temp, lbl, stress_label)
        if len(X) == 0 or len(np.unique(y_raw)) < 2: continue

        bw = _baseline_windows(eda, bvp, temp, lbl, rest_label)
        if len(bw) < 3: continue

        det = ICCM_Direction(vec_fn)
        det.calibrate_baseline(bw)

        # Collect population stress windows from training subjects
        train_vecs, train_v0s = [], []
        for sid in sids:
            if sid == test_sid: continue
            e, b, t, l = all_sigs[sid]
            bw2 = _baseline_windows(e, b, t, l, rest_label)
            if len(bw2) < 2: continue
            v0_tr = np.mean([vec_fn(*w) for w in bw2], axis=0)
            for s in range(0, len(e)-WIN_N, STEP_N):
                if np.mean(l[s:s+WIN_N]==stress_label) >= 0.80:
                    b_s = int(s/EDA_FS*BVP_FS)
                    b_e = int((s+WIN_N)/EDA_FS*BVP_FS)
                    train_vecs.append(vec_fn(e[s:s+WIN_N], b[b_s:b_e], t[s:s+WIN_N]))
                    train_v0s.append(v0_tr)
        if len(train_vecs) < 10: continue
        det.calibrate_population(train_vecs, train_v0s)
        R = det.rgraph(eda, bvp, temp, lbl, stress_label)

        # Train RF
        X_tr, y_tr = [], []
        for sid in sids:
            if sid == test_sid: continue
            Xs, ys, _ = make_windows(*all_sigs[sid][:3], all_sigs[sid][3], stress_label)
            if len(Xs) > 0: X_tr.append(Xs); y_tr.append(ys)
        if not X_tr: continue

        X_train = np.vstack(X_tr); y_train = np.concatenate(y_tr)
        sc  = StandardScaler()
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(sc.fit_transform(X_train), y_train)
        X_sc    = sc.transform(X)
        y_pred  = clf.predict(X_sc)
        y_proba = clf.predict_proba(X_sc)

        acc = accuracy_score(y_raw, y_pred)
        f1  = f1_score(y_raw, y_pred, zero_division=0)
        cm  = confusion_matrix(y_raw, y_pred, labels=[0,1])
        fp  = int(cm[0,1])

        decs = []
        for start in starts:
            b_s = int(start/EDA_FS*BVP_FS)
            b_e = int((start+WIN_N)/EDA_FS*BVP_FS)
            d, _ = det.detect(eda[start:start+WIN_N], bvp[b_s:b_e], temp[start:start+WIN_N])
            decs.append(d)
        decs = np.array(decs)

        classify_mask = (decs == 'CLASSIFY')
        n_classify    = classify_mask.sum()
        n_defer       = (decs == 'DEFER').sum()
        n_abstain     = (decs == 'ABSTAIN').sum()
        n_total       = len(decs)
        coverage      = round(100 * n_classify / max(n_total,1), 1)

        if n_classify > 5 and len(np.unique(y_raw[classify_mask])) > 1:
            acc_g = accuracy_score(y_raw[classify_mask], y_pred[classify_mask])
            f1_g  = f1_score(y_raw[classify_mask], y_pred[classify_mask], zero_division=0)
            cm_g  = confusion_matrix(y_raw[classify_mask], y_pred[classify_mask], labels=[0,1])
            fp_g  = int(cm_g[0,1])
        else:
            acc_g, f1_g, fp_g = acc, f1, fp

        rand = random_abstention_baseline(y_raw, y_pred, coverage)
        conf = confidence_threshold_baseline(y_raw, y_pred, y_proba, coverage)

        worse_strict = acc_g < acc
        worse_2pct   = acc_g < acc - 0.02
        pct_abst     = round(100 * n_abstain / max(n_total,1), 1)
        pct_defer    = round(100 * n_defer   / max(n_total,1), 1)

        print(f'  [{variant_name}] {str(test_sid):>5}: '
              f'pop={acc:.3f}(FP={fp})  '
              f'gated={acc_g:.3f}(FP={fp_g},abst={pct_abst}%,cov={coverage}%)  '
              f'R={R:.3f}{"  WORSE!" if worse_2pct else ""}')

        rows.append({
            'subject':      test_sid,
            'R_graph':      round(R, 4),
            'acc':          acc,
            'f1':           f1,
            'fp':           fp,
            'acc_g':        acc_g,
            'f1_g':         f1_g,
            'fp_g':         fp_g,
            'coverage':     coverage,
            'pct_abst':     pct_abst,
            'pct_defer':    pct_defer,
            'worse_strict': worse_strict,
            'worse_2pct':   worse_2pct,
            'rand_acc':     rand[0] if rand else np.nan,
            'rand_f1':      rand[1] if rand else np.nan,
            'rand_fp':      rand[2] if rand else np.nan,
            'conf_acc':     conf[0] if conf else np.nan,
            'conf_f1':      conf[1] if conf else np.nan,
            'conf_fp':      conf[2] if conf else np.nan,
            'variant':      variant_name,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT TABLES
# ══════════════════════════════════════════════════════════════════════════════

def _row_stats(df):
    """Aggregate statistics for one dataset's results."""
    if df is None or len(df) == 0:
        return dict(N=0, r=0, p=1, acc=0, f1=0, fp_b=0, fp_g=0,
                    red=0, abst=0, cov=0, wf1=0, strict=0, worse=0,
                    rand_fp=0, conf_fp=0)
    v = df.dropna(subset=['R_graph','acc'])
    if len(v) > 3 and np.std(v['R_graph']) > 0 and np.std(v['acc']) > 0:
        r, p = pearsonr(v['R_graph'], v['acc'])
    else:
        r, p = 0, 1
    fp_b = df['fp'].sum()
    fp_g = df['fp_g'].sum()
    return dict(
        N       = len(df),
        r       = r,
        p       = p,
        acc     = df['acc'].mean(),
        f1      = df['f1_g'].mean(),
        fp_b    = fp_b,
        fp_g    = fp_g,
        red     = 100 * (fp_b - fp_g) / max(fp_b, 1),
        abst    = df['pct_abst'].mean(),
        cov     = df['coverage'].mean(),
        wf1     = df['f1_g'].min(),
        strict  = int(df['worse_strict'].sum()),
        worse   = int(df['worse_2pct'].sum()),
        rand_fp = df['rand_fp'].sum(),
        conf_fp = df['conf_fp'].sum(),
    )


def print_ablation_table(w_dfs, s_dfs, w_main, s_main):
    vec_labels = {
        'Corr-ICCM':   'correlation only   (3)',
        'Lag-ICCM':    'corr + max-lag     (6)',
        'Causal-ICCM': 'corr + directed    (6)',
        'Hybrid-ICCM': 'corr+lag+directed  (9)',
    }

    def ablation_row(df, name, label):
        if df is None or len(df) == 0: return
        v = df.dropna(subset=['R_graph','acc'])
        if len(v) > 3 and np.std(v['R_graph']) > 0 and np.std(v['acc']) > 0:
            r, p = pearsonr(v['R_graph'], v['acc'])
            sig  = '*' if p < 0.05 else 'n.s.'
        else:
            r, p, sig = 0, 1, 'n.s.'
        fp_b = df['fp'].sum()
        fp_g = df['fp_g'].sum()
        red  = 100 * (fp_b - fp_g) / max(fp_b, 1)
        abst = df['pct_abst'].mean()
        cov  = df['coverage'].mean()
        worse = int(df['worse_2pct'].sum())
        print(f'  {name:<14} {label:<25} {r:>+7.3f} {sig:>6} '
              f'{red:>8.1f}% {abst:>7.1f}% {cov:>8.1f}% {worse:>10}')

    print()
    print('=' * 88)
    print('ABLATION — WESAD (Frobenius metric)')
    print('=' * 88)
    print(f'  {"Method":<14} {"Coupling vector":<25} {"r":>7} {"sig":>6} '
          f'{"FP red%":>9} {"Abst%":>8} {"Cov%":>9} {">2% worse":>10}')
    print('  ' + '-' * 84)
    for name in vec_labels:
        ablation_row(w_dfs.get(name), name, vec_labels[name])

    if w_main is not None and len(w_main) > 0:
        v = w_main.dropna(subset=['R_graph','acc'])
        r, p = (pearsonr(v['R_graph'],v['acc']) if len(v)>3 else (0,1))
        sig  = '*' if p < 0.05 else 'n.s.'
        fp_b = w_main['fp'].sum(); fp_g = w_main['fp_g'].sum()
        red  = 100*(fp_b-fp_g)/max(fp_b,1)
        abst = w_main['pct_abst'].mean(); cov=w_main['coverage'].mean()
        worse = int(w_main['worse_2pct'].sum())
        print('  ' + '-' * 84)
        print(f'  {"Lag-ICCM":<14} {"corr + max-lag     (6)":<25} {r:>+7.3f} {sig:>6} '
              f'{red:>8.1f}% {abst:>7.1f}% {cov:>8.1f}% {worse:>10}  ← WESAD PROPOSED')
    print('=' * 88)

    print()
    print('=' * 88)
    print('ABLATION — Stress-Predict (direction-aware metric)')
    print('=' * 88)
    print(f'  {"Method":<14} {"Coupling vector":<25} {"r":>7} {"sig":>6} '
          f'{"FP red%":>9} {"Abst%":>8} {"Cov%":>9} {">2% worse":>10}')
    print('  ' + '-' * 84)
    for name in vec_labels:
        ablation_row(s_dfs.get(name), name, vec_labels[name])

    if s_main is not None and len(s_main) > 0:
        v = s_main.dropna(subset=['R_graph','acc'])
        r, p = (pearsonr(v['R_graph'],v['acc']) if len(v)>3 else (0,1))
        sig  = '*' if p < 0.05 else 'n.s.'
        fp_b = s_main['fp'].sum(); fp_g = s_main['fp_g'].sum()
        red  = 100*(fp_b-fp_g)/max(fp_b,1)
        abst = s_main['pct_abst'].mean(); cov=s_main['coverage'].mean()
        worse = int(s_main['worse_2pct'].sum())
        print('  ' + '-' * 84)
        print(f'  {"Hybrid-ICCM":<14} {"corr+lag+directed (9)":<25} {r:>+7.3f} {sig:>6} '
              f'{red:>8.1f}% {abst:>7.1f}% {cov:>8.1f}% {worse:>10}  ← SP PROPOSED')
    print('=' * 88)


def print_main_table(w_df, s_df):
    w = _row_stats(w_df)
    s = _row_stats(s_df)
    w_fp_b = w_df['fp'].sum() if w_df is not None and len(w_df) else 0
    s_fp_b = s_df['fp'].sum() if s_df is not None and len(s_df) else 0

    print()
    print('=' * 80)
    print('TABLE I — PROTOCOL-AWARE ICCM RESULTS')
    print('=' * 80)
    print(f'  {"":42} {"WESAD":>16} {"Stress-Predict":>16}')
    print('  ' + '-' * 76)
    print(f'  {"Subjects (N)":<42} {w["N"]:>16} {s["N"]:>16}')
    print(f'  {"Coupling vector":<42} {"corr + max-lag":>16} {"corr+lag+directed":>16}')
    print(f'  {"ICCM metric":<42} {"Frobenius":>16} {"Direction-aware":>16}')
    print(f'  {"Mean LOSO accuracy":<42} {w["acc"]:>16.1%} {s["acc"]:>16.1%}')
    print(f'  {"Mean LOSO F1 (gated windows)":<42} {w["f1"]:>16.3f} {s["f1"]:>16.3f}')
    print(f'  {"Mean coverage":<42} {w["cov"]:>15.1f}% {s["cov"]:>15.1f}%')
    print(f'  {"Ambiguity r (vs LOSO acc)":<42} {w["r"]:>+16.3f} {s["r"]:>+16.3f}')
    print(f'  {"p-value":<42} {w["p"]:>16.4f} {s["p"]:>16.4f}')
    print(f'  {"Significant (p < 0.05)":<42} '
          f'{"YES *" if w["p"]<0.05 else "NO":>16} '
          f'{"YES *" if s["p"]<0.05 else "NO":>16}')
    print(f'  {"FP (population model)":<42} {int(w_fp_b):>16} {int(s_fp_b):>16}')
    print(f'  {"FP (Protocol-Aware ICCM)":<42} {int(w["fp_g"]):>16} {int(s["fp_g"]):>16}')
    print(f'  {"FP (random abstention, matched)":<42} {int(w["rand_fp"]):>16} {int(s["rand_fp"]):>16}')
    print(f'  {"FP (confidence threshold, matched)":<42} {int(w["conf_fp"]):>16} {int(s["conf_fp"]):>16}')
    print(f'  {"FP reduction vs population":<42} {w["red"]:>15.1f}% {s["red"]:>15.1f}%')
    print(f'  {"Mean abstention rate":<42} {w["abst"]:>15.1f}% {s["abst"]:>15.1f}%')
    print(f'  {"Worst-subject F1 (gated)":<42} {w["wf1"]:>16.3f} {s["wf1"]:>16.3f}')
    print(f'  {"Subjects with any accuracy drop":<42} {w["strict"]:>16} {s["strict"]:>16}')
    print(f'  {"Subjects with >2% accuracy drop":<42} {w["worse"]:>16} {s["worse"]:>16}')
    print('=' * 80)
    print(f'  alpha = {ALPHA}  |  * p < 0.05\n')


def plot_scatter(w_df, s_df, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    datasets = [
        ('WESAD  (N=15)',          w_df, C_PURPLE, 'Frobenius metric  $||\Delta v||_2$'),
        ('Stress-Predict  (N=35)', s_df, C_ORANGE, 'Direction-aware  $1 - \cos(\Delta v, \mu_\Delta)$'),
    ]
    for ax, (title, df, col, metric_label) in zip(axes, datasets):
        if df is None or len(df) < 3: ax.set_title(title); continue
        df = df.dropna(subset=['R_graph','acc'])
        if len(df) < 3: ax.set_title(title); continue
        r, p = pearsonr(df['R_graph'], df['acc'])
        ax.scatter(df['R_graph'], df['acc'], color=col, s=48, zorder=4,
                   edgecolors='white', linewidths=0.5)
        for _, row in df.iterrows():
            ax.annotate(str(row['subject']),
                        xy=(row['R_graph'], row['acc']),
                        xytext=(3,3), textcoords='offset points',
                        fontsize=5.5, color=C_GRAY)
        z  = np.polyfit(df['R_graph'], df['acc'], 1)
        xl = np.linspace(df['R_graph'].min()-0.05, df['R_graph'].max()+0.05, 100)
        ax.plot(xl, np.polyval(z,xl), '--', color=C_DARK, lw=1.0, alpha=0.6)
        sig = '  *' if p < 0.05 else '  n.s.'
        ax.text(0.05, 0.06, f'$r = {r:.3f}${sig}\n$p = {p:.4f}$',
                transform=ax.transAxes, fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='#CCCCCC', linewidth=0.6))
        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
        ax.set_xlabel(f'Structural Ambiguity Score  $R_i$\n({metric_label})', fontsize=8)
        ax.set_ylabel('LOSO Accuracy', fontsize=8)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.yaxis.grid(True, linewidth=0.35, color=C_GRID, zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=1.0)
    for ext in ['pdf','png']:
        path = os.path.join(fig_dir, f'fig3_scatter_v3.{ext}')
        fig.savefig(path); print(f'  → {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(wesad_path, stress_path, fig_dir, results_dir, skip_ablation=False):
    os.makedirs(fig_dir,     exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print('\n' + '='*60 + '\nLoading WESAD...\n' + '='*60)
    w_sigs = load_wesad(wesad_path)

    print('\n' + '='*60 + '\nLoading Stress-Predict...\n' + '='*60)
    s_sigs = load_stress_predict(stress_path)

    variants = {
        'Corr-ICCM':   coupling_vec_corr,
        'Lag-ICCM':    coupling_vec_lag,
        'Causal-ICCM': coupling_vec_causal,
        'Hybrid-ICCM': coupling_vec_hybrid,
    }

    w_ablation, s_ablation = {}, {}

    if not skip_ablation:
        for vname, vfn in variants.items():
            print(f'\n{"="*60}\nABLATION: {vname} — WESAD\n{"="*60}')
            df = run_loso_frobenius(w_sigs, vfn, vname)
            w_ablation[vname] = df
            df.to_csv(os.path.join(results_dir,
                      f'wesad_{vname.lower().replace("-","_")}.csv'), index=False)

            print(f'\n{"="*60}\nABLATION: {vname} — Stress-Predict\n{"="*60}')
            df = run_loso_direction(s_sigs, vfn, vname)
            s_ablation[vname] = df
            df.to_csv(os.path.join(results_dir,
                      f'sp_{vname.lower().replace("-","_")}.csv'), index=False)
    else:
        print('\n[Ablation skipped — run without --skip_ablation for full results]')
        for vname in variants:
            w_ablation[vname] = None
            s_ablation[vname] = None

    # Main method: Protocol-Aware ICCM
    # WESAD (single-protocol): Lag-ICCM — corr + max-lag (6 features)
    # Stress-Predict (multi-protocol): Hybrid-ICCM — corr + max-lag + directed (9 features)
    print(f'\n{"="*60}\nMAIN: Protocol-Aware ICCM — WESAD (Lag coupling)\n{"="*60}')
    w_main = run_loso_frobenius(w_sigs, coupling_vec_lag, 'Lag-ICCM')
    w_main.to_csv(os.path.join(results_dir, 'wesad_protocol_aware_iccm.csv'), index=False)

    print(f'\n{"="*60}\nMAIN: Protocol-Aware ICCM — Stress-Predict (Hybrid + direction-aware)\n{"="*60}')
    s_main = run_loso_direction(s_sigs, coupling_vec_hybrid, 'Hybrid-ICCM')
    s_main.to_csv(os.path.join(results_dir, 'sp_protocol_aware_iccm.csv'), index=False)

    # Results
    print('\n' + '='*60 + '\nRESULTS\n' + '='*60)
    print_ablation_table(w_ablation, s_ablation, w_main, s_main)
    print_main_table(w_main, s_main)
    plot_scatter(w_main, s_main, fig_dir)
    print('DONE.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--wesad_path',    default='data/WESAD')
    ap.add_argument('--stress_path',   default='data/Stress-Predict-Dataset/Raw_data')
    ap.add_argument('--fig_dir',       default='figures')
    ap.add_argument('--results_dir',   default='results')
    ap.add_argument('--skip_ablation', action='store_true')
    args = ap.parse_args()
    main(args.wesad_path, args.stress_path,
         args.fig_dir, args.results_dir,
         skip_ablation=args.skip_ablation)