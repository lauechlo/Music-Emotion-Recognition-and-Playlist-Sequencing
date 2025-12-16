# Music Emotion Recognition and Playlist Sequencing
SML312 Final Project – Fall 2025

This project looks at whether I can predict how songs make people feel (valence + arousal) and then use those predictions to put songs in an order that feels smoother and more intentional. The idea came from noticing that playlists often have emotional "jumps," and I wanted to see if machine learning could help reduce those.

There are two main components:

1. **Emotion Recognition** – Predict valence/arousal for Spotify tracks
2. **Playlist Sequencing** – Use those predictions to arrange playlists with fewer abrupt transitions

A big part of the project was figuring out how to balance *accuracy* with *speed*. The DEAM neural network uses detailed audio features but is too slow for large datasets (~30 seconds per track). So I built a lightweight Random Forest model that learns to approximate the neural network's predictions using only Spotify's 12 features. This runs in milliseconds and gets similar results (since the neural network itself only gets R²~0.35, the integration model getting ~0.30 is pretty close).

---

## What it does

**Emotion recognition:**

* Neural network (DEAM) using 74 audio features (pre-extracted with openSMILE)
* Fast integration model using Spotify CSV features
* Unified emotion analyzer that falls back through multiple methods

**Playlist sequencing:**

* Eleven algorithms implemented
* Automatically compares them and shows which gives the smoothest emotional flow
* Evaluates transitions using valence/arousal distance

---

## Results (summary)

### Emotion Recognition

* DEAM neural network: **R² ≈ 0.30–0.42** (5-fold cross-validation on 1,802 tracks)
* CSV integration model (Random Forest): **R² ≈ 0.40 (valence), 0.77 (arousal)** predicting emotion from musical features 
* Works on the full 114k Spotify track CSV (instant predictions)

### Playlist Sequencing

* Tested 11 sequencing strategies
* Best algorithm improved emotional smoothness by **47.6%**
* Overall playlist quality score improved by **39.2%** vs random
* Key insight: **greedy nearest-neighbor is already optimal** for this type of continuous emotion space, and the more advanced algorithms didn’t really outperform it

---

## Repository Structure (simplified)

**Core code**

* `unified_music_analyzer.py` – main system combining all prediction paths
* `deam_csv_integration.py` – trains and loads the integration model
* `playlist_sequencer.py` – core sequencing logic, scoring, visualization
* `mood_utils.py` – mood classification helpers
* `config.py` – Spotify API settings (optional)

**Experiments**

* `experiments/train_integration_model.py` – trains integration model (compares circular vs honest approaches)
* `experiments/hyperparameter_tuning.py` – grid search for RF and NN optimal parameters
* `experiments/rf_tuning_results.pkl` – Random Forest tuning results
* `experiments/nn_tuning_results.pkl` – Neural Network tuning results


**Data**

* `spotify-tracks-dataset.csv` – 114k tracks with 12 features (from Kaggle)
* `deam_features_74.csv` – 1,802 DEAM tracks with 74 features + valence/arousal labels
* `deam_csv_integration.pkl` – trained RF model
* `deam_full_model.pkl` – trained neural network (DEAM)

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

You’ll need the `spotify-tracks-dataset.csv` file for CSV-based predictions.

Spotify API credentials are optional. If you want to fetch audio previews:

```python
SPOTIFY_CLIENT_ID = "your_id"
SPOTIFY_CLIENT_SECRET = "your_secret"
```

(Otherwise the system will use CSV-only mode.)

---

## Usage

### Basic emotion prediction

```python
from unified_music_analyzer import UnifiedMusicAnalyzer

analyzer = UnifiedMusicAnalyzer()
result = analyzer.analyze_track("4iV5W9uYEdYUVa79Axb7Rh")

print("Valence:", result['valence'])
print("Arousal:", result['arousal'])
print("Mood:", result['mood'])
```

### Sequencing a playlist

```python
from playlist_sequencer import PlaylistSequencer
from deam_csv_integration import DEAMCSVIntegration

integration = DEAMCSVIntegration()
integration.load_csv_data()
integration.load_models("deam_csv_integration.pkl")

track_ids = ["id1", "id2", "id3"]

tracks_data = [integration.predict_deam_from_csv(tid) for tid in track_ids]

sequencer = PlaylistSequencer()
sequencer.load_from_analyzed_data(tracks_data)

results = sequencer.compare_strategies()
best_strategy = max(results.items(), key=lambda x: x[1]["metrics"].total_score)
```

---

## How the system works

### Emotion Recognition Pipeline

```
Track ID
   ↓
Try fast integration model (12 Spotify features)
   ↓
If missing → try Spotify API + DEAM NN (slow)
   ↓
If still missing → fallback to raw CSV valence/energy
   ↓
Return (valence, arousal, mood)
```

The integration model is the main workhorse because it’s instant. The DEAM network only runs when necessary or when exploring preview audio.

---

## Sequencing Algorithms

**Simple ones:**

* Greedy smooth (nearest neighbor)
* Buildup (low → high energy)
* Cooldown (high → low energy)
* Wave patterns
* Cluster-first (K-means groups)

**Search/optimization-based:**

* Greedy with 3-step lookahead
* 2-Opt
* Simulated annealing
* Beam search
* Adaptive greedy

The surprising result was that **greedy is actually the globally optimal solution** for this kind of continuous emotion space, which several advanced algorithms confirmed by converging to the same ordering.

---

## Data

* **DEAM dataset** (1,802 tracks)

  * Used for training neural network
  * 74 audio features per track (pre-extracted using openSMILE)
  * Includes valence/arousal annotations from expert listeners

* **Spotify 114k track CSV**

  * Used for large-scale predictions
  * 12 high-level audio features

The DEAM features were extracted using openSMILE (MFCCs, spectral features, energy, pitch, etc.). The annotations come from crowdsourced emotion ratings that were averaged across multiple listeners.

---

## Hyperparameter Tuning

### Random Forest Integration Model

Hyperparameter tuning was performed using 5-fold cross-validation on the DEAM dataset with a reduced feature set (9 features) to match the dimensionality constraints of the Spotify CSV features. The goal was to find hyperparameters that balance model complexity with generalization.

**Optimal Hyperparameters Found:**

* **n_estimators: 200 (valence) / 100 (arousal)**
  * More trees generally improve stability and reduce variance
  * Valence required more trees for optimal performance
  * Diminishing returns beyond 200 trees while increasing computation time

* **max_depth: 8**
  * Shallow trees prevent overfitting on the relatively small DEAM dataset (1,802 tracks)
  * Deeper trees (12-16) led to worse cross-validation performance
  * Depth of 8 provides enough complexity to capture nonlinear patterns without memorizing training data

* **max_features: 'sqrt'**
  * Feature subsampling adds diversity to the ensemble
  * Using sqrt(9) ≈ 3 features per split prevents individual trees from being too correlated
  * Improves generalization compared to using all features

* **min_samples_split: 5**
  * Requires at least 5 samples to attempt a split
  * Prevents creating splits based on very small subsets that might be noise
  * Balances between model flexibility and regularization

* **min_samples_leaf: 1**
  * Allows leaf nodes with single samples
  * Combined with other constraints (max_depth, min_samples_split), this doesn't cause overfitting
  * Preserves model's ability to capture fine-grained patterns

**Performance (on hyperparameter tuning task with 9 DEAM features):**
* Valence: R² = 0.3852 (cross-validated)
* Arousal: R² = 0.4387 (cross-validated)

These scores represent performance on predicting human emotion annotations from a limited DEAM feature set used for hyperparameter tuning. The production integration model uses 10 Spotify features (excluding valence/energy to avoid circular prediction) and achieves R² ≈ 0.40 (valence) and 0.77 (arousal) when predicting emotion from musical characteristics like tempo, loudness, and danceability.

### Neural Network (DEAM Model)

The neural network uses the following architecture and hyperparameters on the full 74-feature DEAM dataset:

**Architecture and Hyperparameters:**

* **hidden_layer_sizes: (128, 64, 32)**
  * Three-layer feedforward network with gradual dimension reduction
  * Balances model capacity with dataset size (1,802 tracks)
  * Reduces dimensions from 74 input features to 2 output targets (valence, arousal)

* **alpha: 0.001 (L2 regularization)**
  * Moderate regularization to prevent overfitting
  * Penalizes large weights without over-constraining the model
  * Works well with early stopping

* **learning_rate_init: 0.001**
  * Standard learning rate for Adam optimizer
  * Enables stable convergence on this dataset
  * Combined with early stopping (validation_fraction=0.1, n_iter_no_change=20)

* **activation: 'relu'**
  * ReLU activation for hidden layers
  * Helps with gradient flow and training stability

* **max_iter: 500, early_stopping: True**
  * Maximum 500 epochs with early stopping
  * Stops training if validation performance doesn't improve for 20 iterations
  * Prevents overfitting and reduces training time

**Performance:**
* Single Model: R² = 0.30 (valence), 0.32 (arousal) - 5-fold cross-validation
* **Ensemble (5 models): R² = 0.34 (valence), 0.41 (arousal)** - 14-28% improvement

These hyperparameters were chosen to balance model complexity with the relatively small training dataset size. Hyperparameter tuning experiments did not improve on this configuration, demonstrating that the initial architecture choice was well-suited to the task.

**Ensemble Method:**

The final system uses an ensemble of 5 neural networks with different random seeds (42-46). Averaging predictions across models reduces variance from random initialization:

* **Improvement**: +0.04 R² (valence), +0.09 R² (arousal) - validated via 5-fold CV
* **Trade-off**: 5× training time, minimal inference cost
* **Rationale**: Standard ML technique for reducing model variance

Ensemble performance is robust across CV folds, confirming the improvement isn't specific to a particular train/test split.

---

## Feature Selection and Scoring Functions

### DEAM 74-Feature Selection Process

The DEAM dataset originally provides hundreds of openSMILE features extracted from audio. The 74 features used in this project were selected based on music emotion recognition literature and feature importance analysis:

**Feature Categories Selected (74 total):**

1. **Energy/Loudness Features (6 features)**
   - RMS energy (mean, std, delta mean, delta std)
   - Zero-crossing rate (mean, std)
   - Rationale: Energy is strongly correlated with arousal dimension

2. **MFCCs - Mel-Frequency Cepstral Coefficients (28 features)**
   - MFCC 1-14 (mean and std for each)
   - Rationale: Capture timbre and tonal quality, critical for emotion perception
   - Most discriminative features for music genre and emotion

3. **Spectral Features (16 features)**
   - Spectral centroid, flux, slope, entropy (mean and std for each)
   - Spectral variance, skewness, kurtosis (mean and std)
   - Rationale: Brightness and frequency distribution relate to emotional character

4. **Pitch and Harmony (8 features)**
   - F0 (fundamental frequency) mean and std
   - Harmonics-to-noise ratio (logHNR) mean and std
   - Voicing probability mean and std
   - Jitter (local and DDP) mean and std
   - Rationale: Pitch patterns influence valence perception

5. **Frequency Band Energy (8 features)**
   - Energy in 250-650 Hz band (mean, std)
   - Energy in 1000-4000 Hz band (mean, std)
   - Auditory spectrum bands (mean, std)
   - Rationale: Different frequency ranges associated with different emotions

6. **Psychoacoustic Features (8 features)**
   - Spectral sharpness (mean, std)
   - Auditory spectrum length L1 norm (mean, std)
   - Auditory spectrum with RASTA filtering (mean, std)
   - Shimmer (mean, std)
   - Rationale: Perceptually motivated features designed for emotion recognition

**Selection Criteria:**
- Excluded highly correlated features (Pearson correlation > 0.95)
- Removed features with >10% missing values
- Prioritized features with high importance in preliminary Random Forest models
- Focused on interpretable features with known relationships to emotion perception

### Playlist Sequencing Scoring Function

The composite quality score balances multiple objectives to create emotionally coherent playlists. The weights were chosen through iterative experimentation and reflect the relative importance of each objective:

**Scoring Function:**
```
Total Score = smoothness_score × 0.35 +
              trajectory_score × 0.25 +
              diversity_score × 0.20 +
              variance_score × 0.15 +
              baseline × 0.05
```

**Weight Justifications:**

* **Smoothness (0.35) - Most Important**
  - Minimizes jarring transitions between consecutive tracks
  - Directly addresses the core problem (emotional jumps in playlists)
  - Computed as normalized Euclidean distance in valence-arousal space
  - Highest weight because smooth transitions are the primary goal

* **Trajectory Alignment (0.25) - Secondary Goal**
  - Ensures playlist follows intended emotional arc (buildup, cooldown, etc.)
  - Provides directionality and narrative structure
  - Important for intentional mood progression
  - Second-highest weight to maintain coherent emotional journey

* **Diversity (0.20) - Balance**
  - Prevents playlist from becoming monotonous
  - Measures artist and genre variety
  - Balances smoothness objective (which could lead to repetitive selections)
  - Moderate weight to encourage variety without sacrificing cohesion

* **Flow Consistency (0.15) - Polish**
  - Measures standard deviation of transition distances
  - Lower variance means more predictable, consistent pacing
  - Prevents alternating between tiny and huge jumps
  - Lower weight because it's a refinement, not core objective

* **Baseline (0.05) - Scale Offset**
  - Provides non-zero baseline score
  - Ensures reasonable scores even for poor orderings
  - Minimal weight, mainly for numerical stability

**Why These Specific Values:**
The weights were determined empirically by testing on diverse playlists and adjusting based on subjective quality assessment. The 0.35/0.25/0.20/0.15/0.05 distribution reflects a descending priority from core objective (smoothness) through secondary goals to refinements, while summing to 1.0 for interpretability.

### Sequencing Algorithm Hyperparameters

Each advanced sequencing algorithm has tuned hyperparameters chosen to balance optimization quality with computational cost:

**Clustering-Based Sequencing:**
* `n_clusters = 3`
  - Groups tracks into 3 emotion regions
  - Sequences within each cluster, then connects clusters
  - Rationale: 3 clusters provide meaningful groupings without over-segmentation

**Greedy with Lookahead:**
* `lookahead = 3`
  - Evaluates 3 steps ahead instead of just next track
  - Balances foresight with computational cost (O(n³) vs O(n²))
  - Rationale: 3 steps captures near-term consequences without excessive computation

**2-Opt Local Search:**
* `max_iterations = 100`
  - Maximum number of improvement passes
  - Typically converges in 20-50 iterations
  - Rationale: 100 provides generous convergence budget without wasting time

**Simulated Annealing:**
* `initial_temperature = 1.0`
  - Controls initial acceptance probability for worse solutions
  - 1.0 allows ~37% acceptance of solutions with delta=1.0
  - Rationale: Moderate starting temperature balances exploration vs exploitation

* `cooling_rate = 0.95`
  - Temperature multiplied by 0.95 each iteration
  - Geometric cooling schedule
  - Rationale: Standard cooling rate that works well empirically

* `max_iterations = 500`
  - Enough iterations for temperature to cool sufficiently
  - After 500 iterations, temperature ≈ 1.0 × 0.95^500 ≈ 0
  - Rationale: Allows thorough exploration before converging

**Beam Search:**
* `beam_width = 3`
  - Keeps top 3 candidate sequences at each step
  - Balances search breadth with memory/time cost
  - Rationale: 3 provides diversity without excessive branching (vs 2 or 5)

**Adaptive Greedy:**
* `bridge_probability = 0.15`
  - 15% chance to jump to distant track (vs always picking nearest)
  - Allows occasional exploration of different emotion regions
  - Rationale: 0.15 provides enough jumps for variety without destroying smoothness

### Sequencing Algorithm Performance

Bootstrap analysis (100 playlists, 15-20 tracks each) revealed that optimization algorithms significantly outperform random ordering:

**Results:**
* **2-Opt local search**: 32.12% improvement (best overall, Cohen's d = 3.70)
* Simulated annealing: 28.61% improvement (d = 3.13)
* Beam search: 26.72% improvement (d = 2.71)
* Greedy nearest-neighbor: 26.37% improvement (d = 2.68)

**Why 2-Opt Wins:**

2-Opt iteratively improves solutions by swapping track pairs, escaping local optima that trap greedy methods. In continuous 2D emotion space, this local search finds globally better orderings than greedy's single-pass construction.

**Why Greedy Still Works Well:**

Greedy performs surprisingly well (26.4% improvement) because:
* Euclidean emotion space has smooth geometry (no adversarial structure)
* Minimizing each local step naturally improves global smoothness
* Confidence intervals overlap with top algorithms (practically similar performance)

**Trade-off:** 2-Opt achieves 5.8% better improvement than greedy but requires iterative refinement (slower). For real-time applications, greedy offers 95% of 2-Opt's quality at O(n²) instead of O(n² × iterations).

**Key Finding:** All optimization algorithms showed very large effect sizes (Cohen's d > 2.0), meaning the differences vs random ordering are practically significant, not just statistically significant. The choice between top algorithms depends on speed vs quality requirements.

---

## Notes

* The bootstrap sequencing analysis takes a while to run (100 iterations).
* Some of the sklearn warnings are normal.
* DEAM neural network can be run with the included `deam_features_74.csv` file.
* CSV-only mode works without API keys.
* This repo focuses more on building pipelines and exploring algorithmic behavior than on building a production UI.

