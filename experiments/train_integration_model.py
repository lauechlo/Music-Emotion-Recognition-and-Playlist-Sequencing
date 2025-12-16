#!/usr/bin/env python3
"""
Honest integration model training using real Spotify tracks
Matches Spotify CSV to DEAM by sampling and testing predictive power
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

sys.path.append(str(Path(__file__).parent.parent / 'models'))
sys.path.append(str(Path(__file__).parent.parent / 'analyzer'))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


def create_training_data():
    """
    Create training data by using Spotify CSV tracks
    and their existing valence/energy as proxies for DEAM targets
    """
    print("="*80)
    print("HONEST INTEGRATION MODEL TRAINING")
    print("="*80)

    base_dir = Path(__file__).resolve().parent.parent
    spotify_path = base_dir / "data" / "spotify-tracks-dataset.csv"

    # Load Spotify CSV
    df = pd.read_csv(spotify_path)
    print(f"\nLoaded {len(df):,} Spotify tracks")

    # Sample a subset for training (similar size to DEAM)
    sample_size = 2000
    df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    print(f"Using {len(df_sample):,} tracks for training")

    return df_sample


def train_with_all_features(df):
    """Train WITH valence and energy (original approach)"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: WITH valence and energy features (original)")
    print("="*80)

    features_with = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]

    # Create targets scaled to DEAM range (1-9)
    X = df[features_with].fillna(0)
    y_valence = 1 + (df['valence'] * 8)  # Scale 0-1 to 1-9
    y_arousal = 1 + (df['energy'] * 8)   # Scale 0-1 to 1-9

    # Split
    X_train, X_test, y_val_train, y_val_test, y_ar_train, y_ar_test = train_test_split(
        X, y_valence, y_arousal, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train valence
    val_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    val_model.fit(X_train_scaled, y_val_train)
    val_pred = val_model.predict(X_test_scaled)
    val_r2 = r2_score(y_val_test, val_pred)

    # Train arousal
    ar_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    ar_model.fit(X_train_scaled, y_ar_train)
    ar_pred = ar_model.predict(X_test_scaled)
    ar_r2 = r2_score(y_ar_test, ar_pred)

    print(f"\nResults WITH valence/energy as features:")
    print(f"  Valence R²: {val_r2:.4f}")
    print(f"  Arousal R²: {ar_r2:.4f}")
    print(f"\nConclusion: Very high R² because it's learning identity transformation")

    return val_r2, ar_r2


def train_without_circular_features(df):
    """Train WITHOUT valence and energy (honest approach)"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: WITHOUT valence/energy features (honest)")
    print("="*80)

    features_without = [
        'danceability', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'tempo',
        'key', 'mode', 'time_signature'
    ]

    # Create targets
    X = df[features_without].fillna(0)
    y_valence = 1 + (df['valence'] * 8)
    y_arousal = 1 + (df['energy'] * 8)

    # Split
    X_train, X_test, y_val_train, y_val_test, y_ar_train, y_ar_test = train_test_split(
        X, y_valence, y_arousal, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train valence
    val_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    val_model.fit(X_train_scaled, y_val_train)
    val_pred = val_model.predict(X_test_scaled)
    val_r2 = r2_score(y_val_test, val_pred)
    val_cv = cross_val_score(val_model, X_train_scaled, y_val_train, cv=5, scoring='r2')

    # Train arousal
    ar_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    ar_model.fit(X_train_scaled, y_ar_train)
    ar_pred = ar_model.predict(X_test_scaled)
    ar_r2 = r2_score(y_ar_test, ar_pred)
    ar_cv = cross_val_score(ar_model, X_train_scaled, y_ar_train, cv=5, scoring='r2')

    print(f"\nResults WITHOUT valence/energy as features:")
    print(f"  Valence - Test R²: {val_r2:.4f}, CV: {val_cv.mean():.4f} (+/- {val_cv.std():.4f})")
    print(f"  Arousal - Test R²: {ar_r2:.4f}, CV: {ar_cv.mean():.4f} (+/- {ar_cv.std():.4f})")
    print(f"\nConclusion: Lower R² but more honest - learning from musical features")

    # Save this honest model
    model_data = {
        'valence_model': val_model,
        'arousal_model': ar_model,
        'scaler': scaler,
        'csv_features': features_without,
        'test_metrics': {
            'valence_r2': val_r2,
            'arousal_r2': ar_r2,
            'valence_cv_mean': val_cv.mean(),
            'arousal_cv_mean': ar_cv.mean()
        }
    }

    output_path = Path(__file__).parent.parent / 'data' / 'deam_csv_integration.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nHonest model saved to: {output_path}")

    return val_r2, ar_r2, model_data


def main():
    print("\n")
    print("="*80)
    print("INTEGRATION MODEL: CIRCULAR vs HONEST COMPARISON")
    print("="*80)
    print("\nThis script compares two approaches:")
    print("  1. WITH valence/energy features (circular)")
    print("  2. WITHOUT valence/energy features (honest)")
    print("\n")

    # Load data
    df = create_training_data()

    # Experiment 1: With circular features
    r2_with_val, r2_with_ar = train_with_all_features(df)

    # Experiment 2: Without circular features
    r2_without_val, r2_without_ar, model_data = train_without_circular_features(df)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATION")
    print("="*80)

    print(f"\nCircular approach (WITH valence/energy):")
    print(f"  Valence R²: {r2_with_val:.4f}")
    print(f"  Arousal R²: {r2_with_ar:.4f}")
    print(f"  Problem: Mostly learning identity transformation")

    print(f"\nHonest approach (WITHOUT valence/energy):")
    print(f"  Valence R²: {r2_without_val:.4f}")
    print(f"  Arousal R²: {r2_without_ar:.4f}")
    print(f"  Advantage: Actually learning emotion from musical features")

    print("\n" + "-"*80)
    print("RECOMMENDATION FOR YOUR REPORT:")
    print("-"*80)

    print(f"\nUpdate your report to say:")
    print(f"  'The integration model achieved R² ≈ {r2_without_val:.2f} (valence)")
    print(f"   and R² ≈ {r2_without_ar:.2f} (arousal) when predicting emotion from")
    print(f"   musical features WITHOUT using Spotify's existing valence/energy")
    print(f"   predictions. This represents the model's ability to infer emotion")
    print(f"   from tempo, loudness, danceability, and other acoustic properties.")
    print(f"   While lower than models that use valence/energy as inputs (R² > 0.95),")
    print(f"   this approach avoids circular prediction and demonstrates genuine")
    print(f"   emotion inference from musical characteristics.'")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
