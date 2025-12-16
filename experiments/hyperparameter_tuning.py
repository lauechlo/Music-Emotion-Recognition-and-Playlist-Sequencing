#!/usr/bin/env python3
"""
Hyperparameter Tuning for DEAM Models
Run this to find optimal parameters for Random Forest and Neural Network
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import time


def load_deam_data():
    """Load DEAM dataset with same logic as existing code"""
    print("Loading DEAM data...")
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "data" / "deam_features_74.csv"

    df = pd.read_csv(csv_path)

    # Find valence/arousal columns (same as deam_neural_network.py)
    possible_valence = ["valence", "valence_mean", "valence_avg"]
    possible_arousal = ["arousal", "arousal_mean", "arousal_avg"]

    valence_col = next((c for c in possible_valence if c in df.columns), None)
    arousal_col = next((c for c in possible_arousal if c in df.columns), None)

    if valence_col is None or arousal_col is None:
        raise ValueError(f"Could not find valence/arousal columns")

    # Extract features (exclude IDs and targets)
    drop_cols = {valence_col, arousal_col, "track_id", "song_id", "file_id", "filename"}
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols].to_numpy(dtype=float)
    y_valence = df[valence_col].to_numpy(dtype=float)
    y_arousal = df[arousal_col].to_numpy(dtype=float)

    print(f"Loaded {X.shape[0]} tracks with {X.shape[1]} features")
    print(f"Valence range: {y_valence.min():.2f} to {y_valence.max():.2f}")
    print(f"Arousal range: {y_arousal.min():.2f} to {y_arousal.max():.2f}\n")

    return X, y_valence, y_arousal


def load_deam_for_rf():
    """
    Load DEAM data for RF integration model hyperparameter tuning.

    Uses a reduced feature set from DEAM (9 features selected to be analogous
    to Spotify's feature dimensionality) with actual human emotion annotations
    as targets.

    This tests the RF's ability to predict emotion from a limited feature set,
    which matches the production task without data leakage.
    """
    print("Loading DEAM data for RF integration model...")
    base_dir = Path(__file__).resolve().parent.parent
    deam_path = base_dir / "data" / "deam_features_74.csv"

    # Load DEAM dataset
    df = pd.read_csv(deam_path)
    print(f"Loaded {len(df):,} DEAM tracks")

    # Select a reduced feature set from DEAM's 74 features
    # These are chosen to be analogous to Spotify's feature types:
    # - Energy/loudness features
    # - Spectral features (timbre/brightness)
    # - Temporal features (rhythm/tempo proxies)
    # - Pitch/harmony features
    reduced_features = [
        'pcm_RMSenergy_sma_amean',           # Energy (like Spotify's energy)
        'pcm_RMSenergy_sma_stddev',          # Energy variance
        'pcm_fftMag_spectralCentroid_sma_amean',  # Brightness (related to acousticness)
        'pcm_fftMag_spectralFlux_sma_amean',      # Timbre change (related to danceability)
        'F0final_sma_amean',                 # Pitch (related to mode/key)
        'pcm_zcr_sma_amean',                 # Zero-crossing rate (percussiveness)
        'pcm_fftMag_mfcc_sma[1]_amean',      # MFCC 1 (timbre)
        'pcm_fftMag_mfcc_sma[2]_amean',      # MFCC 2 (timbre)
        'audspec_lengthL1norm_sma_amean'     # Spectral energy
    ]

    # Check which features exist
    available_features = [f for f in reduced_features if f in df.columns]
    print(f"Found {len(available_features)}/{len(reduced_features)} DEAM features")

    if len(available_features) < len(reduced_features):
        missing = set(reduced_features) - set(available_features)
        print(f"Warning: Missing features: {missing}")
        if len(available_features) < 7:
            print("Too many missing features for hyperparameter tuning")
            return None, None, None
        reduced_features = available_features

    # Extract features
    X = df[reduced_features].fillna(0).to_numpy(dtype=float)

    # Use actual human annotations as targets (NO DATA LEAKAGE!)
    # These are the ground truth emotion ratings from human listeners
    y_valence = df['valence_mean'].to_numpy(dtype=float)
    y_arousal = df['arousal_mean'].to_numpy(dtype=float)

    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Valence range: {y_valence.min():.2f} to {y_valence.max():.2f}")
    print(f"Arousal range: {y_arousal.min():.2f} to {y_arousal.max():.2f}\n")

    return X, y_valence, y_arousal


def tune_random_forest():
    """Tune Random Forest hyperparameters"""
    print("="*80)
    print("RANDOM FOREST HYPERPARAMETER TUNING")
    print("="*80 + "\n")

    X, y_valence, y_arousal = load_deam_for_rf()

    if X is None:
        print("Skipping RF tuning - data not available\n")
        return None

    # Split data
    X_train, X_test, y_val_train, y_val_test, y_ar_train, y_ar_test = train_test_split(
        X, y_valence, y_arousal, test_size=0.2, random_state=42
    )

    # Scale features (same as your current code)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [8, 12, 16, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    print(f"Testing {3*5*3*3*3} = 135 parameter combinations with 5-fold CV")
    print("This may take 30-60 minutes...\n")

    # Tune for VALENCE
    print("-" * 80)
    print("Tuning for VALENCE prediction...")
    print("-" * 80)
    start_time = time.time()

    grid_search_val = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    grid_search_val.fit(X_train_scaled, y_val_train)

    val_time = time.time() - start_time

    print(f"\nBest Valence Parameters: {grid_search_val.best_params_}")
    print(f"Best CV R²: {grid_search_val.best_score_:.4f}")
    print(f"Time: {val_time/60:.1f} minutes\n")

    # Tune for AROUSAL
    print("-" * 80)
    print("Tuning for AROUSAL prediction...")
    print("-" * 80)
    start_time = time.time()

    grid_search_ar = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    grid_search_ar.fit(X_train_scaled, y_ar_train)

    ar_time = time.time() - start_time

    print(f"\nBest Arousal Parameters: {grid_search_ar.best_params_}")
    print(f"Best CV R²: {grid_search_ar.best_score_:.4f}")
    print(f"Time: {ar_time/60:.1f} minutes\n")

    # Save results
    results = {
        'valence_best_params': grid_search_val.best_params_,
        'valence_best_score': grid_search_val.best_score_,
        'valence_best_model': grid_search_val.best_estimator_,
        'arousal_best_params': grid_search_ar.best_params_,
        'arousal_best_score': grid_search_ar.best_score_,
        'arousal_best_model': grid_search_ar.best_estimator_,
        'scaler': scaler
    }

    with open('rf_tuning_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to 'rf_tuning_results.pkl'\n")

    return results


def tune_neural_network():
    """Tune Neural Network hyperparameters (lighter search)"""
    print("="*80)
    print("NEURAL NETWORK HYPERPARAMETER TUNING")
    print("="*80 + "\n")

    X, y_valence, y_arousal = load_deam_data()

    # Split data
    X_train, X_test, y_val_train, y_val_test, y_ar_train, y_ar_test = train_test_split(
        X, y_valence, y_arousal, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define parameter grid (smaller search - NN is slower)
    param_grid = {
        'hidden_layer_sizes': [
            (64, 32),           # 2 layers
            (128, 64, 32),      # 3 layers (your current)
            (256, 128, 64),     # 3 layers wider
            (128, 128, 64)      # 3 layers different ratio
        ],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.0001, 0.001, 0.01]
    }

    print(f"Testing {4*3*3} = 36 parameter combinations with 3-fold CV")
    print("This may take 2-4 hours...\n")

    # Tune for VALENCE
    print("-" * 80)
    print("Tuning for VALENCE prediction...")
    print("-" * 80)
    start_time = time.time()

    grid_search_val = GridSearchCV(
        MLPRegressor(
            activation='relu',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        ),
        param_grid,
        cv=3,  # 3-fold to save time
        scoring='r2',
        n_jobs=1,  # NN doesn't parallelize well across jobs
        verbose=2
    )
    grid_search_val.fit(X_train_scaled, y_val_train)

    val_time = time.time() - start_time

    print(f"\nBest Valence Parameters: {grid_search_val.best_params_}")
    print(f"Best CV R²: {grid_search_val.best_score_:.4f}")
    print(f"Time: {val_time/60:.1f} minutes\n")

    # Tune for AROUSAL
    print("-" * 80)
    print("Tuning for AROUSAL prediction...")
    print("-" * 80)
    start_time = time.time()

    grid_search_ar = GridSearchCV(
        MLPRegressor(
            activation='relu',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        ),
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=1,
        verbose=2
    )
    grid_search_ar.fit(X_train_scaled, y_ar_train)

    ar_time = time.time() - start_time

    print(f"\nBest Arousal Parameters: {grid_search_ar.best_params_}")
    print(f"Best CV R²: {grid_search_ar.best_score_:.4f}")
    print(f"Time: {ar_time/60:.1f} minutes\n")

    # Save results
    results = {
        'valence_best_params': grid_search_val.best_params_,
        'valence_best_score': grid_search_val.best_score_,
        'valence_best_model': grid_search_val.best_estimator_,
        'arousal_best_params': grid_search_ar.best_params_,
        'arousal_best_score': grid_search_ar.best_score_,
        'arousal_best_model': grid_search_ar.best_estimator_,
        'scaler': scaler
    }

    with open('nn_tuning_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to 'nn_tuning_results.pkl'\n")

    return results


def main():
    """Run hyperparameter tuning"""
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING FOR DEAM EMOTION RECOGNITION")
    print("="*80 + "\n")

    print("This script will tune hyperparameters for:")
    print("1. Random Forest (CSV integration model) - ~30-60 minutes")
    print("2. Neural Network (DEAM model) - ~2-4 hours")
    print("\nYou can run both or just one.\n")

    # Ask which to run
    response = input("Tune Random Forest? (y/n): ").strip().lower()
    if response == 'y':
        rf_results = tune_random_forest()
        if rf_results:
            print("\n" + "="*80)
            print("RANDOM FOREST TUNING SUMMARY")
            print("="*80)
            print(f"\nValence - Best R²: {rf_results['valence_best_score']:.4f}")
            print(f"Parameters: {rf_results['valence_best_params']}")
            print(f"\nArousal - Best R²: {rf_results['arousal_best_score']:.4f}")
            print(f"Parameters: {rf_results['arousal_best_params']}\n")

    response = input("\nTune Neural Network? (y/n): ").strip().lower()
    if response == 'y':
        nn_results = tune_neural_network()
        print("\n" + "="*80)
        print("NEURAL NETWORK TUNING SUMMARY")
        print("="*80)
        print(f"\nValence - Best R²: {nn_results['valence_best_score']:.4f}")
        print(f"Parameters: {nn_results['valence_best_params']}")
        print(f"\nArousal - Best R²: {nn_results['arousal_best_score']:.4f}")
        print(f"Parameters: {nn_results['arousal_best_params']}\n")

    print("="*80)
    print("TUNING COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print("- rf_tuning_results.pkl (if RF was run)")
    print("- nn_tuning_results.pkl (if NN was run)")
    print("\nUse these optimal parameters to update your models.\n")


if __name__ == "__main__":
    main()
