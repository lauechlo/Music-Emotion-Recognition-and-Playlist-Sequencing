"""
Train models to predict DEAM scores from CSV features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from pathlib import Path
from mood_utils import classify_mood_from_deam

class DEAMCSVIntegration:
    def __init__(self):
        self.csv_to_deam_valence = None
        self.csv_to_deam_arousal = None
        self.scaler = StandardScaler()
        # NOTE: Removed 'valence' and 'energy' to avoid circular prediction
        # We want to predict emotion from other features, not from Spotify's own emotion estimates
        self.csv_features = [
            'danceability', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'tempo',
            'key', 'mode', 'time_signature'
        ]
        self.csv_data = None

    def load_csv_data(self, csv_path=None):
        print("Loading Spotify CSV dataset...")
        if csv_path is None:
            # Try multiple possible locations
            possible_paths = [
                'spotify-tracks-dataset.csv',
                'data/spotify-tracks-dataset.csv',
                Path(__file__).parent.parent / 'data' / 'spotify-tracks-dataset.csv'
            ]
            for path in possible_paths:
                try:
                    self.csv_data = pd.read_csv(path)
                    print(f"Loaded {len(self.csv_data):,} tracks from {path}")
                    return self.csv_data
                except FileNotFoundError:
                    continue
            raise FileNotFoundError("Could not find spotify-tracks-dataset.csv")
        else:
            self.csv_data = pd.read_csv(csv_path)
            print(f"Loaded {len(self.csv_data):,} tracks")
            return self.csv_data


    def train_csv_to_deam_models(self, deam_data):
        # train random forest to predict DEAM from CSV features
        print("Training CSV to DEAM prediction models...")

        # Prepare features and targets
        X = deam_data[self.csv_features].fillna(0)
        y_valence = deam_data['deam_valence']
        y_arousal = deam_data['deam_arousal']

        print(f"Training data shape: {X.shape}")
        print(f"Valence range: {y_valence.min():.2f} - {y_valence.max():.2f}")
        print(f"Arousal range: {y_arousal.min():.2f} - {y_arousal.max():.2f}")

        # Split for validation
        X_train, X_test, y_val_train, y_val_test, y_ar_train, y_ar_test = train_test_split(
            X, y_valence, y_arousal, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train valence predictor
        print("\nTraining valence predictor...")
        self.csv_to_deam_valence = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            max_features='sqrt',
            min_samples_leaf=1,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.csv_to_deam_valence.fit(X_train_scaled, y_val_train)

        # Evaluate valence model
        val_pred = self.csv_to_deam_valence.predict(X_test_scaled)
        val_r2 = r2_score(y_val_test, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val_test, val_pred))

        print(f"  Valence R²: {val_r2:.3f}")
        print(f"  Valence RMSE: {val_rmse:.3f}")

        # Train arousal predictor
        print("\nTraining arousal predictor...")
        self.csv_to_deam_arousal = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            max_features='sqrt',
            min_samples_leaf=1,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.csv_to_deam_arousal.fit(X_train_scaled, y_ar_train)

        # Evaluate arousal model
        ar_pred = self.csv_to_deam_arousal.predict(X_test_scaled)
        ar_r2 = r2_score(y_ar_test, ar_pred)
        ar_rmse = np.sqrt(mean_squared_error(y_ar_test, ar_pred))

        print(f"  Arousal R²: {ar_r2:.3f}")
        print(f"  Arousal RMSE: {ar_rmse:.3f}")

        return {
            'valence_r2': val_r2,
            'valence_rmse': val_rmse,
            'arousal_r2': ar_r2,
            'arousal_rmse': ar_rmse
        }

    def predict_deam_from_csv(self, track_id):
        # fast prediction using CSV features
        if self.csv_to_deam_valence is None:
            raise ValueError("Models not trained. Call train_csv_to_deam_models() first.")

        # Get track features from CSV
        track_row = self.csv_data[self.csv_data['track_id'] == track_id]
        if track_row.empty:
            return None

        track_data = track_row.iloc[0]

        # Extract features
        features = [track_data[col] for col in self.csv_features]
        features_scaled = self.scaler.transform([features])

        # Predict DEAM valence and arousal
        deam_valence = self.csv_to_deam_valence.predict(features_scaled)[0]
        deam_arousal = self.csv_to_deam_arousal.predict(features_scaled)[0]

        # Convert to mood category using centralized function
        mood = classify_mood_from_deam(deam_valence, deam_arousal)

        return {
            'track_id': track_id,
            'track_name': track_data['track_name'],
            'artists': track_data['artists'],
            'genre': track_data['track_genre'],
            'deam_valence': float(deam_valence),
            'deam_arousal': float(deam_arousal),
            'mood': mood,
            'spotify_valence': track_data['valence'],
            'spotify_energy': track_data['energy']
        }

    # For backwards compatibility - expose the centralized function
    def classify_mood_from_deam(self, valence, arousal):
        """
        Convert DEAM valence/arousal to mood categories
        Uses centralized mood_utils function
        """
        return classify_mood_from_deam(valence, arousal)

    def save_models(self, model_path="deam_csv_integration.pkl"):
        """Save the trained integration models"""
        if self.csv_to_deam_valence is None:
            raise ValueError("No models to save. Train models first.")

        model_data = {
            'valence_model': self.csv_to_deam_valence,
            'arousal_model': self.csv_to_deam_arousal,
            'scaler': self.scaler,
            'csv_features': self.csv_features
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Integration models saved to {model_path}")

    def load_models(self, model_path="deam_csv_integration.pkl"):
        """Load pre-trained integration models"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.csv_to_deam_valence = model_data['valence_model']
        self.csv_to_deam_arousal = model_data['arousal_model']
        self.scaler = model_data['scaler']
        self.csv_features = model_data['csv_features']

        print(f"Integration models loaded from {model_path}")


def main():
    """Main integration pipeline - loads pre-trained models"""
    print("DEAM-CSV INTEGRATION SYSTEM")
    print("=" * 40)

    # Initialize system
    integration = DEAMCSVIntegration()

    # Load CSV data
    integration.load_csv_data()

    # Load pre-trained models
    try:
        integration.load_models('deam_csv_integration.pkl')
        print("Pre-trained integration models loaded successfully")
    except FileNotFoundError:
        print("\nError: Pre-trained models not found.")
        print("Models should be trained using the DEAM neural network predictions as ground truth.")
        print("See TECHNICAL_NOTES.md for training details.")
        return

    # Test the integrated system
    print("\nTesting integrated predictions...")
    sample_tracks = integration.csv_data.sample(5)

    for _, track in sample_tracks.iterrows():
        result = integration.predict_deam_from_csv(track['track_id'])
        if result:
            print(f"\nTrack: {result['track_name']}")
            print(f"  DEAM Valence: {result['deam_valence']:.2f} (Spotify: {result['spotify_valence']:.2f})")
            print(f"  DEAM Arousal: {result['deam_arousal']:.2f} (Spotify: {result['spotify_energy']:.2f})")
            print(f"  Predicted Mood: {result['mood']}")

    print(f"\nSystem ready! Integration model can predict emotion for 114,000+ tracks.")


if __name__ == "__main__":
    main()
