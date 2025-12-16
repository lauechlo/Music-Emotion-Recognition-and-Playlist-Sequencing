"""
Unified music emotion analysis - combines API, CSV, and DEAM
"""

import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from deam_csv_integration import DEAMCSVIntegration
from mood_utils import classify_mood_from_spotify
import time
import os

try:
    from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
except ImportError:
    SPOTIFY_CLIENT_ID = None
    SPOTIFY_CLIENT_SECRET = None

class UnifiedMusicAnalyzer:
    def __init__(self, spotify_client_id=None, spotify_client_secret=None):
        # tries API first, then CSV, then fallback
        self.spotify = None
        self.csv_integration = DEAMCSVIntegration()
        self.csv_data = None

        # Initialize Spotify API
        if spotify_client_id and spotify_client_secret:
            self.init_spotify_api(spotify_client_id, spotify_client_secret)
        elif SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
            self.init_spotify_api(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
        else:
            print("Warning: No Spotify credentials provided. CSV-only mode.")

        # Load CSV integration models
        try:
            self.csv_integration.load_csv_data()
            self.csv_integration.load_models('deam_csv_integration.pkl')
            print("CSV integration models loaded successfully")
        except:
            print("Warning: CSV integration models not found. Run deam_csv_integration.py first.")

    def init_spotify_api(self, client_id, client_secret):
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            self.spotify = spotipy.Spotify(
                client_credentials_manager=client_credentials_manager
            )
            print("Spotify API initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Spotify API: {e}")
            self.spotify = None

    def get_track_info(self, track_id):
        if not self.spotify:
            return None

        try:
            # Get basic track info
            track = self.spotify.track(track_id)
            if not track:
                return None

            # Get audio features
            audio_features = self.spotify.audio_features([track_id])[0]
            if not audio_features:
                return None

            # Combine information
            track_info = {
                'track_id': track_id,
                'track_name': track['name'],
                'artists': ', '.join([artist['name'] for artist in track['artists']]),
                'album_name': track['album']['name'],
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms'],
                'preview_url': track['preview_url'],
                **audio_features  # Include all Spotify audio features
            }

            return track_info

        except Exception as e:
            print(f"Error getting track info for {track_id}: {e}")
            return None

    def analyze_track(self, track_id, method='auto'):
        # auto tries API -> CSV -> fallback
        # cascading approach makes it work for any track
        if method == 'auto':
            # Try different methods in priority order
            result = self.analyze_with_api(track_id)
            if result:
                return result

            result = self.analyze_with_csv_integration(track_id)
            if result:
                return result

            return self.analyze_with_csv_fallback(track_id)

        elif method == 'api':
            return self.analyze_with_api(track_id)

        elif method == 'csv':
            return self.analyze_with_csv_fallback(track_id)

        elif method == 'integrated':
            return self.analyze_with_csv_integration(track_id)

        else:
            raise ValueError(f"Unknown method: {method}")

    def analyze_with_api(self, track_id):
        # use spotify API + DEAM integration
        if not self.spotify:
            return None

        # Get track information
        track_info = self.get_track_info(track_id)
        if not track_info:
            return None

        # Use CSV→DEAM integration to predict from Spotify features
        if hasattr(self.csv_integration, 'csv_to_deam_valence') and self.csv_integration.csv_to_deam_valence:
            try:
                # Prepare features for DEAM prediction
                features = [
                    track_info.get('danceability', 0),
                    track_info.get('energy', 0),
                    track_info.get('loudness', -10),
                    track_info.get('speechiness', 0),
                    track_info.get('acousticness', 0),
                    track_info.get('instrumentalness', 0),
                    track_info.get('liveness', 0),
                    track_info.get('valence', 0),
                    track_info.get('tempo', 120)
                ]

                features_scaled = self.csv_integration.scaler.transform([features])

                # Predict DEAM valence and arousal
                deam_valence = self.csv_integration.csv_to_deam_valence.predict(features_scaled)[0]
                deam_arousal = self.csv_integration.csv_to_deam_arousal.predict(features_scaled)[0]

                # Classify mood
                mood = self.csv_integration.classify_mood_from_deam(deam_valence, deam_arousal)

                track_info.update({
                    'deam_valence': float(deam_valence),
                    'deam_arousal': float(deam_arousal),
                    'mood': mood,
                    'analysis_method': 'spotify_api_with_deam_integration',
                    'confidence': 0.85  # High confidence for API data
                })

            except Exception as e:
                print(f"Error in DEAM integration: {e}")
                # Fallback to simple mood classification
                track_info.update({
                    'mood': classify_mood_from_spotify(
                        track_info.get('valence', 0.5),
                        track_info.get('energy', 0.5),
                        track_info.get('danceability', 0.5)
                    ),
                    'analysis_method': 'spotify_api_simple',
                    'confidence': 0.7
                })

        else:
            # Simple mood classification from Spotify features
            track_info.update({
                'mood': classify_mood_from_spotify(
                    track_info.get('valence', 0.5),
                    track_info.get('energy', 0.5),
                    track_info.get('danceability', 0.5)
                ),
                'analysis_method': 'spotify_api_simple',
                'confidence': 0.7
            })

        return track_info

    def analyze_with_csv_integration(self, track_id):
        # use CSV -> DEAM model (fastest, pretty accurate)
        if not hasattr(self.csv_integration, 'csv_to_deam_valence') or not self.csv_integration.csv_to_deam_valence:
            return None

        try:
            result = self.csv_integration.predict_deam_from_csv(track_id)
            if result:
                result['analysis_method'] = 'csv_deam_integration'
                result['confidence'] = 0.9  # High confidence for integrated model
            return result
        except Exception as e:
            print(f"Error in CSV integration: {e}")
            return None

    def analyze_with_csv_fallback(self, track_id):
        """Fallback analysis using basic CSV data"""
        if not hasattr(self.csv_integration, 'csv_data') or self.csv_integration.csv_data is None:
            return None

        # Look up track in CSV data
        track_row = self.csv_integration.csv_data[self.csv_integration.csv_data['track_id'] == track_id]
        if track_row.empty:
            return None

        track_data = track_row.iloc[0]

        # Simple mood classification using centralized function
        mood = classify_mood_from_spotify(
            track_data['valence'],
            track_data['energy'],
            track_data.get('danceability', 0.5)
        )

        return {
            'track_id': track_id,
            'track_name': track_data['track_name'],
            'artists': track_data['artists'],
            'genre': track_data['track_genre'],
            'mood': mood,
            'spotify_valence': track_data['valence'],
            'spotify_energy': track_data['energy'],
            'popularity': track_data['popularity'],
            'analysis_method': 'csv_fallback',
            'confidence': 0.6
        }

    def analyze_playlist(self, track_ids, show_progress=True):
        """Analyze multiple tracks efficiently"""
        results = []

        print(f"Analyzing {len(track_ids)} tracks...")

        for i, track_id in enumerate(track_ids):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(track_ids)} tracks...")

            result = self.analyze_track(track_id, method='auto')
            if result:
                results.append(result)

            # Small delay to avoid rate limits
            time.sleep(0.1)

        return results

    def get_recommendations_by_mood(self, target_mood, limit=20, min_popularity=30):
        """Get track recommendations for a specific mood"""
        recommendations = []

        if not hasattr(self.csv_integration, 'csv_data') or self.csv_integration.csv_data is None:
            print("CSV data not available for recommendations")
            return recommendations

        # Filter tracks by mood prediction
        candidate_tracks = self.csv_integration.csv_data[
            self.csv_integration.csv_data['popularity'] >= min_popularity
        ].sample(min(1000, len(self.csv_integration.csv_data)))  # Sample for efficiency

        for _, track in candidate_tracks.iterrows():
            result = self.analyze_track(track['track_id'], method='integrated')
            if result and result.get('mood') == target_mood:
                recommendations.append(result)

            if len(recommendations) >= limit:
                break

        # Sort by confidence and popularity
        recommendations.sort(key=lambda x: (x.get('confidence', 0), x.get('popularity', 0)), reverse=True)

        return recommendations[:limit]

    def compare_methods(self, track_ids):
        """Compare different analysis methods on the same tracks"""
        comparison_results = []

        for track_id in track_ids:
            track_comparison = {'track_id': track_id}

            # Try each method
            methods = ['api', 'integrated', 'csv']
            for method in methods:
                result = self.analyze_track(track_id, method=method)
                if result:
                    track_comparison[f'{method}_mood'] = result.get('mood')
                    track_comparison[f'{method}_confidence'] = result.get('confidence')
                    track_comparison[f'{method}_method'] = result.get('analysis_method')

                    # Store additional info from first successful method
                    if 'track_name' not in track_comparison:
                        track_comparison['track_name'] = result.get('track_name', 'Unknown')
                        track_comparison['artists'] = result.get('artists', 'Unknown')

            comparison_results.append(track_comparison)

        return comparison_results


def main():
    """Demo the unified analysis system"""
    print("UNIFIED MUSIC EMOTION ANALYSIS SYSTEM")
    print("=" * 50)

    # Initialize system
    analyzer = UnifiedMusicAnalyzer()

    # Test tracks (mix from different sources)
    test_tracks = [
        "4iV5W9uYEdYUVa79Axb7Rh",  # Happy - Pharrell Williams
        "7qiZfU4dY1lWllzX7mPBI3",  # Shape of You - Ed Sheeran
        "0VjIjW4GlUZAMYd2vXMi3b",  # Blinding Lights - The Weeknd
    ]

    print("\n1. INDIVIDUAL TRACK ANALYSIS:")
    for track_id in test_tracks:
        print(f"\nAnalyzing track: {track_id}")
        result = analyzer.analyze_track(track_id, method='auto')

        if result:
            print(f"  Track: {result.get('track_name', 'Unknown')} by {result.get('artists', 'Unknown')}")
            print(f"  Mood: {result.get('mood', 'unknown')}")
            print(f"  Method: {result.get('analysis_method', 'unknown')}")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")

            if 'deam_valence' in result:
                print(f"  DEAM Valence: {result['deam_valence']:.2f}")
                print(f"  DEAM Arousal: {result['deam_arousal']:.2f}")
        else:
            print(f"  Could not analyze track {track_id}")

    print(f"\n2. METHOD COMPARISON:")
    comparison = analyzer.compare_methods(test_tracks[:2])  # Limit for demo

    for comp in comparison:
        print(f"\nTrack: {comp.get('track_name', 'Unknown')}")
        for method in ['api', 'integrated', 'csv']:
            mood = comp.get(f'{method}_mood', 'N/A')
            confidence = comp.get(f'{method}_confidence', 0)
            print(f"  {method}: {mood} (conf: {confidence:.2f})")

    print(f"\n3. MOOD-BASED RECOMMENDATIONS:")
    recommendations = analyzer.get_recommendations_by_mood('energetic', limit=5)

    print("Top energetic tracks:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec.get('track_name', 'Unknown')} by {rec.get('artists', 'Unknown')}")
        print(f"     Confidence: {rec.get('confidence', 0):.2f}")

    print(f"\nUnified system ready! Can analyze any Spotify track with multiple fallback methods.")


if __name__ == "__main__":
    main()
