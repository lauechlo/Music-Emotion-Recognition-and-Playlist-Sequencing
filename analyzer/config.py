#!/usr/bin/env python3
"""
Spotify API Configuration
Add your Spotify API credentials here
"""

# Spotify API Credentials
# Get these from: https://developer.spotify.com/dashboard
SPOTIFY_CLIENT_ID = "your_client_id_here"
SPOTIFY_CLIENT_SECRET = "your_client_secret_here"

# Optional: Add redirect URI if you need user authentication later
SPOTIFY_REDIRECT_URI = "http://localhost:8888/callback"

# Validation function
def validate_credentials():
    """Check if credentials are properly set"""
    if SPOTIFY_CLIENT_ID == "your_client_id_here" or SPOTIFY_CLIENT_SECRET == "your_client_secret_here":
        raise ValueError(
            "Please update config.py with your actual Spotify API credentials.\n"
            "Get them from: https://developer.spotify.com/dashboard"
        )
    return True
