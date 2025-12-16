"""
Mood classification utilities
"""

def classify_mood_from_deam(valence, arousal):
    # convert DEAM scores (1-9 scale) to mood categories
    v_norm = (valence - 1) / 8
    a_norm = (arousal - 1) / 8

    # using Russell's circumplex model
    if v_norm >= 0.7 and a_norm >= 0.7:
        return "energetic"
    elif v_norm >= 0.6 and a_norm >= 0.4:
        return "joyful"
    elif v_norm >= 0.5 and a_norm <= 0.4:
        return "calm"
    elif v_norm <= 0.4 and a_norm <= 0.4:
        return "melancholy"
    elif 0.3 <= v_norm <= 0.7 and 0.3 <= a_norm <= 0.7:
        return "focused"
    else:
        return "neutral"


def classify_mood_from_spotify(valence, energy, danceability=None):
    # same as above but for spotify features (0-1 scale)
    if danceability is None:
        danceability = 0.5

    if valence >= 0.7 and energy >= 0.7:
        return "energetic"
    elif valence >= 0.6 and energy >= 0.4 and danceability >= 0.6:
        return "joyful"
    elif valence >= 0.5 and energy <= 0.4:
        return "calm"
    elif valence <= 0.4 and energy <= 0.4:
        return "melancholy"
    elif 0.3 <= valence <= 0.7 and 0.3 <= energy <= 0.7:
        return "focused"
    else:
        return "neutral"


def spotify_to_deam_scale(valence, energy):
    # convert 0-1 to 1-9 scale
    return 1 + (valence * 8), 1 + (energy * 8)


def deam_to_spotify_scale(deam_valence, deam_arousal):
    # inverse of above
    return (deam_valence - 1) / 8, (deam_arousal - 1) / 8
