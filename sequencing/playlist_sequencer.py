"""
Playlist sequencing algorithms
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
import random
from dataclasses import dataclass


@dataclass
class SequencingMetrics:
    # metrics for evaluating sequences
    transition_smoothness: float
    flow_variance: float
    energy_gradient: float
    diversity_score: float
    trajectory_score: float
    total_score: float


class PlaylistSequencer:
    # main sequencer class with 11 different algorithms

    def __init__(self, analyzer=None):
        self.analyzer = analyzer
        self.tracks_data = []

    def load_playlist(self, track_ids):
        # load tracks from analyzer
        if not self.analyzer:
            raise ValueError("Analyzer not initialized")

        print(f"Analyzing {len(track_ids)} tracks...")
        self.tracks_data = []

        for i, track_id in enumerate(track_ids):
            result = self.analyzer.analyze_track(track_id, method='auto')
            if result:
                # Ensure we have valence/arousal scores
                if 'deam_valence' not in result:
                    # Estimate from Spotify valence/energy
                    result['deam_valence'] = 1 + (result.get('spotify_valence', 0.5) * 8)
                    result['deam_arousal'] = 1 + (result.get('spotify_energy', 0.5) * 8)

                self.tracks_data.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(track_ids)} tracks")

        print(f"Successfully loaded {len(self.tracks_data)} tracks")
        return self.tracks_data

    def load_from_analyzed_data(self, tracks_data):
        self.tracks_data = tracks_data
        return self.tracks_data

    def emotional_distance(self, track1, track2):
        # euclidean distance in valence-arousal space
        v1 = (track1.get('deam_valence', 5) - 1) / 8  # Normalize to 0-1
        a1 = (track1.get('deam_arousal', 5) - 1) / 8

        v2 = (track2.get('deam_valence', 5) - 1) / 8
        a2 = (track2.get('deam_arousal', 5) - 1) / 8

        return euclidean([v1, a1], [v2, a2])

    def transition_smoothness(self, sequence):
        # average distance between consecutive tracks
        if len(sequence) < 2:
            return 0.0

        distances = []
        for i in range(len(sequence) - 1):
            dist = self.emotional_distance(sequence[i], sequence[i + 1])
            distances.append(dist)

        return np.mean(distances)

    def flow_variance(self, sequence):
        # variance in transition distances
        if len(sequence) < 2:
            return 0.0

        distances = []
        for i in range(len(sequence) - 1):
            dist = self.emotional_distance(sequence[i], sequence[i + 1])
            distances.append(dist)

        return np.std(distances)

    def energy_gradient(self, sequence):
        # how energy changes over time (positive = increasing)
        if len(sequence) < 2:
            return 0.0

        energy_values = [
            (track.get('deam_arousal', 5) - 1) / 8
            for track in sequence
        ]

        # Linear regression coefficient
        x = np.arange(len(energy_values))
        gradient = np.polyfit(x, energy_values, 1)[0]

        return gradient

    def diversity_score(self, sequence):
        # artist and genre diversity
        if len(sequence) < 2:
            return 0.0

        # Count unique artists
        artists = [track.get('artists', '') for track in sequence]
        unique_artists = len(set(artists))
        artist_diversity = unique_artists / len(sequence)

        # Count unique genres
        genres = [track.get('genre', '') for track in sequence]
        unique_genres = len(set(genres))
        genre_diversity = unique_genres / len(sequence) if unique_genres > 0 else 0.5

        # Average diversity
        return (artist_diversity + genre_diversity) / 2

    def trajectory_alignment(self, sequence, target_arc='auto'):
        # check if sequence matches target emotional trajectory
        if len(sequence) < 3:
            return 0.5

        # Extract valence and arousal trajectories
        valence = [(track.get('deam_valence', 5) - 1) / 8 for track in sequence]
        arousal = [(track.get('deam_arousal', 5) - 1) / 8 for track in sequence]

        # Auto-detect trajectory if not specified
        if target_arc == 'auto':
            energy_slope = np.polyfit(range(len(arousal)), arousal, 1)[0]
            if energy_slope > 0.05:
                target_arc = 'buildup'
            elif energy_slope < -0.05:
                target_arc = 'cooldown'
            else:
                target_arc = 'stable'

        # Score based on target trajectory
        if target_arc == 'buildup':
            # Should have positive energy gradient
            gradient = np.polyfit(range(len(arousal)), arousal, 1)[0]
            return max(0, min(1, gradient * 2 + 0.5))

        elif target_arc == 'cooldown':
            # Should have negative energy gradient
            gradient = np.polyfit(range(len(arousal)), arousal, 1)[0]
            return max(0, min(1, -gradient * 2 + 0.5))

        elif target_arc == 'wave':
            # Should have peaks and valleys
            variance = np.var(arousal)
            return min(1, variance * 5)  # Higher variance = better wave

        elif target_arc == 'stable':
            # Should have low variance
            variance = np.var(arousal)
            return max(0, 1 - variance * 5)  # Lower variance = better stability

        return 0.5

    def calculate_all_metrics(self, sequence, target_arc='auto'):
        # calculate all metrics for a given sequence
        smoothness = self.transition_smoothness(sequence)
        variance = self.flow_variance(sequence)
        gradient = self.energy_gradient(sequence)
        diversity = self.diversity_score(sequence)
        trajectory = self.trajectory_alignment(sequence, target_arc)

        # Calculate weighted total score (higher = better)
        # Normalize smoothness (invert so higher = better)
        smoothness_score = max(0, 1 - smoothness * 2)  # Assuming max distance ~0.5
        variance_score = max(0, 1 - variance * 4)

        total = (
            smoothness_score * 0.35 +  # Most important
            trajectory * 0.25 +
            diversity * 0.20 +
            variance_score * 0.15 +
            0.05  # Baseline
        )

        return SequencingMetrics(
            transition_smoothness=smoothness,
            flow_variance=variance,
            energy_gradient=gradient,
            diversity_score=diversity,
            trajectory_score=trajectory,
            total_score=total
        )

    # basic sequencing algorithms

    def sequence_greedy_smooth(self, tracks=None):
        # greedy nearest neighbor - picks closest track each time
        if tracks is None:
            tracks = self.tracks_data.copy()

        if len(tracks) <= 1:
            return tracks

        # Start with a middle track (balanced emotion)
        valence_scores = [(track.get('deam_valence', 5) - 1) / 8 for track in tracks]
        arousal_scores = [(track.get('deam_arousal', 5) - 1) / 8 for track in tracks]

        # Find most central track
        median_v = np.median(valence_scores)
        median_a = np.median(arousal_scores)

        distances_to_center = [
            euclidean([v, a], [median_v, median_a])
            for v, a in zip(valence_scores, arousal_scores)
        ]
        start_idx = np.argmin(distances_to_center)

        sequence = [tracks[start_idx]]
        remaining = tracks[:start_idx] + tracks[start_idx + 1:]

        # Greedy nearest neighbor
        while remaining:
            current = sequence[-1]

            # Find closest remaining track
            distances = [self.emotional_distance(current, track) for track in remaining]
            nearest_idx = np.argmin(distances)

            sequence.append(remaining[nearest_idx])
            remaining.pop(nearest_idx)

        return sequence

    def sequence_by_trajectory(self, tracks=None, arc_type='buildup'):
        # sequence by energy trajectory (buildup/cooldown/wave/stable)
        if tracks is None:
            tracks = self.tracks_data.copy()

        if len(tracks) <= 1:
            return tracks

        # Sort by arousal (energy)
        arousal_scores = [(track.get('deam_arousal', 5) - 1) / 8 for track in tracks]
        sorted_indices = np.argsort(arousal_scores)

        if arc_type == 'buildup':
            # Low to high energy
            return [tracks[i] for i in sorted_indices]

        elif arc_type == 'cooldown':
            # High to low energy
            return [tracks[i] for i in sorted_indices[::-1]]

        elif arc_type == 'wave':
            # Alternating high/low (create peaks and valleys)
            low_energy = sorted_indices[:len(sorted_indices)//2]
            high_energy = sorted_indices[len(sorted_indices)//2:]

            sequence = []
            for i in range(max(len(low_energy), len(high_energy))):
                if i < len(low_energy):
                    sequence.append(tracks[low_energy[i]])
                if i < len(high_energy):
                    sequence.append(tracks[high_energy[i]])
            return sequence

        elif arc_type == 'stable':
            # Sort by distance from median, keep similar tracks together
            median_arousal = np.median(arousal_scores)
            distances = [abs(a - median_arousal) for a in arousal_scores]
            sorted_indices = np.argsort(distances)
            return [tracks[i] for i in sorted_indices]

        return tracks

    def sequence_clustered(self, tracks=None, n_clusters=3):
        # k-means clustering then sequence within clusters
        if tracks is None:
            tracks = self.tracks_data.copy()

        if len(tracks) <= n_clusters:
            return tracks

        # Extract emotion features
        features = np.array([
            [
                (track.get('deam_valence', 5) - 1) / 8,
                (track.get('deam_arousal', 5) - 1) / 8
            ]
            for track in tracks
        ])

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        # Group by cluster
        clusters = [[] for _ in range(n_clusters)]
        for track, label in zip(tracks, labels):
            clusters[label].append(track)

        # Order clusters by average arousal (energy progression)
        cluster_arousal = [
            np.mean([(t.get('deam_arousal', 5) - 1) / 8 for t in cluster])
            for cluster in clusters
        ]
        cluster_order = np.argsort(cluster_arousal)

        # Sequence within each cluster, then concatenate
        sequence = []
        for cluster_idx in cluster_order:
            cluster_tracks = clusters[cluster_idx]
            # Use greedy smooth within cluster
            if len(cluster_tracks) > 1:
                cluster_seq = self.sequence_greedy_smooth(cluster_tracks)
                sequence.extend(cluster_seq)
            else:
                sequence.extend(cluster_tracks)

        return sequence

    # advanced optimization algorithms

    def sequence_greedy_lookahead(self, tracks=None, lookahead=3):
        # greedy but looks ahead N steps instead of just 1
        # could optimize this more but it works
        if tracks is None:
            tracks = self.tracks_data.copy()

        if len(tracks) <= 1:
            return tracks

        # Start from central track (same as greedy_smooth)
        valence_scores = [(track.get('deam_valence', 5) - 1) / 8 for track in tracks]
        arousal_scores = [(track.get('deam_arousal', 5) - 1) / 8 for track in tracks]

        median_v = np.median(valence_scores)
        median_a = np.median(arousal_scores)

        distances_to_center = [
            euclidean([v, a], [median_v, median_a])
            for v, a in zip(valence_scores, arousal_scores)
        ]
        start_idx = np.argmin(distances_to_center)

        sequence = [tracks[start_idx]]
        remaining = tracks[:start_idx] + tracks[start_idx + 1:]

        # Build sequence with lookahead
        while remaining:
            current = sequence[-1]

            if len(remaining) <= lookahead:
                # Not enough tracks left for lookahead, use simple greedy
                distances = [self.emotional_distance(current, track) for track in remaining]
                best_idx = np.argmin(distances)
            else:
                # Evaluate paths of length 'lookahead'
                best_idx = None
                best_score = float('inf')

                for i, next_track in enumerate(remaining):
                    # Start path with this candidate
                    path_score = self.emotional_distance(current, next_track)

                    # Simulate lookahead steps
                    temp_remaining = remaining[:i] + remaining[i+1:]
                    temp_current = next_track

                    for step in range(min(lookahead - 1, len(temp_remaining))):
                        # Find best next step in simulation
                        distances = [self.emotional_distance(temp_current, t) for t in temp_remaining]
                        next_idx = np.argmin(distances)
                        path_score += distances[next_idx]
                        temp_current = temp_remaining[next_idx]
                        temp_remaining = temp_remaining[:next_idx] + temp_remaining[next_idx+1:]

                    # Average score per step
                    avg_score = path_score / min(lookahead, len(remaining) - i + 1)

                    if avg_score < best_score:
                        best_score = avg_score
                        best_idx = i

            sequence.append(remaining[best_idx])
            remaining.pop(best_idx)

        return sequence

    def sequence_2opt(self, tracks=None, max_iterations=100):
        # 2-opt optimization - swaps pairs to improve
        if tracks is None:
            tracks = self.tracks_data.copy()

        if len(tracks) <= 3:
            return tracks

        # Start with greedy solution
        sequence = self.sequence_greedy_smooth(tracks)

        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try all possible 2-opt swaps
            for i in range(1, len(sequence) - 2):
                for j in range(i + 1, len(sequence)):
                    # Calculate current cost
                    current_cost = (
                        self.emotional_distance(sequence[i-1], sequence[i]) +
                        self.emotional_distance(sequence[j-1], sequence[j])
                    )

                    # Calculate new cost after swap
                    new_cost = (
                        self.emotional_distance(sequence[i-1], sequence[j-1]) +
                        self.emotional_distance(sequence[i], sequence[j])
                    )

                    # If improvement, apply swap
                    if new_cost < current_cost:
                        # Reverse segment between i and j
                        sequence[i:j] = sequence[i:j][::-1]
                        improved = True

        return sequence

    def sequence_simulated_annealing(self, tracks=None, initial_temp=1.0,
                                     cooling_rate=0.95, max_iterations=500):
        # simulated annealing - accepts worse solutions sometimes to escape local optima
        # got this idea from algorithms class, works pretty well
        if tracks is None:
            tracks = self.tracks_data.copy()

        if len(tracks) <= 2:
            return tracks

        # Start with greedy solution
        current_sequence = self.sequence_greedy_smooth(tracks)
        current_score = self.transition_smoothness(current_sequence)

        best_sequence = current_sequence.copy()
        best_score = current_score

        temperature = initial_temp

        for iteration in range(max_iterations):
            # Generate neighbor by swapping two random tracks
            new_sequence = current_sequence.copy()
            i, j = random.sample(range(len(new_sequence)), 2)
            new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]

            # Calculate new score (lower is better)
            new_score = self.transition_smoothness(new_sequence)

            # Calculate acceptance probability
            delta = new_score - current_score

            if delta < 0:
                # Better solution, always accept
                accept = True
            else:
                # Worse solution, accept with probability based on temperature
                accept_prob = np.exp(-delta / temperature)
                accept = random.random() < accept_prob

            if accept:
                current_sequence = new_sequence
                current_score = new_score

                # Update best if this is the best we've seen
                if current_score < best_score:
                    best_sequence = current_sequence.copy()
                    best_score = current_score

            # Cool down temperature
            temperature *= cooling_rate

        return best_sequence

    def sequence_beam_search(self, tracks=None, beam_width=3):
        # beam search - keeps top K candidates at each step
        if tracks is None:
            tracks = self.tracks_data.copy()

        if len(tracks) <= 1:
            return tracks

        # Find central starting track
        valence_scores = [(track.get('deam_valence', 5) - 1) / 8 for track in tracks]
        arousal_scores = [(track.get('deam_arousal', 5) - 1) / 8 for track in tracks]

        median_v = np.median(valence_scores)
        median_a = np.median(arousal_scores)

        distances_to_center = [
            euclidean([v, a], [median_v, median_a])
            for v, a in zip(valence_scores, arousal_scores)
        ]
        start_idx = np.argmin(distances_to_center)
        start_track = tracks[start_idx]

        # Initialize beam with starting track
        # Each candidate is (sequence, remaining_tracks, cumulative_score)
        beam = [([start_track],
                 tracks[:start_idx] + tracks[start_idx + 1:],
                 0.0)]

        # Build sequences
        while beam[0][1]:  # While there are remaining tracks
            new_beam = []

            for sequence, remaining, cum_score in beam:
                # Expand this candidate
                current = sequence[-1]

                # Score all possible next tracks
                candidates = []
                for i, next_track in enumerate(remaining):
                    distance = self.emotional_distance(current, next_track)
                    new_score = cum_score + distance
                    new_remaining = remaining[:i] + remaining[i+1:]
                    candidates.append((sequence + [next_track], new_remaining, new_score))

                new_beam.extend(candidates)

            # Keep only top beam_width candidates
            new_beam.sort(key=lambda x: x[2])  # Sort by cumulative score
            beam = new_beam[:beam_width]

        # Return best complete sequence
        return beam[0][0]

    def sequence_adaptive_greedy(self, tracks=None, bridge_probability=0.15):
        # like greedy but occasionally makes strategic jumps
        if tracks is None:
            tracks = self.tracks_data.copy()

        if len(tracks) <= 1:
            return tracks

        # Start from central track
        valence_scores = [(track.get('deam_valence', 5) - 1) / 8 for track in tracks]
        arousal_scores = [(track.get('deam_arousal', 5) - 1) / 8 for track in tracks]

        median_v = np.median(valence_scores)
        median_a = np.median(arousal_scores)

        distances_to_center = [
            euclidean([v, a], [median_v, median_a])
            for v, a in zip(valence_scores, arousal_scores)
        ]
        start_idx = np.argmin(distances_to_center)

        sequence = [tracks[start_idx]]
        remaining = tracks[:start_idx] + tracks[start_idx + 1:]

        while remaining:
            current = sequence[-1]

            # Decide: greedy or bridge?
            if random.random() < bridge_probability and len(remaining) > 5:
                # Bridge move: find a "connector" track
                connectivity_scores = []
                for candidate in remaining:
                    # Average distance to other remaining tracks
                    avg_dist = np.mean([
                        self.emotional_distance(candidate, other)
                        for other in remaining if other != candidate
                    ])
                    connectivity_scores.append(avg_dist)

                # Pick track with best average connectivity
                best_idx = np.argmin(connectivity_scores)
            else:
                # Greedy move: pick nearest
                distances = [self.emotional_distance(current, track) for track in remaining]
                best_idx = np.argmin(distances)

            sequence.append(remaining[best_idx])
            remaining.pop(best_idx)

        return sequence

    def compare_strategies(self, target_arc='auto'):
        # run all strategies and compare results
        if not self.tracks_data:
            raise ValueError("No tracks loaded")

        strategies = {
            # Baseline
            'original': self.tracks_data.copy(),

            # Basic strategies
            'greedy_smooth': self.sequence_greedy_smooth(),
            'buildup': self.sequence_by_trajectory(arc_type='buildup'),
            'cooldown': self.sequence_by_trajectory(arc_type='cooldown'),
            'wave': self.sequence_by_trajectory(arc_type='wave'),
            'clustered_3': self.sequence_clustered(n_clusters=3),

            # Advanced optimizations
            'greedy_lookahead': self.sequence_greedy_lookahead(lookahead=3),
            '2opt': self.sequence_2opt(max_iterations=100),
            'simulated_annealing': self.sequence_simulated_annealing(
                initial_temp=1.0, cooling_rate=0.95, max_iterations=500
            ),
            'beam_search': self.sequence_beam_search(beam_width=3),
            'adaptive_greedy': self.sequence_adaptive_greedy(bridge_probability=0.15),
        }

        results = {}
        for name, sequence in strategies.items():
            metrics = self.calculate_all_metrics(sequence, target_arc)
            results[name] = {
                'sequence': sequence,
                'metrics': metrics
            }

        return results

    def visualize_sequence(self, sequence, title="Playlist Sequence"):
        # plot emotional trajectory
        if len(sequence) < 2:
            print("Need at least 2 tracks to visualize")
            return

        # Extract data
        positions = list(range(len(sequence)))
        valence = [(track.get('deam_valence', 5) - 1) / 8 for track in sequence]
        arousal = [(track.get('deam_arousal', 5) - 1) / 8 for track in sequence]

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Valence/Arousal scatter with connection lines
        ax1 = axes[0]
        ax1.scatter(valence, arousal, c=positions, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        ax1.plot(valence, arousal, 'gray', alpha=0.3, linewidth=1)

        # Add arrows to show direction
        for i in range(len(sequence) - 1):
            ax1.annotate('', xy=(valence[i+1], arousal[i+1]), xytext=(valence[i], arousal[i]),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1))

        ax1.set_xlabel('Valence (Negative ← → Positive)', fontsize=12)
        ax1.set_ylabel('Arousal (Calm ← → Energetic)', fontsize=12)
        ax1.set_title('Emotional Space Trajectory', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)

        # Add quadrant labels
        ax1.text(0.85, 0.85, 'Energetic\nJoyful', ha='center', va='center', fontsize=9, alpha=0.5)
        ax1.text(0.15, 0.85, 'Tense\nAnxious', ha='center', va='center', fontsize=9, alpha=0.5)
        ax1.text(0.85, 0.15, 'Calm\nPeaceful', ha='center', va='center', fontsize=9, alpha=0.5)
        ax1.text(0.15, 0.15, 'Sad\nMelancholy', ha='center', va='center', fontsize=9, alpha=0.5)

        # 2. Valence over time
        ax2 = axes[1]
        ax2.plot(positions, valence, marker='o', linewidth=2, markersize=6, color='#2ecc71')
        ax2.fill_between(positions, valence, alpha=0.3, color='#2ecc71')
        ax2.set_xlabel('Track Position', fontsize=12)
        ax2.set_ylabel('Valence', fontsize=12)
        ax2.set_title('Valence Trajectory', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)

        # 3. Arousal over time
        ax3 = axes[2]
        ax3.plot(positions, arousal, marker='o', linewidth=2, markersize=6, color='#e74c3c')
        ax3.fill_between(positions, arousal, alpha=0.3, color='#e74c3c')
        ax3.set_xlabel('Track Position', fontsize=12)
        ax3.set_ylabel('Arousal (Energy)', fontsize=12)
        ax3.set_title('Energy Trajectory', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.05, 1.05)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        return fig

    def visualize_comparison(self, comparison_results):
        # plot comparison of all strategies
        strategies = list(comparison_results.keys())

        # Extract metrics
        smoothness = [comparison_results[s]['metrics'].transition_smoothness for s in strategies]
        variance = [comparison_results[s]['metrics'].flow_variance for s in strategies]
        diversity = [comparison_results[s]['metrics'].diversity_score for s in strategies]
        trajectory = [comparison_results[s]['metrics'].trajectory_score for s in strategies]
        total = [comparison_results[s]['metrics'].total_score for s in strategies]

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Transition Smoothness (lower is better)
        ax1 = axes[0, 0]
        bars1 = ax1.bar(strategies, smoothness, color='#3498db')
        ax1.set_ylabel('Avg Distance', fontsize=11)
        ax1.set_title('Transition Smoothness\n(Lower = Smoother)', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        # 2. Flow Variance (lower is better)
        ax2 = axes[0, 1]
        bars2 = ax2.bar(strategies, variance, color='#9b59b6')
        ax2.set_ylabel('Std Dev', fontsize=11)
        ax2.set_title('Flow Consistency\n(Lower = More Consistent)', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)

        # 3. Diversity Score (higher is better)
        ax3 = axes[0, 2]
        bars3 = ax3.bar(strategies, diversity, color='#2ecc71')
        ax3.set_ylabel('Diversity Score', fontsize=11)
        ax3.set_title('Artist/Genre Diversity\n(Higher = More Diverse)', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1)
        ax3.grid(axis='y', alpha=0.3)

        # 4. Trajectory Alignment (higher is better)
        ax4 = axes[1, 0]
        bars4 = ax4.bar(strategies, trajectory, color='#e67e22')
        ax4.set_ylabel('Alignment Score', fontsize=11)
        ax4.set_title('Trajectory Alignment\n(Higher = Better Match)', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 1)
        ax4.grid(axis='y', alpha=0.3)

        # 5. Total Score (higher is better)
        ax5 = axes[1, 1]
        bars5 = ax5.bar(strategies, total, color='#e74c3c')
        ax5.set_ylabel('Total Score', fontsize=11)
        ax5.set_title('Overall Quality Score\n(Higher = Better)', fontsize=12, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.set_ylim(0, 1)
        ax5.grid(axis='y', alpha=0.3)

        # Highlight best strategy
        best_idx = np.argmax(total)
        bars5[best_idx].set_color('#f39c12')
        bars5[best_idx].set_edgecolor('black')
        bars5[best_idx].set_linewidth(2)

        # 6. Summary table
        ax6 = axes[1, 2]
        ax6.axis('off')

        # Create summary text
        best_strategy = strategies[best_idx]
        summary_text = f"BEST STRATEGY: {best_strategy.upper()}\n\n"
        summary_text += f"Total Score: {total[best_idx]:.3f}\n\n"
        summary_text += "Rankings:\n"

        # Rank by total score
        ranked = sorted(zip(strategies, total), key=lambda x: x[1], reverse=True)
        for i, (strat, score) in enumerate(ranked[:5], 1):  # Top 5 only
            summary_text += f"{i}. {strat}: {score:.3f}\n"

        ax6.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('Sequencing Strategy Comparison (11 Algorithms)', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        return fig


def main():
    """Demo the playlist sequencer"""
    print("PLAYLIST SEQUENCING SYSTEM - 11 ALGORITHMS")
    print("=" * 70)

    # For demo, create mock tracks
    print("\nCreating mock playlist for demonstration...")

    mock_tracks = [
        {'track_id': '1', 'track_name': 'Euphoric Dance', 'artists': 'DJ Energy', 'genre': 'edm',
         'deam_valence': 8.5, 'deam_arousal': 8.8, 'mood': 'energetic'},
        {'track_id': '2', 'track_name': 'Rainy Day Blues', 'artists': 'The Sad Ones', 'genre': 'blues',
         'deam_valence': 2.3, 'deam_arousal': 2.8, 'mood': 'melancholy'},
        {'track_id': '3', 'track_name': 'Gym Pump', 'artists': 'Fitness Beats', 'genre': 'electronic',
         'deam_valence': 7.2, 'deam_arousal': 9.0, 'mood': 'energetic'},
        {'track_id': '4', 'track_name': 'Meditation Flow', 'artists': 'Zen Master', 'genre': 'ambient',
         'deam_valence': 6.8, 'deam_arousal': 1.5, 'mood': 'calm'},
        {'track_id': '5', 'track_name': 'Lost Love', 'artists': 'Heartbreak Hotel', 'genre': 'indie',
         'deam_valence': 3.2, 'deam_arousal': 3.5, 'mood': 'melancholy'},
        {'track_id': '6', 'track_name': 'Summer Vibes', 'artists': 'Happy Gang', 'genre': 'pop',
         'deam_valence': 8.0, 'deam_arousal': 7.5, 'mood': 'joyful'},
        {'track_id': '7', 'track_name': 'Study Session', 'artists': 'Lo-Fi Cafe', 'genre': 'lofi',
         'deam_valence': 5.5, 'deam_arousal': 3.2, 'mood': 'focused'},
        {'track_id': '8', 'track_name': 'Rage Mode', 'artists': 'Metal Gods', 'genre': 'metal',
         'deam_valence': 4.2, 'deam_arousal': 8.9, 'mood': 'intense'},
    ]

    # Initialize sequencer
    sequencer = PlaylistSequencer()
    sequencer.load_from_analyzed_data(mock_tracks)

    # Compare strategies
    print("\nComparing all 11 sequencing strategies...")
    results = sequencer.compare_strategies()

    # Sort by total score
    sorted_results = sorted(results.items(),
                          key=lambda x: x[1]['metrics'].total_score,
                          reverse=True)

    print("\n" + "=" * 70)
    print("RESULTS - RANKED BY TOTAL SCORE")
    print("=" * 70)

    for rank, (strategy, data) in enumerate(sorted_results, 1):
        metrics = data['metrics']
        marker = " [BEST]" if rank == 1 else " [TOP 3]" if rank <= 3 else ""
        print(f"\n{rank}. {strategy.upper()}{marker}")
        print(f"   Total Score:        {metrics.total_score:.3f}")
        print(f"   Smoothness:         {metrics.transition_smoothness:.3f} ↓")
        print(f"   Consistency:        {metrics.flow_variance:.3f} ↓")
        print(f"   Diversity:          {metrics.diversity_score:.3f} ↑")

    # Show improvement over baseline
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)

    original_score = results['original']['metrics'].total_score
    best_strategy, best_data = sorted_results[0]
    best_score = best_data['metrics'].total_score

    improvement = ((best_score - original_score) / original_score) * 100

    print(f"\nOriginal (Random):    {original_score:.3f}")
    print(f"Best ({best_strategy}):  {best_score:.3f}")
    print(f"Improvement:          +{improvement:.1f}%")

    print("\n" + "=" * 70)
    print("ALL 11 ALGORITHMS TESTED")
    print("=" * 70)


if __name__ == "__main__":
    main()
