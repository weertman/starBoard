# src/dl/verification_evaluation.py
"""
Evaluation module for measuring verification model performance against user annotations.

Compares verification confidence scores (P(same)) against past match verdicts
to compute both ranking metrics (Rank@K, MRR) and classification metrics (AUC, precision/recall).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("starBoard.dl.verification_evaluation")


@dataclass
class PairEvalResult:
    """Evaluation result for a single query-gallery pair."""
    query_id: str
    gallery_id: str
    verification_score: float  # P(same)
    rank: int  # 1-indexed rank when sorted by verification score (0 if not found)
    verdict: str  # "yes", "no", or "maybe"
    predicted_match: bool  # Whether model predicts match at optimal threshold
    correct: bool  # Whether prediction matches verdict
    found: bool  # Whether both query and gallery were in precomputed data


@dataclass
class VerificationSuggestion:
    """A suggested match for an unmatched query based on verification confidence."""
    query_id: str
    gallery_id: str
    verification_score: float  # P(same)
    rank: int  # 1-indexed rank


@dataclass
class VerificationEvalResults:
    """Complete evaluation results for a verification model."""
    model_key: str
    model_name: str
    
    # Counts
    total_yes_verdicts: int = 0
    total_no_verdicts: int = 0
    total_evaluated: int = 0
    
    # Ranking metrics (comparable to embedding eval)
    rank_at_1: float = 0.0
    rank_at_5: float = 0.0
    rank_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    
    # Classification metrics (unique to verification)
    auc_roc: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    optimal_threshold: float = 0.5
    
    # Score distributions
    match_scores: List[float] = field(default_factory=list)  # P(same) for true matches
    nonmatch_scores: List[float] = field(default_factory=list)  # P(same) for non-matches
    
    # Per-pair results
    pair_results: List[PairEvalResult] = field(default_factory=list)
    
    # Suggestions for unmatched queries
    suggestions: List[VerificationSuggestion] = field(default_factory=list)
    
    # Errors/warnings
    missing_queries: List[str] = field(default_factory=list)
    missing_galleries: List[str] = field(default_factory=list)


def run_verification_evaluation(
    model_key: Optional[str] = None,
    suggestion_threshold: float = 0.7,
    max_suggestions: int = 20
) -> VerificationEvalResults:
    """
    Run evaluation comparing verification scores to user verdicts.
    
    Args:
        model_key: Verification model to evaluate (uses active/default if None)
        suggestion_threshold: Minimum P(same) score for suggestions
        max_suggestions: Maximum number of match suggestions to return
        
    Returns:
        VerificationEvalResults with metrics and per-pair details
    """
    from src.data.past_matches import _latest_label_rows_all_pairs
    from src.dl.registry import DLRegistry, DEFAULT_VERIFICATION_KEY
    from src.dl.verification_lookup import VerificationLookup
    from src.ui.query_state_delegate import get_query_states_batch, QueryState
    
    # Determine model
    registry = DLRegistry.load()
    if model_key is None:
        model_key = registry.active_verification_model or DEFAULT_VERIFICATION_KEY
    
    if model_key not in registry.verification_models:
        raise ValueError(f"Verification model not found: {model_key}")
    
    model_entry = registry.verification_models[model_key]
    if not model_entry.precomputed:
        raise ValueError(f"Verification model not precomputed: {model_key}")
    
    results = VerificationEvalResults(
        model_key=model_key,
        model_name=model_entry.display_name
    )
    
    # Load verification lookup
    log.info("Loading verification lookup for %s...", model_key)
    verif_lookup = VerificationLookup(model_key)
    if not verif_lookup.load():
        raise ValueError(f"Failed to load verification data for {model_key}")
    
    # Collect all verdicts from past matches
    # Build maps: yes_matches, no_matches
    yes_matches: Dict[str, List[str]] = {}  # query_id -> [gallery_ids with yes verdict]
    no_matches: Dict[str, List[str]] = {}   # query_id -> [gallery_ids with no verdict]
    all_decided_queries = set()
    
    for row in _latest_label_rows_all_pairs():
        qid = row.get("query_id", "")
        gid = row.get("gallery_id", "")
        verdict = row.get("verdict", "").lower()
        
        if not qid or not gid:
            continue
        
        all_decided_queries.add(qid)
        
        if verdict == "yes":
            if qid not in yes_matches:
                yes_matches[qid] = []
            yes_matches[qid].append(gid)
        elif verdict == "no":
            if qid not in no_matches:
                no_matches[qid] = []
            no_matches[qid].append(gid)
    
    results.total_yes_verdicts = sum(len(gids) for gids in yes_matches.values())
    results.total_no_verdicts = sum(len(gids) for gids in no_matches.values())
    
    log.info("Found %d yes verdicts, %d no verdicts across %d queries",
             results.total_yes_verdicts, results.total_no_verdicts, len(all_decided_queries))
    
    # Collect scores for classification metrics
    match_scores = []
    nonmatch_scores = []
    ranks = []
    
    # Evaluate yes verdicts (true matches)
    for query_id, gallery_ids in yes_matches.items():
        if not verif_lookup.has_query(query_id):
            results.missing_queries.append(query_id)
            continue
        
        # Get ranked gallery for this query (sorted by verification score)
        all_scores = verif_lookup.get_scores_for_query(query_id)
        ranked = sorted(all_scores.items(), key=lambda x: -x[1])
        ranked_dict = {gid: (i + 1, score) for i, (gid, score) in enumerate(ranked)}
        
        for gallery_id in gallery_ids:
            if not verif_lookup.has_gallery(gallery_id):
                results.missing_galleries.append(gallery_id)
                continue
            
            if gallery_id in ranked_dict:
                rank, score = ranked_dict[gallery_id]
                match_scores.append(score)
                ranks.append(rank)
                
                results.pair_results.append(PairEvalResult(
                    query_id=query_id,
                    gallery_id=gallery_id,
                    verification_score=score,
                    rank=rank,
                    verdict="yes",
                    predicted_match=True,  # Will update after finding optimal threshold
                    correct=True,  # Will update after finding optimal threshold
                    found=True
                ))
            else:
                results.pair_results.append(PairEvalResult(
                    query_id=query_id,
                    gallery_id=gallery_id,
                    verification_score=0.0,
                    rank=0,
                    verdict="yes",
                    predicted_match=False,
                    correct=False,
                    found=False
                ))
    
    # Evaluate no verdicts (non-matches)
    for query_id, gallery_ids in no_matches.items():
        if not verif_lookup.has_query(query_id):
            continue  # Already logged in yes_matches loop
        
        all_scores = verif_lookup.get_scores_for_query(query_id)
        
        for gallery_id in gallery_ids:
            if not verif_lookup.has_gallery(gallery_id):
                continue
            
            score = all_scores.get(gallery_id, 0.0)
            nonmatch_scores.append(score)
            
            # Get rank for this non-match
            ranked = sorted(all_scores.items(), key=lambda x: -x[1])
            rank = next((i + 1 for i, (gid, _) in enumerate(ranked) if gid == gallery_id), 0)
            
            results.pair_results.append(PairEvalResult(
                query_id=query_id,
                gallery_id=gallery_id,
                verification_score=score,
                rank=rank,
                verdict="no",
                predicted_match=True,  # Will update after finding optimal threshold
                correct=False,  # Will update after finding optimal threshold
                found=True
            ))
    
    results.match_scores = match_scores
    results.nonmatch_scores = nonmatch_scores
    results.total_evaluated = len(match_scores) + len(nonmatch_scores)
    
    # Compute ranking metrics
    if ranks:
        ranks_arr = np.array(ranks)
        results.rank_at_1 = float(np.mean(ranks_arr <= 1))
        results.rank_at_5 = float(np.mean(ranks_arr <= 5))
        results.rank_at_10 = float(np.mean(ranks_arr <= 10))
        results.mrr = float(np.mean(1.0 / ranks_arr))
    
    # Compute classification metrics
    if match_scores and nonmatch_scores:
        results.auc_roc, results.optimal_threshold = _compute_auc_and_threshold(
            match_scores, nonmatch_scores
        )
        
        # Compute metrics at optimal threshold
        all_scores = match_scores + nonmatch_scores
        all_labels = [1] * len(match_scores) + [0] * len(nonmatch_scores)
        predictions = [1 if s >= results.optimal_threshold else 0 for s in all_scores]
        
        tp = sum(1 for p, l in zip(predictions, all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, all_labels) if p == 1 and l == 0)
        tn = sum(1 for p, l in zip(predictions, all_labels) if p == 0 and l == 0)
        fn = sum(1 for p, l in zip(predictions, all_labels) if p == 0 and l == 1)
        
        results.accuracy = (tp + tn) / len(all_labels) if all_labels else 0.0
        results.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        results.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results.f1_score = (2 * results.precision * results.recall / 
                          (results.precision + results.recall)) if (results.precision + results.recall) > 0 else 0.0
        
        # Update pair results with predictions at optimal threshold
        for pair in results.pair_results:
            pair.predicted_match = pair.verification_score >= results.optimal_threshold
            if pair.verdict == "yes":
                pair.correct = pair.predicted_match
            elif pair.verdict == "no":
                pair.correct = not pair.predicted_match
    
    # Generate match suggestions for unmatched queries
    log.info("Generating verification-based match suggestions...")
    
    all_query_ids = verif_lookup.get_query_ids()
    query_states = get_query_states_batch(all_query_ids)
    
    unmatched_queries = [
        qid for qid, state in query_states.items()
        if state in (QueryState.NOT_ATTEMPTED, QueryState.ATTEMPTED, QueryState.PINNED)
    ]
    
    suggestions = []
    for query_id in unmatched_queries:
        ranked = verif_lookup.get_ranked_gallery(query_id, top_k=3)
        for rank_idx, (gallery_id, score) in enumerate(ranked):
            if score >= suggestion_threshold:
                suggestions.append(VerificationSuggestion(
                    query_id=query_id,
                    gallery_id=gallery_id,
                    verification_score=score,
                    rank=rank_idx + 1
                ))
    
    # Sort by verification score and limit
    suggestions.sort(key=lambda x: -x.verification_score)
    results.suggestions = suggestions[:max_suggestions]
    
    log.info("Verification evaluation complete: AUC=%.3f, Rank@1=%.1f%%, %d suggestions",
             results.auc_roc, results.rank_at_1 * 100, len(results.suggestions))
    
    return results


def _compute_auc_and_threshold(
    match_scores: List[float],
    nonmatch_scores: List[float]
) -> Tuple[float, float]:
    """
    Compute AUC-ROC and find optimal threshold.
    
    Returns:
        (auc, optimal_threshold)
    """
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        
        all_scores = np.array(match_scores + nonmatch_scores)
        all_labels = np.array([1] * len(match_scores) + [0] * len(nonmatch_scores))
        
        auc = roc_auc_score(all_labels, all_scores)
        
        # Find optimal threshold (maximize Youden's J statistic = TPR - FPR)
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return float(auc), float(optimal_threshold)
        
    except ImportError:
        log.warning("sklearn not available, using simple AUC approximation")
        # Simple AUC approximation: probability that a random positive > random negative
        n_correct = sum(1 for m in match_scores for n in nonmatch_scores if m > n)
        n_total = len(match_scores) * len(nonmatch_scores)
        auc = n_correct / n_total if n_total > 0 else 0.5
        
        # Simple threshold: midpoint between mean match and mean nonmatch
        mean_match = np.mean(match_scores) if match_scores else 0.5
        mean_nonmatch = np.mean(nonmatch_scores) if nonmatch_scores else 0.5
        optimal_threshold = (mean_match + mean_nonmatch) / 2
        
        return auc, optimal_threshold


# ==================== Visualization Functions ====================

def create_confidence_distribution_figure(results: VerificationEvalResults):
    """
    Create a Plotly figure showing the distribution of verification scores
    for matches vs non-matches.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    if not results.match_scores and not results.nonmatch_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No evaluation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Verification Score Distribution")
        return fig
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{"type": "indicator"}, {"type": "indicator"}]],
        row_heights=[0.7, 0.3],
        subplot_titles=("P(same) Distribution: Matches vs Non-Matches", "", "")
    )
    
    # Histogram for matches (true positives should have high scores)
    if results.match_scores:
        fig.add_trace(
            go.Histogram(
                x=results.match_scores,
                name="True Matches",
                marker_color="rgba(27, 158, 119, 0.7)",
                nbinsx=20,
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Histogram for non-matches (should have low scores)
    if results.nonmatch_scores:
        fig.add_trace(
            go.Histogram(
                x=results.nonmatch_scores,
                name="Non-Matches",
                marker_color="rgba(217, 95, 2, 0.7)",
                nbinsx=20,
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Add vertical line at optimal threshold
    fig.add_vline(
        x=results.optimal_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {results.optimal_threshold:.2f}",
        row=1, col=1
    )
    
    # AUC gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=results.auc_roc * 100,
            title={'text': "AUC-ROC (%)"},
            gauge={
                'axis': {'range': [50, 100]},
                'bar': {'color': "#1b9e77"},
                'steps': [
                    {'range': [50, 70], 'color': "#fee08b"},
                    {'range': [70, 85], 'color': "#abdda4"},
                    {'range': [85, 100], 'color': "#1b9e77"}
                ]
            }
        ),
        row=2, col=1
    )
    
    # Accuracy gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=results.accuracy * 100,
            title={'text': "Accuracy (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1b9e77"},
                'steps': [
                    {'range': [0, 60], 'color': "#fee08b"},
                    {'range': [60, 80], 'color': "#abdda4"},
                    {'range': [80, 100], 'color': "#1b9e77"}
                ]
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f"Verification Evaluation: {results.model_name}",
        barmode='overlay',
        height=600,
        xaxis_title="P(same) Score",
        yaxis_title="Count",
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )
    
    # Add summary annotation
    mean_match = np.mean(results.match_scores) if results.match_scores else 0
    mean_nonmatch = np.mean(results.nonmatch_scores) if results.nonmatch_scores else 0
    summary = (
        f"Matches: n={len(results.match_scores)}, mean={mean_match:.3f} | "
        f"Non-matches: n={len(results.nonmatch_scores)}, mean={mean_nonmatch:.3f} | "
        f"Separation: {mean_match - mean_nonmatch:.3f}"
    )
    fig.add_annotation(
        text=summary,
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(size=11)
    )
    
    return fig


def create_verification_rank_figure(results: VerificationEvalResults):
    """
    Create a Plotly figure showing the rank distribution when sorted by verification score.
    Similar to embedding rank distribution for comparison.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Extract ranks for true matches
    ranks = [r.rank for r in results.pair_results if r.verdict == "yes" and r.found and r.rank > 0]
    
    if not ranks:
        fig = go.Figure()
        fig.add_annotation(
            text="No ranking data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Verification Rank Distribution")
        return fig
    
    # Create histogram bins
    max_rank = max(ranks)
    bins = [1, 2, 3, 4, 5, 10, 20, 50, max(100, max_rank + 1)]
    bin_labels = ["1", "2", "3", "4", "5", "6-10", "11-20", "21-50", "50+"]
    
    # Count ranks in each bin
    counts = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        if i < 5:  # Individual ranks 1-5
            count = sum(1 for r in ranks if r == low)
        else:
            count = sum(1 for r in ranks if low <= r < high)
        counts.append(count)
    
    # Truncate trailing zeros
    while counts and counts[-1] == 0:
        counts.pop()
        bin_labels.pop()
    
    # Colors: green for rank 1, gradient to red for higher ranks
    colors = ['#1b9e77', '#66c2a5', '#abdda4', '#e6f598', '#fee08b', 
              '#fdae61', '#f46d43', '#d53e4f', '#9e0142'][:len(counts)]
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{"type": "indicator"}, {"type": "indicator"}]],
        row_heights=[0.7, 0.3],
        subplot_titles=("Where Did Matches Rank by Verification Confidence?", "", "")
    )
    
    # Bar chart
    fig.add_trace(
        go.Bar(
            x=bin_labels,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='outside',
            name="Count"
        ),
        row=1, col=1
    )
    
    # Rank@1 gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=results.rank_at_1 * 100,
            title={'text': "Rank@1 (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1b9e77"},
                'steps': [
                    {'range': [0, 50], 'color': "#fee08b"},
                    {'range': [50, 80], 'color': "#abdda4"},
                    {'range': [80, 100], 'color': "#1b9e77"}
                ]
            }
        ),
        row=2, col=1
    )
    
    # MRR gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=results.mrr * 100,
            title={'text': "MRR (%)"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1b9e77"},
                'steps': [
                    {'range': [0, 50], 'color': "#fee08b"},
                    {'range': [50, 80], 'color': "#abdda4"},
                    {'range': [80, 100], 'color': "#1b9e77"}
                ]
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f"Verification Rank Distribution: {results.model_name}",
        showlegend=False,
        height=600,
        xaxis_title="Rank of Correct Match (by verification confidence)",
        yaxis_title="Number of Queries",
    )
    
    # Add summary annotation
    summary = (
        f"Total evaluated: {len(ranks)} | "
        f"Rank@1: {results.rank_at_1:.1%} | "
        f"Rank@5: {results.rank_at_5:.1%} | "
        f"MRR: {results.mrr:.3f}"
    )
    fig.add_annotation(
        text=summary,
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(size=12)
    )
    
    return fig


def create_roc_curve_figure(results: VerificationEvalResults):
    """
    Create a Plotly figure showing the ROC curve.
    """
    import plotly.graph_objects as go
    
    if not results.match_scores or not results.nonmatch_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for ROC curve",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="ROC Curve")
        return fig
    
    try:
        from sklearn.metrics import roc_curve
        
        all_scores = np.array(results.match_scores + results.nonmatch_scores)
        all_labels = np.array([1] * len(results.match_scores) + [0] * len(results.nonmatch_scores))
        
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        
        # Find optimal point
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {results.auc_roc:.3f})',
            line=dict(color='#1b9e77', width=2)
        ))
        
        # Diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random (AUC = 0.5)',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        # Optimal threshold point
        fig.add_trace(go.Scatter(
            x=[fpr[optimal_idx]],
            y=[tpr[optimal_idx]],
            mode='markers',
            name=f'Optimal (t={results.optimal_threshold:.2f})',
            marker=dict(color='red', size=12, symbol='star')
        ))
        
        fig.update_layout(
            title=f"ROC Curve: {results.model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=500,
            width=600,
            showlegend=True,
            legend=dict(x=0.6, y=0.2)
        )
        
        # Add annotation with metrics
        fig.add_annotation(
            text=(
                f"AUC: {results.auc_roc:.3f}<br>"
                f"Optimal threshold: {results.optimal_threshold:.2f}<br>"
                f"Precision: {results.precision:.1%}<br>"
                f"Recall: {results.recall:.1%}"
            ),
            xref="paper", yref="paper",
            x=0.95, y=0.05,
            showarrow=False,
            font=dict(size=11),
            align="right",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        return fig
        
    except ImportError:
        fig = go.Figure()
        fig.add_annotation(
            text="sklearn required for ROC curve",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="ROC Curve")
        return fig


def create_verification_per_pair_figure(results: VerificationEvalResults):
    """
    Create a Plotly figure showing per-pair evaluation results.
    """
    import plotly.graph_objects as go
    
    # Get pairs with verdicts
    pair_data = [r for r in results.pair_results if r.found]
    pair_data.sort(key=lambda x: -x.verification_score)
    
    if not pair_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No pair evaluation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Color-code by correctness
    def get_color(pair):
        if pair.correct:
            return "#1b9e77"  # Green for correct
        else:
            return "#d95f02"  # Orange/red for incorrect
    
    colors = [get_color(r) for r in pair_data]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Query</b>', '<b>Gallery</b>', '<b>P(same)</b>', '<b>Rank</b>', 
                   '<b>Verdict</b>', '<b>Predicted</b>', '<b>Correct</b>'],
            fill_color='#2d3436',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[
                [r.query_id for r in pair_data],
                [r.gallery_id for r in pair_data],
                [f"{r.verification_score:.3f}" for r in pair_data],
                [r.rank for r in pair_data],
                [r.verdict for r in pair_data],
                ["Match" if r.predicted_match else "No Match" for r in pair_data],
                ["✓" if r.correct else "✗" for r in pair_data]
            ],
            fill_color=[
                ['white'] * len(pair_data),
                ['white'] * len(pair_data),
                ['white'] * len(pair_data),
                ['white'] * len(pair_data),
                ['white'] * len(pair_data),
                ['white'] * len(pair_data),
                colors
            ],
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=f"Per-Pair Verification Results: {results.model_name}",
        height=max(400, 50 + 30 * min(len(pair_data), 20))
    )
    
    # Add summary
    correct_count = sum(1 for r in pair_data if r.correct)
    fig.add_annotation(
        text=f"Showing {len(pair_data)} pairs | Correct: {correct_count} ({correct_count/len(pair_data):.1%})",
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=11, color="#666")
    )
    
    return fig


def create_verification_suggestions_figure(results: VerificationEvalResults):
    """
    Create a Plotly figure showing high-confidence match suggestions.
    """
    import plotly.graph_objects as go
    
    if not results.suggestions:
        fig = go.Figure()
        fig.add_annotation(
            text="No high-confidence suggestions available\n(all queries may already be matched)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Verification-Based Match Suggestions")
        return fig
    
    # Color by confidence score
    def score_color(score):
        if score >= 0.9:
            return "#1b9e77"
        elif score >= 0.8:
            return "#66c2a5"
        elif score >= 0.7:
            return "#abdda4"
        else:
            return "#fee08b"
    
    colors = [score_color(s.verification_score) for s in results.suggestions]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Query</b>', '<b>Suggested Gallery</b>', '<b>P(same)</b>', '<b>Rank</b>'],
            fill_color='#2d3436',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[
                [s.query_id for s in results.suggestions],
                [s.gallery_id for s in results.suggestions],
                [f"{s.verification_score:.3f}" for s in results.suggestions],
                [s.rank for s in results.suggestions]
            ],
            fill_color=[
                ['white'] * len(results.suggestions),
                ['white'] * len(results.suggestions),
                colors,
                ['white'] * len(results.suggestions)
            ],
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=f"High-Confidence Match Suggestions: {results.model_name}",
        height=max(400, 50 + 30 * len(results.suggestions))
    )
    
    # Add explanation
    fig.add_annotation(
        text="Unmatched queries with high verification confidence (P(same) ≥ 0.7)",
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=11, color="#666")
    )
    
    return fig


# ==================== Persistence Functions ====================

def _get_verification_evaluation_path(model_key: str) -> Path:
    """Get the path for storing verification evaluation results."""
    from src.dl.registry import DLRegistry
    model_dir = DLRegistry.get_verification_model_data_dir(model_key)
    return model_dir / "verification_evaluation_results.json"


def save_verification_evaluation(results: VerificationEvalResults) -> bool:
    """
    Save verification evaluation results to disk.
    
    Args:
        results: The evaluation results to save
        
    Returns:
        True if saved successfully
    """
    try:
        path = _get_verification_evaluation_path(results.model_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build compact query-indexed data
        query_suggestions = {}
        for s in results.suggestions:
            if s.query_id not in query_suggestions or s.rank < query_suggestions[s.query_id]["rank"]:
                query_suggestions[s.query_id] = {
                    "gallery_id": s.gallery_id,
                    "score": s.verification_score,
                    "rank": s.rank
                }
        
        data = {
            "version": 1,
            "model_key": results.model_key,
            "model_name": results.model_name,
            "timestamp": datetime.now().isoformat(),
            "ranking_metrics": {
                "rank_at_1": results.rank_at_1,
                "rank_at_5": results.rank_at_5,
                "rank_at_10": results.rank_at_10,
                "mrr": results.mrr,
            },
            "classification_metrics": {
                "auc_roc": results.auc_roc,
                "accuracy": results.accuracy,
                "precision": results.precision,
                "recall": results.recall,
                "f1_score": results.f1_score,
                "optimal_threshold": results.optimal_threshold,
            },
            "counts": {
                "total_evaluated": results.total_evaluated,
                "total_yes_verdicts": results.total_yes_verdicts,
                "total_no_verdicts": results.total_no_verdicts,
            },
            "query_suggestions": query_suggestions,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        log.info("Saved verification evaluation results for %s", results.model_key)
        return True
        
    except Exception as e:
        log.error("Failed to save verification evaluation results: %s", e)
        return False


def load_verification_evaluation(model_key: str) -> Optional[Dict]:
    """
    Load stored verification evaluation results.
    
    Args:
        model_key: The verification model to load evaluation for
        
    Returns:
        Dict with evaluation data or None if not found
    """
    try:
        path = _get_verification_evaluation_path(model_key)
        
        if not path.exists():
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    except Exception as e:
        log.error("Failed to load verification evaluation for %s: %s", model_key, e)
        return None


def has_verification_evaluation(model_key: str) -> bool:
    """Check if verification evaluation results exist."""
    path = _get_verification_evaluation_path(model_key)
    return path.exists()


def get_optimal_threshold_for_active_model() -> Optional[float]:
    """
    Get the optimal threshold from the active verification model's evaluation.
    
    Returns:
        The optimal threshold if evaluation has been run, else None.
    """
    from src.dl.registry import DLRegistry
    
    try:
        registry = DLRegistry.load()
        if not registry.active_verification_model:
            return None
        
        eval_data = load_verification_evaluation(registry.active_verification_model)
        if eval_data and "classification_metrics" in eval_data:
            return eval_data["classification_metrics"].get("optimal_threshold")
        return None
    except Exception:
        return None


def colorize_verification_score(score: float, optimal_threshold: Optional[float] = None) -> Tuple[str, str]:
    """
    Get color and tooltip text for a verification score.
    
    Uses the optimal threshold if provided, otherwise falls back to fixed thresholds.
    
    Args:
        score: The P(same) verification score
        optimal_threshold: Computed optimal threshold from evaluation (or None)
        
    Returns:
        (color_hex, tooltip_text)
    """
    if optimal_threshold is not None:
        # Use threshold-relative coloring
        # Green: at or above threshold (predicted match)
        # Yellow: within 0.15 below threshold (uncertain)
        # Red: more than 0.15 below threshold (predicted non-match)
        margin = 0.15
        if score >= optimal_threshold:
            color = "#27ae60"  # green
        elif score >= optimal_threshold - margin:
            color = "#f39c12"  # yellow/amber
        else:
            color = "#e74c3c"  # red
        
        tooltip = (
            f"Verification model confidence: P(same) = {score:.3f}\n"
            f"Optimal threshold: {optimal_threshold:.2f} (from your past matches)\n"
            f"Above threshold: likely match | Near threshold: review | Below: unlikely"
        )
    else:
        # Fallback to fixed thresholds
        if score >= 0.7:
            color = "#27ae60"  # green
        elif score >= 0.4:
            color = "#f39c12"  # yellow/amber
        else:
            color = "#e74c3c"  # red
        
        tooltip = (
            f"Verification model confidence: P(same) = {score:.3f}\n"
            f"High (>0.7): likely match | Medium (0.4-0.7): review | Low (<0.4): unlikely\n"
            f"Run Verification Evaluation for threshold tuned to your data."
        )
    
    return color, tooltip