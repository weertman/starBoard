# src/dl/evaluation.py
"""
Evaluation module for measuring DL model performance against user annotations.

Compares model similarity rankings against past match verdicts (yes/no/maybe)
to compute retrieval metrics like Rank@K and Mean Reciprocal Rank (MRR).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("starBoard.dl.evaluation")


@dataclass
class QueryEvalResult:
    """Evaluation result for a single query."""
    query_id: str
    gallery_id: str  # The confirmed match (verdict=yes)
    rank: int  # 1-indexed rank of the match in similarity ranking (0 if not found)
    similarity_score: float
    found: bool  # Whether the gallery was in the precomputed data


@dataclass
class MatchSuggestion:
    """A suggested match for an unmatched query."""
    query_id: str
    gallery_id: str
    similarity_score: float
    rank: int  # 1-indexed rank


@dataclass 
class EvaluationResults:
    """Complete evaluation results for a model."""
    model_key: str
    model_name: str
    
    # Counts
    total_yes_verdicts: int = 0
    total_evaluated: int = 0  # Verdicts where both query and gallery were in precomputed data
    
    # Metrics
    rank_at_1: float = 0.0
    rank_at_5: float = 0.0
    rank_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    
    # Per-query results
    query_results: List[QueryEvalResult] = field(default_factory=list)
    
    # Per-gallery aggregated results: {gallery_id: {"hits": n, "total": n, "avg_rank": f}}
    gallery_stats: Dict[str, Dict] = field(default_factory=dict)
    
    # Match suggestions for unmatched queries
    suggestions: List[MatchSuggestion] = field(default_factory=list)
    
    # Errors/warnings
    missing_queries: List[str] = field(default_factory=list)
    missing_galleries: List[str] = field(default_factory=list)


def run_evaluation(
    model_key: Optional[str] = None,
    suggestion_threshold: float = 0.5,
    max_suggestions: int = 20
) -> EvaluationResults:
    """
    Run evaluation comparing model rankings to user verdicts.
    
    Args:
        model_key: Model to evaluate (uses active/default if None)
        suggestion_threshold: Minimum similarity score for suggestions
        max_suggestions: Maximum number of match suggestions to return
        
    Returns:
        EvaluationResults with metrics and per-query details
    """
    from src.data.past_matches import build_past_matches_dataset
    from src.dl.registry import DLRegistry, DEFAULT_MODEL_KEY
    from src.dl.similarity_lookup import SimilarityLookup
    from src.ui.query_state_delegate import get_query_states_batch, QueryState
    
    # Determine model
    registry = DLRegistry.load()
    if model_key is None:
        model_key = registry.active_model or DEFAULT_MODEL_KEY
    
    if model_key not in registry.models:
        raise ValueError(f"Model not found: {model_key}")
    
    model_entry = registry.models[model_key]
    if not model_entry.precomputed:
        raise ValueError(f"Model not precomputed: {model_key}")
    
    results = EvaluationResults(
        model_key=model_key,
        model_name=model_entry.display_name
    )
    
    # Load past matches
    log.info("Loading past matches dataset...")
    pm_dataset = build_past_matches_dataset()
    
    # Load similarity lookup
    log.info("Loading similarity lookup for %s...", model_key)
    sim_lookup = SimilarityLookup(model_key)
    if not sim_lookup.load():
        raise ValueError(f"Failed to load similarity data for {model_key}")
    
    # Extract yes verdicts from past matches
    # per_query_counts has {query_id: Counter(yes=n, maybe=n, no=n)}
    # We need the actual gallery_id for each yes verdict
    # We'll need to iterate through the records directly
    
    # Build a map of query_id -> list of (gallery_id with yes verdict)
    yes_matches: Dict[str, List[str]] = {}
    all_decided_queries = set()
    
    # Access the internal data - we need to rebuild from labels
    from src.data.past_matches import _latest_label_rows_all_pairs
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
    
    results.total_yes_verdicts = sum(len(gids) for gids in yes_matches.values())
    
    # Evaluate each yes verdict
    log.info("Evaluating %d yes verdicts across %d queries...", 
             results.total_yes_verdicts, len(yes_matches))
    
    ranks = []
    gallery_hits: Dict[str, List[int]] = {}  # gallery_id -> list of ranks achieved
    
    for query_id, gallery_ids in yes_matches.items():
        if not sim_lookup.has_query(query_id):
            results.missing_queries.append(query_id)
            continue
        
        # Get ranked gallery for this query
        ranked = sim_lookup.get_ranked_gallery(query_id)
        ranked_dict = {gid: (i + 1, score) for i, (gid, score) in enumerate(ranked)}
        
        for gallery_id in gallery_ids:
            if not sim_lookup.has_gallery(gallery_id):
                results.missing_galleries.append(gallery_id)
                result = QueryEvalResult(
                    query_id=query_id,
                    gallery_id=gallery_id,
                    rank=0,
                    similarity_score=0.0,
                    found=False
                )
            elif gallery_id in ranked_dict:
                rank, score = ranked_dict[gallery_id]
                result = QueryEvalResult(
                    query_id=query_id,
                    gallery_id=gallery_id,
                    rank=rank,
                    similarity_score=score,
                    found=True
                )
                ranks.append(rank)
                
                # Track per-gallery
                if gallery_id not in gallery_hits:
                    gallery_hits[gallery_id] = []
                gallery_hits[gallery_id].append(rank)
            else:
                result = QueryEvalResult(
                    query_id=query_id,
                    gallery_id=gallery_id,
                    rank=0,
                    similarity_score=0.0,
                    found=False
                )
            
            results.query_results.append(result)
    
    results.total_evaluated = len(ranks)
    
    # Compute metrics
    if ranks:
        ranks_arr = np.array(ranks)
        results.rank_at_1 = float(np.mean(ranks_arr <= 1))
        results.rank_at_5 = float(np.mean(ranks_arr <= 5))
        results.rank_at_10 = float(np.mean(ranks_arr <= 10))
        results.mrr = float(np.mean(1.0 / ranks_arr))
    
    # Aggregate per-gallery stats
    for gid, gid_ranks in gallery_hits.items():
        results.gallery_stats[gid] = {
            "hits": len(gid_ranks),
            "avg_rank": float(np.mean(gid_ranks)),
            "best_rank": min(gid_ranks),
            "worst_rank": max(gid_ranks)
        }
    
    # Generate match suggestions for unmatched queries
    log.info("Generating match suggestions...")
    
    # Get query states to find unmatched queries
    all_query_ids = sim_lookup.get_query_ids()
    query_states = get_query_states_batch(all_query_ids)
    
    unmatched_queries = [
        qid for qid, state in query_states.items()
        if state in (QueryState.NOT_ATTEMPTED, QueryState.ATTEMPTED, QueryState.PINNED)
    ]
    
    suggestions = []
    for query_id in unmatched_queries:
        ranked = sim_lookup.get_ranked_gallery(query_id, top_k=3)
        for rank_idx, (gallery_id, score) in enumerate(ranked):
            if score >= suggestion_threshold:
                suggestions.append(MatchSuggestion(
                    query_id=query_id,
                    gallery_id=gallery_id,
                    similarity_score=score,
                    rank=rank_idx + 1
                ))
    
    # Sort by similarity score and limit
    suggestions.sort(key=lambda x: -x.similarity_score)
    results.suggestions = suggestions[:max_suggestions]
    
    log.info("Evaluation complete: Rank@1=%.1f%%, MRR=%.3f, %d suggestions",
             results.rank_at_1 * 100, results.mrr, len(results.suggestions))
    
    return results


def create_rank_distribution_figure(results: EvaluationResults):
    """
    Create a Plotly figure showing the distribution of ranks.
    
    Returns a Plotly figure object.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Extract ranks
    ranks = [r.rank for r in results.query_results if r.found and r.rank > 0]
    
    if not ranks:
        fig = go.Figure()
        fig.add_annotation(
            text="No evaluation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Rank Distribution")
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
        subplot_titles=("Where Did Matches Rank?", "", "")
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
    
    # Metric indicators
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
        title=f"Model Evaluation: {results.model_name}",
        showlegend=False,
        height=600,
        xaxis_title="Rank of Correct Match",
        yaxis_title="Number of Queries",
    )
    
    # Add summary annotation
    summary = (
        f"Total evaluated: {results.total_evaluated} | "
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


def create_per_query_figure(results: EvaluationResults):
    """
    Create a Plotly figure showing per-query evaluation results.
    
    Returns a Plotly figure object with an interactive table.
    """
    import plotly.graph_objects as go
    
    # Prepare data
    query_data = [r for r in results.query_results if r.found]
    query_data.sort(key=lambda x: x.rank)
    
    if not query_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No evaluation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Color-code by rank
    def rank_color(rank):
        if rank == 1:
            return "#1b9e77"
        elif rank <= 5:
            return "#66c2a5"
        elif rank <= 10:
            return "#fee08b"
        else:
            return "#f46d43"
    
    colors = [rank_color(r.rank) for r in query_data]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Query</b>', '<b>Matched Gallery</b>', '<b>Rank</b>', '<b>Similarity</b>'],
            fill_color='#2d3436',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[
                [r.query_id for r in query_data],
                [r.gallery_id for r in query_data],
                [r.rank for r in query_data],
                [f"{r.similarity_score:.3f}" for r in query_data]
            ],
            fill_color=[['white'] * len(query_data), ['white'] * len(query_data), 
                       colors, ['white'] * len(query_data)],
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=f"Per-Query Results: {results.model_name}",
        height=max(400, 50 + 30 * len(query_data))
    )
    
    return fig


def create_gallery_stats_figure(results: EvaluationResults):
    """
    Create a Plotly figure showing per-gallery performance.
    
    Returns a Plotly figure object.
    """
    import plotly.graph_objects as go
    
    if not results.gallery_stats:
        fig = go.Figure()
        fig.add_annotation(
            text="No gallery statistics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Sort by average rank
    sorted_galleries = sorted(
        results.gallery_stats.items(),
        key=lambda x: x[1]['avg_rank']
    )
    
    gallery_ids = [g[0] for g in sorted_galleries]
    avg_ranks = [g[1]['avg_rank'] for g in sorted_galleries]
    hits = [g[1]['hits'] for g in sorted_galleries]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=gallery_ids,
        y=avg_ranks,
        marker_color=['#1b9e77' if r <= 5 else '#fee08b' if r <= 10 else '#f46d43' 
                      for r in avg_ranks],
        text=[f"n={h}" for h in hits],
        textposition='outside',
        name="Avg Rank"
    ))
    
    fig.update_layout(
        title=f"Gallery Performance: {results.model_name}",
        xaxis_title="Gallery Individual",
        yaxis_title="Average Rank (lower is better)",
        height=max(400, 50 + 20 * len(gallery_ids)),
        xaxis={'categoryorder': 'array', 'categoryarray': gallery_ids}
    )
    
    return fig


def create_suggestions_figure(results: EvaluationResults):
    """
    Create a Plotly figure showing match suggestions.
    
    Returns a Plotly figure object with an interactive table.
    """
    import plotly.graph_objects as go
    
    if not results.suggestions:
        fig = go.Figure()
        fig.add_annotation(
            text="No match suggestions available\n(all high-similarity queries already matched)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Match Suggestions")
        return fig
    
    # Color by similarity score
    def score_color(score):
        if score >= 0.8:
            return "#1b9e77"
        elif score >= 0.6:
            return "#66c2a5"
        else:
            return "#fee08b"
    
    colors = [score_color(s.similarity_score) for s in results.suggestions]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Query</b>', '<b>Suggested Gallery</b>', '<b>Similarity</b>', '<b>Rank</b>'],
            fill_color='#2d3436',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[
                [s.query_id for s in results.suggestions],
                [s.gallery_id for s in results.suggestions],
                [f"{s.similarity_score:.3f}" for s in results.suggestions],
                [s.rank for s in results.suggestions]
            ],
            fill_color=[['white'] * len(results.suggestions), 
                       ['white'] * len(results.suggestions),
                       colors,
                       ['white'] * len(results.suggestions)],
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=f"Match Suggestions for Unmatched Queries: {results.model_name}",
        height=max(400, 50 + 30 * len(results.suggestions))
    )
    
    # Add explanation
    fig.add_annotation(
        text="Queries without a 'yes' verdict that have high similarity to gallery individuals",
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=11, color="#666")
    )
    
    return fig


# ==================== Persistence Functions ====================

def _get_evaluation_path(model_key: str) -> Path:
    """Get the path for storing evaluation results for a model."""
    from src.dl.registry import DLRegistry
    model_dir = DLRegistry.get_model_data_dir(model_key)
    return model_dir / "evaluation_results.json"


def save_evaluation(results: EvaluationResults) -> bool:
    """
    Save evaluation results to disk for later use.
    
    Saves a compact format optimized for quick lookup:
    - Per-query top match info for sorting
    - Timestamp of evaluation
    - Model metrics
    
    Args:
        results: The evaluation results to save
        
    Returns:
        True if saved successfully
    """
    try:
        path = _get_evaluation_path(results.model_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build compact query-indexed data for quick lookup
        query_suggestions = {}
        for s in results.suggestions:
            # Only keep the top suggestion per query (rank=1)
            if s.query_id not in query_suggestions or s.rank < query_suggestions[s.query_id]["rank"]:
                query_suggestions[s.query_id] = {
                    "gallery_id": s.gallery_id,
                    "score": s.similarity_score,
                    "rank": s.rank
                }
        
        data = {
            "version": 1,
            "model_key": results.model_key,
            "model_name": results.model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "rank_at_1": results.rank_at_1,
                "rank_at_5": results.rank_at_5,
                "rank_at_10": results.rank_at_10,
                "mrr": results.mrr,
                "total_evaluated": results.total_evaluated,
                "total_yes_verdicts": results.total_yes_verdicts,
            },
            "query_suggestions": query_suggestions,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        log.info("Saved evaluation results for %s: %d suggestions", 
                 results.model_key, len(query_suggestions))
        return True
        
    except Exception as e:
        log.error("Failed to save evaluation results: %s", e)
        return False


@dataclass
class StoredEvaluation:
    """Loaded evaluation data for quick access."""
    model_key: str
    model_name: str
    timestamp: str
    metrics: Dict[str, float]
    # {query_id: {"gallery_id": str, "score": float, "rank": int}}
    query_suggestions: Dict[str, Dict]
    
    def get_sorted_queries(self) -> List[Tuple[str, str, float]]:
        """
        Get queries sorted by similarity score (highest first).
        
        Returns:
            List of (query_id, top_gallery_id, score) tuples
        """
        items = [
            (qid, data["gallery_id"], data["score"])
            for qid, data in self.query_suggestions.items()
        ]
        return sorted(items, key=lambda x: -x[2])
    
    def get_top_match(self, query_id: str) -> Optional[Tuple[str, float]]:
        """
        Get the top suggested match for a query.
        
        Returns:
            (gallery_id, score) or None if no suggestion
        """
        if query_id in self.query_suggestions:
            data = self.query_suggestions[query_id]
            return (data["gallery_id"], data["score"])
        return None
    
    def remove_query(self, query_id: str) -> bool:
        """
        Remove a query from the suggestions (after it's been matched).
        
        Returns:
            True if the query was removed
        """
        if query_id in self.query_suggestions:
            del self.query_suggestions[query_id]
            return True
        return False


def load_evaluation(model_key: str) -> Optional[StoredEvaluation]:
    """
    Load stored evaluation results for a model.
    
    Args:
        model_key: The model to load evaluation for
        
    Returns:
        StoredEvaluation or None if not found/invalid
    """
    try:
        path = _get_evaluation_path(model_key)
        
        if not path.exists():
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return StoredEvaluation(
            model_key=data["model_key"],
            model_name=data["model_name"],
            timestamp=data["timestamp"],
            metrics=data["metrics"],
            query_suggestions=data["query_suggestions"],
        )
        
    except Exception as e:
        log.error("Failed to load evaluation results for %s: %s", model_key, e)
        return None


def has_evaluation(model_key: str) -> bool:
    """Check if evaluation results exist for a model."""
    path = _get_evaluation_path(model_key)
    return path.exists()


def update_stored_evaluation(model_key: str, matched_query_id: str) -> bool:
    """
    Update stored evaluation after a query has been matched.
    
    Removes the matched query from the suggestions.
    
    Args:
        model_key: The model's evaluation to update
        matched_query_id: The query that was just matched
        
    Returns:
        True if updated successfully
    """
    try:
        path = _get_evaluation_path(model_key)
        
        if not path.exists():
            return False
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if matched_query_id in data.get("query_suggestions", {}):
            del data["query_suggestions"][matched_query_id]
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            log.debug("Removed %s from evaluation suggestions", matched_query_id)
            return True
        
        return False
        
    except Exception as e:
        log.error("Failed to update evaluation: %s", e)
        return False

