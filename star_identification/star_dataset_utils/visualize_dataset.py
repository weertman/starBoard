#!/usr/bin/env python
"""
Dataset Visualization for Star Re-ID Dataset

Generates comprehensive visualizations of the star_dataset structure,
focusing on dataset sources and train/test split distributions.

Outputs are saved to: star_dataset_utils/outputs/

Usage:
    python star_dataset_utils/visualize_dataset.py
    python star_dataset_utils/visualize_dataset.py --metadata path/to/metadata.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette - earthy/ocean tones appropriate for marine biology
COLORS = {
    'train': '#2E86AB',      # Ocean blue
    'test': '#E94F37',       # Coral red
    'negative_only': '#F39C12',  # Amber/warning
    'primary': '#1A535C',    # Deep teal
    'secondary': '#4ECDC4',  # Light teal
    'accent': '#FF6B6B',     # Soft red
    'background': '#F7F9FC', # Light gray-blue
}

# Extended palette for dataset sources
SOURCE_PALETTE = sns.color_palette("husl", 21)


class DatasetVisualizer:
    """Visualization toolkit for star_dataset analysis."""
    
    def __init__(
        self,
        metadata_path: Path,
        output_dir: Path,
        dataset_root: Optional[Path] = None,
    ):
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.dataset_root = dataset_root or self.metadata_path.parent
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.df = self._load_metadata()
        
        print(f"Loaded {len(self.df)} images from {self.metadata_path.name}")
        print(f"Output directory: {self.output_dir}")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load and validate metadata CSV."""
        df = pd.read_csv(self.metadata_path)
        
        # Determine which metadata format we have
        if 'negative_only' in df.columns:
            # metadata_temporal.csv format
            required_cols = ['identity', 'path', 'dataset', 'split']
        else:
            # metadata.csv format - need to extract dataset from identity
            required_cols = ['identity', 'path', 'split']
            if 'dataset' not in df.columns:
                # Extract dataset from identity (format: DATASET__individual)
                df['dataset'] = df['identity'].apply(
                    lambda x: x.split('__')[0] if '__' in str(x) else 'unknown'
                )
        
        # Validate required columns
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Parse dates if available
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Ensure negative_only column exists
        if 'negative_only' not in df.columns:
            df['negative_only'] = False
        
        return df
    
    def generate_all_visualizations(self):
        """Generate all dataset visualizations."""
        print("\n" + "="*60)
        print("GENERATING DATASET VISUALIZATIONS")
        print("="*60)
        
        # 1. Dataset source overview
        self.plot_source_composition()
        
        # 2. Train/test split by source
        self.plot_split_by_source()
        
        # 3. Detailed split breakdown (stacked bars)
        self.plot_split_breakdown_stacked()
        
        # 4. Identity distribution per source
        self.plot_identities_per_source()
        
        # 5. Images per identity distribution
        self.plot_images_per_identity()
        
        # 6. Negative-only analysis
        self.plot_negative_only_analysis()
        
        # 7. Comprehensive dashboard
        self.plot_dashboard()
        
        # 8. Generate summary statistics
        self.save_summary_statistics()
        
        print("\n" + "="*60)
        print(f"All visualizations saved to: {self.output_dir}")
        print("="*60)
    
    def plot_source_composition(self):
        """Plot overall dataset source composition."""
        print("\n[1/8] Generating source composition plot...")
        
        # Count images per source
        source_counts = self.df['dataset'].value_counts().sort_values(ascending=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # Left: Horizontal bar chart
        ax1 = axes[0]
        colors = [SOURCE_PALETTE[i % len(SOURCE_PALETTE)] for i in range(len(source_counts))]
        bars = ax1.barh(range(len(source_counts)), source_counts.values, color=colors)
        ax1.set_yticks(range(len(source_counts)))
        ax1.set_yticklabels(source_counts.index, fontsize=8)
        ax1.set_xlabel('Number of Images')
        ax1.set_title('Images per Dataset Source')
        
        # Add value labels
        for bar, val in zip(bars, source_counts.values):
            ax1.text(val + 20, bar.get_y() + bar.get_height()/2, 
                    f'{val:,}', va='center', fontsize=8)
        
        # Right: Pie chart for top sources
        ax2 = axes[1]
        top_n = 8
        top_sources = source_counts.tail(top_n)
        other_count = source_counts.head(len(source_counts) - top_n).sum()
        
        if other_count > 0:
            pie_data = pd.concat([top_sources, pd.Series({'Other': other_count})])
        else:
            pie_data = top_sources
        
        wedges, texts, autotexts = ax2.pie(
            pie_data.values,
            labels=None,
            autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
            colors=SOURCE_PALETTE[:len(pie_data)],
            explode=[0.02] * len(pie_data),
            startangle=90,
        )
        
        # Legend
        ax2.legend(
            wedges, 
            [f'{name} ({val:,})' for name, val in zip(pie_data.index, pie_data.values)],
            title='Dataset Sources',
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
        )
        ax2.set_title(f'Top {top_n} Sources (+ Other)')
        
        plt.suptitle('Star Dataset: Source Composition', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / '01_source_composition.png'
        plt.savefig(save_path)
        plt.close()
        print(f"   Saved: {save_path.name}")
    
    def plot_split_by_source(self):
        """Plot train/test split distribution by dataset source."""
        print("\n[2/8] Generating train/test split by source...")
        
        # Create split counts per source
        split_counts = self.df.groupby(['dataset', 'split']).size().unstack(fill_value=0)
        
        # Ensure both columns exist
        for col in ['train', 'test']:
            if col not in split_counts.columns:
                split_counts[col] = 0
        
        # Sort by total images
        split_counts['total'] = split_counts['train'] + split_counts['test']
        split_counts = split_counts.sort_values('total', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        y_pos = np.arange(len(split_counts))
        bar_height = 0.8
        
        # Plot train bars
        train_bars = ax.barh(
            y_pos, split_counts['train'], 
            height=bar_height, 
            label='Train (Gallery)',
            color=COLORS['train'],
            alpha=0.9,
        )
        
        # Plot test bars (stacked)
        test_bars = ax.barh(
            y_pos, split_counts['test'],
            height=bar_height,
            left=split_counts['train'],
            label='Test (Query)',
            color=COLORS['test'],
            alpha=0.9,
        )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(split_counts.index, fontsize=8)
        ax.set_xlabel('Number of Images')
        ax.set_title('Train/Test Split by Dataset Source', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        
        # Add percentage labels
        for i, (idx, row) in enumerate(split_counts.iterrows()):
            total = row['total']
            if total > 0:
                train_pct = row['train'] / total * 100
                # Add total count at end of bar
                ax.text(
                    total + 20, i,
                    f'{int(total):,} ({train_pct:.0f}% train)',
                    va='center', fontsize=7, color='gray'
                )
        
        plt.tight_layout()
        
        save_path = self.output_dir / '02_split_by_source.png'
        plt.savefig(save_path)
        plt.close()
        print(f"   Saved: {save_path.name}")
    
    def plot_split_breakdown_stacked(self):
        """Plot detailed split breakdown with negative_only distinction."""
        print("\n[3/8] Generating detailed split breakdown...")
        
        # Create detailed categorization
        def categorize(row):
            if row['split'] == 'test':
                return 'test'
            elif row.get('negative_only', False):
                return 'train_negative_only'
            else:
                return 'train_evaluable'
        
        self.df['category'] = self.df.apply(categorize, axis=1)
        
        # Count per source
        cat_counts = self.df.groupby(['dataset', 'category']).size().unstack(fill_value=0)
        
        # Ensure all columns exist
        for col in ['train_evaluable', 'train_negative_only', 'test']:
            if col not in cat_counts.columns:
                cat_counts[col] = 0
        
        # Sort by total
        cat_counts['total'] = cat_counts.sum(axis=1)
        cat_counts = cat_counts.sort_values('total', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        y_pos = np.arange(len(cat_counts))
        bar_height = 0.8
        
        # Stacked bars
        ax.barh(
            y_pos, cat_counts['train_evaluable'],
            height=bar_height,
            label='Train (Evaluable)',
            color=COLORS['train'],
            alpha=0.9,
        )
        ax.barh(
            y_pos, cat_counts['train_negative_only'],
            height=bar_height,
            left=cat_counts['train_evaluable'],
            label='Train (Negative-Only)',
            color=COLORS['negative_only'],
            alpha=0.9,
        )
        ax.barh(
            y_pos, cat_counts['test'],
            height=bar_height,
            left=cat_counts['train_evaluable'] + cat_counts['train_negative_only'],
            label='Test (Query)',
            color=COLORS['test'],
            alpha=0.9,
        )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cat_counts.index, fontsize=8)
        ax.set_xlabel('Number of Images')
        ax.set_title(
            'Detailed Split Breakdown by Source\n'
            '(Negative-Only = single-outing identities, excluded from evaluation)',
            fontsize=11, fontweight='bold'
        )
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        save_path = self.output_dir / '03_split_breakdown_detailed.png'
        plt.savefig(save_path)
        plt.close()
        
        # Clean up temporary column
        self.df.drop('category', axis=1, inplace=True)
        print(f"   Saved: {save_path.name}")
    
    def plot_identities_per_source(self):
        """Plot number of unique identities per source."""
        print("\n[4/8] Generating identities per source plot...")
        
        # Count unique identities per source
        id_counts = self.df.groupby('dataset')['identity'].nunique().sort_values(ascending=True)
        
        # Also get images per identity stats
        img_per_id = self.df.groupby(['dataset', 'identity']).size().reset_index(name='count')
        id_stats = img_per_id.groupby('dataset')['count'].agg(['mean', 'std', 'min', 'max'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # Left: Identity counts
        ax1 = axes[0]
        colors = [SOURCE_PALETTE[i % len(SOURCE_PALETTE)] for i in range(len(id_counts))]
        bars = ax1.barh(range(len(id_counts)), id_counts.values, color=colors)
        ax1.set_yticks(range(len(id_counts)))
        ax1.set_yticklabels(id_counts.index, fontsize=8)
        ax1.set_xlabel('Number of Unique Identities')
        ax1.set_title('Unique Identities per Dataset Source')
        
        for bar, val in zip(bars, id_counts.values):
            ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val}', va='center', fontsize=8)
        
        # Right: Average images per identity
        ax2 = axes[1]
        id_stats_sorted = id_stats.loc[id_counts.index]
        
        ax2.barh(
            range(len(id_stats_sorted)), 
            id_stats_sorted['mean'],
            xerr=id_stats_sorted['std'].fillna(0),
            color=colors,
            capsize=3,
            alpha=0.8,
        )
        ax2.set_yticks(range(len(id_stats_sorted)))
        ax2.set_yticklabels(id_stats_sorted.index, fontsize=8)
        ax2.set_xlabel('Average Images per Identity (± std)')
        ax2.set_title('Images per Identity by Source')
        
        plt.suptitle('Identity Distribution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / '04_identities_per_source.png'
        plt.savefig(save_path)
        plt.close()
        print(f"   Saved: {save_path.name}")
    
    def plot_images_per_identity(self):
        """Plot distribution of images per identity (class imbalance analysis)."""
        print("\n[5/8] Generating images per identity distribution...")
        
        # Calculate images per identity
        img_per_id = self.df.groupby('identity').size()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Top-left: Histogram
        ax1 = axes[0, 0]
        ax1.hist(img_per_id.values, bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='white')
        ax1.axvline(img_per_id.mean(), color=COLORS['accent'], linestyle='--', 
                   label=f'Mean: {img_per_id.mean():.1f}')
        ax1.axvline(img_per_id.median(), color=COLORS['secondary'], linestyle='--',
                   label=f'Median: {img_per_id.median():.1f}')
        ax1.set_xlabel('Images per Identity')
        ax1.set_ylabel('Number of Identities')
        ax1.set_title('Distribution of Images per Identity')
        ax1.legend()
        
        # Top-right: Log-scale histogram (long-tail visualization)
        ax2 = axes[0, 1]
        ax2.hist(img_per_id.values, bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='white')
        ax2.set_yscale('log')
        ax2.set_xlabel('Images per Identity')
        ax2.set_ylabel('Number of Identities (log scale)')
        ax2.set_title('Long-Tail Distribution (Log Scale)')
        
        # Bottom-left: Box plot by source (top 10 sources)
        ax3 = axes[1, 0]
        top_sources = self.df['dataset'].value_counts().head(10).index.tolist()
        plot_data = []
        for source in top_sources:
            source_ids = self.df[self.df['dataset'] == source].groupby('identity').size()
            for val in source_ids.values:
                plot_data.append({'source': source[:15], 'images': val})
        
        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(data=plot_df, x='images', y='source', ax=ax3, palette='husl')
        ax3.set_xlabel('Images per Identity')
        ax3.set_ylabel('Dataset Source')
        ax3.set_title('Images per Identity by Source (Top 10)')
        
        # Bottom-right: CDF
        ax4 = axes[1, 1]
        sorted_counts = np.sort(img_per_id.values)
        cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        ax4.plot(sorted_counts, cdf, color=COLORS['primary'], linewidth=2)
        ax4.fill_between(sorted_counts, cdf, alpha=0.3, color=COLORS['primary'])
        ax4.set_xlabel('Images per Identity')
        ax4.set_ylabel('Cumulative Proportion of Identities')
        ax4.set_title('Cumulative Distribution')
        ax4.grid(True, alpha=0.3)
        
        # Add reference lines
        for pct in [0.5, 0.8, 0.9]:
            idx = np.searchsorted(cdf, pct)
            if idx < len(sorted_counts):
                ax4.axhline(pct, color='gray', linestyle=':', alpha=0.5)
                ax4.axvline(sorted_counts[idx], color='gray', linestyle=':', alpha=0.5)
                ax4.text(sorted_counts[idx], pct, f' {pct*100:.0f}%: ≤{sorted_counts[idx]} imgs',
                        fontsize=8, va='bottom')
        
        plt.suptitle('Class Imbalance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / '05_images_per_identity.png'
        plt.savefig(save_path)
        plt.close()
        print(f"   Saved: {save_path.name}")
    
    def plot_negative_only_analysis(self):
        """Analyze negative-only (single-outing) identities."""
        print("\n[6/8] Generating negative-only analysis...")
        
        if 'negative_only' not in self.df.columns or not self.df['negative_only'].any():
            print("   Skipping: No negative_only data available")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Left: Overall proportion
        ax1 = axes[0]
        neg_only_ids = self.df[self.df['negative_only'] == True]['identity'].nunique()
        eval_ids = self.df[self.df['negative_only'] == False]['identity'].nunique()
        
        ax1.pie(
            [eval_ids, neg_only_ids],
            labels=['Evaluable\n(Multi-Outing)', 'Negative-Only\n(Single-Outing)'],
            autopct='%1.1f%%',
            colors=[COLORS['train'], COLORS['negative_only']],
            explode=[0, 0.05],
            startangle=90,
        )
        ax1.set_title(f'Identity Types\n({eval_ids} evaluable, {neg_only_ids} negative-only)')
        
        # Middle: By source
        ax2 = axes[1]
        neg_by_source = self.df.groupby('dataset').apply(
            lambda x: pd.Series({
                'evaluable': x[x['negative_only'] == False]['identity'].nunique(),
                'negative_only': x[x['negative_only'] == True]['identity'].nunique(),
            })
        )
        neg_by_source = neg_by_source.sort_values('evaluable', ascending=True).tail(15)
        
        y_pos = np.arange(len(neg_by_source))
        ax2.barh(y_pos, neg_by_source['evaluable'], label='Evaluable', 
                color=COLORS['train'], alpha=0.9)
        ax2.barh(y_pos, neg_by_source['negative_only'], 
                left=neg_by_source['evaluable'],
                label='Negative-Only', color=COLORS['negative_only'], alpha=0.9)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(neg_by_source.index, fontsize=8)
        ax2.set_xlabel('Number of Identities')
        ax2.set_title('Identity Types by Source (Top 15)')
        ax2.legend(loc='lower right')
        
        # Right: Images in each category
        ax3 = axes[2]
        neg_only_imgs = self.df[self.df['negative_only'] == True].shape[0]
        eval_imgs = self.df[self.df['negative_only'] == False].shape[0]
        
        bars = ax3.bar(
            ['Evaluable\nIdentities', 'Negative-Only\nIdentities'],
            [eval_imgs, neg_only_imgs],
            color=[COLORS['train'], COLORS['negative_only']],
            alpha=0.9,
        )
        ax3.set_ylabel('Number of Images')
        ax3.set_title('Image Count by Identity Type')
        
        for bar, val in zip(bars, [eval_imgs, neg_only_imgs]):
            ax3.text(bar.get_x() + bar.get_width()/2, val + 50,
                    f'{val:,}', ha='center', fontsize=10, fontweight='bold')
        
        plt.suptitle(
            'Negative-Only Identity Analysis\n'
            '(Single-outing identities that cannot be used for cross-time evaluation)',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        
        save_path = self.output_dir / '06_negative_only_analysis.png'
        plt.savefig(save_path)
        plt.close()
        print(f"   Saved: {save_path.name}")
    
    def plot_dashboard(self):
        """Create a comprehensive dashboard with key metrics."""
        print("\n[7/8] Generating comprehensive dashboard...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # === Row 1 ===
        
        # 1.1: Key metrics text
        ax_metrics = fig.add_subplot(gs[0, 0])
        ax_metrics.axis('off')
        
        total_images = len(self.df)
        total_identities = self.df['identity'].nunique()
        total_sources = self.df['dataset'].nunique()
        train_images = (self.df['split'] == 'train').sum()
        test_images = (self.df['split'] == 'test').sum()
        
        neg_only_ids = 0
        neg_only_imgs = 0
        if 'negative_only' in self.df.columns:
            neg_only_ids = self.df[self.df['negative_only'] == True]['identity'].nunique()
            neg_only_imgs = (self.df['negative_only'] == True).sum()
        
        metrics_text = f"""
        STAR DATASET SUMMARY
        {'='*30}
        
        Total Images:     {total_images:,}
        Total Identities: {total_identities:,}
        Dataset Sources:  {total_sources}
        
        TRAIN/TEST SPLIT
        {'─'*30}
        Train Images:     {train_images:,} ({train_images/total_images*100:.1f}%)
        Test Images:      {test_images:,} ({test_images/total_images*100:.1f}%)
        
        IDENTITY TYPES
        {'─'*30}
        Evaluable IDs:    {total_identities - neg_only_ids:,}
        Negative-Only:    {neg_only_ids:,}
        """
        
        ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=10, fontfamily='monospace', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=COLORS['background'], alpha=0.8))
        ax_metrics.set_title('Key Metrics', fontsize=12, fontweight='bold')
        
        # 1.2: Train/Test pie chart
        ax_pie = fig.add_subplot(gs[0, 1])
        ax_pie.pie(
            [train_images, test_images],
            labels=['Train', 'Test'],
            autopct='%1.1f%%',
            colors=[COLORS['train'], COLORS['test']],
            explode=[0, 0.05],
            startangle=90,
        )
        ax_pie.set_title('Train/Test Split', fontsize=12, fontweight='bold')
        
        # 1.3: Top sources bar
        ax_top = fig.add_subplot(gs[0, 2])
        top_sources = self.df['dataset'].value_counts().head(8)
        bars = ax_top.barh(range(len(top_sources)), top_sources.values[::-1], 
                          color=SOURCE_PALETTE[:8])
        ax_top.set_yticks(range(len(top_sources)))
        ax_top.set_yticklabels(top_sources.index[::-1], fontsize=8)
        ax_top.set_xlabel('Images')
        ax_top.set_title('Top 8 Sources', fontsize=12, fontweight='bold')
        
        # === Row 2 ===
        
        # 2.1-2.2: Split by source (wide)
        ax_split = fig.add_subplot(gs[1, :2])
        split_counts = self.df.groupby(['dataset', 'split']).size().unstack(fill_value=0)
        for col in ['train', 'test']:
            if col not in split_counts.columns:
                split_counts[col] = 0
        split_counts['total'] = split_counts.sum(axis=1)
        split_counts = split_counts.sort_values('total', ascending=True)
        
        y_pos = np.arange(len(split_counts))
        ax_split.barh(y_pos, split_counts['train'], height=0.8, 
                     label='Train', color=COLORS['train'], alpha=0.9)
        ax_split.barh(y_pos, split_counts['test'], height=0.8,
                     left=split_counts['train'], label='Test', 
                     color=COLORS['test'], alpha=0.9)
        ax_split.set_yticks(y_pos)
        ax_split.set_yticklabels(split_counts.index, fontsize=7)
        ax_split.set_xlabel('Images')
        ax_split.set_title('Split Distribution by Source', fontsize=12, fontweight='bold')
        ax_split.legend(loc='lower right')
        
        # 2.3: Images per identity histogram
        ax_hist = fig.add_subplot(gs[1, 2])
        img_per_id = self.df.groupby('identity').size()
        ax_hist.hist(img_per_id.values, bins=30, color=COLORS['primary'], 
                    alpha=0.7, edgecolor='white')
        ax_hist.axvline(img_per_id.mean(), color=COLORS['accent'], linestyle='--',
                       label=f'Mean: {img_per_id.mean():.1f}')
        ax_hist.set_xlabel('Images per Identity')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Class Distribution', fontsize=12, fontweight='bold')
        ax_hist.legend()
        
        # === Row 3 ===
        
        # 3.1: Identities per source
        ax_ids = fig.add_subplot(gs[2, 0])
        id_per_source = self.df.groupby('dataset')['identity'].nunique().sort_values(ascending=True).tail(10)
        ax_ids.barh(range(len(id_per_source)), id_per_source.values,
                   color=SOURCE_PALETTE[:len(id_per_source)])
        ax_ids.set_yticks(range(len(id_per_source)))
        ax_ids.set_yticklabels(id_per_source.index, fontsize=8)
        ax_ids.set_xlabel('Unique Identities')
        ax_ids.set_title('Identities per Source (Top 10)', fontsize=11, fontweight='bold')
        
        # 3.2: Train split percentage by source
        ax_pct = fig.add_subplot(gs[2, 1])
        split_pct = (split_counts['train'] / split_counts['total'] * 100).sort_values()
        colors = [COLORS['train'] if p >= 50 else COLORS['test'] for p in split_pct.values]
        ax_pct.barh(range(len(split_pct)), split_pct.values, color=colors, alpha=0.8)
        ax_pct.axvline(80, color='red', linestyle='--', alpha=0.5, label='80% target')
        ax_pct.set_yticks(range(len(split_pct)))
        ax_pct.set_yticklabels(split_pct.index, fontsize=7)
        ax_pct.set_xlabel('Train %')
        ax_pct.set_title('Train Split % by Source', fontsize=11, fontweight='bold')
        ax_pct.legend()
        
        # 3.3: Box plot of images per identity
        ax_box = fig.add_subplot(gs[2, 2])
        top_5_sources = self.df['dataset'].value_counts().head(5).index.tolist()
        box_data = [
            self.df[self.df['dataset'] == src].groupby('identity').size().values
            for src in top_5_sources
        ]
        bp = ax_box.boxplot(box_data, labels=[s[:12] for s in top_5_sources], 
                           patch_artist=True)
        for patch, color in zip(bp['boxes'], SOURCE_PALETTE[:5]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax_box.set_ylabel('Images per Identity')
        ax_box.set_title('Distribution (Top 5 Sources)', fontsize=11, fontweight='bold')
        plt.setp(ax_box.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        
        plt.suptitle('Star Dataset Overview Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        save_path = self.output_dir / '07_dashboard.png'
        plt.savefig(save_path)
        plt.close()
        print(f"   Saved: {save_path.name}")
    
    def save_summary_statistics(self):
        """Save detailed summary statistics to CSV and text files."""
        print("\n[8/8] Generating summary statistics...")
        
        # === Per-source statistics ===
        source_stats = []
        for source in self.df['dataset'].unique():
            source_df = self.df[self.df['dataset'] == source]
            img_per_id = source_df.groupby('identity').size()
            
            stats = {
                'dataset_source': source,
                'total_images': len(source_df),
                'unique_identities': source_df['identity'].nunique(),
                'train_images': (source_df['split'] == 'train').sum(),
                'test_images': (source_df['split'] == 'test').sum(),
                'train_pct': (source_df['split'] == 'train').mean() * 100,
                'avg_images_per_identity': img_per_id.mean(),
                'std_images_per_identity': img_per_id.std(),
                'min_images_per_identity': img_per_id.min(),
                'max_images_per_identity': img_per_id.max(),
            }
            
            if 'negative_only' in self.df.columns:
                stats['negative_only_identities'] = source_df[source_df['negative_only'] == True]['identity'].nunique()
                stats['negative_only_images'] = (source_df['negative_only'] == True).sum()
            
            source_stats.append(stats)
        
        stats_df = pd.DataFrame(source_stats)
        stats_df = stats_df.sort_values('total_images', ascending=False)
        
        # Save CSV
        csv_path = self.output_dir / 'source_statistics.csv'
        stats_df.to_csv(csv_path, index=False)
        print(f"   Saved: {csv_path.name}")
        
        # === Generate text report ===
        report_lines = [
            "=" * 70,
            "STAR DATASET ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Generated from: {self.metadata_path.name}",
            f"Total Images: {len(self.df):,}",
            f"Total Identities: {self.df['identity'].nunique():,}",
            f"Dataset Sources: {self.df['dataset'].nunique()}",
            "",
            "-" * 70,
            "TRAIN/TEST SPLIT SUMMARY",
            "-" * 70,
            f"Train Images: {(self.df['split'] == 'train').sum():,} ({(self.df['split'] == 'train').mean()*100:.1f}%)",
            f"Test Images: {(self.df['split'] == 'test').sum():,} ({(self.df['split'] == 'test').mean()*100:.1f}%)",
            "",
        ]
        
        if 'negative_only' in self.df.columns:
            neg_ids = self.df[self.df['negative_only'] == True]['identity'].nunique()
            neg_imgs = (self.df['negative_only'] == True).sum()
            report_lines.extend([
                "-" * 70,
                "NEGATIVE-ONLY IDENTITIES (Single-Outing)",
                "-" * 70,
                f"Negative-Only Identities: {neg_ids}",
                f"Negative-Only Images: {neg_imgs:,}",
                f"Evaluable Identities: {self.df['identity'].nunique() - neg_ids}",
                "",
            ])
        
        report_lines.extend([
            "-" * 70,
            "IMAGES PER IDENTITY STATISTICS",
            "-" * 70,
        ])
        
        img_per_id = self.df.groupby('identity').size()
        report_lines.extend([
            f"Mean: {img_per_id.mean():.2f}",
            f"Median: {img_per_id.median():.2f}",
            f"Std: {img_per_id.std():.2f}",
            f"Min: {img_per_id.min()}",
            f"Max: {img_per_id.max()}",
            "",
            "-" * 70,
            "TOP 10 DATASET SOURCES (by image count)",
            "-" * 70,
        ])
        
        for _, row in stats_df.head(10).iterrows():
            report_lines.append(
                f"{row['dataset_source']}: {int(row['total_images']):,} images, "
                f"{int(row['unique_identities'])} identities, "
                f"{row['train_pct']:.1f}% train"
            )
        
        report_lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])
        
        report_path = self.output_dir / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"   Saved: {report_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for the star_dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python star_dataset_utils/visualize_dataset.py
    python star_dataset_utils/visualize_dataset.py --metadata ./star_dataset/metadata_temporal.csv
    python star_dataset_utils/visualize_dataset.py --output ./custom_output_dir
        """
    )
    parser.add_argument(
        '--metadata', '-m',
        type=str,
        default=None,
        help='Path to metadata CSV (default: auto-detect in star_dataset/)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for visualizations (default: star_dataset_utils/outputs/)'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default=None,
        help='Root directory of star_dataset (default: parent of metadata file)'
    )
    
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Find metadata file
    if args.metadata:
        metadata_path = Path(args.metadata)
    else:
        # Try common locations
        candidates = [
            project_root / 'star_dataset' / 'metadata_temporal.csv',
            project_root / 'star_dataset' / 'metadata.csv',
            project_root / 'star_dataset_resized' / 'metadata_temporal.csv',
            project_root / 'star_dataset_resized' / 'metadata.csv',
        ]
        metadata_path = None
        for candidate in candidates:
            if candidate.exists():
                metadata_path = candidate
                break
        
        if metadata_path is None:
            print("ERROR: Could not find metadata file. Please specify with --metadata")
            print("Searched locations:")
            for c in candidates:
                print(f"  - {c}")
            sys.exit(1)
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / 'outputs'
    
    # Dataset root
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    
    # Create visualizer and run
    visualizer = DatasetVisualizer(
        metadata_path=metadata_path,
        output_dir=output_dir,
        dataset_root=dataset_root,
    )
    
    visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()


