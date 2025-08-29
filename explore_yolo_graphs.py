"""
Explore and visualize extracted YOLO graphs for GNN development.

This script helps you understand the graph structure and features
extracted from YOLO predictions.
"""
import argparse
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from collections import defaultdict, Counter

def parse_args():
    ap = argparse.ArgumentParser(description="Explore extracted YOLO graphs")
    ap.add_argument('--graph-file', type=str, required=True,
                   help='Path to extracted graph file (.pkl or .json)')
    ap.add_argument('--output-dir', type=str, default='graph_analysis',
                   help='Directory for analysis outputs')
    ap.add_argument('--show-plots', action='store_true',
                   help='Display plots interactively')
    ap.add_argument('--max-examples', type=int, default=10,
                   help='Max examples to show in detail')
    return ap.parse_args()


def load_graphs(file_path: Path) -> List[Dict[str, Any]]:
    """Load graphs from pickle or json file."""
    if file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def analyze_graphs(graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze graph statistics."""
    stats = {
        'total_graphs': len(graphs),
        'graphs_with_detections': 0,
        'graphs_with_edges': 0,
        'detection_counts': [],
        'node_counts': [],
        'edge_counts': [],
        'class_distributions': Counter(),
        'confidence_distributions': [],
        'spatial_distributions': [],
        'aneurysm_present_counts': Counter(),
        'fold_distributions': Counter()
    }
    
    for graph in graphs:
        if graph['num_detections'] > 0:
            stats['graphs_with_detections'] += 1
            stats['detection_counts'].append(graph['num_detections'])
            stats['node_counts'].append(graph['graph_data']['num_nodes'])
            stats['edge_counts'].append(graph['graph_data']['num_edges'])
            
            if graph['graph_data']['num_edges'] > 0:
                stats['graphs_with_edges'] += 1
            
            # Analyze detections
            for det in graph['detections']:
                stats['class_distributions'][det['class_name']] += 1
                stats['confidence_distributions'].append(det['conf'])
                
                # Spatial position (bbox center)
                x_center, y_center = det['bbox'][:2]
                stats['spatial_distributions'].append([x_center, y_center])
        
        # Label distributions
        if 'aneurysm_present' in graph:
            label = 'positive' if graph['aneurysm_present'] else 'negative'
            stats['aneurysm_present_counts'][label] += 1
            
        if 'fold_id' in graph:
            stats['fold_distributions'][graph['fold_id']] += 1
    
    return stats


def create_visualizations(stats: Dict[str, Any], output_dir: Path, show_plots: bool = False):
    """Create analysis visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Detection count distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    if stats['detection_counts']:
        plt.hist(stats['detection_counts'], bins=20, alpha=0.7)
        plt.xlabel('Number of Detections per Series')
        plt.ylabel('Frequency')
        plt.title('Detection Count Distribution')
        plt.axvline(np.mean(stats['detection_counts']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(stats["detection_counts"]):.1f}')
        plt.legend()
    
    # 2. Node vs Edge counts
    plt.subplot(2, 3, 2)
    if stats['node_counts'] and stats['edge_counts']:
        plt.scatter(stats['node_counts'], stats['edge_counts'], alpha=0.6)
        plt.xlabel('Number of Nodes')
        plt.ylabel('Number of Edges')
        plt.title('Nodes vs Edges')
        
        # Add diagonal line for reference
        max_nodes = max(stats['node_counts'])
        plt.plot([0, max_nodes], [0, max_nodes * (max_nodes - 1) / 2], 
                'r--', alpha=0.5, label='Fully Connected')
        plt.legend()
    
    # 3. Confidence distribution
    plt.subplot(2, 3, 3)
    if stats['confidence_distributions']:
        plt.hist(stats['confidence_distributions'], bins=30, alpha=0.7)
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Detection Confidence Distribution')
        plt.axvline(np.mean(stats['confidence_distributions']), color='red', linestyle='--',
                   label=f'Mean: {np.mean(stats["confidence_distributions"]):.3f}')
        plt.legend()
    
    # 4. Class distribution
    plt.subplot(2, 3, 4)
    if stats['class_distributions']:
        classes = list(stats['class_distributions'].keys())
        counts = list(stats['class_distributions'].values())
        plt.barh(classes, counts)
        plt.xlabel('Detection Count')
        plt.title('Class Distribution')
        plt.gca().tick_params(axis='y', labelsize=8)
    
    # 5. Spatial distribution (heatmap)
    plt.subplot(2, 3, 5)
    if stats['spatial_distributions']:
        positions = np.array(stats['spatial_distributions'])
        plt.hexbin(positions[:, 0], positions[:, 1], gridsize=20, cmap='Blues')
        plt.xlabel('X Position (normalized)')
        plt.ylabel('Y Position (normalized)')
        plt.title('Spatial Distribution of Detections')
        plt.colorbar(label='Density')
    
    # 6. Label distribution
    plt.subplot(2, 3, 6)
    if stats['aneurysm_present_counts']:
        labels = list(stats['aneurysm_present_counts'].keys())
        counts = list(stats['aneurysm_present_counts'].values())
        plt.pie(counts, labels=labels, autopct='%1.1f%%')
        plt.title('Aneurysm Present Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph_analysis.png', dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Additional detailed plots
    
    # Edge density distribution
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    edge_densities = []
    for i, (nodes, edges) in enumerate(zip(stats['node_counts'], stats['edge_counts'])):
        if nodes > 1:
            max_edges = nodes * (nodes - 1) / 2
            density = edges / max_edges if max_edges > 0 else 0
            edge_densities.append(density)
    
    if edge_densities:
        plt.hist(edge_densities, bins=20, alpha=0.7)
        plt.xlabel('Edge Density (edges / max_possible_edges)')
        plt.ylabel('Frequency')
        plt.title('Graph Edge Density Distribution')
        plt.axvline(np.mean(edge_densities), color='red', linestyle='--',
                   label=f'Mean: {np.mean(edge_densities):.3f}')
        plt.legend()
    
    # Confidence vs class
    plt.subplot(1, 2, 2)
    if stats['class_distributions'] and stats['confidence_distributions']:
        # This would need the original graph data to correlate conf with class
        plt.text(0.5, 0.5, 'Confidence vs Class\n(needs raw data)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Confidence vs Class Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_analysis.png', dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def show_examples(graphs: List[Dict[str, Any]], max_examples: int = 10):
    """Show detailed examples of graphs."""
    print("\n=== Graph Examples ===")
    
    # Filter to graphs with detections
    graphs_with_dets = [g for g in graphs if g['num_detections'] > 0]
    
    # Sort by number of detections (descending)
    graphs_with_dets.sort(key=lambda x: x['num_detections'], reverse=True)
    
    for i, graph in enumerate(graphs_with_dets[:max_examples]):
        print(f"\nExample {i+1}: Series {graph['series_id']}")
        print(f"  Aneurysm Present: {graph.get('aneurysm_present', 'Unknown')}")
        print(f"  Fold: {graph.get('fold_id', 'Unknown')}")
        print(f"  Detections: {graph['num_detections']}")
        print(f"  Slices processed: {graph['num_slices_processed']}")
        print(f"  Graph nodes: {graph['graph_data']['num_nodes']}")
        print(f"  Graph edges: {graph['graph_data']['num_edges']}")
        
        if graph['detections']:
            print(f"  Detection details:")
            for j, det in enumerate(graph['detections'][:5]):  # Show first 5 detections
                print(f"    Det {j}: class={det['class_name']}, conf={det['conf']:.3f}, "
                      f"bbox={[f'{x:.3f}' for x in det['bbox']]}")
            if len(graph['detections']) > 5:
                print(f"    ... and {len(graph['detections']) - 5} more")
        
        if graph['graph_data']['node_features']:
            node_features = np.array(graph['graph_data']['node_features'])
            print(f"  Node features shape: {node_features.shape}")
            print(f"  Feature ranges: conf=[{node_features[:, 0].min():.3f}, {node_features[:, 0].max():.3f}], "
                  f"x=[{node_features[:, 1].min():.3f}, {node_features[:, 1].max():.3f}], "
                  f"y=[{node_features[:, 2].min():.3f}, {node_features[:, 2].max():.3f}]")


def main():
    args = parse_args()
    
    # Load graphs
    print(f"Loading graphs from {args.graph_file}")
    graphs = load_graphs(Path(args.graph_file))
    print(f"Loaded {len(graphs)} graphs")
    
    # Analyze
    print("Analyzing graphs...")
    stats = analyze_graphs(graphs)
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Total graphs: {stats['total_graphs']}")
    print(f"Graphs with detections: {stats['graphs_with_detections']} ({stats['graphs_with_detections']/stats['total_graphs']*100:.1f}%)")
    print(f"Graphs with edges: {stats['graphs_with_edges']} ({stats['graphs_with_edges']/max(1,stats['graphs_with_detections'])*100:.1f}% of non-empty)")
    
    if stats['detection_counts']:
        print(f"Detection count stats: mean={np.mean(stats['detection_counts']):.1f}, "
              f"std={np.std(stats['detection_counts']):.1f}, "
              f"max={np.max(stats['detection_counts'])}")
        
    if stats['confidence_distributions']:
        print(f"Confidence stats: mean={np.mean(stats['confidence_distributions']):.3f}, "
              f"std={np.std(stats['confidence_distributions']):.3f}")
    
    print(f"Class distribution:")
    for class_name, count in stats['class_distributions'].most_common():
        print(f"  {class_name}: {count}")
        
    if stats['aneurysm_present_counts']:
        print(f"Label distribution: {dict(stats['aneurysm_present_counts'])}")
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    print(f"Creating visualizations in {output_dir}")
    create_visualizations(stats, output_dir, args.show_plots)
    
    # Show examples
    show_examples(graphs, args.max_examples)
    
    # Save detailed stats
    stats_serializable = {k: v for k, v in stats.items() 
                         if not isinstance(v, (Counter, np.ndarray))}
    stats_serializable['class_distribution'] = dict(stats['class_distributions'])
    stats_serializable['aneurysm_present_distribution'] = dict(stats['aneurysm_present_counts'])
    stats_serializable['fold_distribution'] = dict(stats['fold_distributions'])
    
    with open(output_dir / 'analysis_stats.json', 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()