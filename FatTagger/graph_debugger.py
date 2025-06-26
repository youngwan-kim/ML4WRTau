#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from collections import Counter
import networkx as nx

class GraphDebugger:
    
    def __init__(self):
        self.pfcand_features = ["pt", "eta", "phi", "mass", "pdgId"]
        self.fatjet_features = ["pt", "eta", "phi", "mass", "tau1", "tau2", "tau3", "msoftdrop"]
        
        self.pdg_names = {
            211: "ChgHadPlus", -211: "ChgHadMinus",
            130: "NeuHad", 22: "Photon",
            11: "Electron", -11: "Positron",
            13: "Muon", -13: "AntiMuon",
            1: "HFHad", 2:"HFEM"
        }
    
    def load_graphs(self, pt_file_path: str) -> List:
        try:
            graphs = torch.load(pt_file_path, map_location='cpu', weights_only=False)
            print(f"Loaded {len(graphs)} graphs from {pt_file_path}")
            return graphs
        except Exception as e:
            print(f"Error loading {pt_file_path}: {e}")
            return []
    
    def get_basic_stats(self, graphs: List) -> Dict:
        if not graphs:
            return {}
        
        stats = {
            'total_graphs': len(graphs),
            'node_counts': [g.x.shape[0] for g in graphs],
            'edge_counts': [g.edge_index.shape[1] for g in graphs],
            'bsm_tau_counts': sum(1 for g in graphs if hasattr(g, 'is_bsm_tau') and g.is_bsm_tau),
            'dr_values': [float(g.tau_fatjet_dr) for g in graphs if hasattr(g, 'tau_fatjet_dr')],
        }
        
        stats['min_nodes'] = min(stats['node_counts'])
        stats['max_nodes'] = max(stats['node_counts'])
        stats['avg_nodes'] = np.mean(stats['node_counts'])
        stats['median_nodes'] = np.median(stats['node_counts'])
        
        stats['min_edges'] = min(stats['edge_counts'])
        stats['max_edges'] = max(stats['edge_counts'])
        stats['avg_edges'] = np.mean(stats['edge_counts'])
        
        if stats['dr_values']:
            stats['min_dr'] = min(stats['dr_values'])
            stats['max_dr'] = max(stats['dr_values'])
            stats['avg_dr'] = np.mean(stats['dr_values'])
        
        return stats
    
    def analyze_single_graph(self, graph, graph_idx: int = 0) -> Dict:
        analysis = {
            'graph_idx': graph_idx,
            'num_nodes': graph.x.shape[0],
            'num_edges': graph.edge_index.shape[1],
            'node_features': graph.x.shape[1],
            'global_features': graph.u.shape[0] if hasattr(graph, 'u') else 0,
        }
        
        if hasattr(graph, 'is_bsm_tau'):
            analysis['is_bsm_tau'] = bool(graph.is_bsm_tau)
        if hasattr(graph, 'tau_fatjet_dr'):
            analysis['tau_fatjet_dr'] = float(graph.tau_fatjet_dr)
        if hasattr(graph, 'event_idx'):
            analysis['event_idx'] = int(graph.event_idx)
        if hasattr(graph, 'fatjet_idx'):
            analysis['fatjet_idx'] = int(graph.fatjet_idx)
        
        node_features = graph.x.numpy()
        analysis['node_feature_stats'] = {}
        for i, feat_name in enumerate(self.pfcand_features):
            feat_values = node_features[:, i]
            analysis['node_feature_stats'][feat_name] = {
                'min': float(feat_values.min()),
                'max': float(feat_values.max()),
                'mean': float(feat_values.mean()),
                'std': float(feat_values.std()),
                'zeros': int(np.sum(feat_values == 0)),
                'invalid': int(np.sum(feat_values == -999))
            }
        
        pdg_ids = node_features[:, 4].astype(int)  
        pdg_counter = Counter(pdg_ids)
        analysis['particle_types'] = {
            int(pdg): count for pdg, count in pdg_counter.items()
        }
        analysis['particle_names'] = {
            self.pdg_names.get(pdg, f"PDG_{pdg}"): count 
            for pdg, count in pdg_counter.items()
        }
        
        edge_attr = graph.edge_attr.numpy()
        if edge_attr.shape[1] > 0:
            dr_values = edge_attr[:, 0]   
            analysis['edge_dr_stats'] = {
                'min': float(dr_values.min()),
                'max': float(dr_values.max()),
                'mean': float(dr_values.mean()),
                'std': float(dr_values.std())
            }
        
        if hasattr(graph, 'u'):
            global_features = graph.u.numpy()
            analysis['global_feature_stats'] = {}
            for i, feat_name in enumerate(self.fatjet_features):
                if i < len(global_features):
                    analysis['global_feature_stats'][feat_name] = float(global_features[i])
        
        return analysis
    
    def check_graph_sanity(self, graphs: List) -> Dict: 
        issues = {
            'empty_graphs': [],
            'single_node_graphs': [],
            'extreme_dr_values': [],
            'missing_metadata': [],
            'invalid_features': [],
            'disconnected_graphs': []
        }
        
        for i, graph in enumerate(graphs):
            if graph.x.shape[0] == 0:
                issues['empty_graphs'].append(i)
            
            if graph.x.shape[0] == 1:
                issues['single_node_graphs'].append(i)
            
            if hasattr(graph, 'tau_fatjet_dr'):
                dr = float(graph.tau_fatjet_dr)
                if dr > 0.8 or dr < 0:
                    issues['extreme_dr_values'].append((i, dr))
            
            required_attrs = ['is_bsm_tau', 'tau_fatjet_dr', 'event_idx', 'fatjet_idx']
            missing = [attr for attr in required_attrs if not hasattr(graph, attr)]
            if missing:
                issues['missing_metadata'].append((i, missing))
            
            node_features = graph.x.numpy()
            if np.any(np.isnan(node_features)) or np.any(np.isinf(node_features)):
                issues['invalid_features'].append(i)
            
            if graph.x.shape[0] < 100:  
                edge_index = graph.edge_index.numpy()
                if edge_index.shape[1] == 0:
                    issues['disconnected_graphs'].append(i)
        
        return issues
    
    def plot_overview(self, graphs: List, figsize: Tuple[int, int] = (15, 10)):
        stats = self.get_basic_stats(graphs)
        
        if not stats:
            print("No graphs to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        axes[0, 0].hist(stats['node_counts'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Number of Nodes')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Node Count Distribution')
        axes[0, 0].axvline(stats['avg_nodes'], color='red', linestyle='--', 
                          label=f'Mean: {stats["avg_nodes"]:.1f}')
        axes[0, 0].legend()
        
        axes[0, 1].hist(stats['edge_counts'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Number of Edges')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Edge Count Distribution')
        axes[0, 1].axvline(stats['avg_edges'], color='red', linestyle='--',
                          label=f'Mean: {stats["avg_edges"]:.1f}')
        axes[0, 1].legend()
        
        if stats['dr_values']:
            axes[0, 2].hist(stats['dr_values'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 2].set_xlabel('Tau-FatJet ΔR')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].set_title('ΔR Distribution')
            axes[0, 2].axvline(stats['avg_dr'], color='red', linestyle='--',
                              label=f'Mean: {stats["avg_dr"]:.3f}')
            axes[0, 2].legend()
        
        bsm_count = stats['bsm_tau_counts']
        regular_count = stats['total_graphs'] - bsm_count
        axes[1, 0].pie([regular_count, bsm_count], 
                       labels=['Regular Tau', 'BSM Tau'],
                       autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('BSM vs Regular Tau')
        
        axes[1, 1].scatter(stats['node_counts'], stats['edge_counts'], alpha=0.6)
        axes[1, 1].set_xlabel('Number of Nodes')
        axes[1, 1].set_ylabel('Number of Edges')
        axes[1, 1].set_title('Nodes vs Edges')
        
        summary_text = f"""
        Total Graphs: {stats['total_graphs']}
        Nodes: {stats['min_nodes']} - {stats['max_nodes']} (avg: {stats['avg_nodes']:.1f})
        Edges: {stats['min_edges']} - {stats['max_edges']} (avg: {stats['avg_edges']:.1f})
        BSM Taus: {bsm_count} ({bsm_count/stats['total_graphs']*100:.1f}%)
        """
        if stats['dr_values']:
            summary_text += f"ΔR: {stats['min_dr']:.3f} - {stats['max_dr']:.3f} (avg: {stats['avg_dr']:.3f})"
        
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center')
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.show()
    
    def plot_single_graph_analysis(self, graph, graph_idx: int = 0, figsize: Tuple[int, int] = (15, 8)):
        analysis = self.analyze_single_graph(graph, graph_idx)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        info_text = f"""
        Graph {graph_idx}
        Nodes: {analysis['num_nodes']}
        Edges: {analysis['num_edges']}
        BSM Tau: {analysis.get('is_bsm_tau', 'Unknown')}
        ΔR: {analysis.get('tau_fatjet_dr', 'Unknown'):.3f}
        Event: {analysis.get('event_idx', 'Unknown')}
        FatJet: {analysis.get('fatjet_idx', 'Unknown')}
        """
        axes[0, 0].text(0.1, 0.5, info_text, transform=axes[0, 0].transAxes,
                        fontsize=12, verticalalignment='center')
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Graph Info')
        
        particle_names = analysis['particle_names']
        if particle_names:
            names = list(particle_names.keys())
            counts = list(particle_names.values())
            axes[0, 1].bar(names, counts)
            axes[0, 1].set_xlabel('Particle Type')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Particle Types')
            plt.setp(axes[0, 1].get_xticklabels(), rotation=45)
        
        node_features = graph.x.numpy()
        pt_values = node_features[:, 0] 
        axes[0, 2].hist(pt_values, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('pT [GeV]')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('PFCandidate pT Distribution')
        
        eta_values = node_features[:, 1]
        phi_values = node_features[:, 2]
        scatter = axes[1, 0].scatter(eta_values, phi_values, c=pt_values, 
                                   cmap='viridis', alpha=0.7, s=30)
        
        if hasattr(graph, 'u'):
            fatjet_eta = float(graph.u[1])  
            fatjet_phi = float(graph.u[2])  
            
            axes[1, 0].scatter(fatjet_eta, fatjet_phi, c='red', marker='*', 
                             s=200, label='FatJet Center', edgecolors='black', linewidth=1)
            
            circle = plt.Circle((fatjet_eta, fatjet_phi), 0.8, 
                              fill=False, color='red', linestyle='--', 
                              linewidth=2, label='ΔR = 0.8')
            axes[1, 0].add_patch(circle)
        
        axes[1, 0].set_xlabel('η')
        axes[1, 0].set_ylabel('φ')
        axes[1, 0].set_title('PFCandidate η-φ Distribution')
        axes[1, 0].legend()
        axes[1, 0].set_aspect('equal', adjustable='box')
        plt.colorbar(scatter, ax=axes[1, 0], label='pT [GeV]')
        
        if 'edge_dr_stats' in analysis:
            edge_attr = graph.edge_attr.numpy()
            dr_values = edge_attr[:, 0]
            axes[1, 1].hist(dr_values, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('ΔR between PFCandidates')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Edge ΔR Distribution')
        
        feature_subset = node_features[:, :5]  
        feature_names = self.pfcand_features[:5]
        corr_matrix = np.corrcoef(feature_subset.T)
        im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 2].set_xticks(range(len(feature_names)))
        axes[1, 2].set_yticks(range(len(feature_names)))
        axes[1, 2].set_xticklabels(feature_names, rotation=45)
        axes[1, 2].set_yticklabels(feature_names)
        axes[1, 2].set_title('Feature Correlation')
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.show()
    
    def plot_network_layout(self, graph, graph_idx: int = 0, layout: str = 'spring', 
                           figsize: Tuple[int, int] = (12, 8)):
        
        edge_index = graph.edge_index.numpy()
        G = nx.Graph()
        G.add_nodes_from(range(graph.x.shape[0]))
        
        edges = [(int(edge_index[0, i]), int(edge_index[1, i])) 
                 for i in range(0, edge_index.shape[1], 2)]  
        G.add_edges_from(edges)
        
        node_features = graph.x.numpy()
        pt_values = node_features[:, 0]
        pdg_ids = node_features[:, 4].astype(int)
        
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        nodes1 = nx.draw_networkx_nodes(G, pos, node_color=pt_values, 
                                       cmap='viridis', node_size=50, ax=ax1)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax1)
        ax1.set_title(f'Graph {graph_idx}: Colored by pT')
        plt.colorbar(nodes1, ax=ax1, label='pT [GeV]')
        ax1.axis('off')
        
        unique_pdgs = list(set(pdg_ids))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pdgs)))
        pdg_colors = {pdg: colors[i] for i, pdg in enumerate(unique_pdgs)}
        node_colors = [pdg_colors[pdg] for pdg in pdg_ids]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, ax=ax2)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax2)
        ax2.set_title(f'Graph {graph_idx}: Colored by Particle Type')
        ax2.axis('off')
        
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=pdg_colors[pdg], markersize=8,
                                     label=self.pdg_names.get(pdg, f'PDG_{pdg}'))
                          for pdg in unique_pdgs]
        ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.show()
    
    def export_analysis_report(self, graphs: List, output_path: str):
        
        stats = self.get_basic_stats(graphs)
        issues = self.check_graph_sanity(graphs)
        
        with open(output_path, 'w') as f:
            f.write("Graph Dataset Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("Basic Statistics:\n")
            f.write("-" * 20 + "\n")
            for key, value in stats.items():
                if isinstance(value, list):
                    continue
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("Issues Found:\n")
            f.write("-" * 20 + "\n")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    f.write(f"{issue_type}: {len(issue_list)} instances\n")
                    if len(issue_list) <= 10:  
                        f.write(f"  Details: {issue_list}\n")
                    f.write("\n")
            
            if graphs:
                f.write("Sample Graph Analysis:\n")
                f.write("-" * 20 + "\n")
                sample_analysis = self.analyze_single_graph(graphs[0], 0)
                for key, value in sample_analysis.items():
                    f.write(f"{key}: {value}\n")
        
        print(f"Analysis report exported to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug PyTorch Geometric graph files")
    parser.add_argument("pt_files", nargs="+", help="Graph .pt files to analyze")
    parser.add_argument("--output-report", help="Export analysis report to file")
    parser.add_argument("--show-plots", action="store_true", help="Show visualization plots")
    
    args = parser.parse_args()
    
    debugger = GraphDebugger()
    
    for pt_file in args.pt_files:
        print(f"\nAnalyzing {pt_file}")
        graphs = debugger.load_graphs(pt_file)
        
        if not graphs:
            continue
        
        stats = debugger.get_basic_stats(graphs)
        print(f"Total graphs: {stats['total_graphs']}")
        print(f"Node count range: {stats['min_nodes']} - {stats['max_nodes']}")
        print(f"BSM tau graphs: {stats['bsm_tau_counts']}")
        
        issues = debugger.check_graph_sanity(graphs)
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        if total_issues > 0:
            print(f"Found {total_issues} potential issues")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    print(f"  {issue_type}: {len(issue_list)}")

        if args.output_report:
            report_path = args.output_report.replace('.txt', f'_{Path(pt_file).stem}.txt')
            debugger.export_analysis_report(graphs, report_path)
        
        if args.show_plots:
            debugger.plot_overview(graphs)
            if graphs:
                debugger.plot_single_graph_analysis(graphs[0])


if __name__ == "__main__":
    main()