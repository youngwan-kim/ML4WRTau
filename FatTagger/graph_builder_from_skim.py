#!/usr/bin/env python3

import json
import uproot
import os
import awkward as ak
import vector
import torch
from torch_geometric.data import Data
from typing import List, Dict, Optional
from tqdm import tqdm
from pathlib import Path
import numpy as np
from collections import defaultdict
import argparse

vector.register_awkward()

def delta_r(eta1, phi1, eta2, phi2):
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)

def split_pfcand_by_fatjet(pfcand_array, pfcand_idx_array, fatjet_idx_array):
    pfcand_by_fatjet = defaultdict(list)
    
    for pfcand_idx, fatjet_idx in zip(pfcand_idx_array, fatjet_idx_array):
        pfcand_by_fatjet[fatjet_idx].append(pfcand_idx)
    
    max_fatjet_idx = max(fatjet_idx_array) if len(fatjet_idx_array) > 0 else -1
    result = [pfcand_by_fatjet.get(i, []) for i in range(max_fatjet_idx + 1)]
    
    return result

class FatJetTauMatcher:
    
    def __init__(self, dr_cut=0.8, verbose=False):
        self.dr_cut = dr_cut
        self.verbose = verbose
    

    def tau_from_bsm_decay_with_chain(self, genparts, verbose=False):
        if verbose:
            print("DEBUG [tau_from_bsm_decay_with_chain]: Starting decay chain analysis")
            print(f"DEBUG [tau_from_bsm_decay_with_chain]: Total genparts = {len(genparts)}")
 
        for i, p in enumerate(genparts):
            if abs(p.pdgId) != 15:
                continue  # not a tau

            if verbose:
                print(f"DEBUG [tau_from_bsm_decay_with_chain]: Found tau at idx={i}, pdgId={p.pdgId}, pt={p.pt:.1f}")
            chain = [p]
            idx = i
            
            if verbose:
                print(f"DEBUG [tau_from_bsm_decay_with_chain]: Phase 1 - Searching for τ_R (9900016) from tau idx={i}")
            phase1_steps = 0
            
            while True:
                mother_idx = genparts[idx].genPartIdxMother
                if verbose:
                    print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Step {phase1_steps}: idx={idx} -> mother_idx={mother_idx}")
                
                if mother_idx < 0 or mother_idx == idx:
                    if verbose:
                        print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Phase 1 terminated: invalid mother_idx={mother_idx}")
                    break
                    
                if mother_idx >= len(genparts):
                    if verbose:
                        print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Phase 1 terminated: mother_idx={mother_idx} out of bounds")
                    break
                    
                mother = genparts[mother_idx]
                chain.append(mother)
                if verbose:
                    print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Added to chain: idx={mother_idx}, pdgId={mother.pdgId}, pt={mother.pt:.1f}")
                
                if mother.pdgId == 9900016:
                    if verbose:
                        print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Found τ_R at idx={mother_idx}! Moving to phase 2")
                    idx = mother_idx
                    break
                    
                idx = mother_idx
                phase1_steps += 1
                
                if phase1_steps > 20:
                    if verbose:
                        print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Phase 1 safety break: too many steps ({phase1_steps})")
                    break
            else:
                if verbose:
                    print(f"DEBUG [tau_from_bsm_decay_with_chain]: No τ_R found for tau {i}, skipping")
                continue  # no 9900016 in chain
 
            if verbose:
                print(f"DEBUG [tau_from_bsm_decay_with_chain]: Phase 2 - Searching for W_R (±34) from τ_R idx={idx}")
            phase2_steps = 0
            
            while True:
                mother_idx = genparts[idx].genPartIdxMother
                if verbose:
                    print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Step {phase2_steps}: idx={idx} -> mother_idx={mother_idx}")
                
                if mother_idx < 0 or mother_idx == idx:
                    if verbose:
                        print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Phase 2 terminated: invalid mother_idx={mother_idx}")
                    break
                    
                if mother_idx >= len(genparts):
                    if verbose:
                        print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Phase 2 terminated: mother_idx={mother_idx} out of bounds")
                    break
                    
                mother = genparts[mother_idx]
                if verbose:
                    print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Examining mother: idx={mother_idx}, pdgId={mother.pdgId}, pt={mother.pt:.1f}")
                
                if mother.pdgId != 9900016:
                    chain.append(mother)
                    if verbose:
                        print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Non-τ_R mother found: pdgId={mother.pdgId}")
                    if abs(mother.pdgId) == 34:
                        if verbose:
                            print(f"DEBUG [tau_from_bsm_decay_with_chain]:   SUCCESS! Found W_R at idx={mother_idx}")
                            print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Final chain length: {len(chain)}")
                            for j, particle in enumerate(chain):
                                print(f"DEBUG [tau_from_bsm_decay_with_chain]:     Chain[{j}]: pdgId={particle.pdgId}, pt={particle.pt:.1f}")
                        return True, chain
                    if verbose:
                        print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Not W_R, breaking from phase 2")
                    break
                    
                chain.append(mother)
                if verbose:
                    print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Added τ_R to chain: idx={mother_idx}")
                idx = mother_idx
                phase2_steps += 1
                
                if phase2_steps > 20:
                    if verbose:
                        print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Phase 2 safety break: too many steps ({phase2_steps})")
                    break

            if verbose:
                print(f"DEBUG [tau_from_bsm_decay_with_chain]: FAILED to find complete BSM decay chain for tau {i}")
                print(f"DEBUG [tau_from_bsm_decay_with_chain]: Partial chain length: {len(chain)}")
                for j, particle in enumerate(chain):
                    print(f"DEBUG [tau_from_bsm_decay_with_chain]:   Chain[{j}]: pdgId={particle.pdgId}, pt={particle.pt:.1f}")
        return False, chain
    
    def match_tau_to_fatjet(self, event):
        matched_pairs = []
        
        try:
            tau_data = {
                "pt": event["Tau_pt"],
                "eta": event["Tau_eta"], 
                "phi": event["Tau_phi"],
            }
            
            if "Tau_genPartIdx" in event.fields:
                tau_data["genPartIdx"] = event["Tau_genPartIdx"]
            else:
                tau_data["genPartIdx"] = ak.full_like(event["Tau_pt"], -1)
                
            if "Tau_genPartFlav" in event.fields:
                tau_data["genPartFlav"] = event["Tau_genPartFlav"]
            else:
                tau_data["genPartFlav"] = ak.full_like(event["Tau_pt"], -1)
            
            taus = ak.zip(tau_data)
        except KeyError:
            return matched_pairs
        
        try:
            fatjet_data = {
                "pt": event["FatJet_pt"],
                "eta": event["FatJet_eta"],
                "phi": event["FatJet_phi"],
                "mass": event["FatJet_mass"],
            }
            
            if "FatJet_tau1" in event.fields:
                fatjet_data["tau1"] = event["FatJet_tau1"]
            else:
                fatjet_data["tau1"] = ak.full_like(event["FatJet_pt"], -999)
                
            if "FatJet_tau2" in event.fields:
                fatjet_data["tau2"] = event["FatJet_tau2"]
            else:
                fatjet_data["tau2"] = ak.full_like(event["FatJet_pt"], -999)
                
            if "FatJet_tau3" in event.fields:
                fatjet_data["tau3"] = event["FatJet_tau3"]
            else:
                fatjet_data["tau3"] = ak.full_like(event["FatJet_pt"], -999)
                
            if "FatJet_msoftdrop" in event.fields:
                fatjet_data["msoftdrop"] = event["FatJet_msoftdrop"]
            else:
                fatjet_data["msoftdrop"] = ak.full_like(event["FatJet_pt"], -999)
            
            fatjets = ak.zip(fatjet_data)
        except KeyError:
            return matched_pairs
        
        genparts = None
        if "GenPart_pt" in event.fields:
            try:
                genparts = ak.zip({
                    "pt": event["GenPart_pt"],
                    "eta": event["GenPart_eta"],
                    "phi": event["GenPart_phi"],
                    "pdgId": event["GenPart_pdgId"],
                    "genPartIdxMother": event["GenPart_genPartIdxMother"],
                })
            except KeyError:
                genparts = None
        
        for tau_idx, tau in enumerate(taus):
            for fatjet_idx, fatjet in enumerate(fatjets):
                dr = delta_r(tau.eta, tau.phi, fatjet.eta, fatjet.phi)
                if dr < self.dr_cut:
                    is_bsm_tau = False
                    if genparts is not None and tau.genPartIdx >= 0 and tau.genPartIdx < len(genparts):
                        is_bsm_tau, _ = self.tau_from_bsm_decay_with_chain(genparts, verbose=self.verbose)
                    
                    print(tau_idx, fatjet_idx, tau.genPartIdx, is_bsm_tau, dr)
                    matched_pairs.append({
                        'tau_idx': tau_idx,
                        'fatjet_idx': fatjet_idx,
                        'dr': dr,
                        'tau': tau,
                        'fatjet': fatjet,
                        'is_bsm_tau': is_bsm_tau
                    })
        
        return matched_pairs


class FatJetPFCandGraphBuilder:
    
    def __init__(self):
        self.pfcand_features = ["pt", "eta", "phi", "mass", "pdgId"]
        self.fatjet_features = ["pt", "eta", "phi", "mass", "tau1", "tau2", "tau3", "msoftdrop"]
        
    def extract_pfcandidates(self, event, fatjet_idx):
        try:
            pfcand_data = {}
            for feat in self.pfcand_features:
                branch = f"PFCand_{feat}"
                if branch in event.fields:
                    pfcand_data[feat] = event[branch]
                else:
                    default_val = 0 if feat == "charge" else -999
                    pfcand_data[feat] = ak.full_like(event["PFCand_pt"], default_val)
            
            fatjet_pfcand_idx = event["FatJetPFCand_pfCandIdx"]
            fatjet_idx_array = event["FatJetPFCand_jetIdx"]
            
            pfcand_indices_by_fatjet = split_pfcand_by_fatjet(
                pfcand_data["pt"], fatjet_pfcand_idx, fatjet_idx_array
            )
            
            if fatjet_idx >= len(pfcand_indices_by_fatjet):
                return []
            
            pfcand_indices = pfcand_indices_by_fatjet[fatjet_idx]
            
            pfcandidates = []
            for idx in pfcand_indices:
                pfcand = {}
                for feat in self.pfcand_features:
                    try:
                        pfcand[feat] = float(pfcand_data[feat][idx])
                    except (IndexError, TypeError):
                        pfcand[feat] = 0 if feat == "charge" else -999
                pfcandidates.append(pfcand)
            
            return pfcandidates
            
        except KeyError as e:
            print(f"Missing PFCandidate branch: {e}")
            return []
    
    def build_graph(self, pfcandidates, fatjet_info):
        if len(pfcandidates) < 2:
            return None
        
        node_features = []
        for pfcand in pfcandidates:
            features = [pfcand[feat] for feat in self.pfcand_features]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        num_nodes = len(pfcandidates)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2, with_replacement=False).T
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        edge_attr = []
        for i, j in edge_index.T:
            eta_i, phi_i = pfcandidates[i]["eta"], pfcandidates[i]["phi"]
            eta_j, phi_j = pfcandidates[j]["eta"], pfcandidates[j]["phi"]
            dr = delta_r(eta_i, phi_i, eta_j, phi_j)
            edge_attr.append([dr])
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        global_features = [fatjet_info[feat] for feat in self.fatjet_features]
        u = torch.tensor(global_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)


class SkimmedGraphBuilder:
    
    def __init__(self, dr_cut: float = 0.8, verbose: bool = False):
        self.matcher = FatJetTauMatcher(dr_cut=dr_cut, verbose=verbose)
        self.graph_builder = FatJetPFCandGraphBuilder()
    
    def process_skim_file(self, skim_json_path: str, output_dir: str):
        
        with open(skim_json_path) as f:
            skim_data = json.load(f)
        
        input_file = skim_data['input_file']
        selected_events = skim_data['selected_events']
        
        if not selected_events:
            print(f"No events selected in {skim_json_path}")
            return
        
        print(f"Processing {len(selected_events)} selected events from {input_file}")
        
        graphs = []
        
        with uproot.open(input_file) as f:
            events_tree = f["Events"]
            arrays = events_tree.arrays()
            
            for event_idx in tqdm(selected_events, 
                                desc=f"Processing {os.path.basename(input_file)}", 
                                unit="event"):
                
                event = arrays[event_idx]    
                matches = self.matcher.match_tau_to_fatjet(event)
                
                for match in matches:
                    fatjet_idx = match['fatjet_idx']
                    fatjet_info = match['fatjet']
                    
                    pfcandidates = self.graph_builder.extract_pfcandidates(event, fatjet_idx)
                    
                    if len(pfcandidates) < 2:
                        continue
                    
                    graph = self.graph_builder.build_graph(pfcandidates, fatjet_info)
                    
                    if graph is not None:
                        graph.is_bsm_tau = match['is_bsm_tau']
                        graph.tau_fatjet_dr = match['dr']
                        graph.event_idx = event_idx
                        graph.fatjet_idx = fatjet_idx
                        
                        graphs.append(graph)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        skim_filename = Path(skim_json_path).stem
        out_file = output_path / f"{skim_filename}_graphs.pt"
        torch.save(graphs, out_file)
        print(f"Saved {len(graphs)} graphs to {out_file}")
    
    def process_multiple_skim_files(self, skim_json_paths: List[str], output_dir: str):
        
        print(f"Processing {len(skim_json_paths)} skim files...")
        
        for skim_path in skim_json_paths:
            print(f"\nProcessing skim file: {skim_path}")
            self.process_skim_file(skim_path, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Build graphs from skimmed JSON event indices"
    )
    parser.add_argument("skim_files", nargs="+", 
                       help="JSON skim files to process")
    parser.add_argument("--output-dir", default="graphs_from_skim", 
                       help="Output directory for graph files")
    parser.add_argument("--dr-cut", type=float, default=0.8,
                       help="Delta R cut for tau-fatjet matching")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose debug output")
    
    args = parser.parse_args()
    
    builder = SkimmedGraphBuilder(dr_cut=args.dr_cut, verbose=args.verbose)
    
    builder.process_multiple_skim_files(args.skim_files, args.output_dir)


if __name__ == "__main__":
    main()