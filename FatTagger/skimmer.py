#!/usr/bin/env python3
import json
import uproot
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def delta_r(eta1, phi1, eta2, phi2):
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)

def skim_file(input_path, output_path, dr_cut=0.8):

    selected_events = []
    
    print(f"Processing {input_path}")
    
    with uproot.open(input_path) as f:
        events = f["Events"]
        n_events = events.num_entries
        
        arrays = events.arrays([
            "Tau_eta", "Tau_phi", 
            "FatJet_eta", "FatJet_phi"
        ])
        
        print(f"Checking {n_events} events...")
        
        for i in tqdm(range(n_events), desc="Events", unit="event"):
            tau_eta = arrays["Tau_eta"][i]
            tau_phi = arrays["Tau_phi"][i] 
            fatjet_eta = arrays["FatJet_eta"][i]
            fatjet_phi = arrays["FatJet_phi"][i]
            
            if len(tau_eta) == 0 or len(fatjet_eta) == 0:
                continue
            
            found_match = False
            for t_eta, t_phi in zip(tau_eta, tau_phi):
                for fj_eta, fj_phi in zip(fatjet_eta, fatjet_phi):
                    dr = delta_r(t_eta, t_phi, fj_eta, fj_phi)
                    if dr < dr_cut:
                        found_match = True
                        #print(i,dr)
                        break
                if found_match:
                    break
            
            if found_match:
                selected_events.append(i)
    
    results = {
        "input_file": input_path,
        "dr_cut": dr_cut,
        "total_events": n_events,
        "selected_events": selected_events,
        "efficiency": len(selected_events) / n_events if n_events > 0 else 0.0
    }
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Selected {len(selected_events)}/{n_events} events ({results['efficiency']:.3f})")
    print(f"Saved to {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Simple FatJet-Tau event skimmer")
    parser.add_argument("config", help="JSON config file with input ROOT file paths")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSON files")
    parser.add_argument("--dr-cut", type=float, default=0.8, help="Delta R cut (default: 0.8)")
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    input_paths = config["path"]
    sample_name = config.get("name", "sample")
    
    print(f"Skimming {len(input_paths)} files")
    print(f"Sample: {sample_name}")
    print(f"Delta R cut: {args.dr_cut}")
    
    total_input = 0
    total_selected = 0
    
    for input_path in input_paths:
        input_filename = Path(input_path).stem
        output_filename = f"{sample_name}_{input_filename}_skim.json"
        output_path = Path(args.output_dir) / output_filename
        
        results = skim_file(input_path, output_path, args.dr_cut)
        
        total_input += results["total_events"]
        total_selected += len(results["selected_events"])
        
        print()
    
    overall_efficiency = total_selected / total_input if total_input > 0 else 0.0
    print(f"=== Summary ===")
    print(f"Total events: {total_selected}/{total_input} ({overall_efficiency:.3f})")

if __name__ == "__main__":
    main()