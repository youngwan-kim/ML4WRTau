import json, uproot, os, itertools
import awkward as ak
import vector, torch
from torch_geometric.data import Data
from typing import List, Dict
from tqdm.autonotebook import tqdm
from multiprocessing import Pool
from pathlib import Path
import numpy as np

vector.register_awkward()

class HEPSkimmer:
    def __init__(self, hlt_name="HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1"):
        self.hlt_name = hlt_name

    def event_passes(self, event):
        try:
            if not event[self.hlt_name]:
                return False
        except KeyError:
            return False 

        try:
            tau_pt = event["Tau_pt"]
            if len(tau_pt) < 1:
                return False
            
        except KeyError:
            return False

        return True




class GraphConfig:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self.file_paths = self.config["path"]

def delta_r(eta1, phi1, eta2, phi2):
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)

class HEPEventGraphBuilder:
    def __init__(self):
        self.features_common = ["pt", "eta", "phi", "mass"]
        self.features_by_type = {
            "Muon": self.features_common + ["charge","dxy","dz","miniPFRelIso_all","sip3d"],
            "Electron": self.features_common + ["charge","dxy","dz","eInvMinusPInv","hoe","sip3d","miniPFRelIso_all"],
            "Tau": self.features_common + ["charge", "decayMode","dxy","dz","leadTkDeltaEta","leadTkDeltaPhi","leadTkPtOverTauPt","rawIso"],
            "Jet": self.features_common + ["btagDeepFlavB","chHEF","chEmEF"],
            "FatJet": self.features_common + ["tau1", "tau2", "tau3", "msoftdrop"]
        }
        self.object_type_id = {
            "Muon": 0,
            "Electron": 1,
            "Tau": 2,
            "Jet": 3,
            "FatJet": 4
        }
        self.max_feature_len = max(len(feats) for feats in self.features_by_type.values())

    def build_features(self, events, objname: str, features: List[str]):
        feature_dict = {}
        valid_feature = None
        for feat in features:
            branch = f"{objname}_{feat}"
            if branch in events.fields:
                feature_dict[feat] = events[branch]
                if valid_feature is None:
                    valid_feature = events[branch]
            else:
                shape = ak.num(valid_feature)
                feature_dict[feat] = ak.full_like(valid_feature, -999)
        out = ak.zip(feature_dict)
        out["type"] = ak.full_like(feature_dict[features[0]], self.object_type_id[objname])
        #print(out)
        return out
    
    def build_nodes(self, event):

        node_features = []
        node_types = []

        for obj_type, feature_list in self.features_by_type.items():
            #print("build_nodes")
            #print(obj_type,feature_list)
            branch_name = f"{obj_type}_{feature_list[0]}"
            if branch_name not in event.fields :
                continue
            try:
                num_objects = len(event[f"{obj_type}_{feature_list[0]}"])
                #print("\t",num_objects)
            except (KeyError, TypeError):
                continue  

            for i in range(num_objects) :
                features = []
                for feat in feature_list :
                    branch = f"{obj_type}_{feat}"
                    if branch in event.fields :
                        try : val = event[branch][i]
                        except Exception : val = -999.0
                    else : val = -999.0
                    features.append(val)
                padded_features = features + [-999.0] * (self.max_feature_len - len(features))
                node_features.append(padded_features)
                node_types.append(self.object_type_id[obj_type])
 
        #print(f"Event nodes: {len(node_features)}")
        return node_features, node_types


    def compute_delta_r(self, obj1, obj2):
        return obj1.deltaR(obj2)


    
    def event_to_graph(self, event):
        node_features, node_types = self.build_nodes(event)

        if len(node_features) < 2:
            return None

        x_feat = torch.tensor(node_features, dtype=torch.float)
        node_types_tensor = torch.tensor(node_types, dtype=torch.long) 

        x = x_feat  

        num_nodes = x.shape[0]
        edge_index = torch.combinations(torch.arange(num_nodes), r=2, with_replacement=False).T
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        eta = x_feat[:, 1].numpy()  
        phi = x_feat[:, 2].numpy()

        edge_attr = []
        for i, j in edge_index.T:
            dr = delta_r(eta[i], phi[i], eta[j], phi[j])
            edge_attr.append([dr])
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        try:
            met = float(event["MET_pt"])
        except (KeyError, TypeError):
            met = -999.0

        u = torch.tensor([met], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, node_type=node_types_tensor)



class HEPGraphDatasetBuilder:
    def __init__(self, config: GraphConfig, max_workers=10):
        self.paths = config.file_paths
        self.sample_name = config.config.get("name", "output")
        self.builder = HEPEventGraphBuilder()
        self.output_dir = Path(self.sample_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def _extract_tree_number(self, path: str) -> str:
        filename = os.path.basename(path)
        stem = filename.replace(".root", "")
        return stem.split("_")[-1]
    
    def _process_single_file(self, path: str) -> None:
        graphs = []
        skimmer = HEPSkimmer()
        try:
            with uproot.open(path) as f:
                events = f["Events"].arrays()
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return

        for i in tqdm(range(len(events)), desc=f"Events in {os.path.basename(path)}", unit="event",leave=True):
            event = events[i]
            if not skimmer.event_passes(event):
                continue
            graph = self.builder.event_to_graph(event)
            if graph is not None:
                graphs.append(graph)

        tree_number = self._extract_tree_number(path)
        out_file = self.output_dir / f"tree_{tree_number}.pt"
        torch.save(graphs, out_file)

    def process_files(self):
        print(f"Processing {len(self.paths)} files with up to {self.max_workers} workers...")

        with Pool(processes=self.max_workers) as pool:
            list(tqdm(pool.imap(self._process_single_file, self.paths), total=len(self.paths), desc="Files", unit="file"))

