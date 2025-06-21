import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import os, random,glob
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

def load_graphs_from_dir(path, label):
    
    files = glob.glob(f"{path}/*.pt")
    all_graphs = []
    for f in files:
        data_list = torch.load(f)
        for graph in data_list:
            graph.y = torch.tensor([label], dtype=torch.long)
            all_graphs.append(graph)
    return all_graphs


def load_graphs(signal_path, background_path):
    signal_data = []
    background_data = []
    signal_files = glob.glob(f"{signal_path}/*.pt")
    background_files = glob.glob(f"{background_path}/*.pt")
    for f in tqdm(signal_files,desc="Signal Files",unit="file",leave=False) :
        data_list = torch.load(f,weights_only=False)
        for g in tqdm(data_list,desc="Signal Graphs",unit="graph",leave=False) :
            g.y = torch.tensor([1], dtype=torch.float) 
            signal_data.append(g)
    print(len(signal_data))
    for f in tqdm(background_files,desc="Background Files",unit="file",leave=False) :
        data_list = torch.load(f,weights_only=False)
        for g in tqdm(data_list,desc="Background Graphs",unit="graph",leave=False)  :
            g.y = torch.tensor([0], dtype=torch.float) 
            background_data.append(g)

    full_data = signal_data + background_data
    random.shuffle(full_data)

    return full_data


def load_graphs_dirlist(signal_paths, background_paths):
    signal_data = []
    background_data = []
    for signal_path in signal_paths :
        signal_files = glob.glob(f"{signal_path}/*.pt")
        for f in tqdm(signal_files,desc="Signal Files",unit="file",leave=True) :
            data_list = torch.load(f,weights_only=False)
            for g in tqdm(data_list,desc="Signal Graphs",unit="graph",leave=True) :
                g.y = torch.tensor([1], dtype=torch.float) 
                signal_data.append(g)
    for background_path in background_paths :
        background_files = glob.glob(f"{background_path}/*.pt")
        for f in tqdm(background_files,desc="Background Files",unit="file",leave=True) :
            data_list = torch.load(f,weights_only=False)
            for g in tqdm(data_list,desc="Background Graphs",unit="graph",leave=True)  :
                g.y = torch.tensor([0], dtype=torch.float) 
                background_data.append(g)

    full_data = signal_data + background_data
    random.shuffle(full_data)

    return full_data


class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze(-1)

def train(model, loader, optimizer, criterion, epoch, device):
    model.train()
    total_loss = 0
    batch_bar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    for batch in batch_bar:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_bar.set_postfix(loss=loss.item())
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = torch.sigmoid(out).cpu().numpy()
        label = batch.y.cpu().numpy()
        preds.extend(pred)
        labels.extend(label)

    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, preds > 0.5)
    return acc, auc


def main(): 
    signal_path = "/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoTauTauJJ_WR3000_N200"
    background_path = "/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/TTLJ_powheg"
    
    log_dir = "runs/SimpleGNN_LR1em3_hDim32L4"
    os.makedirs(log_dir,exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    l_signal_path = ["/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoTauTauJJ_WR3000_N200",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N400",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N600",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N800",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N1000",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N1200",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N1400",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N1600",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N1800",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N2000",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N2200",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N2400",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N2600",
                     ]
    l_background_path = ["/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/TTJJ_powheg",
                         #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/TTLJ_powheg",
                         #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/TTLL_powheg",
                         #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WJets_MG",
                         #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/DYJets_MG"
                         ]

    all_data = load_graphs_dirlist(l_signal_path, l_background_path)
    print(f"Loaded {len(all_data)} graphs")

    # Split
    train_data = all_data[:int(0.8 * len(all_data))]
    test_data = all_data[int(0.8 * len(all_data)):]

    randomgraph = random.choice(test_data)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    input_dim = train_data[0].x.shape[1]  # e.g. 5+1
    model = SimpleGNN(input_dim=input_dim).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Class balancing if signal is rare
    n_pos = sum(g.y.item() == 1 for g in train_data)
    n_neg = len(train_data) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to("cuda")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Train
    for epoch in range(1, 51):
        loss = train(model, train_loader, optimizer, criterion, epoch, device="cuda")
        writer.add_scalar("Loss/train", loss,epoch)
        acc, auc = evaluate(model, test_loader, device="cuda")
        print(f"[Epoch {epoch:02d}] Loss: {loss:.4f}, Acc: {acc:.3f}, AUC: {auc:.3f}")
        for name, param in model.named_parameters():
            writer.add_histogram(f"Weights/{name}", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
        writer.add_graph( model )

if __name__ == "__main__":
    main()
