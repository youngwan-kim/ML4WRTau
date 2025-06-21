import os
import torch
import random
import argparse
from tqdm import tqdm
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm
from torch_geometric.nn import global_mean_pool, global_max_pool ,GATv2Conv
from torch.utils.tensorboard import SummaryWriter
from visualize_attention import *

class GATClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=0.2)
        self.norm1 = LayerNorm(hidden_dim * heads)

        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.2)
        self.norm2 = LayerNorm(hidden_dim)


        self.lin = Linear(hidden_dim * 2, 1)

    def forward(self, x, edge_index, batch, return_attention=False):
        attn_weights = {}
    
        if return_attention:
            x, (edge_index1, alpha1) = self.conv1(x, edge_index, return_attention_weights=True)
            attn_weights['conv1'] = (edge_index1, alpha1)
        else:
            x = self.conv1(x, edge_index)
    
        x = self.norm1(x)
        x = F.elu(x)
    
        if return_attention:
            x, (edge_index2, alpha2) = self.conv2(x, edge_index, return_attention_weights=True)
            attn_weights['conv2'] = (edge_index2, alpha2)
        else:
            x = self.conv2(x, edge_index)
    
        x = self.norm2(x)
        x = F.elu(x)
    
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
    
        x = self.lin(x)
        if return_attention:
            return x.view(-1), attn_weights
        else:
            return x.view(-1)

 
def load_graphs_from_dir(directory):
    all_graphs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            path = os.path.join(directory, filename)
            try:
                graphs = torch.load(path,weights_only=False)
                all_graphs.extend(graphs)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    return all_graphs

def load_graphs_from_dirlist(directory_list):
    all_graphs = []
    for directory in directory_list :
        for filename in tqdm(os.listdir(directory),desc="Files",unit="file",leave=True):
            if filename.endswith(".pt"):
                path = os.path.join(directory, filename)
                try:
                    graphs = torch.load(path,weights_only=False)
                    all_graphs.extend(graphs)
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
    return all_graphs

 
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l_signal_path = ["/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoTauTauJJ_WR3000_N200",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N400",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N600",
                     "/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N800",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N1000",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N1200",
                     "/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N1400",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N1600",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N1800",
                     "/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N2000",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N2200",
                     #"/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N2400",
                     "/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WRtoNTautoTauTauJJ_WR3000_N2600",]
    l_background_path = ["/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/TTJJ_powheg",
                         "/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/TTLJ_powheg",
                         "/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/TTLL_powheg",
                         "/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/WJets_MG",
                         "/data6/Users/youngwan/SKNanoAnalyzer/MLSandbox/DYJets_MG"]

    print("Loading signal and background graphs...")
    signal_graphs_ = load_graphs_from_dirlist(l_signal_path)
    background_graphs_ = load_graphs_from_dirlist(l_background_path)

    signal_graphs,background_graphs = [],[]

    for g in tqdm(signal_graphs_, desc="Labeling signal"):
        g.y = torch.tensor([1.0], dtype=torch.float)
        signal_graphs.append(g)

    for g in tqdm(background_graphs_, desc="Labeling background"):
        g.y = torch.tensor([0.0], dtype=torch.float)
        background_graphs.append(g)

    all_graphs = signal_graphs + background_graphs
    random.shuffle(all_graphs)

    split_idx = int(len(all_graphs) * 0.8)
    train_graphs = all_graphs[:split_idx]
    val_graphs = all_graphs[split_idx:]

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)

    input_dim = train_graphs[0].x.shape[1]
    model = GATClassifier(input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    tag = "GraphAttentionV2_LR1em5_hDim32head4"
    log_dir = f"runs/{tag}"
    model_dir = f"models/{tag}"
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(model_dir,exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model_path = f"{model_dir}/gat_epoch_{epoch+1}.pth"
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}",leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        writer.add_scalar("Loss/train",total_loss / len(train_loader),epoch)
        print(f"Epoch {epoch} Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = torch.sigmoid(model(batch.x, batch.edge_index, batch.batch))
                predicted = (pred > 0.5).float()
                correct += (predicted == batch.y).sum().item()
                total += batch.y.size(0)
        acc = correct / total if total > 0 else 0
        writer.add_scalar("Accuracy/Validation",acc,epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(f"Weights/{name}", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
        print(f"Validation Accuracy: {acc:.4f}")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model at: {model_path}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--signal_dir", type=str, required=True, help="Directory with signal .pt files")
    #parser.add_argument("--background_dir", type=str, required=True, help="Directory with background .pt files")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()
    main(args)
