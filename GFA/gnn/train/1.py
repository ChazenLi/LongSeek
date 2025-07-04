import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(GNNModel, self).__init__()
        
        # Graph Convolution Layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        # Final prediction layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_attr, batch=None):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Fourth GCN layer
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Global mean pooling
        if batch is not None:
            x = torch.mean(x, dim=0)
        
        # Final prediction
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def load_and_preprocess_data(pt_dir):
    """Load all .pt files and convert them to PyG Data objects"""
    dataset = []
    for pt_file in glob.glob(os.path.join(pt_dir, "*.pt")):
        try:
            data = torch.load(pt_file)
            # Create PyG Data object
            graph_data = Data(
                x=data['x'],
                edge_index=data['edge_index'],
                edge_attr=data['edge_attr']
            )
            dataset.append(graph_data)
        except Exception as e:
            print(f"Error loading {pt_file}: {str(e)}")
    return dataset

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        # For this example, we'll use mean node features as target
        # Replace this with your actual target values
        target = torch.mean(data.x, dim=1).to(device)
        
        loss = F.mse_loss(output.squeeze(), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            target = torch.mean(data.x, dim=1).to(device)
            loss = F.mse_loss(output.squeeze(), target)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    pt_dir = "E:/APTAMER-GEN/pt"  # Update this path to your pt files directory
    dataset = load_and_preprocess_data(pt_dir)
    
    # Split dataset
    train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    # Initialize model
    input_dim = dataset[0].x.size(1)  # Number of node features
    model = GNNModel(input_dim=input_dim).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Training parameters
    n_epochs = 200
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # Training loop
    for epoch in range(n_epochs):
        train_loss = train_model(model, train_loader, optimizer, device)
        test_loss = evaluate_model(model, test_loader, device)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}')
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
    
    print("Training completed!")
    print(f"Best test loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()