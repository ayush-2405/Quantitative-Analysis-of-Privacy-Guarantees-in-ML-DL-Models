import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset 
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.preprocessing import StandardScaler 
from opacus import PrivacyEngine 
from sklearn.metrics.pairwise import cosine_similarity 
 
###########################################################################
 #### 
# 1) LOAD AND PREPROCESS DATA 
###########################################################################
 #### 
def load_and_preprocess_data(csv_path): 
    # Load the dataset 
    data = pd.read_csv(csv_path) 
     
    # Check for missing values and handle them 
    # Replace zeros with NaN in certain columns where zero doesn't make 
sense 
    cols_to_replace_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 
'Insulin', 'BMI'] 
    for col in cols_to_replace_zeros: 
        data[col] = data[col].replace(0, np.nan) 
     
    # Fill NaN values with the median of each column 
    for col in cols_to_replace_zeros: 
        data[col] = data[col].fillna(data[col].median()) 
     
    # Separate features and target 
    X = data.drop(columns=['Outcome']).values 
    y = data['Outcome'].values 
     
    # Standardize the features 
    scaler = StandardScaler() 
    X = scaler.fit_transform(X) 
     
    return X.astype(np.float32), y.astype(np.int64) 
 
###########################################################################
 #### 
# 2) TORCH DATASET 
###########################################################################
 #### 
class MedicalDataset(Dataset): 
    def __init__(self, X, y): 
        self.X = torch.from_numpy(X) 
        self.y = torch.from_numpy(y) 
 
    def __len__(self): 
        return len(self.X) 
 
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx] 
 
###########################################################################
 #### 
# 3) BUILD A SIMPLE MLP WITH PYTORCH 
###########################################################################
 #### 
class DPCompatibleMLP(nn.Module): 
    def __init__(self, input_dim, num_classes=2): 
        super().__init__() 
        self.fc1 = nn.Linear(input_dim, 256, bias=False) 
        self.fc2 = nn.Linear(256, 128, bias=False) 
        self.fc3 = nn.Linear(128, 64, bias=False) 
        self.fc4 = nn.Linear(64, num_classes, bias=False) 
        self.dropout = nn.Dropout(0.3) 
 
    def forward(self, x): 
        x = torch.relu(self.fc1(x)) 
        x = self.dropout(x) 
        x = torch.relu(self.fc2(x)) 
        x = self.dropout(x) 
        x = torch.relu(self.fc3(x)) 
        x = self.dropout(x) 
        x = self.fc4(x) 
        return x 
 
###########################################################################
 #### 
# 4) TRAIN WITH DP-SGD (Opacus) AND EVALUATE 
###########################################################################
 #### 
def train_dp_sgd(X, y, epochs=10, batch_size=128, lr=0.01,  
                max_grad_norm=1.0, noise_multiplier=1.1, delta=1e-5): 
    # Data preparation 
    N = len(X) 
    indices = np.arange(N) 
    np.random.shuffle(indices) 
    X, y = X[indices], y[indices] 
    split = int(0.8 * N) 
    X_tr, X_te = X[:split], X[split:] 
    y_tr, y_te = y[:split], y[split:] 
     
    train_ds = MedicalDataset(X_tr, y_tr) 
    test_ds = MedicalDataset(X_te, y_te) 
    train_loader = DataLoader(train_ds, batch_size=batch_size, 
shuffle=True) 
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) 
 
    # Model initialization 
    input_dim = X.shape[1] 
    num_classes = len(np.unique(y)) 
    model = DPCompatibleMLP(input_dim, num_classes) 
     
    # Opacus setup 
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
    privacy_engine = PrivacyEngine() 
     
    # Convert to DP model 
    dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private( 
        module=model, 
        optimizer=optimizer, 
        data_loader=train_loader, 
        noise_multiplier=noise_multiplier, 
        max_grad_norm=max_grad_norm, 
    ) 
 
    # Training loop 
    criterion = nn.CrossEntropyLoss() 
    dp_model.train() 
     
    for epoch in range(epochs): 
        running_loss = 0.0 
        correct = 0 
        total = 0 
         
        for X_batch, y_batch in dp_train_loader: 
            dp_optimizer.zero_grad() 
            outputs = dp_model(X_batch) 
            loss = criterion(outputs, y_batch) 
            loss.backward() 
            dp_optimizer.step() 
             
            running_loss += loss.item() * y_batch.size(0) 
            _, preds = torch.max(outputs, dim=1) 
            correct += (preds == y_batch).sum().item() 
            total += y_batch.size(0) 
 
        # Print metrics 
        epsilon = privacy_engine.get_epsilon(delta) 
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/total:.4f} | 
" 
              f"Acc: {correct/total:.4f} | ε: {epsilon:.2f}") 
 
    final_model = DPCompatibleMLP(input_dim, num_classes) 
    final_model.load_state_dict(dp_model._module.state_dict()) 
     
    return final_model 
 
###########################################################################
 #### 
# 4) TRAIN NON-DP MODEL 
###########################################################################
 #### 
def train_non_dp_sgd(X, y, epochs=10, batch_size=128, lr=0.01): 
    N = len(X) 
    indices = np.arange(N) 
    np.random.shuffle(indices) 
    X, y = X[indices], y[indices] 
    split = int(0.8*N) 
    X_tr, X_te = X[:split], X[split:] 
    y_tr, y_te = y[:split], y[split:] 
     
    train_ds = MedicalDataset(X_tr, y_tr) 
    test_ds = MedicalDataset(X_te, y_te) 
    train_loader = DataLoader(train_ds, batch_size=batch_size, 
shuffle=True) 
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) 
 
    input_dim = X.shape[1] 
    model = DeeperMLP(input_dim) 
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
    criterion = nn.CrossEntropyLoss() 
 
    model.train() 
    for epoch in range(epochs): 
        running_loss = 0.0 
        correct = 0 
        total = 0 
        for X_batch, y_batch in train_loader: 
            optimizer.zero_grad() 
            outputs = model(X_batch) 
            loss = criterion(outputs, y_batch) 
            loss.backward() 
            optimizer.step() 
            running_loss += loss.item() * y_batch.size(0) 
            _, preds = torch.max(outputs, dim=1) 
            correct += (preds == y_batch).sum().item() 
            total += y_batch.size(0) 
         
        train_loss = running_loss / total 
        train_acc = correct / total 
        print(f"Non-DP Epoch {epoch+1}/{epochs}, Loss={train_loss:.4f}, 
Acc={train_acc:.4f}") 
 
    model.eval() 
    correct_test = 0 
    total_test = 0 
    with torch.no_grad(): 
        for X_batch, y_batch in test_loader: 
            outputs = model(X_batch) 
            _, preds = torch.max(outputs, dim=1) 
            correct_test += (preds == y_batch).sum().item() 
            total_test += y_batch.size(0) 
     
    test_acc = correct_test / total_test 
    print(f"Non-DP Final Test Acc={test_acc:.4f}") 
    return model 
 
###########################################################################
 #### 
# 5) RECONSTRUCTION ATTACK IMPLEMENTATION 
###########################################################################
 #### 
class ReconstructionAttack: 
    def __init__(self, model, input_dim): 
        self.model = model.eval()  # Completely clean PyTorch model 
        self.input_dim = input_dim 
         
    def attack(self, target_labels, num_samples=100, epochs=1000, lr=0.1): 
        synthetic_data = torch.randn((num_samples, self.input_dim),  
                                   requires_grad=True) 
        optimizer = optim.Adam([synthetic_data], lr=lr) 
        criterion = nn.CrossEntropyLoss() 
         
        for _ in range(epochs): 
            optimizer.zero_grad() 
            outputs = self.model(synthetic_data) 
            loss = criterion(outputs, target_labels) 
            loss.backward()  # Now uses pure PyTorch autograd 
            optimizer.step() 
             
        return synthetic_data.detach().numpy() 
 
def compare_reconstructions(original_data, dp_recon, non_dp_recon): 
    # Calculate metrics 
    dp_mse = np.mean((original_data - dp_recon) ** 2) 
    non_dp_mse = np.mean((original_data - non_dp_recon) ** 2) 
     
    dp_cos = np.diag(cosine_similarity(original_data, dp_recon)).mean() 
    non_dp_cos = np.diag(cosine_similarity(original_data, 
non_dp_recon)).mean() 
 
    # Print results 
    print("\nReconstruction Comparison:") 
    print(f"{'Metric':<20} {'DP Model':<15} {'Non-DP Model':<15}") 
    print(f"{'MSE':<20} {dp_mse:.4f}{'':<10} {non_dp_mse:.4f}") 
    print(f"{'Cosine Similarity':<20} {dp_cos:.4f}{'':<10} 
{non_dp_cos:.4f}") 
 
    # Plot feature comparison 
    plt.figure(figsize=(12, 6)) 
    for i in range(3):  # Plot first 3 features 
        plt.subplot(1, 3, i+1) 
        plt.scatter(original_data[:,i], dp_recon[:,i], alpha=0.5, 
label='DP') 
        plt.scatter(original_data[:,i], non_dp_recon[:,i], alpha=0.5, 
label='Non-DP') 
        plt.plot([-3,3], [-3,3], 'k--') 
        plt.xlabel('Original') 
        plt.ylabel('Reconstructed') 
        plt.title(f'Feature {i+1}') 
        if i == 0: 
            plt.legend() 
    plt.tight_layout() 
    plt.show() 
 
###########################################################################
 #### 
# 6) UPDATED MAIN FUNCTION 
###########################################################################
 #### 
def main(): 
    csv_path = "diabetes (3).csv" 
    X, y = load_and_preprocess_data(csv_path) 
     
    # Train both models 
    print("Training DP Model...") 
    dp_model = train_dp_sgd(X, y, epochs=10, batch_size=64,  
                           lr=0.001, max_grad_norm=1.0, 
                           noise_multiplier=1.3, delta=1e-5) 
     
    print("\nTraining Non-DP Model...") 
    non_dp_model = train_non_dp_sgd(X, y, epochs=10, batch_size=64, 
lr=0.001) 
 
    # Prepare attack data (use first 100 training samples) 
    attack_data = X[:100] 
    attack_labels = y[:100] 
     
    # Perform reconstruction attacks 
    print("\nPerforming Reconstruction Attacks...") 
    # Perform attacks 
    dp_attacker = ReconstructionAttack(dp_model, X.shape[1]) 
    non_dp_attacker = ReconstructionAttack(non_dp_model, X.shape[1]) 
     
    dp_recon = dp_attacker.attack(torch.tensor(attack_labels)) 
    non_dp_recon = non_dp_attacker.attack(torch.tensor(attack_labels)) 
     
    # Compare results 
    compare_reconstructions(attack_data, dp_recon, non_dp_recon) 
 
if __name__ == "__main__": 
    main()