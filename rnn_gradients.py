import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def generate_long_term_dependency_data(seq_length, num_sequences):
    X = np.random.randn(num_sequences, seq_length, 1).astype(np.float32)
    y = (X[:, 0, 0] > 0).astype(np.float32)  # Target depends only on the first element
    return torch.from_numpy(X), torch.from_numpy(y)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hn = self.rnn(x)
        return torch.sigmoid(self.fc(hn.squeeze(0)))

def train_rnn(model, X, y, num_epochs, learning_rate):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    gradient_norms = []
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        
        grad_norms = []
        for name, param in model.named_parameters():
            if 'rnn' in name and 'weight' in name:
                grad_norms.append(param.grad.norm().item())
        gradient_norms.append(grad_norms)
        
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return gradient_norms, losses

def visualize_results(gradient_norms, losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    ax1.plot(gradient_norms)
    ax1.set_title('Gradient Norms Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Gradient Norm')
    ax1.set_yscale('log')
    ax1.legend(['Input-to-Hidden', 'Hidden-to-Hidden'])
    ax1.grid(True)
    
    ax2.plot(losses)
    ax2.set_title('Loss Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

seq_length = 1000
num_sequences = 1000
input_size = 1
hidden_size = 20
output_size = 1
num_epochs = 1000
learning_rate = 0.001

X, y = generate_long_term_dependency_data(seq_length, num_sequences)
model = SimpleRNN(input_size, hidden_size, output_size)
gradient_norms, losses = train_rnn(model, X, y, num_epochs, learning_rate)

visualize_results(gradient_norms, losses)

model.eval()
with torch.no_grad():
    predictions = (model(X).squeeze() > 0.5).float()
    accuracy = (predictions == y).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')