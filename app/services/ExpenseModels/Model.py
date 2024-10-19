import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

class ExpenseDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length - 10
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + 10]
        return x, y
# Load the saved model and scaler
def save_model(model, scaler, save_dir='saved_models'):
    """Save the trained model and scaler"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the PyTorch model
    model_path = os.path.join(save_dir, 'expense_predictor.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save the scaler
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
class ExpensePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_length):
        super(ExpensePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_length = output_length
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Changed to output the same number of features for each prediction
        self.fc = nn.Linear(hidden_size, output_length * input_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] 
        out = self.fc(out)
        # Reshape output to match target dimensions: [batch_size, sequence_length, features]
        out = out.view(batch_size, self.output_length, self.input_size)
        return out

def prepare_data(df, date_column='date', amount_column='amount'):
    # Convert date strings to datetime objects and extract features
    df[date_column] = pd.to_datetime(df[date_column])
    df['hour'] = df[date_column].dt.hour
    df['day'] = df[date_column].dt.day
    df['month'] = df[date_column].dt.month
    df['day_of_week'] = df[date_column].dt.dayofweek
    
    # Scale the features
    scaler = MinMaxScaler()
    features = ['hour', 'day', 'month', 'day_of_week', amount_column]
    scaled_data = scaler.fit_transform(df[features])
    
    return scaled_data, scaler

def train_model(model, train_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def predict_future_expenses(model, last_sequence, scaler, device):
    model.eval()
    with torch.no_grad():
        input_seq = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        predictions = model(input_seq)
        
        # Get predictions back to CPU and numpy
        predictions = predictions.cpu().numpy()[0]  # Remove batch dimension
        
        # Inverse transform the entire sequence
        inversed_predictions = scaler.inverse_transform(predictions)
        
        # Extract only the amount predictions (last column)
        amount_predictions = inversed_predictions[:, -1]
        
        return amount_predictions

def main():
    # Hyperparameters
    sequence_length = 20
    hidden_size = 64
    num_layers = 2
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    prediction_length = 10
    
    # Example data creation for demonstration purposes (will replace with actual data)
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='h')
    df = pd.DataFrame({
        'date': dates,
        'amount': np.random.normal(1000, 200, len(dates))
    })
    
    # Prepare data
    scaled_data, scaler = prepare_data(df)
    input_size = scaled_data.shape[1]  # Number of features
    
    # Create dataset and dataloader
    dataset = ExpenseDataset(scaled_data, sequence_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExpensePredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_length=prediction_length
    ).to(device)
    
    # Train model
    train_model(model, train_loader, num_epochs, learning_rate, device)
    save_model(model, scaler)
    # Make predictions
    last_sequence = scaled_data[-sequence_length:]
    predictions = predict_future_expenses(model, last_sequence, scaler, device)
    print("Predicted next 10 expenses:", predictions)

if __name__ == "__main__":
    main()