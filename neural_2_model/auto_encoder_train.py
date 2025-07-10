import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os


class AutoEncoder(nn.Module):
    """
    Autoencoder for dimensionality reduction of neural signals.
    
    This autoencoder compresses high-dimensional neural data (271,633 dimensions)
    to match GPT's hidden layer dimensionality (768 dimensions) as described in
    the methodology section.
    
    Architecture:
        - Encoder: Linear layer (271,633 -> 768)
        - Decoder: Linear layer with Sigmoid activation (768 -> 271,633)
    """
    
    def __init__(self, input_dim=271633, hidden_dim=768):
        super(AutoEncoder, self).__init__()
        
        # Encoder: compress from high-dimensional neural data to 768d
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )
        
        # Decoder: reconstruct from 768d back to original dimensions
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            encoded: Encoded representation (batch_size, hidden_dim)
            decoded: Reconstructed data (batch_size, input_dim)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def load_subject_data():
    """
    Load subject IDs and experimental data.
    
    Returns:
        subjects: List of subject IDs
        id_2_dur: Dictionary mapping trial IDs to sound duration
        id_2_trail_type: Dictionary mapping trial IDs to trial types
    """
    # Load subject information
    read_file = pd.read_excel('/make_tsv/fmri_map_id.xlsx')
    subjects = list(read_file['BIDS_ID'])
    
    # Load experimental material data
    df_ma = pd.read_excel('/make_tsv/MATERIAL.xlsx')
    id_2_dur = dict(zip(df_ma['TrialID'], df_ma['soundDur']))
    
    # Construct trial sentences from word columns
    trail_list = [str(df_ma['word1'][i]) + ' ' + str(df_ma['word2'][i]) + ' ' + 
                  str(df_ma['word3'][i]) + ' ' + str(df_ma['word4'][i]) + ' ' + 
                  str(df_ma['word5'][i]) + ' ' + str(df_ma['word6'][i]) + ' ' + 
                  str(df_ma['word7'][i]) + ' ' + str(df_ma['word8'][i])
                  for i in range(len(df_ma))]
    
    # Clean sentences by removing 'nan' entries
    trail_list = [sentence.replace("nan", "").strip() for sentence in trail_list]
    
    # Remove duplicate words from sentences
    def remove_duplicates(sentence):
        words = sentence.split()
        for i in range(7):
            if len(words) >= 2 and words[-1] == words[-2]:
                words.pop(-1)
        return ' '.join(words)
    
    new_list = [remove_duplicates(sentence) for sentence in trail_list]
    id_2_trail_type = dict(zip(list(df_ma['TrialID']), new_list))
    
    return subjects, id_2_dur, id_2_trail_type


def process_trial_data():
    """
    Process trial data to handle sentence ordering and counting.
    
    Returns:
        count_new_list: List of processed sentences with counts
    """
    # Load reference sentences from file
    file_path = '/home/nllsgyang/Documents/language_learning/output.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    item_listxt = [line.strip() for line in lines]
    
    # Separate sentences based on reference list
    subjects, _, id_2_trail_type = load_subject_data()
    new_list = list(id_2_trail_type.values())
    
    new_list_co = []
    new_list_unc = []
    for sentence in new_list:
        if sentence in item_listxt:
            new_list_co.append(sentence)
        else:
            new_list_unc.append(sentence)
    
    new_list = new_list_co + new_list_unc
    
    # Generate sentence counts
    count_dict = {}
    count_new_list = []
    
    for sentence in new_list:
        if sentence in count_dict:
            count_dict[sentence] += 1
        else:
            count_dict[sentence] = 0
        
        count_new_list.append(f"{sentence} {count_dict[sentence]}")
    
    return count_new_list


def train_autoencoder(subjects, neural_data_path, output_path):
    """
    Train autoencoder using leave-one-out cross-validation.
    
    Args:
        subjects: List of subject IDs
        neural_data_path: Path to neural data files
        output_path: Path to save trained models and encoded data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training parameters
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001
    patience = 3
    
    for sub_id, subject in enumerate(subjects):
        print(f"Processing subject: {subject}")
        
        # Leave-one-out cross-validation setup
        test_indices = [sub_id]
        train_indices = list(range(len(subjects)))
        train_indices.remove(sub_id)
        
        # Initialize model, loss function, and optimizer
        autoencoder = AutoEncoder().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
        
        # Training tracking variables
        autoencoder_list = []
        test_loss_list = []
        encoded_np_list = []
        save_mark = 0
        no_improve_count = 0
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train on all subjects except the test subject
            for idx in train_indices:
                # Load neural data for current training subject
                neural_data_file = f"{neural_data_path}/sub-{subjects[idx]}_neural_vector.npy"
                vector = torch.Tensor(np.load(neural_data_file)).to(device)
                
                # Create data loader for batch processing
                dataset = torch.utils.data.TensorDataset(vector)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # Training step
                for batch in dataloader:
                    inputs = batch[0].to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass
                    encoded, decoded = autoencoder(inputs)
                    
                    # Calculate reconstruction loss
                    loss = criterion(decoded, inputs)
                    loss.backward()
                    optimizer.step()
            
            # Save model state for this epoch
            autoencoder_list.append(autoencoder.state_dict())
            print(f"Training loss: {loss.item():.4f}")
            
            # Evaluate on test subject
            with torch.no_grad():
                test_neural_file = f"{neural_data_path}/sub-{subject}_neural_vector.npy"
                test_vector = torch.Tensor(np.load(test_neural_file)).to(device)
                
                encoded, decoded = autoencoder(test_vector)
                test_loss = criterion(decoded, test_vector)
                
                encoded_np = encoded.detach().cpu().numpy()
                encoded_np_list.append(encoded_np)
                test_loss_list.append(test_loss.item())
                
                print(f"Test loss: {test_loss.item():.4f}")
            
            # Early stopping based on test loss
            if epoch > 0 and test_loss.item() > min(test_loss_list):
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    min_loss_index = test_loss_list.index(min(test_loss_list))
                    save_best_model(subject, encoded_np_list[min_loss_index], 
                                   autoencoder_list[min_loss_index], output_path)
                    save_mark = 1
                    break
            else:
                no_improve_count = 0
        
        # Save best model if early stopping was not triggered
        if save_mark == 0:
            min_loss_index = test_loss_list.index(min(test_loss_list))
            save_best_model(subject, encoded_np_list[min_loss_index], 
                           autoencoder_list[min_loss_index], output_path)


def save_best_model(subject, encoded_data, model_state, output_path):
    """
    Save the best model and encoded data for a subject.
    
    Args:
        subject: Subject ID
        encoded_data: Encoded neural data
        model_state: Model state dictionary
        output_path: Base output path
    """
    subject_output_dir = f"{output_path}/neural_data_all_brain_auto_encoder_768_without_sub_{subject}"
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # Save encoded neural data
    encoded_file = f"{subject_output_dir}/sub-{subject}_neural_vector.npy"
    np.save(encoded_file, encoded_data)
    
    # Save autoencoder model
    model_file = f"{subject_output_dir}/autoencoder.pth"
    torch.save(model_state, model_file)
    
    print(f"Saved results for subject {subject}")


def main():
    """
    Main function to execute the dimensionality reduction pipeline.
    """
    # Load subject data
    subjects, id_2_dur, id_2_trail_type = load_subject_data()
    
    # Process trial data
    count_new_list = process_trial_data()
    
    # Set paths
    neural_data_path = "neural_data_all_brain_no_reduced_mni"
    output_path = "neural_data_leave_one_valid_mni"
    
    # Train autoencoder for dimensionality reduction
    train_autoencoder(subjects, neural_data_path, output_path)
    
    print("Dimensionality reduction completed successfully.")


if __name__ == "__main__":
    main()