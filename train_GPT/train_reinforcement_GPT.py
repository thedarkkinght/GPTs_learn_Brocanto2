import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from os.path import join
from tqdm.notebook import tqdm
from scipy.ndimage import gaussian_filter1d

# Transformers imports
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, GPT2ForSequenceClassification,
    AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    set_seed, AdamW, get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader

# ML utilities
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score

# Custom imports
from tokenise import BPE_token
from right_corpus import NP, NPVP, SOV


def setup_environment():
    """Initialize random seeds and device configuration for reproducibility."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_experimental_data():
    """Load and preprocess experimental data from Excel file."""
    # Load experimental materials
    df_ma = pd.read_excel('/home/nllsgyang/Documents/make_tsv/MATERIAL.xlsx')
    id_2_dur = dict(zip(df_ma['TrialID'], df_ma['soundDur']))
    
    # Construct trial sentences from word columns
    trail_list = [
        ' '.join([str(df_ma[f'word{i}'][j]) for i in range(1, 9)])
        for j in range(len(df_ma))
    ]
    
    # Remove 'nan' strings and extra whitespace
    trail_list = [sentence.replace("nan", "").strip() for sentence in trail_list]
    
    return trail_list, id_2_dur


def remove_duplicates(sentence):
    """Remove duplicate words from the end of sentences."""
    words = sentence.split()
    for i in range(7):
        if len(words) >= 2 and words[-1] == words[-2]:
            words.pop(-1)
    return ' '.join(words)


def load_corpus_data():
    """Load and process corpus data for training."""
    # Generate complete corpus
    all_corpus = NP + NPVP + SOV
    print(f"Total corpus size: {len(all_corpus)}")
    
    # Load experimental data
    trail_list, id_2_dur = load_experimental_data()
    
    # Process sentences to remove duplicates
    processed_list = [remove_duplicates(sentence) for sentence in trail_list]
    
    # Create unique sentence list
    unique_sentences = []
    for item in processed_list:
        if item not in unique_sentences:
            unique_sentences.append(item)
    
    # Load grammatical sentences from file
    file_path = '/home/nllsgyang/Documents/language_learning/output.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        grammatical_sentences = [line.strip() for line in file.readlines()]
    
    # Separate grammatical and ungrammatical sentences
    grammatical_list = []
    ungrammatical_list = []
    
    for sentence in processed_list:
        if sentence in grammatical_sentences:
            grammatical_list.append(sentence)
        else:
            ungrammatical_list.append(sentence)
    
    # Combine all sentences
    all_sentences = grammatical_list + ungrammatical_list
    
    # Create labels based on grammaticality
    labels = []
    for sentence in all_sentences:
        if sentence in all_corpus:
            labels.append('true')
        else:
            labels.append('false')
    
    return all_sentences, labels


def setup_tokenizer():
    """Initialize and configure the tokenizer."""
    tokenizer = BPE_token()
    save_path = 'tokenized_data'
    
    # Load pre-trained tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(save_path)
    
    # Add special tokens
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "</s>",
        "mask_token": "<mask>"
    })
    
    # Set padding side for GPT models
    tokenizer.padding_side = 'left'
    
    return tokenizer


def split_data(sentences, labels, train_ratio=0.8):
    """Split data into training and validation sets."""
    assert len(sentences) == len(labels), "Sentences and labels must have same length"
    
    # Calculate number of training samples
    num_train = int(train_ratio * len(sentences))
    
    # Generate random indices for training
    train_indices = random.sample(range(len(sentences)), num_train)
    
    # Get remaining indices for validation
    val_indices = list(set(range(len(sentences))) - set(train_indices))
    
    return train_indices, val_indices


class BrocantoDataset(Dataset):
    """PyTorch Dataset class for Brocanto2 language learning data."""
    
    def __init__(self, sentences, labels, indices):
        """
        Initialize dataset with sentences, labels, and indices.
        
        Args:
            sentences: List of sentence strings
            labels: List of corresponding labels
            indices: List of indices to use from sentences/labels
        """
        self.texts = [sentences[i] for i in indices]
        self.labels = [labels[i] for i in indices]
        self.n_examples = len(self.labels)
    
    def __len__(self):
        """Return number of examples in dataset."""
        return self.n_examples
    
    def __getitem__(self, item):
        """
        Get item by index.
        
        Args:
            item: Index of item to retrieve
            
        Returns:
            Dictionary containing text and label
        """
        return {
            'text': self.texts[item],
            'label': self.labels[item]
        }


class AllCorpusDataset(Dataset):
    """Dataset containing all grammatical Brocanto2 sentences."""
    
    def __init__(self):
        """Initialize with complete grammatical corpus."""
        all_corpus = NP + NPVP + SOV
        self.texts = all_corpus
        self.labels = ['true'] * len(all_corpus)
        self.n_examples = len(self.labels)
    
    def __len__(self):
        """Return number of examples in dataset."""
        return self.n_examples
    
    def __getitem__(self, item):
        """
        Get item by index.
        
        Args:
            item: Index of item to retrieve
            
        Returns:
            Dictionary containing text and label
        """
        return {
            'text': self.texts[item],
            'label': self.labels[item]
        }


class Gpt2ClassificationCollator:
    """
    Data collator for GPT2 classification tasks.
    
    Converts text and labels to tensors suitable for model input.
    """
    
    def __init__(self, tokenizer, labels_encoder, max_sequence_len=None):
        """
        Initialize collator.
        
        Args:
            tokenizer: Tokenizer for text processing
            labels_encoder: Dictionary mapping labels to integers
            max_sequence_len: Maximum sequence length for padding/truncation
        """
        self.tokenizer = tokenizer
        self.max_sequence_len = (
            tokenizer.model_max_length if max_sequence_len is None 
            else max_sequence_len
        )
        self.labels_encoder = labels_encoder
    
    def __call__(self, sequences):
        """
        Process batch of sequences.
        
        Args:
            sequences: List of dictionaries containing text and labels
            
        Returns:
            Dictionary of tensors ready for model input
        """
        # Extract texts and labels
        texts = [seq['text'] for seq in sequences]
        labels = [seq['label'] for seq in sequences]
        
        # Encode labels
        encoded_labels = [self.labels_encoder[label] for label in labels]
        
        # Tokenize texts
        inputs = self.tokenizer(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_sequence_len
        )
        
        # Add labels to inputs
        inputs.update({'labels': torch.tensor(encoded_labels)})
        
        return inputs


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Execute one training epoch.
    
    Args:
        model: GPT model for training
        dataloader: Training data loader
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler
        device: Device for tensor operations
        
    Returns:
        Tuple of (true_labels, predicted_labels, average_loss)
    """
    model.train()
    
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader), disable=True):
        # Store true labels for evaluation
        true_labels.extend(batch['labels'].numpy().flatten().tolist())
        
        # Move batch to device
        batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}
        
        # Clear gradients
        model.zero_grad()
        
        # Forward pass
        outputs = model(**batch)
        loss, logits = outputs[:2]
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        # Store predictions
        logits = logits.detach().cpu().numpy()
        predictions_labels.extend(logits.argmax(axis=-1).flatten().tolist())
    
    avg_loss = total_loss / len(dataloader)
    return true_labels, predictions_labels, avg_loss


def validate_epoch(model, dataloader, device):
    """
    Execute one validation epoch.
    
    Args:
        model: GPT model for validation
        dataloader: Validation data loader
        device: Device for tensor operations
        
    Returns:
        Tuple of (true_labels, predicted_labels, average_loss)
    """
    model.eval()
    
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), disable=True):
            # Store true labels for evaluation
            true_labels.extend(batch['labels'].numpy().flatten().tolist())
            
            # Move batch to device
            batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss, logits = outputs[:2]
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Store predictions
            logits = logits.detach().cpu().numpy()
            predictions_labels.extend(logits.argmax(axis=-1).flatten().tolist())
    
    avg_loss = total_loss / len(dataloader)
    return true_labels, predictions_labels, avg_loss


def plot_training_curves(train_acc, val_acc, save_path=None):
    """
    Plot training and validation accuracy curves.
    
    Args:
        train_acc: List of training accuracies
        val_acc: List of validation accuracies
        save_path: Optional path to save the plot
    """
    epochs = range(1, len(train_acc) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='red', linewidth=2)
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def extract_hidden_representations(model, tokenizer, sentences, device, max_length=25):
    """
    Extract hidden representations from trained model.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer for text processing
        sentences: List of sentences to process
        device: Device for tensor operations
        max_length: Maximum sequence length
        
    Returns:
        List of hidden representations for each layer
    """
    model.eval()
    model.to(device)
    
    # Initialize storage for each layer (embedding + 2 GPT blocks)
    hidden_representations = [[] for _ in range(3)]
    
    with torch.no_grad():
        for sentence in sentences:
            # Tokenize input
            inputs = tokenizer(
                text=sentence,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=max_length
            ).to(device)
            
            # Get model outputs with hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract hidden states from each layer
            for layer_idx in range(3):
                hidden_state = outputs.hidden_states[layer_idx]
                # Use the last token's hidden state
                last_hidden = hidden_state[:, -1, :][0]
                hidden_representations[layer_idx].append(last_hidden)
    
    # Convert to numpy arrays
    for i, layer_repr in enumerate(hidden_representations):
        hidden_representations[i] = torch.stack(layer_repr).detach().cpu().numpy()
    
    return hidden_representations


def main():
    """Main training and evaluation pipeline."""
    # Configuration
    CONFIG = {
        'layers': 2,
        'learning_rate': 5e-4,
        'max_length': 25,
        'epochs': 200,
        'batch_size': 16,
        'labels_ids': {'true': 0, 'false': 1}
    }
    
    # Setup
    device = setup_environment()
    sentences, labels = load_corpus_data()
    tokenizer = setup_tokenizer()
    
    # Split data
    train_indices, val_indices = split_data(sentences, labels, train_ratio=0.8)
    
    # Create datasets
    train_dataset = BrocantoDataset(sentences, labels, train_indices)
    val_dataset = BrocantoDataset(sentences, labels, val_indices)
    
    print(f'Created train_dataset with {len(train_dataset)} examples')
    print(f'Created val_dataset with {len(val_dataset)} examples')
    
    # Create data collator
    collator = Gpt2ClassificationCollator(
        tokenizer=tokenizer,
        labels_encoder=CONFIG['labels_ids'],
        max_sequence_len=CONFIG['max_length']
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collator
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collator
    )
    
    # Model configuration
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_layer=CONFIG['layers'],
        num_labels=len(CONFIG['labels_ids'])
    )
    
    # Initialize model
    model = GPT2ForSequenceClassification(model_config).to(device)
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], eps=1e-8)
    total_steps = len(train_dataloader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training tracking
    training_history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    print('Starting training...')
    
    # Training loop
    for epoch in tqdm(range(CONFIG['epochs'])):
        # Training
        train_labels, train_predictions, train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, device
        )
        train_acc = accuracy_score(train_labels, train_predictions)
        
        # Validation
        val_labels, val_predictions, val_loss = validate_epoch(
            model, val_dataloader, device
        )
        val_acc = accuracy_score(val_labels, val_predictions)
        
        # Log progress
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: "
              f"train_loss: {train_loss:.5f} - val_loss: {val_loss:.5f} - "
              f"train_acc: {train_acc:.5f} - val_acc: {val_acc:.5f}")
        
        # Store metrics
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        
        # Save model checkpoint
        output_dir = f"./models/classification_epoch_{epoch}_val_acc_{val_acc:.4f}"
        model.save_pretrained(output_dir)
    
    # Plot training curves
    plot_training_curves(
        training_history['train_acc'],
        training_history['val_acc'],
        save_path='./plots/training_curves.png'
    )
    
    print(f"Training completed. Best validation accuracy: {max(training_history['val_acc']):.4f}")
    
    # Extract hidden representations
    print("Extracting hidden representations...")
    hidden_reprs = extract_hidden_representations(model, tokenizer, sentences, device)
    
    # Save representations
    for i, layer_repr in enumerate(hidden_reprs):
        np.save(f'./representations/layer_{i}_representations.npy', layer_repr)
        print(f"Layer {i} representation shape: {layer_repr.shape}")


if __name__ == "__main__":
    main()