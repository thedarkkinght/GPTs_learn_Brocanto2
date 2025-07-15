from tokenise import BPE_token
from pathlib import Path
import os
import pandas as pd
from os.path import join
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import pickle
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    TextDataset, LineByLineTextDataset, set_seed,
    AutoTokenizer, GPT2ForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

# Import corpus generation functions
from right_corpus import NP, NPVP, SOV

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Configuration parameters
LAYERS = 2
LEARNING_RATE = 5e-4
TRAIN_EPOCHS = 200
MAX_LENGTH = 25
BATCH_SIZE = 20
TEMPERATURE = 1.0
VALIDATION_SPLIT = 0.2

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_brocanto2_data():
    """
    Load and preprocess Brocanto2 corpus data.
    
    Returns:
        tuple: (sentences, labels, train_data, val_data)
    """
    print("Loading Brocanto2 corpus...")
    print(f"Total corpus size: {len(NP + NPVP + SOV)}")
    
    # Load material data
    df_ma = pd.read_excel('/home/nllsgyang/Documents/make_tsv/MATERIAL.xlsx')
    id_2_dur = dict(zip(df_ma['TrialID'], df_ma['soundDur']))
    
    # Construct sentence list from material data
    trail_list = [
        str(df_ma['word1'][i]) + ' ' + str(df_ma['word2'][i]) + ' ' + 
        str(df_ma['word3'][i]) + ' ' + str(df_ma['word4'][i]) + ' ' + 
        str(df_ma['word5'][i]) + ' ' + str(df_ma['word6'][i]) + ' ' + 
        str(df_ma['word7'][i]) + ' ' + str(df_ma['word8'][i])
        for i in range(len(df_ma))
    ]
    
    # Clean sentences by removing NaN values
    trail_list = [sentence.replace("nan", "").strip() for sentence in trail_list]
    
    # Remove duplicate words within sentences
    def remove_duplicates(sentence):
        words = sentence.split()
        for i in range(7):
            if len(words) > 1 and words[-1] == words[-2]:
                words.pop(-1)
        return ' '.join(words)
    
    new_list = [remove_duplicates(sentence) for sentence in trail_list]
    
    # Create trial type mapping
    id_2_trail_type = dict(zip(list(df_ma['TrialID']), new_list))
    
    # Remove duplicates from sentence list
    new_new_list = []
    for item in new_list:
        if item not in new_new_list:
            new_new_list.append(item)
    
    # Load full corpus from file
    file_path = '/home/nllsgyang/Documents/language_learning/output.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    item_listxt = [line.strip() for line in lines]
    
    # Separate grammatical and ungrammatical sentences
    new_list_co = []
    new_list_unc = []
    for co in new_list:
        if co in NP + NPVP + SOV:
            new_list_co.append(co)
        else:
            new_list_unc.append(co)
    
    # Combine to maintain 1:1 ratio as specified in methodology
    new_list = new_list_co + new_list_unc
    
    # Generate labels (0 for grammatical, 1 for ungrammatical)
    y_train = []
    y_train_tensor = []
    for item in new_list:
        if item in NP + NPVP + SOV:
            y_train.append('true')
            y_train_tensor.append(torch.tensor(0).to(device))
        else:
            y_train.append('false')
            y_train_tensor.append(torch.tensor(1).to(device))
    
    # Train-validation split (80-20 as specified)
    assert len(new_list) == len(y_train)
    num_elements = int(0.8 * len(new_list))
    random_indices = random.sample(range(len(new_list)), num_elements)
    remaining_indices = list(set(range(len(new_list))) - set(random_indices))
    
    train_list = [new_list[i] for i in random_indices]
    train_labels = [y_train_tensor[i] for i in random_indices]
    val_list = [new_list[i] for i in remaining_indices]
    val_labels = [y_train_tensor[i] for i in remaining_indices]
    
    return new_list, y_train_tensor, train_list, train_labels, val_list, val_labels

def initialize_tokenizer():
    """
    Initialize and configure the BPE tokenizer for Brocanto2.
    
    Returns:
        GPT2Tokenizer: Configured tokenizer
    """
    tokenizer = BPE_token()
    save_path = 'tokenized_data'
    tokenizer = GPT2Tokenizer.from_pretrained(save_path)
    
    # Add special tokens
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>", 
        "unk_token": "<unk>",
        "pad_token": "</s>",
        "mask_token": "<mask>"
    })
    
    # Set padding side for GPT
    tokenizer.padding_side = 'left'
    
    return tokenizer

def create_gpt_model(tokenizer, model_type='integrated'):
    """
    Create GPT model with specified configuration.
    
    Args:
        tokenizer: Tokenizer instance
        model_type: Type of model ('prediction', 'reinforcement', 'integrated')
    
    Returns:
        tuple: (model, linear_classifier)
    """
    # Create configuration as specified in methodology
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_layer=LAYERS,
    )
    
    # Create model and move to device
    model = GPT2LMHeadModel(config).to(device)
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Create linear classifier for reinforcement/integrated models
    linear_net = nn.Linear(in_features=768, out_features=2, bias=True).to(device)
    
    return model, linear_net

def evaluate_generation_capability(model, tokenizer, test_dataset, temp=1.0):
    """
    Evaluate model's generation capability on NP, NPVP, and SOV constructions.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer instance
        test_dataset: Test dataset
        temp: Temperature for generation
    
    Returns:
        tuple: Evaluation metrics (accuracy, new_rate, new_generated, etc.)
    """
    np_generated = []
    npvp_generated = []
    sov_generated = []
    model.eval()
    correct_count = 0
    total_generated = 0
    new = 0
    cor = []
    cor_new = []
    
    # Load full corpus and training corpus
    with open("output.txt", "r", encoding="utf-8") as f:
        full_corpus = f.read().splitlines()
    with open("text/train_human.txt", "r", encoding="utf-8") as ff:
        train_corpus = ff.read().splitlines()
    
    # Calculate novel sentences (not in training)
    minus = [x for x in full_corpus if x not in train_corpus]
    
    for example in test_dataset:
        # Get first two tokens for generation seed
        first_token = example["input_ids"][0]
        second_token = example["input_ids"][1]
        
        # Construct generation input
        input_ids = torch.tensor([[0, first_token, second_token]]).to(model.device)
        
        new_mark = 0
        # Try different generation lengths
        for gen_len in range(5, 20):
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_length=gen_len,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(input_ids),
                    return_dict_in_generate=True,
                    temperature=temp,
                    do_sample=True
                )
            
            # Decode generated sequence
            generated_sequence = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True)
            
            # Check if generated sequence is in full corpus
            if generated_sequence in full_corpus:
                correct_count += 1
                cor.append(generated_sequence)
                
                # Check if novel (not in training)
                if generated_sequence in minus:
                    new += 1
                    new_mark = 1
                    cor_new.append(generated_sequence)
                
                # Categorize by construction type
                if generated_sequence in NP:
                    np_generated.append(generated_sequence)
                if generated_sequence in NPVP:
                    npvp_generated.append(generated_sequence)
                if generated_sequence in SOV:
                    sov_generated.append(generated_sequence)
                break
        
        # Try one more generation step if not novel
        if new_mark == 0 and generated_sequence in full_corpus:
            generated_ids = model.generate(
                input_ids,
                max_length=gen_len + 1,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(input_ids),
                return_dict_in_generate=True,
                temperature=temp,
                do_sample=True
            )
            generated_sequence = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True)
            
            if generated_sequence in minus:
                new += 1
                new_mark = 1
                cor.append(generated_sequence)
                cor_new.append(generated_sequence)
            
            # Categorize by construction type
            if generated_sequence in NP:
                np_generated.append(generated_sequence)
            if generated_sequence in NPVP:
                npvp_generated.append(generated_sequence)
            if generated_sequence in SOV:
                sov_generated.append(generated_sequence)
        
        total_generated += 1
        
        # Final categorization
        if generated_sequence in full_corpus:
            cor.append(generated_sequence)
        if generated_sequence in minus:
            cor_new.append(generated_sequence)
        if generated_sequence in NP:
            np_generated.append(generated_sequence)
        if generated_sequence in NPVP:
            npvp_generated.append(generated_sequence)
        if generated_sequence in SOV:
            sov_generated.append(generated_sequence)
    
    # Calculate metrics
    new_acc = new / (new + (total_generated - correct_count)) if (new + (total_generated - correct_count)) > 0 else 0
    accuracy = correct_count / total_generated if total_generated > 0 else 0
    new_generated = len(set(cor)) / total_generated if total_generated > 0 else 0
    new_no_re = len(set(cor_new)) / len(set(cor)) if len(set(cor)) > 0 else 0
    np_per = len(set(np_generated)) / len(set(cor)) if len(set(cor)) > 0 else 0
    npvp_per = len(set(npvp_generated)) / len(set(cor)) if len(set(cor)) > 0 else 0
    sov_per = len(set(sov_generated)) / len(set(cor)) if len(set(cor)) > 0 else 0
    
    return (accuracy, new_acc, new_generated, new_no_re, np_per, npvp_per, sov_per)

def train_integrated_model(model, linear_net, tokenizer, train_list, train_labels, val_list, val_labels):
    """
    Train the integrated prediction-reinforcement model.
    
    Args:
        model: GPT model
        linear_net: Linear classifier
        tokenizer: Tokenizer
        train_list: Training sentences
        train_labels: Training labels
        val_list: Validation sentences
        val_labels: Validation labels
    
    Returns:
        dict: Training results and metrics
    """
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Load test dataset for evaluation
    test_file = "output.txt"
    test_dataset_g = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=test_file,
        block_size=128
    )
    
    # Separate grammatical sentences for language modeling
    new_list_co = [sentence for sentence in train_list if sentence in NP + NPVP + SOV]
    
    # Tracking variables
    cla_acc_list = []
    loss_weight = []
    epochs_record = []
    
    # Evaluation result storage
    evaluation_results = []
    evaluation_new_results = []
    evaluation_new_hit_results = []
    evaluation_new_hit_gg = []
    evaluation_np_per = []
    evaluation_npvp_per = []
    evaluation_sov_per = []
    
    print("Starting training...")
    
    for epoch in range(TRAIN_EPOCHS):
        epoch_loss_weight = []
        loss_all = torch.zeros(1).to(device)
        lm_loss = torch.zeros(1).to(device)
        
        # Training loop
        for cor_id, corpus in enumerate(train_list):
            # Forward pass
            output = model(
                torch.tensor([tokenizer.encode(corpus)]).to(model.device),
                labels=torch.tensor([tokenizer.encode(corpus)]).to(device),
                output_hidden_states=True
            )
            
            # Classification loss
            model_label = linear_net(output.hidden_states[-1][-1][-1])
            loss_c = criterion(model_label.unsqueeze(0), train_labels[cor_id].unsqueeze(0))
            loss_all += loss_c
            
            # Language modeling loss (only for grammatical sentences)
            if corpus in new_list_co:
                lm_loss += output.loss
            
            # Update every 50 samples
            if ((cor_id + 1) % 50) == 0:
                # Dynamic loss weighting as specified in methodology
                if epoch > 0 and 'new_rate' in locals() and new_rate > cla_accuracy:
                    lm_weight = 2
                else:
                    lm_weight = np.log(79 / 2)  # Based on vocabulary size and binary classification
                
                # Combined loss
                combined_loss = lm_weight * lm_loss + loss_all
                epoch_loss_weight.append(lm_weight)
                
                # Backward pass
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()
                
                # Reset accumulated losses
                loss_all = torch.zeros(1).to(device)
                lm_loss = torch.zeros(1).to(device)
        
        # Record average loss weight for epoch
        if epoch_loss_weight:
            loss_weight.append(sum(epoch_loss_weight) / len(epoch_loss_weight))
        
        # Evaluation
        epochs_record.append(epoch)
        
        # Classification accuracy evaluation
        with torch.no_grad():
            correct_predictions = 0
            total_predictions = 0
            
            for cor_id, corpus in enumerate(val_list):
                output = model(
                    torch.tensor([tokenizer.encode(corpus)]).to(model.device),
                    output_hidden_states=True
                )
                model_label = linear_net(output.hidden_states[-1][-1][-1])
                
                _, predicted_label = torch.max(model_label, dim=0)
                true_label = val_labels[cor_id].item()
                
                if predicted_label.item() == true_label:
                    correct_predictions += 1
                total_predictions += 1
        
        # Calculate classification accuracy
        cla_accuracy = correct_predictions / total_predictions if total_predictions else 0
        cla_acc_list.append(cla_accuracy)
        
        # Generation evaluation
        eval_result = evaluate_generation_capability(model, tokenizer, test_dataset_g, TEMPERATURE)
        accuracy, new_rate, new_hit, new_gp, np_per, npvp_per, sov_per = eval_result
        
        # Store evaluation results
        evaluation_results.append(accuracy)
        evaluation_new_results.append(new_rate)
        evaluation_new_hit_results.append(new_hit)
        evaluation_new_hit_gg.append(new_gp)
        evaluation_np_per.append(np_per)
        evaluation_npvp_per.append(npvp_per)
        evaluation_sov_per.append(sov_per)
        
        # Save model checkpoint
        output_dir = f"./models/epoch_{epoch}_new_rate_{new_rate:.4f}_cla_{cla_accuracy:.4f}_model"
        model.save_pretrained(output_dir)
        
        # Print progress
        print(f"Epoch {epoch + 1}/{TRAIN_EPOCHS}")
        print(f"  Classification Accuracy: {cla_accuracy:.4f}")
        print(f"  Generation Accuracy: {accuracy:.4f}")
        print(f"  New Rate: {new_rate:.4f}")
        print(f"  NP/NPVP/SOV percentages: {np_per:.4f}/{npvp_per:.4f}/{sov_per:.4f}")
    
    # Return training results
    results = {
        'evaluation_results': evaluation_results,
        'evaluation_new_results': evaluation_new_results,
        'evaluation_new_hit_results': evaluation_new_hit_results,
        'evaluation_new_hit_gg': evaluation_new_hit_gg,
        'evaluation_np_per': evaluation_np_per,
        'evaluation_npvp_per': evaluation_npvp_per,
        'evaluation_sov_per': evaluation_sov_per,
        'cla_acc_list': cla_acc_list,
        'loss_weight': loss_weight,
        'epochs_record': epochs_record
    }
    
    return results

def save_results(results, output_path):
    """
    Save training results to file.
    
    Args:
        results: Dictionary containing training results
        output_path: Path to save results
    """
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {output_path}")

def plot_training_curves(results):
    """
    Plot training curves for analysis.
    
    Args:
        results: Dictionary containing training results
    """
    epochs = range(len(results['evaluation_new_results']))
    
    plt.figure(figsize=(12, 8))
    
    # Plot new rate
    plt.subplot(2, 2, 1)
    smoothed_new_results = gaussian_filter1d(results['evaluation_new_results'], sigma=2)
    plt.plot(epochs, smoothed_new_results, '#075B72', label='New Rate', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('New Rate')
    plt.title('Generation New Rate')
    plt.grid(True)
    plt.legend()
    
    # Plot classification accuracy
    plt.subplot(2, 2, 2)
    smoothed_cla_acc = gaussian_filter1d(results['cla_acc_list'], sigma=2)
    plt.plot(epochs, smoothed_cla_acc, '#30309A', label='Classification Accuracy', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Plot loss weight dynamics
    plt.subplot(2, 2, 3)
    plt.plot(results['loss_weight'], linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Weight')
    plt.title('Dynamic Loss Weighting')
    plt.grid(True)
    
    # Plot construction percentages
    plt.subplot(2, 2, 4)
    plt.plot(epochs, results['evaluation_np_per'], label='NP', linewidth=2)
    plt.plot(epochs, results['evaluation_npvp_per'], label='NPVP', linewidth=2)
    plt.plot(epochs, results['evaluation_sov_per'], label='SOV', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.title('Construction Type Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def extract_representations(model, tokenizer, sentences, output_dir):
    """
    Extract hidden representations from trained model.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        sentences: List of sentences
        output_dir: Directory to save representations
    """
    print("Extracting representations...")
    model.eval()
    
    # Storage for representations from each layer
    lists_hidden = [[] for _ in range(3)]  # Embedding + 2 GPT blocks
    
    with torch.no_grad():
        for corpus in sentences:
            output = model(
                torch.tensor([tokenizer.encode(corpus)]).to(model.device),
                output_hidden_states=True
            )
            
            # Extract hidden states from each layer
            for i in range(3):
                hidden_states = output.hidden_states[i]
                # Get last time step representation
                last_hidden_state = hidden_states[:, -1, :][0]
                lists_hidden[i].append(last_hidden_state)
    
    # Save representations
    os.makedirs(output_dir, exist_ok=True)
    for i, layer_representations in enumerate(lists_hidden):
        layer_matrix = torch.stack(layer_representations)
        layer_matrix = layer_matrix.detach().cpu().numpy()
        
        save_path = os.path.join(output_dir, f'layer_{i}_representations.npy')
        np.save(save_path, layer_matrix)
        print(f"Layer {i} representations saved: {layer_matrix.shape}")

def main():
    """
    Main training and evaluation pipeline.
    """
    print("Initializing Brocanto2 GPT Training Pipeline...")
    
    # Load data
    new_list, y_train_tensor, train_list, train_labels, val_list, val_labels = load_brocanto2_data()
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer()
    
    # Create integrated model (GPT-PR)
    model, linear_net = create_gpt_model(tokenizer, 'integrated')
    
    # Train model
    results = train_integrated_model(
        model, linear_net, tokenizer, 
        train_list, train_labels, 
        val_list, val_labels
    )
    
    # Save results
    save_results(results, 'training_results.pkl')
    
    # Plot training curves
    plot_training_curves(results)
    
    # Extract representations
    extract_representations(model, tokenizer, new_list, 'representations/')
    
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
