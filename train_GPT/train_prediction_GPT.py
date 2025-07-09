from tokenise import BPE_token
from pathlib import Path
import os

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from transformers import TextDataset, LineByLineTextDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Initialize tokenizer and train on corpus
def initialize_tokenizer():
    """Initialize and train BPE tokenizer on Brocanto2 corpus"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BPE_token()
    paths = [str(x) for x in Path("./text/").glob("**/*.txt")]
    
    # Train tokenizer on corpus
    tokenizer.bpe_train(paths)
    
    save_path = 'tokenized_data'
    tokenizer.save_tokenizer(save_path)
    
    return tokenizer, save_path, device

# Load Brocanto2 grammar constructions
def load_grammar_constructions():
    """Load NP, NPVP, and SOV constructions from corpus"""
    from right_corpus import NP, NPVP, SOV
    total_constructions = len(NP + NPVP + SOV)
    print(f"Total grammar constructions: {total_constructions}")
    return NP, NPVP, SOV

# Evaluation function for generation tasks
def evaluate_generation_performance(model, tokenizer, test_dataset, train_percent, temperature):
    """
    Evaluate model performance on generation tasks
    Returns accuracy, new_rate, and construction-specific metrics
    """
    NP, NPVP, SOV = load_grammar_constructions()
    
    np_generated = []
    npvp_generated = []
    sov_generated = []
    
    model.eval()
    correct_count = 0
    total_generated = 0
    novel_correct = 0
    all_correct = []
    novel_generated = []
    
    # Load full corpus and training subset
    with open("output.txt", "r", encoding="utf-8") as f:
        full_corpus = f.read().splitlines()
    with open(f"text/train{train_percent}.txt", "r", encoding="utf-8") as f:
        train_corpus = f.read().splitlines()
    
    # Determine novel items (not in training set)
    novel_items = [x for x in full_corpus if x not in train_corpus]
    
    for example in test_dataset:
        # Extract first two tokens for generation seed
        first_token = example["input_ids"][0]
        second_token = example["input_ids"][1]
        
        input_ids = torch.tensor([[0, first_token, second_token]]).to(model.device)
        
        # Generate with varying lengths
        for gen_length in range(5, 20):
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_length=gen_length,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(input_ids),
                    return_dict_in_generate=True,
                    temperature=temperature,
                    do_sample=True
                )
            
            generated_sequence = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True)
            
            novel_flag = False
            
            if generated_sequence in full_corpus:
                correct_count += 1
                all_correct.append(generated_sequence)
                
                if generated_sequence in novel_items:
                    novel_correct += 1
                    novel_flag = True
                    novel_generated.append(generated_sequence)
                
                # Categorize by construction type
                if generated_sequence in NP:
                    np_generated.append(generated_sequence)
                elif generated_sequence in NPVP:
                    npvp_generated.append(generated_sequence)
                elif generated_sequence in SOV:
                    sov_generated.append(generated_sequence)
                break
        
        # Additional generation attempt if no novel item found
        if not novel_flag and generated_sequence in full_corpus:
            generated_ids = model.generate(
                input_ids,
                max_length=gen_length + 1,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(input_ids),
                return_dict_in_generate=True,
                temperature=temperature,
                do_sample=True
            )
            
            generated_sequence = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True)
            
            if generated_sequence in novel_items:
                novel_correct += 1
                all_correct.append(generated_sequence)
                novel_generated.append(generated_sequence)
                
                # Categorize by construction type
                if generated_sequence in NP:
                    np_generated.append(generated_sequence)
                elif generated_sequence in NPVP:
                    npvp_generated.append(generated_sequence)
                elif generated_sequence in SOV:
                    sov_generated.append(generated_sequence)
        
        total_generated += 1
        
        # Final categorization
        if generated_sequence in full_corpus:
            all_correct.append(generated_sequence)
        if generated_sequence in novel_items:
            novel_generated.append(generated_sequence)
        if generated_sequence in NP:
            np_generated.append(generated_sequence)
        elif generated_sequence in NPVP:
            npvp_generated.append(generated_sequence)
        elif generated_sequence in SOV:
            sov_generated.append(generated_sequence)
    
    # Calculate metrics
    new_rate = novel_correct / (novel_correct + (total_generated - correct_count))
    accuracy = correct_count / total_generated
    generation_diversity = len(set(all_correct)) / total_generated
    
    if len(set(all_correct)) != 0:
        novel_proportion = len(set(novel_generated)) / len(set(all_correct))
        np_proportion = len(set(np_generated)) / len(set(all_correct))
        npvp_proportion = len(set(npvp_generated)) / len(set(all_correct))
        sov_proportion = len(set(sov_generated)) / len(set(all_correct))
    else:
        novel_proportion = 0
        np_proportion = 0
        npvp_proportion = 0
        sov_proportion = 0
    
    return (accuracy, new_rate, generation_diversity, novel_proportion, 
            np_proportion, npvp_proportion, sov_proportion)

# Training configuration
def setup_training_config():
    """Setup training hyperparameters"""
    config = {
        'layers': 2,
        'train_datasets': ['_human_corrected'],
        'temperature': 1.0,
        'learning_rate': 5e-4,
        'train_epochs': 200,
        'target_accuracy': 0.91
    }
    return config

# Main training loop
def train_gpt_prediction_model():
    """Main training function for GPT prediction model"""
    
    # Initialize components
    tokenizer, save_path, device = initialize_tokenizer()
    config = setup_training_config()
    
    # Initialize result storage
    num_datasets = len(config['train_datasets'])
    results = {
        'accuracy': [[] for _ in range(num_datasets)],
        'new_rate': [[] for _ in range(num_datasets)],
        'generation_diversity': [[] for _ in range(num_datasets)],
        'novel_proportion': [[] for _ in range(num_datasets)],
        'np_proportion': [[] for _ in range(num_datasets)],
        'npvp_proportion': [[] for _ in range(num_datasets)],
        'sov_proportion': [[] for _ in range(num_datasets)]
    }
    
    # Train on each dataset
    for dataset_idx, train_dataset in enumerate(config['train_datasets']):
        
        # Setup tokenizer with special tokens
        tokenizer = GPT2Tokenizer.from_pretrained(save_path)
        tokenizer.add_special_tokens({
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"
        })
        
        # Create model configuration
        model_config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            n_layer=config['layers'],
        )
        
        # Initialize model
        model = GPT2LMHeadModel(model_config).to(device)
        model.config.pad_token_id = tokenizer.eos_token_id
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        
        # Load training data
        train_file = f"text/train{train_dataset}.txt"
        train_dataset_obj = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=train_file,
            block_size=128
        )
        
        # Load test data
        test_file = "output.txt"
        test_dataset_obj = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=test_file,
            block_size=128
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir="./output",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=config['learning_rate'],
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset_obj,
        )
        
        # Training and evaluation loop
        for epoch in range(config['train_epochs']):
            trainer.train()
            
            # Evaluate model performance
            evaluation_results = evaluate_generation_performance(
                model, tokenizer, test_dataset_obj, train_dataset, config['temperature']
            )
            
            # Store results
            accuracy, new_rate, gen_diversity, novel_prop, np_prop, npvp_prop, sov_prop = evaluation_results
            
            results['accuracy'][dataset_idx].append(accuracy)
            results['new_rate'][dataset_idx].append(new_rate)
            results['generation_diversity'][dataset_idx].append(gen_diversity)
            results['novel_proportion'][dataset_idx].append(novel_prop)
            results['np_proportion'][dataset_idx].append(np_prop)
            results['npvp_proportion'][dataset_idx].append(npvp_prop)
            results['sov_proportion'][dataset_idx].append(sov_prop)
            
            # Save model checkpoint
            output_dir = (f"./layer={config['layers']}_lr={config['learning_rate']}_"
                         f"{train_dataset}_corpus/epoch_{epoch}_new_rate_{new_rate:.4f}_"
                         f"acc_{accuracy:.4f}_model")
            model.save_pretrained(output_dir)
            
            # Print training progress
            print(f"Epoch {epoch+1}/{config['train_epochs']} - Dataset: {train_dataset}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  New Rate: {new_rate:.4f}")
            print(f"  Generation Diversity: {gen_diversity:.4f}")
            print(f"  Novel Proportion: {novel_prop:.4f}")
            print(f"  NP Proportion: {np_prop:.4f}")
            print(f"  NPVP Proportion: {npvp_prop:.4f}")
            print(f"  SOV Proportion: {sov_prop:.4f}")
    
    return results

# Visualization function
def plot_training_results(results, config):
    """Generate training progress visualization"""
    
    # Extract model paths and performance metrics
    model_paths = []
    new_rates = []
    
    base_path = f"./layer={config['layers']}_lr={config['learning_rate']}_human_corrected_corpus/"
    
    for epoch in range(config['train_epochs']):
        for model_name in os.listdir(base_path):
            if (model_name.startswith('epoch_') and 
                model_name.split('_')[1] == str(epoch)):
                if float(model_name.split('_')[4]) not in new_rates:
                    model_paths.append(os.path.join(base_path, model_name))
                    new_rates.append(float(model_name.split('_')[4]))
    
    # Create visualization
    epochs = range(config['train_epochs'])
    
    for i, dataset in enumerate(config['train_datasets']):
        plt.figure(figsize=(8, 8))
        
        # Apply Gaussian smoothing
        original_data = new_rates
        smoothed_data = gaussian_filter1d(original_data, sigma=2)
        
        # Adjust smoothed curve to match original maximum
        original_max = max(original_data)
        smoothed_max = max(smoothed_data)
        offset = original_max - smoothed_max
        adjusted_smoothed = smoothed_data + offset
        
        # Plot smoothed curve
        plt.plot(epochs, adjusted_smoothed, '#075B72', 
                label='New Rate', linewidth=5)
        
        # Add maximum value annotation
        max_value = max(adjusted_smoothed)
        max_epoch = epochs[adjusted_smoothed.tolist().index(max_value)]
        plt.axhline(y=max_value, color='#075B72', 
                   linestyle='--', linewidth=1.5, alpha=0.8)
        plt.text(50, max_value + 0.02, f'Max: {max_value:.2f}', 
                fontsize=25, color='#075B72')
        
        # Configure plot
        plt.ylim(0, 1.0)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('Epochs', fontsize=25)
        plt.ylabel('New Rate', fontsize=25)
        plt.grid(True)
        plt.legend(loc='lower right', fontsize=20)
        
        # Save plot
        plt.savefig(f'./plots/gpt_prediction_training_{dataset}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

# Representation extraction function
def extract_model_representations(model_path, stimuli_file):
    """Extract hidden state representations from trained model"""
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained('tokenized_data')
    
    # Add special tokens
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load stimulus materials
    df_materials = pd.read_excel(stimuli_file)
    
    # Process stimuli
    stimulus_list = []
    for i in range(len(df_materials)):
        sentence = ' '.join([
            str(df_materials[f'word{j}'][i]) for j in range(1, 9)
        ])
        sentence = sentence.replace("nan", "").strip()
        stimulus_list.append(sentence)
    
    # Remove duplicate words within sentences
    def remove_word_duplicates(sentence):
        words = sentence.split()
        for _ in range(7):
            if len(words) >= 2 and words[-1] == words[-2]:
                words.pop(-1)
        return ' '.join(words)
    
    processed_stimuli = [remove_word_duplicates(s) for s in stimulus_list]
    unique_stimuli = list(dict.fromkeys(processed_stimuli))
    
    # Extract representations for each layer
    layer_representations = [[] for _ in range(3)]
    
    for stimulus in unique_stimuli:
        with torch.no_grad():
            input_ids = torch.tensor([tokenizer.encode(stimulus)]).to(device)
            outputs = model(input_ids, output_hidden_states=True)
            
            # Extract final token representation from each layer
            for layer_idx in range(3):
                hidden_states = outputs.hidden_states[layer_idx]
                final_token_repr = hidden_states[:, -1, :][0]
                layer_representations[layer_idx].append(final_token_repr)
    
    # Convert to numpy arrays and save
    for layer_idx, representations in enumerate(layer_representations):
        repr_array = torch.stack(representations).detach().cpu().numpy()
        np.save(f'./representations/layer_{layer_idx}_representations.npy', repr_array)
        print(f"Layer {layer_idx} representations shape: {repr_array.shape}")
    
    return layer_representations

# Main execution
if __name__ == "__main__":
    
    # Train GPT prediction model
    training_results = train_gpt_prediction_model()
    
    # Generate visualization
    config = setup_training_config()
    plot_training_results(training_results, config)
    
    # Extract representations from best model
    best_model_path = "./layer=2_lr=5e-4_human_corrected_corpus/best_model"
    stimuli_file = "./materials/MATERIAL.xlsx"
    
    representations = extract_model_representations(best_model_path, stimuli_file)
    
    print("Training and analysis completed successfully.")