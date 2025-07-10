import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import ( GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup,)
from scipy.spatial.distance import cosine
from sklearn.model_selection import KFold
from collections import OrderedDict
from joblib import Parallel, delayed
import pickle

# Configuration and hyperparameters
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = 2
        self.temp = 1
        self.lr = 5e-6
        self.labels_ids = {'true': 0, 'false': 1}
        self.n_labels = len(self.labels_ids)
        self.max_length = 25
        self.epochs = 100
        self.batch_size = 20
        
        # Loss function weights based on paper methodology
        # α = ln(79)/2 for 79-way vs 2-way classification difficulty balance
        self.alpha = np.log(79) / 2
        # β = 768/4 for neural MSE vs cross-entropy scale balance
        self.beta = 768 / 4
        # γ = 1 as reference scaling factor
        self.gamma = 1

config = Config()

def load_tokenizer(tokenizer_path):
    """Load and configure GPT2 tokenizer with special tokens"""
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "</s>",
        "mask_token": "<mask>"
    })
    tokenizer.padding_side = 'left'  # GPT-style padding
    return tokenizer

def load_corpus_data(corpus_path):
    """Load language corpus data (NP, NPVP, SOV structures)"""
    # Import corpus structures - replace with your actual corpus loading logic
    from right_corpus import NP, NPVP, SOV  # Replace with actual import
    return NP, NPVP, SOV

def load_subjects_data(subjects_file):
    """Load subject IDs from file"""
    read_file = pd.read_excel(subjects_file)
    return list(read_file['BIDS_ID'])

def load_material_data(material_file):
    """Load experimental material data"""
    df_ma = pd.read_excel(material_file)
    id_2_dur = dict(zip(df_ma['TrialID'], df_ma['soundDur']))
    
    # Create trial list from word columns
    trail_list = []
    for i in range(len(df_ma)):
        words = [str(df_ma[f'word{j}'][i]) for j in range(1, 9)]
        sentence = ' '.join(words)
        trail_list.append(sentence)
    
    # Clean sentences
    trail_list = [sentence.replace("nan", "").strip() for sentence in trail_list]
    
    return df_ma, id_2_dur, trail_list

def remove_duplicates(sentence):
    """Remove duplicate words from sentence"""
    words = sentence.split()
    for i in range(7):
        if len(words) > 1:
            if words[-1] == '#':
                words.pop(-1)
            elif words[-1] == words[-2]:
                words.pop(-1)
    return ' '.join(words)

def prepare_corpus_labels(trail_list, corpus_structures):
    """Prepare corpus and labels for training"""
    NP, NPVP, SOV = corpus_structures
    
    # Clean sentences
    new_list_raw = [remove_duplicates(sentence) for sentence in trail_list]
    
    # Create labels
    y_train_raw = []
    for item in new_list_raw:
        if item in NP + NPVP + SOV:
            y_train_raw.append('true')
        else:
            y_train_raw.append('false')
    
    # Create mapping for reorganized corpus
    mapping = {}
    new_list_co = []
    new_list_unc = []
    
    for n_num, co in enumerate(new_list_raw):
        if co in (NP + NPVP + SOV):
            new_list_co.append(co)
            mapping[n_num] = len(new_list_co) - 1
        else:
            new_list_unc.append(co)
            mapping[n_num] = 144 + len(new_list_unc) - 1
    
    new_list = new_list_co + new_list_unc
    
    # Create final labels
    y_train = ['true' if item in (NP + NPVP + SOV) else 'false' for item in new_list]
    
    # Convert to tensors
    y_train_tensor = []
    for item in new_list:
        if item in NP + NPVP + SOV:
            y_train_tensor.append(torch.tensor(0).to(config.device))
        else:
            y_train_tensor.append(torch.tensor(1).to(config.device))
    
    return new_list, y_train, y_train_tensor, mapping, y_train_raw

def load_behavioral_data(subject, day, data_dir):
    """Load behavioral data for specific subject and day"""
    id_2_beh_list = []
    
    # Load data from different blocks
    for blk_num in range(1, 3):
        csv_file = f'{data_dir}/BCT_fMRI_GJT_blk{blk_num}-{subject}-{day}.csv'
        df_sub = pd.read_csv(csv_file)
        train_id_list = list(map(int, df_sub['TrialID'].dropna()))
        sub_beh = list(map(int, df_sub['Stimuli3.ACC'].dropna()))
        id_2_beh = dict(zip(train_id_list, sub_beh))
        id_2_beh_list.append(id_2_beh)
    
    # Process additional blocks with ID offset
    for blk_num in range(3, 5):
        csv_file = f'{data_dir}/BCT_fMRI_GJT_blk{blk_num}-{subject}-{day}.csv'
        df_sub = pd.read_csv(csv_file)
        train_id_list = list(map(int, df_sub['TrialID'].dropna()))
        sub_beh = list(map(int, df_sub['Stimuli3.ACC'].dropna()))
        train_id_list = [i + 96 for i in train_id_list]
        id_2_beh = dict(zip(train_id_list, sub_beh))
        id_2_beh_list.append(id_2_beh)
    
    for blk_num in range(5, 7):
        csv_file = f'{data_dir}/BCT_fMRI_GJT_blk{blk_num}-{subject}-{day}.csv'
        df_sub = pd.read_csv(csv_file)
        train_id_list = list(map(int, df_sub['TrialID'].dropna()))
        train_id_list = [i + 192 for i in train_id_list]
        sub_beh = list(map(int, df_sub['Stimuli3.ACC'].dropna()))
        id_2_beh = dict(zip(train_id_list, sub_beh))
        id_2_beh_list.append(id_2_beh)
    
    # Merge all behavioral data
    merged_id_2_beh = {}
    for id_2_beh in id_2_beh_list:
        merged_id_2_beh.update(id_2_beh)
    
    return OrderedDict(sorted(merged_id_2_beh.items()))

def process_behavioral_labels(behavioral_data, y_train_raw, mapping, new_list):
    """Process behavioral data into labels"""
    y_train_sub_score_raw = []
    
    for train_id, behavior in behavioral_data.items():
        if behavior == 1:  # Correct response
            y_train_sub_score_raw.append(y_train_raw[train_id - 1])
        else:  # Incorrect response - flip label
            original_label = y_train_raw[train_id - 1]
            flipped_label = 'false' if original_label == 'true' else 'true'
            y_train_sub_score_raw.append(flipped_label)
    
    # Map to new list structure
    y_train_sub_score = [None] * len(new_list)
    for raw_index, new_index in mapping.items():
        y_train_sub_score[new_index] = y_train_sub_score_raw[raw_index]
    
    return y_train_sub_score

def load_generalization_test_data(subject, data_dir):
    """Load generalization test data for day 7"""
    df_subject = pd.read_csv(f'{data_dir}/GJT_generalization-{subject}-7.csv')
    
    # Create trial list
    trail_list = []
    for i in range(len(df_subject)):
        words = [str(df_subject[f'word{j}'][i]) for j in range(1, 9)]
        sentence = ' '.join(words)
        trail_list.append(sentence)
    
    trail_list = [sentence.replace("nan", "").strip() for sentence in trail_list]
    
    # Remove duplicates and filter empty sentences
    sub_item_list = [remove_duplicates(sentence) for sentence in trail_list]
    used_sub_item_list = [sentence for sentence in sub_item_list if sentence]
    
    return used_sub_item_list

def evaluate_generation_performance(model, tokenizer, test_dataset, corpus_structures, temp=1):
    """Evaluate model generation performance on different linguistic structures"""
    NP, NPVP, SOV = corpus_structures
    
    # Load full corpus for evaluation
    with open("corpus_full.txt", "r", encoding="utf-8") as f:
        full_corpus = f.read().splitlines()
    
    with open("corpus_train.txt", "r", encoding="utf-8") as f:
        train_corpus = f.read().splitlines()
    
    novel_corpus = [x for x in full_corpus if x not in train_corpus]
    
    np_generated = []
    npvp_generated = []
    sov_generated = []
    
    model.eval()
    correct_count = 0
    total_generated = 0
    novel_count = 0
    correct_sentences = []
    novel_sentences = []
    
    for example in test_dataset:
        # Get first two tokens for generation seed
        first_token = example["input_ids"][0]
        second_token = example["input_ids"][1]
        
        input_ids = torch.tensor([[0, first_token, second_token]]).to(model.device)
        
        generated_sequence = None
        novel_flag = False
        
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
            
            generated_sequence = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True)
            
            if generated_sequence in full_corpus:
                correct_count += 1
                correct_sentences.append(generated_sequence)
                
                if generated_sequence in novel_corpus:
                    novel_count += 1
                    novel_flag = True
                    novel_sentences.append(generated_sequence)
                
                # Classify by linguistic structure
                if generated_sequence in NP:
                    np_generated.append(generated_sequence)
                elif generated_sequence in NPVP:
                    npvp_generated.append(generated_sequence)
                elif generated_sequence in SOV:
                    sov_generated.append(generated_sequence)
                
                break
        
        # Try one more length if not novel
        if not novel_flag and generated_sequence in full_corpus:
            with torch.no_grad():
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
            
            if generated_sequence in novel_corpus:
                novel_count += 1
                novel_sentences.append(generated_sequence)
        
        total_generated += 1
    
    # Calculate metrics
    accuracy = correct_count / total_generated if total_generated > 0 else 0
    novel_accuracy = novel_count / (novel_count + (total_generated - correct_count)) if total_generated > 0 else 0
    generation_rate = len(set(correct_sentences)) / total_generated if total_generated > 0 else 0
    
    if len(set(correct_sentences)) > 0:
        novel_rate = len(set(novel_sentences)) / len(set(correct_sentences))
        np_rate = len(set(np_generated)) / len(set(correct_sentences))
        npvp_rate = len(set(npvp_generated)) / len(set(correct_sentences))
        sov_rate = len(set(sov_generated)) / len(set(correct_sentences))
    else:
        novel_rate = np_rate = npvp_rate = sov_rate = 0
    
    return accuracy, novel_accuracy, generation_rate, novel_rate, np_rate, npvp_rate, sov_rate

def create_model_and_classifier(tokenizer, config):
    """Create GPT model and classification head"""
    gpt_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_layer=config.layers,
    )
    
    model = GPT2LMHeadModel(gpt_config).to(config.device)
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Classification head for behavioral prediction
    classifier = nn.Linear(in_features=768, out_features=2, bias=True).to(config.device)
    
    return model, classifier

def train_subject_model(subject, data_paths, config):
    """Train subject-specific multi-task GPT model"""
    print(f'Training subject {subject}...')
    
    # Load data
    tokenizer = load_tokenizer(data_paths['tokenizer'])
    corpus_structures = load_corpus_data(data_paths['corpus'])
    
    # Load behavioral data
    day1_behavioral = load_behavioral_data(subject, 1, data_paths['behavioral'])
    day2_behavioral = load_behavioral_data(subject, 2, data_paths['behavioral'])
    
    # Load neural data
    neural_data = np.load(data_paths['neural'].format(subject=subject))
    neural_tensor = torch.tensor(neural_data, dtype=torch.float32).to(config.device)
    
    # Load generalization test data
    gen_test_data = load_generalization_test_data(subject, data_paths['generalization'])
    
    # Prepare corpus and labels
    new_list, y_train, y_train_tensor, mapping, y_train_raw = prepare_corpus_labels(
        trail_list, corpus_structures
    )
    
    # Process behavioral labels
    day1_labels = process_behavioral_labels(day1_behavioral, y_train_raw, mapping, new_list)
    day2_labels = process_behavioral_labels(day2_behavioral, y_train_raw, mapping, new_list)
    
    # Convert to tensors
    day1_tensor = [torch.tensor(0 if label == 'true' else 1).to(config.device) for label in day1_labels]
    day2_tensor = [torch.tensor(0 if label == 'true' else 1).to(config.device) for label in day2_labels]
    
    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(new_list)))):
        print(f'Fold {fold_idx + 1}/5')
        
        # Create model and classifier
        model, classifier = create_model_and_classifier(tokenizer, config)
        
        # Prepare training data
        train_sentences = [new_list[i] for i in train_idx]
        train_day1_labels = [day1_tensor[i] for i in train_idx]
        train_day2_labels = [day2_tensor[i] for i in train_idx]
        train_neural = [neural_tensor[i] for i in train_idx]
        
        # Prepare validation data
        val_sentences = [new_list[i] for i in val_idx]
        val_day1_labels = [day1_tensor[i] for i in val_idx]
        val_day2_labels = [day2_tensor[i] for i in val_idx]
        val_neural = [neural_tensor[i] for i in val_idx]
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            list(model.parameters()) + list(classifier.parameters()),
            lr=5e-4,
            eps=1e-8
        )
        
        total_steps = len(train_sentences) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Loss functions
        criterion_classification = nn.CrossEntropyLoss()
        criterion_neural = nn.MSELoss()
        
        # Training loop
        for epoch in range(config.epochs):
            model.train()
            classifier.train()
            
            total_loss = 0
            behavioral_loss = 0
            neural_loss = 0
            lm_loss = 0
            
            for batch_idx, sentence in enumerate(train_sentences):
                # Forward pass through GPT
                input_ids = torch.tensor([tokenizer.encode(sentence)]).to(config.device)
                
                outputs = model(
                    input_ids,
                    labels=input_ids,
                    output_hidden_states=True
                )
                
                # Get hidden state for classification and neural prediction
                hidden_state = outputs.hidden_states[-1][-1][-1]  # Last layer, last token
                
                # Classification loss (behavioral prediction)
                behavioral_logits = classifier(hidden_state)
                
                # Alternate between day 1 and day 2 labels based on epoch
                if (epoch // 20) % 2 == 0:
                    behavioral_target = train_day2_labels[batch_idx]
                else:
                    behavioral_target = train_day1_labels[batch_idx]
                
                loss_behavioral = criterion_classification(
                    behavioral_logits.unsqueeze(0),
                    behavioral_target.unsqueeze(0)
                )
                
                # Neural prediction loss
                loss_neural = criterion_neural(hidden_state, train_neural[batch_idx])
                
                # Language modeling loss (only for grammatical sentences)
                if sentence in corpus_structures[0] + corpus_structures[1] + corpus_structures[2]:
                    loss_lm = outputs.loss
                else:
                    loss_lm = torch.tensor(0.0).to(config.device)
                
                # Combined loss with paper's weighting scheme
                combined_loss = (
                    config.alpha * loss_behavioral +
                    config.beta * loss_neural +
                    config.gamma * loss_lm
                )
                
                total_loss += combined_loss.item()
                behavioral_loss += loss_behavioral.item()
                neural_loss += loss_neural.item()
                lm_loss += loss_lm.item()
                
                # Backward pass every batch_size samples
                if (batch_idx + 1) % config.batch_size == 0:
                    optimizer.zero_grad()
                    combined_loss.backward()
                    optimizer.step()
                    scheduler.step()
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                classifier.eval()
                
                with torch.no_grad():
                    val_accuracy = 0
                    val_neural_similarity = 0
                    
                    for i, sentence in enumerate(val_sentences):
                        input_ids = torch.tensor([tokenizer.encode(sentence)]).to(config.device)
                        outputs = model(input_ids, output_hidden_states=True)
                        hidden_state = outputs.hidden_states[-1][-1][-1]
                        
                        # Classification accuracy
                        logits = classifier(hidden_state)
                        pred_label = torch.argmax(logits).item()
                        true_label = val_day1_labels[i].item()
                        val_accuracy += (pred_label == true_label)
                        
                        # Neural similarity
                        similarity = 1 - cosine(
                            hidden_state.cpu().numpy(),
                            val_neural[i].cpu().numpy()
                        )
                        val_neural_similarity += similarity
                    
                    val_accuracy /= len(val_sentences)
                    val_neural_similarity /= len(val_sentences)
                    
                    print(f'Epoch {epoch}: Val Acc = {val_accuracy:.3f}, '
                          f'Neural Sim = {val_neural_similarity:.3f}')
        
        fold_results.append({
            'model': model.state_dict(),
            'classifier': classifier.state_dict(),
            'val_accuracy': val_accuracy,
            'val_neural_similarity': val_neural_similarity
        })
    
    return fold_results

def main():
    """Main training pipeline"""
    # Configuration
    config = Config()
    
    # Data paths - modify these according to your data structure
    data_paths = {
        'tokenizer': './tokenized_data',
        'corpus': './corpus_data',
        'subjects': './subjects_data.xlsx',
        'material': './material_data.xlsx',
        'behavioral': './behavioral_data',
        'neural': './neural_data/sub-{subject}_neural_vector.npy',
        'generalization': './generalization_data'
    }
    
    # Load subjects
    subjects = load_subjects_data(data_paths['subjects'])
    
    # Initialize result storage
    all_results = {}
    
    # Train models for all subjects
    results = Parallel(n_jobs=5)(
        delayed(train_subject_model)(subject, data_paths, config)
        for subject in subjects
    )
    
    # Save results
    for idx, subject in enumerate(subjects):
        all_results[subject] = results[idx]
    
    # Save to file
    output_path = './results/multitask_gpt_results.pkl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"Training completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()