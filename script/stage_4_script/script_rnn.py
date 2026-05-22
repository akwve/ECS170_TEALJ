import csv
import os
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_4_code.Training_Convergence_Plot import save_training_loss_curve
import numpy as np
import torch


def _tokenize_generation_text(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[.!?]", text.lower())


def _load_text_generation_corpus(corpus_file_path):
    jokes = []
    with open(corpus_file_path, 'r', encoding='utf-8', errors='ignore', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            joke = (row.get('Joke') or '').strip()
            if joke:
                jokes.append(joke)
    return jokes


def _build_trigram_model(texts):
    transitions = defaultdict(Counter)
    for text in texts:
        tokens = _tokenize_generation_text(text)
        if len(tokens) < 3:
            continue

        padded_tokens = ['<START>', '<START>'] + tokens + ['<END>']
        for index in range(len(padded_tokens) - 2):
            context = (padded_tokens[index], padded_tokens[index + 1])
            next_token = padded_tokens[index + 2]
            transitions[context][next_token] += 1

    return transitions


def _weighted_choice(counter, rng):
    choices = list(counter.items())
    tokens = [token for token, _ in choices]
    weights = [weight for _, weight in choices]
    return rng.choices(tokens, weights=weights, k=1)[0]


def _batch_negative_log_likelihood(transitions, texts, smoothing=1.0):
    total_loss = 0.0
    total_tokens = 0
    vocabulary = set()
    for context, counter in transitions.items():
        vocabulary.update(context)
        vocabulary.update(counter.keys())
    vocabulary_size = max(1, len(vocabulary))

    for text in texts:
        tokens = _tokenize_generation_text(text)
        if len(tokens) < 3:
            continue

        padded_tokens = ['<START>', '<START>'] + tokens + ['<END>']
        for index in range(len(padded_tokens) - 2):
            context = (padded_tokens[index], padded_tokens[index + 1])
            next_token = padded_tokens[index + 2]
            counter = transitions.get(context)
            context_total = sum(counter.values()) if counter else 0
            token_count = counter.get(next_token, 0) if counter else 0
            probability = (token_count + smoothing) / (context_total + smoothing * vocabulary_size)
            total_loss -= math.log(probability)
            total_tokens += 1

    return total_loss / max(1, total_tokens)


def _train_generation_model(texts, batch_size=20, smoothing=1.0):
    transitions = defaultdict(Counter)
    training_loss_history = []

    shuffled_texts = list(texts)
    random.Random(2).shuffle(shuffled_texts)

    for start_index in range(0, len(shuffled_texts), batch_size):
        batch_texts = shuffled_texts[start_index:start_index + batch_size]
        if transitions:
            training_loss_history.append(_batch_negative_log_likelihood(transitions, batch_texts, smoothing=smoothing))
        else:
            training_loss_history.append(_batch_negative_log_likelihood(_build_trigram_model(batch_texts), batch_texts, smoothing=smoothing))

        batch_model = _build_trigram_model(batch_texts)
        for context, counter in batch_model.items():
            transitions[context].update(counter)

    return transitions, training_loss_history


def _format_generated_tokens(tokens):
    text = ' '.join(tokens)
    text = re.sub(r'\s+([.!?])', r'\1', text)
    return text.strip()


def _generate_text(transitions, seed_words, max_new_tokens=20, seed=2):
    rng = random.Random(seed)
    prompt_tokens = [word.lower() for word in seed_words if word]
    if len(prompt_tokens) < 3:
        raise ValueError('seed_words must contain at least three words')

    generated_tokens = prompt_tokens[:]
    context = tuple(generated_tokens[-2:])

    for _ in range(max_new_tokens):
        counter = transitions.get(context)
        if counter is None:
            fallback_contexts = [key for key in transitions if key[1] == context[-1]]
            if fallback_contexts:
                context = rng.choice(fallback_contexts)
                counter = transitions.get(context)
            else:
                context = rng.choice(list(transitions.keys()))
                counter = transitions[context]

        next_token = _weighted_choice(counter, rng)
        if next_token == '<END>':
            break

        generated_tokens.append(next_token)
        context = (generated_tokens[-2], generated_tokens[-1])

    return _format_generated_tokens(generated_tokens)


def _print_generation_examples():
    corpus_file_path = PROJECT_ROOT / 'data' / 'stage_4_data' / 'text_generation' / 'data'
    if not corpus_file_path.exists():
        print(f'Generation corpus not found at {corpus_file_path}')
        return

    jokes = _load_text_generation_corpus(corpus_file_path)
    transitions, training_loss_history = _train_generation_model(jokes, batch_size=20)

    if not transitions:
        print('No generation transitions could be built from the corpus')
        return

    plot_file_path = PROJECT_ROOT / 'result' / 'stage_4_result' / 'generator_training_loss_curve_generator.png'
    save_training_loss_curve(
        [training_loss_history],
        str(plot_file_path),
        'Text Generator Training Loss Convergence',
        line_color='tab:orange'
    )
    print(f'Generator training loss curve saved to {plot_file_path}')

    dataset_seeds = []
    for joke in jokes:
        tokens = _tokenize_generation_text(joke)
        if len(tokens) >= 3:
            dataset_seeds.append(tokens[:3])
        if len(dataset_seeds) >= 2:
            break

    print('\n========== Text Generation Examples ==========')
    for seed_words in dataset_seeds:
        generated_text = _generate_text(transitions, seed_words, max_new_tokens=20, seed=2)
        print(f'Seed: {" ".join(seed_words)}')
        print(f'Generated: {generated_text}')
        print('')
    print('============================================\n')

# ---- RNN Text Classification script ----
if __name__ == '__main__':
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)

    config = {
        'max_epoch': 30,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.5,
        'max_sequence_length': 500,
        'min_word_frequency': 5
    }

    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    print('Loading dataset...')
    data_obj = Dataset_Loader('IMDb', 'IMDb movie reviews sentiment classification')
    data_obj.dataset_source_folder_path = str(PROJECT_ROOT / 'data' / 'stage_4_data' / 'text_classification')

    evaluate_obj = Evaluate_Accuracy('evaluating_metrics', '')
    method_obj = Method_RNN('RNN_Text_Classification', 'RNN for text classification')

    # Apply configuration
    for key, val in config.items():
        if hasattr(data_obj, key):
            setattr(data_obj, key, val)
        if hasattr(method_obj, key):
            setattr(method_obj, key, val)

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    print(f'Configuration: {config}')

    # Load data
    data = data_obj.load()
    method_obj.data = data

    # Apply final configuration
    for key, val in config.items():
        if hasattr(method_obj, key):
            setattr(method_obj, key, val)

    # Train and test
    print('Running method...')
    result = method_obj.run()

    # Evaluate
    evaluate_obj.data = {'true_y': result['true_y'], 'pred_y': result['pred_y']}
    metrics = evaluate_obj.evaluate()
    print(f'\n========== Test Results ==========')
    print(f'Accuracy:  {metrics["accuracy"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall:    {metrics["recall"]:.4f}')
    print(f'F1 Score:  {metrics["f1"]:.4f}')
    print(f'==================================\n')

    # Plot training loss
    plot_file_path = str(PROJECT_ROOT / 'result' / 'stage_4_result' / 'training_loss_curve_rnn.png')
    os.makedirs(os.path.dirname(plot_file_path), exist_ok=True)

    if method_obj.training_loss_history:
        save_training_loss_curve(
            [method_obj.training_loss_history],
            plot_file_path,
            'RNN Training Loss Convergence',
            line_color='tab:blue'
        )
        print(f'Training loss curve saved to {plot_file_path}')

    _print_generation_examples()

    print('************ End ************')
