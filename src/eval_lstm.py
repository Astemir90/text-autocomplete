# функция для расчета ROUGE-1 и ROUGE-2, а также получения списков пяти удачных и пяти неудачных прогнозов
from evaluate import load
import yaml

# функция чтения конфиг
with open('configs/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# функция
def evaluate_rouge(model, loader, reverse_vocab, good_cases_displayed=config['eval_lstm']['good_cases_displayed'], 
                   bad_cases_displayed=config['eval_lstm']['good_cases_displayed']):
    model.eval()
    rouge = load('rouge')
    generated_text, target_text = [], []
    bad_cases, good_cases = [], []
    bad_cases_count, good_cases_count = 0, 0
    for batch in loader:
        context_ids = batch['context_ids'].tolist()
        context_tokens = [[x for x in sublist if x != 0] for sublist in context_ids]
        target_ids = batch['target_ids'].tolist()
        target_tokens = [[x for x in sublist if x != 0] for sublist in target_ids]
        for i in range(len(context_tokens)):
            if len(context_tokens[i]) == 0:
                continue
            new_text = model.generate_sequence(context_tokens[i], stop_token=1)
            new_text = ' '.join([reverse_vocab[token] for token in new_text])
            target = ' '.join([reverse_vocab[token] for token in target_tokens[i] if token != 2])
            if new_text == target and good_cases_count < good_cases_displayed:
                good_cases.append((new_text, target))
                good_cases_count += 1
            elif new_text != target and bad_cases_count < bad_cases_displayed:
                bad_cases.append((new_text, target))
                bad_cases_count += 1
            else:
                continue
            generated_text.append(new_text)
            target_text.append(target)
    result = rouge.compute(predictions=generated_text, references=target_text, use_stemmer=True, use_aggregator=True)
    return result['rouge1'], result['rouge2'], bad_cases, good_cases