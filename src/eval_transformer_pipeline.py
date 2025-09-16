# функция для расчета ROUGE предобученной модели
from evaluate import load

def pretrained_model_rouge(generator, loader, reverse_vocab, top_k=50, max_new_tokens=20):
    rouge = load('rouge')
    generated_text, target_text = [], []
    bad_cases, good_cases = [], []
    bad_cases_count, good_cases_count = 0, 0
    result = []
    for batch in loader:
        context_ids = batch['context_ids'].tolist()
        context_tokens = [[x for x in sublist if x != 0] for sublist in context_ids]
        context_texts = [' '.join([reverse_vocab[i] for i in sublist]) for sublist in context_tokens]
        target_ids = batch['target_ids'].tolist()
        target_tokens = [[x for x in sublist if x != 0] for sublist in target_ids]
        target_texts = [' '.join([reverse_vocab[i] for i in sublist]) for sublist in target_tokens]
        for i in range(len(context_texts)):
            if len(context_texts[i]) == 0:
                continue
            new_text = generator(context_texts[i], do_sample=True, top_k=top_k, max_new_tokens=max_new_tokens, 
                                 pad_token_id = generator.tokenizer.eos_token_id)[0]['generated_text']
            new_text = new_text[len(context_texts[i]) + 1:]
            target = context_texts[i] + ' ' + target_texts[i]
            if new_text == target and good_cases_count < 5:
                good_cases.append((new_text, target))
                good_cases_count += 1
            elif new_text != target and bad_cases_count < 5:
                bad_cases.append((new_text, target))
                bad_cases_count += 1
            else:
                continue
            generated_text.append(new_text)
            target_text.append(target)
    rouge_dict = rouge.compute(predictions=generated_text, references=target_text, use_stemmer=True, use_aggregator=True)
    result.append(f'val_ROUGE-1: {rouge_dict['rouge1']:.4f}, val_ROUGE-2: {rouge_dict['rouge2']:.4f}')
    result.append('Matched cases on validation (generated, target):')
    for case in good_cases:
        result.append(case)
    result.append('Unmatched cases on validation (generated, target):')
    for case in bad_cases:
        result.append(case)
    return result

    