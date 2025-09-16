# обучение модели LSTM и выведение метрик
from src.eval_lstm import evaluate_rouge

def model_train(model, optimizer, criterion, vocab_size, train_loader, val_loader, reverse_vocab, n_epochs):
    model.train()
    result = []
    for epoch in range(n_epochs):
        for batch in train_loader:
            context_ids = batch['context_ids']
            target_ids = batch['target_ids']
            optimizer.zero_grad()
            outputs, _ = model(context_ids)
            loss = criterion(outputs.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
        if epoch+1 == n_epochs:
            rouge1, rouge2, bad_cases, good_cases = evaluate_rouge(model, val_loader, reverse_vocab)
            result.append(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, val_ROUGE-1: {rouge1:.4f}, val_ROUGE-2: {rouge2:.4f}')
            result.append('Matched cases on validation (generated, target):')
            for case in good_cases:
                result.append(case)
            result.append('Unmatched cases on validation (generated, target):')
            for case in bad_cases:
                result.append(case)
        else:
            rouge1, rouge2, bad_cases, good_cases = evaluate_rouge(model, val_loader, reverse_vocab)
            result.append(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, val_ROUGE-1: {rouge1:.4f}, val_ROUGE-2: {rouge2:.4f}')
    return result