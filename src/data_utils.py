import re
from collections import Counter

# функция очистки текста
def clean_string(text):
    # приводим к нижнему регистру
    text = text.lower()
    # удаляем ссылки (http://, https://, www.)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # удаляем упоминания (@username)
    text = re.sub(r'@\w+', '', text)
    # удаляем смайлы
    text = re.sub(r'[:;][-o]?(?:\)|\(|d|p|/|\\|\'|")', '', text)
    # оставляем только буквы, цифры, пробелы и базовые знаки препинания
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# создание словарей
def create_vocabs(texts):
    all_words = ' '.join(texts).split()
    counter = Counter(all_words)
    vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common())}
    vocab['<PAD>'] = 0
    vocab['<EOS>'] = 1
    vocab_size = len(vocab)
    reverse_vocab = {i: word for word, i in vocab.items()}
    return vocab, reverse_vocab, vocab_size

# функция токенизации
def tokenize_text(texts, vocab):
    return [[vocab[word] for word in sentence.split()] for sentence in texts]