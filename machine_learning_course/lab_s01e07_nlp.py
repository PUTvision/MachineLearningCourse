import time

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import torchtext

from torch.utils.data import DataLoader
import torch

from lab_s01e07_nlp_utils import TextClassificationModel, train, evaluate


def ex_1():
    text = "I'm having a wonderful time at WZUM laboratories!"
    print(text.split())

    text = "I'm having a wonderful time at WZUM laboratories!"
    tokenizer = torchtext.data.get_tokenizer("basic_english")
    print(tokenizer(text))

    t1 = "John likes to watch movies. Mary likes movies too."
    t2 = "Mary also likes to watch football games."
    t1 = tokenizer(t1)
    t2 = tokenizer(t2)
    vocab = set(t1 + t2)
    vocab = dict.fromkeys(vocab, 0)
    print(vocab)

    t1_vocab = vocab.copy()
    for word in t1:
        t1_vocab[word] += 1

    t2_vocab = vocab.copy()
    for word in t2:
        t2_vocab[word] += 1

    print(t1_vocab)
    print(t2_vocab)

    t1 = list(t1_vocab.values())
    t2 = list(t2_vocab.values())


def ex_2():
    tokenizer = torchtext.data.get_tokenizer("basic_english")

    t1 = "John likes to watch movies. Mary likes movies too."
    t2 = "Mary also likes to watch football games."

    def yield_tokens(data):  # generator zwracający tokenizowany tekst
        for t in data:
            yield tokenizer(t)

    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens([t1, t2]), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])  # ustawienie, że dla nieznanych słów słownik ma zwracać indeks 0

    print(f'{vocab["football"]=}')
    print(f'{vocab["himalaje"]=}')
    print(f'{vocab["cokolwiek"]=}')
    print(f'{vocab["likes"]=}')

    for tokens in yield_tokens([t1, t2]):
        for word in tokens:
            # print(f'{vocab[word]=}')
            print(f'vocab[{word}]={vocab[word]}')


def ex_3():
    train_data, test_data = torchtext.datasets.AG_NEWS()
    print(next(train_data)[1])

    # for a, b in train_data:
    #     print(f'{a=}, {b=}')

    tokenizer = torchtext.data.get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    tekst = "I'm having a wonderful time at WZUM laboratories!"
    print(vocab(tokenizer(tekst)))

    text_pipeline = lambda x: vocab(tokenizer(x))
    print(text_pipeline(tekst))

    label_pipeline = lambda x: int(x) - 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    num_class = len(set([label for (label, text) in torchtext.datasets.AG_NEWS(split='train')]))
    print(f'{num_class=}')
    vocab_size = len(vocab)
    print(f'{vocab_size=}')
    model = TextClassificationModel(vocab_size, 64, num_class).to(device)

    BATCH_SIZE = 64

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    train_iter, test_iter = torchtext.datasets.AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    for epoch in range(1, 2 + 1):
        epoch_start_time = time.time()
        train(model, train_dataloader, optimizer, criterion, epoch)
        accu_val = evaluate(model, valid_dataloader, criterion)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print(f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | valid accuracy {accu_val:8.3f}')
        print('-' * 59)

    print('Checking the results of test dataset.')
    accu_test = evaluate(model, test_dataloader, criterion)
    print(f'test accuracy: {accu_test:8.3f}')


def main():
    # ex_1()
    # ex_2()
    ex_3()


if __name__ == '__main__':
    main()
