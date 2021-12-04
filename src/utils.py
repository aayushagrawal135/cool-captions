import torch
from torch import nn
from nltk.tokenize import word_tokenize

from vocab import Vocabulary
# Now we need to window pad our training examples. We have already defined a
# function to handle window padding. We are including it here again so that
# everything is in one place.
def pad_window(sentence, window_size, vocab:Vocabulary):
    window = [vocab.word_to_idx[vocab.pad]] * window_size
    return window + sentence + window

# Now we need to turn words in our training examples to indices. We are
# copying the function defined earlier for the same reason as above.
def convert_tokens_to_indices(sentence, vocab:Vocabulary):
    return [vocab.word_to_idx.get(token, vocab.word_to_idx["<UNK>"]) for token in sentence]

def add_start_end_of_sentence(sentence, vocab:Vocabulary):
    return [vocab.word_to_idx[vocab.sos]] + sentence + [vocab.word_to_idx[vocab.eos]]

def custom_collate_fn(batch, window_size, vocab:Vocabulary):
    # Break our batch into the training examples (x) and labels (y)
    # We are turning our x and y into tensors because nn.utils.rnn.pad_sequence
    # method expects tensors. This is also useful since our model will be
    # expecting tensor inputs.
    images, labels = zip(*batch)
    images = torch.stack(list(images), dim=0)
    captions = [label['comment'] for label in labels]
    captions = [word_tokenize(caption) for caption in captions]

    # Convert the train examples into indices.
    captions = [convert_tokens_to_indices(caption, vocab) for caption in captions]
    # Append start sentence and end sentence
    captions = [add_start_end_of_sentence(caption, vocab) for caption in captions]

    # Get lengths of captions
    lengths = [len(caption) for caption in captions]
    # pad_sequence function expects the input to be a tensor, so we turn x into one
    captions = [torch.LongTensor(caption_indices) for caption_indices in captions]
    padded_captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=vocab.word_to_idx[vocab.pad])

    # We are now ready to return our variables. The order we return our variables
    # here will match the order we read them in our training loop.
    return images, padded_captions, lengths
