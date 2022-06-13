import re


separator = ' +++$+++ '
start_symbol = '^'
end_symbol = '$'
padding_symbol = '*'
unknown_symbol = '#'


def read_lines(filename, mode=''):
    lines = open(filename, encoding='iso-8859-1').readlines()
    for line in lines:
        line = line.strip().split(separator)
        if mode == 'dictionary':
            yield line[-1].strip().lower()
        elif mode == 'lines2text':
            yield line[0], line[-1].strip().lower()
        else:
            line = line[-1].strip()
            line = re.sub(r'[^\w,]', '', line).split(',')
            yield line


def remove_characters(line):
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    line = re.sub('\^', '', line)
    line = re.sub('\#', '', line)
    line = re.sub('\$', '', line)
    line = re.sub('\*', '', line)
    return line


def tokenizer(text):
    words_ = []
    _word_split = re.compile("([.,!?\-<>:;)(])")
    for fragment in text.strip().lower().split():
        for token in re.split(_word_split, fragment):
            if not token:
                continue
            words_.append(token)
    return words_


def create_vocabulary(filename):
    word2id = {padding_symbol: 0, start_symbol: 1, end_symbol: 2, unknown_symbol: 3}
    next_id = max([i for i in word2id.values()]) + 1
    for text in read_lines(filename, mode='dictionary'):
        words = tokenizer(remove_characters(text))
        for word in words:
            if word not in word2id.keys():
                word2id[word] = next_id
                next_id += 1
    return word2id


def get_lines(filename):
    lin2text = {}
    for lin, text in read_lines(filename, mode='lines2text'):
        lin2text[lin] = text
    return lin2text


def prepare_dialog(filename):
    memory_n, input_sequence, output_sequence = {}, {}, {}
    for l in read_lines(filename):
        length = len(l)
        if length == 2:
            if 0 in input_sequence.keys():
                input_sequence[0].append(l[0])
                output_sequence[0].append(l[1])
            else:
                input_sequence[0] = [l[0]]
                output_sequence[0] = [l[1]]
        elif length > 2:
            if 0 in input_sequence.keys():
                input_sequence[0].append(l[0])
                output_sequence[0].append(l[1])
            else:
                input_sequence[0] = [l[0]]
                output_sequence[0] = [l[1]]
            position = 0
            while position <= length - 2:
                memory = 1
                while position + memory - 1 < length - 2:
                    i = 0
                    memory_list = []
                    while i < memory:
                        memory_list.append(l[position + i])
                        i += 1
                    if memory in memory_n.keys():
                        memory_n[memory].append(memory_list)
                        input_sequence[memory].append(l[position + i])
                        output_sequence[memory].append(l[position + i + 1])
                    else:
                        memory_n[memory] = [memory_list]
                        input_sequence[memory] = [l[position + i]]
                        output_sequence[memory] = [l[position + i + 1]]
                    memory += 1
                position += 1
    return input_sequence, output_sequence, memory_n


def sentence_to_ids(sentence, word2id, padded_len=0):
    sent_ids = []
    sentence = remove_characters(sentence)
    sentence = tokenizer(sentence)
    if padded_len == 0:
        il = len(sentence)
    else:
        il = min(len(sentence), padded_len - 1)
    for x in range(0, il):
        try:
            sent_ids.append(word2id[sentence[x]])
        except:
            sent_ids.append(word2id['#'])

    sent_ids.append(word2id['$'])
    sent_len = len(sent_ids)
    while len(sent_ids) < padded_len:
        sent_ids.append(word2id['*'])
    return sent_ids, sent_len


def batch_to_ids(sentences, word2id, max_len=0):
    if max_len > 0:
        max_len_in_batch = min(max(len(tokenizer(s)) for s in sentences) + 1, max_len)
    else:
        max_len_in_batch = max(len(tokenizer(s)) for s in sentences) + 1
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, word2id, max_len_in_batch)
        batch_ids.append(ids)
        batch_ids_len.append(ids_len)
    return batch_ids, batch_ids_len


def generate_batches_no_memory(input_samples, output_samples, get_movie_lines, batch_size=64):
    X, Y = [], []
    for i, (x, y) in enumerate(zip(input_samples, output_samples), 1):
        X.append(get_movie_lines[x])
        Y.append(get_movie_lines[y])
        if i % batch_size == 0:
            yield X, Y
            X, Y = [], []
    if X and Y:
        yield X, Y


def generate_batches_memory(input_samples, output_samples, memory_samples, get_movie_lines, batch_size=64):
    input_seq, output_seq, memory = [], [], []
    for i, (x, y, z) in enumerate(zip(input_samples, output_samples, memory_samples), 1):
        input_seq.append(get_movie_lines[x])
        output_seq.append(get_movie_lines[y])
        for pos in range(0, len(z)):
            memory.append(get_movie_lines[z[pos]])
        if i % batch_size == 0:
            yield input_seq, output_seq, memory
            input_seq, output_seq, memory = [], [], []
    if input_seq and output_seq and memory:
        yield input_seq, output_seq, memory


def ids_to_sentence(ids, id2word):
    return [id2word[i] for i in ids]