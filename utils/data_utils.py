from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
import os, io, json, torch, re
import numpy as np
from utils.model_utils import OrderedCounter

class TextDataLoader(Dataset):
    def __init__(self, data_name, data_dir, split, create_data, **kwargs):
        super(DataLoader, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 3)

        self.raw_data_path = os.path.join(data_dir, data_name+'.'+split+'.txt')
        self.data_file = data_name+'.'+split+'.json'
        self.vocab_file = data_name+'.vocab.json'

        if create_data:
            print("Creating new {} {} data.".format(split.upper()), data_name)
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("{} preprocessed file not found at {}. Creating new.".format(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def _load_data(self,vocab=True):
        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)

        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self,vocab=True):
         with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        tokenizer = TweetTokenizer(preserve_case=False)
        data = defaultdict(dict)

        with open(self.raw_data_path, 'r') as file:
            for i, line in enumerate(file):
                words = tokenizer.tokenize(line)
                input = ['<sos>'] + words
                input = input[:self.max_sequence_length]
                target = words[:self.max_sequence_length-1]
                target = target + ['<eos>']

                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)

                input.extend(['<pad>'] * (self.max_sequence_length-length))
                target.extend(['<pad>'] * (self.max_sequence_length-length))

                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):
        assert self.split == 'train', "Vocablurary can only be created for training file."

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(self.raw_data_path, 'r') as file:

            for i, line in enumerate(file):
                words = tokenizer.tokenize(line)
                w2c.update(words)

            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)
        assert len(w2i) == len(i2w)
        print("Vocablurary of {} keys created.".format(len(w2i)))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w



def clean_sentences(file):
    try:
        f = open(file,'r')
        cleaned = []
        remove = ['<eos>','“','”','"','"“','“','<eos>"','•','·','●','-']
        generated_sentences = f.readlines()
        for idx,line in enumerate(generated_sentences):
            line = re.sub(r'<eos>',"",line)
            splitted = line.split()
            for word in splitted:
                if word in remove:
                    splitted.remove(word)
            string = ""
            punc = [',','.',"’",'!','?',"'",'(',')', ':']
            apostrophe1 = "’"
            apostrophe2 = "'"
            apostrophe_check=0
            line_begin = 1

            for clrs in splitted:
                if clrs in punc:
                    if clrs == apostrophe1 or clrs == apostrophe2:
                        string += "'"
                        apostrophe_check = 1
                    else:
                        string += clr
                else:
                    if apostrophe_check == 0:
                        if line_begin == 1:
                            string += clrs
                            line_begin = 0
                        else:
                            string += " " + clrs
                    else:
                        string += clrs
                        apostrophe_check = 0
            if 'caz' in string:
                string = string.replace('caz', 'c az')
            if 'cparçalı' in string:
                string = string.replace('cparçalı','c parçalı')
            if '(' in string:
                string = string.replace('( ','(')
            if 'i̇' in string:
                string = string.replace('i̇','i')
            if '‘ ' in string:
                string = string.replace('‘ ',"'")
            if '‘' in string:
                string = string.replace('‘',"'")

            cleaned.append(string)
            string = ""
        return cleaned
    except:
        raise FileExistsError()


def select_k(samples: list, k=1, unk_threshold = 1, repeat_threshold = 3, printall = False):
    if printall:
        print(*samples,sep='\n')

    sentences = []
    for sentence in samples:
        wordfreq = defaultdict(int)
        not_necessary = ['','<eos>','\“','\”','\"',',','.','?','!',':']
        if '<eos>' in sentence:
            words = sentence.split()
            for raw_word in words:
                if raw_word not in not_necessary:
                    wordfreq[raw_word] +=1
            if len(wordfreq) != 0:
                approval_counter = 0
                general_counter = 0
                if wordfreq['<unk>'] <= unk_threshold:
                    for word, freq in wordfreq.items():
                        if word not in not_necessary:
                            general_counter += 1
                            if freq <=repeat_threshold:
                                approval_counter += 1
                if (approval_counter == general_counter) and (general_counter != 0 and approval_counter!=0) and (len(sentence.split())  > k):
                    sentences.append(sentence)
                approval_counter = 0
                general_counter = 0
    return sentences

def save_sentences(sentences: list, file: str, population=False):
    if not population:
        for sentence in sentences:
            try:
                with open(file, 'a') as f:
                    f.write(sentence+"\n")
            except:
                raise FileExistsError()
    else:
        for samples in sentences:
            for sentence in samples:
                try:
                    with open(file, 'a') as f:
                        f.write(sentence+"\n")
                except:
                    raise FileExistsError()

def clear_duplicates(file: str):
    try:
        with open(file, "r") as f:
            lines = [line.rstrip() for line in f]
        lines = list(dict.fromkeys(lines))

        with open("augmentations.txt", "w") as f:
            for row in lines:
                s = "".join(map(str, row))
                file.write(s+'\n')
        print(f"Total augmented sentences: {len(lines)}")
    except:
        OSError()

