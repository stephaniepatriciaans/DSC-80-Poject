# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    if not hasattr(get_book, "_delay"):
        robots_url = "https://www.gutenberg.org/robots.txt"
        try:
            r = requests.get(robots_url)
            delay = 0.5
            for line in r.text.splitlines():
                if line.strip().lower().startswith("crawl-delay"):
                    try:
                        delay = float(line.split(":",1)[1].strip())
                    except ValueError:
                        pass
                    break
        except Exception:
            delay = 0.5
        get_book._delay = delay

    if not hasattr(get_book, "_last_request_time"):
        get_book._last_request_time = 0.0
    elapsed = time.time() - get_book._last_request_time
    
    if elapsed < get_book._delay:
        time.sleep(get_book._delay - elapsed)

    response = requests.get(url)
    get_book._last_request_time = time.time()
    text = response.text.replace("\r\n", "\n")

    markers = list(re.finditer(r"\*\*\*.*?\*\*\*", text))
    if len(markers) >= 2:
        return text[ markers[0].end() : markers[-1].start() ]
    else:
        return text



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    text = book_string.replace('\r\n', '\n')
    text = text.strip()
    pc = '\x00'
    text = re.sub(r'\n(?:[ \t]*\n)+', pc, text)
    text = '\x02' + text + '\x03'
    text = text.replace(pc, '\x03\x02')
    tokens = re.findall(r'\x02|\x03|\w+|[^\s\w]', text)
    return tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM:
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        unique_tokens = pd.Series(tokens).unique()
        prob = 1 / len(unique_tokens)
        return pd.Series(prob, index=unique_tokens)
    
    def probability(self, words):
        for word in words:
            if word not in self.mdl:
                return 0.0
        return np.prod([self.mdl[word] for word in words])
    
    def sample(self, M):
        return ' '.join(np.random.choice(self.mdl.index, size=M, p=self.mdl.values))


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM:
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        return pd.Series(tokens).value_counts(normalize=True)
    
    def probability(self, words):
        prob = 1.0
        for word in words:
            if word not in self.mdl:
                return 0.0
            prob *= self.mdl[word]
        return prob
    
    def sample(self, M):
        return ' '.join(np.random.choice(self.mdl.index, size=M, p=self.mdl.values))


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

from collections import Counter

class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)


    def create_ngrams(self, tokens):
        return [tuple(tokens[i : i + self.N]) for i in range(len(tokens) - self.N + 1)]
 
    def train(self, ngrams):
        ngram_count  = Counter(ngrams)
        n1gram_count = Counter(ng[:-1] for ng in ngrams)

        row = []
        for ngram, count in ngram_count.items():
            prefix = ngram[:-1]
            row.append({
                'ngram':  ngram,
                'n1gram': prefix,
                'prob':   count / n1gram_count[prefix]
            })

        df = pd.DataFrame(row)
        df = df.sort_values(by=['prob', 'ngram'],ascending=[True, True]).reset_index(drop=True)
        return df

    def probability(self, words):
        if not words:
            return 0.0

        probability = 1.0
        n = self.N

        maximum = min(n - 1, len(words))
        for order in range(1, maximum + 1):
            current_model = self
            for _ in range(n - order):
                current_model = current_model.prev_mdl

            if order == 1:
                token = words[0]
                if token not in current_model.mdl.index:
                    return 0.0
                probability *= current_model.mdl.loc[token]
            else:
                prefix = tuple(words[:order])
                match = current_model.mdl.loc[current_model.mdl['ngram'] == prefix, 'prob']
                if match.empty:
                    return 0.0
                probability *= match.iloc[0]

        for i in range(len(words) - n + 1):
            ngram = tuple(words[i : i + n])
            match = self.mdl.loc[self.mdl['ngram'] == ngram, 'prob']
            if match.empty:
                return 0.0
            probability *= match.iloc[0]

        return probability

    def sample(self, M):
        history = ['\x02']
        base = 2
        for _ in range(M-1):
            if base < self.N:
                current = self
                
                for _ in range(self.N - base):
                    current = current.prev_mdl
                    
                context = tuple(history[0:base-1])
                token = current.mdl[current.mdl['n1gram'] == context]
                if token.empty:
                    history.append('\x03')
                else:
                    next = token['ngram'].apply(lambda ng: ng[-1]).values
                    probs = token['prob'].values
                    choice = np.random.choice(next, p=probs / probs.sum())
                    history.append(choice)
                    base += 1
                
            else:
                context = tuple(history[-(self.N - 1):])
                token = self.mdl[self.mdl['n1gram'] == context]
                
                if token.empty:
                    history.append('\x03')
                else:
                    next = token['ngram'].apply(lambda ng: ng[-1]).values
                    probs = token['prob'].values
                    choice = np.random.choice(next, p=probs / probs.sum())
                    history.append(choice)

        while len(history) - 1 < M:
            history.append('\x03')

        return ' '.join(history)