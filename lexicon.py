import os

NEGATORS = {
    'not', 'no', "n't", 'never', 'neither', 'nor', 'nobody',
    'nothing', 'nowhere', 'hardly', 'scarcely', 'barely',
    'without', 'lack', 'lacking'
}

class SentimentLexicon:
    def __init__(self, lexicon_dir):
        self.positive = set()
        self.negative = set()
        pos_path = os.path.join(lexicon_dir, 'positive-words.txt')
        neg_path = os.path.join(lexicon_dir, 'negative-words.txt')
        for path, target in [(pos_path, self.positive), (neg_path, self.negative)]:
            if os.path.exists(path):
                with open(path, encoding='latin-1') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith(';'):
                            target.add(line.lower())

    def polarity(self, word):
        w = word.lower()
        if w in self.positive:
            return 2
        elif w in self.negative:
            return 0
        return 1

    def build_features(self, tokens):
        n = len(tokens)
        polarities = [self.polarity(t) for t in tokens]
        negations = [0] * n
        for i, t in enumerate(tokens):
            w = t.lower()
            if w in NEGATORS or w.endswith("n't"):
                for j in range(i + 1, min(i + 4, n)):
                    negations[j] = 1
        return [polarities[i] * 2 + negations[i] for i in range(n)]
