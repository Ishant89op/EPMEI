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
            return 1
        elif w in self.negative:
            return -1
        return 0

    def build_features(self, tokens):
        """
        Features:
        0: Neutral / non-opinion (untouched)
        1: Positive
        2: Negated Positive
        3: Negative
        4: Negated Negative
        """
        n = len(tokens)
        raw_polarities = [self.polarity(t) for t in tokens]
        negations = [0] * n
        for i, t in enumerate(tokens):
            w = t.lower()
            if w in NEGATORS or w.endswith("n't"):
                for j in range(i + 1, min(i + 4, n)):
                    negations[j] = 1

        feats = [0] * n
        for i in range(n):
            pol = raw_polarities[i]
            if pol == 1:
                feats[i] = 2 if negations[i] else 1
            elif pol == -1:
                feats[i] = 4 if negations[i] else 3
            else:
                feats[i] = 0
        return feats
