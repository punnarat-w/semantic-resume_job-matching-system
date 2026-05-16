from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
CLS_TOKEN = "<cls>"
SEP_TOKEN = "<sep>"


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9+#.']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1, max_size: int | None = None) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))

        token_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            CLS_TOKEN: 2,
            SEP_TOKEN: 3,
        }
        for token, count in counter.most_common():
            if max_size is not None and len(token_to_id) >= max_size:
                break
            if count >= min_freq and token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id=token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    def encode_pair(self, requirement: str, resume: str, max_length: int) -> list[int]:
        tokens = [CLS_TOKEN]
        tokens.extend(tokenize(requirement))
        tokens.append(SEP_TOKEN)
        tokens.extend(tokenize(resume))
        ids = [self.token_to_id.get(token, self.token_to_id[UNK_TOKEN]) for token in tokens]
        ids = ids[:max_length]
        if len(ids) < max_length:
            ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids

