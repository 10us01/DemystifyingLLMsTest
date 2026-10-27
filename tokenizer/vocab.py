"""Vocabulary manager. Assigned: Sarthak. Bug: E4"""
from typing import Dict, Optional
from config.constants import PAD_TOKEN_ID, UNK_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)

        self.pad_token_id = self._add_special_token(PAD_TOKEN, PAD_TOKEN_ID)
        self.unk_token_id = self._add_special_token(UNK_TOKEN, UNK_TOKEN_ID)
        self.bos_token_id = self._add_special_token(BOS_TOKEN, BOS_TOKEN_ID)
        self.eos_token_id = self._add_special_token(EOS_TOKEN, EOS_TOKEN_ID)

    def _add_special_token(self, token: str, preferred_id: int) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if preferred_id in self._id_to_token:
            preferred_id = next(
                (candidate for candidate in range(max(self._id_to_token, default=-1) + 2)
                 if candidate not in self._id_to_token),
                max(self._id_to_token, default=-1) + 1,
            )
        return self.add_token(token, preferred_id)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = len(self._token_to_id)
            while idx in self._id_to_token:
                idx += 1
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            raise ValueError(f"token ID must be a non-negative integer, got {idx}")
        if idx in self._id_to_token:
            raise ValueError(f"token ID {idx} is already assigned to {self._id_to_token[idx]!r}")
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        return self.add_token(token)

    def id_to_token(self, idx: int) -> str:
        return self._id_to_token.get(idx, UNK_TOKEN)

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
