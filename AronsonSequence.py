from enum import Enum
from num2words import num2words

STR_START = " is the "
STR_END = " letter"


def n2w(n):
    os = num2words(n, ordinal=True).replace(" and", "")
    return os.replace(", ", "").replace(" ", "").replace("-", "")


class ReferralType(Enum):
    BACKWARD = 1
    SELF = 2
    FORWARD = 3


class AronsonSequence:
    def __init__(self, letter: str, indices: list[int], forward: bool):
        self.letter = letter.lower()
        self.indices = indices
        self.forward = forward
        self.human_readable = self._build_string()
        self.string_repr = self.human_readable.replace(" ", "").replace(",", "")
        self.referral_dict = self._build_referral_dict()

    def _build_string(self):
        indices = self.indices if self.forward else self.indices[::-1]
        parts = [self.letter + STR_START] + [n2w(i) for i in indices] + [STR_END]
        return ''.join(parts)

    def _build_referral_dict(self):
        d = {}
        for idx in self.indices:
            if idx in d:
                # in case we are setting
                continue
            target_idx = idx - 1
            rep = n2w(idx)
            pos = self.string_repr.find(rep)
            if target_idx < pos:
                d[idx] = ReferralType.BACKWARD
            elif pos <= target_idx < pos + len(rep):
                d[idx] = ReferralType.SELF
            else:
                d[idx] = ReferralType.FORWARD
        return d

    def has_forward_referring(self):
        return any(ref == ReferralType.FORWARD for ref in self.referral_dict.values())

    def get_string_repr(self):
        """Getter for string representation."""
        return self.string_repr

    def get_direction(self):
        """Getter for the direction of the sequence (forward or not)."""
        return self.forward

    def get_indices(self):
        """Getter for the sequence indices"""
        return self.indices
    def set_indices(self, new_indices: list[int]):
        """Setter for indices. Updates string_repr, human_readable, and referral_dict."""
        self.indices = new_indices
        self.human_readable = self._build_string()
        self.string_repr = self.human_readable.replace(" ", "").replace(",", "")
        self.referral_dict = self._build_referral_dict()


    def __repr__(self):
        return self.human_readable

    def __eq__(self, other):
        return isinstance(other, AronsonSequence) and self.indices == other.indices and self.letter == other.letter \
            and self.forward == other.direction

    def __hash__(self):
        return hash((tuple(self.indices), self.letter, self.forward))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        """
        Returns the index at the specified position in the Aronson sequence.

        @param index The position of the index to return from the Aronson sequence.
        @return The index at the specified position.
        """
        if index < 0 or index >= len(self.indices):
            raise IndexError("Index out of range")
        return self.indices[index]
