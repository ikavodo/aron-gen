from itertools import islice, cycle
from AronsonSequence import AronsonSequence, n2w, PREFIX, SUFFIX, ReferralType, Direction
from typing import Callable

# upper bound for searching for singleton Aronson sequences
SINGLETON_UPPER = 40


# Exception classes
class VerificationError(Exception):
    """
    Custom exception raised when the Aronson sequence verification fails.

    :param message: The error message to be shown.
    :param input_sentence: The input data that caused the failure.
    """

    def __init__(self, message="Verifier failed", input_sentence=None):
        self.message = message
        self.input_sentence = input_sentence
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}: {self.input_sentence}"


class GenError(Exception):
    """
        Custom exception raised when generating from the input sequence is impossible.

        :param message: The error message to be shown.
        :param input_sentence: The input data that caused the failure.
        """

    def __init__(self, message="Generating failed", input_sentence=None, forward_ref=False, seq_len=None):
        self.message = message
        self.input_sentence = input_sentence
        self.forward_ref = forward_ref
        self.len = seq_len
        super().__init__(self.message)

    def __str__(self):
        if self.forward_ref:
            suffix = f"has type {ReferralType.FORWARD}"
        elif self.len:
            suffix = f"could not generate {self.len} elements"
        else:
            suffix = ""
        return f"{self.message}: {self.input_sentence} {suffix}"


# This class currently has one goal: enumerating all sequences within set with length up to n
# this should be done in iterations using four generation rules: backwards (gen_sequence), forwards (
# Others can be extended by one element using Aronson algorithm (backward-refer).
# For example: enumerate() -> AronsonSequence('t', [1], True), AronsonSequence('t', [4], True),
# AronsonSequence('t', [11], True)
# Use partition of referring to make equivalence sets?
# But how to make sure we generate *all* sequences of length n without reverting to brute-force search?
# Need to encode logic somehow.
# Be allowed to find intersection of AronsonSets with different letters?
class AronsonSet:
    """
    Class for generating AronsonSequence objects. Is equivalent to sets A_x(->) or A_x(<-) where x is some letter.
    Provides methods for generating Aronson sequences and verifying their correctness.
    AronsonSequence is primitive class
    :param letter: The letter used for generating sequences.
    :param direction: sequences generation direction.
    """

    # allow to generate from a sequence?
    def __init__(self, letter: str, direction: Direction):
        self.letter = letter.lower()  # Letter used for generating sequences
        self.direction = direction  # Sequence direction
        self.seen_seqs = set()  # Sequences seen so far. This field may change

    @property
    def display_letter(self):
        return self.letter.upper()

    @classmethod
    def from_sequence(cls, seq: AronsonSequence):
        """
        constructor from AronsonSequence
        :param seq:
        :return: this instance
        """
        obj = cls(seq.get_letter(), seq.get_direction())
        obj.seen_seqs.add(seq)  # remember which seq generated class
        return obj

    def _ismember(self, seq: AronsonSequence, conditional: Callable[[AronsonSequence], bool]) -> bool:
        """
        helper for conditioning an AronsonSequence with regard to the current AronsonSet instance
        :param seq: AronsonSequence to be conditioned
        :param conditional: by which to condition
        :return: True/False
        """
        return (
                seq.get_letter() == self.display_letter and
                seq.get_direction() == self.direction and conditional(seq)
        )

    def is_correct(self, seq: AronsonSequence):
        """
        wrapper for _ismember(), allows for verification of sequence
        :param seq: AronsonSequence to be verified
        :return: True/False
        """
        return self._ismember(seq, AronsonSequence.is_correct)

    def is_complete(self, seq: AronsonSequence):
        """
        wrapper for _ismember(), allows for checking completeness of sequence
        :param seq: AronsonSequence to be checked
        :return: True/False
        """
        return self._ismember(seq, AronsonSequence.is_complete)

    def add_sequence(self, seq: AronsonSequence):
        """
        Add a new sequence to set of seen sequences
        :param seq:
        :return:
        """
        if seq in self.seen_seqs:
            # do nothing
            return
        if not self.is_correct(seq):
            raise VerificationError
        self.seen_seqs.add(seq)

    def gen_sequence(self, n: int, seq: AronsonSequence) -> AronsonSequence:
        """
        Generates a new AronsonSequence of length n either from scratch or by extending an existing sequence.

        :param n: The length of the desired AronsonSequence.
        :param seq: An optional input AronsonSequence to extend. If None, a new sequence is generated.
        :return: A new AronsonSequence object.
        """

        # do some error checking here
        # Use AronsonSequence as seed for generation
        if not self.is_correct(seq):  # is of correct letter and direction
            raise VerificationError(input_sentence=str(seq))  # Verifier failed
        elif seq.has_forward_referring() or seq.is_complete():
            # can't be extended
            raise GenError(input_sentence=seq, forward_ref=True)
        # extend input using generator
        new_elements = list(islice(self._agen(seq=seq), n - len(seq)))
        # update sequence
        seq.append_elements(new_elements)

        if len(seq) != n:
            # could not extend sequence or generate from scratch up to desired length
            raise GenError(input_sentence=seq, forward_ref=False, seq_len=n)
        # add to sequences generated so far.
        # Would be nice to easily retrieve this from a dictionary of AronsonSequences to lists of AronsonSequences!!!
        self.add_sequence(seq)
        return seq

    def _agen(self, seq: AronsonSequence = None):
        """
        Internal generator for generating indices based on the current sequence or from scratch.

        :param seq: An optional AronsonSequence to continue generating from.
        :return: A generator yielding new indices for the sequence.
        """
        if seq:
            # Start from the last index of the input sequence. Inspect this later
            idx = max(seq.get_elements())
            # slice string_repr to start from relevant string buffer, check is correct!
            s = seq.get_sentence()[idx:-len(SUFFIX.replace(" ", ""))]
        else:
            # generate from scratch
            idx = 0
            s = (self.letter + PREFIX).replace(" ", "") if self.direction == Direction.FORWARD else \
                SUFFIX[::-1].replace(" ", "")

        while True:
            idx_rel = 1 + s.find(self.letter)  # Find the relative position of the letter
            if idx_rel <= 0:  # Letter not found in string buffer
                break
            idx += idx_rel
            yield idx
            extend = n2w(idx) if self.direction == Direction.FORWARD else n2w(idx)[::-1]  # Extend the string buffer
            s = s[idx_rel:] + extend

    def generate_singletons(self):
        """
        Generates all single-index valid AronsonSequences (singletons).

        :return: A generator yielding valid singleton AronsonSequence objects.
        """
        for idx in range(1, SINGLETON_UPPER):
            candidate = AronsonSequence(self.letter, [idx], self.direction)
            try:
                # runs verifier on candidate
                self.add_sequence(candidate)
                yield candidate
            except VerificationError:
                # is not correct
                continue

    def generate_variations(self, n: int, seq: AronsonSequence = None):
        """
        Generates variations of AronsonSequences of length n, either from scratch or by modifying an existing sequence.

        :param n: The length of the AronsonSequences to generate.
        :param seq: An optional input AronsonSequence to modify. If None, sequences are generated from scratch.
        :return: A generator yielding new AronsonSequence variations.
        """
        # update this to take seen_seqs into account!
        # start from generating original sequence from input, or original Aronson sequence if no such
        orig = self.gen_sequence(n, seq)
        # used to generate orig first
        start_idx = -1
        # Stack keeps track of sequences generated so far
        stack = [(orig, start_idx)]
        while stack:
            # keep track of index used to generate current sequence
            cur, cur_idx = stack.pop()
            yield cur
            # generate variations
            for idx in range(cur_idx + 1, n - 1):
                elements = cur.get_elements()
                new_seq = AronsonSequence(seq.get_letter(), elements[:idx] + [elements[idx + 1]],
                                          seq.get_direction())
                try:
                    # sequence is of length less than n
                    extend = self.gen_sequence(n, new_seq)
                except VerificationError or GenError:
                    continue  # Skip invalid variations
                stack.append((extend, idx))

    def get_generator(self, n: int, seq: AronsonSequence = None):
        """
        Returns a generator for generating variations of AronsonSequences.

        :param n: The length of the sequences to generate.
        :param seq: An optional input AronsonSequence to modify. If None, sequences are generated from scratch.
        :return: A generator yielding new AronsonSequence variations.
        """
        return self.generate_variations(n, seq)

    def intersect_aronson_sets(self, other: 'AronsonSet', n: int, seq1: AronsonSequence = None,
                               seq2: AronsonSequence = None):
        """
        Computes the intersection of two AronsonSets by comparing their generated sequences and finding common subsets.

        :param other: The other AronsonSet to intersect with.
        :param n: The length of the sequences to generate.
        :param seq1: An optional input AronsonSequence for the first generator.
        :param seq2: An optional input AronsonSequence for the second generator.
        :return: A set of common valid AronsonSequences that belong to both sets.
        """
        # can use self.seen_seqs and other.seen_seqs here! More elegant
        self_set, other_set = set(), set()
        # generate all sequences up to length n from each generator
        for i in range(n):
            for seq1, seq2 in zip(self.generate_variations(i, seq1), other.get_generator(i, seq2)):
                self_set.update(seq1)
                other_set.update(seq2)
        # intersection of sequences generated
        common_subsets = self_set & other_set

        return {subset for subset in common_subsets if self.is_correct(subset) and other.is_correct(subset)}

    def union(self, other_set: 'AronsonSet', n: int):
        """
        Yields all unique AronsonSequences of length n from both sets in alternating order.

        :param other_set: The other AronsonSet to union with.
        :param n: The length of the sequences to generate.
        :return: A generator yielding unique AronsonSequences from both sets.
        """
        seen = set()
        generators = [self.get_generator(n), other_set.get_generator(n)]
        for g in cycle(generators):
            try:
                seq = next(g)
                if seq not in seen:
                    seen.add(seq)
                    yield seq
            except StopIteration:
                break

    def get_seen_seqs(self):
        return self.seen_seqs
