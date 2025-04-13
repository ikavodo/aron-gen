from itertools import islice, combinations, permutations
from AronsonSequence import AronsonSequence, n2w, Refer, Direction, LEN_PREFIX, LEN_SUFFIX
from collections import defaultdict
from typing import Callable

# upper bound for searching for singleton Aronson sequences
SINGLETON_UPPER = 40
# approximation of max ordinal lengths per number of bits
ORD_TABLE = {i: 10 * (i + 1) for i in range(12)}


# Exception classes
class VerificationError(Exception):
    """
    Custom exception raised when the Aronson sequence verification fails.

    :param message: The error message to be shown.
    :param input_seq: The input data that caused the failure.
    """

    def __init__(self, message="Verifier failed", input_seq=None):
        self.message = message
        self.input_seq = input_seq
        super().__init__(self.message)

    def __str__(self):
        # will this show representation of input_seq?
        return f"{self.message}: {self.input_seq}"


class GenError(Exception):
    """
        Custom exception raised when generating from the input sequence is impossible.

        :param message: The error message to be shown.
        :param seq_len: expected length of sequence
        """

    def __init__(self, message="Generating failed", seq_len=None):
        self.message = message
        self.len = seq_len
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}: could not generate {self.len} elements"


# This class generates correct Aronson sequences via various generation rules.
class AronsonSet:
    """
    Class for generating correct AronsonSequence objects.
    Notice we have for any instance S this class S ∈ Aₓ(→/←), where x is some letter and arrow refers to
    direction (see https://ikavodo.github.io/aronson-1/)
    :param letter: The letter used for generating sequences.
    :param direction: sequences generation direction.
    """

    # allow to generate from a sequence?
    def __init__(self, letter: str, direction: Direction):
        self.letter = letter.lower()  # Letter used for generating sequences
        self.direction = direction  # Sequence direction
        self.sets_by_iteration = defaultdict(set)
        self.cur_iter = 0
        self.seen_seqs = set(AronsonSequence(self.letter, [], self.direction))
        # Each AronsonSet object contains the matching empty AronsonSequence object
        self.sets_by_iteration[self.cur_iter] = self.seen_seqs.copy()

    @property
    def display_letter(self):
        return self.letter.upper()

    @classmethod
    def from_sequence(cls, seq: AronsonSequence):
        """
        constructor from AronsonSequence. Good for checking generation rules on a particular sequence
        :param seq: AronsonSequence
        :return: instance
        """
        # (inefficient to construct before checking correctness, change this later)
        obj = cls(seq.get_letter(), seq.get_direction())
        if not obj.is_correct(seq):
            # Let the user know
            raise VerificationError(input_seq=seq)
        obj.add_sequences({seq})  # does verification, throws VerificationError if sequence in correct
        return obj

    def is_correct(self, seq: AronsonSequence):
        """
        wrapper for _ismember(), allows for verification of sequence
        :param seq: AronsonSequence to be verified
        :return: True/False
        """
        return (
                seq.get_letter() == self.display_letter and
                seq.get_direction() == self.direction and self.is_correct(seq)
        )

    def is_complete(self, seq: AronsonSequence):
        """
        wrapper for _ismember(), allows for checking completeness of sequence
        :param seq: AronsonSequence to be checked
        :return: True/False
        """
        return (
                seq.get_letter() == self.display_letter and
                seq.get_direction() == self.direction and self.is_complete(seq)
        )

    def add_sequences(self, seqs: set[AronsonSequence]):
        """
        Add a new sequence to set of seen sequences
        :param seqs:
        :return:
        """
        valid_seqs = {seq for seq in seqs if seq not in self.seen_seqs and self.is_correct(seq)}
        self.seen_seqs.update(valid_seqs)

    # Don't use this in case there is anything forward-referring, as that element will need to be fixed! Use
    # forward_fix() for that.
    @staticmethod
    def backward_search(seq: AronsonSequence):
        """
        Find all occurrences missing in sentence prefix and
        :param seq:
        :return:
        """
        # does nothing if sequence is prefix complete!
        occurrences = seq.get_prefix_missing()
        return {seq.copy().append_elements(occ) for occ in occurrences}

    def _agen(self, seq: AronsonSequence):
        """
        Internal generator for backwards-based generation.

        :param seq: An optional AronsonSequence to generate from.
        :return: A generator yielding new indices for the sequence.
        """
        idx = seq.get_prefix() if not seq.is_empty() else 0
        s = seq.get_sentence()
        s = s[idx:-LEN_SUFFIX] if self.direction == Direction.FORWARD else s[LEN_PREFIX:-idx][::-1]
        while True:
            # generator yields as many indices as required
            idx_rel = 1 + s.find(self.letter)  # Find the relative position of the letter
            if idx_rel <= 0:  # Letter not found in string buffer, StopIter
                break
            idx += idx_rel
            yield idx
            s = s[idx_rel:] + (n2w(idx) if self.direction == Direction.FORWARD else n2w(idx)[::-1])

    # For good old well-behaved sequences (no forward referring)
    def backward_generate(self, n: int, seq: AronsonSequence = None):

        # generate empty AronsonSequence if no argument
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        # generate as many new elements as necessary
        new_elements = list(islice(self._agen(seq=seq), n - len(seq)))
        # update sequence
        seq.append_elements(new_elements)

        if len(seq) < n:
            # could not extend sequence up to desired length
            raise GenError(seq_len=n)
        # return set for add_sequences() method
        return {seq}

    # Wrapper for backward_generate() method with default arguments
    def generate_aronson(self, n: int) -> set[AronsonSequence]:
        # generates the standard, prefix-complete Aronson sequence
        return self.backward_generate(n)

    def swap(self, seq: AronsonSequence = None):
        """
        are allowed to swap two elements if the position to which any element points is unaffected by the swap
        :param seq: AronsonSequence object
        :return: set of newly generated sets in which swapping is legal
        """
        new_sets = set()
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        for i, j in combinations(range(len(seq)), 2):
            # not generator- is exhumed after one iteration!
            ranges = [seq.get_range(x) for x in [i, j]]
            # anything in between is affected
            range_of_effect = range(min(r.start for r in ranges), max(r.stop for r in ranges))
            if all(x not in range_of_effect for x in seq.get_elements()):
                swapped = seq.get_elements()
                swapped[i], swapped[j] = swapped[j], swapped[i]
                new_sets.add(swapped)
        return new_sets

    # Idea: try to generate something at the end of the sequence.
    # For sequences with only backward and self-referring elements (otherwise use forward_fix)
    def forward_generate(self, seq: AronsonSequence):
        """
        Generate by searching within a bounded search space for valid ordinals to append
        :param seq:
        :return:
        """
        if seq.is_empty():
            # for generating all singletons
            lower_bound = 0
        else:
            lower_bound = len(seq.get_sentence()[:-LEN_SUFFIX]) if seq.direction == Direction.FORWARD else \
                len(seq.get_sentence()[LEN_PREFIX:])
        # maximum length of added ordinal
        ord_key = len(str((seq.get_prefix())))
        upper_bound = len(seq.get_sentence()) + ORD_TABLE[ord_key]
        new_seqs = set()
        for elem in range(lower_bound, upper_bound):
            seq_cpy = seq.copy()
            candidate = seq_cpy.append_elements(elem)
            if self.is_correct(candidate):
                new_seqs.add(candidate)
        return new_seqs

    def generate_singletons(self):
        """
        Generates all single-index valid AronsonSequences (singletons).

        :return: set of all correct singleton AronsonSequences
        """
        return self.forward_generate(AronsonSequence(self.letter, [], self.direction))

    def forward_fix(self, seq: AronsonSequence) -> set[AronsonSequence]:
        new_seqs = set()
        elements = seq.get_elements()
        forward_refs = {x for x in elements if seq.get_ref(x) == Refer.FORWARD and all(
            y not in range(min(seq.get_range(x)), len(seq.get_sentence())) for y in elements if y != x)}

        occurrences = seq.get_occurences() - forward_refs

        # probably only one such element, if at all
        for elem in forward_refs:
            for occ in occurrences:
                # generate new sequence by appending a missing occurrence
                seq_cpy = seq.copy()
                seq_cpy.append_elements(occ)

                # Try fixing the forward-ref element. Notice that candidate == elem at some point, meaning we also
                # check correctness of elem itself, while also taking into account other possibilities
                for candidate in range(min(seq.get_range(elem)), len(seq.get_sentence())):
                    # shallow copy
                    new_elements = seq.get_elements().copy()
                    new_elements[new_elements.index(elem)] = candidate
                    new_seq = AronsonSequence(seq.letter, new_elements, self.direction)
                    if self.is_correct(new_seq):
                        new_seqs.add(new_seq)

        return new_seqs

    # backward_search, backward_generate, swap, forward_generate, forward_fix
    def generate_from_rules(self, n_iterations: int):
        if n_iterations < 0:
            raise ValueError("Num of iterations must be non-negative")

        if n_iterations == 0:
            # do nothing
            return

        while self.cur_iter < n_iterations:
            cur_seqs = set()
            prev_seqs = self.sets_by_iteration[self.cur_iter]

            for seq in prev_seqs:
                if seq.is_empty():
                    cur_seqs.update(self.generate_singletons())
                elif seq.has_forward_referring():
                    cur_seqs.update(self.forward_fix(seq))
                else:
                    cur_seqs.update(self.backward_search(seq))
                    cur_seqs.update(self.backward_generate(1, seq))
                    cur_seqs.update(self.forward_generate(seq))
                # can do this either way
                cur_seqs.update(self.swap(seq))

            self.cur_iter += 1
            self.sets_by_iteration[self.cur_iter] = cur_seqs
            # total sets generated so far
            self.seen_seqs.update(cur_seqs)

    def get_seen_seqs(self):
        return self.seen_seqs

    # For generating ground truths up to n=3 and comparing missing sets in other implementation
    def generate_brute_force(self, n_iterations: int):
        def generate_unique_lists(n, bound):
            if n > bound:
                raise ValueError("n can't be greater than the number of available unique values")
            return [list(p) for p in permutations(range(1, bound + 1), n)]  # Start from 1

        if n_iterations < 0:
            raise ValueError("Num of iterations must be non-negative")

        if n_iterations == 0:
            # do nothing
            return

        cur_ord_key = 1  # just in case. For 0 -> upper is
        while self.cur_iter < n_iterations:
            # enlarge search range
            self.cur_iter += 1
            # grows linearly with iterations
            upper_bound = self.cur_iter * ORD_TABLE[cur_ord_key] + 2 * max(LEN_PREFIX, LEN_SUFFIX)
            if upper_bound >= 10 ** (cur_ord_key + 1):
                cur_ord_key += 1
            seqs = (AronsonSequence(self.letter, list(perm), self.direction) for perm in
                    generate_unique_lists(self.cur_iter, upper_bound))
            cur_seqs = {s for s in seqs if self.is_correct(s)}
            self.sets_by_iteration[self.cur_iter] = cur_seqs
            # total sets generated so far
            self.seen_seqs.update(cur_seqs)

    @staticmethod
    def _set_operation_core(set1: 'AronsonSet', set2: 'AronsonSet', n: int, ignore_letters: bool,
                            set_op: Callable[[set, set], set]):
        """
        Core logic for set operations on AronsonSets.
        """
        set1.generate_from_rules(n)
        set2.generate_from_rules(n)

        get_repr = (lambda s: s.get_elements()) if ignore_letters else (lambda s: s)

        elems1 = {get_repr(s) for s in set1.get_seen_seqs()}
        elems2 = {get_repr(s) for s in set2.get_seen_seqs()}

        combined = set_op(elems1, elems2)

        if ignore_letters:
            letter = 'χ'  # or 'Π'
            return {
                (AronsonSequence(letter, el, Direction.FORWARD),
                 AronsonSequence(letter, el, Direction.BACKWARD))
                for el in combined
            }

        if set1.display_letter != set2.display_letter:
            raise ValueError("Mismatched letters. Use ignore_letters=True to override.")

        return {
            (AronsonSequence(set1.display_letter, el, Direction.FORWARD),
             AronsonSequence(set1.display_letter, el, Direction.BACKWARD))
            for el in combined
        }

    # Set operations utilize _set_operation_core() helper method
    # Intersection
    def intersect_aronson_sets(self, other: 'AronsonSet', n: int, ignore_letters: bool = False):
        return self._set_operation_core(self, other, n, ignore_letters, set.intersection)

    @staticmethod
    def intersect_aronson_sequences(seq1: AronsonSequence, seq2: AronsonSequence, n: int, ignore_letters: bool = False):
        set1 = AronsonSet.from_sequence(seq1)
        set2 = AronsonSet.from_sequence(seq2)
        return AronsonSet._set_operation_core(set1, set2, n, ignore_letters, set.intersection)

    # Union
    def union_aronson_sets(self, other: 'AronsonSet', n: int, ignore_letters: bool = False):
        return self._set_operation_core(self, other, n, ignore_letters, set.union)

    @staticmethod
    def union_aronson_sequences(seq1: AronsonSequence, seq2: AronsonSequence, n: int, ignore_letters: bool = False):
        set1 = AronsonSet.from_sequence(seq1)
        set2 = AronsonSet.from_sequence(seq2)
        return AronsonSet._set_operation_core(set1, set2, n, ignore_letters, set.union)

    # Difference
    def difference_aronson_sets(self, other: 'AronsonSet', n: int, ignore_letters: bool = False):
        return self._set_operation_core(self, other, n, ignore_letters, set.difference)

    @staticmethod
    def difference_aronson_sequences(seq1: AronsonSequence, seq2: AronsonSequence, n: int,
                                     ignore_letters: bool = False):
        set1 = AronsonSet.from_sequence(seq1)
        set2 = AronsonSet.from_sequence(seq2)
        return AronsonSet._set_operation_core(set1, set2, n, ignore_letters, set.difference)
