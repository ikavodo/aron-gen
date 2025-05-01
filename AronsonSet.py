from enum import Enum
from itertools import islice, combinations, permutations
from math import log2, ceil

from AronsonSequence import AronsonSequence, Direction, LEN_PREFIX, LEN_SUFFIX
from collections import defaultdict
from typing import Callable
from functools import reduce
from contextlib import suppress

# global: dictionary with maximum ordinal lengths per number of bits in decimal representation
ORD_TABLE = {i + 1: j for i, j in enumerate([7, 14, 26, 39, 45, 56, 69, 75, 87, 99])}
# initial key for maximum ordinal length
ORD_INITIAL = 2


# Exception classes


class VerificationError(Exception):
    """
    Custom exception raised when AronsonSequence verification (is_correct()) fails.
    :param message: The error message to be shown.
    :param input_seq: The input data that caused the failure.
    """

    def __init__(self, message="Verifier failed", input_seq=None):
        self.message = message
        self.input_seq = input_seq
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}: {self.input_seq}"


class ErrorMode(Enum):
    """
    Enum for modes

    BACKWARD: Refers to a previous position.
    SELF: Refers to its own position.
    FORWARD: Refers to a later position.
    """
    ITER_MODE = 1
    ELEM_MODE = 2


class GenError(Exception):
    """
        Custom exception raised when generating from the input sequence is impossible.
        :param message: The error message to be shown.
        :param n: Num of iterations or num of elements, depending on error mode
        """

    def __init__(self, n, message="Generating failed", error_mode=ErrorMode.ITER_MODE):
        self.n = n
        self.message = message
        self.error_mode = error_mode
        super().__init__(self.message)

    def __str__(self):
        if self.error_mode == ErrorMode.ITER_MODE:
            return f"{self.message}: converged after {self.n} iterations"
        else:
            return f"{self.message}: stopped after {self.n} elements"


# This class generates correct Aronson sequences up to a given number of elements
class AronsonSet:
    """
    Class for generating correct AronsonSequence objects of a given letter and direction,
    see https://ikavodo.github.io/aronson-1/
    :param letter: The letter used for generating sequences.
    :param direction: sequences generation direction.
    """

    def __init__(self, letter: str, direction: Direction = Direction.FORWARD):
        AronsonSequence.check_letter(letter)
        AronsonSequence.check_direction(direction)
        # Prune generation search space, integers appear in no correct sequence in given direction
        self.non_elements = {2, 3, 5, 6, 8, 9} if direction == Direction.FORWARD else {1, 2, 5, 6, 9, 10}
        self.letter = letter.lower()  # Letter used for generating sequences
        self.direction = direction  # Sequence direction
        self.iter_dict = defaultdict(set)  # init empty dictionary
        self.cur_iter = 0  # no generating iterations yet
        self.seen_seqs = {AronsonSequence(self.letter, [], self.direction)}  # every set contains the empty sequence
        self.iter_dict[self.cur_iter] = self.seen_seqs.copy()  # initial set
        self.subset_dict = defaultdict(set)  # used for fast generation

    @property
    def display_letter(self):
        """ Display in uppercase"""
        return self.letter.upper()

    # constructors
    @classmethod
    def from_sequence(cls, seq: AronsonSequence = None):
        """
        constructor of singleton set (holding a single non-empty AronsonSequence instance).
        :param seq: AronsonSequence
        :return: instance
        """
        seq = seq if seq is not None else AronsonSequence('t')
        if not seq.is_correct():
            # sequence must be correct
            raise VerificationError(input_seq=seq)
        obj = cls(seq.get_letter(), seq.get_direction())
        obj._update_iter({seq})  # update seen_seqs and iter_dict
        return obj

    @classmethod
    def from_set(cls, seqs: set[AronsonSequence] = None):
        """
        constructor from set of AronsonSequence instances.
        :param seqs: set of AronsonSequence instances
        :return: class instance
        """
        seqs = {AronsonSequence('t')} if (seqs is None or not seqs) else seqs
        field_set = set()
        sets = []
        for seq in seqs:
            s = cls.from_sequence(seq)  # use previous constructor
            field_set.add((s.get_letter(), s.get_direction()))
            if len(field_set) > 1:
                # conflicting directions or letters
                raise ValueError("All sequences must have same letter and direction")
            sets.append(s)
        # Union over singleton instances, see __or__() method
        return reduce(lambda a, b: a | b, sets)

    def is_correct(self, seq: AronsonSequence):
        """
        Is given sequence correct with regard to class instance
        :param seq: AronsonSequence to be verified
        :return: True/False
        """

        return (
                seq.get_letter() == self.display_letter and
                seq.get_direction() == self.direction and seq.is_correct()
        )

    def is_complete(self, seq: AronsonSequence):
        """
        If given sequence is complete with regard to class instance
        :param seq: AronsonSequence to be checked
        :return: True/False
        """
        return (
                seq.get_letter() == self.display_letter and
                seq.get_direction() == self.direction and seq.is_complete()
        )

    def _update_iter(self, seqs: set[AronsonSequence]):
        """
        Update iteration dictionary and seen sequences
        :param seqs: for updating
        :return: None
        """
        # Update sets and dicts
        self.iter_dict[self.cur_iter].update(seqs)
        self.seen_seqs.update(seqs)

    # Valid only if no forward-referring element in the input sequence,
    # does nothing if sequence is prefix-complete
    @staticmethod
    def backward_search(seq: AronsonSequence):
        """
        Find all occurrences missing in sentence prefix
        :param seq: for which to find occurrences
        :return: set of new sequences with appended missing occurrences
        """
        occurrences = seq.get_prefix_missing()
        # using __add__() operator of AronsonSequence class
        return {seq.copy() + AronsonSequence(seq.letter, [occ], seq.direction) for occ in occurrences}

    def _agen(self, seq: AronsonSequence):
        """
        Internal generator for backwards-based generation.
        :param seq: An optional AronsonSequence to generate from.
        :return: A generator yielding new indices for the sequence.
        """
        idx, s = seq.get_prefix(), seq.get_sentence()
        s = s[idx:-LEN_SUFFIX] if self.direction == Direction.FORWARD else s[LEN_PREFIX: -idx if idx else None][::-1]
        while True:
            # generator yields as many indices as required
            idx_rel = 1 + s.find(self.letter)  # Find the relative position of the letter
            if idx_rel <= 0:  # Letter not found in string buffer, StopIter
                break
            idx += idx_rel
            yield idx
            ordinal = seq.n2w(idx)
            s = s[idx_rel:] + (ordinal if self.direction == Direction.FORWARD else ordinal[::-1])

    # Valid only if no forward referring elements in input sequence
    def backward_generate(self, n: int, seq: AronsonSequence = None):
        """
        Generate new sequences from well-behaved sequences (with backward/self-referring elements only)
        :param n: num of elements to generate
        :param seq: for generation, optional
        :return: generated sequence with n new elements
        """
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        if n < 0:
            raise ValueError(" n must be non negative")
        new_elements = list(islice(self._agen(seq=seq), n))
        new_seq = seq + AronsonSequence(seq.letter, new_elements, seq.direction)
        if len(new_seq) < len(seq) + n:
            # could not extend sequence up to desired length
            raise GenError(len(seq) + n, ErrorMode.ELEM_MODE)

        # singleton set
        return {new_seq}

    def generate_aronson(self, n: int) -> set[AronsonSequence]:
        """ Wrapper for backward_generate() method with default arguments"""
        return self.backward_generate(n)

    def swap(self, seq: AronsonSequence = None):
        """
        Swap two elements if the positions to which all elements point are unaffected by the swap
        :param seq: AronsonSequence object
        :return: set of newly generated sets in which swapping is legal
        """
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        new_sets = set()
        for i, j in combinations(range(len(seq)), 2):
            ranges = [seq.get_range(seq[x]) for x in [i, j]]
            range_of_effect = range(min(r.start for r in ranges), max(r.stop for r in ranges))
            if all(x - 1 not in range_of_effect for x in seq.get_elements()):
                # all elements unaffected
                swapped = seq.get_elements().copy()
                swapped[i], swapped[j] = swapped[j], swapped[i]
                new_sets.add(AronsonSequence(self.letter, swapped, self.direction))
        return new_sets

    @staticmethod
    def _subset_to_index_pairs(seq_length):
        """
        Map each index pair (i, j) to all non-empty subsets of indices strictly in between.
        If i and j are adjacent, also include each individually as possible subsets.
        :param seq_length: for which to find index pairs
        :return: relevant pairs
        """
        index_pair_subsets = defaultdict(list)
        for i in range(seq_length):
            for j in range(i + 1, seq_length):
                mid = list(range(i + 1, j))  # indices strictly in between
                if not mid:
                    # i and j are adjacent → allow deleting i or j individually
                    index_pair_subsets[(i, j)].append((i,))
                    index_pair_subsets[(i, j)].append((j,))
                else:
                    for r in range(1, len(mid) + 1):
                        for sub in combinations(mid, r):
                            index_pair_subsets[(i, j)].append(sub)
        return index_pair_subsets

    # Valid only for backward-referring sequences.
    def subset(self, seq: AronsonSequence = None):
        """
        Take subset if other elements aren't affected.
        :param self:
        :param seq: AronsonSequence object
        :return: set of newly generated sets in which swapping is legal
        """

        new_sets = set()
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        if len(seq) > self.cur_iter:
            self.subset_dict = self._subset_to_index_pairs(len(seq))
        # pick arbitrary pairs and delete every subset in between
        for i, j in combinations(range(len(seq)), 2):
            for sub in self.subset_dict[(i, j)]:
                ranges = [seq.get_range(seq[x]) for x in sub]
                # anything after first discarded element is affected
                range_min = min(r.start for r in ranges)
                if all(x - 1 < range_min for x in seq.get_elements()):
                    # All elements refer backwards before index of omitted element
                    sub_seq = seq.get_elements().copy()
                    [sub_seq.remove(seq[s]) for s in sub]
                    new_sets.add(AronsonSequence(self.letter, sub_seq, self.direction))
        return new_sets

    # Valid for sequences with no forward referring elements
    def forward_generate(self, seq: AronsonSequence = None):
        """
        Generate by searching within a bounded search space for valid ordinals
        :param seq: to append to
        :return: generated sequence
        """
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        sentence_len = len(seq.get_sentence())
        # generate all singletons if no input
        lower_bound = 1 if seq.is_empty() else sentence_len - (
            LEN_SUFFIX if seq.direction == Direction.FORWARD else LEN_PREFIX) - 1
        new_seqs = set()
        # number of digits in sentence length
        ord_key = len(str(sentence_len))
        if seq.is_empty():
            upper_bound = ORD_TABLE[ord_key] + sentence_len
        else:
            upper_bound = len(seq) * ORD_TABLE[ord_key] + sentence_len

        for elem in range(lower_bound, upper_bound + 1):
            # exhaustive search
            if elem in self.non_elements:
                # skip
                continue
            candidate = seq + AronsonSequence(seq.letter, [elem], seq.direction)
            if self.is_correct(candidate):
                new_seqs.add(candidate)
        return new_seqs

    def generate_singletons(self):
        """
        Wrapper for generating set of singleton AronsonSequences
        :return: set of correct singleton AronsonSequences
        """
        return self.forward_generate(AronsonSequence(self.letter, [], self.direction))

    # Currently infeasible from n >= 4
    def generate_full(self, n_iterations: int, error_rate: float = 0.01):
        """
        Exhaustive generation of all Aronson sequences up to given length
        :param error_rate: degree of precision, with 0. corresponding to no error, 1. to complete
        :param n_iterations: max length of generated sequences
        :return: None
        """
        if n_iterations <= 0:
            return

        def is_valid_extension(elem, current_perm):
            """
            helper for checking validity of new sequence elements
            :param elem: to add
            :param current_perm: which elem is added to
            :return: True or False
            """
            if elem in self.non_elements or (elem - 1 in current_perm and elem - 2 in current_perm) or \
                    (elem + 1 in current_perm and elem + 2 in current_perm):
                return False
            return True

        def backtrack(current_perm, current_sum, remaining, max_len):
            """
            helper for looking over potential sequences
            :param current_perm: input to add to
            :param current_sum: ongoing computation of metric
            :param remaining: elements to add
            :param max_len: of generation
            :return: None
            """
            if len(current_perm) == max_len:
                mean = sum(current_perm) / len(current_perm)
                metric = max(x - mean for x in current_perm)
                upper_metric_bound = ceil(log2(len(current_perm)) * ORD_TABLE[cur_ord_key])
                # if max_len >= 4 and error_rate <= 1e-3:
                #     # misses a few sequences of length 4
                #     upper_metric_bound += 1
                if metric <= (1 - error_rate) * upper_metric_bound:
                    # no extreme outliers
                    yield current_perm.copy()
                return

            for elem in remaining:
                if is_valid_extension(elem, current_perm):
                    current_perm.append(elem)
                    remaining.remove(elem)
                    yield from backtrack(current_perm, current_sum + elem, remaining, max_len)
                    remaining.add(current_perm.pop())

        # generation engine
        while self.cur_iter < n_iterations:
            self.cur_iter += 1
            cur_ord_key = ORD_INITIAL
            upper_bound = self.cur_iter * ORD_TABLE[cur_ord_key] + 2 * LEN_PREFIX
            if upper_bound >= 10 ** (cur_ord_key + 1):
                cur_ord_key += 1

            initial_remaining = {x for x in range(1, upper_bound) if x not in self.non_elements}

            cur_seqs = set()
            for perm in backtrack([], 0, initial_remaining, self.cur_iter):
                # check only relevant sequences
                seq = AronsonSequence(self.letter, perm, self.direction)
                if self.is_correct(seq):
                    cur_seqs.add(seq)

            self._update_iter(cur_seqs)

    def generate_fast(self, n_iterations: int, forward_generate=False):
        """
        Optimized generation using swap/subset operations
        :param forward_generate: more expensive computationally
        :param n_iterations: to generate for
        :return: None
        """
        if n_iterations < 0:
            raise ValueError("Num of iterations must be non-negative")
        if n_iterations == 0:
            return

        while self.cur_iter < n_iterations:

            prev_seqs = self.iter_dict[self.cur_iter]
            self.cur_iter += 1
            # update subset_dict
            self.subset_dict = self._subset_to_index_pairs(self.cur_iter)

            if self.cur_iter == 1:
                cur_seqs = self.generate_singletons()
            else:
                cur_seqs = set()
                for seq in (s for s in prev_seqs):
                    cur_seqs.update(self.swap(seq))
                    cur_seqs.update(self.subset(seq))
                    if not seq.has_forward_ref():
                        cur_seqs.update(self._handle_backward_rules(seq))
                        if forward_generate:
                            # Becomes computationally expensive
                            cur_seqs.update(self.forward_generate(seq))

            filtered = {seq for seq in cur_seqs if seq not in self.seen_seqs}
            if not filtered:
                # converged
                raise GenError(self.cur_iter)
            self._update_iter(filtered)

    def _handle_backward_rules(self, seq):
        """
        Handle backward generation rules
        :param seq: to generate from
        :return: all generated sequences
        """
        results = set()
        if not seq.is_prefix_complete():
            results.update(self.backward_search(seq))
        with suppress(GenError):
            # ignore whatever failed to generate
            results.update(self.backward_generate(1, seq))
        return results

    # getters
    def filter_elements(self, elems):
        """
        return all seen sequences containing elements
        :param elems: for sequences
        :return: set of seen sequences including elems

        """
        if not isinstance(elems, set):
            raise ValueError("input argument must be a set")
        return AronsonSet.from_set({seq for seq in self.seen_seqs if all(elem in seq for elem in elems)})

    def filter_symmetric(self, seq_len=0):
        """
        return all sequences for which all permutations are also in set
        :param seq_len: to start filtering from
        :return: set of such sequences
        """
        sym_set: set[AronsonSequence] = set()
        for seq in self.seen_seqs:
            seq_perm = {AronsonSequence(self.letter, list(perm), self.direction) for perm in
                        permutations(seq, len(seq))}
            if all(perm in self.seen_seqs for perm in seq_perm) and len(seq) >= seq_len:
                sym_set.update(seq_perm)
        return AronsonSet.from_set(sym_set)

    # Buggy?
    def find_non_elements(self, n_iter=None):
        """
        Find all elements up to maximum element in set which do not appear in any sequence
        :param n_iter: to look within
        :return: elements
        """
        if n_iter is not None and (n_iter < 0 or n_iter > self.cur_iter):
            raise ValueError("must choose correct iteration")
        if len(self.seen_seqs) == 1:
            # trivially empty
            return set()
        search_set = self.iter_dict[n_iter] if n_iter is not None else self.seen_seqs
        seen_elems = {elem for seq in search_set for elem in seq}
        # set has at least one non-empty sequence-> max is defined
        return set(range(1, self.max)) - seen_elems

    def get_elements(self):
        """ get all elements appearing in some sequence in set"""
        return {elem for seq in self.seen_seqs for elem in seq}

    def get_unique_element_sequences(self, other: 'AronsonSet'):
        """ Get all sequences holding elements not appearing in any sequence in other set"""
        missing_elements = self.get_elements() - other.get_elements()
        return {seq for seq in self.seen_seqs if any(elem in seq for elem in missing_elements)}

    def filter_refs(self, refs: set[Direction]):
        """
        return all seen sequences including elements with reference pointers in the set refs
        :param refs: pointers for elements in sequence
        :return: set of seen sequences including refs

        """
        if not isinstance(refs, set):
            raise ValueError("input argument must be a set")
        # val is (range[], Refer.Type) for val in seq.get_refer_dict().values
        seqs_set = {seq for seq in self.seen_seqs if refs.issubset({ref[1] for ref in seq.get_refer_dict().values()})}
        return AronsonSet.from_set(seqs_set)

    # Utility methods
    def copy(self):
        """ shallow copy for new instance"""
        new_set = AronsonSet(self.letter, self.direction)
        new_set.set_iter_dict(self.iter_dict)
        return new_set

    def clear(self):
        """ wrapper for clearing a set"""
        self.set_iter_dict({})

    # Setters
    def set_iter_dict(self, new_dict):
        """ main setter, of iteration dictionary and relevant set of seen sequences"""
        # Set always includes empty AronsonSequence
        new_dict = {0: {AronsonSequence(self.letter, [], self.direction)}} if not new_dict else new_dict
        field_set = {(s.get_letter(), s.get_direction()) for sets in new_dict.values() for s in sets}

        # Verify all sequences share the same letter and direction
        if len(field_set) > 1:
            raise ValueError("All sequences must have the same letter and direction")

        self.iter_dict.clear()
        self.iter_dict.update(new_dict)
        self.seen_seqs.clear()
        for s in new_dict.values():
            self.seen_seqs.update(s)
        self.cur_iter = max(new_dict.keys())

    def flip_direction(self):
        """ Flip direction of an empty set"""
        if len(self.seen_seqs) > 1:
            raise ValueError("Can't flip direction of non-default AronsonSet instance.")
        self.direction = self.direction.flip
        for seq in self.seen_seqs:
            # there is only sequence here
            seq.flip_direction()

    def _set_operation_core(self: 'AronsonSet', other: 'AronsonSet', set_op: Callable[[set, set], set],
                            n: int = 0) -> 'AronsonSet':
        """
        Core logic for set operations on AronsonSets.
        Produces a new AronsonSet that is the result of applying `set_op` (e.g., union, intersection)
        on the two inputs. Tracks the iteration index where each sequence was first seen in the inputs,
        and builds the new iter_dict accordingly.
        :param other: set
        :param set_op: to apply
        :param n: generation iterations
        :return: new instance with set_op applied
        """

        def search_dict(cur_seq, cur_set):
            """ helper for getting iterations in which a sequence is added to iter_dict"""
            return (i for i, s in cur_set.iter_dict.items() if cur_seq in s)

        if self.letter != other.letter:
            raise ValueError("Mismatched letters: sets must use the same letter.")

        # don't modify inputs
        set1 = self.copy()
        set2 = other.copy()
        set1.generate_full(n)
        set2.generate_full(n)
        # can't use AronsonSequence instances themselves for set operation
        op_result = set_op({tuple(seq.get_elements()) for seq in set1},
                           {tuple(seq.get_elements()) for seq in set2})

        # Build new iter_dict: for each sequence, track min of iter found in seq1 or seq2
        new_iter_dict = defaultdict(set)
        new_iter_dict[0].add(AronsonSequence(set1.letter, [], set1.direction))

        for elem in op_result:
            for_seq = list(elem)
            # float('inf') is default val in case not found
            iter1 = next(search_dict(AronsonSequence(set1.letter, for_seq, set1.direction), set1), float('inf'))
            iter2 = next(search_dict(AronsonSequence(set2.letter, for_seq, set2.direction), set2), float('inf'))
            gen_iter = min(iter1, iter2)
            # set direction be first set
            new_iter_dict[gen_iter].add(AronsonSequence(set1.letter, for_seq, set1.direction))
        # Construct new AronsonSet, set direction as that of first set
        new_set = AronsonSet(set1.get_letter(), set1.get_direction())
        new_set.set_iter_dict(new_iter_dict)
        return new_set

    # getters
    def get_seen_seqs(self):
        return self.seen_seqs

    def get_iter_dict(self):
        return self.iter_dict

    def get_letter(self):
        """
        :return: letter in upper case
        """
        return self.display_letter

    def get_n_iterations(self):
        return self.cur_iter

    def get_direction(self):
        return self.direction

    def peek(self):
        """ Recover an arbitrary sequence from seen sequences"""
        s = self.seen_seqs.pop()
        self.seen_seqs.add(s)
        return s

    def discard(self, seq: AronsonSequence):
        """
        discard a sequence from a set. Discarding a set of sequences can be accomplished via
        :param seq: to be discarded
        :return: None
        """
        self.__isub__(AronsonSet.from_set({seq}))

    @property
    def max(self):
        """ maximum element seen in some sequence in the set"""
        if len(self.seen_seqs) == 1:
            raise ValueError("Set contains only the empty sequence")
        return max(seq.get_prefix() for seq in self.iter_dict[self.cur_iter])

    # operator overloading
    def __and__(self, other: 'AronsonSet', n: int = 0):
        """
        & operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :return: set instance
        """
        return self._set_operation_core(other, set.intersection, n)

    def __iand__(self, other, n: int = 0):
        """
        &= operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :return: set instance
        """
        result = self._set_operation_core(other, set.intersection, n)
        self.cur_iter = n
        self.set_iter_dict(result.get_iter_dict())
        return self

    def __or__(self, other: 'AronsonSet', n: int = 0):
        """
        | operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :return: set instance
        """
        if self.direction != other.direction:
            raise ValueError("sets must have same direction")
        return self._set_operation_core(other, set.union, n)

    def __ior__(self, other: 'AronsonSet', n: int = 0):
        """
        |= operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :return: set instance
        """
        if self.direction != other.direction:
            raise ValueError("sets must have same direction")
        result = self._set_operation_core(other, set.union, n)
        self.cur_iter = n
        self.set_iter_dict(result.get_iter_dict())
        return self

    def __sub__(self, other: 'AronsonSet', n: int = 0):
        """
        - operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :return: set instance
        """
        # take difference with intersection (which has same direction) if directions not aligned
        return self._set_operation_core(other, set.difference, n) if self.direction == other.direction else \
            self._set_operation_core(self & other, set.difference, n)

    def __isub__(self, other: 'AronsonSet', n: int = 0):
        """
        -= operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :return: set instance
        """
        # take difference with intersection (which has same direction) if directions not aligned
        result = self._set_operation_core(other, set.difference, n) if self.direction == other.direction else \
            self._set_operation_core(self & other, set.difference, n)
        self.cur_iter = n
        self.set_iter_dict(result.get_iter_dict())
        return self

    def __getitem__(self, index: int):
        """
        [] operator, returns the generated sequences at specified iteration
        :param index: of iterations for which AronsonSequences are to be retrieved.
        :return: The relevant generated sets
        """
        return self.iter_dict[index]

    def __iter__(self):
        """
        for operator, returns an iterator over the seen sequences.
        :return: An iterator for the sequences.
        """
        return iter(self.seen_seqs)

    def __len__(self):
        """
        len() operator, returns the number of seen AronsonSequence instances
        :return: The length of the Aronson sequence.
        """
        return len(self.seen_seqs)

    def __contains__(self, item):
        """in operator, returns True or False if a sequence is in set of seen sequences"""
        return item in self.seen_seqs

    @property
    def _hashable_iter_dict(self):
        """
        Returns a frozenset representation of iter_dict for hashing,
        """
        return frozenset(
            (i, frozenset(seq_set)) for i, seq_set in self.iter_dict.items()
        )

    def __eq__(self, other):
        """ = operator"""
        if not isinstance(other, AronsonSet):
            return NotImplemented
        return (
                self.letter == other.letter and
                self.direction == other.direction and
                self._hashable_iter_dict == other._hashable_iter_dict
        )

    def __hash__(self):
        """ hash() operator"""
        return hash((
            self.letter,
            self.direction,
            self._hashable_iter_dict
        ))
