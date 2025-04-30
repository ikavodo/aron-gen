from itertools import islice, combinations
from math import log2, ceil

from AronsonSequence import AronsonSequence, Direction, LEN_PREFIX, LEN_SUFFIX
from collections import defaultdict
from typing import Callable
from functools import reduce
from contextlib import suppress

# global: dictionary with maximum ordinal lengths per number of bits in decimal representation
ORD_TABLE = {i + 1: j for i, j in enumerate([7, 14, 26, 39, 45, 56, 69, 75, 87, 99])}


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
        :param n: expected length of sequence
        """

    def __init__(self, message="Generating failed", n=None):
        self.message = message
        self.n = n
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}" if self.n is None else f"{self.message}: could not generate {self.n} elements"


# This class generates correct Aronson sequences via various generation rules.
def _subset_to_index_pairs(seq_length):
    """ Map each index pair (i, j) to all non-empty subsets of indices strictly in between.
        If i and j are adjacent, also include each individually as possible subsets.
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


class AronsonSet:
    """
    Class for generating correct AronsonSequence objects.
    Notice we have for any instance S this class S ∈ Aₓ(→/←), where x is some letter and arrow refers to
    direction (see https://ikavodo.github.io/aronson-1/)
    :param letter: The letter used for generating sequences.
    :param direction: sequences generation direction.
    """

    # allow to generate from a sequence?
    def __init__(self, letter: str, direction: Direction = Direction.FORWARD):
        # do some tests on letter and direction
        AronsonSequence.check_letter(letter)
        AronsonSequence.check_direction(direction)
        # used for pruning brute-force search space
        self.non_elements = {2, 3, 5, 6, 8, 9} if direction == Direction.FORWARD else {1, 2, 5, 6, 9, 10}
        self.letter = letter.lower()  # Letter used for generating sequences
        self.direction = direction  # Sequence direction
        self.iter_dict = defaultdict(set)  # init empty dictionary
        self.cur_iter = 0  # no generating iterations yet
        self.seen_seqs = {AronsonSequence(self.letter, [], self.direction)}  # every set contains empty sequence
        self.iter_dict[self.cur_iter] = self.seen_seqs.copy()
        self.subset_dict = defaultdict(set)  # used for fast generation

    @property
    def display_letter(self):
        """ Display in uppercase"""
        return self.letter.upper()

    # constructors
    @classmethod
    def from_sequence(cls, seq: AronsonSequence = None):
        """
        constructor from AronsonSequence.
        :param seq: AronsonSequence
        :return: instance
        """
        # default
        seq = seq if seq is not None else AronsonSequence('t')
        if not seq.is_correct():
            raise VerificationError(input_seq=seq)
        obj = cls(seq.get_letter(), seq.get_direction())
        obj._update_iter({seq})
        return obj

    # Nice implementation: take union of singleton Sets
    @classmethod
    def from_set(cls, seqs: set[AronsonSequence] = None):
        """
        constructor from set of AronsonSequence instances.
        :param seqs: set of instances
        :return: class instance
        """
        seqs = {AronsonSequence('t')} if (seqs is None or not seqs) else seqs
        field_set = set()
        sets = []
        for seq in seqs:
            s = cls.from_sequence(seq)
            sets.append(s)
            field_set.add((s.get_letter(), s.get_direction()))
            if len(field_set) > 1:
                raise ValueError("All sequences must have same letter and direction")
        # Return one AronsonSet consisting of all sequences
        return reduce(lambda a, b: a | b, sets)

    def is_correct(self, seq: AronsonSequence):
        """
        If given sequence is correct with regard to class instance
        :param seq: AronsonSequence to be verified
        :return: True/False
        """

        # key = tuple(seq.get_elements())
        # if key not in self.correctness_cache:
        #     self.correctness_cache[key] = seq.is_correct()

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
        # Update stuff where necessary
        self.iter_dict[self.cur_iter].update(seqs)
        self.seen_seqs.update(seqs)

    # Don't use this in case there is any forward-referring element in the sequence
    @staticmethod
    def backward_search(seq: AronsonSequence):
        """
        Find all occurrences missing in sentence prefix and
        :param seq: for which to find occurrences
        :return: set of new sequences with appended missing occurrences
        """
        # does nothing if sequence is prefix complete!
        occurrences = seq.get_prefix_missing()
        new_seqs = set()
        for occ in occurrences:
            seq_cpy = seq.copy()
            seq_cpy.append_elements([occ])
            new_seqs.add(seq_cpy)
        return new_seqs

    def _agen(self, seq: AronsonSequence):
        """
        Internal generator for backwards-based generation.

        :param seq: An optional AronsonSequence to generate from.
        :return: A generator yielding new indices for the sequence.
        """
        idx = seq.get_prefix()
        s = seq.get_sentence()
        s = s[idx:-LEN_SUFFIX] if self.direction == Direction.FORWARD else s[LEN_PREFIX: (-idx if idx else None)][::-1]
        while True:
            # generator yields as many indices as required
            idx_rel = 1 + s.find(self.letter)  # Find the relative position of the letter
            if idx_rel <= 0:  # Letter not found in string buffer, StopIter
                break
            idx += idx_rel
            yield idx
            s = s[idx_rel:] + (seq.n2w(idx) if self.direction == Direction.FORWARD else seq.n2w(idx)[::-1])

    # For well-behaved sequences (no forward referring!)
    def backward_generate(self, n: int, seq: AronsonSequence = None):
        """
        Generate new sequences from well-behaved sequences (with backward/self-referring elements only)
        :param n: num of elements to generate
        :param seq: for generation, optional
        :return: generated sequence with n new elements
        """
        # generate empty AronsonSequence if no argument
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        if n < 0:
            raise ValueError(" n must be non negative")
        new_elements = list(islice(self._agen(seq=seq), n))
        # don't modify original
        seq_cpy = seq.copy()
        seq_cpy.append_elements(new_elements)

        if len(seq_cpy) < len(seq) + n:
            # could not extend sequence up to desired length
            raise GenError(n=n)
        return {seq_cpy}

    def generate_aronson(self, n: int) -> set[AronsonSequence]:
        """ Wrapper for backward_generate() method with default arguments"""
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
            ranges = [seq.get_range(seq[x]) for x in [i, j]]
            # anything in between is affected
            range_of_effect = range(min(r.start for r in ranges), max(r.stop for r in ranges))
            # need to look at all elements.
            if all(x - 1 not in range_of_effect for x in seq.get_elements()):
                swapped = seq.get_elements().copy()
                swapped[i], swapped[j] = swapped[j], swapped[i]
                new_sets.add(AronsonSequence(self.letter, swapped, self.direction))
        return new_sets

    def subset(self, seq: AronsonSequence = None):
        """
        Can take subset if other elements aren't affected. This only works for backward-referring sequences.
        :param self:
        :param seq: AronsonSequence object
        :return: set of newly generated sets in which swapping is legal
        """

        new_sets = set()
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        if len(seq) > self.cur_iter:
            self.subset_dict = _subset_to_index_pairs(len(seq))
        # Idea is to pick arbitrary pairs and delete every subset in between
        for i, j in combinations(range(len(seq)), 2):
            for sub in self.subset_dict[(i, j)]:
                ranges = [seq.get_range(seq[x]) for x in sub]
                # anything after first omitted is affected
                range_min = min(r.start for r in ranges)
                # All elements need to refer backwards before point of first point of omission
                if all(x - 1 < range_min for x in seq.get_elements()):
                    sub_seq = seq.get_elements().copy()
                    [sub_seq.remove(seq[s]) for s in sub]
                    new_sets.add(AronsonSequence(self.letter, sub_seq, self.direction))
        return new_sets

    # For sequences with only backward and self-referring elements (otherwise use forward_fix)
    def forward_generate(self, seq: AronsonSequence = None):
        """
        Generate by searching within a bounded search space for valid ordinals
        :param seq: to append to
        :return: generated sequence
        """
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        sentence_len = len(seq.get_sentence())
        # for generating all singleton AronsonSequences in first iteration
        lower_bound = 1 if seq.is_empty() else sentence_len - (
            LEN_SUFFIX if seq.direction == Direction.FORWARD else LEN_PREFIX) - 1
        new_seqs = set()
        ord_key = len(str(sentence_len))
        if seq.is_empty():
            upper_bound = ORD_TABLE[ord_key] + sentence_len
        else:
            upper_bound = len(seq) * ORD_TABLE[ord_key] + sentence_len

        # make more efficient?
        for elem in range(lower_bound, upper_bound + 1):
            if elem in self.non_elements:
                continue
            candidate = seq.copy()
            candidate.append_elements([elem])
            # brute-force search within bounded search space
            if self.is_correct(candidate):
                new_seqs.add(candidate)
        return new_seqs

    def generate_singletons(self):
        """
        Wrapper for generating set of singleton AronsonSequences
        :return: set of correct singleton AronsonSequences
        """
        return self.forward_generate(AronsonSequence(self.letter, [], self.direction))

    # For generating ground truths in first three iterations (infeasible afterwards)
    def generate_full(self, n_iterations: int):
        if n_iterations <= 0:
            return

        def is_valid_extension(elem, current_perm):
            if elem in self.non_elements or (elem - 1 in current_perm and elem - 2 in current_perm) or \
                    (elem + 1 in current_perm and elem + 2 in current_perm):
                return False
            return True

        def backtrack(current_perm, current_sum, remaining, max_len):
            if len(current_perm) == max_len:
                # Apply metric check only now
                mean = sum(current_perm) / len(current_perm)
                metric = max(x - mean for x in current_perm)
                if metric <= ceil(log2(len(current_perm)) * ORD_TABLE[cur_ord_key]):
                    yield current_perm.copy()
                return
            for elem in sorted(remaining):
                if is_valid_extension(elem, current_perm):
                    current_perm.append(elem)
                    remaining.remove(elem)
                    yield from backtrack(current_perm, current_sum + elem, remaining, max_len)
                    remaining.add(current_perm.pop())

        cur_ord_key = 2
        while self.cur_iter < n_iterations:
            self.cur_iter += 1

            upper_bound = self.cur_iter * ORD_TABLE[cur_ord_key] + 2 * LEN_PREFIX
            if upper_bound >= 10 ** (cur_ord_key + 1):
                cur_ord_key += 1

            allowed_elements = [x for x in range(1, upper_bound) if x not in self.non_elements]
            initial_remaining = set(allowed_elements)

            cur_seqs = set()
            for perm in backtrack([], 0, initial_remaining, self.cur_iter):
                seq = AronsonSequence(self.letter, perm, self.direction)
                if self.is_correct(seq):
                    cur_seqs.add(seq)

            self._update_iter(cur_seqs)

    def generate_fast(self, n_iterations: int):
        """Optimized generation using swap/subset operations"""
        if n_iterations < 0:
            raise ValueError("Num of iterations must be non-negative")
        if n_iterations == 0:
            return

        while self.cur_iter < n_iterations:

            prev_seqs = self.iter_dict[self.cur_iter]
            self.cur_iter += 1
            # update subset_dict
            self.subset_dict = _subset_to_index_pairs(self.cur_iter)
            cur_seqs = self.generate_singletons() if self.cur_iter == 1 else set().union(
                *(self.swap(seq) | self.subset(seq) for seq in prev_seqs))
            for seq in (s for s in prev_seqs if not s.has_forward_ref()):
                cur_seqs.update(self._handle_backward_rules(seq))

            self._update_filtered(cur_seqs)

            # Helper methods

    def _handle_backward_rules(self, seq):
        """Handle backward-looking generation rules"""
        results = set()
        if not seq.is_prefix_complete():
            results.update(self.backward_search(seq))
        with suppress(GenError):
            results.update(self.backward_generate(1, seq))
            # currently finds 5 sequences forward_fix doesn't. why?
        results.update(self.forward_generate(seq))
        return results

    def _update_filtered(self, cur_seqs):
        """Update seen sequences with filtered results"""
        filtered = {seq for seq in cur_seqs if seq not in self.seen_seqs}
        if not filtered:
            raise GenError("converged")
        self._update_iter(filtered)

    def filter_elements(self, elems):
        """
        return all seen sequences including elements in elems
        :param elems: for sequences
        :return: set of seen sequences including elems

        """
        if not isinstance(elems, set):
            raise ValueError("input argument must be a set")
        return AronsonSet.from_set({seq for seq in self.seen_seqs if all(elem in seq for elem in elems)})

    def find_non_elements(self, n_iter):
        # idea: look at all generated sequences, look for elements which appear in non.
        if n_iter <= 0:
            return set()
        seen_elems = reduce(set.__or__, [set(seq) for seq in self.iter_dict[n_iter]])
        return set(range(1, max(seen_elems))) - seen_elems

    def get_elements(self):
        return {elem for seq in self.seen_seqs for elem in seq}

    def get_exotic_sequences(self, other: 'AronsonSet'):
        # want to get sequences in other set which contain elements missing in the current set
        missing_elements = self.get_elements() - other.get_elements()
        return {seq for seq in self.seen_seqs if any(elem in seq for elem in missing_elements)}

    def filter_refs(self, refs):
        """
        return all seen sequences including reference pointers in refs
        :param refs: pointers for elements in sequence
        :return: set of seen sequences including refs

        """
        if not isinstance(refs, set):
            raise ValueError("input argument must be a set")
        return AronsonSet.from_set({
            seq for seq in self.seen_seqs
            if refs.issubset({ref[1] for ref in seq.get_refer_dict().values()})
        })

    # Utility methods
    def copy(self):
        new_set = AronsonSet(self.letter, self.direction)
        new_set.set_iter_dict(self.iter_dict)
        return new_set

    def clear(self):
        """ wrapper for clearing a set"""
        self.set_iter_dict({})

    # Setters
    def set_subset_dict(self, seq: AronsonSequence):
        """ Helper"""
        self.subset_dict = _subset_to_index_pairs(len(seq))

    def set_iter_dict(self, new_dict):
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
        # is empty
        self.direction = self.direction.opposite()
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
        # Compute the result of the set operation
        result = set_op({tuple(seq.get_elements()) for seq in set1.get_seen_seqs()},
                        {tuple(seq.get_elements()) for seq in set2.get_seen_seqs()})

        # Build new iter_dict: for each sequence, track min of iter found in seq1 or seq2
        new_iter_dict = defaultdict(set)
        if result:
            # Always retain the empty set
            new_iter_dict[0].add(AronsonSequence(set1.letter, [], set1.direction))

        for elem in result:
            # float('inf') is default val in case not found
            for_seq = list(elem)
            iter1 = next(search_dict(AronsonSequence(set1.letter, for_seq, set1.direction), set1), float('inf'))
            iter2 = next(search_dict(AronsonSequence(set2.letter, for_seq, set2.direction), set2), float('inf'))
            gen_iter = min(iter1, iter2)
            # set direction be first set
            new_iter_dict[gen_iter].add(AronsonSequence(set1.letter, for_seq, set1.direction))
        # Construct new AronsonSet, set direction as that of first set
        result = AronsonSet(set1.get_letter(), set1.get_direction())
        result.set_iter_dict(new_iter_dict)
        return result

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

    @property
    def max(self):
        if len(self.seen_seqs) == 1:
            raise ValueError("Set contains only the empty sequence")
        return max(seq.get_prefix() for seq in self.iter_dict[self.cur_iter])
