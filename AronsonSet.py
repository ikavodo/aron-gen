from itertools import islice, combinations, permutations
from AronsonSequence import AronsonSequence, n2w, Refer, Direction, LEN_PREFIX, LEN_SUFFIX
from collections import defaultdict
from typing import Callable
from functools import reduce
from contextlib import suppress

# upper bound over max ordinal lengths per number of bits in integer representation
# ORD_TABLE = {i: 12 * (i - 1) for i in range(1, 12)}
# {2: 7, 3: 14, 4: 26, 5: 39, 6: 44, 7: 56, 8: 67, 9: 73, 10: 85, 11: 96}

ORD_TABLE = {i + 1: j for i, j in enumerate([7, 14, 26, 39, 45, 56, 69, 75, 87, 99])}


# ORD_TABLE = {i: 10*i for i in range(12)}

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

    def __init__(self, message="Generating failed", n=None):
        self.message = message
        self.n = n
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}: could not generate {self.n} elements"


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
    def __init__(self, letter: str, direction: Direction = Direction.FORWARD):
        # do some tests on letter and direction
        AronsonSequence.check_letter(letter)
        AronsonSequence.check_direction(direction)

        self.letter = letter.lower()  # Letter used for generating sequences
        self.direction = direction  # Sequence direction
        self.iter_dict = defaultdict(set)
        self.cur_iter = 0
        # This is wrong!!! Aha. The set operator tries to
        self.seen_seqs = {AronsonSequence(self.letter, [], self.direction)}
        # This counts as iteration 0
        self.iter_dict[self.cur_iter] = self.seen_seqs.copy()

    @property
    def display_letter(self):
        """ Display in uppercase"""
        return self.letter.upper()

    # constructors
    @classmethod
    def from_sequence(cls, seq: AronsonSequence = None):
        """
        constructor from AronsonSequence. Good for checking generation rules on a particular sequence
        :param seq: AronsonSequence
        :return: instance
        """
        # default
        seq = seq if seq is not None else AronsonSequence('t')
        # (inefficient to construct before checking correctness, change this later)
        if not seq.is_correct():
            # Let the user know
            raise VerificationError(input_seq=seq)
        obj = cls(seq.get_letter(), seq.get_direction())
        # should the user even be allowed to add sequences? For a second let's decide on the contrary
        obj._update_iter({seq})
        return obj

    # Nice implementation: take union of singleton Sets
    @classmethod
    def from_set(cls, seqs: set[AronsonSequence] = None):
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
        :param seq: AronsonSequence to be verified
        :return: True/False
        """
        return (
                seq.get_letter() == self.display_letter and
                seq.get_direction() == self.direction and seq.is_correct()
        )

    def is_complete(self, seq: AronsonSequence):
        """
        wrapper for _ismember(), allows for checking completeness of sequence
        :param seq: AronsonSequence to be checked
        :return: True/False
        """
        return (
                seq.get_letter() == self.display_letter and
                seq.get_direction() == self.direction and seq.is_complete()
        )

    def _update_iter(self, seqs: set[AronsonSequence]):
        """
        Add new sequences to set of seen sequences
        :param seqs:
        :return:
        """
        # At the moment no verification for generation rules! Change this if generation rules incorrect
        # valid_seqs = {seq for seq in seqs if seq not in self.seen_seqs and self.is_correct(seq)}

        # Update stuff where necessary
        self.iter_dict[self.cur_iter].update(seqs)
        self.seen_seqs.update(seqs)

    # Don't use this in case there is any forward-referring element in the sequence
    @staticmethod
    def backward_search(seq: AronsonSequence):
        """
        Find all occurrences missing in sentence prefix and
        :param seq:
        :return:
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
            s = s[idx_rel:] + (n2w(idx) if self.direction == Direction.FORWARD else n2w(idx)[::-1])

    # For good old well-behaved sequences (no forward referring)
    def backward_generate(self, n: int, seq: AronsonSequence = None):
        """
        Generate new sequences from well-behaved sequences (with backward/self-referring elements only)
        """
        # generate empty AronsonSequence if no argument
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        # generate as many new elements as necessary
        if n < 0:
            raise ValueError(" n must be non negative")
        new_elements = list(islice(self._agen(seq=seq), n))
        # update sequence, don't modify original
        seq_cpy = seq.copy()
        seq_cpy.append_elements(new_elements)

        if len(seq_cpy) < len(seq) + n:
            # could not extend sequence up to desired length
            raise GenError(n=n)
        return {seq_cpy}

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
            ranges = [seq.get_range(seq[x]) for x in [i, j]]
            # anything in between is affected
            range_of_effect = range(min(r.start for r in ranges), max(r.stop for r in ranges))
            # need to look at all elements.
            if all(x - 1 not in range_of_effect for x in seq.get_elements()):
                swapped = seq.get_elements().copy()
                swapped[i], swapped[j] = swapped[j], swapped[i]
                new_sets.add(AronsonSequence(self.letter, swapped, self.direction))
        return new_sets

    # TODO change this to work also for sequence with two elements
    def subset(self, seq: AronsonSequence = None):
        """
        Can take subset if other elements aren't affected. This only works for backward-referring sequences (?).
        :param seq: AronsonSequence object
        :return: set of newly generated sets in which swapping is legal
        """

        def subset_to_index_pairs(seq_length):
            """ Map each index pair (i, j) to all non-empty subsets of indices strictly between i and j.
                If i and j are adjacent, also include each individually as possible subsets.
            """
            index_pair_subsets = defaultdict(list)
            for i in range(seq_length):
                for j in range(i + 1, seq_length):
                    mids = list(range(i + 1, j))  # indices strictly between i and j
                    if not mids:
                        # i and j are adjacent → allow deleting i or j individually
                        index_pair_subsets[(i, j)].append((i,))
                        index_pair_subsets[(i, j)].append((j,))
                    else:
                        for r in range(1, len(mids) + 1):
                            for sub in combinations(mids, r):
                                index_pair_subsets[(i, j)].append(sub)
            return index_pair_subsets

        new_sets = set()
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        subset_dict = subset_to_index_pairs(len(seq))
        # Idea- try to pick arbitrary pairs and delete every subset in between
        for i, j in combinations(range(len(seq)), 2):
            # if j > i + 1:
            for sub in subset_dict[(i, j)]:
                ranges = [seq.get_range(seq[x]) for x in sub]
                # anything after first omitted is affected
                range_min = min(r.start for r in ranges)
                # All elements need to refer backwards before point of first point of omission
                if all(x - 1 < range_min for x in seq.get_elements()):
                    sub_seq = seq.get_elements().copy()
                    [sub_seq.remove(seq[s]) for s in sub]
                    new_sets.add(AronsonSequence(self.letter, sub_seq, self.direction))
        return new_sets

    # Idea: try to generate something at the end of the sequence.
    # For sequences with only backward and self-referring elements (otherwise use forward_fix)
    def forward_generate(self, seq: AronsonSequence = None):
        """
        Generate by searching within a bounded search space for valid ordinals to append
        :param seq:
        :return:
        """
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        if seq.is_empty():
            # for generating all singleton AronsonSequences in first iteration, seq.prefix = 0
            lower_bound = 1
        else:
            lower_bound = len(seq.get_sentence()[:-LEN_SUFFIX]) if seq.direction == Direction.FORWARD else \
                len(seq.get_sentence()[LEN_PREFIX:])
        # overshoot.
        sentence_len = len(seq.get_sentence())
        ord_key = len(str(sentence_len))
        upper_bound = len(seq.get_sentence()) + ORD_TABLE[ord_key]
        new_seqs = set()
        for elem in range(lower_bound, upper_bound + 1):
            candidate = seq.copy()
            candidate.append_elements([elem])
            # brute-force search within bounded search space
            if self.is_correct(candidate):
                new_seqs.add(candidate)
        return new_seqs

    # Is there a better way to generate all singletons?
    def generate_singletons(self):
        """
        Generates all single-index valid AronsonSequences (singletons).

        :return: set of all correct singleton AronsonSequences
        """
        return self.forward_generate(AronsonSequence(self.letter, [], self.direction))

    # For generating ground truths up to cur_iters=3
    def generate_brute_force(self, n_iterations: int):
        def generate_unique_lists(n, bound):
            """ helper for generating permutations"""
            if n > bound:
                raise ValueError("n can't be greater than the number of available unique values")
            return [list(p) for p in permutations(range(1, bound + 1), n)]  # Start from 1

        if n_iterations < 0:
            raise ValueError("Num of iterations must be non-negative")

        if n_iterations == 0:
            # do nothing
            return

        cur_ord_key = 2  # maximal length of size 2
        while self.cur_iter < n_iterations:
            # enlarge search range
            self.cur_iter += 1
            # grows linearly with iterations
            upper_bound = self.cur_iter * ORD_TABLE[cur_ord_key] + 2 * max(LEN_PREFIX, LEN_SUFFIX)
            if upper_bound >= 10 ** (cur_ord_key + 1):
                # enlarge search range
                cur_ord_key += 1
            seqs = (AronsonSequence(self.letter, list(perm), self.direction) for perm in
                    generate_unique_lists(self.cur_iter, upper_bound))
            cur_seqs = {s for s in seqs if self.is_correct(s)}
            self._update_iter(cur_seqs)

    def forward_fix(self, seq: AronsonSequence, expensive=False) -> set[AronsonSequence]:
        new_seqs = set()
        elements = seq.get_elements()
        # all elements which refer forward, where no other elements refer to an area which will be affected by
        # changing such elements
        forward_refs = {x for x in elements if seq.get_ref(x) == Refer.FORWARD and all(
            y not in range(min(seq.get_range(x)), len(seq.get_sentence())) for y in elements if y != x)}
        sentence_len = len(seq.get_sentence())
        ord_key = len(str(sentence_len))
        upper_bound = len(seq.get_sentence()) + ORD_TABLE[ord_key]
        # currently inefficient in expensive case. Think how to optimize
        occurrences = seq.get_occurrences() - forward_refs if not expensive else range(1, upper_bound)
        for elem in forward_refs:
            for candidate_back in occurrences:
                # made larger
                lower_bound = min(seq.get_range(elem))
                for candidate_forward in range(lower_bound, upper_bound):
                    new_elements = seq.get_elements().copy()
                    new_elements.append(candidate_back)
                    # Try replacing elem and checking if correct.
                    new_elements[new_elements.index(elem)] = candidate_forward
                    new_seq = AronsonSequence(seq.letter, new_elements, self.direction)
                    if self.is_correct(new_seq):
                        new_seqs.add(new_seq)
        return new_seqs

    # backward_search, backward_generate, swap, forward_generate, forward_fix.
    def generate_from_rules(self, n_iterations: int, expensive=False):
        if n_iterations < 0:
            raise ValueError("Num of iterations must be non-negative")

        if n_iterations == 0:
            # do nothing
            return

        while self.cur_iter < n_iterations:
            # generate new empty set, to update
            prev_seqs = self.iter_dict[self.cur_iter]
            self.cur_iter += 1
            cur_seqs = set()
            for seq in prev_seqs:

                if seq.has_forward_ref():
                    # most computationally intensive, takes most factors into account
                    cur_seqs.update(self.forward_fix(seq, expensive))
                else:
                    if not seq.is_prefix_complete():
                        # there are missing occurrences to be added
                        cur_seqs.update(self.backward_search(seq))
                    with suppress(GenError):
                        # do nothing if error comes up
                        cur_seqs.update(self.backward_generate(1, seq))
                    cur_seqs.update(self.forward_generate(seq))

                # can always do this
                cur_seqs.update(self.swap(seq))
            filtered = {seq for seq in cur_seqs if seq not in self.seen_seqs}
            # takes care of updating where necessary
            self._update_iter(filtered)

    def copy(self):
        new_set = AronsonSet(self.letter, self.direction)
        # this also sets new seen_seqs object. 
        new_set.set_iter_dict(self.iter_dict)
        return new_set

    def clear(self):
        """ wrapper"""
        self.set_iter_dict({})

    def set_iter_dict(self, new_dict):
        new_dict = {0: {AronsonSequence(self.letter, [], self.direction)}} if not new_dict else new_dict
        # error checking!
        field_set = {(s.get_letter(), s.get_direction()) for sets in new_dict.values() for s in sets}

        # Verify all sequences share the same letter and direction
        if len(field_set) > 1:
            raise ValueError("All sequences must have the same letter and direction")

        # what is expected behavior when input argument iter_dict is empty?
        self.iter_dict.clear()
        self.iter_dict.update(new_dict)
        # This should automatically change the seen sequences.
        self.seen_seqs.clear()
        for s in new_dict.values():
            self.seen_seqs.update(s)
        self.cur_iter = max(new_dict.keys())

    def flip_direction(self):
        if len(self.seen_seqs) > 1:
            raise ValueError("Can't flip direction of non-default AronsonSet instance.")
        # is empty
        self.direction = Direction.BACKWARD if self.direction == Direction.FORWARD else Direction.FORWARD
        for seq in self.seen_seqs:
            seq.flip_direction()

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

    # Helper for operator overloading
    def _set_operation_core(self: 'AronsonSet', other: 'AronsonSet', set_op: Callable[[set, set], set],
                            n: int = 0, expensive=False) -> 'AronsonSet':
        """
        Core logic for set operations on AronsonSets.
        Produces a new AronsonSet that is the result of applying `set_op` (e.g., union, intersection)
        on the seen_seqs of self and other.
        Tracks the correct iteration index where each sequence was first seen in the inputs,
        and builds the new iter_dict accordingly.
        """
        if self.letter != other.letter:
            raise ValueError("Mismatched letters: sets must use the same letter.")

        # copy!
        set1 = self.copy()
        set2 = other.copy()
        set1.generate_from_rules(n, expensive)
        set2.generate_from_rules(n, expensive)
        # implement version ignoring letters later.
        # Compute the result of the set operation
        result = set_op({tuple(seq.get_elements()) for seq in set1.get_seen_seqs()},
                        {tuple(seq.get_elements()) for seq in set2.get_seen_seqs()})

        # Build new iter_dict: for each sequence, track min of iter found in seq1 or seq2
        new_iter_dict = defaultdict(set)
        if result:
            # Always retain the empty set
            new_iter_dict[0].add(AronsonSequence(set1.letter, [], set1.direction))
        search_dict = lambda cur_seq, cur_set: (i for i, s in cur_set.iter_dict.items() if cur_seq in s)
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

    def __and__(self, other: 'AronsonSet', n: int = 0, expensive=False):
        return self._set_operation_core(other, set.intersection, n, expensive)

    def __iand__(self, other, n: int = 0, expensive=False):
        result = self._set_operation_core(other, set.intersection, n, expensive)
        self.cur_iter = n
        self.set_iter_dict(result.get_iter_dict())
        return self

    def __or__(self, other: 'AronsonSet', n: int = 0, expensive=False):
        return self._set_operation_core(other, set.union, n, expensive)

    def __ior__(self, other: 'AronsonSet', n: int = 0, expensive=False):
        result = self._set_operation_core(other, set.union, n, expensive)
        self.cur_iter = n
        self.set_iter_dict(result.get_iter_dict())
        return self

    def __sub__(self, other: 'AronsonSet', n: int = 0, expensive=False):
        return self._set_operation_core(other, set.difference, n, expensive)

    def __isub__(self, other: 'AronsonSet', n: int = 0, expensive=False):
        result = self._set_operation_core(other, set.difference, n, expensive)
        self.cur_iter = n
        self.set_iter_dict(result.get_iter_dict())
        return self

    def __getitem__(self, index: int):
        """
        Returns the generated sequences at specified iteration

        :param index: of iterations for which AronsonSequences are to be retrieved.
        :return: The relevant generated sets
        """
        return self.iter_dict[index]

    def __iter__(self):
        """
        Returns an iterator over the elements of the sequence.

        :return: An iterator for the elements.
        """
        return iter(self.seen_seqs)

    def __len__(self):
        """
        Returns the length of the Aronson sequence (i.e., the number of elements).

        :return: The length of the Aronson sequence.
        """
        return len(self.seen_seqs)

    def __contains__(self, item):
        return item in self.seen_seqs

    @property
    def _hashable_iter_dict(self):
        """
        Returns a frozenset representation of iter_dict,
        where each entry is a tuple (iteration, frozenset(sequences))
        """
        return frozenset(
            (i, frozenset(seq_set)) for i, seq_set in self.iter_dict.items()
        )

    def __eq__(self, other):
        if not isinstance(other, AronsonSet):
            return NotImplemented
        return (
                self.letter == other.letter and
                self.direction == other.direction and
                self._hashable_iter_dict == other._hashable_iter_dict
        )

    def __hash__(self):
        return hash((
            self.letter,
            self.direction,
            self._hashable_iter_dict
        ))
