from itertools import islice, combinations, permutations
from AronsonSequence import AronsonSequence, Refer, Direction, LEN_PREFIX, LEN_SUFFIX
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
        self.iter_dict = defaultdict(set)  # init empty dictionary
        self.cur_iter = 0  # no generating iterations yet
        self.seen_seqs = {AronsonSequence(self.letter, [], self.direction)}  # every set contains empty sequence
        self.iter_dict[self.cur_iter] = self.seen_seqs.copy()
        self.subset_dict = defaultdict(set)

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

    #   currently unused
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

    def _subset_to_index_pairs(self, seq_length):
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

    def subset(self, seq: AronsonSequence = None):
        """
        Can take subset if other elements aren't affected. This only works for backward-referring sequences.
        :param seq: AronsonSequence object
        :return: set of newly generated sets in which swapping is legal
        """

        new_sets = set()
        seq = seq if seq is not None else AronsonSequence(self.letter, [], self.direction)
        if len(seq) > self.cur_iter:
            self.subset_dict = self._subset_to_index_pairs(len(seq))
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
        if seq.is_empty():
            # for generating all singleton AronsonSequences in first iteration
            lower_bound = 1
        else:
            lower_bound = sentence_len - (LEN_SUFFIX if seq.direction == Direction.FORWARD else LEN_PREFIX)
        ord_key = len(str(sentence_len))
        upper_bound = sentence_len + ORD_TABLE[ord_key]
        new_seqs = set()
        # make more efficient?
        for elem in range(lower_bound, upper_bound + 1):
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

        cur_ord_key = 2  # maximal ordinal with 2 digits
        while self.cur_iter < n_iterations:
            self.cur_iter += 1
            upper_bound = self.cur_iter * ORD_TABLE[cur_ord_key] + 2 * max(LEN_PREFIX, LEN_SUFFIX)
            if upper_bound >= 10 ** (cur_ord_key + 1):
                # enlarge search range
                cur_ord_key += 1
            seqs = (AronsonSequence(self.letter, list(perm), self.direction) for perm in
                    generate_unique_lists(self.cur_iter, upper_bound))
            cur_seqs = {s for s in seqs if self.is_correct(s)}
            self._update_iter(cur_seqs)

    def forward_fix(self, seq: AronsonSequence) -> set[AronsonSequence]:
        """
        Generate by searching within a bounded search space for valid ordinals to append.
        Also, "correct" existing sequence elements if necessary
        :param seq: to append to
        :return: generated sequences
        """
        new_seqs = set()
        elements = seq.get_elements()
        sentence_len = len(seq.get_sentence())

        # all elements which refer forward and can be changed without breaking correctness
        forward_indices = {i for i, x in enumerate(elements) if seq.get_ref(x) == Refer.FORWARD and all(
            y not in range(min(seq.get_range(x)), sentence_len) for y in elements if y != x)}
        ord_key = len(str(sentence_len))
        upper_bound = sentence_len + ORD_TABLE[ord_key]
        for i in forward_indices:
            for candidate_back in range(1, upper_bound):
                lower_bound = min(seq.get_range(seq[i]))
                for candidate_forward in range(lower_bound, upper_bound):
                    new_elements = seq.get_elements().copy()
                    new_elements.append(candidate_back)
                    new_elements[i] = candidate_forward
                    new_seq = AronsonSequence(seq.letter, new_elements, self.direction)
                    if self.is_correct(new_seq):
                        new_seqs.add(new_seq)
        return new_seqs

    def generate_from_rules(self, n_iterations: int, full=True):
        """
        Generate sequences using forward/backward resolution rules
        :param n_iterations: Number of generations to perform
        :param full: Whether to use exhaustive forward reference resolution
        """
        if n_iterations < 0:
            raise ValueError("Num of iterations must be non-negative")
        if n_iterations == 0:
            return

        while self.cur_iter < n_iterations:
            prev_seqs = self.iter_dict[self.cur_iter]
            self.cur_iter += 1
            cur_seqs, forward_seqs = set(), []

            for seq in prev_seqs:
                if seq.has_forward_ref():
                    (cur_seqs.update(self.forward_fix(seq)) if full else forward_seqs.append(seq))
                else:
                    cur_seqs.update(self._handle_backward_rules(seq))

            if forward_seqs and not full:
                cur_seqs.update(self.forward_fix(max(forward_seqs,
                                                     key=lambda x: len(x.get_sentence()))))

            self._update_filtered(cur_seqs)

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
            self.subset_dict = self._subset_to_index_pairs(self.cur_iter)
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
        results.update(self.forward_generate(seq))
        return results

    def _update_filtered(self, cur_seqs):
        """Update seen sequences with filtered results"""
        filtered = {seq for seq in cur_seqs if seq not in self.seen_seqs}
        if not filtered:
            raise GenError("converged")
        self._update_iter(filtered)

    def filter_elems(self, elems):
        """
        return all seen sequences including elements in elems
        :param elems: for sequences
        :return: set of seen sequences including elems

        """
        if not isinstance(elems, set):
            raise ValueError("elements must be a set")
        return {seq for seq in self.seen_seqs if all(elem in seq for elem in elems)}

    def filter_refs(self, refs):
        """
        return all seen sequences including reference pointers in refs
        :param refs: pointers for elements in sequence
        :return: set of seen sequences including refs

        """
        if not isinstance(refs, set):
            raise ValueError("refs must be a set")
        return {
            seq for seq in self.seen_seqs
            if refs.issubset({ref[1] for ref in seq.get_refer_dict().values()})
        }

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
        self.subset_dict = self._subset_to_index_pairs(len(seq))

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
        self.direction = Direction.BACKWARD if self.direction == Direction.FORWARD else Direction.FORWARD
        for seq in self.seen_seqs:
            # there is only sequence here
            seq.flip_direction()

    def _set_operation_core(self: 'AronsonSet', other: 'AronsonSet', set_op: Callable[[set, set], set],
                            n: int = 0, full=False) -> 'AronsonSet':
        """
        Core logic for set operations on AronsonSets.
        Produces a new AronsonSet that is the result of applying `set_op` (e.g., union, intersection)
        on the two inputs. Tracks the iteration index where each sequence was first seen in the inputs,
        and builds the new iter_dict accordingly.
        :param other: set
        :param set_op: to apply
        :param n: generation iterations
        :param full: to use for generation
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
        set1.generate_from_rules(n, full)
        set2.generate_from_rules(n, full)
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
    def __and__(self, other: 'AronsonSet', n: int = 0, full=False):
        """
        & operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :param full: for generating
        :return: set instance
        """
        return self._set_operation_core(other, set.intersection, n, full)

    def __iand__(self, other, n: int = 0, full=False):
        """
        &= operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :param full: for generating
        :return: set instance
        """
        result = self._set_operation_core(other, set.intersection, n, full)
        self.cur_iter = n
        self.set_iter_dict(result.get_iter_dict())
        return self

    def __or__(self, other: 'AronsonSet', n: int = 0, full=False):
        """
        | operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :param full: for generating
        :return: set instance
        """
        if self.direction != other.direction:
            raise ValueError("sets must have same direction")
        return self._set_operation_core(other, set.union, n, full)

    def __ior__(self, other: 'AronsonSet', n: int = 0, full=False):
        """
        |= operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :param full: for generating
        :return: set instance
        """
        if self.direction != other.direction:
            raise ValueError("sets must have same direction")
        result = self._set_operation_core(other, set.union, n, full)
        self.cur_iter = n
        self.set_iter_dict(result.get_iter_dict())
        return self

    def __sub__(self, other: 'AronsonSet', n: int = 0, full=False):
        """
        - operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :param full: for generating
        :return: set instance
        """
        # take difference with intersection (which has same direction) if directions not aligned
        return self._set_operation_core(other, set.difference, n, full) if self.direction == other.direction else \
            self._set_operation_core(self & other, set.difference, n, full)

    def __isub__(self, other: 'AronsonSet', n: int = 0, full=False):
        """
        -= operator over sets, wrapping set operation helper
        :param other: set
        :param n: generation iterations
        :param full: for generating
        :return: set instance
        """
        # take difference with intersection (which has same direction) if directions not aligned
        result = self._set_operation_core(other, set.difference, n, full) if self.direction == other.direction else \
            self._set_operation_core(self & other, set.difference, n, full)
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
