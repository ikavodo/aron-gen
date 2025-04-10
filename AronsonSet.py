from itertools import islice, cycle, combinations
from AronsonSequence import AronsonSequence, n2w, STR_START, STR_END

# upper bound for searching for singleton Aronson sequences,
# computed using the find_singleton_upper_bound method
SINGLETON_UPPER = 40


def find_singleton_upper_bound():
    """
    Computes the minimal  upper bound for finding singleton Aronson sequences

    :return: The upper bound integer for singleton sequences search space.
    """
    lower = len(STR_START.replace(" ", ""))
    upper = 100  # >= len("t is the one-hundredth letter"), len(ordinal(n)) >= len(ordinal(100)) for n >= 100

    for m in range(2 * lower, upper):
        len_ord_m = len(n2w(m))
        # Check that all larger ordinals are at least as long
        is_candidate = all([len(n2w(k)) >= len_ord_m for k in range(m + 1, upper)])
        # Ensure m is long enough relative to surrounding text
        is_long_enough = (m >= 2 * lower + len_ord_m)

        if is_candidate and is_long_enough:
            return m


class VerifierError(Exception):
    """
    Custom exception raised when the Aronson sequence verification fails.

    :param message: The error message to be shown.
    :param input_data: The input data that caused the failure.
    """

    def __init__(self, message="Verifier failed", input_data=None):
        self.message = message
        self.input_data = input_data
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}: {self.input_data}"


class GenError(Exception):
    """
        Custom exception raised when the Aronson sequence verification fails.

        :param message: The error message to be shown.
        :param input_data: The input data that caused the failure.
        """

    def __init__(self, message="generating failed", input_data=None):
        self.message = message
        self.input_data = input_data
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}: {self.input_data} is forward-referring"


class ExtendError(Exception):
    """
        Custom exception raised when the Aronson sequence verification fails.

        :param message: The error message to be shown.
        :param input_data: The input data that caused the failure.
        """

    def __init__(self, n, message="Extension failed", input_data=None):
        self.message = message
        self.len = n
        self.input_data = input_data
        super().__init__(self.message)

    def __str__(self):
        prefix = f"{self.input_data} " if self.input_data is not None else ""
        return f"{self.message}: {prefix}could not be extended by {self.len}"


class AronsonSet:
    """
    Class for generating AronsonSequence objects. This is equivalent to A_t(->) or A_t(<-).
    Provides methods for generating Aronson sequences and verifying their validity.

    :param letter: The letter used for generating sequences.
    :param forward: Whether the sequences should be generated in the forward direction.
    """

    def __init__(self, letter: str, forward: bool):
        self.letter = letter  # Letter used for generating sequences
        self.forward = forward  # Sequence direction

    def verifier(self, sequence: AronsonSequence) -> bool:
        """
        Verifies that an AronsonSequence belongs to the set by checking its string representation
        and elements with respect to the specified letter.

        :param sequence: The AronsonSequence to be verified.
        :return: True if the sequence is valid, False otherwise.
        """
        s = sequence.get_sentence()
        if not self.forward:
            s = s[::-1]  # Reverse the string representation if backward direction
        try:
            return all(s[i - 1] == self.letter.lower() for i in sequence.elements)
        except IndexError:
            return False

    def gen_sequence(self, n: int, inp: AronsonSequence = None) -> AronsonSequence:
        """
        Generates a new AronsonSequence of length n either from scratch or by extending an existing sequence.

        :param n: The length of the desired AronsonSequence.
        :param inp: An optional input AronsonSequence to extend. If None, a new sequence is generated.
        :return: A new AronsonSequence object.
        """
        if inp is None:
            # generate new object
            seq = list(islice(self._agen(), n))  # Generate Aronson's sequence from scratch
            inp = AronsonSequence(self.letter, seq, self.forward)

        else:
            if not self.verifier(inp):
                raise VerifierError(input_data=inp.get_elements())  # Verifier failed
            elif inp.has_forward_referring():
                # can't be extended without compromising correctness
                raise GenError(input_data=inp)
            # extend input using generator
            new_indices = list(islice(self._agen(inp=inp), n - len(inp)))
            # update sequence
            inp.set_elements(new_indices, append=True)

        if len(inp) != n:
            # could not extend sequence or generate from scratch up to desired length
            raise ExtendError(n, input_data=inp)
        return inp

    def _agen(self, inp: AronsonSequence = None):
        """
        Internal generator for generating indices based on the current sequence or from scratch.

        :param inp: An optional AronsonSequence to continue generating from.
        :return: A generator yielding new indices for the sequence.
        """
        if inp:
            # Start from the last index of the input sequence to prevent complications. Check this later
            idx = max(inp.get_elements())
            # slice string_repr to start from relevant place
            s = inp.get_sentence()[idx:-len(STR_END.replace(" ", ""))]
        else:
            idx = 0
            s = (self.letter + STR_START).replace(" ", "") if self.forward else STR_END[::-1].replace(" ", "")

        while True:
            idx_rel = 1 + s.find(self.letter)  # Find the relative position of the letter
            if idx_rel <= 0:  # Letter not found in string buffer
                break
            idx += idx_rel
            yield idx
            extend = n2w(idx) if self.forward else n2w(idx)[::-1]  # Extend the string buffer
            s = s[idx_rel:] + extend

    def generate_singletons(self):
        """
        Generates all single-index valid AronsonSequences (singletons).

        :return: A generator yielding valid singleton AronsonSequence objects.
        """
        for idx in range(1, SINGLETON_UPPER):
            candidate = AronsonSequence(self.letter, [idx], self.forward)
            if self.verifier(candidate):  # If valid, yield the singleton sequence
                yield candidate

    def generate_variations(self, n: int, inp: AronsonSequence = None):
        """
        Generates variations of AronsonSequences of length n, either from scratch or by modifying an existing sequence.

        :param n: The length of the AronsonSequences to generate.
        :param inp: An optional input AronsonSequence to modify. If None, sequences are generated from scratch.
        :return: A generator yielding new AronsonSequence variations.
        """
        orig = self.gen_sequence(n, inp)
        start_idx = -1
        stack = [(orig, start_idx)]  # Stack to keep track of the current state
        while stack:
            # start from yielding original
            cur, cur_idx = stack.pop()
            yield cur
            for idx in range(cur_idx + 1, n - 1):
                new_indices = cur.indices[:idx] + [cur.indices[idx + 1]]  # Modify indices for new variation
                new_inp = AronsonSequence(inp.letter, new_indices, inp.get_direction())
                try:
                    extend = self.gen_sequence(n, new_inp)
                except VerifierError:
                    continue  # Skip invalid variations
                stack.append((extend, idx))

    def get_generator(self, n: int, inp: AronsonSequence = None):
        """
        Returns a generator for generating variations of AronsonSequences.

        :param n: The length of the sequences to generate.
        :param inp: An optional input AronsonSequence to modify. If None, sequences are generated from scratch.
        :return: A generator yielding new AronsonSequence variations.
        """
        return self.generate_variations(n, inp)

    def intersect_aronson_sets(self, other: 'AronsonSet', n: int, inp1: AronsonSequence = None,
                               inp2: AronsonSequence = None):
        """
        Computes the intersection of two AronsonSets by comparing their generated sequences and finding common subsets.

        :param other: The other AronsonSet to intersect with.
        :param n: The length of the sequences to generate.
        :param inp1: An optional input AronsonSequence for the first generator.
        :param inp2: An optional input AronsonSequence for the second generator.
        :return: A set of common valid AronsonSequences that belong to both sets.
        """
        seen_self, seen_other = set(), set()
        # generate all sequences up to length n from each generator
        for i in range(n):
            for seq1, seq2 in zip(self.generate_variations(i, inp1), other.get_generator(i, inp2)):
                seen_self.update(seq1)
                seen_other.update(seq2)
        # intersection of seen sets
        common_subsets = seen_self & seen_other

        return {subset for subset in common_subsets if self.verifier(subset) and other.verifier(subset)}

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
