from itertools import islice, cycle, combinations
from AronsonSequence import AronsonSequence, n2w, STR_START, STR_END

# upper bound for searching for singleton Aronson sequences,
# computed using the find_singleton_upper_bound method
SINGLETON_UPPER = 40


def find_singleton_upper_bound():
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
    def __init__(self, message="Verifier failed", input_data=None):
        self.message = message
        self.input_data = input_data
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}: {self.input_data}"


class AronsonSet:
    """
    class for generating AronsonSequence objects. Equivalent to A_t(->) or A_t(<-).
    See https://ikavodo.github.io/aronson-1/
    """

    def __init__(self, letter: str, forward: bool):
        # letter to be used for generating sequences
        self.letter = letter
        # sequence direction
        self.forward = forward

    def verifier(self, sequence: AronsonSequence) -> bool:
        """
        Used to verify that an AronsonSequence belongs to set.
        :param sequence: to be verified
        :return: True or False
        """
        # use getter function- better.
        s = sequence.string_repr
        if not self.forward:
            # reverse string_repr
            s = s[::-1]
        try:
            # comparison of indices with occurences of letter in string_repr
            return all(s[i - 1] == self.letter.lower() for i in sequence.indices)
        except IndexError:
            # AronsonSequence('t',[10000],forward=True)<->'t is the ten-thousandth letter' throws IndexError
            return False

    def gen_aronson(self, n: int, inp: AronsonSequence = None) -> AronsonSequence:
        """
        generate a new AronsonSequence of length n from input or from scratch
        :param n: length of AronsonSequence
        :param inp: to generate from
        :return: new AronsonSequence object
        """
        if inp is None:
            # gen from scratch
            seq = list(islice(self._agen(), n))
        else:
            if not self.verifier(inp):
                # input incorrect
                raise VerifierError(input_data=inp.indices)
            # generate continuation of indices from input
            seq = inp.indices + list(
                islice(self._agen(inp=inp), n - len(inp.indices)))
        return AronsonSequence(self.letter, seq, self.forward)

    def _agen(self, inp: AronsonSequence = None):
        """
        internal generator, based on Michael Branicky's implementation in https://oeis.org/A005224
        :param inp: to be used
        :return: new indices
        """
        if inp:
            # we can use inp's internal string representation to continue generating
            idx = max(inp.indices)
            s = inp.string_repr[idx:-len(STR_END.replace(" ", ""))]
        else:
            # generate from scratch
            idx = 0
            s = (self.letter + STR_START).replace(" ", "") if self.forward else STR_END[::-1].replace(" ", "")

        while True:
            # loop for generating new backwards-referring indices
            idx_rel = 1 + s.find(self.letter)
            if idx_rel <= 0:
                # letter not found in string buffer
                break
            idx += idx_rel
            yield idx
            # extension to string buffer
            extend = n2w(idx) if self.forward else n2w(idx)[::-1]
            s = s[idx_rel:] + extend

    def generate_singletons(self):
        """
        Yields all single-index valid AronsonSequences.
        :return: generator of singleton AronsonSequence objects
        """
        # bounded search space resulting from ordinal lengths
        for idx in range(1, SINGLETON_UPPER):
            candidate = AronsonSequence(self.letter, [idx], self.forward)
            if self.verifier(candidate):
                # is singleton
                yield candidate

    def generate_variations(self, n, inp: AronsonSequence = None):
        """
        generator for new AronsonSequences of length n, either from scratch or from existing input
        :param n: length of new AronsonSequences
        :param inp: to generate from
        :return: generator
        """
        # where to start generating variations from
        orig = self.gen_aronson(n, inp)
        start_idx = -1
        stack = [(orig, start_idx)]
        # first generated instance is orig
        while stack:
            # keep track of which idx was omitted
            cur, cur_idx = stack.pop()
            yield cur
            # only remove indices forward
            for idx in range(cur_idx + 1, n - 1):
                # important: keep track of which index was used to generate the current sequence
                new_indices = cur.indices[:idx] + [cur.indices[idx + 1]]
                new_inp = AronsonSequence(inp.letter, new_indices, inp.forward)
                try:
                    extend = self.gen_aronson(n, new_inp)
                except VerifierError:
                    # verifier returned false, meaning sequence is incorrect and can't be extended
                    continue
                stack.append((extend, idx))

    def intersect_aronson_sets(self, other, n):
        def power_set(seq: AronsonSequence):
            return set(
                AronsonSequence(seq.letter, list(sub), seq.forward)
                for r in range(len(seq) + 1)
                for sub in combinations(seq.indices, r)
            )

        power_sets = [set(), set()]

        for seq1, seq2 in zip(self.generate_variations(n), other.generate_variations(n)):
            power_sets[0].update(power_set(seq1))
            power_sets[1].update(power_set(seq2))

        common_subsets = power_sets[0] & power_sets[1]

        return {
            subset for subset in common_subsets
            if self.verifier(subset) and other.verifier(subset)
        }

    def union(self, other_set: 'AronsonSet', n: int):
        """Yields all unique sequences of length n from both sets (alternating order)."""
        seen = set()
        generators = [self.generate_variations(n), other_set.generate_variations(n)]
        for g in cycle(generators):
            try:
                seq = next(g)
                if seq not in seen:
                    seen.add(seq)
                    yield seq
            except StopIteration:
                break
