`from itertools import islice, cycle, combinations
from AronsonSequence import AronsonSequence, n2w, STR_START, STR_END

# upper bound for searching for singleton Aronson sequences, computed using the next method
SINGLETON_UPPER = 40

def find_singleton_upper_bound():
    lower = len(STR_START.replace(" ", ""))
    upper = 100  # For n >= 100, len(ordinal(n)) >= len("one-hundred"), 100 >= len("t is the one-hundredth letter")

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
    def __init__(self, letter: str, forward: bool):
        self.letter = letter
        self.forward = forward

    def verifier(self, sequence: AronsonSequence) -> bool:
        s = sequence.string_repr
        if not self.forward:
            s = s[::-1]
        try:
            return all(s[i - 1] == self.letter.lower() for i in sequence.indices)
        except IndexError:
            return False

    def gen_aronson(self, n: int, inp: AronsonSequence = None) -> AronsonSequence:
        if inp is None:
            seq = list(islice(self._agen(), n))
        else:
            if not self.verifier(inp):
                raise VerifierError(input_data=inp.indices)
            seq = inp.indices + list(
                islice(self._agen(inp=inp), n - len(inp.indices)))
        return AronsonSequence(self.letter, seq, self.forward)

    def _agen(self, inp: AronsonSequence = None):
        if inp:
            idx = max(inp.indices)
            s = inp.string_repr[idx:-len(STR_END.replace(" ", ""))]
        else:
            idx = 0
            s = (self.letter + STR_START).replace(" ", "") if self.forward else STR_END[::-1].replace(" ", "")

        while True:
            idx_rel = 1 + s.find(self.letter)
            if idx_rel <= 0:
                break
            idx += idx_rel
            yield idx
            extend = n2w(idx) if self.forward else n2w(idx)[::-1]
            s = s[idx_rel:] + extend

    def generate_singletons(self):
        """Yields all single-index valid AronsonSequences."""
        for idx in range(1, SINGLETON_UPPER):
            candidate = AronsonSequence(self.letter, [idx], self.forward)
            if self.verifier(candidate):
                yield candidate

    def generate_variations(self, n, inp: AronsonSequence = None):
        # where to start generating variations from
        orig = self.gen_aronson(n, inp)
        start_idx = -1
        stack = [(orig, start_idx)]
        while stack:
            # keep track of which idx was omitted
            cur, cur_idx = stack.pop()
            yield cur
            # only remove indices forward
            for idx in range(cur_idx + 1, n - 1):
                # important: keep track of which index was used to generate the current sequence
                inp = cur[:idx] + [cur[idx + 1]]
                try:
                    extend = self.gen_aronson(n, inp)
                except VerifierError:
                    # verifier returned false- meaning sequence is incorrect and can't be extended
                    continue
                stack.append((extend, idx))

    def intersect_aronson_sets(self, other, n):
        def power_set(seq: AronsonSequence):
            return set(
                AronsonSequence(seq.letter, list(sub), seq.direction)
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

    def union(self, other_set: 'AronsonSet'):
        """Yields all unique sequences from both sets (alternating order)."""
        seen = set()
        generators = [self.generate_variations(1_000), other_set.generate_variations(1_000)]
        for g in cycle(generators):
            try:
                seq = next(g)
                if seq not in seen:
                    seen.add(seq)
                    yield seq
            except StopIteration:
                break
`