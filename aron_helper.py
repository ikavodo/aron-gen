from copy import deepcopy
import random
from timeit import default_timer
from itertools import islice, chain, combinations
from functools import reduce

from num2words import num2words

# consts
START = " is the "
END = " letter"
# use this for finding sequences starting with ordinals outside of prefix/suffix range
SEARCH_LOWER_BOUND = max(len(START.replace(" ", "")), len(END.replace(" ", "")))


# Define the custom exception class
class VerifierError(Exception):
    def __init__(self, message="Verifier failed", input_data=None):
        self.message = message
        self.input_data = input_data  # Store the input that caused the failure
        super().__init__(self.message)

    def __str__(self):
        # Customize the string representation to include input data
        return f"{self.message}: {self.input_data}"


# nifty func to be used for simpler analysis- 1-> 1st, 2-> 2nd etc...
def ordinal(n):
    s = ('th', 'st', 'nd', 'rd') + ('th',) * 10
    v = n % 100
    if v > 13:
        return f'{n}{s[v % 10]}'
    else:
        return f'{n}{s[v]}'


# helper
def n2w(n):
    os = num2words(n, ordinal=True).replace(" and", "")
    return os.replace(", ", "").replace(" ", "").replace("-", "")




# Helper- generates sequence from generator and input
def aronson(letter, n, forward=True, inp=None, forward_referring=True):
    # generates 'the rest' from input
    if inp is None:
        seq = list(islice(agen_better(letter, forward=forward), n))
    else:
        seq = inp + list(
            islice(agen_better(letter, forward=forward, inp=inp, forward_referring=forward_referring), n - len(inp)))
    # bug: if input is of length n then list(islice(...,0)) doesn't evaluate generator- no VerifierError.
    return seq


# Extension of Michael Branicky's implementation in OEIS, allowing for generating backwards-true sequences, as well
# as starting from a point in the middle
# see https://oeis.org/A005224
def agen_better(letter, forward=True, inp=None, forward_referring=True):
    def update_from_inp(forward, inp, letter, s):
        # check is correct, be pessimistic
        if not verifier(letter, inp, forward, forward_referring):
            raise VerifierError(input_data=inp)
        for i in inp:
            s += (n2w(i) if forward else n2w(i)[::-1])
        idx = max(inp)  # inp[-1] if monotonically rising
        s = s[idx:]
        return s, idx

    # Initialize sequence based on direction
    s = (letter + START).replace(" ", "") if forward else END.replace(" ", "")[::-1]
    idx = 0
    if inp:
        # update to given point
        s, idx = update_from_inp(forward, inp, letter, s)
    while True:
        try:
            # Find the next occurrence of the letter. If not inside s.find() returns -1, so repeats infinitely
            idx_rel = 1 + s.find(letter)
        except ValueError:
            # not sure this is necessary
            break  # Stop if letter is not found
        idx += idx_rel
        yield idx
        # Adjust the sequence for the next iteration
        s = s[idx_rel:] + (n2w(idx) if forward else n2w(idx)[::-1])


# correct but slow because requires generating the entire sequence
def verifier_slow(letter, indices, forward=True):
    sequence = get_sequence(letter, indices, forward)
    # Reverse the sequence if forward is False
    if not forward:
        sequence = sequence[::-1]
    # Check if the input letter matches at the specified indices
    try:
        return all(sequence[i - 1] == letter.lower() for i in indices)
    except IndexError:
        return False


def test_verifiers(letter, n, forward, *verifiers):
    seq = aronson(letter, n)
    execution_times = {}

    # Loop over verifiers
    for _ in range(n):
        # Generate a new random seq_copy for each iteration
        seq_copy = deepcopy(seq)
        random_int = random.randint(1, n - 1)
        seq_copy[(random_int - 1):(random_int + 1)] = range((seq_copy[random_int] - 1), (seq_copy[random_int] + 1))

        # Loop over provided verifiers
        for verifier in verifiers:
            # Time the current verifier using the same seq_copy
            start_time = default_timer()
            # make sure no such sequence is actually correct
            verifier(letter, seq_copy, forward)
            end_time = default_timer()

            # Store the execution time for the verifier
            execution_times.setdefault(verifier.__name__, []).append(end_time - start_time)

    # Calculate and print average times
    for verifier_name, times in execution_times.items():
        avg_time = sum(times) / n
        print(f"Average execution time for {verifier_name}: {avg_time:.4f} seconds")


# Use slow verifier for debugging
def verifier(letter, indices, forward=True, forward_referring=True):
    if forward_referring:
        return verifier_slow(letter, indices, forward)
    else:
        # prior knowledge about monotonicity allows function to run faster
        return verifier_fast(letter, indices, forward)


# only works for non-forwards referring sequences!
def verifier_fast(letter, indices, forward=True):
    # Determine the initial sequence based on the direction (forward or reverse)
    s = (letter + START).replace(" ", "") if forward else END.replace(" ", "")[::-1]
    for idx in indices:
        # Add the ordinal to the sequence
        s += (n2w(idx) if forward else n2w(idx)[::-1])
        try:
            # Check if the letter matches the expected position
            if s[idx - 1] != letter.lower():
                return False
        except IndexError:
            # If the index is out of range, return False
            return False
    return True


def find_singletons(letter):
    def find_singleton_upper_bound():
        lower = len(START.replace(" ", ""))
        upper = 100  # For n >= 100, len(ordinal(n)) >= len("one-hundred")

        for m in range(2 * lower, upper):
            len_ord_m = len(n2w(m))
            # Check that all larger ordinals are at least as long
            is_candidate = all([len(n2w(k)) >= len_ord_m for k in range(m + 1, upper)])
            # Ensure m is long enough relative to surrounding text
            is_long_enough = (m >= 2 * lower + len_ord_m)

            if is_candidate and is_long_enough:
                return m

    upper = find_singleton_upper_bound()
    return {(i,) for i in range(upper) if verifier(letter, (i,), True) and verifier(letter, (i,), False)}


# Generate sentence from sequence input
def get_sequence(letter, indices, forward=True, delimited=False):
    delim = ', ' if delimited else ''
    idx_ord = indices if forward else indices[::-1]
    pref =  letter + START if delimited else  (letter + START).replace(" ", "")
    mid = (num2words(idx, ordinal=True) if delimited else n2w(idx) for idx in idx_ord)
    suff = END if delimited else END.replace(" ", "")
    return f"{pref}{delim.join(mid)}{suff}"


# wrapper for printing, will become __repr__ in object-oriented approach
def print_sequence(letter, indices, forward=True):
    print(get_sequence(letter, indices, forward, True))


def generate_variations(letter, n, forward=True, inp=None):
    # where to start generating variations from
    orig = aronson(letter, n, forward, inp)
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
                extend = aronson(letter, n, forward, inp)
            except VerifierError:
                # verifier returned false- meaning sequence is incorrect and can't be extended
                continue
            stack.append((extend, idx))


def intersect_aronson_sets_small(letter, n):
    # helper for generating power set
    def power_set(nums):
        return set(chain.from_iterable(combinations(nums, r) for r in range(len(nums) + 1)))

    forwards = aronson(letter, n, forward=True)
    backwards = aronson(letter, n, forward=False)
    # Look at subsequence intersection
    intersection = power_set(forwards) & power_set(backwards)
    # Subsequences need to return True for both verifiers
    return {i for i in intersection if
            verifier(letter, i, forward=True, forward_referring=True) and verifier(letter, i, forward=False,
                                                                                   forward_referring=True)}


def intersect_aronson_sets(letter, n):
    def power_set(seq):
        return set(chain.from_iterable(combinations(seq, r) for r in range(len(seq) + 1)))

    # Initialize sets to accumulate power sets for both directions
    power_sets = [set(), set()]
    # Create generators for both directions
    gen_forward = generate_variations(letter, n, True)
    gen_backward = generate_variations(letter, n, False)
    for seq_fwd, seq_bwd in zip(gen_forward, gen_backward):
        power_sets[0].update(power_set(seq_fwd))
        power_sets[1].update(power_set(seq_bwd))

    # Compute intersection of both power sets
    common_subsets = power_sets[0] & power_sets[1]
    # Filter by verifier
    return {subset for subset in common_subsets if verifier(letter, subset, True) and
            verifier(letter, subset, False)}


# These two will be addressed next time. a is the letter-> a is an outlier
def is_outlier(letter):
    s_pref, s_suff = (letter + START).replace(" ", ""), END.replace(" ", "")
    s = s_pref + s_suff
    return s[::-1].find(letter) == len(s) - 1


# Generate singleton classes via a correction approach. Prove that it terminates
# (numbers grow larger than their ordinal reps)
def gen_outliers(letter):
    s_pref, s_suff = (letter + START).replace(" ", ""), END.replace(" ", "")
    seen_idxs = {None}  # Start with the initial index 0
    new_indices = {None}  # Set of indices to check
    while new_indices:
        next_indices = set()  # Set to collect new indices for the next iteration
        # Iterate over the current new indices to check and extend them
        # problem with yield when singleton set? new_indices={20}
        for n_i in new_indices:
            # Form the string to check the correct index positions
            if n_i is not None:
                s = s_pref + n2w(n_i) + s_suff
                if n_i >= len(s):
                    continue
                # Check if the index is correct
                if s[::-1][n_i - 1] == letter:
                    # TypeError: 'set_iterator' object is not subscriptable
                    yield n_i  # Yield the correct index

            else:
                s = s_pref + s_suff
            # ignore stuff after last ordinal!!!
            indices = {i for i, char in enumerate(s[::-1]) if char == letter}
            next_indices.update(indices.difference(seen_idxs))  # Add only unseen indices

        # Update the seen indices and new indices for the next iteration
        seen_idxs.update(next_indices)
        new_indices = next_indices  # Set new_indices for the next loop
