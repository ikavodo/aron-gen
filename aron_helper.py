from itertools import islice
from num2words import num2words
import string
import sys

# consts
BASE_START = " is the "
BASE_END = " letter"
# use this for finding sequences starting with ordinals outside of prefix/suffix range
SEARCH_LOWER_BOUND = max(len(BASE_START.replace(" ", "")), len(BASE_END.replace(" ", "")))


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


def pows_two():
    last = 1
    l = [last]
    while True:
        last *= 2
        if last >= sys.maxsize:
            break
        l.append(last)
    return l


# Extension of Michael Branicky's implementation in OEIS, allowing for generating backwards-true sequences, as well
# as starting from a point in the middle
# see https://oeis.org/A005224
def agen_better(letter, forward=True, inp=None):
    def update_from_inp(forward, inp, letter, s):
        # check is correct
        if not verifier(letter, inp, forward):
            raise ValueError('incorrect inputs')
        for i in inp:
            s += (n2w(i) if forward else n2w(i)[::-1])
        idx = max(inp)  # inp[-1] if monotonically rising
        s = s[idx:]
        return s, idx

    # Initialize sequence based on direction
    s = (letter + BASE_START).replace(" ", "") if forward else BASE_END.replace(" ", "")[::-1]
    idx = 0
    if inp:
        # update to given point
        s, idx = update_from_inp(forward, inp, letter, s)
    while True:
        try:
            # Find the next occurrence of the letter. If not inside s.find() returns -1, so repeats infinitely
            idx_rel = 1 + s.find(letter)
        except ValueError:
            break  # Stop if letter is not found
        idx += idx_rel
        yield idx
        # Adjust the sequence for the next iteration
        s = s[idx_rel:] + (n2w(idx) if forward else n2w(idx)[::-1])

#correct but slow because requires generating the entire sequence
def verifier_slow(letter, indices, forward=True):
    sequence = get_sequence(letter, indices, forward)
    # Reverse the sequence if forward is False
    if not forward:
        sequence = sequence[::-1]
    # Check if the input letter matches at the specified indices
    return all(sequence[i - 1] == letter for i in indices)


#only works for backwards refering sequences
def verifier(letter, indices, forward=True):
    # Determine the initial sequence based on the direction (forward or reverse)
    s = (letter + BASE_START).replace(" ", "") if forward else BASE_END.replace(" ", "")[::-1]
    prev_idx = 0
    is_mono = all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))
    for idx in indices:
        # For monotonic sequences, compute relative index
        idx_rel = idx - prev_idx if is_mono else 0

        # Add the ordinal to the sequence
        s += (n2w(idx) if forward else n2w(idx)[::-1])

        try:
            # Check if the letter matches the expected position
            if s[(idx_rel if is_mono else idx) - 1] != letter:
                return False
        except IndexError:
            # If the index is out of range, return False
            return False

        prev_idx = idx
        s = s[idx_rel:]  # Adjust sequence for next iteration (monotonic case)

    return True

# Generate sentence from sequence input
def get_sequence(letter, indices, forward=True, delimited=False):
    delim = ', ' if delimited else ''
    idx_ord = indices if forward else indices[::-1]
    pref = (letter + BASE_START).replace(" ", "") if not delimited else letter + BASE_START
    mid = (num2words(idx, ordinal=True) if delimited else n2w(idx) for idx in idx_ord)
    suff = BASE_END.replace(" ", "") if not delimited else BASE_END
    return f"{pref}{delim.join(mid)}{suff}"


# wrapper for printing, will become __repr__ in object-oriented approach
def print_sequence(letter, indices, forward=True):
    print(get_sequence(letter, indices, forward, True))


# Need to understand what I'm doing. a is the letter-> a is an outlier
def is_outlier(letter):
    s_pref, s_suff = (letter + BASE_START).replace(" ", ""), BASE_END.replace(" ", "")
    s = s_pref + s_suff
    return s[::-1].find(letter) == len(s) - 1


# Generate singleton classes via a correction approach. Prove that it terminates
# (numbers grow larger than their ordinal reps)
def gen_outliers(letter):
    s_pref, s_suff = (letter + BASE_START).replace(" ", ""), BASE_END.replace(" ", "")
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


#
# #Iteratively adds and checks ordinals, for non-monotonic examples, such as 't is the twelfth, seventh letter' (->)
# def verifier(letter, indices, forward=True):
#     s = (letter + BASE_START).replace(" ", "") if forward else BASE_END.replace(" ", "")[::-1]
#     for idx in indices:
#         s += (n2w(idx) if forward else n2w(idx)[::-1])
#         try:
#             # Find the next occurrence of the letter
#             if s[idx-1] != letter:
#                 return False
#         except IndexError:
#             # Inherently false
#             return False
#         # Adjust the sequence for the next iteration
#     return True
#
#
# # This should be used for monotonic sequences- more space efficient. Otherwise this should be put inside the other one,
# # where the other one builds the string iteratively in entirety
# def verifier_monotonic(letter, indices, forward=True):
#     #problem- what if receive a generator?
#     # if len(indices)>1:
#     #     #check if monotonically increasing
#     #     if not all(indices[i] <= indices[i+1] for i in range(len(indices) - 1)):
#     #         print('Use general verifier method')
#     #         return None
#
#     s = (letter + BASE_START).replace(" ", "") if forward else BASE_END.replace(" ", "")[::-1]
#     prev_idx = 0
#     for idx in indices:
#         idx_rel = idx - prev_idx
#         s += (n2w(idx) if forward else n2w(idx)[::-1])
#         try:
#             # Find the next occurrence of the letter
#             if s[idx_rel - 1] != letter:
#                 return False
#         except IndexError:
#             #Inherently false
#             return False
#         s = s[idx_rel:]
#         prev_idx = idx
#         # Adjust the sequence for the next iteration
#     return True
