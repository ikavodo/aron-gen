import string
import timeit
import aron_helper
from itertools import islice


# Test script
def test_sequence():
    max_tests = 10  # Number of indices to test per letter/algorithm
    for letter in string.ascii_lowercase:
        # Test forward algorithm
        print(f"Testing forward algorithm for letter '{letter}'...")
        generator = aron_helper.agen_better(letter, True)
        indices = list(islice(generator, max_tests))
        assert aron_helper.verifier(letter, indices), f"Forward algorithm failed for letter '{letter}'"

        # Test reverse algorithm
        print(f"Testing reverse algorithm for letter '{letter}'...")
        # what if I try testing for outliers anyway? what will happen?
        if aron_helper.is_outlier(letter):
            generator = aron_helper.gen_outliers(letter)
        else:
            generator = aron_helper.agen_better(letter, False)
        # generator = agen(letter, False)
        indices = list(islice(generator, max_tests))
        # Reconstruct the sequence to verify
        assert aron_helper.verifier(letter, indices, False), f"Reverse algorithm failed for letter '{letter}'"

    print("All tests passed!")


if __name__ == '__main__':
    seen = []
    n = 5
    letter = 't'
    # [print(seq) for seq in aron_helper.generate_variations(n, letter, False)
    #        if seq not in seen and not seen.append(seq)]
    seqs = aron_helper.intersect_aronson_sets(letter, n)
    for seq in seqs:
        assert(aron_helper.verifier(letter, seq, True))
        assert (aron_helper.verifier(letter, seq, False))


    assert (aron_helper.verifier('t', [18, 8], forward=False, forward_referring=True))
    assert (aron_helper.verifier('l', indices=[1, 23], forward=True, forward_referring=True))

    #  check working from input
    letter = 't'
    num_ind = 5
    for inp in [[], [1, 11], [10, 12]]:
        # notice second and third converge
        extend = list(islice(aron_helper.agen_better(letter, forward=True, inp=inp), num_ind - len(inp)))
        series = inp + extend
        aron_helper.print_sequence(letter, series)
        # breaks mono implementation....
    print(str(aron_helper.verifier(letter, [10, 12, 7, 17], True)))

# if __name__ == '__main__':
    # # Set up argument parser
    # parser = argparse.ArgumentParser(description="Generate sequence based on the provided letter.")
    # parser.add_argument("letter", type=str, help="A single letter from a-z.")
    # parser.add_argument(
    #     "--algorithm",
    #     choices=["forward", "reverse"],
    #     default="forward",
    #     help="Choose the algorithm for generating the sequence.",
    # )
    # parser.add_argument(
    #     "--count",
    #     type=int,
    #     default=10,
    #     help="Number of words to generate. Default is 10.",
    # )
    #
    # # Parse the command-line arguments
    # args = parser.parse_args()
    #
    # # Validate the input
    # letter = args.letter.strip().lower()
    # if len(letter) != 1 or letter not in string.ascii_lowercase:
    #     print("Invalid input! Please enter a single letter (a-z).")
    #     exit()
    #
    # count = args.count
    # if count <= 0:
    #     print("Invalid count! Please specify a positive integer.")
    #     exit()
    #
    # # Select and run the algorithm
    #
    # # Select and run the algorithm
    # if args.algorithm == "forward":
    #     s_suff = ''
    #     generator = agen(letter) #this alg takes care of all cases
    #     indices = list(islice(generator, count))
    #     for i in indices:
    #         s_suff = s_suff + num2words(i, ordinal=True) + DELIMITER
    #     print(letter + S_PREF + s_suff[:-2] + S_END)
    #
    # else:  # args.algorithm == "reverse"
    #
    #     # Handle outlier letters
    #     if is_outlier(letter):
    #         indices = list(islice(gen_outliers(letter), count))
    #         if not indices:
    #             print("No sequence exists")
    #             exit()
    #         # Print all outlier options for this letter
    #         [print(letter + S_PREF + num2words(i, ordinal=True) + S_END) for i in indices]
    #     else:
    #         generator = agen_rev(letter)
    #         #same for forward and backward
    #         indices = list(islice(generator, count))
    #         s_suff = S_END
    #         for i, ord in enumerate(indices):
    #             s_suff = num2words(ord, ordinal=True) + (DELIMITER if i else '') + s_suff
    #             fixed = s_suff.replace(" ", "").replace("-", "").replace(",", "")
    #             if fixed[::-1][ord] != letter:
    #                 raise AssertionError
    #         print(letter + S_PREF + s_suff)
