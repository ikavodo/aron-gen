import unittest
from num2words import num2words
from AronsonSequence import n2w, AronsonSequence, PREFIX, SUFFIX, ReferralType, Direction, REPR_FORWARD, \
    REPR_BACKWARD  # Adjust your_module as needed


# unit test for AronsonSequence
class AronsonSequenceTests(unittest.TestCase):

    def test_construction(self):
        seq = AronsonSequence('T', [1, 2, 3], Direction.FORWARD)
        # check is converted to lower_case
        self.assertEqual(seq.letter, 't')
        self.assertEqual(seq.elements, [1, 2, 3])
        self.assertTrue(seq.direction)

        # Invalid letters
        for invalid_letter in ['tt', '!', '']:
            with self.subTest(letter=invalid_letter):
                with self.assertRaises(ValueError):
                    AronsonSequence(invalid_letter, [1], Direction.FORWARD)

        # Invalid elements
        invalid_elements_list = [
            [],  # empty list
            [1, 'x'],  # contains non-int
            [0, 1],  # contains non-positive
            [1, 1]  # contains duplicates
        ]
        for invalid_elements in invalid_elements_list:
            with self.subTest(elements=invalid_elements):
                with self.assertRaises(ValueError):
                    AronsonSequence('t', invalid_elements, Direction.FORWARD)

    def test_repr(self):
        # test that human representation is correct
        seq = AronsonSequence('t', [25], Direction.FORWARD)
        self.assertTrue(repr(seq)[0].isupper())
        self.assertIn(PREFIX + num2words(25, ordinal=True) + SUFFIX, repr(seq))
        self.assertIn(REPR_FORWARD, repr(seq))
        self.assertEqual(str(seq), repr(seq))
        seq = AronsonSequence('t', [100, 101], Direction.BACKWARD)
        for elem in seq.get_elements():
            self.assertIn(num2words(elem, ordinal=True), str(seq))
        self.assertIn(REPR_BACKWARD, repr(seq))

    def test_sentence(self):
        # test that the internal sentence is correct, use something with hyphen
        elem = 25
        suffix = PREFIX.replace(" ", "") + n2w(elem) + SUFFIX.replace(" ", "")
        elems = [3, 4]
        test_sequences = [
            (AronsonSequence('t', [elem], Direction.FORWARD), suffix),
            (AronsonSequence('t', elems, Direction.BACKWARD), ''.join([n2w(i) for i in elems[::-1]]))
        ]

        for seq, part in test_sequences:
            with self.subTest(seq=seq):
                self.assertIn(part, seq.get_sentence())

    def test_has_forward_refer(self):
        # check that has_forward_referring() method works correctly
        test_cases = [
            # backwards referring in both directions
            (AronsonSequence('t', [1, 4, 11], Direction.FORWARD), False),
            (AronsonSequence('t', [3, 4], Direction.BACKWARD), False),

            # self referring in both directions
            (AronsonSequence('t', [10, 12], Direction.FORWARD), False),
            (AronsonSequence('t', [8, 14], Direction.BACKWARD), False),

            # forward referring in both directions
            (AronsonSequence('t', [19, 100], Direction.FORWARD), True),
            (AronsonSequence('t', [19, 100], Direction.BACKWARD), True),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.has_forward_referring(), expected)

    def test_all_refers(self):
        # check that refer_dictionaries are correct
        test_cases = [
            # all referring types- BACKWARD, SELF, BACKWARD, SELF, FORWARD
            (AronsonSequence('t', [1, 12, 17, 30, 100], Direction.FORWARD), True),
            (AronsonSequence('t', [3, 13, 21, 34, 100], Direction.BACKWARD), True)
        ]

        # problem in backwards case. Why?
        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                ref_vals = set(seq.get_referral_dict().values())
                self.assertEqual(ref_vals, set(ReferralType))

    def test_is_correct(self):
        test_cases = [
            # all of these are correct sentences

            # backward referring
            (AronsonSequence('t', [1, 4, 11], Direction.FORWARD), True),
            (AronsonSequence('t', [3, 4, 11], Direction.BACKWARD), True),

            # self referring
            (AronsonSequence('t', [10, 12], Direction.FORWARD), True),
            (AronsonSequence('t', [8, 14], Direction.BACKWARD), True),

            # forward referring
            (AronsonSequence('t', [19], Direction.FORWARD), True),
            (AronsonSequence('t', [19], Direction.BACKWARD), True),

            # all referring
            (AronsonSequence('t', [1, 12, 17, 30], Direction.FORWARD), True),
            (AronsonSequence('t', [3, 13, 21, 34], Direction.BACKWARD), True),

            # false
            (AronsonSequence('t', [19, 100], Direction.FORWARD), False),
            (AronsonSequence('t', [19, 100], Direction.BACKWARD), False),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.is_correct(), expected)

    def test_complete(self):
        test_cases = [
            (AronsonSequence('l', [1, 23], Direction.FORWARD), True),
            # also true
            (AronsonSequence('l', [23, 1], Direction.FORWARD), True),
            (AronsonSequence('l', [1], Direction.FORWARD), False),
            (AronsonSequence('j', [24], Direction.BACKWARD), True),
            (AronsonSequence('l', [6], Direction.BACKWARD), False)
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.is_complete(), expected)

    def test_set_elements_append(self):
        suffix_ind = len(SUFFIX) + len(REPR_FORWARD)
        for append_flag in [False, True]:
            seq = AronsonSequence('t', [1, 2], Direction.FORWARD)
            seq_repr = str(seq)
            prev_elements = seq.get_elements().copy()

            # Set elements with the given append flag
            seq.set_elements([3, 4], append=append_flag)

            if append_flag:
                # When append=True, ensure the original elements are retained
                for elem in prev_elements:
                    self.assertIn(elem, seq.get_elements())
                # check string representations work as expected
                self.assertIn(seq_repr[:-suffix_ind], str(seq))
            else:
                # When append=False, the previous elements should be removed
                for elem in prev_elements:
                    self.assertNotIn(elem, seq.get_elements())
                self.assertNotIn(seq_repr[:-suffix_ind], str(seq))

    def test_set_elements_duplicates(self):
        # Test appending only duplicate elements
        seq = AronsonSequence('t', [1, 2], Direction.FORWARD)
        prev_elements = seq.get_elements().copy()
        seq.set_elements([1, 2], append=True)
        # Ensure no new elements are added
        self.assertEqual(seq.get_elements(), prev_elements)
        duplicates = [5, 1, 2, 6]
        # Test appending new elements with some duplicates
        seq.set_elements(duplicates, append=True)
        # Ensure original elements (1, 2) remain and new elements (5, 6) are added in correct order
        self.assertEqual(sorted(duplicates), seq.get_elements())

    def test_set_elements_empty(self):
        seq = AronsonSequence('t', [1, 2], Direction.FORWARD)

        # Test setting elements with an empty list
        # Make sure throws ValueError
        for append_flag in [True, False]:
            with self.subTest(append=append_flag):
                with self.assertRaises(ValueError):
                    seq.set_elements([], append_flag)

    def test_cpy(self):
        seq = AronsonSequence('t', [1, 2], Direction.FORWARD)
        seq_cpy = seq.copy()
        self.assertEqual(seq, seq_cpy)
        seq.append_elements([3])
        # elements are mutable-> copy should not be shallow
        self.assertNotEqual(seq, seq_cpy)

    def test_append_element_wrapper(self):
        seq = AronsonSequence('t', [1, 2], Direction.FORWARD)
        seq_cpy = seq.copy()
        seq.set_elements([3, 4], append=True)
        # check these two do the same thing
        seq_cpy.append_elements([3, 4])
        self.assertEqual(seq, seq_cpy)

    # === Other Method Tests ===

    def test_iter_and_getitem(self):
        seq = AronsonSequence('t', [5, 6, 7], Direction.FORWARD)
        self.assertEqual(list(seq), [5, 6, 7])
        self.assertEqual(seq[1], 6)

    def test_len(self):
        seq = AronsonSequence('x', [10, 20], Direction.FORWARD)
        self.assertEqual(len(seq), 2)

    def test_eq_and_hash(self):
        a = AronsonSequence('z', [1, 2], Direction.FORWARD)
        b = AronsonSequence('z', [1, 2], Direction.FORWARD)
        c = AronsonSequence('z', [1, 2], Direction.BACKWARD)
        d = AronsonSequence('d', [1, 2], Direction.FORWARD)
        e = AronsonSequence('z', [1, 3], Direction.FORWARD)

        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertNotEqual(a, e)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(hash(a), hash(c))
        self.assertNotEqual(hash(a), hash(d))
        self.assertNotEqual(hash(a), hash(e))

    def test_large_sequence(self):
        # Create a very large sequence
        seq = AronsonSequence('t', list(range(1, 1001)), Direction.FORWARD)

        # Test some basic properties
        self.assertEqual(len(seq), 1000)
        self.assertEqual(seq[999], 1000)
        self.assertEqual(seq[0], 1)

    def test_flip_direction(self):
        seq = AronsonSequence('t', [1, 2, 3], Direction.FORWARD)
        seq.set_elements([4, 5, 6], append=True)

        # First set to other direction
        seq.flip_direction()
        seq.set_elements([7, 8], append=True)

        # Now check if the order of elements is consistent
        self.assertEqual(seq.get_sentence(),
                         AronsonSequence('t', [1, 2, 3, 4, 5, 6, 7, 8], Direction.BACKWARD).get_sentence())

    def test_string_elements(self):
        seq = AronsonSequence('t', [1, 2, 3], Direction.FORWARD)
        # Ensure it raises a ValueError because strings should not be allowed
        with self.assertRaises(ValueError):
            seq.set_elements(['a', 'b', 'c'], append=True)

    def test_is_hashable(self):
        seq = AronsonSequence('t', [1, 2, 3], Direction.FORWARD)
        s = set()
        # seq is hashable
        s.add(seq)
        self.assertIn(seq, s)

    def test_get_letter(self):
        seq = AronsonSequence('t', [1, 2, 3], Direction.FORWARD)
        # make sure display_letter() method works as expected
        self.assertNotEqual(seq.get_letter(), 't')

if __name__ == '__main__':
    unittest.main()
