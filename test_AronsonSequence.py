import unittest
from num2words import num2words
from AronsonSequence import n2w, AronsonSequence, STR_START, STR_END, ReferralType  # Adjust your_module as needed


# unit test for AronsonSequence
class AronsonSequenceTests(unittest.TestCase):

    def test_construction(self):
        seq = AronsonSequence('T', [1, 2, 3], True)
        # check is converted to lower_case
        self.assertEqual(seq.letter, 't')
        self.assertEqual(seq.elements, [1, 2, 3])
        self.assertTrue(seq.forward)

        # Invalid letters
        for invalid_letter in ['tt', '!', '']:
            with self.subTest(letter=invalid_letter):
                with self.assertRaises(ValueError):
                    AronsonSequence(invalid_letter, [1], True)

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
                    AronsonSequence('t', invalid_elements, True)

    def test_repr(self):
        # test that human representation is correct
        seq = AronsonSequence('t', [25], True)
        self.assertIn(STR_START + num2words(25, ordinal=True) + STR_END, repr(seq))
        self.assertEqual(str(seq), repr(seq))
        seq = AronsonSequence('t', [100, 101], True)
        for elem in seq.get_elements():
            self.assertIn(num2words(elem, ordinal=True), str(seq))

    def test_sentence(self):
        # test that the internal sentence is correct, use somehting with hyphen
        elem = 25
        suffix = STR_START.replace(" ", "") + n2w(elem) + STR_END.replace(" ", "")
        inds = [3, 4]
        test_sequences = [
            (AronsonSequence('t', [elem], True), suffix),
            (AronsonSequence('t', inds, False), ''.join([n2w(i) for i in inds[::-1]]))
        ]

        for seq, part in test_sequences:
            with self.subTest(seq=seq):
                self.assertIn(part, seq.get_sentence())

    def test_has_forward_refer(self):
        # check that has_forward_referring() method works correctly
        test_cases = [
            # backwards referring in both directions
            (AronsonSequence('t', [1, 4, 11], True), False),
            (AronsonSequence('t', [3, 4], False), False),

            # self referring in both directions
            (AronsonSequence('t', [10, 12], True), False),
            (AronsonSequence('t', [8, 14], False), False),

            # forward referring in both directions
            (AronsonSequence('t', [19, 100], True), True),
            (AronsonSequence('t', [19, 100], False), True),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.has_forward_referring(), expected)

    def test_all_refers(self):
        # check that refer_dictionaries are correct
        test_cases = [
            # all referring types- BACKWARD, SELF, BACKWARD, SELF, FORWARD
            (AronsonSequence('t', [1, 12, 17, 30, 100], True), True),
            (AronsonSequence('t', [3, 13, 21, 34, 100], False), True)
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                ref_vals = set(seq.get_referral_dict().values())
                self.assertEqual(ref_vals, set(ReferralType))

    def test_is_correct(self):
        test_cases = [
            # all of these are correct sentences

            # backward referring
            (AronsonSequence('t', [1, 4, 11], True), True),
            (AronsonSequence('t', [3, 4, 11], False), True),

            # self referring
            (AronsonSequence('t', [10, 12], True), True),
            (AronsonSequence('t', [8, 14], False), True),

            # forward referring
            (AronsonSequence('t', [19], True), True),
            (AronsonSequence('t', [19], False), True),

            # all referring
            (AronsonSequence('t', [1, 12, 17, 30], True), True),
            (AronsonSequence('t', [3, 13, 21, 34], False), True),

            # false
            (AronsonSequence('t', [19, 100], True), False),
            (AronsonSequence('t', [19, 100], False), False),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.is_correct(), expected)

    def test_self_contained(self):
        test_cases = [
            # all of these are correct sentences
            (AronsonSequence('l', [1, 23], True), True),
            (AronsonSequence('l', [1], True), False),
            (AronsonSequence('j', [24], False), True),
            (AronsonSequence('l', [6], False), False)
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(seq.is_self_contained(), expected)

    # I am currently here. Do I need an append_elements() method?
    def test_set_elements_append(self):
        for append_flag in [False, True]:
            seq = AronsonSequence('t', [1, 2], True)
            seq_repr = str(seq)
            prev_elements = seq.get_elements().copy()

            # Set elements with the given append flag
            seq.set_elements([3, 4], append=append_flag)

            if append_flag:
                # When append=True, ensure the original elements are retained
                for elem in prev_elements:
                    self.assertIn(elem, seq.get_elements())
                # check string represntations work as expected
                self.assertIn(seq_repr[:-len(STR_END)], str(seq))
            else:
                # When append=False, the previous elements should be removed
                for elem in prev_elements:
                    self.assertNotIn(elem, seq.get_elements())
                self.assertNotIn(seq_repr[:-len(STR_END)], str(seq))

    def test_set_elements_duplicates(self):
        # Test appending only duplicate elements
        seq = AronsonSequence('t', [1, 2], True)
        prev_elements = seq.get_elements().copy()
        seq.set_elements([1, 2], append=True)
        # Ensure no new elements are added
        self.assertEqual(seq.get_elements(), prev_elements)
        dups = [5, 1, 2, 6]
        # Test appending new elements with some duplicates
        seq.set_elements(dups, append=True)
        # Ensure original elements (1, 2) remain and new elements (5, 6) are added in correct order
        self.assertEqual(sorted(dups), seq.get_elements())

    def set_elements_empty(self):
        seq = AronsonSequence('t', [1, 2], True)

        # Test appending with an empty list
        seq.set_elements([], append=True)
        # Ensure the sequence remains unchanged as we appended an empty list
        self.assertEqual(seq.get_elements(), [1, 2])

        # Test setting elements with an empty list (not appending)
        # Make sure throws ValueError
        with self.assertRaises(ValueError):
            seq.set_elements([], append=False)

    # === Other Method Tests ===

    def test_iter_and_getitem(self):
        seq = AronsonSequence('t', [5, 6, 7], True)
        self.assertEqual(list(seq), [5, 6, 7])
        self.assertEqual(seq[1], 6)

    def test_len(self):
        seq = AronsonSequence('x', [10, 20], True)
        self.assertEqual(len(seq), 2)

    def test_eq_and_hash(self):
        a = AronsonSequence('z', [1, 2], True)
        b = AronsonSequence('z', [1, 2], True)
        c = AronsonSequence('z', [1, 2], False)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(hash(a), hash(b))

    def test_large_sequence(self):
        # Create a very large sequence
        seq = AronsonSequence('t', list(range(1, 1001)), True)

        # Test some basic properties
        self.assertEqual(len(seq), 1000)
        self.assertEqual(seq[999], 1000)
        self.assertEqual(seq[0], 1)

    def test_flip_direction(self):
        seq = AronsonSequence('t', [1, 2, 3], True)
        seq.set_elements([4, 5, 6], append=True)

        # First set to forward direction
        seq.flip_direction()
        seq.set_elements([7, 8], append=True)

        # Now check if the order of elements is consistent
        self.assertEqual(seq.get_sentence(), AronsonSequence('t', [1, 2, 3, 4, 5, 6, 7, 8], False).get_sentence())

    def test_string_elements(self):
        seq = AronsonSequence('t', [1, 2, 3], True)
        # Ensure it raises a ValueError because strings should not be allowed
        with self.assertRaises(ValueError):
            seq.set_elements(['a', 'b', 'c'], append=True)


if __name__ == '__main__':
    unittest.main()
