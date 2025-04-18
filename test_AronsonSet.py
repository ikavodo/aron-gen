import unittest
from AronsonSet import AronsonSet, GenError, VerificationError, ORD_TABLE, n2w
from AronsonSequence import AronsonSequence, Direction, Refer


class AronsonSetTests(unittest.TestCase):

    def test_ord_dict(self):
        """
        Checks that upper bound on lengths of ordinals per range is correct
        :return:
        """

        # prime generator
        def generate_primes():
            """Generate an infinite sequence of prime numbers."""
            # dictionary
            D = {}
            q = 2  # First prime number
            while True:
                if q not in D:
                    yield q
                    # add square to dictionary
                    D[q * q] = [q]
                else:
                    # already added to dictionary
                    for p in D[q]:
                        # use to update new values, we have p divides q-> (p+q) divides q
                        D.setdefault(p + q, []).append(p)
                    # don't need previous value- no new information
                    del D[q]
                q += 1

        cur = 1
        # use primes to test ordinal length more quickly
        gen = generate_primes()
        # not actually testing all. Too many primes. Find sparser distribution
        for i in range(1, 7):
            upper = 10 ** i
            limit = ORD_TABLE[i]
            while cur < upper:
                cur = next(gen)
                self.assertTrue(len(n2w(cur)) <= limit)

    # What to tests in AronsonSet?
    def test_initialization(self):
        # Test valid initialization
        for direction in [Direction.FORWARD, Direction.BACKWARD]:
            with self.subTest(direction=direction):
                aset = AronsonSet('t', direction)
                self.assertEqual(aset.display_letter, 'T')
                self.assertEqual(aset.direction, direction)
                self.assertEqual(len(aset.get_seen_seqs()), 1)  # Empty sequence
                self.assertEqual(aset.get_n_iterations(), 0)

        # Invalid letters
        for invalid_letter in ['tt', '!', '']:
            with self.subTest(letter=invalid_letter):
                with self.assertRaises(ValueError):
                    AronsonSet(invalid_letter)

        invalid_directions_list = [True, 0]
        for invalid_direction in invalid_directions_list:
            with self.subTest(direction=invalid_direction):
                with self.assertRaises(ValueError):
                    AronsonSet('t', invalid_direction)

    def test_from_sequence(self):
        # Construction from empty sequence should result in same thing as constructing a new AronsonSet instance
        empty_seq = AronsonSequence('t', [], Direction.BACKWARD)
        valid_seq = AronsonSequence('t', [3, 4, 11], Direction.BACKWARD)
        aset = AronsonSet.from_sequence(valid_seq)
        self.assertIn(valid_seq, aset.get_seen_seqs())
        self.assertIn(empty_seq, aset.get_seen_seqs())

        # Invalid sequence
        invalid_seq = AronsonSequence('t', [1, 2, 3])
        with self.assertRaises(VerificationError):
            AronsonSet.from_sequence(invalid_seq)

    # Need these for testing further constructors
    def test_copy(self):
        seq = AronsonSequence('t', [1], Direction.FORWARD)
        aset = AronsonSet.from_sequence(seq)
        aset_cpy = aset.copy()
        self.assertEqual(aset_cpy.get_iter_dict(), aset.get_iter_dict())
        self.assertEqual(aset_cpy.get_n_iterations(), aset.get_n_iterations())
        self.assertEqual(aset_cpy.get_direction(), aset.get_direction())
        self.assertEqual(aset_cpy.get_letter(), aset.get_letter())
        self.assertEqual(aset_cpy.get_seen_seqs(), aset.get_seen_seqs())
        # equivalent to
        self.assertEqual(aset_cpy, aset)
        aset = AronsonSet('t')
        # iter_dict and seen_seqs are not the same
        self.assertNotEqual(aset, aset_cpy)

    def test_eq(self):
        empty_seq = AronsonSequence('t')
        aset_empty = AronsonSet.from_sequence(empty_seq)
        # checks __eq__
        self.assertEqual(aset_empty, AronsonSet('t'))
        self.assertNotEqual(aset_empty, AronsonSet('t', Direction.BACKWARD))
        self.assertNotEqual(aset_empty, AronsonSet('g'))
        # don't have same seen sets.
        self.assertNotEqual(aset_empty, AronsonSet.from_sequence(AronsonSequence('t', [1])))
        # Now try something where seen sets are the same, but on different iterations
        fake_iter_dict = {0: {empty_seq}, 1: {empty_seq}}
        fake_aset = aset_empty.copy()
        fake_aset.set_iter_dict(fake_iter_dict)
        self.assertEqual(fake_aset.get_seen_seqs(), aset_empty.get_seen_seqs())
        self.assertNotEqual(aset_empty, fake_aset)

    def test_iter(self):
        empty_set = AronsonSet('t')
        empty_seq = AronsonSequence('t')
        for seq in empty_set:
            # only sequence in set
            self.assertEqual(empty_seq, seq)
        valid_set = AronsonSequence('t', [3, 4, 11], Direction.BACKWARD)
        # Iterating through set also gives non empty sequence
        self.assertTrue(any(seq != empty_seq for seq in valid_set))

    # Too early to try these.
    def test_or_singleton(self):
        empty_seq = AronsonSequence('t', [], Direction.BACKWARD)
        valid_seq1 = AronsonSequence('t', [3, 4, 11], Direction.BACKWARD)
        empty = AronsonSet.from_sequence(empty_seq)
        singleton1 = AronsonSet.from_sequence(valid_seq1)
        union_set1 = singleton1 | empty
        self.assertEqual(union_set1, singleton1)
        valid_seq2 = AronsonSequence('t', [3, 4], Direction.BACKWARD)
        singleton2 = AronsonSet.from_sequence(valid_seq2)
        union_set2 = singleton1 | singleton2
        # check that all sequences mapped to zeroeth interation
        self.assertTrue(
            seq in union_set2.get_iter_dict()[union_set2.get_n_iterations()] for seq in union_set2.get_seen_seqs())
        # Check
        self.assertNotEqual(union_set1, union_set2)

    def test_or_nonzero_iterations(self):
        # TODO after checking generating rules
        pass

    def test_from_set(self):
        emp_seq = AronsonSequence('t')
        # Check this works fine!
        aset = AronsonSet.from_set({emp_seq})
        self.assertEqual(aset, AronsonSet.from_sequence(emp_seq))
        # all of these are same- generate set with default empty AronsonSequence
        self.assertEqual(aset, AronsonSet.from_set())

        invalid_cases = [{AronsonSequence('t', [], Direction.BACKWARD), AronsonSequence('t')},
                         {AronsonSequence('a'), AronsonSequence('b')}]
        for case in invalid_cases:
            with self.assertRaises(ValueError):
                # Don't allow invalid cases
                AronsonSet.from_set(case)

        # Check valid case
        sets = {AronsonSequence('t', [3], Direction.BACKWARD), AronsonSequence('t', [3, 4], Direction.BACKWARD),
                AronsonSequence('t', [3, 4, 11], Direction.BACKWARD)}
        new_aset = AronsonSet.from_set(sets)
        for s in sets:
            self.assertIn(s, new_aset.get_seen_seqs())

    def test_is_correct(self):
        test_cases = [

            (AronsonSequence('t'), True),
            (AronsonSequence('t', [], Direction.BACKWARD), True),

            (AronsonSequence('t', [1, 4, 11], Direction.FORWARD), True),
            (AronsonSequence('t', [3, 4, 11], Direction.BACKWARD), True),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(AronsonSet.from_sequence(seq).is_correct(seq), expected)

        # Return incorrectness for incorrect sequences or those with mismatched directions and letters compared to
        # AronsonSet
        aset = AronsonSet('t')
        for seq in [AronsonSequence('t', [19, 100], Direction.FORWARD),
                    AronsonSequence('t', [19, 100], Direction.BACKWARD), AronsonSequence('t', [], Direction.BACKWARD),
                    AronsonSequence('b')]:
            self.assertFalse(aset.is_correct(seq))

    def test_is_complete(self):
        test_cases = [
            (AronsonSequence('l', [1, 23], Direction.FORWARD), True),
            # also true
            (AronsonSequence('l', [23, 1], Direction.FORWARD), True),
            (AronsonSequence('j', [24], Direction.BACKWARD), True),
            (AronsonSequence('l', [1], Direction.FORWARD), False),
            (AronsonSequence('l', [6], Direction.BACKWARD), False),
            (AronsonSequence('t'), False),
            (AronsonSequence('t', [], Direction.BACKWARD), False),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(AronsonSet.from_sequence(seq).is_complete(seq), expected)

        invalid_cases = [AronsonSequence('l', [1, 23], Direction.FORWARD),
                         AronsonSequence('t', [], Direction.BACKWARD)]
        aset = AronsonSet('t')
        for case in invalid_cases:
            self.assertFalse(aset.is_complete(case))



    def test_set_iter_dict(self):
        aset = AronsonSet('t')

    def test_generate_aronson(self):
        # Test standard T sequence generation
        aset = AronsonSet('t', Direction.FORWARD)
        sequences = aset.generate_aronson(3)
        expected = AronsonSequence('t', [1, 4, 11], Direction.FORWARD)
        self.assertIn(expected, sequences)

        # Test backward generation
        aset_back = AronsonSet('t', Direction.BACKWARD)
        sequences = aset_back.generate_aronson(2)
        expected = AronsonSequence('t', [3, 4], Direction.BACKWARD)
        self.assertIn(expected, sequences)

    def test_swap_operation(self):
        # Test valid swap
        original = AronsonSequence('t', [1, 4, 11], Direction.FORWARD)
        aset = AronsonSet.from_sequence(original)
        swapped_seqs = aset.swap(original)

        # Expected swap that maintains validity
        expected = AronsonSequence('t', [4, 1, 11], Direction.FORWARD)
        self.assertIn(expected, swapped_seqs)

        # Test invalid swap (would break references)
        invalid_swap = AronsonSequence('t', [11, 4, 1], Direction.FORWARD)
        self.assertNotIn(invalid_swap, swapped_seqs)

    def test_forward_generation(self):
        # Test finding next valid element
        initial = AronsonSequence('t', [1], Direction.FORWARD)
        aset = AronsonSet.from_sequence(initial)
        generated = aset.forward_generate(initial)

        # Should find position 4 as next valid element
        expected = AronsonSequence('t', [1, 4], Direction.FORWARD)
        self.assertIn(expected, generated)

    def test_singleton_generation(self):
        # Test valid singleton sequences
        aset = AronsonSet('t', Direction.FORWARD)
        singletons = aset.generate_singletons()

        valid_singletons = {
            AronsonSequence('t', [1], Direction.FORWARD),
            AronsonSequence('t', [4], Direction.FORWARD)
        }
        self.assertTrue(valid_singletons.issubset(singletons))

    def test_generate_from_rules(self):
        aset = AronsonSet('t', Direction.FORWARD)
        aset.generate_from_rules(3)

        # Should contain standard sequence after 3 iterations
        expected = AronsonSequence('t', [1, 4, 11], Direction.FORWARD)
        self.assertIn(expected, aset.get_seen_seqs())

    def test_intersection_operations(self):
        # Create two sets with overlapping sequences
        aset1 = AronsonSet('t', Direction.FORWARD)
        aset1.generate_from_rules(3)

        aset2 = AronsonSet('t', Direction.FORWARD)
        aset2.generate_from_rules(2)

        # Get intersection
        intersection = aset1.intersect_aronson_sets(aset2, 2)
        expected = AronsonSequence('t', [1, 4], Direction.FORWARD)
        self.assertIn(expected, intersection)

    def test_edge_cases(self):
        # Test empty sequence handling
        aset = AronsonSet('x', Direction.FORWARD)
        self.assertTrue(aset.get_seen_seqs().pop().is_empty())

        # Test generation failure handling
        aset = AronsonSet('q', Direction.FORWARD)
        with self.assertRaises(GenError):
            aset.generate_aronson(5)

    def test_property_methods(self):
        # Test prefix complete check
        seq = AronsonSequence('t', [1, 4], Direction.FORWARD)
        aset = AronsonSet.from_sequence(seq)
        self.assertTrue(seq.is_prefix_complete())

        # Test has_forward_referring
        forward_ref_seq = AronsonSequence('t', [19], Direction.FORWARD)
        aset.add_sequences({forward_ref_seq})
        self.assertTrue(forward_ref_seq.has_forward_referring())

    def test_brute_force_generation(self):
        aset = AronsonSet('t', Direction.FORWARD)
        aset.generate_brute_force(2)

        # Should contain all valid 2-element sequences
        valid_sequences = {
            AronsonSequence('t', [1, 4], Direction.FORWARD),
            AronsonSequence('t', [4, 1], Direction.FORWARD)
        }
        self.assertTrue(valid_sequences.issubset(aset.get_seen_seqs()))

    def test_sequence_validation(self):
        # Test valid sequence
        valid_seq = AronsonSequence('t', [1, 4, 11], Direction.FORWARD)
        aset = AronsonSet('t', Direction.FORWARD)
        self.assertTrue(aset.is_correct(valid_seq))

        # Test invalid sequence
        invalid_seq = AronsonSequence('t', [1, 2, 3], Direction.FORWARD)
        self.assertFalse(aset.is_correct(invalid_seq))


if __name__ == '__main__':
    unittest.main()
