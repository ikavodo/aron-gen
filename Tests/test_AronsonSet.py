import random
import unittest
import time
from functools import reduce
from itertools import combinations

from AronsonSet import AronsonSet, GenError, VerificationError, ORD_TABLE, n2w
from AronsonSequence import AronsonSequence, Direction, Refer


class AronsonSetTests(unittest.TestCase):
    """ unit test for AronsonSet"""

    def test_ord_dict(self):
        """
        Checks that upper bound on lengths of ordinals per range is correct. Not an exact test but good enough
        :return:
        """
        # can't test all ordinals, this takes too long
        for i in range(2, 11):
            lower, upper = 10 ** (i - 2), 10 ** (i - 1)
            limit = ORD_TABLE[i]
            for _ in range(100):
                cur = random.randint(lower, upper)
                self.assertTrue(len(n2w(cur)) <= limit)

    # Constructors
    def test_initialization(self):
        # Test valid initialization
        for direction in list(Direction):
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

    def test_from_sequence(self):
        # Construction from empty sequence should result in same thing as constructing a new AronsonSet instance
        valid_seq = AronsonSequence('t', [3, 4, 11], Direction.BACKWARD)
        aset = AronsonSet.from_sequence(valid_seq)
        self.assertIn(valid_seq, aset.get_seen_seqs())
        empty_seq = AronsonSequence('t', [], Direction.BACKWARD)
        self.assertIn(empty_seq, aset.get_seen_seqs())

        # Invalid sequence
        invalid_seq = AronsonSequence('t', [1, 2, 3])
        with self.assertRaises(VerificationError):
            AronsonSet.from_sequence(invalid_seq)

    # Need these for testing further constructors
    def test_eq(self):
        empty_seq = AronsonSequence('t')
        aset_empty = AronsonSet.from_sequence(empty_seq)
        self.assertEqual(aset_empty, AronsonSet('t'))
        # different direction or letter
        self.assertNotEqual(aset_empty, AronsonSet('t', Direction.BACKWARD))
        self.assertNotEqual(aset_empty, AronsonSet('g'))
        # don't have same seen sets.
        self.assertNotEqual(aset_empty, AronsonSet.from_sequence(AronsonSequence('t', [1])))
        # different iter_dicts
        fake_aset = AronsonSet.from_sequence(empty_seq)
        fake_aset.set_iter_dict({i: {empty_seq} for i in range(2)})
        self.assertEqual(fake_aset.get_seen_seqs(), aset_empty.get_seen_seqs())
        self.assertNotEqual(aset_empty, fake_aset)

    def test_copy(self):
        seq = AronsonSequence('t', [1])
        aset = AronsonSet.from_sequence(seq)
        aset_cpy = aset.copy()
        self.assertEqual(aset_cpy, aset)
        aset = AronsonSet('t')
        # iter_dict and seen_seqs are not the same
        self.assertNotEqual(aset, aset_cpy)

    def test_iter(self):
        empty_set = AronsonSet('t')
        empty_seq = AronsonSequence('t')
        # only sequence in set
        self.assertEqual(empty_seq, empty_set.get_seen_seqs().pop())
        valid_set = AronsonSet.from_sequence(AronsonSequence('t', [3, 4, 11], Direction.BACKWARD))
        # Iterating through set also gives non-empty sequence
        self.assertTrue(any(seq != empty_seq for seq in valid_set))

    def test_or_singleton(self):
        empty_seq = AronsonSequence('t', [], Direction.BACKWARD)
        empty_set = AronsonSet.from_sequence(empty_seq)
        valid = AronsonSequence('t', [3, 4, 11], Direction.BACKWARD)
        singleton1 = AronsonSet.from_sequence(valid)
        union_set1 = singleton1 | empty_set
        # or with empty set does nothing
        self.assertEqual(union_set1, singleton1)
        valid = AronsonSequence('t', [3, 4], Direction.BACKWARD)
        singleton2 = AronsonSet.from_sequence(valid)
        union_set2 = singleton1 | singleton2
        # check that all sequences mapped to zero-eth interation
        same_iter = union_set2.get_iter_dict()[union_set2.get_n_iterations()]
        self.assertTrue(
            seq in same_iter for seq in union_set2.get_seen_seqs())
        self.assertNotEqual(union_set1, union_set2)

    def test_from_set(self):
        emp_seq = AronsonSequence('t')
        aset = AronsonSet.from_set({emp_seq})
        # all are the same
        self.assertEqual(aset, AronsonSet.from_sequence(emp_seq))
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
            # contains all sequences used for construction
            self.assertIn(s, new_aset.get_seen_seqs())

    def test_is_correct(self):
        test_cases = [
            (AronsonSequence('t'), True),
            (AronsonSequence('t', [], Direction.BACKWARD), True),
            (AronsonSequence('t', [1, 4, 11]), True),
            (AronsonSequence('t', [3, 4, 11], Direction.BACKWARD), True),
        ]

        for seq, expected in test_cases:
            with self.subTest(seq=seq):
                self.assertEqual(AronsonSet.from_sequence(seq).is_correct(seq), expected)

        # incorrect sequences or mismatched directions and letters
        aset = AronsonSet('t')
        for seq in [AronsonSequence('t', [], Direction.BACKWARD),
                    AronsonSequence('b'), AronsonSequence('t', [19, 100]),
                    AronsonSequence('t', [19, 100], Direction.BACKWARD)]:
            self.assertFalse(aset.is_correct(seq))

    def test_is_complete(self):
        test_cases = [
            (AronsonSequence('l', [1, 23], Direction.FORWARD), True),
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

        # mismatched letter or direction
        invalid_cases = [AronsonSequence('l', [1, 23], Direction.FORWARD),
                         AronsonSequence('t', [], Direction.BACKWARD)]
        aset = AronsonSet('t')
        for case in invalid_cases:
            self.assertFalse(aset.is_complete(case))

    def test_set_iter_dict(self):
        aset = AronsonSet('t')
        aset_cpy = aset.copy()
        empty_set = {AronsonSequence('t')}
        aset_cpy.set_iter_dict({})
        # does nothing
        self.assertEqual(aset, aset_cpy)

        # invalid case
        invalid_set = empty_set.copy()
        invalid_set.update({AronsonSequence('t', [], Direction.BACKWARD), AronsonSequence('j')})
        invalid_dict = {0: invalid_set}
        with self.assertRaises(ValueError):
            aset_cpy.set_iter_dict(invalid_dict)

        valid_set = empty_set.copy()
        valid_seq = AronsonSequence('t', [1])
        valid_set.add(valid_seq)
        valid_dict = {0: valid_set}
        aset_cpy.set_iter_dict(valid_dict)
        # are same
        self.assertEqual(aset_cpy, AronsonSet.from_sequence(valid_seq))

        # Check iters updated using dict with more than one iteration
        more_iters_dict = {i: valid_set for i in range(2)}
        aset_cpy.set_iter_dict(more_iters_dict)
        self.assertTrue(aset_cpy.get_n_iterations() == max(more_iters_dict.keys()))

    def test_clear(self):
        empty_set = AronsonSet('t')
        aset_cpy = empty_set.copy()
        empty_set.clear()
        # does nothing
        self.assertEqual(empty_set, aset_cpy)
        aset = AronsonSet.from_sequence(AronsonSequence('t', [1]))
        aset.clear()
        self.assertEqual(aset, empty_set)

    def test_backward_search(self):

        def get_new_seqs(aset):
            """ helper for getting rid of empty set in backward_search()"""
            new_seqs = [aset.backward_search(seq) for seq in aset.get_seen_seqs()]
            return reduce(lambda a, b: a.union(b), new_seqs)

        aset_emp = AronsonSet('t')
        empty_seq = AronsonSequence('t')
        # Does nothing
        self.assertEqual(aset_emp.backward_search(empty_seq), set())

        for elems, direction in zip([[4, 1], [4, 3]], list(Direction)):
            targ_seq = AronsonSequence('t', elems, direction)
            # Is valid
            self.assertTrue(targ_seq.is_correct())
            self.assertTrue(targ_seq.is_prefix_complete())
            aset = AronsonSet.from_sequence(AronsonSequence('t', [4], direction=direction))
            new_seqs = get_new_seqs(aset)
            self.assertEqual(new_seqs, {targ_seq})

        for direction in list(Direction):
            # forward referring
            forward_ref = AronsonSequence('t', [19], direction)
            aset = AronsonSet.from_sequence(forward_ref)
            new_seqs = get_new_seqs(aset)
            # Check for correctness when using backward_search on forward_referring sequences
            self.assertFalse(all([seq.is_correct() for seq in new_seqs]))

        # prefix complete
        for elems, direction in zip([[1], [3]], list(Direction)):
            new_aset = AronsonSet.from_sequence(AronsonSequence('t', elems, direction))
            self.assertTrue(all(seq.is_prefix_complete()) for seq in new_aset.get_seen_seqs())
            new_seqs = get_new_seqs(new_aset)
            # Returns nothing
            self.assertEqual(new_seqs, set())

    def test_generate_aronson(self):
        # Simple case of generate_backwards()
        for direction, elems in zip(list(Direction),
                                    [[1, 4, 11], [3, 4, 11]]):
            aset = AronsonSet('t', direction)
            for n in range(len(elems)):
                # singleton
                singleton_set = aset.generate_aronson(n)
                expected = AronsonSequence('t', elems[:n], direction)
                self.assertIn(expected, singleton_set)
                seq = singleton_set.pop()
                self.assertTrue(seq.is_prefix_complete())

        # Letter doesn't allow generating more.
        for direction in set(Direction):
            aset = AronsonSet('a', direction)
            with self.assertRaises(GenError):
                aset.generate_aronson(2)

    def test_backward_generate(self):

        for seq, direction in zip([[10], [19]], list(Direction)):
            # can fail for either forward-referring or self-referring, catch exception
            aset = AronsonSet.from_sequence(AronsonSequence('t', seq, direction))
            with self.assertRaises(GenError):
                [aset.backward_generate(1, seq) for seq in aset.get_seen_seqs()]

        # try generating more than 1 new element
        for elems in [[1, 11], [10, 12]]:
            seq = AronsonSequence('t', elems)
            aset = AronsonSet.from_sequence(seq)
            new_seqs = [aset.backward_generate(2, seq) for seq in aset.get_seen_seqs() if not seq.is_empty()]
            new_seq = reduce(lambda a, b: a.union(b), new_seqs).pop()
            # last element of the two series is the same
            self.assertEqual(new_seq[-1], 24)

    def test_flip_direction(self):
        """ only works for empty sets"""
        empty_set = AronsonSet('t')
        empty_set.flip_direction()
        self.assertEqual(empty_set.get_direction(), Direction.BACKWARD)
        # check seen sequence direction is also changed
        seq = empty_set.get_seen_seqs().copy().pop()
        self.assertEqual(seq.get_direction(), Direction.BACKWARD)
        aset = AronsonSet.from_sequence(AronsonSequence('t', [1]))
        # can't flip anymore
        with self.assertRaises(ValueError):
            aset.flip_direction()

    def test_swap_operation(self):
        empty_set = AronsonSet('t')
        for s in [empty_set.get_seen_seqs(), empty_set.generate_aronson(1)]:
            seq = s.pop()
            # swapping empty or singleton sequences does nothing
            self.assertEqual(empty_set.swap(seq), set())

        all_seqs = [([1, 11, 4], [4, 1, 11], [4, 11, 1]), ([3, 11, 4], [4, 3, 11], [4, 11, 3])]
        for seqs, direction in zip(all_seqs, list(Direction)):
            swapped1, swapped2, out = seqs
            empty_set = AronsonSet('t', direction)
            seq = empty_set.generate_aronson(3).pop()
            swapped_seqs = empty_set.swap(seq)
            self.assertIn(AronsonSequence('t', swapped1, direction), swapped_seqs)
            # do another swap
            second_gen_swap = empty_set.swap(AronsonSequence('t', swapped2, direction))
            self.assertIn(AronsonSequence('t', out, direction), second_gen_swap)

        # Case in which pair can't be swapped
        empty_set.flip_direction()
        seq = AronsonSequence('t', [10, 12])
        # is empty
        swapped_seqs = empty_set.swap(seq)
        self.assertEqual(swapped_seqs, set())

    def test_subset_operation(self):
        # Test finds all swaps
        empty_set = AronsonSet('t')
        empty_seq = empty_set.get_seen_seqs().pop()
        # swapping empty sequence elements does nothing
        self.assertEqual(empty_set.subset(empty_seq), set())

        for direction in list(Direction):
            empty_set = AronsonSet('t', direction)
            seq = empty_set.generate_aronson(3).pop()
            subbed_seqs = empty_set.subset(seq)
            self.assertIn(AronsonSequence('t', seq[::2], direction), subbed_seqs)

        elems = [1, 4]
        seq = AronsonSequence('t', elems)
        subbed_seqs = AronsonSet('t').subset(seq)
        for e in elems:
            self.assertIn(AronsonSequence('t', [e]), subbed_seqs)

    def test_singleton_generation(self):
        # Test valid singleton sequences
        all_elems = [[1, 4, 10, 12, 19, 21, 22], [3, 4, 8, 19, 23, 24]]
        for elems, direction in zip(all_elems, list(Direction)):
            aset = AronsonSet('t', direction)
            singletons = aset.generate_singletons()
            # All valid forward singletons
            valid = {AronsonSequence('t', [p], direction) for p in elems}
            self.assertTrue(valid.issubset(singletons))

    def test_forward_generation(self):
        # Test finding next valid element
        all_elems = [[11, 12, 15, 17, 25, 26, 27], [11, 13, 17, 20, 25, 27]]
        for elems, direction in zip(all_elems, list(Direction)):
            aset = AronsonSet('t', direction)
            seq = aset.generate_aronson(1).pop()
            continuations = aset.forward_generate(seq)
            # valid forward continuations
            first = 1 if direction == Direction.FORWARD else 3
            valid_continuations = {AronsonSequence('t', [first, p], direction) for p in elems}
            self.assertTrue(valid_continuations.issubset(continuations))

    def test_forward_fix(self):
        # doesn't find AronsonSequence('t', [28,17])
        all_elems = [[AronsonSequence('t', [25, 1]), AronsonSequence('t', [26, 1]),
                      AronsonSequence('t', [26, 4]), AronsonSequence('t', [29, 11])],
                     [AronsonSequence('t', [20, 3], Direction.BACKWARD),
                      AronsonSequence('t', [26, 4], Direction.BACKWARD),
                      AronsonSequence('t', [25, 3], Direction.BACKWARD),
                      AronsonSequence('t', [26, 8], Direction.BACKWARD),
                      ]]
        for elems, direction in zip(all_elems, list(Direction)):
            aset = AronsonSet('t', direction)
            seq = AronsonSequence('t', [19], direction)
            self.assertTrue(seq.has_forward_ref())
            check_seqs = aset.forward_fix(seq)
            for s in elems:
                self.assertTrue(aset.is_correct(s))
                self.assertIn(s, check_seqs)

    def check_generation_method(self, method_name):
        all_elems = [([1, 4, 10, 12, 19, 21, 22], [11, 12, 15, 17, 25, 26, 27]),
                     ([3, 4, 8, 19, 23, 24], [11, 13, 17, 20, 25, 27])]

        for elems, direction in zip(all_elems, list(Direction)):
            singletons, continuations = elems
            aset = AronsonSet('t', direction)

            # Dynamically call the method with depth=2
            getattr(aset, method_name)(2)

            first = 1 if direction == Direction.FORWARD else 3
            filtered = {
                seq for seq in aset.get_seen_seqs()
                if not seq.is_empty() and (len(seq) == 1 or seq[0] == first)
            }

            valid_singletons = {AronsonSequence('t', [p], direction) for p in singletons}
            valid_continuations = {AronsonSequence('t', [first, p], direction) for p in continuations}

            for valid in [valid_singletons, valid_continuations]:
                self.assertTrue(valid.issubset(filtered))

    # This is just for getting ground truths for later
    def test_brute_force_generation(self):
        self.check_generation_method("generate_brute_force")

    def test_generate_from_rules(self):
        self.check_generation_method("generate_from_rules")

    def test_generate_fast(self):
        # Test singleton generation in first iteration
        valid_singletons = {
            Direction.FORWARD: {1, 4, 10, 12, 19, 21, 22},
            Direction.BACKWARD: {3, 4, 8, 19, 23, 24}
        }

        for direction in Direction:
            aset = AronsonSet('t', direction)
            aset.generate_fast(1)

            # Verify first iteration contains valid singletons
            for seq in valid_singletons[direction]:
                aseq = AronsonSequence('t', [seq], direction)
                self.assertTrue(aseq.is_correct())
                self.assertIn(aseq, aset[1])

        # Test swap/subset operations in subsequent iterations
        test_cases = [
            ([1, 4], [4, 1]),  # Simple swap
            # This is incorrect?
            ([10, 12], [12, 10]),
            ([3, 4], [4, 3])  # Backward direction
        ]

        for initial, swapped in test_cases:
            for direction in Direction:
                aset = AronsonSet('t', direction)
                aset.generate_fast(2)
                for elems in [initial, swapped]:
                    # Check original and swapped versions exist in second iteration
                    seq = AronsonSequence('t', elems, direction)
                    if seq.is_correct():
                        self.assertIn(seq, aset)

        # Test subset operations
        aset = AronsonSet('t')
        aset.generate_fast(3)

        for seq in aset:
            if len(seq) > 1:
                # Verify at least one subset exists in next iteration
                subsets = {AronsonSequence('t', list(s), seq.direction)
                           for s in combinations(seq, len(seq) - 1)}
                self.assertTrue(any(s in aset for s in subsets if s.is_correct()))

        # Test convergence detection
        with self.assertRaises(GenError):
            aset = AronsonSet('r')
            aset.generate_fast(5)  # Should converge before 5 iterations

    def test_generation_speed(self):
        """Verify generate_fast() is faster than generate_from_rules() for equivalent iterations"""
        iterations = 3
        speedups = []

        for direction in Direction:
            # Time generate_from_rules
            rules_set = AronsonSet('t', direction)
            start_rules = time.perf_counter()
            rules_set.generate_from_rules(iterations, full=False)
            time_rules = time.perf_counter() - start_rules

            # Time generate_fast
            fast_set = AronsonSet('t', direction)
            start_fast = time.perf_counter()
            fast_set.generate_fast(iterations)
            time_fast = time.perf_counter() - start_fast

            # Calculate speedup factor
            speedups.append(time_rules / max(time_fast, 1e-9))  # Avoid division by zero

            # Verify fast method contains all key sequences from rules method
            for s in fast_set:
                # all correct
                self.assertTrue(s in rules_set)

        # Require minimum 2x speedup in at least one direction
        self.assertTrue(max(speedups) >= 2.0,
                        f"Insufficient speedup: {speedups}")

    def test_sub(self):
        aset_empty = AronsonSet('t')
        singletons = aset_empty.generate_singletons()
        aset_singletons = AronsonSet.from_set(singletons)
        self.assertEqual(aset_singletons - aset_singletons, aset_empty)
        diff = aset_singletons - aset_empty
        # should include the empty set!
        self.assertEqual(diff, aset_singletons)

    def test_missing_sentences(self):
        aset_brute = AronsonSet('t')
        aset_brute.generate_brute_force(2)
        aset_rules = AronsonSet('t')
        aset_rules.generate_from_rules(2, full=True)
        missing = aset_brute - aset_rules
        # Works
        self.assertEqual(missing, AronsonSet('t'))

    def test_generate_from_rules_harder(self):
        # This is too hard.
        # seqs_per_iter = [[1, 8, 73, 955], [1, 7, 67, 771]]
        seqs_per_iter = [[1, 8, 73], [1, 7, 67]]

        for direction, n_seqs in zip(list(Direction), seqs_per_iter):
            for i, n in enumerate(n_seqs):
                aset = AronsonSet('t', direction)
                aset.generate_from_rules(i, full=False)
                self.assertTrue(all(aset.is_correct(s) for s in aset.get_seen_seqs()))
                # doesn't work for 3!
                self.assertEqual(len(aset), n)

    def test_and(self):
        # check same
        aset = AronsonSet('t')
        emp_set = aset.copy()
        aset.generate_from_rules(1)
        self.assertEqual(aset & emp_set, emp_set)
        aset_back = AronsonSet('t', Direction.BACKWARD)
        intersect = (aset.__and__(aset_back, 1, full=True)).get_seen_seqs()
        for seq in {AronsonSequence('t'), AronsonSequence('t', [4]), AronsonSequence('t', [19])}:
            self.assertIn(seq, intersect)

    def test_iand(self):
        # Test in-place intersection
        aset1 = AronsonSet('t')
        aset2 = aset1.copy()
        aset1.generate_from_rules(1)
        aset2.generate_from_rules(1)

        # Intersection with self should be identity
        aset1 &= aset2
        self.assertEqual(aset1, aset2)

        # Intersection with empty set
        empty_set = AronsonSet('t')
        aset1 &= empty_set
        self.assertEqual(aset1, empty_set)

    def test_ior(self):
        # Test in-place union
        empty_set = AronsonSet('t', Direction.BACKWARD)
        valid_seq = AronsonSequence('t', [3, 4, 11], Direction.BACKWARD)
        valid_set = AronsonSet.from_sequence(valid_seq)

        # Union with empty set
        empty_set |= valid_set
        self.assertEqual(empty_set, valid_set)

        # Union with non-empty set
        another_seq = AronsonSequence('t', [19], Direction.BACKWARD)
        both_set = AronsonSet.from_set({valid_seq, another_seq})
        another_set = AronsonSet.from_sequence(another_seq)
        empty_set |= another_set
        self.assertEqual(empty_set, both_set)

    def test_isub(self):
        # Test in-place subtraction
        aset = AronsonSet('t')
        seq1 = AronsonSequence('t', [4])
        seq2 = AronsonSequence('t', [19])
        aset |= AronsonSet.from_set({seq1, seq2})

        # Subtract identical set
        aset -= AronsonSet.from_set({seq1, seq2})
        self.assertEqual(aset, AronsonSet('t'))

        # Subtract subset with n=1
        aset |= AronsonSet.from_set({seq1, seq2})
        subset = AronsonSet.from_set({seq1})
        aset -= subset
        self.assertEqual(aset, AronsonSet.from_sequence(seq2))

    def test_filter_elems(self):
        all_elems = [[11, 12, 15, 17, 25, 26, 27], [11, 13, 17, 20, 25, 27]]
        for elems, direction in zip(all_elems, list(Direction)):
            aset = AronsonSet('t', direction)
            aset.generate_from_rules(2)
            first = 1 if direction == Direction.FORWARD else 3
            valid_continuations = {AronsonSequence('t', [first, p], direction) for p in elems}
            filtered = aset.filter_elems({first})
            for continuation in valid_continuations:
                self.assertIn(continuation, filtered)

    def test_filter_refs(self):
        all_elems = [({1, 4}, {10, 12}, {19, 21, 22}), ({3, 4}, {8}, {19, 23, 24})]
        for elems, direction in zip(all_elems, list(Direction)):
            aset = AronsonSet('t', direction)
            aset.generate_from_rules(1)
            for elem, ref in zip(elems, list(Refer)):
                filtered = aset.filter_refs({ref})
                for e in elem:
                    self.assertIn(AronsonSequence('t', [e], direction), filtered)

    def test_getitem(self):
        aset = AronsonSet('t')
        aset.generate_from_rules(1)
        self.assertTrue(all(len(seq) == i for seq in aset[i]) for i in range(2))

    def test_contains(self):
        aset = AronsonSet('t')
        self.assertTrue(AronsonSequence('t') in aset)
        aset.generate_from_rules(1)
        singleton_elems = [1, 4, 10, 12, 19, 21, 22]
        for elem in singleton_elems:
            self.assertTrue(AronsonSequence('t', [elem]) in aset)

    def test_get_letter(self):
        aset = AronsonSet('t')
        # make sure display_letter() method works as expected
        self.assertNotEqual(aset.get_letter(), 't')

    def test_len(self):
        aset = AronsonSet.from_sequence(AronsonSequence('t', [1]))
        self.assertEqual(len(aset), 2)
        aset.clear()
        self.assertEqual(len(aset), 1)

    def test_hash(self):
        aset = AronsonSet('t')
        for other in {AronsonSet('t', Direction.BACKWARD), AronsonSet.from_sequence(AronsonSequence('t', [1])),
                      AronsonSet('d')}:
            self.assertEqual(hash(aset) == hash(other), aset == other)

    def test_is_hashable(self):
        aset = AronsonSequence('t')
        s = set()
        # set is hashable
        s.add(aset)
        self.assertIn(aset, s)


if __name__ == '__main__':
    unittest.main()
