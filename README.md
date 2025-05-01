# Aronson Sequence Generator

A Python implementation of two classes representing self-referential sentences (AronsonSequence), and collections thereof (AronsonSet). 
The prototype which inspired the first core class `AronsonSequence` is ["Aronson's sequence"](https://oeis.org/A005224), first coined by J. K. Aronson, and quoted by D. R. Hofstadter in his book "Methamagical Themas" (1983).
The  `AronsonSet` (collections of sequences) class constitutes a generalization of Aronson's sequence to all sentences "Ω is the X, Y, Z... letter", where Ω \in Σ is a letter in the alphabet and the sentence is semantically correct (meaning X, Y, Z are ordinals which map to occurrences of the letter Ω in the sentence).  

See [this](https://ikavodo.github.io/aronson-1/) for more details.

[![Tests](https://img.shields.io/badge/tests-90%25%20coverage-green)]

## Features

### `AronsonSequence` Class
Models self-referential sentences with:
- Positional reference tracking (forward/backward/self)
- Automatic sentence generation & validation
- Element manipulation (append/swap/clear)
- Direction flipping (forward ↔ backward)
- Comprehensive correctness checks

```python
# Create and validate sequence
aronson_initial = [1, 4, 11] # first three terms in Aronson's sequence
letter = 't'
seq1 = AronsonSequence(letter, aronson_initial)
print(seq1)  # "T is the first, fourth, eleventh letter in this sentence, not counting commas and spaces"
print(seq1.is_correct())  # True
print({seq1.get_ref(elem) for elem in aronson_initial}) # {Refer.BACKWARD}, meaning all elements refer before their positions
print(seq.is_prefix_complete()) # True, as all occurrences of T up to max(aronson_initial) are accounted for
print(seq.is_complete()) # False, the first n terms of the Aronson sequence are never complete (making the series infinite)
seq1.append_elements([16]) # Next element in Aronson
seq1.is_correct() # True

aronson_reverse = [3, 4, 11]
seq2 = AronsonSequence(letter, aronson_reverse, Direction.BACKWARD) # Reverse Aronson's sequence
print(seq2)  # "Not counting commas and spaces, in this sentence backwards T is the eleventh, fourth, third letter"
print(seq2.is_correct())  # True
print(seq.is_prefix_complete()) # True
seq1.append_elements([12]) # Wrong next element
seq1.is_correct() # False

```

### `AronsonSet` Class
Manages collections of valid sequences with:

- Multiple generation strategies (pruned brute-force/fast rule-based)
- Set operations (union/intersection/difference)
- Filter operations (by element/reference/symmetry)

```python
# Generate and analyze sequences
aset1 = AronsonSet('t', Direction.BACKWARD)
empty_seq = aset1.peek() 
print(empty_seq) # "T is the letter"
seq1 = aset1.generate_aronson(3).pop() # AronsonSequence('t', [3, 4, 11], Direction.BACKWARD)
print(aset1.is_correct(seq1)) # True
aset2 = AronsonSet('t') # Forward
seq2 = aset2.generate_aronson(3).pop()
print(aset1.is_correct(seq2)) # False, sequence is incorrect in set context
print(aset2.is_correct(seq1)) # False
```

## Advanced Usage
### Hybrid Generation
```python
aset = AronsonSet('t', Direction.BACKWARD)
aset.generate_full(2) # Exhaustive search
print(len(aset)) # 67
aset.generate_fast(3, forward_generate=True)  # Optimized continuation
print(len(aset)) # 198
```

### Set Operations
```python
# Combine sequence sets
seq1 = AronsonSequence('t', [1, 4, 11])
seq2 = AronsonSequence('t', [10, 12])
set1 = AronsonSet.from_sequence(seq1)
set2 = AronsonSet.from_sequence(seq2) 

union_set = set1 | set2 
assert(union_set == AronsonSet.from_set({seq1, seq2})) # same as constructor from_set() 
intersection_set = set1 & set2 
assert(intersection_set == AronsonSet('t')) # intersection is empty forward set
difference_set = set1 - set2 
assert(difference_set == set1) # sets are complementary
```

### Filter Operations
```python
# Combine sequence sets
seq1 = AronsonSequence('t', [1, 4, 11])
aset = AronsonSet.from_sequence(seq1)
n_iters = 2
aset.generate_full(n_iters)
filtered = aset.filter_symmetric(n_iters)
new_aset.filter_elements({new_aset.max})
[print(seq) for seq in new_aset if not seq.is_empty()]
# "T is the thirty-second, thirty-third letter in this sentence, not counting commas and spaces",
# "T is the thirty-third, thirty-second letter in this sentence, not counting commas and spaces"
```

## Installation
```bash
git clone https://github.com/ikavodo/aronson-generator.git
cd aronson
pip install -r requirements.txt  # Requires num2words
```

### Testing Framework
Comprehensive test suite covering:

- Sequence validation and reference resolution
- Generation method comparisons
- Set operation correctness
- Edge case handling
- Performance benchmarks

Run tests with:
```bash
python -m unittest test_AronsonSet.py test_AronsonSequence.py
```
