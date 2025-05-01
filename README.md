# Aronson Sequence Generator

A Python implementation of two classes representing self-referential sentences (AronsonSequence), and collections thereof (AronsonSet). 
The prototype which inspired the first core class `AronsonSequence` is ["Aronson's sequence"](https://oeis.org/A005224), first coined by J. K. Aronson, and quoted by D. R. Hofstadter in his book "Methamagical Themas" (1983).
The  `AronsonSet` (collections of sequences) class constitutes a generalization of Aronson's sequence to all sentences "Ω is the X, Y, Z... letter", where Ω \in Σ is a letter in the alphabet and the sentence is semantically correct (meaning X, Y, Z are ordinals which map to occurrences of the letter Ω in the sentence).  

See [this](https://ikavodo.github.io/aronson-1/) for more details.

[![Tests](https://img.shields.io/badge/tests-90%25%20coverage-green)](https://github.com/ikavodo/aronson-generator/actions)

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
print(seq1)  # "T is the first, fourth, eleventh letter in this sentence..."
print(seq1.is_correct())  # True
print({seq1.get_ref(elem) for elem in aronson_initial}) # {Refer.BACKWARD}, meaning all elements refer before their positions
print(seq.is_prefix_complete()) # True, as all occurrences of T up to max(aronson_initial) are accounted for
print(seq.is_complete()) # False, the first n terms of the Aronson sequence are never complete (making the series infinite)

seq2 = AronsonSequence(letter, aronson_initial[::-1])
print(seq.is_permutation(seq2) # True
print(seq2.is_correct())  # False, the eleventh index is not a T 
print({seq1.get_refer_val(elem) for elem in aronson_initial}) # {Refer.BACKWARD, Refer.SELF}, first element refers an index within its own ordinal representation
print(seq.is_prefix_complete()) # False

seq1.append_elements([16]) # Next element in Aronson
seq1.is_correct() # True
```

### `AronsonSet` Class
Manages collections of valid sequences with:

- Multiple generation strategies (brute-force/rule-based)
- Set operations (union/intersection/difference)
- Performance-optimized generate_fast() method

```python
# Generate and analyze sequences
aset = AronsonSet('t', Direction.BACKWARD)
seq1 = AronsonSequence('t', [1, 4, 11])
seq2 = aset.generate_aronson(3) # generates AronsonSequence('t', [3, 4, 11], Direction.BACKWARD)
print(aset.is_correct(seq2)) # True
print(aset.is_correct(seq1)) # False, sequence is incorrect in set context
```

## Advanced Usage
### Hybrid Generation
```python
aset = AronsonSet('t', Direction.BACKWARD)
aset.generate_full(2) # Exhaustive search
print(len(aset)) # 67
aset.generate_fast(3)  # Optimized continuation
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
