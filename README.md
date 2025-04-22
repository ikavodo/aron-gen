# Aronson Sequence Generator

A Python implementation of self-referential sentences inspired by the ["Aronson sequence"](https://oeis.org/A005224), featuring combinatorial generation and validation of linguistic puzzles. Built around two core classes: `AronsonSequence` (individual sequences) and `AronsonSet` (collections of sequences).

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
seq1 = AronsonSequence('t', [1,4,11])
print(seq1.is_correct())  # True
print(seq1)  # "T is the first, fourth, eleventh letter in this sentence..."
```

### `AronsonSet` Class
Manages collections of valid sequences with:

- Multiple generation strategies (brute-force/rule-based)
- Set operations (union/intersection/difference)
- Performance-optimized generate_fast() method

```python
# Generate and analyze sequences
aset = AronsonSet('t', Direction.BACKWARD)
seq2 = aset.generate_aronson(3) # generates AronsonSequence('t', [3,4,11], Direction.BACKWARD)
print(aset.is_correct(seq2)) # True
print(aset.is_correct(seq1)) # False, sequence is incorrect in set context
```

## Advanced Usage
### Hybrid Generation
```python
aset.generate_from_rules(2, full=True) # Exhaustive search
print(len(aset)) # 67
aset.generate_fast(3)  # Optimized continuation
print(len(aset)) # 198
```

### Set Operations
```python
# Combine sequence sets
set1 = AronsonSet.from_sequence(seq1) # same as 
set2 = AronsonSet.from_sequence(seq2) 
union_set = set1 | set2 # same as AronsonSet.from_set({seq1, seq2})
intersection_set = set1 & set2 # returns empty forward set, same as AronsonSet('t') 
difference_set = set1 - set2 # equals set1 because sets are complementary
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
