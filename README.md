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
seq = AronsonSequence('t', [1,4,11], Direction.FORWARD)
print(seq.is_correct())  # True/False
print(seq)  # "T is the first, fourth, eleventh letter in this sentence..."
```

### `AronsonSet` Class
Manages collections of valid sequences with:

- Multiple generation strategies (brute-force/rule-based)
- Set operations (union/intersection/difference)
- Performance-optimized generate_fast() method

```python
# Generate and analyze sequences
aset = AronsonSet('t')
aset.generate_fast(3)  # Optimized generation
missing = brute_set - rule_set  # Set operations
```

### Installation
```bash
git clone https://github.com/ikavodo/aronson-generator.git
cd aronson-generator
pip install -r requirements.txt  # Requires num2words
```

## Advanced Usage
### Hybrid Generation
```python
aset = AronsonSet('t', Direction.BACKWARD)
aset.generate_from_rules(2, full=True)  # Exhaustive search
aset.generate_fast(3)  # Optimized continuation
```

### Set Operations
```python
# Combine sequence sets
union_set = set1 | set2
intersection_set = set1 & set2
difference_set = set1 - set2
```

## Testing Framework
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
