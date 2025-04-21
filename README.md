# Aronson Sequence Generator

A Python implementation of self-referential sentences inspired by Douglas Hofstadter's "Aronson sequence", featuring combinatorial generation and validation of linguistic puzzles. Built around two core classes: `AronsonSequence` (individual sequences) and `AronsonSet` (collections of sequences).
See https://ikavodo.github.io/aronson-1/ for more details

## Key Components

### `AronsonSequence` Class
Models self-referential sentences like _"T is the first, fourth, eleventh... letter..."_ with:

- **Core Functionality**
  ```python
  # Create sequence for letter 'T' with elements [1,4,11]
  seq = AronsonSequence('t', [1,4,11], Direction.FORWARD)
  
  # Validate self-referential integrity
  print(seq.is_correct())  # True/False
  
  # Get human-readable format
  print(seq)  # "T is the first, fourth, eleventh letter in this sentence..."
Features

Positional reference tracking (forward/backward/self)

Sentence generation & validation

Element manipulation (append/swap/clear)

Direction flipping (forward ↔ backward)

AronsonSet Class
Manages collections of valid sequences with:

Generation Methods

```python
aset = AronsonSet('t')

# Brute-force generation (small n)
aset.generate_brute_force(2)

# Rule-based generation (larger n)
aset.generate_from_rules(3)
```

Installation
Clone repository:

```bash
git clone https://github.com/ikavodo/aronson-generator.git
cd aronson-generator
```
Install dependencies:

```bash
pip install -r requirements.txt  # Requires num2words
```
Advanced Usage
```python
# Hybrid generation strategies
aset = AronsonSet('t', Direction.BACKWARD)
aset.generate_from_rules(2, expensive=True)
