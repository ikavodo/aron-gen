# Aronson Sequence Generator

A Python implementation of self-referential sentences inspired by Douglas Hofstadter's "Aronson sequence", featuring combinatorial generation and validation of linguistic puzzles. Built around two core classes: `AronsonSequence` (individual sequences) and `AronsonSet` (collections of sequences).

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
