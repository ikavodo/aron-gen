from enum import Enum
from num2words import num2words

PREFIX = " is the "
SUFFIX = " letter"
REPR_FORWARD = " in this sentence, not counting commas and spaces"
# delete "in this sentence"?
REPR_BACKWARD = "Not counting commas and spaces, backwards "


def n2w(n):
    """
    Converts a number n to its ordinal word representation.

    :param n: The number to convert.
    :return: The ordinal word representation of the number.
    """
    os = num2words(n, ordinal=True).replace(" and", "")
    return os.replace(", ", "").replace(" ", "").replace("-", "")


class ReferralType(Enum):
    """
    Enum for keeping track of where elements refer to relatively to their ordinal representation.

    BACKWARD: Refers to a previous position.
    SELF: Refers to its own position.
    FORWARD: Refers to a later position.
    """
    BACKWARD = 1
    SELF = 2
    FORWARD = 3


class Direction(Enum):
    """
    Direction by which ordinals refer to indices in the string representation of the sequence
    FORWARD: left to right
    BACKWARD: right to left
    """
    FORWARD = 1
    BACKWARD = 2


class AronsonSequence:
    """
    Represents an Aronson sequence, which is a sequence of elements where each element refers to the ordinal of a letter
    in a string that describes the letter's position in the sequence.

    """

    def __init__(self, letter: str, elements: list[int], direction: Direction):
        """
        Initializes the AronsonSequence with the given letter, elements, and direction.

        :param letter: The letter used for the sequence.
        :param elements: A list of elements in the sequence.
        :param direction: Whether the sequence is generated forward or backward
        """
        # Validate letter
        self._check_letter(letter)

        # Validate elements
        self._check_elements(elements)

        self._check_direction(direction)

        # Set attributes, save in lower_case
        self.letter = letter.lower()
        self.elements = elements
        self.direction = direction
        self.sentence_repr = self._build_string()
        self.sentence = self.sentence_repr.replace(", ", "").replace(" ", "").replace("-", "")
        self.referral_dict = self._build_referral_dict()

    @property
    def display_letter(self):
        return self.letter.upper()

    @staticmethod
    def _check_direction(direction: Direction):
        # Validate direction
        if not isinstance(direction, Direction):
            raise ValueError(f"Invalid direction value: {direction}. Must be a Direction.")

    @staticmethod
    def _check_elements(elements):
        if not elements:
            raise ValueError("elements must be non-empty")
        if not isinstance(elements, list) or not all(isinstance(i, int) and i > 0 for i in elements) or len(
                set(elements)) != len(elements):
            raise ValueError(f"Invalid elements: {elements}. Must be a list of non-repeating positive integers.")

    @staticmethod
    def _check_letter(letter):
        if not isinstance(letter, str) or len(letter) != 1 or not letter.isalpha():
            raise ValueError(f"Invalid letter: {letter!r}. Must be a single alphabetic character.")

    def _update_sentence(self):
        """
        Updates the string representation of the sequence and its sanitized form.

        This method constructs the `sentence_repr` and the `sentence` attributes.
        """
        self.sentence_repr = self._build_string()
        self.sentence = self.sentence_repr.replace(", ", "").replace(" ", "").replace("-", "")

    def _build_string(self):
        """
        Returns the human-readable string representation of the AronsonSequence.

        :return: The string representation of the sequence.
        """
        idx_ord = self.elements if self.direction == Direction.FORWARD else self.elements[::-1]
        return f"{self.letter + PREFIX}{', '.join(num2words(i, ordinal=True) for i in idx_ord)}{SUFFIX}"

    def _get_referral_type(self, idx):
        """
        Determines the referral type for a specific index in the sequence.

        :param idx: The index of the element in the sequence.
        :return: A value from `ReferralType` indicating the referral type.
        """
        target_idx = idx - 1
        rep = n2w(idx)
        if self.direction == Direction.FORWARD:
            pos = self.sentence.find(rep)
        else:
            pos = self.sentence[::-1].find(rep[::-1])

        if target_idx < pos:
            return ReferralType.BACKWARD
        elif pos <= target_idx < pos + len(rep):
            return ReferralType.SELF
        else:
            return ReferralType.FORWARD

    def _build_referral_dict(self):
        """
        Builds the referral dictionary that maps each element to its corresponding referral type.

        :return: A dictionary mapping each element to its referral type.
        """
        return {idx: self._get_referral_type(idx) for idx in self.elements}

    def _update_referral_dict(self, new_elements):
        """
        Updates the referral dictionary with new elements.

        :param new_elements: A list of new elements to add to the dictionary.
        """
        # no key duplicities- no overwrites
        self.referral_dict.update({
            idx: self._get_referral_type(idx) for idx in new_elements
        })

    def has_forward_referring(self):
        """
        Checks if there are any forward referring elements in the sequence.

        :return: True if there are forward referring elements, False otherwise.
        """
        return any(ref == ReferralType.FORWARD for ref in self.referral_dict.values())

    def _get_occurrences(self):
        """
        Returns the 1-based positions of `self.letter` in the sentence, respecting direction.

        :return: A list of positions where the letter occurs.
        """
        s = self.sentence if self.direction == Direction.FORWARD else self.sentence[::-1]
        return {i + 1 for i, char in enumerate(s) if char == self.letter}

    # Add notion of prefix_complete (up to SUFFIX)
    def is_complete(self):
        """
        Checks if the sequence is self-contained, i.e., the positions of the letter in the sentence match the elements.

        :return: True if the sequence is self-contained, False otherwise.
        """
        return self._get_occurrences() == set(self.elements)

    def is_correct(self):
        """
        Verifies if the sequence is valid by checking if all elements occur at the correct positions.

        :return: True if the sequence is valid, False otherwise.
        """
        return all(ind in self._get_occurrences() for ind in self.elements)

    # setters
    def set_elements(self, new_elements: list[int], append=False):
        """
        Setter for the elements of the sequence. Updates the sentence, sentence_repr, and referral_dict.

        :param new_elements: The new elements for the sequence.
        :param append: Whether to append or replace elements.
        """

        # check input
        self._check_elements(new_elements)

        if append:
            # Append new elements while avoiding duplicates
            self.elements.extend(elem for elem in new_elements if elem not in self.elements)
            self._update_sentence()
            self._update_referral_dict(new_elements)
        else:
            # Replace elements
            self.elements = new_elements
            self._update_sentence()
            self.referral_dict = self._build_referral_dict()

    def append_elements(self, new_elements: list[int]):
        """
        Wrapper function to append new elements to the sequence.

        :param new_elements: A list of new elements to append.
        """
        self.set_elements(new_elements, append=True)

    def set_letter(self, letter):
        """
        Sets the letter for the sequence and updates the sentence accordingly.

        :param letter: The new letter to set for the sequence.
        """
        self._check_letter(letter)
        self.letter = letter.lower()
        # need to update sentence
        self._update_sentence()

    def flip_direction(self):
        """
        Flips the direction of the sequence

        Updates the sentence accordingly.
        """
        self.direction = Direction.BACKWARD if self.direction == Direction.FORWARD else Direction.FORWARD
        # need to update sentence
        self._update_sentence()

    # getters
    def get_letter(self):
        return self.display_letter

    def get_sentence(self):
        """
        Getter for the string representation of the sequence.

        :return: The string representation of the sequence.
        """
        return self.sentence

    def get_direction(self):
        """
        Getter for the direction of the sequence

        :return: Direction.
        """
        return self.direction

    def get_elements(self):
        """
        Getter for the elements of the sequence.

        :return: The elements of the sequence.
        """
        return self.elements

    def get_referral_dict(self):
        """
        Getter for the referral dictionary.

        :return: The referral dictionary.
        """
        return self.referral_dict

    def __repr__(self):
        """
        Returns the human-readable representation of the Aronson sequence.

        :return: The human-readable string representation of the sequence.
        """
        # return upper case
        s = self.display_letter + self.sentence_repr[1:]
        return s + REPR_FORWARD if self.direction == Direction.FORWARD else REPR_BACKWARD + s

    def __eq__(self, other):
        """
        Compares two Aronson sequences for equality based on letter, elements, and direction.

        :param other: The other AronsonSequence to compare with.
        :return: True if the sequences are equal, False otherwise.
        """
        return isinstance(other, AronsonSequence) and self.elements == other.get_elements() and \
            self.display_letter == other.get_letter() and self.direction == other.get_direction()

    def copy(self):
        return AronsonSequence(
            self.letter,
            self.elements.copy(),  # avoid sharing mutable state
            self.direction
        )

    def __hash__(self):
        """
        Returns a hash value for the AronsonSequence object based on its letter, elements, and direction.

        :return: A hash value for the sequence.
        """
        return hash((tuple(self.elements), self.letter, self.direction))

    def __iter__(self):
        """
        Returns an iterator over the elements of the sequence.

        :return: An iterator for the elements.
        """
        return iter(self.elements)

    def __len__(self):
        """
        Returns the length of the Aronson sequence (i.e., the number of elements).

        :return: The length of the Aronson sequence.
        """
        return len(self.elements)

    def __getitem__(self, index: int):
        """
        Returns the element at a specified position in the Aronson sequence.

        :param index: The index position to retrieve.
        :return: The element at the specified position.
        """
        return self.elements[index]
