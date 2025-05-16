import argparse
import os

from AronsonSet import AronsonSet, Direction

def main(n):
    os.makedirs('../data', exist_ok=True)

    directions = [
        (Direction.FORWARD, 'forward'),
        (Direction.BACKWARD, 'backward')
    ]

    # Data structures to hold elements and statistics
    elements = {'forward': [], 'backward': []}
    stats = {'forward': {}, 'backward': {}}

    for direction, dir_key in directions:
        # Initialize AronsonSet for the direction
        aset = AronsonSet('t', direction)
        try:
            aset.generate_full(n)
        except Exception as e:
            print(f"Warning: Generation stopped due to {e}")

        # Collect all non-empty sequences up to n elements, sorted lex, write length etc...y
        dir_elements = []
        for iter_num in range(n + 1):
            sorted_seqs = sorted([list(seq) for seq in aset[iter_num]])
            dir_elements.extend(sorted_seqs)
            elements[dir_key] = dir_elements
            stats[dir_key][iter_num] = len(aset[iter_num]) + (0 if not iter_num else stats[dir_key][iter_num-1])

    with open('../data/ground_truth_seqs.py', 'w') as f:
        f.write("# Elements corresponding to all Aronson sequences of length up to n\n")
        f.write(f"forward_elems = [\n")
        for elem in elements['forward']:
            f.write(f"    {elem},\n")
        f.write("]\n\n")
        f.write(f"backward_elems = [\n")
        for elem in elements['backward']:
            f.write(f"    {elem},\n")
        f.write("]\n")

    # Write ground_truth_stats.py
    with open('../data/ground_truth_stats.py', 'w') as f:
        f.write("# Ground truth for number of sets per iteration:\n")
        for dir_key in ['forward', 'backward']:
            f.write(f"# Direction.{dir_key.upper()}\n")
            for i in range(n + 1):
                count = stats[dir_key][i]
                f.write(f"# n = {i}: {count}\n")
            f.write("\n")
        f.write(f"forward_stats = {stats['forward']}\n")
        f.write(f"backward_stats = {stats['backward']}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Aronson sequence data files.')
    parser.add_argument('n', type=int, help='Number of iterations to generate')
    args = parser.parse_args()
    main(args.n)