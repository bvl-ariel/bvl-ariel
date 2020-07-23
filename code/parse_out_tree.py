import re


def parse_tree(file):
    groups = []
    depth = 0
    with open(file) as f:
        for x in f:
            m = re.search("(yes|no): +\[([0-9]+)\]:", x)
            if m is not None:
                print(x.strip())


parse_tree(r"D:\Matej\ariel\explore\cluster_targets\original.out")
