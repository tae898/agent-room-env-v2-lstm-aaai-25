"""Utility functions"""

import json
import os
import random
import subprocess
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml


def sample_max_value_key(
    prob_dict: dict[Any, float], keys_to_exclude: list = None
) -> Any:
    """Sample the key with the maximum value.

    Args:
        prob_dict: dict of probabilities
        keys_to_exclude: keys to exclude

    """
    if keys_to_exclude is not None:
        prob_dict = {k: v for k, v in prob_dict.items() if k not in keys_to_exclude}
    else:
        prob_dict = prob_dict
    max_key = max(prob_dict, key=prob_dict.get)

    return max_key


def seed_everything(seed: int) -> None:
    """Seed every randomness to seed"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_lines(fname: str) -> list:
    """Read lines from a text file.

    There is some path magic going on here. This is to account for both the production
    and development mode. Don't use this function for a general purpose.

    """
    if fname.startswith("/"):
        fullpath = fname
    else:
        fullpath = os.path.join(os.path.dirname(__file__), fname)

    with open(fullpath, "r") as stream:
        names = stream.readlines()
    names = [line.strip() for line in names]

    return names


def read_json(fname: str) -> None:
    """Read json"""
    with open(fname, "r") as stream:
        return json.load(stream)


def write_json(content: dict, fname: str) -> None:
    """Write json"""
    with open(fname, "w") as stream:
        json.dump(content, stream, indent=4, sort_keys=False)


def read_json_prod(fname: str) -> None:
    """Read json.

    There is some path magic going on here. This is to account for both the production
    and development mode. Don't use this function for a general purpose.

    """
    fullpath = os.path.join(os.path.dirname(__file__), fname)

    with open(fullpath, "r") as stream:
        return json.load(stream)


def write_json_prod(content: dict, fname: str) -> None:
    """Write json.

    There is some path magic going on here. This is to account for both the production
    and development mode. Don't use this function for a general purpose.

    """
    fullpath = os.path.join(os.path.dirname(__file__), fname)

    with open(fullpath, "w") as stream:
        json.dump(content, stream, indent=4, sort_keys=False)


def read_yaml(fname: str) -> None:
    """Read yaml.

    There is some path magic going on here. This is to account for both the production
    and development mode. Don't use this function for a general purpose.

    """
    if fname.startswith("/"):
        fullpath = fname
    else:
        fullpath = os.path.join(os.path.dirname(__file__), fname)
    with open(fullpath, "r") as stream:
        return yaml.safe_load(stream)


def write_yaml(content: dict, fname: str) -> None:
    """write yaml."""
    with open(fname, "w") as stream:
        yaml.dump(content, stream, indent=2, sort_keys=False)


def read_data(data_path: str) -> dict:
    """Read train, val, test spilts.

    Args:
        data_path: path to data.

    Returns:
        data: {'train': list of training obs,
            'val': list of val obs,
            'test': list of test obs}

    """
    data = read_json(data_path)

    return data


def argmax(iterable) -> int:
    """argmax"""
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def split_by_possessive(name_entity: str) -> tuple[str, str]:
    """Separate name and entity from the given string.

    Args:
        name_entity: e.g., "Bob's laptop"

    Returns:
        name: e.g., Bob
        entity: e.g., laptop

    """
    if "'s " in name_entity:
        name, entity = name_entity.split("'s ")
    else:
        name, entity = None, None

    return name, entity


def get_duplicate_dicts(search: dict, target: list) -> list:
    """Find if there are duplicate dicts.

    Args:
        search: dict
        target: target list to look up.

    Returns:
        duplicates: a list of dicts or None

    """
    assert isinstance(search, dict)
    duplicates = []

    for candidate in target:
        assert isinstance(candidate, dict)
        if set(search).issubset(set(candidate)):
            if all([val == candidate[key] for key, val in search.items()]):
                duplicates.append(candidate)

    return duplicates


def list_duplicates_of(seq, item) -> list:
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def find_connected_nodes(graph_):
    graph = deepcopy(graph_)

    def dfs(node, row, col):
        if (
            row < 0
            or col < 0
            or row >= len(graph)
            or col >= len(graph[0])
            or graph[row][col] == 0
        ):
            return

        connected_nodes.append((row, col))
        graph[row][col] = 0  # Mark the node as visited

        # Check the neighbors
        dfs(node, row - 1, col)  # Up
        dfs(node, row + 1, col)  # Down
        dfs(node, row, col - 1)  # Left
        dfs(node, row, col + 1)  # Right

    connected_components = []
    for row in range(len(graph)):
        for col in range(len(graph[row])):
            if graph[row][col] == 1:
                connected_nodes = []
                dfs(1, row, col)
                if connected_nodes:
                    connected_components.append(connected_nodes)

    return connected_components


def is_running_notebook() -> bool:
    """See if the code is running in a notebook or not."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
