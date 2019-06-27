from pathlib import Path
from typing import List
from typing import Dict

from yaml import load
from yaml import Loader


def parse_yaml(filepath: Path):
    with open(filepath, "r") as f:
        stream = "".join(f.readlines())
        return load(stream, Loader=Loader)


def name2dir(name: str):
    return "_".join([s for s in name.split(" ")])


def name2link(name: str):
    """Used for hyperlink anchors"""
    if not isinstance(name, str):
        name = str(name)

    return "-".join([s.lower() for s in name.split(" ")])


def dir2name(name: str):
    if not isinstance(name, str):
        name = str(name)

    # return " ".join([w[0].upper() + w[1:] for w in name.split("_")])
    return " ".join([w for w in name.split("_")])


def find_subdirectories(path: Path):
    if not isinstance(path, Path):
        path = Path(path)

    return sorted(list(filter(lambda f: f.is_dir() and f.name != "__pycache__", path.glob("*"))))


def sort(
        data: List[Dict],
        sort_key: str,
        sort_reverse: bool,
):
    if sort_key is not None:
        data = sorted(data, key=lambda k: k[sort_key], reverse=sort_reverse)

    return data
