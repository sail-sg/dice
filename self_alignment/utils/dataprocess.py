import re

import pandas as pd


def identity_mapping(x: list):
    return x


def remove_repeat_answers(x: list, sep: str = "---"):
    x = pd.Series(x)
    x = x.apply(lambda x: x.split(sep)[0])
    return x.tolist()


def extract_by_regex(x: list, pattern: str = r"Answer: (.*)"):
    x = pd.Series(x)
    x = x.apply(
        lambda x: re.search(pattern, x).group(1) if re.search(pattern, x) else None
    )
    return x.tolist()
