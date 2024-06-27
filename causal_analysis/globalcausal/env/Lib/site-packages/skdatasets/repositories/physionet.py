from __future__ import annotations

import ast
import math
import re
import urllib
from html.parser import HTMLParser
from pathlib import Path
from typing import (
    Any,
    Final,
    List,
    Literal,
    Mapping,
    Sequence,
    Tuple,
    overload,
)

import numpy as np
import pandas as pd
import wfdb.io
from sklearn.utils import Bunch

from skdatasets.repositories.base import dataset_from_dataframe

from .base import DatasetNotFoundError, fetch_zip

BASE_URL: Final = "https://physionet.org/static/published-projects"
INFO_STRING_SEMICOLONS_ONE_STR: Final = r"(\S*): (\S*)\s*"
INFO_STRING_SEMICOLONS_SEVERAL_REGEX: Final = re.compile(
    fr"(?:{INFO_STRING_SEMICOLONS_ONE_STR})+",
)
INFO_STRING_SEMICOLONS_ONE_REGEX: Final = re.compile(
    INFO_STRING_SEMICOLONS_ONE_STR,
)


class _ZipNameHTMLParser(HTMLParser):
    """Class for parsing the zip name in PhysioNet directory listing."""

    def __init__(self, *, convert_charrefs: bool = True) -> None:
        super().__init__(convert_charrefs=convert_charrefs)

        self.zip_name: str | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: List[Tuple[str, str | None]],
    ) -> None:
        if tag == "a":
            for attr in attrs:
                if attr[0] == "href" and attr[1] and attr[1].endswith(".zip"):
                    self.zip_name = attr[1]


def _get_zip_name_online(dataset_name: str) -> str:
    """Get the zip name of the dataset."""
    parser = _ZipNameHTMLParser()

    url_request = urllib.request.Request(url=f"{BASE_URL}/{dataset_name}")
    try:
        with urllib.request.urlopen(url_request) as url_file:
            url_content = url_file.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise DatasetNotFoundError(dataset_name) from e
        raise

    parser.feed(url_content)

    if parser.zip_name is None:
        raise ValueError(f"No zip file found for dataset '{dataset_name}'")

    return parser.zip_name


def _parse_info_string_value(value: str) -> Any:
    if value.lower() == "nan":
        return math.nan
    try:
        value = ast.literal_eval(value)
    except Exception:
        pass

    return value


def _get_info_strings(comments: Sequence[str]) -> Mapping[str, Any]:

    info_strings_semicolons = {}
    info_strings_spaces = {}

    for comment in comments:
        if comment[0] not in {"-", "#"}:
            if re.fullmatch(INFO_STRING_SEMICOLONS_SEVERAL_REGEX, comment):
                for result in re.finditer(
                    INFO_STRING_SEMICOLONS_ONE_REGEX,
                    comment,
                ):
                    key = result.group(1)
                    if key[0] == "<" and key[-1] == ">":
                        key = key[1:-1]
                    info_strings_semicolons[key] = (
                        _parse_info_string_value(result.group(2))
                    )
            else:
                split = comment.rsplit(maxsplit=1)
                if len(split) == 2:
                    key, value = split
                    info_strings_spaces[key] = _parse_info_string_value(value)

    if info_strings_semicolons:
        return info_strings_semicolons

    # Check for absurd things in spaces
    if (
        len(info_strings_spaces) == 1
        or any(key.count(" ") > 3 for key in info_strings_spaces)
    ):
        return {}

    return info_strings_spaces


def _join_info_dicts(
    dicts: Sequence[Mapping[str, Any]],
) -> Mapping[str, np.typing.NDArray[Any]]:

    joined = {}

    n_keys = len(dicts[0])
    if not all(len(d) == n_keys for d in dicts):
        return {}

    for key in dicts[0]:
        joined[key] = np.array([d[key] for d in dicts])

    return joined


def _constant_attrs(register: wfdb.Record) -> Sequence[Any]:
    return (register.n_sig, register.sig_name, register.units, register.fs)


@overload
def fetch(
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
    target_column: str | Sequence[str] | None = None,
) -> Bunch:
    pass


@overload
def fetch(
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
    target_column: None = None,
) -> Tuple[np.typing.NDArray[Any], None]:
    pass


@overload
def fetch(
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
    target_column: str | Sequence[str],
) -> Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]:
    pass


@overload
def fetch(
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: Literal[True],
    as_frame: Literal[True],
    target_column: None = None,
) -> Tuple[pd.DataFrame, None]:
    pass


@overload
def fetch(
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: Literal[True],
    as_frame: Literal[True],
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    pass


@overload
def fetch(
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: Literal[True],
    as_frame: Literal[True],
    target_column: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pass


def fetch(
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: bool = False,
    as_frame: bool = False,
    target_column: str | Sequence[str] | None = None,
) -> (
    Bunch
    | Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any] | None]
    | Tuple[pd.DataFrame, pd.Series | pd.DataFrame | None]
):

    zip_name = _get_zip_name_online(name)

    path = fetch_zip(
        dataname=name,
        urlname=f"{BASE_URL}/{name}/{zip_name}",
        subfolder="physionet",
        data_home=data_home,
    )

    subpath = path / Path(zip_name).stem
    if subpath.exists():
        path = subpath

    with open(path / "RECORDS") as records_file:
        records = [
            wfdb.io.rdrecord(str(path / record_name.rstrip('\n')))
            for record_name in records_file
        ]

    info_strings = [_get_info_strings(r.comments) for r in records]
    info = _join_info_dicts(info_strings)

    assert all(
        _constant_attrs(r) == _constant_attrs(records[0])
        for r in records
    )
    data = {
        "signal": [r.p_signal for r in records],
    }

    dataframe = pd.DataFrame(
        {**info, **data},
        index=[r.record_name for r in records],
    )
    dataframe["signal"].attrs.update(
        sig_name=records[0].sig_name,
        units=records[0].units,
        fs=records[0].fs,
    )

    return dataset_from_dataframe(
        dataframe,
        return_X_y=return_X_y,
        as_frame=as_frame,
        target_column=target_column,
    )
