"""
@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import itertools as it
from dataclasses import dataclass
from functools import reduce
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    overload,
)

import numpy as np
import pandas as pd
from scipy.stats import (
    friedmanchisquare,
    kruskal,
    mannwhitneyu,
    rankdata,
    wilcoxon,
)
from scipy.stats.stats import ttest_ind_from_stats, ttest_rel
from statsmodels.sandbox.stats.multicomp import multipletests

CorrectionLike = Literal[
    None,
    'bonferroni',
    'sidak',
    'holm-sidak',
    'holm',
    'simes-hochberg',
    'hommel',
    'fdr_bh',
    'fdr_by',
    'fdr_tsbh',
    'fdr_tsbky',
]

MultitestLike = Literal['kruskal', 'friedmanchisquare']

TestLike = Literal['mannwhitneyu', 'wilcoxon']


@dataclass
class SummaryRow:
    values: np.typing.NDArray[Any]
    greater_is_better: bool | None = None


@dataclass
class ScoreCell:
    mean: float
    std: float | None
    rank: int
    significant: bool


def average_rank(
    ranks: np.typing.NDArray[np.integer[Any]],
    **kwargs: Any,
) -> SummaryRow:
    """Compute rank averages."""
    return SummaryRow(
        values=np.mean(ranks, axis=0),
        greater_is_better=False,
    )


def average_mean_score(
    means: np.typing.NDArray[np.floating[Any]],
    greater_is_better: bool,
    **kwargs: Any,
) -> SummaryRow:
    """Compute score mean averages."""
    return SummaryRow(
        values=np.mean(means, axis=0),
        greater_is_better=greater_is_better,
    )


def _is_significant(
    scores1: np.typing.NDArray[np.floating[Any]],
    scores2: np.typing.NDArray[np.floating[Any]],
    mean1: np.typing.NDArray[np.floating[Any]],
    mean2: np.typing.NDArray[np.floating[Any]],
    std1: np.typing.NDArray[np.floating[Any]],
    std2: np.typing.NDArray[np.floating[Any]],
    *,
    nobs: int | None = None,
    two_sided: bool = True,
    paired_test: bool = False,
    significancy_level: float = 0.05,
) -> bool:

    alternative = "two-sided" if two_sided else "greater"

    if paired_test:
        assert scores1.ndim == 1
        assert scores2.ndim == 1

        _, pvalue = ttest_rel(
            scores1,
            scores2,
            axis=-1,
            alternative=alternative,
        )

    else:
        assert nobs

        _, pvalue = ttest_ind_from_stats(
            mean1=mean1,
            std1=std1,
            nobs1=nobs,
            mean2=mean2,
            std2=std2,
            nobs2=nobs,
            equal_var=False,
            alternative=alternative,
        )

    return pvalue < significancy_level


def _all_significants(
    scores: np.typing.NDArray[np.floating[Any]],
    means: np.typing.NDArray[np.floating[Any]],
    stds: np.typing.NDArray[np.floating[Any]] | None,
    ranks: np.typing.NDArray[np.integer[Any]],
    *,
    nobs: int | None = None,
    two_sided: bool = True,
    paired_test: bool = False,
    significancy_level: float = 0,
) -> np.typing.NDArray[np.bool_]:

    significant_matrix = np.zeros_like(ranks, dtype=np.bool_)

    if stds is None or significancy_level <= 0:
        return significant_matrix

    for row, (scores_row, mean_row, std_row, rank_row) in enumerate(
        zip(scores, means, stds, ranks),
    ):
        for column, (scores1, mean1, std1, rank1) in enumerate(
            zip(scores_row, mean_row, std_row, rank_row),
        ):
            # Compare every element with all the ones with immediate below rank
            # It must be significantly better than all of them
            index2 = np.flatnonzero(rank_row == (rank1 + 1))

            is_significant = len(index2) > 0 and all(
                _is_significant(
                    scores1,
                    scores_row[idx],
                    mean1,
                    mean_row[idx],
                    std1,
                    std_row[idx],
                    nobs=nobs,
                    two_sided=two_sided,
                    paired_test=paired_test,
                    significancy_level=significancy_level,
                )
                for idx in index2
            )

            if is_significant:
                significant_matrix[row, column] = True

    return significant_matrix


def _set_style_classes(
    table: pd.DataFrame,
    *,
    all_ranks: np.typing.NDArray[np.integer[Any]],
    significants: np.typing.NDArray[np.bool_],
    n_summary_rows: int,
) -> pd.io.formats.style.Styler:
    rank_class_names = np.char.add(
        "rank",
        all_ranks.astype(str),
    )

    is_summary_row = np.zeros_like(all_ranks, dtype=np.bool_)
    is_summary_row[-n_summary_rows:, :] = True

    summary_rows_class_name = np.char.multiply(
        "summary",
        is_summary_row.astype(int),
    )

    significant_class_name = np.char.multiply(
        "significant",
        np.insert(
            significants,
            (len(significants),) * n_summary_rows,
            0,
            axis=0,
        ).astype(int),
    )

    styler = table.style.set_td_classes(
        pd.DataFrame(
            reduce(
                np.char.add,
                (
                    rank_class_names,
                    " ",
                    summary_rows_class_name,
                    " ",
                    significant_class_name,
                ),
            ),
            index=table.index,
            columns=table.columns,
        ),
    )

    return styler


def _set_style_formatter(
    styler: pd.io.formats.style.Styler,
    *,
    precision: int,
) -> pd.io.formats.style.Styler:

    def _formatter(
        data: object,
    ) -> str:
        if isinstance(data, str):
            return data
        elif isinstance(data, int):
            return str(int)
        elif isinstance(data, float):
            return f"{data:.{precision}f}"
        elif isinstance(data, ScoreCell):
            str_repr = f'{data.mean:.{precision}f}'
            if data.std is not None:
                str_repr += f' ± {data.std:.{precision}f}'
            str_repr += f' ({data.rank:.0f})'
            return str_repr
        else:
            return ""

    return styler.format(
        _formatter,
    )


def _set_default_style_html(
    styler: pd.io.formats.style.Styler,
    *,
    n_summary_rows: int,
) -> pd.io.formats.style.Styler:

    last_rows_mask = np.zeros(len(styler.data), dtype=int)
    last_rows_mask[-n_summary_rows:] = 1

    styler = styler.set_table_styles(
        [
            {
                "selector": ".summary",
                "props": [("font-style", "italic")],
            },
            {
                "selector": ".rank1",
                "props": [("font-weight", "bold")],
            },
            {
                "selector": ".rank2",
                "props": [("text-decoration", "underline")],
            },
            {
                "selector": ".significant::after",
                "props": [
                    ("content", "\"*\""),
                    ("width", "0px"),
                    ("display", "inline-block"),
                ],
            },
            {
                "selector": ".col_heading",
                "props": [("font-weight", "bold")],
            },
        ],
    )

    styler = styler.apply_index(
        lambda _: np.char.multiply(
            "font-style: italic; font-weight: bold",
            last_rows_mask,
        ),
        axis=0,
    )

    styler = styler.apply_index(
        lambda idx: ["font-weight: bold"] * len(idx),
        axis=1,
    )

    return styler


def _set_style_from_class(
    styler: pd.io.formats.style.Styler,
    class_name: str,
    style: str,
) -> pd.io.formats.style.Styler:
    style_matrix = np.full(styler.data.shape, style)

    for row in range(style_matrix.shape[0]):
        for column in range(style_matrix.shape[1]):
            classes = styler.cell_context.get(
                (row, column),
                "",
            ).split()

            if class_name not in classes:
                style_matrix[row, column] = ""

    return styler.apply(lambda x: style_matrix, axis=None)


def _set_default_style_latex(
    styler: pd.io.formats.style.Styler,
    *,
    n_summary_rows: int,
) -> pd.io.formats.style.Styler:

    last_rows_mask = np.zeros(len(styler.data), dtype=int)
    last_rows_mask[-n_summary_rows:] = 1

    styler.set_table_styles(
        [
            {
                'selector': r'newcommand{\summary}',
                'props': r':[1]{\textit{#1}};',
            },
            {
                'selector': r'newcommand{\significant}',
                'props': r':[1]{#1*};',
            },
            {
                'selector': r'newcommand{\rank}',
                'props': (
                    r':[2]{\ifnum#1=1 \textbf{#2} \else '
                    r'\ifnum#1=2 \underline{#2} \fi\fi};'
                ),
            },
        ],
        overwrite=False,
    )

    for rank in range(styler.data.shape[1]):
        styler = _set_style_from_class(
            styler,
            f"rank{rank}",
            f"rank{{{rank}}}:--rwrap; ",
        )

    for class_name in ("summary", "significant"):

        styler = _set_style_from_class(
            styler,
            class_name,
            f"{class_name}:--rwrap; ",
        )

    styler = styler.apply_index(
        lambda _: np.char.multiply(
            "textbf:--rwrap;summary:--rwrap;",
            last_rows_mask,
        ),
        axis=0,
    )

    styler = styler.apply_index(
        lambda idx: ["textbf:--rwrap"] * len(idx),
        axis=1,
    )

    return styler


def _set_default_style(
    styler: pd.io.formats.style.Styler,
    *,
    n_summary_rows: int,
    default_style: Literal["html", "latex", None],
) -> pd.io.formats.style.Styler:

    if default_style == "html":
        styler = _set_default_style_html(
            styler,
            n_summary_rows=n_summary_rows,
        )
    elif default_style == "latex":
        styler = _set_default_style_latex(
            styler,
            n_summary_rows=n_summary_rows,
        )

    return styler


def scores_table(
    scores: np.typing.ArrayLike,
    stds: np.typing.ArrayLike | None = None,
    *,
    datasets: Sequence[str],
    estimators: Sequence[str],
    nobs: int | None = None,
    greater_is_better: bool = True,
    method: Literal['average', 'min', 'max', 'dense', 'ordinal'] = 'min',
    significancy_level: float = 0,
    paired_test: bool = False,
    two_sided: bool = True,
    default_style: Literal["html", "latex", None] = "html",
    precision: int = 2,
    summary_rows: Sequence[Tuple[str, Callable[..., SummaryRow]]] = (
        ("Average rank", average_rank),
    ),
) -> pd.io.formats.style.Styler:
    """
    Scores table.

    Prints a table where each row represents a dataset and each column
    represents an estimator.

    Parameters
    ----------
    scores: array-like
        Matrix of scores where each column represents a model.
        Either the full matrix with all experiment results or the
        matrix with the mean scores can be passed.
    stds: array-like, default=None
        Matrix of standard deviations where each column represents a
        model. If ``scores`` is the full matrix with all results
        this is automatically computed from it and should not be passed.
    datasets: sequence of :external:class:`str`
        List of dataset names.
    estimators: sequence of :external:class:`str`
        List of estimator names.
    nobs: :external:class:`int`
        Number of repetitions of the experiments. Used only for computing
        significances when ``scores`` is not the full matrix.
    greater_is_better: boolean, default=True
        Whether a greater score is better (score) or worse
        (loss).
    method: {'average', 'min', 'max', 'dense', 'ordinal'}, default='average'
        Method used to solve ties.
    significancy_level: :external:class:`float`, default=0
        Significancy level for considerin a result significant. If nonzero,
        significancy is calculated using a t-test. In that case, if
        ``paired_test`` is ``True``, ``scores`` should be the full matrix
        and a paired test is performed. Otherwise, the t-test assumes
        independence, and either ``scores`` should be the full matrix
        or ``nobs`` should be passed.
    paired_test: :external:class:`bool`, default=False
        Whether to perform a paired test or a test assuming independence.
        If ``True``, ``scores`` should be the full matrix.
        Otherwise, either ``scores`` should be the full matrix
        or ``nobs`` should be passed.
    two_sided: :external:class:`bool`, default=True
        Whether to perform a two sided t-test or a one sided t-test.
    default_style: {'html', 'latex', None}, default='html'
        Default style for the table. Use ``None`` for no style. Note that
        the CSS classes and textual formatting are always set.
    precision: :external:class:`int`
        Number of decimals used for floating point numbers.
    summary_rows: sequence
        List of (name, callable) tuples for additional summary rows.
        By default, the rank average is computed.

    Returns
    -------
    table: array-like
        Table of mean and standard deviation of each estimator-dataset
        pair. A ranking of estimators is also generated.

    """
    scores = np.asanyarray(scores)
    stds = None if stds is None else np.asanyarray(stds)

    assert scores.ndim in {2, 3}
    means = scores if scores.ndim == 2 else np.mean(scores, axis=-1)
    if scores.ndim == 3:
        assert stds is None
        assert nobs is None
        stds = np.std(scores, axis=-1)
        nobs = scores.shape[-1]

    ranks = np.asarray([
        rankdata(-m, method=method)
        if greater_is_better
        else rankdata(m, method=method)
        for m in means
    ])

    significants = _all_significants(
        scores,
        means,
        stds,
        ranks,
        nobs=nobs,
        two_sided=two_sided,
        paired_test=paired_test,
        significancy_level=significancy_level,
    )

    table = pd.DataFrame(data=means, index=datasets, columns=estimators)
    for i, d in enumerate(datasets):
        for j, e in enumerate(estimators):
            table.loc[d, e] = ScoreCell(
                mean=means[i, j],
                std=None if stds is None else stds[i, j],
                rank=ranks[i, j],
                significant=significants[i, j],
            )

    # Create additional summary rows
    additional_ranks = []
    for name, summary_fun in summary_rows:
        row = summary_fun(
            scores=scores,
            means=means,
            stds=stds,
            ranks=ranks,
            greater_is_better=greater_is_better,
        )
        table.loc[name] = row.values

        if row.greater_is_better is None:
            additional_ranks.append(np.full(len(row.values), -1))
        else:
            additional_ranks.append(
                rankdata(-row.values, method=method)
                if row.greater_is_better
                else rankdata(row.values, method=method),
            )

    styler = _set_style_classes(
        table,
        all_ranks=np.vstack([ranks] + additional_ranks),
        significants=significants,
        n_summary_rows=len(summary_rows),
    )

    styler = _set_style_formatter(
        styler,
        precision=precision,
    )

    return _set_default_style(
        styler,
        n_summary_rows=len(summary_rows),
        default_style=default_style,
    )


def hypotheses_table(
    samples: np.typing.ArrayLike,
    models: Sequence[str],
    *,
    alpha: float = 0.05,
    multitest: Optional[MultitestLike] = None,
    test: TestLike = 'wilcoxon',
    correction: CorrectionLike = None,
    multitest_args: Optional[Mapping[str, Any]] = None,
    test_args: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Hypotheses table.

    Prints a hypothesis table with a selected test and correction.

    Parameters
    ----------
    samples: array-like
        Matrix of samples where each column represent a model.
    models: array-like
        Model names.
    alpha: float in [0, 1], default=0.05
        Significance level.
    multitest: {'kruskal', 'friedmanchisquare'}, default=None
        Ranking multitest used.
    test: {'mannwhitneyu', 'wilcoxon'}, default='wilcoxon'
        Ranking test used.
    correction: {'bonferroni', 'sidak', 'holm-sidak', 'holm', \
                 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', \
                 'fdr_tsbky'}, default=None
        Method used to adjust the p-values.
    multitest_args: dict
        Optional ranking test arguments.
    test_args: dict
        Optional ranking test arguments.

    Returns
    -------
    multitest_table: array-like
        Table of p-value and rejection/non-rejection for the
        multitest hypothesis.
    test_table: array-like
        Table of p-values and rejection/non-rejection for each test
        hypothesis.

    """
    if multitest_args is None:
        multitest_args = {}

    if test_args is None:
        test_args = {}

    samples = np.asanyarray(samples)

    versus = list(it.combinations(range(len(models)), 2))
    comparisons = [
        f"{models[first]} vs {models[second]}"
        for first, second in versus
    ]

    multitests = {
        'kruskal': kruskal,
        'friedmanchisquare': friedmanchisquare,
    }
    tests = {
        'mannwhitneyu': mannwhitneyu,
        'wilcoxon': wilcoxon,
    }

    multitest_table = None
    if multitest is not None:
        multitest_table = pd.DataFrame(
            index=[multitest],
            columns=['p-value', 'Hypothesis'],
        )
        _, pvalue = multitests[multitest](
            *samples.T,
            **multitest_args,
        )
        reject_str = 'Rejected' if pvalue <= alpha else 'Not rejected'
        multitest_table.loc[multitest] = ['{0:.2f}'.format(pvalue), reject_str]

        # If the multitest does not detect a significative difference,
        # the individual tests are not meaningful, so skip them.
        if pvalue > alpha:
            return multitest_table, None

    pvalues = [
        tests[test](
            samples[:, first],
            samples[:, second],
            **test_args,
        )[1] for first, second in versus
    ]

    if correction is not None:
        reject_bool, pvalues, _, _ = multipletests(
            pvalues,
            alpha,
            method=correction,
        )
        reject = [
            'Rejected'
            if r
            else 'Not rejected'
            for r in reject_bool
        ]
    else:
        reject = [
            'Rejected'
            if pvalue <= alpha
            else 'Not rejected'
            for pvalue in pvalues
        ]

    data = [
        ('{0:.2f}'.format(p), r)
        for p, r in zip(pvalues, reject)
    ]

    test_table = pd.DataFrame(
        data,
        index=comparisons,
        columns=['p-value', 'Hypothesis'],
    )

    return multitest_table, test_table
