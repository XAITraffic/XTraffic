"""
@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import itertools
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from inspect import signature
from tempfile import NamedTemporaryFile, mkdtemp
from time import perf_counter, process_time, sleep
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from warnings import warn

import joblib
import numpy as np
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver, MongoObserver, RunObserver
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import check_cv
from sklearn.utils import Bunch, is_scalar_nan

from incense import ExperimentLoader, FileSystemExperimentLoader
from incense.experiment import FileSystemExperiment

SelfType = TypeVar("SelfType")


class DataLike(Protocol):

    def __getitem__(
        self: SelfType,
        key: np.typing.NDArray[int],
    ) -> SelfType:
        pass

    def __len__(self) -> int:
        pass


DataType = TypeVar("DataType", bound=DataLike, contravariant=True)
TargetType = TypeVar("TargetType", bound=DataLike)
IndicesType = Tuple[np.typing.NDArray[int], np.typing.NDArray[int]]
ExplicitSplitType = Tuple[
    np.typing.NDArray[float],
    np.typing.NDArray[Union[float, int]],
    np.typing.NDArray[float],
    np.typing.NDArray[Union[float, int]],
]

ConfigLike = Union[
    Mapping[str, Any],
    str,
]


class EstimatorProtocol(Protocol[DataType, TargetType]):

    def fit(self: SelfType, X: DataType, y: TargetType) -> SelfType:
        pass

    def predict(self, X: DataType) -> TargetType:
        pass


class CVSplitter(Protocol):

    def split(
        self,
        X: np.typing.NDArray[float],
        y: None = None,
        groups: None = None,
    ) -> Iterable[IndicesType]:
        pass

    def get_n_splits(
        self,
        X: np.typing.NDArray[float],
        y: None = None,
        groups: None = None,
    ) -> int:
        pass


CVLike = Union[
    CVSplitter,
    Iterable[IndicesType],
    int,
    None,
]

EstimatorLike = Union[
    EstimatorProtocol[Any, Any],
    Callable[..., EstimatorProtocol[Any, Any]],
    Tuple[Callable[..., EstimatorProtocol[Any, Any]], ConfigLike],
]

DatasetLike = Union[
    Bunch,
    Callable[..., Bunch],
    Tuple[Callable[..., Bunch], ConfigLike],
]


@dataclass
class ScoresInfo:
    r"""
    Class containing the scores of several related experiments.

    Attributes
    ----------
    dataset_names : Sequence of :external:class:`str`
        Name of the datasets, with the same order in which are present
        in the rows of the scores.
    estimator_names : Sequence of :external:class:`str`
        Name of the estimators, with the same order in which are present
        in the columns of the scores.
    scores : :external:class:`numpy.ndarray`
        Test scores. It has size ``n_datasets`` :math:`\times` ``n_estimators``
        :math:`\times` ``n_partitions``.
    scores_mean : :external:class:`numpy.ndarray`
        Test score means. It has size ``n_datasets``
        :math:`\times` ``n_estimators``.
    scores_std : :external:class:`numpy.ndarray`
        Test score standard deviations. It has size ``n_datasets``
        :math:`\times` ``n_estimators``.

    See Also
    --------
    fetch_scores

    """
    dataset_names: Sequence[str]
    estimator_names: Sequence[str]
    scores: np.typing.NDArray[float]
    scores_mean: np.typing.NDArray[float]
    scores_std: np.typing.NDArray[float]


def _append_info(experiment: Experiment, name: str, value: Any) -> None:
    info_list = experiment.info.get(name, [])
    info_list.append(value)
    experiment.info[name] = info_list


@contextmanager
def _add_timing(experiment: Experiment, name: str) -> Iterator[None]:
    initial_time = perf_counter()
    try:
        yield None
    finally:
        final_time = perf_counter()
        elapsed_time = final_time - initial_time
        _append_info(experiment, name, elapsed_time)


def _iterate_outer_cv(
    outer_cv: CVLike | Iterable[
        Tuple[DataType, TargetType, DataType, TargetType]
    ],
    estimator: EstimatorProtocol[DataType, TargetType],
    X: DataType,
    y: TargetType,
) -> Iterable[Tuple[DataType, TargetType, DataType, TargetType]]:
    """Iterate over multiple partitions."""
    if isinstance(outer_cv, Iterable):
        outer_cv, cv_copy = itertools.tee(outer_cv)
        if len(next(cv_copy)) == 4:
            yield from outer_cv

    cv = check_cv(outer_cv, y, classifier=is_classifier(estimator))
    yield from (
        (X[train], y[train], X[test], y[test])
        for train, test in cv.split(X, y)
    )


def _benchmark_from_data(
    experiment: Experiment,
    *,
    estimator: BaseEstimator,
    X_train: DataType,
    y_train: TargetType,
    X_test: DataType,
    y_test: TargetType,
    save_estimator: bool = False,
    save_train: bool = False,
) -> None:
    with _add_timing(experiment, "fit_time"):
        estimator.fit(X_train, y_train)

    if save_estimator:
        _append_info(experiment, "fitted_estimator", estimator)

    best_params = getattr(estimator, "best_params_", None)
    if best_params:
        _append_info(experiment, "search_best_params", best_params)

    best_score = getattr(estimator, "best_score_", None)
    if best_params:
        _append_info(experiment, "search_best_score", best_score)

    with _add_timing(experiment, "score_time"):
        test_score = estimator.score(X_test, y_test)

    _append_info(experiment, "test_score", float(test_score))

    if save_train:
        train_score = estimator.score(X_train, y_train)
        _append_info(experiment, "train_score", float(train_score))

    for output in ("transform", "predict"):
        method = getattr(estimator, output, None)
        if method is not None:
            with _add_timing(experiment, f"{output}_time"):
                _append_info(experiment, f"{output}", method(X_test))


def _compute_means(experiment: Experiment) -> None:

    experiment.info["score_mean"] = float(
        np.nanmean(experiment.info["test_score"])
    )
    experiment.info["score_std"] = float(
        np.nanstd(experiment.info["test_score"])
    )


def _benchmark_one(
    experiment: Experiment,
    *,
    estimator: BaseEstimator,
    data: Bunch,
    save_estimator: bool = False,
    save_train: bool = False,
) -> None:
    """Use only one predefined partition."""
    X = data.data
    y = data.target

    train_indices = getattr(data, "train_indices", [])
    validation_indices = getattr(data, "validation_indices", [])
    test_indices = getattr(data, "test_indices", [])

    X_train_val = (
        X[train_indices + validation_indices]
        if train_indices
        else X
    )
    y_train_val = (
        y[train_indices + validation_indices]
        if train_indices
        else y
    )

    X_test = X[test_indices]
    y_test = y[test_indices]

    _benchmark_from_data(
        experiment=experiment,
        estimator=estimator,
        X_train=X_train_val,
        y_train=y_train_val,
        X_test=X_test,
        y_test=y_test,
        save_estimator=save_estimator,
        save_train=save_train,
    )

    _compute_means(experiment)


def _benchmark_partitions(
    experiment: Experiment,
    *,
    estimator: BaseEstimator,
    data: Bunch,
    save_estimator: bool = False,
    save_train: bool = False,
    outer_cv: CVLike | Literal["dataset"] = None,
) -> None:
    """Use several partitions."""
    outer_cv = data.outer_cv if outer_cv == "dataset" else outer_cv

    for X_train, y_train, X_test, y_test in _iterate_outer_cv(
        outer_cv=outer_cv,
        estimator=estimator,
        X=data.data,
        y=data.target,
    ):

        _benchmark_from_data(
            experiment=experiment,
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            save_estimator=save_estimator,
            save_train=save_train,
        )

    _compute_means(experiment)


def _benchmark(
    experiment: Experiment,
    *,
    estimator: BaseEstimator,
    data: Bunch,
    save_estimator: bool = False,
    save_train: bool = False,
    outer_cv: CVLike | Literal[False, "dataset"] = None,
) -> None:
    """Run the experiment."""
    if outer_cv is False:
        _benchmark_one(
            experiment=experiment,
            estimator=estimator,
            data=data,
            save_estimator=save_estimator,
            save_train=save_train,
        )
    else:
        _benchmark_partitions(
            experiment=experiment,
            estimator=estimator,
            data=data,
            save_estimator=save_estimator,
            save_train=save_train,
            outer_cv=outer_cv,
        )


def experiment(
    dataset: Callable[..., Bunch],
    estimator: Callable[..., BaseEstimator],
    *,
    save_estimator: bool = False,
    save_train: bool = False,
) -> Experiment:
    """
    Prepare a Scikit-learn experiment as a Sacred experiment.

    Prepare a Scikit-learn experiment indicating a dataset and an estimator and
    return it as a Sacred experiment.

    Parameters
    ----------
    dataset : function
        Dataset fetch function. Might receive any argument. Must return a
        :external:class:`sklearn.utils.Bunch` with ``data``, ``target``
        (might be ``None``), ``inner_cv`` (might be ``None``) and ``outer_cv``
        (might be ``None``).
    estimator : function
        Estimator initialization function. Might receive any keyword argument.
        Must return an initialized sklearn-compatible estimator.

    Returns
    -------
    experiment : Experiment
        Sacred experiment, ready to be run.

    """
    dataset_ingredient = Ingredient("dataset")
    dataset = dataset_ingredient.capture(dataset)
    estimator_ingredient = Ingredient("estimator")
    estimator = estimator_ingredient.capture(estimator)
    experiment = Experiment(
        ingredients=(
            dataset_ingredient,
            estimator_ingredient,
        ),
    )

    @experiment.main
    def run() -> None:
        """Run the experiment."""
        data = dataset()

        # Metaparameter search
        cv = getattr(data, "inner_cv", None)

        try:
            e = estimator(cv=cv)
        except TypeError as exception:
            warn(f"The estimator does not accept cv: {exception}")
            e = estimator()

        # Model assessment
        _benchmark(
            experiment=experiment,
            estimator=e,
            data=data,
            save_estimator=save_estimator,
            save_train=save_train,
        )

        # Ensure that everything is in the info dict at the end
        # See https://github.com/IDSIA/sacred/issues/830
        sleep(experiment.current_run.beat_interval + 1)

    return experiment


def _get_estimator_function(
    experiment: Experiment,
    estimator: EstimatorLike,
) -> Callable[..., EstimatorProtocol[Any, Any]]:

    if hasattr(estimator, "fit"):
        def estimator_function() -> EstimatorProtocol:
            return estimator
    else:
        estimator_function = estimator

    return experiment.capture(estimator_function)


def _get_dataset_function(
    experiment: Experiment,
    dataset: DatasetLike,
) -> Callable[..., Bunch]:

    if callable(dataset):
        dataset_function = dataset
    else:
        def dataset_function() -> Bunch:
            return dataset

    return experiment.capture(dataset_function)


def _create_one_experiment(
    *,
    estimator_name: str,
    estimator: EstimatorLike,
    dataset_name: str,
    dataset: DatasetLike,
    storage: RunObserver,
    config: ConfigLike,
    inner_cv: CVLike | Literal[False, "dataset"] = None,
    outer_cv: CVLike | Literal[False, "dataset"] = None,
    save_estimator: bool = False,
    save_train: bool = False,
) -> Experiment:
    experiment = Experiment()

    experiment.add_config(config)

    experiment.add_config({"estimator_name": estimator_name})
    if isinstance(estimator, tuple):
        estimator, estimator_config = estimator
        experiment.add_config(estimator_config)

    experiment.add_config({"dataset_name": dataset_name})
    if isinstance(dataset, tuple):
        dataset, dataset_config = dataset
        experiment.add_config(dataset_config)

    experiment.observers.append(storage)

    estimator_function = _get_estimator_function(experiment, estimator)
    dataset_function = _get_dataset_function(experiment, dataset)

    @experiment.main
    def run() -> None:
        """Run the experiment."""
        dataset = dataset_function()

        # Metaparameter search
        cv = dataset.inner_cv if inner_cv == "dataset" else inner_cv

        estimator = estimator_function()
        if hasattr(estimator, "cv") and cv is not False:
            estimator.cv = cv

        # Model assessment
        _benchmark(
            experiment=experiment,
            estimator=estimator,
            data=dataset,
            save_estimator=save_estimator,
            save_train=save_train,
            outer_cv=outer_cv,
        )

    return experiment


def create_experiments(
    *,
    datasets: Mapping[str, DatasetLike],
    estimators: Mapping[str, EstimatorLike],
    storage: RunObserver | str,
    config: ConfigLike | None = None,
    inner_cv: CVLike | Literal[False, "dataset"] = False,
    outer_cv: CVLike | Literal[False, "dataset"] = None,
    save_estimator: bool = False,
    save_train: bool = False,
) -> Sequence[Experiment]:
    """
    Create several Sacred experiments.

    It receives a set of estimators and datasets, and create Sacred experiment
    objects for them.

    Parameters
    ----------
    datasets : Mapping
        Mapping where each key is the name for a dataset and each value
        is either:

        * A :external:class:`sklearn.utils.Bunch` with the fields explained
          in :doc:`/structure`. Only ``data`` and ``target`` are
          mandatory.
        * A function receiving arbitrary config values and returning a
          :external:class:`sklearn.utils.Bunch` object like the one explained
          above.
        * A tuple with such a function and additional configuration (either
          a mapping or a filename).
    estimators : Mapping
        Mapping where each key is the name for a estimator and each value
        is either:

        * A scikit-learn compatible estimator.
        * A function receiving arbitrary config values and returning a
          scikit-learn compatible estimator.
        * A tuple with such a function and additional configuration (either
          a mapping or a filename).
    storage : :external:class:`sacred.observers.RunObserver` or :class:`str`
        Where the experiments will be stored. Either a Sacred observer, for
        example to store in a Mongo database, or the name of a directory, to
        use a file observer.
    config : Mapping, :class:`str` or ``None``, default ``None``
        A mapping or filename with additional configuration for the experiment.
    inner_cv : CV-like object, ``"datasets"`` or ``False``, default ``False``
        For estimators that perform cross validation (they have a ``cv``
        parameter) this sets the cross validation strategy, as follows:

        * If ``False`` the original value of ``cv`` is unchanged.
        * If ``"dataset"``, the :external:class:`sklearn.utils.Bunch` objects
          for the datasets must have a ``inner_cv`` attribute, which will
          be the one used.
        * Otherwise, ``cv`` is changed to this value.
    outer_cv : CV-like object, ``"datasets"`` or ``False``, default ``None``
        The strategy used to evaluate different partitions of the data, as
        follows:

        * If ``False`` use only one partition: the one specified in the
          dataset. Thus the :external:class:`sklearn.utils.Bunch` objects
          for the datasets should have defined at least a train and a test
          partition.
        * If ``"dataset"``, the :external:class:`sklearn.utils.Bunch` objects
          for the datasets must have a ``outer_cv`` attribute, which will
          be the one used.
        * Otherwise, this will be passed to
          :external:func:`sklearn.model_selection.check_cv` and the resulting
          cross validator will be used to define the partitions.
    save_estimator : bool, default ``False``
        Whether to save the fitted estimator. This is useful for debugging
        and for obtaining extra information in some cases, but for some
        estimators it could consume much storage.
    save_train : bool, default ``False``
        If ``True``, compute and store also the score over the train data.

    Returns
    -------
    experiments : Sequence of :external:class:`sacred.Experiment`
        Sequence of Sacred experiments, ready to be run.

    See Also
    --------
    run_experiments
    fetch_scores

    """
    if isinstance(storage, str):
        storage = FileStorageObserver(storage)

    if config is None:
        config = {}

    return [
        _create_one_experiment(
            estimator_name=estimator_name,
            estimator=estimator,
            dataset_name=dataset_name,
            dataset=dataset,
            storage=storage,
            config=config,
            inner_cv=inner_cv,
            outer_cv=outer_cv,
            save_estimator=save_estimator,
            save_train=save_train,
        )
        for estimator_name, estimator in estimators.items()
        for dataset_name, dataset in datasets.items()
    ]


def run_experiments(
    experiments: Sequence[Experiment],
) -> Sequence[int]:
    """
    Run Sacred experiments.

    Parameters
    ----------
    experiments : Sequence of :external:class:`sacred.Experiment`
        Sequence of Sacred experiments to be run.

    Returns
    -------
    ids : Sequence of :external:class:`int`
        Sequence of identifiers for each experiment.

    See Also
    --------
    create_experiments
    fetch_scores

    """
    return [e.run()._id for e in experiments]


def _loader_from_observer(
    storage: RunObserver | str,
) -> ExperimentLoader | FileSystemExperimentLoader:

    if isinstance(storage, str):
        return FileSystemExperimentLoader(storage)
    elif isinstance(storage, FileStorageObserver):
        return FileSystemExperimentLoader(storage.basedir)
    elif isinstance(storage, MongoObserver):
        database = storage.runs.database
        client = database.client
        url, port = list(
            client.topology_description.server_descriptions().keys(),
        )[0]

        return ExperimentLoader(
            mongo_uri=f"mongodb://{url}:{port}/",
            db_name=database.name,
            unpickle=False,
        )

    raise ValueError(f"Observer {storage} is not supported.")


def _get_experiments(
    *,
    storage: RunObserver | str,
    ids: Sequence[int] | None = None,
    dataset_names: Sequence[str] | None = None,
    estimator_names: Sequence[str] | None = None,
) -> Sequence[Experiment]:

    loader = _loader_from_observer(storage)

    if (
        (ids, dataset_names, estimator_names) == (None, None, None)
        or isinstance(loader, FileSystemExperimentLoader) and ids is None
    ):
        find_all_fun = getattr(
            loader,
            "find_all",
            lambda: [
                FileSystemExperiment.from_run_dir(run_dir)
                for run_dir in loader._runs_dir.iterdir()
            ],
        )

        experiments = find_all_fun()

    elif (
        (dataset_names, estimator_names) == (None, None)
        or isinstance(loader, FileSystemExperimentLoader)
    ):
        load_ids_fun = getattr(
            loader,
            "find_by_ids",
            lambda id_seq: [
                loader.find_by_id(experiment_id)
                for experiment_id in id_seq
            ],
        )

        experiments = load_ids_fun(ids)

    else:

        conditions: List[
            Mapping[
                str,
                Mapping[str, Sequence[Any]],
            ]
        ] = []

        if ids is not None:
            conditions.append({"_id": {"$in": ids}})

        if estimator_names is not None:
            conditions.append(
                {"config.estimator_name": {"$in": estimator_names}})

        if dataset_names is not None:
            conditions.append({"config.dataset_name": {"$in": dataset_names}})

        query = {"$and": conditions}

        experiments = loader.find(query)

    if isinstance(loader, FileSystemExperimentLoader):
        # Filter experiments by dataset and estimator names
        experiments = [
            e for e in experiments
            if (
                (
                    estimator_names is None
                    or e.config["estimator_name"] in estimator_names
                )
                and (
                    dataset_names is None
                    or e.config["dataset_name"] in dataset_names
                )
            )
        ]

    return experiments


def fetch_scores(
    *,
    storage: RunObserver | str,
    ids: Sequence[int] | None = None,
    dataset_names: Sequence[str] | None = None,
    estimator_names: Sequence[str] | None = None,
) -> ScoresInfo:
    """
    Fetch scores from Sacred experiments.

    By default, it retrieves every experiment. The parameters ``ids``,
    ``estimator_names`` and ``dataset_names`` can be used to restrict the
    number of experiments returned.

    Parameters
    ----------
    storage : :external:class:`sacred.observers.RunObserver` or :class:`str`
        Where the experiments are stored. Either a Sacred observer, for
        example for a Mongo database, or the name of a directory, to
        use a file observer.
    ids : Sequence of :external:class:`int` or ``None``, default ``None``
        If not ``None``, return only experiments whose id is contained
        in the sequence.
    dataset_names : Sequence of :class:`str` or ``None``, default ``None``
        If not ``None``, return only experiments whose dataset names are
        contained in the sequence.
        The order of the names is also the one used for datasets when
        combining the results.
    estimator_names : Sequence of :class:`str` or ``None``, default ``None``
        If not ``None``, return only experiments whose estimator names are
        contained in the sequence.
        The order of the names is also the one used for estimators when
        combining the results.

    Returns
    -------
    info : :class:`ScoresInfo`
        Class containing information about experiments scores.

    See Also
    --------
    run_experiments
    fetch_scores

    """

    experiments = _get_experiments(
        storage=storage,
        ids=ids,
        dataset_names=dataset_names,
        estimator_names=estimator_names,
    )

    dict_experiments: Dict[
        str,
        Dict[str, Tuple[np.typing.NDArray[float], float, float]],
    ] = {}
    estimator_list = []
    dataset_list = []

    nobs = 0

    for experiment in experiments:
        estimator_name = experiment.config["estimator_name"]
        if estimator_name not in estimator_list:
            estimator_list.append(estimator_name)
        dataset_name = experiment.config["dataset_name"]
        if dataset_name not in dataset_list:
            dataset_list.append(dataset_name)
        scores = experiment.info.get("test_score", np.array([]))
        score_mean = experiment.info.get("score_mean", np.nan)
        score_std = experiment.info.get("score_std", np.nan)

        nobs = max(nobs, len(scores))

        assert np.isnan(score_mean) or score_mean == np.mean(scores)
        assert np.isnan(score_std) or score_std == np.std(scores)

        if estimator_name not in dict_experiments:
            dict_experiments[estimator_name] = {}

        if dataset_name in dict_experiments[estimator_name]:
            raise ValueError(
                f"Repeated experiment: ({estimator_name}, {dataset_name})",
            )

        dict_experiments[estimator_name][dataset_name] = (
            scores,
            score_mean,
            score_std,
        )

    estimator_names = (
        tuple(estimator_list) if estimator_names is None else estimator_names
    )
    dataset_names = (
        tuple(dataset_list) if dataset_names is None else dataset_names
    )
    matrix_shape = (len(dataset_names), len(estimator_names))

    scores = np.full(matrix_shape + (nobs,), np.nan)
    scores_mean = np.full(matrix_shape, np.nan)
    scores_std = np.full(matrix_shape, np.nan)

    for i, dataset_name in enumerate(dataset_names):
        for j, estimator_name in enumerate(estimator_names):
            dict_estimator = dict_experiments.get(estimator_name, {})
            s, mean, std = dict_estimator.get(
                dataset_name,
                (np.array([]), np.nan, np.nan),
            )
            if len(s) == nobs:
                scores[i, j] = s
            scores_mean[i, j] = mean
            scores_std[i, j] = std

    scores = np.array(scores.tolist())

    return ScoresInfo(
        dataset_names=dataset_names,
        estimator_names=estimator_names,
        scores=scores,
        scores_mean=scores_mean,
        scores_std=scores_std,
    )
