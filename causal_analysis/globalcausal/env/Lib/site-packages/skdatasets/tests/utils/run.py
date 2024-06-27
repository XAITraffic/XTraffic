#! /usr/bin/env python

"""
@author: David Diaz Vico
@license: MIT
"""

import argparse
from sacred.observers import FileStorageObserver

from skdatasets import fetch
from skdatasets.utils.estimator import json2estimator
from skdatasets.utils.experiment import experiment


def main(dataset=fetch, estimator=json2estimator,
         observers=[FileStorageObserver('.results')]):
    parser = argparse.ArgumentParser(description='Run an experiment.')
    parser.add_argument('-r', '--repository', type=str, help='repository')
    parser.add_argument('-c', '--collection', type=str, default=None,
                        help='collection')
    parser.add_argument('-d', '--dataset', type=str, help='dataset')
    parser.add_argument('-e', '--estimator', type=str, help='estimator')
    args = parser.parse_args()
    e = experiment(dataset, estimator)
    e.observers.extend(observers)
    e.run(config_updates={'dataset': {'repository': args.repository,
                                      'collection': args.collection,
                                      'dataset': args.dataset},
                          'estimator': {'estimator': args.estimator}})


if __name__ == "__main__":
    main()
