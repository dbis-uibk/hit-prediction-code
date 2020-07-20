#!/usr/bin/env python
"""Command line tool to extract results as plots."""
import os
from pathlib import Path

import click
from dbispipeline.db import DB
from dbispipeline.db import DbModel
from dbispipeline.utils import load_project_config
import matplotlib.pyplot as plt
import pandas as pd

RESULT_DIR = 'results'


@click.group()
def cli():
    """Sets up the click command group."""
    pass


@cli.command()
@click.option(
    '--metric',
    '-m',
    default=None,
    type=str,
    help='Restricts the extraction to a single metric.',
)
@click.option(
    '--project-name',
    '-p',
    default=None,
    type=str,
    help='Change the project name.',
)
def epoch_plots(metric, project_name):
    """Extracts plots that show the performance over epochs.

    Args:
        metric: to print plots for.
        project_name: the project to extract results for.

    """
    if project_name is None:
        project_name = load_project_config()['project']['name']

    if metric is None:
        for results, run in get_cv_epoch_evaluator_results(
                project=project_name):
            for current_metric, result in results.items():
                if current_metric.startswith('neg_'):
                    result = result * -1

                file_name = _create_file_name(
                    project_name,
                    run.id,
                    current_metric,
                    'png',
                )
                _store_plot(result, current_metric, file_name)
    else:
        for result, run in get_cv_epoch_evaluator_results(
                project=project_name, requested_metric=metric):
            file_name = _create_file_name(project_name, run.id, metric, 'png')
            _store_plot(result, metric, file_name)


@cli.command()
@click.option(
    '--metric',
    '-m',
    default=None,
    type=str,
    help='Restricts the extraction to a single metric.',
)
@click.option(
    '--project-name',
    '-p',
    default=None,
    type=str,
    help='Change the project name.',
)
def csv_results(metric, project_name):
    """Extracts results as csv that show the performance over epochs.

    Args:
        metric: to print plots for.
        project_name: the project to extract results for.

    """
    if project_name is None:
        project_name = load_project_config()['project']['name']

    if metric is None:
        for results, run in get_cv_epoch_evaluator_results(
                project=project_name):
            for current_metric, result in results.items():
                file_name = _create_file_name(
                    project_name,
                    run.id,
                    current_metric,
                    'csv',
                )
                result.to_csv(file_name)
    else:
        for result, run in get_cv_epoch_evaluator_results(
                project=project_name, requested_metric=metric):
            file_name = _create_file_name(project_name, run.id, metric, 'csv')
            result.to_csv(file_name)


def _create_file_name(project_name, run_id, metric, file_ext):
    project_result_path = os.path.join(RESULT_DIR, project_name)
    Path(project_result_path).mkdir(parents=True, exist_ok=True)

    figure_name = str(run_id) + '_' + metric + '.' + file_ext
    return os.path.join(project_result_path, figure_name)


def _store_plot(data, title, file_name):
    try:
        data.plot(title=title)
        plt.savefig(file_name)
        plt.close()
    except TypeError as ex:
        print(ex)


def get_cv_epoch_evaluator_results(project=None, requested_metric=None):
    """Extracts CvEpochEvaluator results from the database.

    Args:
        project: the name of the project to extract results. If None, the
            project in the dbispipeline.ini is used.
        requested_metric: allows to restrict the results to a single metric.

    Returns: A tuple containing the prepared results as first element and the
        whole db entry as the second entry. The prepared results are eighter a
        pandas dataframe if a metric is requested or a dict containing a pandas
        dataframe per metric.
    """
    if project is None:
        project = load_project_config()['project']['name']

    session = DB.session()
    query = session.query(DbModel).order_by(
        DbModel.id).filter_by(project_name=project)

    for run in query:
        if run.evaluator['class'] == 'CvEpochEvaluator':
            outcome = pd.DataFrame(run.outcome)

            if requested_metric is None:
                results = {}
                for available_metric in run.evaluator['scoring']:
                    results[available_metric] = _extract_metric_results(
                        outcome,
                        available_metric,
                    )
                yield results, run
            elif requested_metric in run.evaluator['scoring']:
                yield _extract_metric_results(outcome, requested_metric), run


def _extract_metric_results(outcome, requested_metric):
    return outcome.apply(
        lambda row: row.apply(lambda value: value[requested_metric]))


if __name__ == '__main__':
    cli()
