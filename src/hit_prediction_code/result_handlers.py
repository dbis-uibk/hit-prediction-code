"""This module contains result handlers for the dbispipeline."""
import json
from typing import Any

from logzero import logger


def print_results_as_json(results: Any, **kwargs: Any) -> None:
    """Prints the results as a json string.

    Args:
        results (Any): the results that should be printed.
    """
    logger.info('Results as JSON:')
    print(json.dumps(results, **kwargs))
