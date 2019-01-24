import os
from pymongo import MongoClient


def _db():
    client = None

    if os.environ.get('MONGO_DB_SERVER') is None:
        client = MongoClient()
    else:
        client = MongoClient(os.environ.get('MONGO_DB_SERVER'))

    return client.mt


def _datasets():
    """
    Returns:
        (pymongo.collection): The datasets collection.
    """

    return _db().datasets


def _recommenders():
    """
    Returns:
        (pymongo.collection): The recommenders collection.
    """

    return _db().recommenders


def _results():
    """
    Returns:
        (pymongo.collection): The results collection.
    """

    return _db().results


def _cv_run():
    """
    Returns:
        (pymongo.collection): The cross-validation run collection.
    """

    return _db().cv_run


def load_dataset(name):
    """ Loads a dataset from the database.

    Args:
        name (str): name of the dataset.

    Returns:
        dict: the dataset in JSON format.

    Raises:
        LookupError: if the dataset does not exist.
    """

    dataset = _datasets().find_one({"name": name})

    if dataset is None:
        raise LookupError('Dataset does not exist.')

    return dataset


def store_dataset(dataset):
    """Stores a dataset in JSON format in the database.

    Args:
        dataset (dict): JSON format dataset object.

    Raises:
        LookupError: if the entry already exists.
    """

    result = _datasets().find_one({"name": dataset['name']})
    if result is not None:
        raise LookupError('Dataset does already exist.')

    _datasets().insert_one(dataset)


def load_recommender(name):
    """ Loads a recommender from the database.

    Args:
        name (str): name of the recommender.

    Returns:
        dict: the recommender config in JSON format.

    Raises:
        LookupError: if the recommender does not exist.
    """

    recommender = _recommenders().find_one({"name": name})

    if recommender is None:
        raise LookupError('Recommender does not exist.')

    return recommender


def store_recommender(recommender):
    """Stores a recommender in JSON format in the database.

    Args:
        recommender (dict): JSON format recommender object.

    Raises:
        LookupError: if the entry already exists.
    """

    result = _recommenders().find_one({"name": recommender['name']})
    if result is not None:
        raise LookupError('Recommender does already exist.')

    _recommenders().insert_one(recommender)


def store_result(result):
    """Stores a result in JSON format in the database.

    Args:
        result (dict): JSON format result object.
    """

    _results().insert_one(result)


def store_cv_run(run):
    """Stores a cross-validation run in JSON format in the database.

    Args:
        run (dict): JSON format cross-validation run object.
    """

    _cv_run().insert_one(run)


def list_datasets():
    """
    Returns:
        list: of datasets stored in the database.
    """

    return _datasets().find({}, {"name": 1, "info": 1})


def list_recommenders():
    """
    Returns:
        list: of recommenders stored in the database.
    """

    return _recommenders().find({}, {"name": 1})


def list_results():
    """
    Returns:
        list: of results.
    """

    results = _results().find(
        {},
        {
            "dataset_id": 1,
            "load_mode": 1,
            "recommender_id": 1
        }
    )

    result = []

    for r in results:
        dataset = _datasets().find_one(r['dataset_id'], {"name": 1})
        recsys = _recommenders().find_one(r['recommender_id'], {"name": 1})

        res = {
            "dataset": dataset,
            "load_mode": r['load_mode'],
            "recommender": recsys
        }

        result.append(res)

    return result


def get_recsys_by_id(recsys_id):
    """Gets the recommender system with the given object id from the DB.

    Args:
        recsys_id (ObjectId): The object id of the recommender system.

    Returns:
        dict: The recommender system if it is stored in the DB, otherwise None.
    """
    recsys = _recommenders().find_one({'_id': recsys_id})

    return recsys


def get_dataset_by_id(dataset_id):
    """Gets the dataset with the given object id from the DB.

    Args:
        dataset_id (ObjectId): The object id of the dataset.

    Returns:
        dict: The dataset if it is stored in the DB, otherwise None.
    """
    recsys = _datasets().find_one({'_id': dataset_id})

    return recsys


def get_dataset_by_name(dataset_name):
    """Gets the dataset with the given name from the DB.

    Args:
        dataset_id (str): The name of the dataset.

    Returns:
        dict: The dataset if it is stored in the DB, otherwise None.
    """
    recsys = _datasets().find_one({'name': dataset_name})

    return recsys
