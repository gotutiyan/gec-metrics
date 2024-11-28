from .metrics import METRIC_ID2CLS, MetricBase
from .meta_eval import METAEVAL_ID2CLS, MetaEvalBase

def get_metric_ids() -> list[str]:
    '''Generate a list of ids with the class name in lower case.
    '''
    ids = list(METRIC_ID2CLS.keys())
    return ids

def get_metric(name: str) -> dict[str, MetricBase]:
    '''Generate a dictionary of ids and classes with the class name in lower case as the key.
    '''
    if not name in get_metric_ids():
        raise ValueError(f'The id should be {get_metric_ids()}.')
    return METRIC_ID2CLS[name]

def get_meta_eval_ids() -> list[str]:
    '''Generate a list of ids with the class name in lower case.
    '''
    ids = list(METAEVAL_ID2CLS.keys())
    return ids

def get_meta_eval(name: str) -> dict[str, MetaEvalBase]:
    '''Generate a dictionary of ids and classes with the class name in lower case as the key.
    '''
    if not name in get_meta_eval_ids():
        raise ValueError(f'The id should be {get_meta_eval_ids()}.')
    return METAEVAL_ID2CLS[name]