import gec_metrics.metrics
from gec_metrics.metrics import MetricBase
import inspect

def get_metric_ids() -> list[str]:
    '''Generate a list of ids with the class name in lower case.
    '''
    metric_ids = [
        elem[0].lower() for elem in inspect.getmembers(gec_metrics.metrics, inspect.isclass) \
              if not elem[0].lower().startswith('metricbase')
    ]
    return metric_ids

def get_metric(metric_id: str) -> dict[str, MetricBase]:
    '''Generate a dictionary of ids and classes with the class name in lower case as the key.
    '''
    if not metric_id in get_metric_ids():
        raise ValueError(f'The metric_id should be {get_metric_ids()}.')
    metric_dict = {
        elem[0].lower(): elem[1] for elem in inspect.getmembers(gec_metrics.metrics, inspect.isclass) \
              if not elem[0].lower().startswith('metricbase')
    }
    return metric_dict[metric_id]