import gec_metrics
import inspect

def main():
    metric_ids = gec_metrics.get_metric_ids()
    for metric_id in metric_ids:
        cls = gec_metrics.get_metric(metric_id)
        print(f'{metric_id}:')
        config = cls.Config()
        for attr in config.__dict__:
            print(f' {attr}: {config.__dict__[attr]}')
        print()

if __name__ == '__main__':
    main()