# Analysis Introduction

The analysis module is defined for the purpose of further analysis of metrics. Currently, edit-level attribution methods are implemented.

## Edit-level attribution

To improve the explainability of metrics that only evaluate at the sentence level, sentence-level scores are attributed to individual edits. The calculated edit-level scores help visualize how sentence-level metrics account for each edit.

```python
from gec_metrics import get_metric
from gec_metrics.analysis import get_attributor
import pprint

metric = get_metric('impara')()
att_cls = get_attributor('shapley')
attributor = att_cls(att_cls.Config(metric=metric))
src = 'Further more by these evidence u will agree'
hyp = 'Further more , with this evidence , you will agree .'
output = attributor.attribute(src=src, hyp=hyp)
pprint.pprint(output)
"""
AttributionOutput(sent_score=-0.027205130085349083,
                  src_score=0.027205130085349083,
                  attribution_scores=[0.06834515836089851,
                                      0.029289511901636915,
                                      0.12393246696641047,
                                      0.14501377101987606,
                                      -0.3611897512649497,
                                      -0.03259614178289969],
                  edits=[<errant.edit.Edit object at 0x7f4b87af2250>,
                         <errant.edit.Edit object at 0x7f4ba7d44790>,
                         <errant.edit.Edit object at 0x7f4b87af23d0>,
                         <errant.edit.Edit object at 0x7f4b87af2690>,
                         <errant.edit.Edit object at 0x7f4b87af2650>,
                         <errant.edit.Edit object at 0x7f4b87af2550>],
                  src='Further more by these evidence u will agree')
"""
```