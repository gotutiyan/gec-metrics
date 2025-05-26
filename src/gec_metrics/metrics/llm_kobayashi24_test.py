from .llm_kobayashi24 import LLMKobayashi24HFEdit
from .llm_kobayashi24 import LLMKobayashi24HFSent
from .llm_kobayashi24 import LLMKobayashi24OpenAIEdit
from .llm_kobayashi24 import LLMKobayashi24OpenAISent
import pytest

SRC = 'This sentences contain gramamtical error .'
HYP = 'This sentence contains a grammatical error .'
instruction_sent = '''The goal of this task is to rank the presented targets based on the quality of the sentences.
After reading the source sentence and target sentences, please assign a score from a minimum of 1 point to a maximum of 5 points to each target based on the quality of the sentence (note that you can assign the same score multiple times).

# source
This sentences contain gramamtical error .

# targets
This sentence contains a grammatical error .

# output format
The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":
```json
{
"target1_score": int // assigned score for target 1
...
"targetN_score": int // assigned score for target N
}
```
'''

instruction_edit = '''The goal of this task is to rank the presented targets based on the quality of the sentences.
After reading the source sentence and target sentences, please assign a score from a minimum of 1 point to a maximum of 5 points to each target based on the quality of the sentence (note that you can assign the same score multiple times).
For targets without any edits, if the sentence is correct, they will be awarded 5 points; if there is an error, they will receive 1 point.
The edits in each target are indicated as follows:
Insert "the": [→the]
Delete "the": [the→]
Replace "the" with "a": [the→a]

# context
This sentences contain gramamtical error .

# targets
[sentences→sentence] [contain→contains] [→a] [gramamtical→grammatical]

# output format
The output should be a markdown code snippet formatted in the
following schema, including the leading and trailing "```json" and "```":
```json
{
"target1_score": int // assigned score for target 1
...
"targetN_score": int // assigned score for target N
}
```'''

cases = [
    (LLMKobayashi24HFSent, instruction_sent),
    (LLMKobayashi24HFEdit, instruction_edit),
    (LLMKobayashi24OpenAISent, instruction_sent),
    (LLMKobayashi24OpenAIEdit, instruction_edit),
]

class TestLLMKobayashi24:
    @pytest.mark.parametrize("metric_cls,gold_instruction", cases)
    def test_metric(self, metric_cls, gold_instruction):
        print(f"{metric_cls=}")
        model = 'openai-community/gpt2'
        metric = metric_cls(metric_cls.Config(
            model=model
        ))
        template = metric.config.instruction_template[:]
        reformed_hyps = [metric.hyp_form(SRC, HYP)]
        instruction = template.replace(
            '[SOURCE]', SRC
        ).replace(
            '[CORRECTION]', '\n'.join(reformed_hyps)
        )
        print(instruction)
        print(gold_instruction)
        assert instruction.rstrip() == gold_instruction.rstrip()

    
    