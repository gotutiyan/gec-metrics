from gec_metrics.metrics import MetricBaseForReferenceFree
from dataclasses import dataclass
from openai import OpenAI
from pydantic import BaseModel
import hashlib
import json
import os
import itertools
import random
from collections import Counter
from gecommon import CachedERRANT
import abc
import outlines
from transformers import BitsAndBytesConfig, AutoTokenizer

class LLMKobayashi24(MetricBaseForReferenceFree):
    @dataclass
    class Config(MetricBaseForReferenceFree.Config):
        model: str = None
        cache: str = None
        seed: int = 777
        verbose: bool = False
        criteria: str = None
        instruction_template: str = '''The goal of this task is to rank the presented targets based on the quality of the sentences.
After reading the source sentence and target sentences, please assign a score from a minimum of 1 point to a maximum of 5 points to each target based on the quality of the sentence (note that you can assign the same score multiple times).

# source
[SOURCE]

# targets
[CORRECTION]

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
    class LLMSentOutputFormat1(BaseModel):
        target_score1: int

    class LLMSentOutputFormat2(BaseModel):
        target_score1: int
        target_score2: int

    class LLMSentOutputFormat3(BaseModel):
        target_score1: int
        target_score2: int
        target_score3: int

    class LLMSentOutputFormat4(BaseModel):
        target_score1: int
        target_score2: int
        target_score3: int
        target_score4: int

    class LLMSentOutputFormat5(BaseModel):
        target_score1: int
        target_score2: int
        target_score3: int
        target_score4: int
        target_score5: int
    
    # The output format is different 
    #   depending on the number of corrected sentences.
    output_formats = [
        LLMSentOutputFormat1,
        LLMSentOutputFormat2,
        LLMSentOutputFormat3,
        LLMSentOutputFormat4,
        LLMSentOutputFormat5
    ]

    def __init__(self, config: Config = None):
        super().__init__(config)
        random.seed(self.config.seed)
        self.client = self.load_client()
        if self.config.cache is None:
            # Use model name as a cache name.
            self.config.cache = self.config.model.replace('/', '-') + '.cache'
        if os.path.exists(self.config.cache):                                                       
            self.cache = self.load_json(self.config.cache)
        else:
            self.cache = dict()
        assert '[SOURCE]' in self.config.instruction_template
        assert '[CORRECTION]' in self.config.instruction_template
        self.criteria2prompt = None
    
    def serialize(self, obj):
        if isinstance(obj, dict):
            return {k: self.serialize(v) for k, v in obj.items()}
        elif hasattr(obj, "__dict__"):
            return self.serialize(obj.__dict__)
        elif isinstance(obj, list):
            return [self.serialize(v) for v in obj]
        else:
            return obj
        
    def create_hash(self, prompt):                                                                                     
        return hashlib.md5(prompt.encode()).hexdigest()

    def load_json(self, file_name):
        data = dict()
        with open(file_name) as f:
            json_lines = f.readlines()
            for line in json_lines:
                json_obj = json.loads(line)
                data[json_obj['id']] = json_obj['results']
        return data

    def append_to_jsonl(self, file_name, data):
        with open(file_name, 'a') as file:
            json_str = json.dumps(data, ensure_ascii=False)
            file.write(json_str + '\n')

    def sample_sentences(
        self, hypotheses: list[str], max_n: int = 5
    ):
        '''Sample max_n sentences from hypotheses.
        LLMKobayashi24** metrics receives the hypotheses up to five.
        So if the number of distinct hypotheses is larger than five, we need to sample five sentences.
        In this implementation, we employ simple strategy: choose the five hypothesis from high frequency.
        '''
        hyp2freq = Counter(hypotheses)
        if len(hyp2freq) <= max_n:
            # Number of distinct hypotheses is less than max_n.
            hyps = [h for h in hyp2freq.keys()]
        else:
            # If larger than max_n, we choose high-frequency sentences.
            hyps = [e[0] for e in sorted(hyp2freq.items(), key=lambda x:x[1], reverse=True)][:max_n]
        # sorted() is used to fix the order of input.
        # Different evaluation results could be obtained in different orders,
        #   leading to a loss of reproducibility.
        return sorted(hyps)
    
    def index_multiple(
        self, elems: list, target_e
    ) -> list[int]:
        '''Multiple version of list.index()
        '''
        return [i for i, e in enumerate(elems) if e == target_e]
    
    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str]
    ):
        raise NotImplementedError(
            '''The LLM**Kobayashi24 does not support score_sentence.
            - This is because this method samples N sentences from the number of distinct hypotheses for each source,
                so not sampled sentences cannot be scored.
            - If you want to use rank_systems(), set aggregation="trueskill".
            - If you want to use corr_system() for the meta-evaluation, set aggregation="trueskill".
            '''
        )
    
    @abc.abstractmethod
    def load_client(self):
        '''This function loads LLM client, e.g. OpenAI() 
            or .from_pretrained() forHuggingface model.
        '''
        raise NotImplementedError()
    
    @abc.abstractmethod
    def call_client(self, instruction: str, output_format: BaseModel):
        '''Write forward scripts given instruction.
            You can refer output format.
        
        '''
        raise NotImplementedError

    def hyp_form(self, src: str, hyp: str) -> str:
        '''This is used for chaning format of the hypothsis, e.g., edit representation.
        Args:
            src (str): Source sentence.
            hyp (str): Hypothesis sentence.

        Return
            str: Another representation of the hypothesis.
        '''
        return hyp
    
    def score_pairwise(
        self,
        sources: list[str],
        hypotheses: list[list[str]],
    ):
        '''Calculate pairwise scores for all of combinations of hypotheses.
        By default, it simply compares the sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
                The shape is (num_sentences, )
            hypotheses (list[list[str]]): Corrected sentences.
                The shape is (num_systems, num_sentences).
            references (list[list[str]]): Reference sentences.
                The shape is (num_references, num_sentences).
        
        Returns:
            list[list[list]]: Pairwise comparison resutls.
                The shape is (num_sentences, num_systems, num_systems).
                Each element is -1, 0, or 1:
                    0 : tie
                    1 : sys_id1 wins sys_id2
                    -1: sys_id1 loses sys_id2
        '''
        pairwise_scores = []
        num_sents = len(sources)
        num_sys = len(hypotheses)
        template = self.config.instruction_template[:]
        if self.config.criteria:
            template = template.replace(
                "\n\n# context",
                f"\n{self.criteria2prompt[self.config.criteria]}\n\n# context"
            )
        for sent_id in range(num_sents):
            src = sources[sent_id]
            hyps = [hypotheses[sys_id][sent_id] for sys_id in range(num_sys)]
            sampled_hyps = self.sample_sentences(hyps, max_n=5)
            # In LLM-E evaluation, hyp_form() converts the sentence into an edit sequence.
            # (In LLM-S, self.hyp_form() returns hypotheses as is.)
            reformed_hyps = [self.hyp_form(src, h) for h in sampled_hyps]
            instruction = template.replace(
                '[SOURCE]', src
            ).replace(
                '[CORRECTION]', '\n'.join(reformed_hyps)
            )
            _hash = self.create_hash(instruction)
            if _hash not in self.cache:
                # forward the input to model.
                response = self.call_client(
                    instruction,
                    self.output_formats[len(sampled_hyps) - 1]
                )
                # To avoid call client twice for the same input,
                #    it caches the results.
                response = self.serialize(response)
                save_data = {'id': _hash, 'results': response}
                self.append_to_jsonl(self.config.cache, save_data)
                self.cache[_hash] = response
            else:
                # If the same input was already processed,
                #   it simply restores the results from the cache, thus no forwarding occurs.
                response = self.cache[_hash]
            scores_dict = json.loads(
                response['choices'][0]['message']['content']
            )
            scores = [scores_dict[f'target_score{i+1}'] for i in range(len(sampled_hyps))]
            assert len(sampled_hyps) == len(scores)
            
            # The computed scores are expands to the systems that have the same hypothesis.
            pairwise_table = [[None for _ in range(num_sys)] for _ in range(num_sys)]
            for i1, i2 in itertools.combinations(range(len(sampled_hyps)), 2):
                hyp1 = sampled_hyps[i1]  # Its score is scores[i1]
                hyp2 = sampled_hyps[i2]  # Its score is scores[i2]
                score = 0
                if scores[i1] > scores[i2]:
                    score = 1
                if scores[i1] < scores[i2]:
                    score = -1
                # index_multiple() returns the index of the system with the same output as hyp1 (or hyp2).
                for hyp1_idx in self.index_multiple(hyps, hyp1):
                    for hyp2_idx in self.index_multiple(hyps, hyp2):
                        pairwise_table[hyp1_idx][hyp2_idx] = score
                        pairwise_table[hyp2_idx][hyp1_idx] = -score
            # Record a tie between systems with the same corrected sentences.
            for i in range(len(sampled_hyps)):
                hyp = sampled_hyps[i]
                ids = self.index_multiple(hyps, hyp)
                for hyp1_idx in ids:
                    for hyp2_idx in ids:
                        pairwise_table[hyp1_idx][hyp2_idx] = 0
                        pairwise_table[hyp2_idx][hyp1_idx] = 0
            pairwise_scores.append(pairwise_table)
        return pairwise_scores
        

class LLMKobayashi24OpenAISent(LLMKobayashi24):
    '''LLM-S with OpenAI models.'''
    @dataclass
    class Config(LLMKobayashi24.Config):
        '''OpenAI configuration
            - model (str): model name.
            - organization (str): Your organization key.
            - api_key (str): Your api key.
            - base_url (str): You can igonore this when using OpenAI model.
                When using Gemini models, specify an appropriate url.
        '''
        model: str = 'gpt-4o-mini-2024-07-18'
        organization: str = None
        api_key: str = None
        base_url: str = None
        
    def __init__(self, config: Config = None):
        super().__init__(config)
        self.criteria2prompt = {
            "grammaticality": "Please evaluate each target with a focus on the grammaticality of the sentence.",
            "fluency": "Please evaluate each target with a focus on the fluency of the sentence.",
            "meaning": "Please evaluate each target with a focus on preserving the meaning between each target and the source, which is the middle sentence in the context."
        }
        
    def load_client(self):
        return OpenAI(
            organization=self.config.organization,
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
    def call_client(self, instruction, response_format):
        # raise ValueError('do not call')
        additonal_config = {}
        if 'gemini' not in self.config.model:
            additonal_config['seed'] = self.config.seed
        return self.client.beta.chat.completions.parse(                                                                            
            model=self.config.model,
            messages=[                                         
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ],
            response_format=response_format,
            **additonal_config
        )

class LLMKobayashi24OpenAIEdit(LLMKobayashi24OpenAISent):
    '''LLM-E with OpenAI models.'''
    @dataclass
    class Config(LLMKobayashi24OpenAISent.Config):
        instruction_template: str = '''The goal of this task is to rank the presented targets based on the quality of the sentences.
After reading the source sentence and target sentences, please assign a score from a minimum of 1 point to a maximum of 5 points to each target based on the quality of the sentence (note that you can assign the same score multiple times).
For targets without any edits, if the sentence is correct, they will be awarded 5 points; if there is an error, they will receive 1 point.
The edits in each target are indicated as follows:
Insert "the": [→the]
Delete "the": [the→]
Replace "the" with "a": [the→a]

# context
[SOURCE]

# targets
[CORRECTION]

# output format
The output should be a markdown code snippet formatted in the
following schema, including the leading and trailing "```json" and "```":
```json
{
"target1_score": int // assigned score for target 1
...
"targetN_score": int // assigned score for target N
}
```
'''
    def __init__(self, config: Config = None):
        super().__init__(config)
        self.criteria2prompt = {
            "difficulty": "Please evaluate each edit in the target with a focus on the difficulty of corrections.",
            "impact": "Please evaluate each edit in the target with a focus on its impact on the sentence.",
        }
        self.errant = CachedERRANT()

    def hyp_form(self, src: str, hyp: str) -> str:
        '''This is used for chaning format of the hypothsis, e.g., edit representation.
        Args:
            src (str): Source sentence.
            hyp (str): Hypothesis sentence.

        Return
            str: Another representation of the hypothesis.
        '''
        edits = self.errant.extract_edits(src, hyp)
        return ' '.join([f"{e.o_str}→{e.c_str}" for e in edits])

class LLMKobayashi24HFSent(LLMKobayashi24):
    '''LLM-S with huggingface models.'''
    @dataclass
    class Config(LLMKobayashi24.Config):
        model: str = 'meta-llama/Llama-2-13b-chat-hf'

    def __init__(self, config = None):
        super().__init__(config)
        self.criteria2prompt = {
            "grammaticality": "Please evaluate each target with a focus on the grammaticality of the sentence.",
            "fluency": "Please evaluate each target with a focus on the fluency of the sentence.",
            "meaning": "Please evaluate each target with a focus on preserving the meaning between each target and the source, which is the middle sentence in the context."
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)

    def load_client(self):
        return outlines.models.transformers(
            self.config.model,
            device='auto',
            model_kwargs={
                'quantization_config': BitsAndBytesConfig(
                    # load_in_4bit=True,
                    load_in_8bit=True,
                )
            }
        )
    
    def call_client(self, instruction, response_format):
        try:
            instruction = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False
            )
        except ValueError:
            # The model does not have chat template.
            pass
        generator = outlines.generate.json(self.client, response_format)
        output = generator(instruction, seed=self.config.seed)
        # Use the same output format as OpenAI.
        return {'choices': [
            {'message': {'content': output.json()}}
        ]}

class LLMKobayashi24HFEdit(LLMKobayashi24HFSent):
    '''LLM-E with huggingface models.'''
    @dataclass
    class Config(LLMKobayashi24HFSent.Config):
        instruction_template: str = '''The goal of this task is to rank the presented targets based on the quality of the sentences.
After reading the source sentence and target sentences, please assign a score from a minimum of 1 point to a maximum of 5 points to each target based on the quality of the sentence (note that you can assign the same score multiple times).
For targets without any edits, if the sentence is correct, they will be awarded 5 points; if there is an error, they will receive 1 point.
The edits in each target are indicated as follows:
Insert "the": [→the]
Delete "the": [the→]
Replace "the" with "a": [the→a]

# context
[SOURCE]

# targets
[CORRECTION]

# output format
The output should be a markdown code snippet formatted in the
following schema, including the leading and trailing "```json" and "```":
```json
{
"target1_score": int // assigned score for target 1
...
"targetN_score": int // assigned score for target N
}
```
'''
    def __init__(self, config: Config = None):
        super().__init__(config)
        self.criteria2prompt = {
            "difficulty": "Please evaluate each edit in the target with a focus on the difficulty of corrections.",
            "impact": "Please evaluate each edit in the target with a focus on its impact on the sentence.",
        }
        self.errant = CachedERRANT()
    
    def hyp_form(self, src: str, hyp: str) -> str:
        '''Convert hypothesis sentence into edit sequence.
        Args:
            src (str): Source sentence.
            hyp (str): Hypothesis sentence.

        Return
            str: Another representation of the hypothesis.
        '''
        edits = self.errant.extract_edits(src, hyp)
        return ' '.join([f"{e.o_str}→{e.c_str}" for e in edits])