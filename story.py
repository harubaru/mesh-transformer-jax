import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from dev_inference import tpus_load, tpus_infer

from transformers import (
        LogitsProcessorList,
        NoBadWordsLogitsProcessor,

        TemperatureLogitsWarper,
        TopPLogitsWarper,
        StoppingCriteriaList,
        MaxLengthCriteria
)

from warpers import (
        TailFreeSamplingLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        PhraseBiasProcessor
)

def init_model(args):
        args.input_stack = []
        args.bias = 0.0
        args.phrase_biases = []
        args.banned_phrases = []
        args.model = AutoModelForCausalLM.from_pretrained(args.model_name)
#        args.model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        args.model.config.pad_token_id = args.model.config.eos_token_id

def run_model(args, input_str):
        # Push the input_str into a list, and popping off the last members if it is past args.past_length

        phrase_bias_ids = []
        if args.phrase_biases:
                for phrase in args.phrase_biases:
                        phrase_bias_ids.append(args.tokenizer.encode(phrase))

        ban_ids = []
        if args.banned_phrases:
                for ban in args.banned_phrases:
                        ban_ids.append(args.tokenizer.encode(ban))

        input_ids = args.tokenizer.encode(input_str, return_tensors='pt')

        warpers = [
                TemperatureLogitsWarper(args.temperature),
                TopPLogitsWarper(args.top_p),
                TailFreeSamplingLogitsWarper(args.tfs)
        ]

        processors = [
                RepetitionPenaltyLogitsProcessor(penalty=args.rep_p, slope=args.rep_p_slope, penalize_last=2048)
        ]

        if args.phrase_biases:
                processors.append(PhraseBiasProcessor(phrase_bias_ids, args.bias))
        if args.banned_phrases:
                processors.append(NoBadWordsLogitsProcessor(ban_ids, None))

        stopping_criteria = StoppingCriteriaList([
                MaxLengthCriteria(args.output_length + len(input_ids[0]))
        ])

        logits_warper = LogitsProcessorList(warpers)
        logits_processor = LogitsProcessorList(processors)

        outputs = args.model.sample(input_ids=input_ids, logits_warper=logits_warper, logits_processor=logits_processor, stopping_criteria=stopping_criteria)

        result = args.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        if not result:
                return 'Please retry or alter your input. The model was unable to generate your request.'
                
        return result
