import torch
import json
import re
from transformers import AutoTokenizer
from dev_inference import tpus_load, tpus_infer

def init_model(args):
        args.input_stack = []
        args.bias = 0.0
        args.phrase_biases = []
        args.banned_phrases = []
        args.net = tpus_load()
        args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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

        result = tpus_infer(args.net, args.tokenizer, input_str, args.top_p, args.temperature, args.top_k, args.output_length)

        if not result:
                return 'Please retry or alter your input. The model was unable to generate your request.'
                
        return result[0]
