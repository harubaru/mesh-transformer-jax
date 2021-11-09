import time

import jax
from jax.experimental import maps
import numpy as np
import optax
import transformers

from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

params = {
        "layers": 28,
        "d_model": 4096,
        "n_heads": 16,
        "n_vocab": 50400,
        "norm": "layernorm",
        "pe": "rotary",
        "pe_rotary_dims": 64,
        "seq": 2048,
        "cores_per_replica": 8,
        "per_replica_batch": 1,
}

def tpus_load(model_path="../step_15193/"):
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]
    seq = params["seq"]
    params["sampler"] = nucleaus_sample
    params["optimizer"] = optax.scale(0)

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)
    maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))

    network = CausalTransformer(params)
    network.state = read_ckpt(network.state, model_path, devices.shape[1])
    network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))
    
    return network

def tpus_infer(network, tokenizer, context, top_p=0.9, temp=0.55, top_k=140, gen_len=512):
    tokens = tokenizer.encode(context)
    provided_ctx = len(tokens)
    pad_amount = params["seq"] - provided_ctx
    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    total_batch = params["per_replica_batch"] * 8 // params["cores_per_replica"]
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

    output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(total_batch)*top_p, "temp": np.ones(total_batch)*temp, "top_k": np.ones(total_batch)*top_k})

    samples = []
    decoded_tokens = output[1][0]

    for o in decoded_tokens[:, :, 0]:
        samples.append(tokenizer.decode(o))

    return samples

#tokenizer = transformers.AutoTokenizer.from_pretrained('hakurei/lit-6B')

#net = tpus_load()
#print(tpus_infer(net, tokenizer, "[ Title: The Dunwich Horror; Author: H. P. Lovecraft; Genre: Horror ]\n***\nThe western"))
