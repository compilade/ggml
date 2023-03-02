# Convert GPT NeoX transformer model to ggml format
#
# Load the model using GPTNeoXForCausalLM.
# Iterate over all variables and write them to a binary file.
#
# For each variable, write the following:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (float[n_dims])
#
# By default, the bigger matrices are converted to 16-bit floats.
# This can be disabled by adding the "use-f32" CLI argument.
#
# At the start of the ggml file we write the model parameters
# and vocabulary.
#

import sys
import struct
import json
import torch
import numpy as np

from transformers import GPTNeoXForCausalLM

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# gpt-neox's added_tokens contains ASCII space characters, so instead of failing,
# simply return back the key (0x20 in the above example)
class ByteDecoder(dict):
    def __missing__(self, key):
        return ord(key)

if len(sys.argv) < 2:
    print("Usage: convert-hf-to-ggml.py dir-model [use-f32]\n")
    sys.exit(1)

# output in the same directory as the model
dir_model = sys.argv[1]
fname_out = sys.argv[1] + "/ggml-model.bin"

with open(dir_model + "/tokenizer.json", "r") as f:
    tokenizer = json.load(f)

with open(dir_model + "/config.json", "r") as f:
    hparams = json.load(f)

encoder = tokenizer["model"]["vocab"]
encoder_added = tokenizer["added_tokens"]

# use 16-bit or 32-bit floats
use_f16 = True
if len(sys.argv) > 2:
    use_f16 = False
    fname_out = sys.argv[1] + "/ggml-model-f32.bin"

model = GPTNeoXForCausalLM.from_pretrained(dir_model, low_cpu_mem_usage=True)
#print (model)

list_vars = model.state_dict()
#print (list_vars)

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["max_position_embeddings"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["num_attention_heads"]))
fout.write(struct.pack("i", hparams["num_hidden_layers"]))
# See https://github.com/huggingface/transformers/blob/v4.22.2/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L84-L85
# This is the equivalent of n_rot in GPT-J, probably
fout.write(struct.pack("i", round((hparams["hidden_size"] // hparams["num_attention_heads"]) * hparams["rotary_pct"])))
fout.write(struct.pack("i", use_f16))

byte_encoder = bytes_to_unicode()
byte_decoder = ByteDecoder({v:k for k, v in byte_encoder.items()})

# NOTE: encoder_added's length is not actually its length, since it also contains tokens from encoder
# So here, count the number of items in encoder_added with an "id" greater than len(encoder)
fout.write(struct.pack("i", len(encoder) + sum(tok["id"] >= len(encoder) for tok in encoder_added)))

for key in encoder:
    text = bytearray([byte_decoder[c] for c in key])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

for token in encoder_added:
    # Skip tokens that were already seen
    # I don't know why added_tokens has duplicates of model.vocab in gpt-neox's tokenizer.json
    if token["id"] < len(encoder):
        continue
    text = bytearray([byte_decoder[c] for c in token["content"]])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape)

    # we don't need these
    if name.endswith(".attention.masked_bias") or \
       name.endswith(".attention.bias") or \
       name.endswith(".attention.rotary_emb.inv_freq"):
        print("  Skipping variable: " + name)
        continue

    n_dims = len(data.shape);

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype = 0;
    if use_f16:
        if name[-7:] == ".weight" and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype = 0

    # for efficiency - transpose these matrices:
    #  "transformer.h.*.mlp.fc_in.weight
    #  "transformer.h.*.attn.out_proj.weight
    #  "transformer.h.*.attn.q_proj.weight"
    #  "transformer.h.*.attn.k_proj.weight"
    #  "transformer.h.*.attn.v_proj.weight"
    """if name.endswith(".mlp.fc_in.weight")     or \
       name.endswith(".attn.out_proj.weight") or \
       name.endswith(".attn.q_proj.weight")   or \
       name.endswith(".attn.k_proj.weight")   or \
       name.endswith(".attn.v_proj.weight"):
        print("  Transposing")
        data = data.transpose()"""

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str);

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
