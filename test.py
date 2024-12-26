import os

import faulthandler
faulthandler.enable()


# from google.colab import userdata # `userdata` is a Colab API.

kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')

GEMMA_VARIANT = 'gemma2-2b-it' # @param ['gemma2-2b', 'gemma2-2b-it'] {type:"string"}

import kagglehub

GEMMA_PATH = kagglehub.model_download(f'google/gemma-2/flax/{GEMMA_VARIANT}')


print('GEMMA_PATH:', GEMMA_PATH)

CKPT_PATH = os.path.join(GEMMA_PATH, GEMMA_VARIANT)
TOKENIZER_PATH = os.path.join(GEMMA_PATH, 'tokenizer.model')
print('CKPT_PATH:', CKPT_PATH)
print('TOKENIZER_PATH:', TOKENIZER_PATH)

from gemma import params as params_lib

params = params_lib.load_and_format_params(CKPT_PATH)

import sentencepiece as spm

vocab = spm.SentencePieceProcessor()
vocab.Load(TOKENIZER_PATH)
print("Vocab size:", vocab.GetPieceSize())


from gemma import transformer as transformer_lib

transformer_config = transformer_lib.TransformerConfig.from_params(
    params=params,
    cache_size=1024
)

transformer = transformer_lib.Transformer(transformer_config)

print("Transformer ready!")

from gemma import sampler as sampler_lib

sampler = sampler_lib.Sampler(
    transformer=transformer,
    vocab=vocab,
    params=params['transformer'],
)

print("Sampler ready!")

prompt = [
    # "what is JAX in 3 bullet points?",
    "台灣的總人口數是多少？並列出統計年份。",
]

print("Prompting... %s" % prompt)

# sampler with verbose=True will print out the logits for each token.
reply = sampler(
    input_strings=prompt,
    # total_generation_steps=128,    
    total_generation_steps=32,
    )

print("Reply:", reply.text)


for input_string, out_string in zip(prompt, reply.text):
    print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")
