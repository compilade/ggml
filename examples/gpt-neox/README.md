# GPT-NeoX

## ⚠ This is a work-in-progress!!! For now, it does not quite work properly ⚠

I only tested this with Pythia-70m and Pythia-1b models, **and they don't work properly**, yet.

The first few words seem okay, ***but then it degrades***, so there is probably a very subtle mistake I made somewhere.

I'm looking for ways to debug my re-implementation of GPT-NeoX, but it might take me a few days, weeks, or months until I finally figure out what I did wrong.

# About [gpt-neox](https://github.com/EleutherAI/gpt-neox) models

The [gpt-neox](https://github.com/EleutherAI/gpt-neox) library not only supports the GPT-NeoX-20B model, but also the Pythia models, which are a bit better than the smaller OPT models.
I hope this implementation on `ggml` will also support all of the GPT-NeoX models, but this is not guaranteed at all.

Since I only have 8 GB of RAM on my laptop, I can't really run the GPT-NeoX-20B model, so I used the Pythia models, which are supposed to use the same model architecture.

# Build instructions

From the root of the `ggml` project,

```console
$ mkdir build
$ cd build
$ cmake ..
$ make -j4 gpt-neox
```

The binary should be built and available at `./bin/gpt-neox`.
