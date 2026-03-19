# llama-cpp-d

[![CI Build](https://github.com/kassane/llama-cpp-d/actions/workflows/ci.yml/badge.svg)](https://github.com/kassane/llama-cpp-d/actions/workflows/ci.yml)
![Latest release](https://img.shields.io/github/v/release/kassane/llama-cpp-d?include_prereleases&label=latest)
[![Static Badge](https://img.shields.io/badge/v2.111.0%20(stable)-f8240e?logo=d&logoColor=f8240e&label=frontend)](https://dlang.org/download.html)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kassane/llama-cpp-d)

D bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Requirements

| Tool | Minimum |
|------|---------|
| LDC or DMD | ≥ 2.111 (`importC` required) |
| CMake | ≥ 3.14 |
| C++17 compiler | GCC / Clang / MSVC |

## How to use

```sh
dub add llama-cpp-d
```

## Examples

```sh
# Text completion
dub run :simple -- -m model.gguf -n 64 "Tell me a joke"

# Tokenization inspector
dub run :tokenize -- -m model.gguf -s "Hello, world!"
```

Flags for `simple`: `-m <path>` (required), `-n <tokens>` (default 32), `-ngl <gpu-layers>` (default 99).
Flags for `tokenize`: `-m <path>` (required), `-s` (include BOS/EOS special tokens).

## Quick start

```d
import llama;

void main()
{
    loadAllBackends();

    // D-string overload; second arg is GPU layer count (0 = CPU only)
    auto model = LlamaModel.loadFromFile("model.gguf", 99);
    assert(model);

    // Context window = model default; batch size = number of prompt tokens
    auto tokens = tokenize(model.vocab, "Hello");
    auto ctx    = LlamaContext.fromModel(model,
                      cast(uint) tokens.length + 32,  // nCtx
                      cast(uint) tokens.length);       // nBatch
    assert(ctx);

    // Two-statement form: SamplerChain is non-copyable, so no chaining on init
    auto smpl = SamplerChain.create();
    smpl.topK(40).topP(0.9f).temp(0.8f).dist();

    auto batch = batchGetOne(tokens);
    ctx.decode(batch);

    auto next = smpl.sample(ctx); // samples from the last output position
}
```

## License

[MIT](./LICENSE)
