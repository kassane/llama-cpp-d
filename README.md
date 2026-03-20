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

## Tools

### hf-download

List and download GGUF files from HuggingFace Hub:

```sh
cd tools && dub build --build=release

# List available .gguf files in a repository
./build/hf-download -r unsloth/Qwen3.5-0.8B-GGUF

# Download a specific file
./build/hf-download -r unsloth/Qwen3.5-0.8B-GGUF -f Qwen3.5-0.8B-Q4_K_M.gguf -o ~/models

# With authentication (private repos / higher rate limits)
HF_TOKEN=hf_xxx ./build/hf-download -r myorg/mymodel -f model.gguf
```

| Flag | Description |
|------|-------------|
| `-r owner/repo` | HuggingFace repository (required) |
| `-f filename` | File to download; omit to list `.gguf` files |
| `-o outdir` | Output directory (default: `.`) |
| `-t token` | HF access token (or `HF_TOKEN` env var) |

## Examples

```sh
# Text completion
dub run :simple -- -m model.gguf -n 64 "Tell me a joke"

# Tokenization inspector
dub run :tokenize -- -m model.gguf -s "Hello, world!"

# Sentence embeddings (cosine similarity between prompts)
dub run :embedding -- -m model.gguf
dub run :embedding -- -m model.gguf -p "custom sentence"

# Context state save/load (verifies two runs produce identical output)
dub run :save-load-state -- -m model.gguf -n 32

# Multimodal (vision/audio) — text only
dub run :multimodal -c default -- -m model.gguf --mmproj mmproj.gguf -n 200 "Describe this."

# Multimodal with an image
dub run :multimodal -c default -- -m model.gguf --mmproj mmproj.gguf -i photo.jpg "What do you see?"
```

| Example | Required flags | Optional flags |
|---------|----------------|----------------|
| `simple` | `-m <path>` | `-n <tokens>` (default 32), `-ngl <gpu-layers>` (default 99) |
| `tokenize` | `-m <path>` | `-s` include BOS/EOS |
| `embedding` | `-m <path>` | `-p <text>`, `-ngl` (default 99) |
| `save-load-state` | `-m <path>` | `-n <tokens>` (default 16), `-ngl`, `--state-file <path>` |
| `multimodal` | `-m <path>`, `--mmproj <path>` | `-i <image>`, `-n <tokens>` (default 512), `-ngl` (default 99), `--no-gpu` |

### Configurations

| Config | Description |
|--------|-------------|
| `default` | CPU only |
| `mtmd` | CPU multimodal (llama + libmtmd) |
| `cuda` | CUDA GPU acceleration |
| `vulkan` | Vulkan GPU acceleration |
| `metal` | Apple Metal (macOS) |
| `hipblas` | AMD ROCm/HIP |
| `openblas` | OpenBLAS |
| `openmp` | OpenMP threading |
| `sycl` | Intel oneAPI SYCL |

## Quick start

### Text completion

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

### Multimodal (vision/audio)

```d
import llama;

void main() @trusted
{
    loadAllBackends();

    auto model = LlamaModel.loadFromFile("model.gguf", 99);
    assert(model);

    auto mparams = mtmd_context_params_default();
    mparams.use_gpu = true;

    auto mtmd = MtmdContext.initFromFile("mmproj.gguf", model.ptr, mparams);
    assert(mtmd);

    // Load an image (or skip for text-only)
    auto bitmap = mtmd.loadBitmap("photo.jpg");
    assert(bitmap);

    import std.string : fromStringz;
    string marker    = fromStringz(mtmd_default_marker()).idup;
    string prompt    = marker ~ "\nDescribe the image.";
    auto   chunks    = InputChunks.create();
    auto   inputTxt  = mtmd_input_text(&prompt[0], true, true);
    const(mtmd_bitmap)*[1] bitmaps = [bitmap.ptr];
    mtmd.tokenize(chunks, inputTxt, bitmaps[]);

    auto ctx = LlamaContext.fromModel(model,
                   cast(uint)(chunks.nTokens + 256),
                   512);
    assert(ctx);

    llama_pos nPast;
    mtmd.evalChunks(ctx.ptr, chunks, 0, 0, 512, true, nPast);

    auto smpl = SamplerChain.create();
    smpl.temp(0.8f).topK(40).topP(0.95f).dist();

    // Generation loop
    llama_token[1] buf;
    foreach (i; 0 .. 256)
    {
        auto tok = smpl.sample(ctx);
        if (isEog(model.vocab, tok)) break;
        import std.stdio : write;
        write(tokenToString(model.vocab, tok));
        smpl.accept(tok);
        buf[0] = tok;
        ctx.decode(batchGetOne(buf[]));
    }
}
```

## License

[MIT](./LICENSE)
