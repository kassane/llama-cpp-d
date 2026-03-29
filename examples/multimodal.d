/++
Multimodal inference CLI — feed an image (and optional text) to a vision model.

Usage:
---
multimodal -m model.gguf --mmproj mmproj.gguf \
           [-i image.jpg] [-n n_predict] [-ngl n_gpu_layers] [prompt]
---

The language model and projector must be compatible (same architecture).
If no image is supplied the tool behaves like a plain text-completion CLI.
+/
module multimodal;

import llama;
import llama.mtmd;
import std.stdio  : write, writeln, writefln, stderr;
import std.string : fromStringz;
import core.stdc.locale : setlocale, LC_NUMERIC;
import core.stdc.stdio  : printf;

// ────────────────────────────────────────────────────────────────────────────

int main(string[] args) @trusted
{
    setlocale(LC_NUMERIC, "C");

    ModelConfig    mcfg;
    SamplingConfig scfg;

    mcfg.nPredict = 512;
    mcfg.prompt   = "Describe the image in detail.";

    if (!parseConfig(mcfg, args, "model options:") ||
        !parseConfig(scfg, args, "sampling options:"))
        return 0;

    // Example-specific flags not covered by ModelConfig.
    string mmprojPath;
    string imagePath;
    bool   useGpu = true;

    for (int i = 1; i < cast(int) args.length; )
    {
        if (args[i] == "--mmproj" && i + 1 < cast(int) args.length)
        {
            mmprojPath = args[i + 1];
            args = args[0 .. i] ~ args[i + 2 .. $];
        }
        else if (args[i] == "-i" && i + 1 < cast(int) args.length)
        {
            imagePath = args[i + 1];
            args = args[0 .. i] ~ args[i + 2 .. $];
        }
        else if (args[i] == "--no-gpu")
        {
            useGpu = false;
            args = args[0 .. i] ~ args[i + 1 .. $];
        }
        else
            i++;
    }

    // Remaining positional arg overrides the prompt.
    if (args.length > 1) mcfg.prompt = args[1];

    if (mcfg.modelPath.length == 0 || mmprojPath.length == 0)
        return printUsage(args[0]);

    // ── Backend + model ──────────────────────────────────────────────────────
    loadAllBackends();

    auto model = LlamaModel.loadFromFile(mcfg.modelPath, mcfg.nGpuLayers);
    if (!model)
    {
        stderr.writeln("error: failed to load language model '", mcfg.modelPath, "'");
        return 1;
    }

    // ── Multimodal projector ─────────────────────────────────────────────────
    auto mparams          = MtmdContext.params();
    mparams.use_gpu       = useGpu;
    mparams.print_timings = true;

    auto mtmd = MtmdContext.initFromFile(mmprojPath, model.ptr, mparams);
    if (!mtmd)
    {
        stderr.writeln("error: failed to load mmproj '", mmprojPath, "'");
        return 1;
    }

    // ── Build prompt with optional media marker ──────────────────────────────
    bool   haveImage  = imagePath.length > 0;
    string marker     = fromStringz(defaultMarker).idup;
    string fullPrompt = haveImage ? marker ~ "\n" ~ mcfg.prompt : mcfg.prompt;

    // ── Tokenise ─────────────────────────────────────────────────────────────
    auto chunks       = InputChunks.create();
    auto inputTxt     = inputText(&fullPrompt[0], /*add_special=*/true,
                                                        /*parse_special=*/true);

    if (haveImage)
    {
        // MtmdBitmap is non-copyable; it stays alive until after tokenize().
        auto bitmapStore = mtmd.loadBitmap(imagePath);
        if (!bitmapStore)
        {
            stderr.writeln("error: failed to load image '", imagePath, "'");
            return 1;
        }
        BitmapPtr bitmapPtrs = [bitmapStore.ptr];
        if (auto err = mtmd.tokenize(chunks, inputTxt, bitmapPtrs[]))
        {
            stderr.writefln("error: mtmd tokenization failed (code %d)", err);
            return 1;
        }
    }
    else
    {
        if (auto err = mtmd.tokenize(chunks, inputTxt))
        {
            stderr.writefln("error: mtmd tokenization failed (code %d)", err);
            return 1;
        }
    }

    // ── Context ──────────────────────────────────────────────────────────────
    uint nCtxNeeded = cast(uint)(chunks.nTokens + mcfg.nPredict);
    auto ctx = LlamaContext.fromModel(model,
                   nCtxNeeded,
                   mcfg.nBatch);
    if (!ctx)
    {
        stderr.writeln("error: failed to create llama_context");
        return 1;
    }

    // ── Eval all chunks (text + image embeddings) ────────────────────────────
    llama_pos nPast;
    if (auto err = mtmd.evalChunks(ctx.ptr, chunks, 0, /*seq_id=*/0,
                                   mcfg.nBatch, /*logits_last=*/true, nPast))
    {
        stderr.writefln("error: chunk eval failed (code %d)", err);
        return 1;
    }

    // ── Generation ───────────────────────────────────────────────────────────
    const vocab = model.vocab;

    auto smpl = buildSamplerChain(scfg);

    llama_token[1] nextBuf;

    for (int gen = 0; gen < mcfg.nPredict; gen++)
    {
        auto tok = smpl.sample(ctx);
        if (isEog(vocab, tok)) break;

        write(tokenToString(vocab, tok));

        import std.stdio : stdout;
        stdout.flush();
        smpl.accept(tok);
        nextBuf[0] = tok;
        auto batch = batchGetOne(nextBuf[]);
        if (ctx.decode(batch))
        {
            stderr.writeln("error: decode failed");
            break;
        }
        nPast++;
    }

    writeln();

    ctx.printPerf();
    smpl.printPerf();
    writeln();

    return 0;
}

// ────────────────────────────────────────────────────────────────────────────

int printUsage(string prog) @trusted nothrow
{
    printf(
        "\nusage: %s -m model.gguf --mmproj mmproj.gguf [options] [prompt]\n\n" ~
        "  -m        path to the GGUF language model\n" ~
        "  --mmproj  path to the multimodal projector GGUF\n" ~
        "  -i        input image or audio file (optional)\n" ~
        "  -ngl      GPU layers (default: 99)\n" ~
        "  -n        tokens to predict (default: 512)\n" ~
        "  -b        batch size (default: 512)\n" ~
        "  -t        temperature (default: 0.8; 0 = greedy)\n" ~
        "  -k        top-K (default: 40)\n" ~
        "  --top-p   top-P nucleus (default: 0.95)\n" ~
        "  --no-gpu  run projector on CPU only\n" ~
        "  prompt    text prompt (default: 'Describe the image in detail.')\n\n",
        prog.ptr);
    return 1;
}
