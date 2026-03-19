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
import std.conv   : to;
import std.string : fromStringz;
import core.stdc.locale : setlocale, LC_NUMERIC;
import core.stdc.stdio  : printf;

// ────────────────────────────────────────────────────────────────────────────

int main(string[] args) @trusted
{
    setlocale(LC_NUMERIC, "C");

    string modelPath;
    string mmprojPath;
    string imagePath;
    string prompt   = "Describe the image in detail.";
    int    ngl      = 99;
    int    nPredict = 512;
    int    nBatch   = 512;
    bool   useGpu   = true;

    for (int i = 1; i < cast(int) args.length; i++)
    {
        switch (args[i])
        {
            case "-m":
                if (++i < cast(int) args.length) modelPath   = args[i];
                else return printUsage(args[0]);
                break;
            case "--mmproj":
                if (++i < cast(int) args.length) mmprojPath  = args[i];
                else return printUsage(args[0]);
                break;
            case "-i":
                if (++i < cast(int) args.length) imagePath   = args[i];
                else return printUsage(args[0]);
                break;
            case "-n":
                if (++i < cast(int) args.length) nPredict    = args[i].to!int;
                else return printUsage(args[0]);
                break;
            case "-ngl":
                if (++i < cast(int) args.length) ngl         = args[i].to!int;
                else return printUsage(args[0]);
                break;
            case "--no-gpu":
                useGpu = false;
                break;
            default:
                prompt = args[i];
                break;
        }
    }

    if (modelPath.length == 0 || mmprojPath.length == 0)
        return printUsage(args[0]);

    // ── Backend + model ──────────────────────────────────────────────────────
    loadAllBackends();

    auto model = LlamaModel.loadFromFile(modelPath, ngl);
    if (!model)
    {
        stderr.writeln("error: failed to load language model '", modelPath, "'");
        return 1;
    }

    // ── Multimodal projector ─────────────────────────────────────────────────
    auto mparams          = mtmd_context_params_default();
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
    string marker     = fromStringz(mtmd_default_marker()).idup;
    string fullPrompt = haveImage ? marker ~ "\n" ~ prompt : prompt;

    // ── Tokenise ─────────────────────────────────────────────────────────────
    auto chunks   = InputChunks.create();
    auto inputTxt = mtmd_input_text(&fullPrompt[0], /*add_special=*/true,
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
        const(mtmd_bitmap)*[1] bitmapPtrs = [bitmapStore.ptr];
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
    uint nCtxNeeded = cast(uint)(chunks.nTokens + nPredict);
    auto ctx = LlamaContext.fromModel(model,
                   nCtxNeeded,
                   cast(uint) nBatch);
    if (!ctx)
    {
        stderr.writeln("error: failed to create llama_context");
        return 1;
    }

    // ── Eval all chunks (text + image embeddings) ────────────────────────────
    llama_pos nPast;
    if (auto err = mtmd.evalChunks(ctx.ptr, chunks, 0, /*seq_id=*/0,
                                   nBatch, /*logits_last=*/true, nPast))
    {
        stderr.writefln("error: chunk eval failed (code %d)", err);
        return 1;
    }

    // ── Generation ───────────────────────────────────────────────────────────
    const vocab = model.vocab;

    auto smpl = SamplerChain.create();
    smpl.temp(0.8f).topK(40).topP(0.95f).dist();

    llama_token[1] nextBuf;

    for (int gen = 0; gen < nPredict; gen++)
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
        "\nusage: %s -m model.gguf --mmproj mmproj.gguf\n" ~
        "          [-i image.jpg] [-n n_predict] [-ngl n_gpu_layers]\n" ~
        "          [--no-gpu] [prompt]\n\n" ~
        "  -m       path to the GGUF language model\n" ~
        "  --mmproj path to the multimodal projector GGUF\n" ~
        "  -i       input image or audio file (optional)\n" ~
        "  -n       number of tokens to predict (default: 512)\n" ~
        "  -ngl     number of GPU layers (default: 99)\n" ~
        "  --no-gpu run projector on CPU only\n" ~
        "  prompt   text prompt (default: 'Describe the image in detail.')\n\n",
        prog.ptr);
    return 1;
}
