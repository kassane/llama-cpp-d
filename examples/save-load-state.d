/++
Demonstrate context-state save and load for reproducible generation.

Encodes a prompt into context 1, saves the state to a file, then restores
it in context 2 and generates the same token sequence — verifying that both
runs match.

Usage: `save-load-state -m model.gguf [-n n_predict] [-ngl n_gpu_layers]
                        [-s state_file] [prompt]`
+/
module save_load_state;

import llama;
import std.stdio  : writef, writefln, writeln, stderr;
import std.conv   : to;
import core.stdc.locale : setlocale, LC_NUMERIC;
import core.stdc.stdio  : printf;

int main(string[] args)
{
    setlocale(LC_NUMERIC, "C");

    string modelPath;
    string prompt    = "The quick brown fox";
    string stateFile = "dump_state.bin";
    int    ngl       = 0;
    int    nPredict  = 16;

    for (int i = 1; i < cast(int) args.length; i++)
    {
        switch (args[i])
        {
            case "-m":
                if (++i < cast(int) args.length) modelPath = args[i];
                else return printUsage(args[0]);
                break;
            case "-n":
                if (++i < cast(int) args.length) nPredict = args[i].to!int;
                else return printUsage(args[0]);
                break;
            case "-ngl":
                if (++i < cast(int) args.length) ngl = args[i].to!int;
                else return printUsage(args[0]);
                break;
            case "-s":
                if (++i < cast(int) args.length) stateFile = args[i];
                else return printUsage(args[0]);
                break;
            default:
                prompt = args[i];
                break;
        }
    }
    if (modelPath.length == 0) return printUsage(args[0]);

    loadAllBackends();

    auto model = LlamaModel.loadFromFile(modelPath, ngl);
    if (!model)
    {
        stderr.writefln("error: failed to load model '%s'", modelPath);
        return 1;
    }

    const vocab  = model.vocab;
    int   nBatch = 512;

    // ── Context 1: encode prompt + save state ────────────────────────────────
    // kv_unified required for recurrent (SSM/Mamba) models when n_seq_max=1.
    auto cp1       = contextParams(cast(uint)(nBatch + nPredict + 1), cast(uint) nBatch);
    cp1.kv_unified = true;
    auto ctx1 = LlamaContext.fromModel(model, cp1);
    if (!ctx1)
    {
        stderr.writeln("error: failed to create context 1");
        return 1;
    }

    auto tokens = tokenize(vocab, prompt, /*addSpecial=*/true);
    if (tokens is null)
    {
        stderr.writeln("error: tokenization failed");
        return 1;
    }

    // Decode all but the last token; save state BEFORE the final token so that
    // both runs can decode it from the same recurrent state (important for
    // hybrid SSM/Mamba models where replaying at a new position changes state).
    if (tokens.length > 1)
    {
        auto ob = allocBatch(cast(int)(tokens.length - 1));
        foreach (i, tok; tokens[0 .. $ - 1])
            batchAdd(ob.get(), tok, cast(llama_pos) i, 0, false);

        if (ctx1.decode(ob.get()))
        {
            stderr.writeln("error: prompt pre-decode failed");
            return 1;
        }
    }

    if (!ctx1.stateSaveFile(stateFile, tokens[0 .. $ - 1]))
    {
        stderr.writeln("error: stateSaveFile failed");
        return 1;
    }
    stderr.writefln("state saved to '%s'", stateFile);

    // Decode the final prompt token to get generation-ready logits.
    {
        auto ob = allocBatch(1);
        batchAdd(ob.get(), tokens[$ - 1], cast(llama_pos)(tokens.length - 1), 0, true);
        if (ctx1.decode(ob.get()))
        {
            stderr.writeln("error: last-token decode failed");
            return 1;
        }
    }

    // ── First generation run (ctx1) ──────────────────────────────────────────
    auto smpl1 = SamplerChain.create();
    smpl1.dist(1234u);

    string result1;
    {
        writef("\nfirst run: %s", prompt);

        // nPast = tokens.length because we decoded all prompt tokens above.
        llama_pos nPast = cast(llama_pos) tokens.length;
        auto ob = allocBatch(1);

        for (int g; g < nPredict; g++)
        {
            auto tok = smpl1.sample(ctx1);
            if (isEog(vocab, tok)) break;

            string piece = tokenToString(vocab, tok);
            writef("%s", piece);
            result1 ~= piece;

            smpl1.accept(tok);
            batchClear(ob.get());
            batchAdd(ob.get(), tok, nPast, 0, true);

            if (ctx1.decode(ob.get()))
            {
                stderr.writeln("\nerror: decode failed");
                return 1;
            }
            nPast++;
        }
        writeln("\n");
    }

    // ── Context 2: restore state + re-generate ───────────────────────────────
    auto cp2       = contextParams(cast(uint)(nBatch + nPredict + 1), cast(uint) nBatch);
    cp2.kv_unified = true;
    auto ctx2 = LlamaContext.fromModel(model, cp2);
    if (!ctx2)
    {
        stderr.writeln("error: failed to create context 2");
        return 1;
    }

    auto smpl2 = SamplerChain.create();
    smpl2.dist(1234u);

    string result2;
    {
        auto sessionToks = new llama_token[](tokens.length);
        size_t nLoaded;

        if (!ctx2.stateLoadFile(stateFile, sessionToks, &nLoaded))
        {
            stderr.writeln("error: stateLoadFile failed");
            return 1;
        }
        stderr.writefln("loaded state: %d tokens", nLoaded);

        // Decode the same final prompt token from the same base state that
        // run 1 used. This guarantees identical logits for hybrid SSM models.
        llama_pos nPast = cast(llama_pos) nLoaded;
        {
            auto ob = allocBatch(1);
            batchAdd(ob.get(), tokens[$ - 1], nPast, 0, true);
            if (ctx2.decode(ob.get()))
            {
                stderr.writeln("error: last-token decode failed (ctx2)");
                return 1;
            }
            nPast++;
        }

        writef("second run: %s", prompt);
        auto ob = allocBatch(1);

        for (int g; g < nPredict; g++)
        {
            auto tok = smpl2.sample(ctx2);
            if (isEog(vocab, tok)) break;

            string piece = tokenToString(vocab, tok);
            writef("%s", piece);
            result2 ~= piece;

            smpl2.accept(tok);
            batchClear(ob.get());
            batchAdd(ob.get(), tok, nPast, 0, true);

            if (ctx2.decode(ob.get()))
            {
                stderr.writeln("\nerror: decode failed");
                return 1;
            }
            nPast++;
        }
        writeln("\n");
    }

    if (result1 != result2)
    {
        stderr.writeln("FAIL: the two runs produced different output");
        stderr.writefln("  run 1: %s", result1);
        stderr.writefln("  run 2: %s", result2);
        return 1;
    }

    stderr.writeln("OK: both runs match");
    return 0;
}

int printUsage(string prog) @trusted nothrow
{
    printf(
        "\nusage: %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers]\n"
        ~ "                       [-s state_file] [prompt]\n\n"
        ~ "  -m  model path (GGUF)\n"
        ~ "  -n  tokens to predict (default: 16)\n"
        ~ "  -ngl GPU layers (default: 0)\n"
        ~ "  -s  state file (default: dump_state.bin)\n\n",
        prog.ptr);
    return 1;
}
