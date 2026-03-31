/++
Minimal text-completion example.
Usage: `simple -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]`
+/
module simple;

import llama;
import std.stdio  : write, writeln, writefln, stderr;
import core.stdc.locale : setlocale, LC_NUMERIC;
import core.stdc.stdio  : printf;

int main(string[] args)
{
    setlocale(LC_NUMERIC, "C");

    ModelConfig    mcfg;
    SamplingConfig scfg;

    if (!parseConfig(mcfg, args, "model options:") ||
        !parseConfig(scfg, args, "sampling options:"))
        return 0;

    // Remaining positional argument overrides the -p prompt.
    if (args.length > 1) mcfg.prompt = args[1];

    if (mcfg.modelPath.length == 0)
        return printUsage(args[0]);

    loadAllBackends();

    auto model = LlamaModel.loadFromFile(mcfg.modelPath, mcfg.nGpuLayers);
    if (!model) { stderr.writeln("error: cannot load model"); return 1; }

    const vocab = model.vocab;

    auto tokens = tokenize(vocab, mcfg.prompt);
    if (tokens.length == 0) { stderr.writeln("error: tokenize failed"); return 1; }

    // Echo prompt.
    foreach (t; tokens) write(tokenToString(vocab, t));

    auto ctx = LlamaContext.fromModel(model,
                   cast(uint)(tokens.length + mcfg.nPredict - 1),
                   mcfg.nBatch);
    if (!ctx) { stderr.writeln("error: cannot create context"); return 1; }

    auto smpl  = buildSamplerChain(scfg);
    auto batch = batchGetOne(tokens);

    if (model.hasEncoder)
    {
        if (ctx.encode(batch)) { stderr.writeln("error: encode failed"); return 1; }
        llama_token[1] startBuf = [model.decoderStartToken];
        batch = batchGetOne(startBuf[]);
    }

    import core.time : MonoTime;
    auto tStart = MonoTime.currTime;
    int  nDecode;
    llama_token[1] nextBuf;

    for (int nPos; nPos + batch.n_tokens < cast(int) tokens.length + mcfg.nPredict; )
    {
        if (ctx.decode(batch)) { stderr.writeln("error: decode failed"); return 1; }
        nPos += batch.n_tokens;

        auto newTok = smpl.sample(ctx);
        smpl.accept(newTok);
        if (isEog(vocab, newTok)) break;

        write(tokenToString(vocab, newTok));

        import std.stdio : stdout;
        stdout.flush();

        nextBuf[0] = newTok;
        batch = batchGetOne(nextBuf[]);
        nDecode++;
    }
    writeln();

    // Integer arithmetic avoids writefln+double template bug under -preview=all.
    long elapsedMs = (MonoTime.currTime - tStart).total!"msecs";
    long tokPerSec = (nDecode > 0 && elapsedMs > 0) ? nDecode * 1000 / elapsedMs : 0;

    stderr.writefln("decoded %d tokens in %d.%02ds, speed: %d t/s",
                    nDecode, elapsedMs / 1000, (elapsedMs % 1000) / 10, tokPerSec);
    stderr.writeln();
    smpl.printPerf();
    ctx.printPerf();
    stderr.writeln();

    return 0;
}

int printUsage(string prog) @trusted nothrow
{
    printf(
        "\nusage: %s -m model.gguf [options] [prompt]\n\n"
        ~ "  -m        model path (.gguf)\n"
        ~ "  -ngl      GPU layers (default: 99)\n"
        ~ "  -c        context size (default: 0 = model default)\n"
        ~ "  -b        batch size (default: 512)\n"
        ~ "  -n        tokens to predict (default: 128)\n"
        ~ "  -p        prompt text (or pass as positional arg)\n"
        ~ "  -t        temperature (default: 0.8; 0 = greedy)\n"
        ~ "  -k        top-K (default: 40)\n"
        ~ "  --top-p   top-P nucleus (default: 0.95)\n"
        ~ "  --min-p   min-P floor (default: 0.05)\n"
        ~ "  --seed    RNG seed\n\n",
        prog.ptr);
    return 1;
}
