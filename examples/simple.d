/// Minimal text-completion example.
/// Usage: simple -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]
module simple;

import llama;
import std.stdio  : write, writeln, writefln, stderr;
import std.conv   : to;
import core.stdc.locale : setlocale, LC_NUMERIC;
import core.stdc.stdio  : printf;

int main(string[] args)
{
    setlocale(LC_NUMERIC, "C");

    string modelPath;
    string prompt   = "Hello my name is";
    int    ngl      = 99;
    int    nPredict = 32;

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
        stderr.writeln("error: unable to load model '", modelPath, "'");
        return 1;
    }

    const vocab = model.vocab;

    auto tokens = tokenize(vocab, prompt);
    if (tokens.length == 0)
    {
        stderr.writeln("error: failed to tokenize prompt");
        return 1;
    }

    // Print prompt as decoded text.
    foreach (t; tokens) write(tokenToString(vocab, t));

    auto ctx = LlamaContext.fromModel(model,
                   cast(uint)(tokens.length + nPredict - 1),
                   cast(uint) tokens.length);
    if (!ctx)
    {
        stderr.writeln("error: failed to create llama_context");
        return 1;
    }

    // Two-statement form: avoids copy-constructing the non-copyable SamplerChain.
    auto smpl = SamplerChain.create();
    smpl.greedy();

    auto batch = batchGetOne(tokens);

    if (model.hasEncoder)
    {
        if (ctx.encode(batch)) { stderr.writeln("error: encode failed"); return 1; }
        // Stack array: avoids pointer-slicing under -preview=safer.
        llama_token[1] startBuf = [model.decoderStartToken];
        batch = batchGetOne(startBuf[]);
    }

    import core.time : MonoTime;
    auto tStart = MonoTime.currTime;
    int  nDecode;
    llama_token[1] nextBuf; // reusable single-token buffer

    for (int nPos; nPos + batch.n_tokens < cast(int) tokens.length + nPredict; )
    {
        if (ctx.decode(batch)) { stderr.writeln("error: decode failed"); return 1; }
        nPos += batch.n_tokens;

        auto newTok = smpl.sample(ctx);
        if (isEog(vocab, newTok)) break;

        write(tokenToString(vocab, newTok));

        import std.stdio : stdout;
        stdout.flush();

        nextBuf[0] = newTok;
        batch = batchGetOne(nextBuf[]);
        nDecode++;
    }
    writeln();

    // Integer arithmetic: avoids writefln+double template bug under -preview=all.
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
    printf("\nusage: %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n\n",
           prog.ptr);
    return 1;
}
