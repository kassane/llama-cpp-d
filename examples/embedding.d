/++
Compute normalized text embeddings and print cosine similarity.

Usage: `embedding -m embed-model.gguf [-p text] [-ngl n_gpu_layers]`

The model must be an embedding model (e.g. nomic-embed-text, bge-*).
If no `-p` is given, two built-in sentences are embedded and compared.
+/
module embedding;

import llama;
import std.stdio  : writefln, writeln, stderr;
import std.conv   : to;
import std.math   : sqrt;
import core.stdc.locale : setlocale, LC_NUMERIC;
import core.stdc.stdio  : printf;

int main(string[] args)
{
    setlocale(LC_NUMERIC, "C");

    string   modelPath;
    string[] prompts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
    ];
    int ngl    = 99;
    int nBatch = 512;

    for (int i = 1; i < cast(int) args.length; i++)
    {
        switch (args[i])
        {
            case "-m":
                if (++i < cast(int) args.length) modelPath = args[i];
                else return printUsage(args[0]);
                break;
            case "-p":
                if (++i < cast(int) args.length) prompts = [args[i]];
                else return printUsage(args[0]);
                break;
            case "-ngl":
                if (++i < cast(int) args.length) ngl = args[i].to!int;
                else return printUsage(args[0]);
                break;
            default:
                prompts = args[i .. $];
                i = cast(int) args.length;
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

    const vocab = model.vocab;
    int   nEmbd = model.nEmbd;

    // Enable embedding extraction; required for generative models.
    auto ctxp        = contextParams(cast(uint) nBatch, cast(uint) nBatch);
    ctxp.embeddings  = true;
    auto ctx = LlamaContext.fromModel(model, ctxp);
    if (!ctx)
    {
        stderr.writeln("error: failed to create context");
        return 1;
    }

    int poolType = ctx.poolingType;
    writefln("pooling type : %d  (%s)",
             poolType,
             poolType == 0 ? "none — using token embeddings" : "sequence pooling");
    writefln("n_embd       : %d", nEmbd);
    writeln();

    auto embeddings = new float[][](prompts.length, nEmbd);

    // Embed each prompt independently (simple, non-batched path).
    foreach (si, prompt; prompts)
    {
        auto tokens = tokenize(vocab, prompt, /*addSpecial=*/true);
        if (tokens is null)
        {
            stderr.writefln("error: tokenization failed for prompt %d", si);
            return 1;
        }

        // Clear the KV cache between prompts.
        ctx.memoryClear(/*data=*/true);

        // Always use seq_id=0: memory is cleared between prompts.
        auto ob = allocBatch(cast(int) tokens.length);
        foreach (j, tok; tokens)
            batchAdd(ob.get(), tok, cast(llama_pos) j, 0, /*logits=*/true);

        if (ctx.decode(ob.get()))
        {
            stderr.writefln("error: decode failed for prompt %d", si);
            return 1;
        }

        // Retrieve embeddings: pooled (mean/CLS) or last-token.
        const(float)[] raw;
        if (poolType == 0) // LLAMA_POOLING_TYPE_NONE
            raw = ctx.getEmbeddingsIth(ob.get().n_tokens - 1);
        else
            raw = ctx.getEmbeddingsSeq(0);

        if (raw is null)
        {
            stderr.writefln("error: no embeddings returned for prompt %d", si);
            return 1;
        }

        // L2-normalize into the output buffer.
        float norm = 0;
        foreach (v; raw) norm += v * v;
        norm = sqrt(norm);
        if (norm > 1e-9f)
            foreach (j; 0 .. nEmbd) embeddings[si][j] = raw[j] / norm;
    }

    // Print a compact summary (first 8 components) — @trusted wrapper avoids
    // -preview=safer restrictions on ptr and printf varargs.
    foreach (i, prompt; prompts)
        printEmbeddingRow(prompt, i, embeddings[i], nEmbd);

    // Pairwise cosine similarity (vectors are already unit-length).
    if (prompts.length >= 2)
    {
        double dot = 0;
        foreach (j; 0 .. nEmbd) dot += embeddings[0][j] * embeddings[1][j];
        printCosine(dot);
    }

    return 0;
}

// @trusted helpers — isolate all raw-pointer / printf-vararg operations.

void printEmbeddingRow(string prompt, size_t idx,
                       const(float)[] embd, int nEmbd) @trusted nothrow
{
    printf("prompt[%zu]: \"", idx);
    foreach (c; prompt) printf("%c", cast(int) c);
    printf("\"\n");
    printf("embedding  : [");
    int show = nEmbd < 8 ? nEmbd : 8;
    foreach (j; 0 .. show)
    {
        if (j > 0) printf(", ");
        printf("%.6f", cast(double) embd[j]);
    }
    if (nEmbd > 8) printf(", ...");
    printf("]\n\n");
}

void printCosine(double sim) @trusted nothrow
{
    printf("cosine similarity [0, 1]: %.6f\n", sim);
}

int printUsage(string prog) @trusted nothrow
{
    printf(
        "\nusage: %s -m embed-model.gguf [-p text] [-ngl n_gpu_layers]\n\n"
        ~ "  -m    embedding model (GGUF)\n"
        ~ "  -p    text to embed (default: two built-in sentences)\n"
        ~ "  -ngl  GPU layers (default: 99)\n\n",
        prog.ptr);
    return 1;
}
