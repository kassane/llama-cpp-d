/// Tokenize text and print each token id + its string piece.
/// Usage: tokenize -m model.gguf [-s] [text]
///   -s   include BOS/EOS special tokens; default: off
module tokenize_example;

import llama;
import std.stdio  : writefln, writeln, stderr;
import core.stdc.locale : setlocale, LC_NUMERIC;
import core.stdc.stdio  : printf;

int main(string[] args)
{
    setlocale(LC_NUMERIC, "C");

    string modelPath;
    string text      = "Hello, world!";
    bool   addSpecial;

    for (int i = 1; i < cast(int) args.length; i++)
    {
        switch (args[i])
        {
            case "-m":
                if (++i < cast(int) args.length) modelPath = args[i];
                else return printUsage(args[0]);
                break;
            case "-s":
                addSpecial = true;
                break;
            default:
                text = args[i];
                break;
        }
    }
    if (modelPath.length == 0) return printUsage(args[0]);

    loadAllBackends();

    // Vocab-only: skips loading weight tensors; all we need for tokenization.
    auto model = LlamaModel.loadVocabOnly(modelPath);
    if (!model)
    {
        stderr.writefln("error: unable to load model '%s'", modelPath);
        return 1;
    }

    const vocab = model.vocab;

    writefln("vocab size : %d", model.nVocab);
    writefln("BOS token  : %d", bosToken(vocab));
    writefln("EOS token  : %d", eosToken(vocab));
    writefln("EOT token  : %d", eotToken(vocab));
    writeln();

    auto tokens = tokenize(vocab, text, addSpecial);
    if (tokens is null)
    {
        stderr.writeln("error: tokenization failed");
        return 1;
    }

    writefln("text       : \"%s\"", text);
    writefln("tokens     : %d", tokens.length);
    writeln();

    writefln("%-8s  %-6s  %s", "id", "type", "piece");
    writefln("%-8s  %-6s  %s", "--------", "------", "-----");
    foreach (t; tokens)
    {
        string tag = isEog(vocab, t) ? "EOG" : (t == bosToken(vocab) ? "BOS" : "tok");
        writefln("%-8d  %-6s  \"%s\"", t, tag, tokenToString(vocab, t));
    }
    writeln();

    // Round-trip check.
    string roundTrip = detokenize(vocab, tokens);
    writefln("round-trip : \"%s\"", roundTrip);
    writeln(roundTrip == text
        ? "round-trip : OK"
        : "round-trip : differs (expected if special tokens were added)");

    return 0;
}

int printUsage(string prog) @trusted nothrow
{
    printf("\nusage: %s -m model.gguf [-s] [text]\n\n"
           ~ "  -m  path to a GGUF model file\n"
           ~ "  -s  add BOS/EOS special tokens around the text\n\n",
           prog.ptr);
    return 1;
}
