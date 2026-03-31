/++
Interactive multi-turn chat using the model's embedded chat template.

Usage:
    chat -m model.gguf [-ngl n_gpu_layers] [-c ctx_size] [-n n_predict]
         [-t temp] [-k top_k] [-s system_prompt]
         [--chat-template-kwargs JSON]

--chat-template-kwargs JSON
    Pass template kwargs as a JSON object, e.g. {"enable_thinking":false}.
    Recognised key: enable_thinking (bool).
      true  — force CoT (appends /think to messages, shows <think> output).
      false — disable CoT (appends /no-think, suppresses <think> output).

Type your message and press Enter. Enter an empty line or Ctrl-D to quit.
+/
module chat;

import llama;
import std.stdio : write, writeln, writefln, stderr, readln, stdin, stdout;
import std.string : strip, toStringz;
import core.stdc.locale : setlocale, LC_NUMERIC;
import core.stdc.stdio : printf;

int main(string[] args)
{
    setlocale(LC_NUMERIC, "C");

    ModelConfig mcfg;
    SamplingConfig scfg;
    string systemPrompt = "You are a helpful assistant.";
    int enableThinking = -1; // -1 = default, 0 = disable CoT, 1 = force CoT

    // Extra flags not covered by the standard configs.
    string[] rest = args.dup;
    for (int i = 1; i < cast(int) rest.length;)
    {
        if (rest[i] == "-s" && i + 1 < cast(int) rest.length)
        {
            systemPrompt = rest[i + 1];
            rest = rest[0 .. i] ~ rest[i + 2 .. $];
        }
        else if (rest[i] == "--chat-template-kwargs" && i + 1 < cast(int) rest.length)
        {
            // Minimal parse: look for "enable_thinking": true|false in the JSON string.
            import std.string : indexOf;

            string kw = rest[i + 1];
            ptrdiff_t pos = kw.indexOf("enable_thinking");
            if (pos >= 0)
            {
                // Scan past the key to find the value token.
                ptrdiff_t colon = kw.indexOf(':', pos);
                if (colon >= 0)
                {
                    string tail = kw[colon + 1 .. $];
                    if (tail.indexOf("true") >= 0)
                        enableThinking = 1;
                    else if (tail.indexOf("false") >= 0)
                        enableThinking = 0;
                }
            }
            rest = rest[0 .. i] ~ rest[i + 2 .. $];
        }
        else
            i++;
    }

    if (!parseConfig(mcfg, rest, "chat options:") ||
        !parseConfig(scfg, rest, "sampling options:"))
        return 0;

    if (mcfg.modelPath.length == 0)
        return printUsage(args[0]);

    loadAllBackends();

    auto model = LlamaModel.loadFromFile(mcfg.modelPath, mcfg.nGpuLayers);
    if (!model)
    {
        stderr.writeln("error: cannot load model");
        return 1;
    }

    const vocab = model.vocab;
    auto tmpls = ChatTemplates.fromModel(model.ptr);
    if (!tmpls)
    {
        stderr.writeln("error: cannot load chat template");
        return 1;
    }

    auto cp = contextParams(mcfg.nCtx, mcfg.nBatch);
    auto ctx = LlamaContext.fromModel(model, cp);
    if (!ctx)
    {
        stderr.writeln("error: cannot create context");
        return 1;
    }

    // Conversation history as a GC-managed array of (role, content) pairs.
    llama_chat_message[] history;

    // Add system message if non-empty.
    if (systemPrompt.length > 0)
        history ~= llama_chat_message("system", systemPrompt.toStringz);

    // Thinking tags are discovered on the first apply() call (requires a user
    // message) and cached; they are stable for the lifetime of the template.
    string thinkStart, thinkEnd;

    writeln("Chat session started. Empty line or Ctrl-D to quit.");
    writeln("─────────────────────────────────────────────────────");

    llama_pos nPast = 0;

    while (true)
    {
        write("\nYou: ");
        stdout.flush();

        string line = readln();
        if (line is null)
            break; // Ctrl-D / EOF
        line = line.strip();
        if (line.length == 0)
            break;

        // Apply template with proper enable_thinking kwarg.
        // apply returns prompt + grammar + thinking tags.
        history ~= llama_chat_message("user", line.toStringz);
        auto params = tmpls.apply(history, enableThinking, /*addAss=*/ true);
        if (params.prompt.length == 0)
        {
            stderr.writeln("error: template apply failed");
            return 1;
        }

        // Cache thinking tags on first successful apply (stable across turns).
        if (thinkStart.length == 0 && params.thinkingStart.length)
        {
            thinkStart = params.thinkingStart;
            thinkEnd = params.thinkingEnd;
        }

        // Rebuild sampler each turn; add grammar when template provides one
        // (grammar enforces thinking suppression at token-selection level).
        auto smpl = buildSamplerChain(scfg);
        if (params.grammar.length > 0)
            smpl.grammar(vocab, params.grammar);

        // Tokenise only the NEW portion (history grows, so we re-tokenise the
        // delta: full formatted string minus what we already processed).
        auto allTokens = tokenize(vocab, params.prompt, /*addSpecial=*/ true);
        if (allTokens is null)
        {
            stderr.writeln("error: tokenize failed");
            return 1;
        }

        // Decode only the new tokens (after nPast).
        if (cast(int) allTokens.length > nPast)
        {
            auto newToks = allTokens[nPast .. $];
            auto ob = allocBatch(cast(int) newToks.length);
            foreach (i, tok; newToks)
                batchAdd(ob.get(), tok, nPast + cast(llama_pos) i, 0, false);
            // Request logits for the last position only.
            (() @trusted { ob.get().logits[ob.get().n_tokens - 1] = true; })();

            if (ctx.decode(ob.get()))
            {
                stderr.writeln("error: decode failed");
                return 1;
            }
            nPast = cast(llama_pos) allTokens.length;
        }

        // Generate assistant response.
        write("Assistant: ");
        stdout.flush();

        string assistantReply;
        auto ob = allocBatch(1);
        bool inThink = false; // true while inside a thinking block

        for (int n = 0; n < mcfg.nPredict; n++)
        {
            auto tok = smpl.sample(ctx);
            smpl.accept(tok);

            if (isEog(vocab, tok))
                break;

            string piece = tokenToString(vocab, tok);

            // Track think-block boundaries using actual template tags.
            // When noThink: grammar already suppresses thinking at token level;
            // this filter is a safety net for models that ignore the grammar.
            if (thinkStart.length && piece == thinkStart)
            {
                inThink = true;
                if (enableThinking == 0)
                    goto next;
            }
            else if (thinkEnd.length && piece == thinkEnd)
            {
                inThink = false;
                if (enableThinking == 0)
                    goto next;
            }
            else if (enableThinking == 0 && inThink)
                goto next;

            write(piece);
            stdout.flush();
            assistantReply ~= piece;
        next:

            batchClear(ob.get());
            batchAdd(ob.get(), tok, nPast, 0, true);
            if (ctx.decode(ob.get()))
            {
                stderr.writeln("\nerror: decode");
                return 1;
            }
            nPast++;
        }
        writeln();

        // Store assistant turn.
        history ~= llama_chat_message("assistant", assistantReply.toStringz);
    }

    writeln("\n─────────────────────────────────────────────────────");
    ctx.printPerf();
    return 0;
}

int printUsage(string prog) @trusted nothrow
{
    printf(
        "\nusage: %s -m model.gguf [options]\n\n"
            ~ "  -m      model path (.gguf)\n"
            ~ "  -ngl    GPU layers (default: 99)\n"
            ~ "  -c      context size (default: 0 = model default)\n"
            ~ "  -n      max tokens per reply (default: 128)\n"
            ~ "  -t      temperature (default: 0.8)\n"
            ~ "  -k      top-k (default: 40)\n"
            ~ "  -s      system prompt (default: 'You are a helpful assistant.')\n"
            ~ "  --chat-template-kwargs JSON  e.g. {\"enable_thinking\":false}\n\n",
        prog.ptr);
    return 1;
}
