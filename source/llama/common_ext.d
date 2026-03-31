/++
D wrappers for llama.cpp common library (via C shims).

Provides chat-template application with full kwargs support (enable_thinking),
matching the behaviour of the llama.cpp CLI `--chat-template-kwargs` flag.

Usage:
---
import llama;

auto tmpls = ChatTemplates.fromModel(model.ptr);
if (tmpls && tmpls.supportsThinking)
    writeln("model supports <think> blocks");

string prompt = tmpls.apply(history, enableThinking: false, addAss: true);
---
+/
module llama.common_ext;

import llama.llama : llama_model, llama_chat_message;

// Private: common_shims.h includes llama.h → would duplicate llama symbols.
private import c.common_stubs;

// ── ChatParams ────────────────────────────────────────────────────────────
struct ChatParams
{
    string prompt;
    string grammar;
    bool grammarLazy;
    string thinkingStart; /// e.g. "<think>", empty if not supported
    string thinkingEnd; /// e.g. "</think>", empty if not supported
    string[] stops;
}

// ── ChatTemplates ─────────────────────────────────────────────────────────

/++
Wraps an opaque `lcpp_chat_templates*` handle.

Construct via `ChatTemplates.fromModel`.  The handle is freed on
destruction.  Copy is disabled; use `move` if ownership transfer is needed.
+/
struct ChatTemplates
{
    private lcpp_chat_templates* _ptr;

    @disable this(this);

    ~this() @trusted @nogc nothrow
    {
        if (_ptr)
        {
            lcpp_chat_templates_free(_ptr);
            _ptr = null;
        }
    }

    /// Cast to bool: `if (tmpls)` is true when successfully loaded.
    bool opCast(T : bool)() const @nogc nothrow
    {
        return _ptr !is null;
    }

    /// Raw pointer (for passing to C shims directly).
    @property lcpp_chat_templates* ptr() @nogc nothrow
    {
        return _ptr;
    }

    /++
    Load templates from a model.
    `tmplOverride` may be empty/null to use the model's embedded template.
    Returns a falsy handle on failure.
    +/
    static ChatTemplates fromModel(
        const(llama_model)* model,
        string tmplOverride = null) @trusted nothrow
    {
        if (model is null)
            return ChatTemplates(null);
        import std.string : toStringz;

        const(char)* over = tmplOverride.length ? tmplOverride.toStringz : null;
        return ChatTemplates(lcpp_chat_templates_init(cast(llama_model*) model, over));
    }

    /// Returns true if the loaded template supports the `enable_thinking` kwarg.
    @property bool supportsThinking() const @trusted @nogc nothrow
    {
        return lcpp_chat_templates_supports_thinking(
            cast(lcpp_chat_templates*) _ptr) != 0;
    }

    /++
    Apply the chat template and return the full `ChatParams`.

    Params:
      msgs            = Conversation history as `llama_chat_message[]`.
      enableThinking  = -1 model default, 0 disable, 1 enable.
      addAss          = Append the assistant-turn prefix (for generation).

    Returns a `ChatParams` with prompt, grammar, thinking tags and stop strings.
    +/
    ChatParams apply(
        scope const(llama_chat_message)[] msgs,
        int enableThinking = -1,
        bool addAss = true) @trusted
    {
        if (!_ptr || msgs.length == 0)
            return ChatParams.init;

        auto mp = cast(llama_chat_message*) msgs.ptr;
        lcpp_chat_params* p = lcpp_chat_templates_apply_full(
            _ptr, mp, msgs.length, enableThinking, addAss);
        if (!p)
            return ChatParams.init;
        scope (exit)
            lcpp_chat_params_free(p);

        import std.string : fromStringz;

        ChatParams r;
        r.prompt = p.prompt ? p.prompt.fromStringz.idup : "";
        r.grammar = p.grammar ? p.grammar.fromStringz.idup : "";
        r.grammarLazy = p.grammar_lazy != 0;
        r.thinkingStart = p.thinking_start ? p.thinking_start.fromStringz.idup : "";
        r.thinkingEnd = p.thinking_end ? p.thinking_end.fromStringz.idup : "";
        r.stops.length = p.n_stops;
        foreach (i; 0 .. p.n_stops)
            r.stops[i] = p.stops[i] ? p.stops[i].fromStringz.idup : "";
        return r;
    }

    /++
    Convenience helper — returns only the rendered prompt string.
    Use `apply` when you also need grammar or thinking tags.
    +/
    string applyPrompt(
        scope const(llama_chat_message)[] msgs,
        int enableThinking = -1,
        bool addAss = true) @trusted
    {
        if (!_ptr || msgs.length == 0)
            return "";

        auto mp = cast(llama_chat_message*) msgs.ptr;
        int n = lcpp_chat_templates_apply(
            _ptr, mp, msgs.length, enableThinking, addAss, null, 0);
        if (n <= 0)
            return "";

        auto buf = new char[](n + 1);
        lcpp_chat_templates_apply(
            _ptr, mp, msgs.length, enableThinking, addAss, buf.ptr, cast(int) buf.length);
        return buf[0 .. n].idup;
    }

    private this(lcpp_chat_templates* p) @nogc nothrow
    {
        _ptr = p;
    }
}
