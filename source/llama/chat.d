module llama.chat;

import llama.llama;

/++
Apply a Jinja chat template to a list of messages.

`tmpl` may be `null` to use an empty string (the model's embedded template is selected elsewhere).
Set `addAss = true` to append the assistant-turn prefix so the model continues from there.

Returns the number of bytes written to `buf`. If the return value exceeds `buf.length`,
reallocate and call again.
+/
int chatApplyTemplate(const(char)* tmpl,
                      scope const(llama_chat_message)[] chat,
                      bool addAss,
                      scope char[] buf) @trusted @nogc nothrow
{
    return llama_chat_apply_template(tmpl,
                                     cast(llama_chat_message*) chat.ptr, chat.length,
                                     addAss, buf.ptr, cast(int) buf.length);
}

/++
Apply a chat template and return the result as a D string.
Allocates a buffer and retries once if it is too small.
+/
string applyTemplate(const(char)* tmpl,
                     scope const(llama_chat_message)[] chat,
                     bool addAss = true) @trusted
{
    char[] buf = new char[](1024);
    auto mp = cast(llama_chat_message*) chat.ptr;
    int n = llama_chat_apply_template(tmpl, mp, chat.length,
                                      addAss, buf.ptr, cast(int) buf.length);
    if (n > cast(int) buf.length)
    {
        buf = new char[](n + 1);
        n = llama_chat_apply_template(tmpl, mp, chat.length,
                                      addAss, buf.ptr, cast(int) buf.length);
    }
    return n > 0 ? buf[0 .. n].idup : "";
}

/++
Names of all built-in chat templates supported by `llama_chat_apply_template`.
Returns a GC-allocated slice of D strings.
+/
string[] builtinTemplates() @trusted
{
    // First call: query count.
    int n = llama_chat_builtin_templates(null, 0);
    if (n <= 0) return [];

    auto ptrs = new const(char)*[](n);
    llama_chat_builtin_templates(ptrs.ptr, n);

    import std.string : fromStringz;
    auto result = new string[](n);
    foreach (i, p; ptrs)
        result[i] = fromStringz(p).idup;
    return result;
}
