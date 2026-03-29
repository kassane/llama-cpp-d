/// Wrappers for the llama/ggml logging and system-info API.
module llama.logging;

import llama.llama;
import llama.ggml;

/++
Set the global log callback.
`fn` receives every log message; pass `null` to restore the default (stderr).
`ud` is forwarded to `fn` as the `user_data` argument.
+/
void setLogCallback(ggml_log_callback fn, void* ud = null) @nogc nothrow
{
    llama_log_set(fn, ud);
}

/++
Retrieve the currently active log callback and user-data pointer.
Either output parameter may be `null` if you only need one of them.
+/
void getLogCallback(ggml_log_callback* fn, void** ud = null) @nogc nothrow
{
    llama_log_get(fn, ud);
}

/++
Silence all llama/ggml log output by installing a no-op callback.
Call `setLogCallback(null)` to restore stderr logging.
+/
void suppressLogs() @nogc nothrow
{
    extern(C) static void noop(ggml_log_level, const(char)*, void*) nothrow @nogc {}
    llama_log_set(&noop, null);
}

/++
Return the llama.cpp system-info string (CPU features, build flags, etc.).
The pointer is valid for the lifetime of the process; do not free it.
+/
const(char)* systemInfoRaw() @nogc nothrow
{
    return llama_print_system_info();
}

/++
Return the llama.cpp system-info string as a D string (GC-allocated copy).
+/
string systemInfo() @trusted
{
    import std.string : fromStringz;
    return fromStringz(llama_print_system_info()).idup;
}
