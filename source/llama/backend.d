module llama.backend;

import llama.ggml_backend : ggml_backend_load_all;
import llama.llama : llama_backend_init, llama_backend_free;

/// Calls `llama_backend_free` when it goes out of scope.
struct BackendGuard
{
    @disable this();
    @disable this(this);

    // Private sentinel constructor — only callable from within this module.
    private this(bool) @nogc nothrow {}

    ~this() @nogc nothrow
    {
        llama_backend_free();
    }
}

/// Initialises the llama backend and returns a guard that frees it on scope exit.
BackendGuard initBackend() @nogc nothrow
{
    llama_backend_init();
    return BackendGuard(false);
}

/// Loads all available ggml backends (CPU, CUDA, Metal, etc.).
void loadAllBackends() @nogc nothrow
{
    ggml_backend_load_all();
}
