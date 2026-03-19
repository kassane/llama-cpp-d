module llama.backend;

import llama.ggml_backend : ggml_backend_load_all;
import llama.llama : llama_backend_init, llama_backend_free;

/// Frees the llama backend on scope exit.
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

/// Inits the llama backend; the returned guard calls `llama_backend_free` on exit.
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
