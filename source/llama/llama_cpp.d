/++
 + D declarations mirroring `llama-cpp.h`.
 +
 + llama-cpp.h is a C++-only header (it guards against C inclusion with
 + `#error "This header is for C++ only"`).  D's importC feature only
 + handles plain C headers, so these declarations must be written by hand
 + using `extern(C++)`.
 +/
module llama.llama_cpp;

import llama.llama : llama_model, llama_context, llama_sampler,
                     llama_model_free, llama_free, llama_sampler_free;

// ---------------------------------------------------------------------------
// C++ struct declarations (extern(C++) maps D structs to C++ structs)
// ---------------------------------------------------------------------------

/// Functor deleter for `llama_model*`; mirrors `llama_model_deleter`
extern(C++) struct llama_model_deleter
{
    /// Calls `llama_model_free(model)`.
    void opCall(llama_model* model) nothrow;
}

/// Functor deleter for `llama_context*`; mirrors `llama_context_deleter`
extern(C++) struct llama_context_deleter
{
    /// Calls `llama_free(context)`.
    void opCall(llama_context* context) nothrow;
}

/// Functor deleter for `llama_sampler*`; mirrors `llama_sampler_deleter`
extern(C++) struct llama_sampler_deleter
{
    /// Calls `llama_sampler_free(sampler)`.
    void opCall(llama_sampler* sampler) nothrow;
}

// llama_adapter_lora forward-declare (the typedef is in llama.h as an
// opaque struct; importC gives us `llama_adapter_lora` already).
import llama.llama : llama_adapter_lora, llama_adapter_lora_free;

/// Functor deleter for `llama_adapter_lora*`; mirrors `llama_adapter_lora_deleter`
extern(C++) struct llama_adapter_lora_deleter
{
    /// Calls `llama_adapter_lora_free(adapter)`.
    void opCall(llama_adapter_lora* adapter) nothrow;
}
