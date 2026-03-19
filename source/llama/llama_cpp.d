/++
D declarations for `llama-cpp.h` (C++-only header).

Since importC only handles plain C headers, these structs are written by
hand using `extern(C++)`.
+/
module llama.llama_cpp;

import llama.llama : llama_model, llama_context, llama_sampler,
                     llama_model_free, llama_free, llama_sampler_free;

// ---------------------------------------------------------------------------
// C++ struct declarations (extern(C++) maps D structs to C++ structs)
// ---------------------------------------------------------------------------

/// C++ deleter for `llama_model*`; used by `std::unique_ptr<llama_model>`.
extern(C++) struct llama_model_deleter
{
    void opCall(llama_model* model) nothrow;
}

/// C++ deleter for `llama_context*`; used by `std::unique_ptr<llama_context>`.
extern(C++) struct llama_context_deleter
{
    void opCall(llama_context* context) nothrow;
}

/// C++ deleter for `llama_sampler*`; used by `std::unique_ptr<llama_sampler>`.
extern(C++) struct llama_sampler_deleter
{
    void opCall(llama_sampler* sampler) nothrow;
}

// llama_adapter_lora forward-declare (the typedef is in llama.h as an
// opaque struct; importC gives us `llama_adapter_lora` already).
import llama.llama : llama_adapter_lora, llama_adapter_lora_free;

/// C++ deleter for `llama_adapter_lora*`; used by `std::unique_ptr<llama_adapter_lora>`.
extern(C++) struct llama_adapter_lora_deleter
{
    void opCall(llama_adapter_lora* adapter) nothrow;
}

// ---------------------------------------------------------------------------
// mtmd C++ deleter structs (from mtmd.h namespace mtmd)
// ---------------------------------------------------------------------------
import llama.mtmd :
    mtmd_context,  mtmd_free,
    mtmd_bitmap,   mtmd_bitmap_free,
    mtmd_input_chunks, mtmd_input_chunks_free,
    mtmd_input_chunk,  mtmd_input_chunk_free;

extern(C++, "mtmd"):

/// C++ deleter for `mtmd_context*`.
extern(C++) struct mtmd_context_deleter
{
    void opCall(mtmd_context* val) nothrow;
}

/// C++ deleter for `mtmd_bitmap*`.
extern(C++) struct mtmd_bitmap_deleter
{
    void opCall(mtmd_bitmap* val) nothrow;
}

/// C++ deleter for `mtmd_input_chunks*`.
extern(C++) struct mtmd_input_chunks_deleter
{
    void opCall(mtmd_input_chunks* val) nothrow;
}

/// C++ deleter for `mtmd_input_chunk*`.
extern(C++) struct mtmd_input_chunk_deleter
{
    void opCall(mtmd_input_chunk* val) nothrow;
}
