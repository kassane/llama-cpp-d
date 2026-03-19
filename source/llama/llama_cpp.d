/**
 * D declarations mirroring `llama-cpp.h`.
 *
 * llama-cpp.h is a C++-only header (it guards against C inclusion with
 * `#error "This header is for C++ only"`).  D's importC feature only
 * handles plain C headers, so these declarations must be written by hand
 * using `extern(C++)`.
 *
 * The structs below are the four custom-deleter types that llama.cpp
 * exposes so that callers can use `std::unique_ptr` for RAII lifetime
 * management in C++.  In D we already have our own RAII wrappers
 * (`LlamaModel`, `LlamaContext`, `SamplerChain`) in the other modules,
 * so these extern(C++) declarations are provided primarily for:
 *
 *   1. Demonstrating/documenting how D's C++ FFI works for simple structs
 *      with member functions declared in a C++ namespace or global scope.
 *   2. Interoperating with C++ code that passes the `*_ptr` smart-pointer
 *      types across a DLL/SO boundary where the raw pointer is extracted
 *      via `.get()` and then handed to D.
 *
 * NOTE: The `std::unique_ptr<T, Deleter>` typedefs in llama-cpp.h cannot
 * be represented directly in D because LDC's/DMD's stdcpp bindings only
 * provide a skeletal `std.unique_ptr` shim (not shipped in LDC 1.42 at
 * all).  For the common need of extracting a raw pointer from a
 * `llama_model_ptr` before passing it to D code, call `.release()` or
 * `.get()` on the C++ side and hand the raw pointer to
 * `LlamaModel.fromPtr()` (a factory provided in llama.model).
 */
module llama.llama_cpp;

import llama.llama : llama_model, llama_context, llama_sampler,
                     llama_model_free, llama_free, llama_sampler_free;

// ---------------------------------------------------------------------------
// C++ struct declarations (extern(C++) maps D structs to C++ structs)
// ---------------------------------------------------------------------------

/// Functor deleter for `llama_model*`; mirrors `llama_model_deleter` in llama-cpp.h.
extern(C++) struct llama_model_deleter
{
    /// Calls `llama_model_free(model)`.
    void opCall(llama_model* model) nothrow;
}

/// Functor deleter for `llama_context*`; mirrors `llama_context_deleter` in llama-cpp.h.
extern(C++) struct llama_context_deleter
{
    /// Calls `llama_free(context)`.
    void opCall(llama_context* context) nothrow;
}

/// Functor deleter for `llama_sampler*`; mirrors `llama_sampler_deleter` in llama-cpp.h.
extern(C++) struct llama_sampler_deleter
{
    /// Calls `llama_sampler_free(sampler)`.
    void opCall(llama_sampler* sampler) nothrow;
}

// llama_adapter_lora forward-declare (the typedef is in llama.h as an
// opaque struct; importC gives us `llama_adapter_lora` already).
import llama.llama : llama_adapter_lora, llama_adapter_lora_free;

/// Functor deleter for `llama_adapter_lora*`; mirrors `llama_adapter_lora_deleter` in llama-cpp.h.
extern(C++) struct llama_adapter_lora_deleter
{
    /// Calls `llama_adapter_lora_free(adapter)`.
    void opCall(llama_adapter_lora* adapter) nothrow;
}

// ---------------------------------------------------------------------------
// Why no std::unique_ptr bindings here?
// ---------------------------------------------------------------------------
//
// LDC 1.42 ships without a `core.stdcpp.unique_ptr` or equivalent module.
// DMD's druntime has `core/stdcpp/` stubs but they are not fully
// implemented for all STL types, and are only useful when linking against
// the MSVC or libc++ C++ runtime on the matching platform.
//
// Attempting to bind `std::unique_ptr<llama_model, llama_model_deleter>`
// as an `extern(C++)` struct would look like:
//
//   extern(C++, "std") struct unique_ptr(T, Deleter)
//   {
//       T* _Ptr;        // internal layout differs between STL implementations
//       Deleter _Del;
//       T*   get()     nothrow;
//       T*   release() nothrow;
//       void reset(T* p = null) nothrow;
//   }
//   alias llama_model_ptr   = unique_ptr!(llama_model,   llama_model_deleter);
//   alias llama_context_ptr = unique_ptr!(llama_context, llama_context_deleter);
//   …
//
// This is fragile because the internal layout of `std::unique_ptr` is
// implementation-defined (libstdc++, libc++, and MSVC STL all differ).
// It is therefore left out in favour of passing raw pointers across the
// ABI boundary and wrapping them with D's own RAII structs.
