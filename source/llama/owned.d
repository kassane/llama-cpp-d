/++
Mixin template for single-ownership wrappers around C pointers.

Injects the boilerplate shared by all handle types: a private `_ptr` field,
a private constructor, a destructor that calls `freeFn`, copy-disable
directives, and `bool` / `ptr` conversions.

Usage:
---
struct LlamaModel {
    mixin Owned!(llama_model, llama_model_free);
    // add factory methods and domain-specific properties below
}
---

The injected private constructor is accessible inside the hosting struct, so
factory static methods can return `StructName(rawPtr)` normally.
+/
module llama.owned;

/++
Ownership mixin. `T` is the pointed-to C type; `freeFn` is called with the
pointer on destruction.
+/
mixin template Owned(T, alias freeFn)
{
    private T* _ptr;

    @disable this();
    @disable this(this);

    private this(T* p) @nogc nothrow { _ptr = p; }

    ~this() @nogc nothrow
    {
        if (_ptr) { freeFn(_ptr); _ptr = null; }
    }

    /// True when the handle holds a non-null pointer.
    bool opCast(B : bool)() const @nogc nothrow { return _ptr !is null; }

    /// Raw C pointer (mutable).
    @property       T* ptr()       @nogc nothrow { return _ptr; }
    /// Raw C pointer (const view).
    @property const(T)* ptr() const @nogc nothrow { return _ptr; }
}
