/++
Wrappers for the llama.cpp fine-tuning / optimisation API.

llama.cpp exposes a gradient-based optimisation loop (`llama_opt_*`) that
supports both AdamW and SGD.  This module provides D-friendly helpers around
that API.

Typical workflow:
---
import llama, llama.opt;

auto dataset = ggml_opt_dataset_init(nEmbd, nLabel, nData, nBatch);
auto resultTrain = ggml_opt_result_init();
auto resultEval  = ggml_opt_result_init();

auto params = defaultOptParams();
params.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
optInit(ctx, model, params);

foreach (epoch; 0 .. 10)
{
    ggml_opt_dataset_shuffle(null, dataset, -1);
    optEpoch(ctx, dataset, resultTrain, resultEval, iDataSplit);
    // read loss: double loss, unc; ggml_opt_result_loss(resultTrain, &loss, &unc);
}

ggml_opt_result_free(resultTrain);
ggml_opt_result_free(resultEval);
ggml_opt_dataset_free(dataset);
---
+/
module llama.opt;

import llama.llama; // llama_opt_*, llama_opt_params, llama_opt_param_filter_all
import llama.ggml_opt; // ggml_opt_dataset_t, ggml_opt_result_t, ggml_opt_epoch_callback, etc.
import llama.ctx : LlamaContext;
import llama.model : LlamaModel;

/++
Build a default `llama_opt_params`.

Uses AdamW, trains all weight tensors, and reads optimizer hyper-parameters
from `ggml_opt_get_default_optimizer_params`.
+/
llama_opt_params defaultOptParams() @trusted @nogc nothrow
{
    // Use local import to disambiguate: both c.llama_stubs and c.ggml_opt_stubs
    // expose this symbol (llama.h includes ggml-opt.h); qualify from ggml_opt_stubs.
    import c.ggml_opt_stubs : ggml_opt_default_pars = ggml_opt_get_default_optimizer_params;

    llama_opt_params p = llama_opt_params.init;
    p.n_ctx_train = 0; // use context size from llama_context
    p.param_filter = &llama_opt_param_filter_all;
    p.param_filter_ud = null;
    p.get_opt_pars = &ggml_opt_default_pars;
    p.get_opt_pars_ud = null;
    p.optimizer_type = ggml_opt_optimizer_type.GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    return p;
}

/++
Initialise the optimisation state for a context + model pair.
Must be called once before the first `optEpoch` call.
+/
void optInit(ref LlamaContext ctx, ref LlamaModel model,
    llama_opt_params params) @nogc nothrow
{
    llama_opt_init(ctx.ptr, model.ptr, params);
}

/++
Run one pass over `dataset`.

Data indices `[0, iDataSplit)` are used for training; `[iDataSplit, N)` for
evaluation.  Pass `iDataSplit = ggml_opt_dataset_ndata(dataset)` to skip
the evaluation pass.

`callbackTrain` / `callbackEval` are optional per-batch callbacks with the
signature `void cb(ggml_opt_context_t, ggml_opt_result_t)`.
+/
void optEpoch(ref LlamaContext ctx,
    ggml_opt_dataset_t dataset,
    ggml_opt_result_t resultTrain,
    ggml_opt_result_t resultEval,
    long iDataSplit,
    ggml_opt_epoch_callback callbackTrain = null,
    ggml_opt_epoch_callback callbackEval = null) @nogc nothrow
{
    llama_opt_epoch(ctx.ptr, dataset, resultTrain, resultEval,
        iDataSplit, callbackTrain, callbackEval);
}

/++
Read the loss and uncertainty from an opt result.
`unc` is the standard error of the mean; pass `null` to ignore it.
+/
void optResultLoss(ggml_opt_result_t result,
    out double loss, double* unc = null) @trusted @nogc nothrow
{
    ggml_opt_result_loss(result, &loss, unc);
}
