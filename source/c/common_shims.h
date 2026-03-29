/*
 * common_shims.h — C-compatible shims for llama.cpp common library.
 *
 * Imported by common_stubs.c (D importC) and included by common_shims.cpp.
 */
#pragma once
#include <llama.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle wrapping common_chat_templates. */
typedef struct lcpp_chat_templates lcpp_chat_templates;

/*
 * Create a chat-templates handle from a loaded model.
 * tmpl_override may be NULL to use the model's embedded template.
 * Returns NULL on failure.  Must be freed with lcpp_chat_templates_free().
 */
lcpp_chat_templates *lcpp_chat_templates_init(const struct llama_model *model,
                                              const char *tmpl_override);

/* Destroy a templates handle created by lcpp_chat_templates_init(). */
void lcpp_chat_templates_free(lcpp_chat_templates *tmpls);

/*
 * Apply a chat template to an array of llama_chat_message values.
 *
 *   msgs             C array of {role, content} pairs
 *   n_msgs           number of messages
 *   enable_thinking  -1 = model default, 0 = disable, 1 = enable
 *   add_ass          append the assistant-turn start token/prefix
 *   buf / buf_len    output buffer; pass NULL/0 to query the required size
 *
 * Returns the number of bytes in the rendered prompt (not counting NUL).
 * If the return value >= buf_len the buffer was too small; reallocate and
 * retry.  Returns < 0 on error.
 */
int lcpp_chat_templates_apply(lcpp_chat_templates *tmpls,
                              const struct llama_chat_message *msgs,
                              size_t n_msgs, int enable_thinking, bool add_ass,
                              char *buf, int buf_len);

/*
 * Returns 1 if the loaded template supports the enable_thinking kwarg,
 * 0 otherwise.
 */
int lcpp_chat_templates_supports_thinking(const lcpp_chat_templates *tmpls);

#ifdef __cplusplus
}
#endif
