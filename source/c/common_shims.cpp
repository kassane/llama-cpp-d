/*
 * common_shims.cpp — extern "C" wrappers for llama.cpp common library.
 *
 * Compiled separately (c++) and archived into libcommon-shims.a.
 * D code imports the C header common_shims.h via importC.
 */
#include "common_shims.h"
#include "chat.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

/* ── opaque handle ───────────────────────────────────────────────────────── */

struct lcpp_chat_templates {
  common_chat_templates_ptr ptr;
};

/* ── extern "C" implementations ─────────────────────────────────────────── */

extern "C" {

lcpp_chat_templates *lcpp_chat_templates_init(const struct llama_model *model,
                                              const char *tmpl_override) {
  if (!model)
    return nullptr;
  auto p = common_chat_templates_init(
      model, tmpl_override ? std::string(tmpl_override) : std::string(""),
      /* tmpl_override_ins = */ std::string(""));
  if (!p)
    return nullptr;
  auto *h = new lcpp_chat_templates{};
  h->ptr = std::move(p);
  return h;
}

void lcpp_chat_templates_free(lcpp_chat_templates *tmpls) { delete tmpls; }

int lcpp_chat_templates_apply(lcpp_chat_templates *tmpls,
                              const struct llama_chat_message *msgs,
                              size_t n_msgs, int enable_thinking, bool add_ass,
                              char *buf, int buf_len) {
  if (!tmpls)
    return -1;

  common_chat_templates_inputs inputs;
  inputs.messages.reserve(n_msgs);
  for (size_t i = 0; i < n_msgs; i++) {
    common_chat_msg msg;
    msg.role = msgs[i].role ? msgs[i].role : "";
    msg.content = msgs[i].content ? msgs[i].content : "";
    inputs.messages.push_back(std::move(msg));
  }
  inputs.add_generation_prompt = add_ass;
  if (enable_thinking == 0)
    inputs.enable_thinking = false;
  else if (enable_thinking == 1)
    inputs.enable_thinking = true;
  /* -1 → leave at the struct default (true) */

  common_chat_params result =
      common_chat_templates_apply(tmpls->ptr.get(), inputs);
  const std::string &prompt = result.prompt;

  int n = static_cast<int>(prompt.size());
  if (buf && buf_len > 0) {
    int to_copy = n < buf_len - 1 ? n : buf_len - 1;
    std::memcpy(buf, prompt.c_str(), to_copy);
    buf[to_copy] = '\0';
  }
  return n;
}

int lcpp_chat_templates_supports_thinking(const lcpp_chat_templates *tmpls) {
  if (!tmpls)
    return 0;
  return common_chat_templates_support_enable_thinking(tmpls->ptr.get()) ? 1
                                                                         : 0;
}

} /* extern "C" */
