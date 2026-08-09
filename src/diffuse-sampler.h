#pragma once

#include "diffuse-common.h"
#include "diffuse.h"

std::vector<int32_t> diffuse_sample(
    diffuse_context * ctx,
    const std::vector<int32_t> & prompt_tokens,
    int n_generate,
    const diffuse_sampler_params & params,
    diffuse_step_callback callback);
