import tqdm
import contextlib
from compressed_tensors.utils import replace_module, delete_offload_module, register_offload_module
from transformers import PreTrainedModel

from llmcompressor.modeling.deepseek_v3 import replace as replace_deepseekv3
from llmcompressor.modeling.llama4 import replace as replace_llama4
from llmcompressor.modeling.qwen3_moe import replace as replace_Qwen3MoE
from llmcompressor.modeling.gpt_oss import GptOssExpertsLinear
from llmcompressor.utils.helpers import patch_attr

__all__ = ["replace_modules_for_calibration"]

# ---------------------- module replacements; permanent -------------------------
replacements = {
    "DeepseekV3MoE": replace_deepseekv3,
    "Llama4TextMoe": replace_llama4,
}


def replace_modules_for_calibration(model: PreTrainedModel) -> PreTrainedModel:
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in replacements:
            new_module = replacements[cls_name](config=model.config, module=module)
            replace_module(model, name, new_module)

    return model


# ------------------- module replacements; during calibration --------------------


def update_qwen3_moe(model, stack):
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name == "Qwen3MoeDecoderLayer":
            # Optionally update the model.config to pass in other arguments
            stack.enter_context(
                patch_attr(
                    module,
                    "mlp",
                    replace_Qwen3MoE(config=model.config, module=module.mlp),
                )
            )


def update_gpt_oss_moe(model: PreTrainedModel, stack):
    @contextlib.contextmanager
    def replace_context(parent, name, module):
        linear = GptOssExpertsLinear(module)
        del module
        delete_offload_module(parent, name)
        register_offload_module(parent, name, linear)

        yield

        restored = linear.to_original()
        del linear
        delete_offload_module(parent, name)
        register_offload_module(parent, name, restored)

    # TODO: need to consider when replace module is duplicated in structure
    modules = list(model.named_modules())
    for name, module in tqdm.tqdm(modules, desc="Checking modules for replacements"):
        children = list(module.named_children())
        for child_name, child in children:
            if child.__class__.__name__ == "GptOssExperts":
                stack.enter_context(replace_context(module, child_name, child))
    


moe_context = {
    "Qwen3MoeForCausalLM": update_qwen3_moe,
    "GptOssForCausalLM": update_gpt_oss_moe,
}


def moe_calibration_context(model: PreTrainedModel, stack):
    # Temporarily updates the MoE modules within the context
    # Once the context exists, parameter updates persist
    cls_name = model.__class__.__name__
    if cls_name in moe_context:
        moe_context.get(cls_name)(model, stack)



# import torch
# from accelerate.hooks import (
#         AlignDevicesHook,
# def replace_offload_module(base: torch.nn.Module, name: str, module: torch.nn.Module):
#     hook = getattr(base, name)._hf_hook
#     delete_offload_module(base, name)

#     weights_map = PrefixedDataset(
#         hook.weights_map.dataset, prefix=f"{hook.weights_map.prefix.remove_suffix(name + ".")}"
#     )

#     parent_hook = AlignDevicesHook(
#         execution_device=hook.execution_device,
#         offload=hook.offload,
#         io_same_device=False,
#         weights_map=weights_map,
#         offload_buffers=offload_buffers,
#         place_submodules=place_submodules,
#         skip_keys=None,
#         tied_params_map=hook.tied_params_map,
#     )
