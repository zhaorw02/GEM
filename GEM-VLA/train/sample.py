from collections import defaultdict

import torch
import torch.nn.functional as F


@torch.no_grad()
def log_sample_res(
    vision_language_model, selected_layers,
    rdt, args,
    accelerator, weight_dtype, dataloader, logger
):
    logger.info(
        f"Running sampling for {args.num_sample_batches} batches..."
    )

    rdt.eval()

    loss_for_log = defaultdict(float)
    loss_counter = defaultdict(int)
    for step, batch in enumerate(dataloader):
        if step >= args.num_sample_batches:
            break

        actions = batch["actions"].to(dtype=weight_dtype)
        states = batch["states"].to(dtype=weight_dtype)

        # Qwen3VL forward to get hidden states
        lang_attn_mask = batch["vision_language_model_inputs"]["attention_mask"].to(dtype=torch.bool)
        vlm_outputs = vision_language_model(
            **batch["vision_language_model_inputs"],
            output_hidden_states=True,
            use_cache=False,
        )

        if isinstance(selected_layers, list):
            lang_hidden = torch.stack(
                [vlm_outputs.hidden_states[i] for i in selected_layers],
                dim=1,
            )  # (B, depth, seq_len, vlm_hidden_size)
        else:
            lang_hidden = vlm_outputs.hidden_states[selected_layers]
            # (B, seq_len, vlm_hidden_size)

        pred_actions = rdt.predict_action(
            lang_tokens=lang_hidden,
            lang_attn_mask=lang_attn_mask,
            state_tokens=states,
        )

        loss = F.mse_loss(pred_actions, actions, reduction='none').float()

        mse_loss = loss.mean()
        mse_loss = accelerator.gather(mse_loss).mean().item()
        loss_for_log["sample_mse"] += mse_loss
        loss_counter["sample_mse"] += 1

    for name in loss_for_log:
        loss_for_log[name] = round(loss_for_log[name] / loss_counter[name], 4)

    rdt.train()
    torch.cuda.empty_cache()

    return dict(loss_for_log)
