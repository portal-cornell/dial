import torch
import torch.nn as nn
import torch.nn.functional as F
from dial.model import Model
from dial.arguments import Arguments

def select_loss(outputs: torch.Tensor, targets: torch.Tensor, args: Arguments):
    if args.loss_type == "binary_cross_entropy":
        extra_metrics = {"accuracy": ((outputs > 0) == targets).float().mean().item()}
        loss = F.binary_cross_entropy_with_logits(outputs, targets.float())
    elif args.loss_type == "mse":
        extra_metrics = {}
        loss = F.mse_loss(outputs, targets.float())
    elif args.loss_type == "cross_entropy":
        extra_metrics = {"accuracy": (outputs.argmax(dim=1) == targets).float().mean().item()}
        loss = F.cross_entropy(outputs, targets.flatten())
    else:
        raise ValueError(f"Invalid loss type: {args.loss_type}")
    return loss, extra_metrics

def contrastive_loss(batch, model: Model, args: Arguments):
    (prompts, responses_chosen, responses_rejected), _ = batch
    inputs_chosen = [prompt+response for prompt, response in zip(prompts, responses_chosen)]
    inputs_rejected = [prompt+response for prompt, response in zip(prompts, responses_rejected)]
    outputs_chosen, embeddings_chosen = model.run_class(inputs_chosen)
    outputs_rejected, embeddings_rejected = model.run_class(inputs_rejected)

    loss = -F.logsigmoid((outputs_chosen - outputs_rejected)).mean()
    extra_metrics = {
        "accuracy": (outputs_chosen > outputs_rejected).float().mean().item(),
    }

    outputs = torch.stack([outputs_chosen, outputs_rejected], dim=-1)
    return loss, extra_metrics, outputs

def mcq_loss(batch, model: Model, args: Arguments):
    all_strings = batch[0]
    prompts = all_strings[0]
    responses_chosen = all_strings[1]
    all_responses_rejected = all_strings[2:]
    inputs_chosen = [prompt+response for prompt, response in zip(prompts, responses_chosen)]
    inputs_rejected = [[prompt+response for prompt, response in zip(prompts, responses_rejected)] for responses_rejected in all_responses_rejected]
    outputs_chosen, embeddings_chosen = model.run_class(inputs_chosen)
    outputs_rejected = [model.run_class(inputs_rejected_i)[0] for inputs_rejected_i in inputs_rejected]

    outputs_rejected = torch.stack(outputs_rejected, dim=1)

    loss = -F.logsigmoid((outputs_chosen[:, None] - outputs_rejected)).mean()
    extra_metrics = {
        "train_all_accuracy": (outputs_chosen > outputs_rejected.max(dim=1).values).float().mean().item(),
        "accuracy": (outputs_chosen[:, None] > outputs_rejected).float().mean().item(),
    }

    outputs = torch.cat([outputs_chosen[:, None], outputs_rejected], dim=1)
    return loss, extra_metrics, outputs

def sequence_prediction_loss(batch, model: Model, args: Arguments):
    inputs, targets = batch
    if len(inputs) == 2:
        prompts, responses = inputs
        inputs = [prompt+response for prompt, response in zip(prompts, responses)]
    else:
        inputs = inputs[0]
    outputs, embeddings = model.run_class(inputs)
    targets = targets.to(embeddings.device)
    loss, extra_metrics = select_loss(outputs, targets, args)
    return loss, extra_metrics, outputs

def domain_adaptation_loss(batch, model: Model, args: Arguments):
    source_inputs, source_targets, target_inputs, target_targets = batch
    if len(source_inputs) == 2:
        prompts, responses = source_inputs
        source_inputs = [prompt+response for prompt, response in zip(prompts, responses)]
    else:
        source_inputs = source_inputs[0]
    
    if len(target_inputs) == 2:
        prompts, responses = target_inputs
        target_inputs = [prompt+response for prompt, response in zip(prompts, responses)]
    else:
        target_inputs = target_inputs[0]
    source_outputs, source_embeddings = model.run_class(source_inputs)
    target_outputs, target_embeddings = model.run_class(target_inputs)
    source_targets, target_targets = source_targets.to(source_embeddings.device), target_targets.to(target_embeddings.device)
    source_loss, source_extra_metrics = select_loss(source_outputs, source_targets, args)
    target_loss, target_extra_metrics = select_loss(target_outputs, target_targets, args)
    all_embeddings = model.noise_layer(torch.cat([source_embeddings, target_embeddings], dim=0))
    domain_labels = torch.cat([torch.zeros(len(source_embeddings)), torch.ones(len(target_embeddings))]).to(all_embeddings.device)
    da_loss, _ = model.da_base.get_da_loss([all_embeddings], domain_labels)
    loss = source_loss + da_loss
    extra_metrics = {
        "source_accuracy": source_extra_metrics.get("accuracy", 0),
        "target_accuracy": target_extra_metrics.get("accuracy", 0),
        "da_loss": da_loss.item(),
        "source_loss": source_loss.item(),
        "target_loss": target_loss.item(),
    }
    return loss, extra_metrics, source_outputs

def domain_adaptation_contrastive_loss(batch, model: Model, args: Arguments):
    source_inputs, _, target_inputs, _ = batch
    (source_prompts, source_responses_chosen, source_responses_rejected) = source_inputs
    (target_prompts, target_responses_chosen, target_responses_rejected) = target_inputs

    source_inputs_chosen = [prompt+response for prompt, response in zip(source_prompts, source_responses_chosen)]
    source_inputs_rejected = [prompt+response for prompt, response in zip(source_prompts, source_responses_rejected)]

    target_inputs_chosen = [prompt+response for prompt, response in zip(target_prompts, target_responses_chosen)]
    target_inputs_rejected = [prompt+response for prompt, response in zip(target_prompts, target_responses_rejected)]

    source_outputs_chosen, source_embeddings_chosen = model.run_class(source_inputs_chosen)
    source_outputs_rejected, source_embeddings_rejected = model.run_class(source_inputs_rejected)
    target_outputs_chosen, target_embeddings_chosen = model.run_class(target_inputs_chosen)
    target_outputs_rejected, target_embeddings_rejected = model.run_class(target_inputs_rejected)

    source_loss = -F.logsigmoid((source_outputs_chosen - source_outputs_rejected)).mean()
    target_loss = -F.logsigmoid((target_outputs_chosen - target_outputs_rejected)).mean()
    source_outputs = torch.stack([source_outputs_chosen, source_outputs_rejected], dim=-1)
    target_outputs = torch.stack([target_outputs_chosen, target_outputs_rejected], dim=-1)
    all_outputs = [source_outputs, target_outputs]

    source_embeddings = model.noise_layer(torch.cat([source_embeddings_chosen, source_embeddings_rejected], dim=0))
    target_embeddings = model.noise_layer(torch.cat([target_embeddings_chosen, target_embeddings_rejected], dim=0))

    all_embeddings = model.noise_layer(torch.cat([source_embeddings, target_embeddings], dim=0))
    domain_labels = torch.cat([torch.zeros(len(source_embeddings)), torch.ones(len(target_embeddings))]).to(all_embeddings.device)
    da_loss, additional_losses = model.da_base.get_da_loss([all_embeddings], domain_labels)

    additional_losses["embed_losses"] = additional_losses["embed_losses"][0]
    additional_losses = {k: v.item() for k, v in additional_losses.items()}

    loss = source_loss + da_loss
    extra_metrics = {
        "source_accuracy": ((source_outputs_chosen > source_outputs_rejected).float()).mean().item(),
        "target_accuracy": ((target_outputs_chosen > target_outputs_rejected).float()).mean().item(),
        "da_loss": da_loss.item(),
        "source_loss": source_loss.item(),
        "target_loss": target_loss.item(),
        **additional_losses,
    }
    return loss, extra_metrics, all_outputs

def oracle_loss(batch, model: Model, args: Arguments):
    source_inputs, _, target_inputs, _ = batch
    (source_prompts, source_responses_chosen, source_responses_rejected) = source_inputs
    (target_prompts, target_responses_chosen, target_responses_rejected) = target_inputs

    source_inputs_chosen = [prompt+response for prompt, response in zip(source_prompts, source_responses_chosen)]
    source_inputs_rejected = [prompt+response for prompt, response in zip(source_prompts, source_responses_rejected)]

    target_inputs_chosen = [prompt+response for prompt, response in zip(target_prompts, target_responses_chosen)]
    target_inputs_rejected = [prompt+response for prompt, response in zip(target_prompts, target_responses_rejected)]

    source_outputs_chosen, _ = model.run_class(source_inputs_chosen)
    source_outputs_rejected, _ = model.run_class(source_inputs_rejected)
    target_outputs_chosen, _ = model.run_class(target_inputs_chosen)
    target_outputs_rejected, _ = model.run_class(target_inputs_rejected)

    source_loss = -F.logsigmoid((source_outputs_chosen - source_outputs_rejected)).mean()
    target_loss = -F.logsigmoid((target_outputs_chosen - target_outputs_rejected)).mean()
    source_outputs = torch.stack([source_outputs_chosen, source_outputs_rejected], dim=-1)
    target_outputs = torch.stack([target_outputs_chosen, target_outputs_rejected], dim=-1)
    all_outputs = [source_outputs, target_outputs]

    loss = source_loss + target_loss
    extra_metrics = {
        "source_accuracy": ((source_outputs_chosen > source_outputs_rejected).float()).mean().item(),
        "target_accuracy": ((target_outputs_chosen > target_outputs_rejected).float()).mean().item(),
        "source_loss": source_loss.item(),
        "target_loss": target_loss.item(),
    }
    return loss, extra_metrics, all_outputs

def baseline_loss(batch, model: Model, args: Arguments):
    source_inputs, _, target_inputs, _ = batch
    (source_prompts, source_responses_chosen, source_responses_rejected) = source_inputs
    source_inputs_chosen = [prompt+response for prompt, response in zip(source_prompts, source_responses_chosen)]
    source_inputs_rejected = [prompt+response for prompt, response in zip(source_prompts, source_responses_rejected)]
    source_outputs_chosen, _ = model.run_class(source_inputs_chosen)
    source_outputs_rejected, _ = model.run_class(source_inputs_rejected)
    source_loss = -F.logsigmoid((source_outputs_chosen - source_outputs_rejected)).mean()
    source_outputs = torch.stack([source_outputs_chosen, source_outputs_rejected], dim=-1)

    (target_prompts, target_responses_chosen, target_responses_rejected) = target_inputs
    target_inputs_chosen = [prompt+response for prompt, response in zip(target_prompts, target_responses_chosen)]
    target_inputs_rejected = [prompt+response for prompt, response in zip(target_prompts, target_responses_rejected)]

    additional_losses = {}
    if args.source_sft:
        sft_loss = model.run_sft(source_prompts, source_responses_chosen, dual_model=True).causal_loss
        additional_losses["source_sft_loss"] = sft_loss.item()
    
    if args.target_ntp:
        ntp_loss = model.run_ntp(target_inputs_chosen + target_inputs_rejected, dual_model=True).causal_loss
        additional_losses["target_ntp_loss"] = ntp_loss.item()

    loss = source_loss
    for key, value in additional_losses.items():
        loss += value
    extra_metrics = {
        "source_accuracy": ((source_outputs_chosen > source_outputs_rejected).float()).mean().item(),
        "source_loss": source_loss.item(),
        **additional_losses,
    }
    return loss, extra_metrics, source_outputs # all_outputs

def domain_adaptation_mcq_loss(batch, model: Model, args: Arguments):
    source_all_strings, _, target_all_strings, _ = batch
    
    source_prompts = source_all_strings[0]
    source_responses_chosen = source_all_strings[1]
    source_all_responses_rejected = source_all_strings[2:]
    source_inputs_chosen = [prompt+response for prompt, response in zip(source_prompts, source_responses_chosen)]
    source_inputs_rejected = [[prompt+response for prompt, response in zip(source_prompts, responses_rejected)] for responses_rejected in source_all_responses_rejected]
    
    target_prompts = target_all_strings[0]
    target_responses_chosen = target_all_strings[1]
    target_all_responses_rejected = target_all_strings[2:]
    target_inputs_chosen = [prompt+response for prompt, response in zip(target_prompts, target_responses_chosen)]
    target_inputs_rejected = [[prompt+response for prompt, response in zip(target_prompts, responses_rejected)] for responses_rejected in target_all_responses_rejected]
    
    source_outputs_chosen, source_embeddings_chosen = model.run_class(source_inputs_chosen)
    target_outputs_chosen, target_embeddings_chosen = model.run_class(target_inputs_chosen)

    source_full_outputs_rejected = [model.run_class(inputs_rejected_i) for inputs_rejected_i in source_inputs_rejected]
    source_outputs_rejected = torch.stack([x[0] for x in source_full_outputs_rejected], dim=1)
    source_embeddings_rejected = torch.cat([x[1] for x in source_full_outputs_rejected], dim=0)

    target_full_outputs_rejected = [model.run_class(inputs_rejected_i) for inputs_rejected_i in target_inputs_rejected]
    target_outputs_rejected = torch.stack([x[0] for x in target_full_outputs_rejected], dim=1)
    target_embeddings_rejected = torch.cat([x[1] for x in target_full_outputs_rejected], dim=0)

    source_loss = -F.logsigmoid((source_outputs_chosen[:, None] - source_outputs_rejected)).mean()
    target_loss = -F.logsigmoid((target_outputs_chosen[:, None] - target_outputs_rejected)).mean()
    source_outputs = torch.cat([source_outputs_chosen[:, None], source_outputs_rejected], dim=1)
    target_outputs = torch.cat([target_outputs_chosen[:, None], target_outputs_rejected], dim=1)
    all_outputs = [source_outputs, target_outputs]

    source_embeddings = model.noise_layer(torch.cat([source_embeddings_chosen, source_embeddings_rejected], dim=0))
    target_embeddings = model.noise_layer(torch.cat([target_embeddings_chosen, target_embeddings_rejected], dim=0))

    all_embeddings = model.noise_layer(torch.cat([source_embeddings, target_embeddings], dim=0))
    domain_labels = torch.cat([torch.zeros(len(source_embeddings)), torch.ones(len(target_embeddings))]).to(all_embeddings.device)
    da_loss, additional_losses = model.da_base.get_da_loss([all_embeddings], domain_labels)

    additional_losses["embed_losses"] = additional_losses["embed_losses"][0]
    additional_losses = {k: v.item() for k, v in additional_losses.items()}

    loss = source_loss + da_loss
    extra_metrics = {
        "source_accuracy": ((source_outputs_chosen[:, None] > source_outputs_rejected).float()).mean().item(),
        "source_train_all_accuracy": ((source_outputs_chosen > source_outputs_rejected.max(dim=1).values).float()).mean().item(),
        "target_accuracy": ((target_outputs_chosen[:, None] > target_outputs_rejected).float()).mean().item(),
        "target_train_all_accuracy": ((target_outputs_chosen > target_outputs_rejected.max(dim=1).values).float()).mean().item(),
        "da_loss": da_loss.item(),
        "source_loss": source_loss.item(),
        "target_loss": target_loss.item(),
        **additional_losses,
    }
    return loss, extra_metrics, all_outputs

str_to_loss_mapping = {
    "sequence_prediction": sequence_prediction_loss,
    "domain_adaptation": domain_adaptation_loss,
    "contrastive": contrastive_loss,
    "domain_adaptation_contrastive": domain_adaptation_contrastive_loss,
    "mcq": mcq_loss,
    "domain_adaptation_mcq": domain_adaptation_mcq_loss,
    "baseline": baseline_loss,
    "oracle": oracle_loss,
}

def get_loss_fn(args: Arguments, mode="train"):
    mode = args.train_mode if mode == "train" else args.eval_mode
    return str_to_loss_mapping[mode]