import matplotlib.pyplot as plt

# TODO: add arg for model size
# TODO: implements differents backend for plotting (plotly etc.)


def plot_token_to_token(matrices, tokens, label, **kwargs):
    """
    Generates a token-to-token matplotlib plot. The purpose of kwargs are used to setup matplotlib parameter.

    Args:
        matrices (_type_): _description_
        tokens (_type_): _description_
        label (_type_): _description_

    Returns:
        _type_: _description_
    """
    fig = plt.figure(figsize=kwargs.get("figsize", (20, 20)))
    fontdict = kwargs.get("fontdict", {"fontsize": 7})

    for idx, matrice in enumerate(matrices):
        ax = fig.add_subplot(4, 3, idx + 1)
        im = ax.imshow(matrice, cmap=kwargs.get("cmap", "viridis"))

        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(
            tokens, fontdict=fontdict, rotation=kwargs.get("rotation", 90)
        )
        ax.set_yticklabels(tokens, fontdict=fontdict)
        ax.set_title(
            f"{label} {idx + 1}",
            fontsize=fontdict["fontsize"] + 2,
        )
        fig.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return plt.gcf()


def plot_token_to_head(matrices, tokens):
    """
    Generates a token-to-head matplotlib plot.

    Args:
        matrices (_type_): _description_
        tokens (_type_): _description_

    Returns:
        _type_: _description_
    """
    fig = plt.figure(figsize=(15, 15))

    for idx, matrice in enumerate(matrices):
        ax = fig.add_subplot(6, 2, idx + 1)
        im = ax.matshow(matrice, cmap="viridis")

        fontdict = {"fontsize": 8}

        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(matrice)))
        ax.set_xticklabels(tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(range(len(matrice)), fontdict=fontdict)
        ax.set_xlabel(f"Layer {idx + 1}")
        fig.colorbar(im, fraction=0.046, pad=0.04)

    # plt.tight_layout()
    return plt.gcf()


# def visualize_attention_matrices(
#     device, model, description, tokenizer, layer, preprocessing_func=None
# ):
#     """
#     Vizualize attention matrices given a model, a tokenizer a layer and a description
#     """

#     if preprocessing_func is not None:
#         description = preprocessing_func(description)
#     all_tokens = tokenizer.convert_ids_to_tokens(tokenizer(description)["input_ids"])
#     batch = tokenizer(
#         description,
#         truncation=True,
#         padding=True,
#         max_length=512,
#         return_attention_mask=True,
#         return_tensors="pt",
#     )
#     attentions = model(
#         batch["input_ids"].to(device),
#         batch["attention_mask"].to(device),
#     ).attentions
#     output_attentions_all = torch.stack(attentions)
#     _visualize_token2token_scores(
#         output_attentions_all[layer].squeeze().detach().cpu().numpy(),
#         all_tokens,
#         "Head",
#     )


# def visualize_head_norm_matrices(
#     device, model, description, tokenizer, preprocessing_func=None
# ):
#     """
#     Vizualize L2 norm across head axis for all layers given a model, a tokenizer and a description
#     """

#     if preprocessing_func is not None:
#         description = preprocessing_func(description)
#     all_tokens = tokenizer.convert_ids_to_tokens(tokenizer(description)["input_ids"])
#     batch = tokenizer.encode_plus(
#         description,
#         truncation=True,
#         padding=True,
#         max_length=512,
#         return_attention_mask=True,
#         return_tensors="pt",
#     )
#     attentions = model(
#         batch["input_ids"].to(device),
#         batch["attention_mask"].to(device),
#     ).attentions
#     output_attentions_all = torch.stack(attentions)
#     _visualize_token2token_scores(
#         torch.norm(output_attentions_all, dim=2).squeeze().detach().cpu().numpy(),
#         all_tokens,
#         "Layer",
#     )


# def _forward(inputs, model, ids, device):
#     pred = model(
#         ids["input_ids"].to(device),
#         ids["attention_mask"].to(device))
#     return pred.logits.max(1).values


# def visualize_vector_norm(
#     device, model, description, tokenizer, preprocessing_func=None, compute="norm", layer=None
# ):
#     if preprocessing_func is not None:
#         description = preprocessing_func(description)
#     all_tokens = tokenizer.convert_ids_to_tokens(tokenizer(description)["input_ids"])
#     ids = tokenizer.encode_plus(
#         description,
#         truncation=True,
#         padding=True,
#         max_length=512,
#         return_attention_mask=True,
#         return_tensors="pt",
#     )
#     output = model(
#         ids["input_ids"].to(device),
#         ids["attention_mask"].to(device),
#     )
#     attentions = output.attentions
#     output_attentions_all = torch.stack(attentions)

#     output_attentions_all_shape = output_attentions_all.shape
#     batch = output_attentions_all_shape[1]
#     num_heads = output_attentions_all_shape[2]
#     head_size = 64
#     all_head_size = 768

#     layers = [
#         model.base_model.encoder.layer[layer].attention.self.value
#         for layer in range(len(model.base_model.encoder.layer))
#     ]

#     input_embeddings = output.hidden_states[0]

#     la = captum.attr.LayerActivation(_forward, layers)
#     value_layer_acts = la.attribute(input_embeddings, additional_forward_args=(
#         model,
#         ids,
#         device
#     ))
#     # shape -> layer x batch x seq_len x all_head_size
#     value_layer_acts = torch.stack(value_layer_acts)

#     new_x_shape = value_layer_acts.size()[:-1] + (num_heads, head_size)
#     value_layer_acts = value_layer_acts.view(*new_x_shape)

#     # layer x batch x neum_heads x 1 x head_size
#     value_layer_acts = value_layer_acts.permute(0, 1, 3, 2, 4)

#     value_layer_acts = value_layer_acts.permute(0, 1, 3, 2, 4).contiguous()
#     value_layer_acts_shape = value_layer_acts.size()

#     # layer x batch x seq_length x num_heads x 1 x head_size
#     value_layer_acts = value_layer_acts.view(value_layer_acts_shape[:-1] + (1, value_layer_acts_shape[-1],))

#     dense_acts = torch.stack([dlayer.attention.output.dense.weight for dlayer in model.base_model.encoder.layer])
#     dense_acts = dense_acts.view(len(layers), all_head_size, num_heads, head_size)

#     # layer x num_heads x head_size x all_head_size
#     dense_acts = dense_acts.permute(0, 2, 3, 1).contiguous()

#     # layers, batch, seq_length, num_heads, 1, all_head_size
#     f_x = torch.stack([value_layer_acts_i.matmul(dense_acts_i) for value_layer_acts_i, dense_acts_i in zip(value_layer_acts, dense_acts)])

#     # layer x batch x seq_length x num_heads x 1 x all_head_size)
#     f_x_shape = f_x.size()
#     f_x = f_x.view(f_x_shape[:-2] + (f_x_shape[-1],))
#     f_x = f_x.permute(0, 1, 3, 2, 4).contiguous()

#     #(layers x batch, num_heads, seq_length, all_head_size)
#     f_x_shape = f_x.size()


#     # ||f(x)||
#     #(layers x batch, num_heads, seq_length)
#     f_x_norm = torch.linalg.norm(f_x, dim=-1)
#     if compute == "norm":
#         _visualize_token2head_scores(f_x_norm.squeeze().detach().cpu().numpy(), all_tokens)


#     # ||alpha * f(x)||
#     # layer x batch x num_heads x seq_length x seq_length x all_head_size
#     alpha_f_x = torch.einsum('lbhks,lbhsd->lbhksd', output_attentions_all, f_x)

#     # layer x batch x num_heads x seq_length x seq_length
#     alpha_f_x_norm = torch.linalg.norm(alpha_f_x, dim=-1)
#     if compute == "alpha-norm":
#         _visualize_token2token_scores(
#             alpha_f_x_norm[layer].squeeze().detach().cpu().numpy(),
#             all_tokens,
#             "Head"
#         )

#     # || SUM alpha * f(x)||
#     summed_alpha_f_x = alpha_f_x.sum(dim=2)

#     # layers x batch x seq_length x seq_length
#     summed_alpha_f_x_norm = torch.linalg.norm(summed_alpha_f_x, dim=-1)
#     if compute == "sum-norm":
#         _visualize_token2token_scores(
#             summed_alpha_f_x_norm.squeeze().cpu().detach().numpy(),
#             all_tokens,
#             "Layer"
#         )
