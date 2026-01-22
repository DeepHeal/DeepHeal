import torch
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_feature_importance(model, input_tensors, layer_index):
    """
    Compute the importance of each input feature to the specified layer's neurons.

    Parameters:
    - model (nn.Module): The model containing the layers.
    - input_tensors (torch.Tensor): The input data.
    - layer_index (int): The index of the layer to visualize.

    Returns:
    - importance_matrix (numpy.ndarray): The computed importance matrix (n_features, n_neurons).
    """

    if layer_index >= len(model.encoder):
        raise ValueError(f"layer_index {layer_index} is greater than the total number of layers in the encoder {len(model.encoder)}")

    # Ensure input requires grad
    if not input_tensors.requires_grad:
        input_tensors.requires_grad = True

    x = input_tensors

    if model.use_importance_mask:
        importance_weights = model.get_importance_weights()
        x = x * importance_weights

    # Forward pass until the specified layer
    # model.encoder is assumed to be an nn.Sequential
    for idx, layer in enumerate(model.encoder):
        x = layer(x)
        if idx == layer_index:
            break

    encoded_output = x
    n_input_features = input_tensors.size(1)
    n_encoded_neurons = encoded_output.size(1)

    importance_matrix = torch.zeros((n_input_features, n_encoded_neurons))

    # Iterate over each encoded neuron to compute gradients
    # Note: This loop can be slow for many neurons.
    # For very large layers, consider batched gradient computation if possible,
    # but for typical latent dims (e.g. 32-128) this is acceptable.
    for i in range(n_encoded_neurons):
        model.zero_grad()
        # Sum of activation of neuron i across the batch
        encoded_output[:, i].sum().backward(retain_graph=True)
        # Gradient w.r.t input features, averaged over batch
        if input_tensors.grad is not None:
            importance_matrix[:, i] = input_tensors.grad.mean(dim=0)
            input_tensors.grad.zero_()
        else:
             logger.warning(f"No gradient for neuron {i}. Check connectivity.")

    return importance_matrix.cpu().numpy()

def prepare_ranked_list(importance_matrix, feature_names, expression_df=None):
    """
    Prepare a ranked list of genes based on their importance weights for each neuron.

    Parameters:
    - importance_matrix (numpy.ndarray): The importance matrix with shape (n_input_features, n_encoded_neurons).
    - feature_names (list or pd.Index): List of feature names corresponding to rows of importance_matrix.
    - expression_df (pd.DataFrame, optional): Expression data (samples x features) to compute mean expression levels.
                                             Columns must match feature_names.

    Returns:
    - ranked_lists (dict): A dictionary with neuron names as keys and ranked gene lists (DataFrames) as values.
    """

    n_neurons = importance_matrix.shape[1]
    encoded_neuron_names = [f'Neuron {i}' for i in range(n_neurons)]

    if len(feature_names) != importance_matrix.shape[0]:
        raise ValueError(f"Length of feature_names ({len(feature_names)}) must match number of rows in importance_matrix ({importance_matrix.shape[0]})")

    df_importance = pd.DataFrame(importance_matrix, index=feature_names, columns=encoded_neuron_names)

    if expression_df is not None:
        # Check if features align
        common_features = [f for f in feature_names if f in expression_df.columns]
        if len(common_features) < len(feature_names):
             logger.warning("Not all feature names found in expression_df. Statistics might be incomplete.")

        # Compute stats on the passed df
        # Reindex to ensure order matches feature_names
        expr_subset = expression_df.reindex(columns=feature_names)

        expr_levels = expr_subset.mean(axis=0).values
        nonzero_levels = (expr_subset > 0).mean(axis=0).values

        df_importance['Expression Level'] = expr_levels
        df_importance['Nonzero Fraction'] = nonzero_levels

    ranked_lists = {}
    for neuron in encoded_neuron_names:
        # Sort by importance
        ranked_list = df_importance[[neuron]].sort_values(by=neuron, ascending=False)
        ranked_list = ranked_list.rename(columns={neuron: 'Importance'})

        if expression_df is not None:
             ranked_list = ranked_list.join(df_importance[['Expression Level', 'Nonzero Fraction']])

        ranked_list.index.name = 'Gene'
        ranked_list = ranked_list.reset_index()

        ranked_lists[neuron] = ranked_list

    return ranked_lists
