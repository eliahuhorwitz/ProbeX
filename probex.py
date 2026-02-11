import torch
from torch import nn


class ProbeXClassification(nn.Module):
    def __init__(self, input_shape, num_classes, n_probes, proj_dim, rep_dim):
        """
        :param input_shape: The shape of the input weight matrix (2d matrix) (in the notation of the paper this has shape [d_H, d_W]).
        :param num_classes: The number for the classifier (in the notation of the paper this is d_Y).
        :param n_probes: The number of probes to use (r_U in the paper).
        :param proj_dim: The dimension into which the probe responses should be projected into (r_V in the paper).
        :param rep_dim: The dimension of the representation (r_T in the paper).
        """
        super(ProbeXClassification, self).__init__()
        self.probes = nn.Linear(n_probes, input_shape[1], bias=False)
        self.shared_probe_proj = nn.Linear(input_shape[0], proj_dim, bias=False)

        self.per_probe_encoders = nn.Linear(proj_dim * n_probes, rep_dim)
        self.classification_head = nn.Linear(rep_dim, num_classes)

    def forward(self, x):
        probe_responses = x @ self.probes.weight
        probe_responses_projected = self.shared_probe_proj(probe_responses.transpose(1, 2))

        probe_responses_projected = torch.relu(probe_responses_projected)
        probe_responses_projected_flat = probe_responses_projected.reshape(probe_responses_projected.shape[0], -1)
        representation = self.per_probe_encoders(probe_responses_projected_flat)

        logits = self.classification_head(representation)
        return logits


class ProbeXZeroshot(nn.Module):
    def __init__(self, input_shape, n_probes, proj_dim, rep_dim, clip_dim=768):
        """
        :param input_shape: The shape of the input weight matrix (2d matrix) (in the notation of the paper this has shape [d_H, d_W]).
        :param n_probes: The number of probes to use (r_U in the paper).
        :param proj_dim: The dimension into which the probe responses should be projected into (r_V in the paper).
        :param rep_dim: The dimension of the representation (r_T in the paper).
        :param clip_dim: The dimension of the CLIP text embeddings (default: 768 for ViT-L/14).
        """
        super(ProbeXZeroshot, self).__init__()
        self.probes = nn.Linear(n_probes, input_shape[1], bias=False)
        self.shared_probe_proj = nn.Linear(input_shape[0], proj_dim, bias=False)

        self.per_probe_encoders = nn.Linear(proj_dim * n_probes, rep_dim)
        self.regression_head = nn.Linear(clip_dim, rep_dim)

    def forward(self, x, clip_embedding):
        representation = self.extract_representation(x)

        clip_z = self.regression_head(clip_embedding)
        clip_z = clip_z / clip_z.norm(dim=-1, keepdim=True)
        representation = representation / representation.norm(dim=-1, keepdim=True)

        logits = representation @ clip_z.T
        return logits

    def extract_representation(self, x):
        probe_responses = x @ self.probes.weight
        probe_responses_projected = self.shared_probe_proj(probe_responses.transpose(1, 2))

        probe_responses_projected = torch.relu(probe_responses_projected)
        probe_responses_projected_flat = probe_responses_projected.reshape(probe_responses_projected.shape[0], -1)
        representation = self.per_probe_encoders(probe_responses_projected_flat)
        
        return representation