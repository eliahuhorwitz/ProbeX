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
        :param num_classes: The number for the classifier (in the notation of the paper this is d_Y).
        :param n_probes: The number of probes to use (r_U in the paper).
        :param proj_dim: The dimension into which the probe responses should be projected into (r_V in the paper).
        :param rep_dim: The dimension of the representation (r_T in the paper).
        """
        super(ProbeXZeroshot, self).__init__()
        self.probes = nn.Linear(n_probes, input_shape[1], bias=False)
        self.shared_probe_proj = nn.Linear(input_shape[0], proj_dim, bias=False)

        self.per_probe_encoders = nn.Linear(proj_dim * n_probes, rep_dim)
        self.regression_head = nn.Linear(clip_dim, rep_dim)

    def forward(self, x, clip_embedding):
        probe_responses = x @ self.probes.weight
        probe_responses_projected = self.shared_probe_proj(probe_responses.transpose(1, 2))

        probe_responses_projected = torch.relu(probe_responses_projected)
        probe_responses_projected_flat = probe_responses_projected.reshape(probe_responses_projected.shape[0], -1)
        representation = self.per_probe_encoders(probe_responses_projected_flat)

        clip_z = self.regression_head(clip_embedding)
        clip_z = clip_z / clip_z.norm(dim=-1, keepdim=True)
        representation = representation / representation.norm(dim=-1, keepdim=True)

        logits = representation @ clip_z.T
        return logits


    # TODO: Implement the extract_representation method for the ProbeXZeroshot model. (Look at what the commented code below did)
    def extract_representation(self, x):
        raise NotImplementedError("This method is not implemented for the ProbeXZeroshot model.")

#
#
#
#
# class ProbesNet(nn.Module):
#     def __init__(self, input_shape, n_probes=64, n_features=128, clip_dim=768):
#         super(ProbesNet, self).__init__()
#         num_probes = int(input_shape[1] * n_probes)
#         num_features = int(input_shape[0] * n_features)
#         self.W_in = nn.Linear(input_shape[1], num_probes, bias=False)
#         self.W_r = nn.Linear(input_shape[0], num_features, bias=False)
#         # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.clip_to_proj = nn.Linear(clip_dim, num_features)
#         self.regression_head = nn.Linear(num_features * num_probes, num_features)
#
#     def forward(self, x, clip_img):
#         probes = self.W_in(x).transpose(1, 2)
#         W_z = self.W_r(probes)
#         # checking if activation helps
#         W_z = torch.relu(W_z)
#         W_z = W_z.reshape(W_z.shape[0], -1)
#         W_z = self.regression_head(W_z)
#
#         clip_z = self.clip_to_proj(clip_img)
#
#         clip_z = clip_z / clip_z.norm(dim=-1, keepdim=True)
#         W_z = W_z / W_z.norm(dim=-1, keepdim=True)
#
#         logits_per_weights = W_z @ clip_z.T
#         return logits_per_weights
#
#     def get_W_features(self, x):
#         probes = self.W_in(x).transpose(1, 2)
#         W_z = self.W_r(probes)
#         # todo: remember to do activation if we use it eventually
#         W_z = W_z.reshape(W_z.shape[0], -1)
#         W_z = self.regression_head(W_z)
#         return W_z