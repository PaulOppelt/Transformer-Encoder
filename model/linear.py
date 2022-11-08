import torch.nn as nn
import torch
from torch import Tensor

class LinearClassification(nn.Module):
    r"""
    implement a linear classifier that can be used to classify the output of the bert model
    args:
        d_model: embedding dimension of the input tokens
        n_domains_incl: number of domains in the input
        window_size: size of the input window
        n_labels: number of labels in the classification task
    """
    def __init__(self, d_model: int, n_domains_incl: int, window_size: int, n_labels: int) -> None:
        super().__init__()

        self.n_domains_incl = n_domains_incl
        self.window_size = window_size
        self.d_model = d_model

        self.fc = nn.Linear(d_model, d_model * self.n_domains_incl)
        self.activ = torch.nn.functional.gelu
        self.drop = nn.Dropout(p=0.4)
        self.norm = nn.LayerNorm(d_model * self.n_domains_incl)
        self.classifier = nn.Linear(d_model * self.n_domains_incl, n_labels)

    def forward(self, input: "Tensor") -> "Tensor":
        r"""
        Args:
            input: Tensor, shape: batch, sequence_lenght, d_model
        Returns:
            output: Tensor, shape: batch, n_labels, sequence_lenght
        """
        input = input[:,:: self.n_domains_incl] # take every n-th element in the second dimension (sequence length)
        pooled_h = self.norm(self.activ(self.fc(input)))  # [0]
        logits = self.classifier(pooled_h)
        logits = torch.permute(
            logits, (1, 2, 0)
        )  
        return logits

