import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepEnsemble(nn.Module):
    def __init__(self, split_n, num_classes, hidden_size):
        super(DeepEnsemble, self).__init__()
        self.split_n = split_n
        self.num_classes = num_classes

        # Meta-classifier (a neural network)
        self.meta_classifier = nn.Sequential(
            nn.Linear(split_n * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, class_logits):
        # class_logits shape: [batch_size, split_n, num_classes]
        batch_size = class_logits.shape[0]

        # Flatten the logits for the meta-classifier
        flattened_logits = class_logits.view(batch_size, -1)  # [batch_size, split_n * num_classes]

        # Meta-classifier
        final_logits = self.meta_classifier(flattened_logits)  # [batch_size, num_classes]

        return final_logits

class SplitSelfAttentionWithHeads(nn.Module):
    def __init__(self, split_n, num_classes, num_layers=12, cla_head=DeepEnsemble):
        super(SplitSelfAttentionWithHeads, self).__init__()
        self.vocab_size = 32100
        self.embed_size = 768
        self.split_n = split_n
        self.split_dim = 768 // split_n
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.cla_head = cla_head(split_n=split_n,num_classes=num_classes,hidden_size=self.split_dim)

        assert (
            self.split_dim * split_n == self.embed_size
        ), "Embedding size needs to be divisible by split_n"

        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        self.multi_layer_self_attentions = nn.ModuleList([
            MultiLayerSelfAttention(self.split_dim, num_layers) for _ in range(split_n)
        ])

        # Separate classification heads for each split
        self.classification_heads = nn.ModuleList([
            nn.Linear(self.split_dim, num_classes) for _ in range(split_n)
        ])

    def forward(self, input_ids, attention_mask):
        # x shape: [batch_size, sequence_length]
        batch_size, sequence_length = input_ids.shape

        # Embedding the input
        input_ids = self.embedding(input_ids)  # [batch_size, sequence_length, embed_size]

        # Split the embedding into `split_n` parts
        input_ids = input_ids.view(batch_size, sequence_length, self.split_n, self.split_dim)
        input_ids = input_ids.permute(0, 2, 1, 3)  # [batch_size, split_n, sequence_length, split_dim]

        # Apply separate Self-Attention and classification heads to each split
        class_logits = []
        for i in range(self.split_n):
            # Self-Attention for the i-th split
            split_out = self.multi_layer_self_attentions[i](input_ids[:, i, :, :], attention_mask)  # [batch_size, sequence_length, split_dim]

            # Classification head for the i-th split
            split_out = split_out.mean(dim=1)  # Mean over sequence length: [batch_size, split_dim]
            class_logits.append(split_out)

        # Stack the classification results
        class_logits = torch.stack(class_logits, dim=1)  # [batch_size, split_n, num_classes]

        return self.cla_head(class_logits)

class MultiLayerSelfAttention(nn.Module):
    def __init__(self, split_dim, num_layers):
        super(MultiLayerSelfAttention, self).__init__()
        self.split_dim = split_dim
        self.num_layers = num_layers

        # Stack multiple Self-Attention layers
        self.layers = nn.ModuleList([
            SingleHeadSelfAttention(split_dim) for _ in range(num_layers)
        ])

        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(split_dim) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        # x shape: [batch_size, sequence_length, split_dim]
        for i in range(self.num_layers):
            # Self-Attention
            attn_out = self.layers[i](x, mask)

            # Residual connection and layer normalization
            x = self.layer_norms[i](x + attn_out)

        return x


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, split_dim):
        super(SingleHeadSelfAttention, self).__init__()
        self.split_dim = split_dim

        # Linear layers for values, keys, and queries
        self.values = nn.Linear(split_dim, split_dim, bias=False)
        self.keys = nn.Linear(split_dim, split_dim, bias=False)
        self.queries = nn.Linear(split_dim, split_dim, bias=False)

    def forward(self, x, mask=None):
        # x shape: [batch_size, sequence_length, split_dim]
        batch_size, sequence_length, _ = x.shape

        # Split into queries, keys, and values
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        # Scaled dot-product attention
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / (self.split_dim ** 0.5)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)
            mask_matrix = mask & mask.transpose(-2, -1)
            energy = energy.masked_fill(mask_matrix == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.matmul(attention, values)  # [batch_size, sequence_length, split_dim]

        return out



