import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from hydra.utils import instantiate

torch.set_float32_matmul_precision('medium')

'''
*** problem setup (bag-level supervision + weak/noisy instance-level labels) ***
* multiple instances learning: predict node-level also do graph-level
* label noise problem: node radius does not cover the annotation => no node is labeled as positive (node label is noisy),
  => but this graph is actually positive  (graph label is trusted)
* possible criterion for loss function: consistency(node label, graph label)

*** strategy ***
* MIL pooling for graph supervision
* Confidence-weighted node supervision
* Pseudo-label EM
* Ranking at graph level
* graph = bag of instances
'''

def graph_pairwise_ranking_loss(graph_logits, graph_labels, margin=1.0):
    """
    Pairwise ranking loss for graphs in a batch.
    Encourages positive graphs to have higher scores than negative graphs.

    Args:
        graph_logits (Tensor): Shape [batch_size, 1], raw graph scores.
        graph_labels (Tensor): Shape [batch_size, 1], binary labels {0, 1}.
        margin (float): Margin for ranking loss.

    Returns:
        Tensor: Scalar ranking loss.
    """
    pos_scores = graph_logits[graph_labels.squeeze() == 1]
    neg_scores = graph_logits[graph_labels.squeeze() == 0]

    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return torch.tensor(0.0, device=graph_logits.device)

    diff = neg_scores.view(-1, 1) - pos_scores.view(1, -1) + margin
    return torch.clamp(diff, min=0).mean()


class GNNClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        """
        Lightning module for node-level and graph-level classification
        with weak/noisy labels and consistency regularization.

        Args:
            model (nn.Module): Graph neural network model.
            cfg (OmegaConf): Configuration with training parameters.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.cfg = cfg

        # Node-level classification loss with pos_weight
        pos_weight = torch.ones([1]) * cfg.pos_weight
        self.node_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Graph-level classification loss
        self.graph_loss_fn = nn.BCEWithLogitsLoss()

        # Metrics
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_node_auroc = torchmetrics.AUROC(task="binary")

        self.automatic_optimization = False

    def training_step(self, data, _):
        node_labels = data.y.float()
        graph_labels = data.cls_labels.view(-1, 1).float()

        node_logits, graph_logits = self.model(data)

        # Node-level loss (handles noisy labels)
        node_loss_raw = self.node_loss_fn(node_logits[:, 0], node_labels)
        node_loss = node_loss_raw.mean()

        # Graph-level MIL loss
        graph_loss = self.graph_loss_fn(graph_logits, graph_labels)

        # Graph-level ranking loss
        ranking_loss = graph_pairwise_ranking_loss(graph_logits, graph_labels, margin=self.cfg.margin)

        # Consistency regularization: node probabilities should align with graph labels
        batch = data.batch
        node_probs = torch.sigmoid(node_logits[:, 0])
        graph_labels_expanded = graph_labels[batch]
        consistency_loss = nn.BCELoss()(node_probs, graph_labels_expanded.squeeze())

        # Total loss with weighted components
        loss = (
            node_loss
            + self.cfg.lambda_graph * graph_loss
            + self.cfg.lambda_ranking * ranking_loss
            + self.cfg.lambda_consistency * consistency_loss
        )

        # Manual optimization step
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        # Logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_node_loss", node_loss, on_step=False, on_epoch=True)
        self.log("train_graph_loss", graph_loss, on_step=False, on_epoch=True)
        self.log("train_ranking_loss", ranking_loss, on_step=False, on_epoch=True)
        self.log("train_consistency_loss", consistency_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, data, _):
        node_labels = data.y.float()
        cls_labels = data.cls_labels.view(-1, 1).float()

        node_logits, graph_logits = self.model(data)

        node_loss = self.node_loss_fn(node_logits[:, 0], node_labels)
        graph_loss = self.graph_loss_fn(graph_logits, cls_labels)

        # Metrics
        self.val_cls_auroc.update(graph_logits.detach(), cls_labels.long())
        self.val_node_auroc.update(node_logits.detach(), node_labels.long())

        self.log("val_loss", node_loss + graph_loss, on_step=False, on_epoch=True, prog_bar=True)
        return node_loss + graph_loss

    def on_validation_epoch_start(self):
        self.val_cls_auroc.reset()
        self.val_node_auroc.reset()

    def on_validation_epoch_end(self):
        self.log("val_cls_auroc", self.val_cls_auroc.compute(), prog_bar=True)
        self.log("val_node_auroc", self.val_node_auroc.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())
        return {"optimizer": optimizer}