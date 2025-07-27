from torch import nn
import torch
import torchmetrics as tm
from models.layer import BasicPointLayer, EdgeGraphConvBlock, DilatedEdgeGraphConvBlock, ResidualBasicPointLayer, \
    PointFeatureImportance, STNkd
import lightning as L


class TunableDilatedToothSegmentationNetwork(nn.Module):
    def __init__(self, num_classes=17, feature_dim=24, 
                 edge_conv_channels=24, edge_conv_k=32, edge_conv_hidden=24,
                 dilated_channels=60, dilated_k=32, dilated_hidden=60,
                 dilation_k_values=[200, 900, 1800],
                 global_hidden=1024, res_channels=512, res_hidden=512,
                 dropout_rate=0.1):
        """
        Tunable version of the dilated tooth segmentation network
        """
        super(TunableDilatedToothSegmentationNetwork, self).__init__()
        self.num_classes = num_classes

        self.stnkd = STNkd(k=feature_dim)

        # Edge Graph Conv blocks with tunable parameters
        self.edge_graph_conv_block1 = EdgeGraphConvBlock(
            in_channels=feature_dim, 
            out_channels=edge_conv_channels, 
            k=edge_conv_k,
            hidden_channels=edge_conv_hidden,
            edge_function="local_global"
        )
        self.edge_graph_conv_block2 = EdgeGraphConvBlock(
            in_channels=edge_conv_channels, 
            out_channels=edge_conv_channels, 
            k=edge_conv_k,
            hidden_channels=edge_conv_hidden,
            edge_function="local_global"
        )
        self.edge_graph_conv_block3 = EdgeGraphConvBlock(
            in_channels=edge_conv_channels, 
            out_channels=edge_conv_channels, 
            k=edge_conv_k,
            hidden_channels=edge_conv_hidden,
            edge_function="local_global"
        )

        # Local hidden layer
        self.local_hidden_layer = BasicPointLayer(
            in_channels=edge_conv_channels * 3, 
            out_channels=dilated_channels
        )

        # Dilated Edge Graph Conv blocks with tunable parameters
        self.dilated_edge_graph_conv_block1 = DilatedEdgeGraphConvBlock(
            in_channels=dilated_channels, 
            hidden_channels=dilated_hidden,
            out_channels=dilated_channels, 
            k=dilated_k,
            dilation_k=dilation_k_values[0], 
            edge_function="local_global"
        )
        self.dilated_edge_graph_conv_block2 = DilatedEdgeGraphConvBlock(
            in_channels=dilated_channels, 
            hidden_channels=dilated_hidden,
            out_channels=dilated_channels, 
            k=dilated_k,
            dilation_k=dilation_k_values[1], 
            edge_function="local_global"
        )
        self.dilated_edge_graph_conv_block3 = DilatedEdgeGraphConvBlock(
            in_channels=dilated_channels, 
            hidden_channels=dilated_hidden,
            out_channels=dilated_channels, 
            k=dilated_k,
            dilation_k=dilation_k_values[2], 
            edge_function="local_global"
        )

        # Global hidden layer
        self.global_hidden_layer = BasicPointLayer(
            in_channels=dilated_channels * 4, 
            out_channels=global_hidden
        )

        # Feature importance
        self.feature_importance = PointFeatureImportance(in_channels=global_hidden)

        # Residual blocks
        self.res_block1 = ResidualBasicPointLayer(
            in_channels=global_hidden, 
            out_channels=res_channels, 
            hidden_channels=res_hidden
        )
        self.res_block2 = ResidualBasicPointLayer(
            in_channels=res_channels, 
            out_channels=res_channels//2, 
            hidden_channels=res_hidden//2
        )
        
        # Output layer
        self.out = BasicPointLayer(
            in_channels=res_channels//2, 
            out_channels=num_classes, 
            is_out=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, pos):
        # precompute pairwise distance of points
        cd = torch.cdist(pos, pos)
        x = self.stnkd(x)

        # Edge Graph Conv blocks
        x1, _ = self.edge_graph_conv_block1(x, pos)
        x1 = self.dropout(x1)
        x2, _ = self.edge_graph_conv_block2(x1)
        x2 = self.dropout(x2)
        x3, _ = self.edge_graph_conv_block3(x2)
        x3 = self.dropout(x3)

        x = torch.cat([x1, x2, x3], dim=2)
        x = self.local_hidden_layer(x)

        # Dilated Edge Graph Conv blocks
        x1, _ = self.dilated_edge_graph_conv_block1(x, pos, cd=cd)
        x1 = self.dropout(x1)
        x2, _ = self.dilated_edge_graph_conv_block2(x1, pos, cd=cd)
        x2 = self.dropout(x2)
        x3, _ = self.dilated_edge_graph_conv_block3(x2, pos, cd=cd)
        x3 = self.dropout(x3)

        x = torch.cat([x, x1, x2, x3], dim=2)
        x = self.global_hidden_layer(x)

        x = self.feature_importance(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.out(x)
        return x


class LitTunableDilatedToothSegmentationNetwork(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create model with configuration
        self.model = TunableDilatedToothSegmentationNetwork(
            num_classes=config['num_classes'],
            feature_dim=config['feature_dim'],
            edge_conv_channels=config['edge_conv_channels'],
            edge_conv_k=config['edge_conv_k'],
            edge_conv_hidden=config['edge_conv_hidden'],
            dilated_channels=config['dilated_channels'],
            dilated_k=config['dilated_k'],
            dilated_hidden=config['dilated_hidden'],
            dilation_k_values=config['dilation_k_values'],
            global_hidden=config['global_hidden'],
            res_channels=config['res_channels'],
            res_hidden=config['res_hidden'],
            dropout_rate=config['dropout_rate']
        )
        
        # Loss function
        if config['loss_function'] == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        elif config['loss_function'] == 'focal_loss':
            class FocalLoss(nn.Module):
                def __init__(self, alpha=0.25, gamma=2.0):
                    super(FocalLoss, self).__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                
                def forward(self, inputs, targets):
                    ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
                    pt = torch.exp(-ce_loss)
                    focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                    return focal_loss.mean()
            
            self.loss = FocalLoss(
                alpha=config['focal_alpha'], 
                gamma=config['focal_gamma']
            )
        
        # Metrics
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=config['num_classes'])
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=config['num_classes'])
        self.train_miou = tm.JaccardIndex(task="multiclass", num_classes=config['num_classes'])
        self.val_miou = tm.JaccardIndex(task="multiclass", num_classes=config['num_classes'])
        
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, C = x.shape
        x = x.float()
        y = y.view(B, N).float()
        pred = self.model(x, pos)
        pred = pred.transpose(2, 1)
        loss = self.loss(pred, y.long())
        self.train_acc(pred, y)
        self.train_miou(pred, y)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_miou", self.train_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, C = x.shape
        x = x.float()
        y = y.view(B, N).float()
        pred = self.model(x, pos)
        pred = pred.transpose(2, 1)
        loss = self.loss(pred, y.long())
        self.val_acc(pred, y)
        self.val_miou(pred, y)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss 
    
    def test_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, C = x.shape
        x = x.float()
        y = y.view(B, N).float()
        pred = self.model(x, pos)
        pred = pred.transpose(2, 1)
        loss = self.loss(pred, y.long())
        self.val_acc(pred, y)
        self.val_miou(pred, y)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def predict_labels(self, data):
        with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
            with torch.no_grad():
                pos, x, y = data
                pos = pos.unsqueeze(0).to(self.device)
                x = x.unsqueeze(0).to(self.device)
                B, N, C = x.shape
                x = x.float()
                y = y.view(B, N).float()
                pred = self.model(x, pos)
                pred = pred.transpose(2, 1)
                pred = torch.argmax(pred, dim=1)
                return pred.squeeze()

    def configure_optimizers(self):
        # Optimizer
        if self.config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.config['learning_rate'], 
                betas=(self.config['beta1'], self.config['beta2']), 
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.config['learning_rate'], 
                betas=(self.config['beta1'], self.config['beta2']), 
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.config['learning_rate'], 
                momentum=0.9, 
                weight_decay=self.config['weight_decay']
            )
        
        # Scheduler
        if self.config['scheduler'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.config['scheduler_step_size'], 
                gamma=self.config['scheduler_gamma'], 
                verbose=True
            )
        elif self.config['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=100, 
                eta_min=1e-6
            )
        elif self.config['scheduler'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=self.config['scheduler_gamma'], 
                patience=10, 
                verbose=True
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        } 