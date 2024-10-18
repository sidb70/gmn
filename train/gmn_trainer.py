import torch.nn as nn
import torch
from train.utils import split

class GMNTrainer:
    def __init__(self,model, device: torch.device):
        self.model = model
        self.device = device
        
    def train_epoch(self, feats, labels, batch_size, criterion, optimizer):
        epoch_running_loss =0 
        num_batches = 0
        for i in range(0, len(feats), batch_size):
            outs = []
            for j in range(i, min(i + batch_size, len(feats))):
                #print("j: ", type(feats[j][0]), type(feats[j][1]))
                (node_feat, edge_index, edge_feat), hpo_vec = feats[j]
                node_feat, edge_index, edge_feat, hpo_vec = (
                    torch.tensor(node_feat).to(self.device),
                    torch.tensor(edge_index).to(self.device),
                    torch.tensor(edge_feat).to(self.device),
                    torch.tensor(hpo_vec).to(self.device),
                )
                out = self.model.forward(node_feat, edge_index, edge_feat, hpo_vec)
                outs.append(out)
            outs = torch.cat(outs, dim=1).squeeze(0).to(self.device)
            y = torch.tensor(labels[i : i + batch_size]).to(self.device)
            loss = criterion(outs, y)
            epoch_running_loss += loss.item()
            num_batches += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("Loss: ", loss.item())
            # print("Predictions: ", outs)
            # print("Labels: ", y)
        print("Epoch Training Loss: ", epoch_running_loss / num_batches)

    def eval_step(self, feats, labels, batch_size, criterion):
        self.model.eval()
        # freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        val_running_loss = 0
        num_batches = 0
        for i in range(0, len(feats), batch_size):
            outs = []
            for j in range(i, min(i + batch_size, len(feats))):
                (node_feat, edge_index, edge_feat), hpo_vec = feats[j]
                node_feat, edge_index, edge_feat, hpo_vec = (
                    torch.tensor(node_feat).to(self.device),
                    torch.tensor(edge_index).to(self.device),
                    torch.tensor(edge_feat).to(self.device),
                    torch.tensor(hpo_vec).to(self.device),
                )
                out = self.model.forward(node_feat, edge_index, edge_feat, hpo_vec)
                outs.append(out)
            outs = torch.cat(outs, dim=1).squeeze(0).to(self.device)
            y = torch.tensor(labels[i : i + batch_size]).to(self.device)
            loss = criterion(outs, y)
            val_running_loss += loss.item()
            num_batches += 1
        print("Validation Loss: ", val_running_loss / num_batches)
        return val_running_loss / num_batches
    
    def train(self, features, labels, epochs, batch_size, lr, valid_size=0.1, test_size=0.1):
        train_set, valid_set, test_set = split( features, labels, test_size, valid_size)
        train_feats, train_labels = train_set
        valid_feats, valid_labels = valid_set
        test_feats, test_labels = test_set  
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            self.train_epoch(train_feats, train_labels, batch_size, criterion, optimizer)
            val_loss = self.eval_step(valid_feats, valid_labels, batch_size, criterion)
            print("Validation Loss: ", val_loss)

        test_loss = self.eval_step(test_feats, test_labels, batch_size, criterion)
        print("Test Loss: ", test_loss)
 


class GMNHPOTrainer(GMNTrainer):
    def __init__(self,model, device: torch.device, hpo_grad_steps):
        super().__init__(model, device)
        self.hpo_grad_steps = hpo_grad_steps

    def eval_step(self, feats, labels, batch_size, criterion):
        self.model.eval()
        # freeze model
        self.model.requires_grad = False
        val_running_loss = 0
        num_batches = 0
        for i in range(0, len(feats), batch_size):
            outs = []
            for j in range(i, min(i + batch_size, len(feats))):
                (node_feat, edge_index, edge_feat), hpo_vec = feats[j]
                node_feat, edge_index, edge_feat, hpo_vec = (
                    torch.tensor(node_feat, dtype=torch.float32).to(self.device),
                    torch.tensor(edge_index).to(self.device),
                    torch.tensor(edge_feat, dtype=torch.float32).to(self.device),
                    torch.tensor(hpo_vec).to(self.device),
                )
                # enable grad for hpo_vec
                hpo_vec.requires_grad = True
                for _ in range(self.hpo_grad_steps):
                    out = self.model(node_feat, edge_index, edge_feat, hpo_vec)
                    loss = criterion(out, torch.tensor(labels[j]).to(self.device))
                    loss.backward()
                    hpo_vec.data = hpo_vec.data - 0.01 * hpo_vec.grad
                    hpo_vec.grad.zero_()
                with torch.no_grad():
                    out = self.model(node_feat, edge_index, edge_feat, hpo_vec)
                    outs.append(out)
            outs = torch.cat(outs, dim=1).squeeze(0).to(self.device)
            y = torch.tensor(labels[i : i + batch_size]).to(self.device)
            loss = criterion(outs, y)
            val_running_loss += loss.item()
            num_batches += 1
        print("Validation Loss: ", val_running_loss / num_batches)
        # unfreeze model
        self.model.requires_grad = True
        return val_running_loss / num_batches
 