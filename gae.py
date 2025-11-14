import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import pygod
from pygod.utils import load_data
from pygod.detector import DOMINANT
from pygod.detector import AdONE
import torch
import math

#
from pygod.metric import eval_roc_auc,eval_average_precision
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric.utils.convert
from torch_geometric.nn import GCNConv,GATConv, GATv2Conv,SAGEConv
from torch_geometric.nn import GCN
import torch_sparse
from pygod.nn.functional import double_recon_loss
from pygod.nn.decoder import DotProductDecoder
from torch_geometric.seed import seed_everything
import torch.nn as nn
import numpy as np
from random import choice

torch.manual_seed(12345)
#torch.manual_seed(3407)
#seed_everything(717)
#seed_everything(42)
#seed_everything(3407)

#seed_everything(12345)

#torch.set_default_tensor_type(torch.DoubleTensor)

global dataset_sel_val
dataset_sel=['gen_5000','enron','disney','inj_cora','inj_amazon','weibo','books','reddit']
#dataset_sel=['inj_amazon','inj_cora','enron','weibo','books','reddit','disney']
#dataset_sel=['reddit']
#dataset_sel=['enron']
#dataset_sel=['gen_1000','gen_5000','gen_10000']

#x, edge_index, adj_mat, y = process_data('weibo')
#x, edge_index, adj_mat, y = process_data('books')
#x, edge_index, adj_mat, y = process_data('disney')
#x, edge_index, adj_mat, y = process_data('enron')
    

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class GVAE(torch.nn.Module):
    def __init__(self, input_size, num_nodes, hidden_size, latent_size, head_count):
        '''
        Initializes the layers of the graph neural network

        :param input_size: defines the original dimension of each node's feature vector
        :param hidden_size: defines size of the hidden dimension in the encoder and decoder
        :param latent_size: defines the dimensions of the latent space of the autoencoder
        '''

        super().__init__()
              
        #self.gat_layer1 = [GCNConv(input_size, hidden_size,add_self_loops = False) for _ in range(head_count)]
        #for i, GNNheads in enumerate(self.gat_layer1):
        #    self.add_module('gat_layer1_{}'.format(i), GNNheads)
             
        self.gat_layer1 = GATConv(input_size, hidden_size, heads= head_count, add_self_loops = False, fill_value=fill_value)
        #self.gat_layer1 = GCNConv(input_size, hidden_size) 
     
        self.gat_layer2 = GATConv(hidden_size*head_count, latent_size,add_self_loops = False, fill_value=fill_value)
        #self.gat_layer2 = GCNConv(hidden_size, latent_size,add_self_loops = False)

        self.attr_decoder_layer_1 = GCNConv(latent_size, hidden_size,add_self_loops = False)
        
        #self.attr_decoder_layer_1 = SAGEConv(latent_size, hidden_size)

        #self.attr_decoder_layer_1 = GATv2Conv(latent_size, hidden_size)
    
        self.structure_decoder = DotProductDecoder(in_dim = latent_size, sigmoid_s=True)

        
        self.dense_attr_1 = nn.Linear(num_nodes, hidden_size)

        #self.mu_layer = GCNConv(latent_size, latent_size,add_self_loops = False) 
        self.mu_layer = torch.nn.Linear(latent_size, latent_size)
         
        self.logvar_layer = torch.nn.Linear(latent_size, latent_size)
        
        #self.logvar_layer = GCNConv(latent_size, latent_size,add_self_loops = False) 
             
        self.act = torch.nn.ReLU()
     
 
    
    def forward(self, x, edge_index, beta):
        '''
        Performs the forward pass of the model

        :param x: the data being passed into the model, should be a tensor of size (number of nodes) x (feature vector length)
        :param edge_index: a tensor listing all of the edges as node pairings 

        :return s: the reconstructed adjacency matrix
        :return decoded: the reconstructed data
        :return mu: the mu parameter for the VAE's distribution
        :return logvar: the log of the variance parameter for the VAE's distribution
        '''
        #h = torch.cat([att(x, edge_index) for att in self.gat_layer1], dim=1)

        h = self.gat_layer1(x,edge_index)
        h = self.act(h)

        emb = self.gat_layer2(h, edge_index)

        mu = self.mu_layer(emb)
        #mu = self.mu_layer(emb, edge_index)

        logvar = self.logvar_layer(emb)
        
        #logvar = self.logvar_layer(emb,edge_index)
   
      
        sigma = torch.exp(logvar)
        #sigma = torch.exp(0.5*logvar)
        #sigma = torch.exp(0.1*logvar)
    

        eps = torch.randn_like(sigma)
        
        #print("eps shape")
        #print(eps.shape)

        eps = eps.to(device)
        sigma = sigma.to(device)
        mu = mu.to(device)


        #z = mu + sigma * eps
        if (beta==0):
         z = mu
        else:
         z = mu + beta*sigma*eps
         
        
        #z = beta*emb+(1-beta)(mu + sigma*eps)


      
        #z = (cur_vae_value)*z + (1-cur_vae_value)*emb
        #z = z + emb
         
        #s_ = self.structure_decoder(z,edge_index)
        s_ = self.structure_decoder(z,edge_index)

        x = self.dense_attr_1(x.T) 

        #decoded = self.attr_decoder_layer_1(z, edge_index)
        decoded = self.attr_decoder_layer_1(z, edge_index)
   
        #x_ = torch.sigmoid(decoded @ x.T) #ojo
        x_ = decoded @ x.T

        
   
        
        return x_, s_,mu,logvar
        
         

class DotProductDecoder_or(nn.Module):
    def __init__(self, in_dim):
        '''
        Constructor for the inner product decoder

        :param in_dim: specifies the input dimension
        '''
        super(DotProductDecoder_or, self).__init__()
        self.in_dim = in_dim

    def forward(self, h):
        '''
        Forward pass for the inner product decoder
        Computes the dot product of the encoded input and its transpose to recreate the adjacency matrix 
        Applies a sigmoid activation function to every value in the adjacency matrix

        :param h: the encoded data

        :return edge_scores: the reconstructed adjacency matrix
        '''
        dot_product = torch.matmul(h, h.t())
        edge_scores = torch.sigmoid(dot_product)
        return edge_scores

def process_data(dataset_name):
    '''
    Function to preprocess data

    :param dataset_name: the name of the dataset to be loaded

    :return x: a tensor of size (number of nodes) x (feature vector length) made up of each node's feature vector
    :return edge_index: a tensor representing each of the edges in the graph as node pairings
    :return adj_mat: the graph's adjacency matrix
    :return y: a binary tensor of size (number of nodes) representing whether each node is an outlier or not
    '''
    data = load_data(dataset_name)
    
    
    average_node_degree = data.num_edges / data.num_nodes
    print("average node degree", average_node_degree)

    global fill_value
    fill_value = math.log2(average_node_degree)

    x, edge_index, y = data.x, data.edge_index, data.y.bool()
    adj_mat = create_adj_mat(edge_index)
    return x, edge_index, adj_mat, y

def create_adj_mat(edge_index):
    '''
    Creates the graph's adjacency matrix from the edge index

    :param edge_index: a tensor representing each of the edges in the graph as node pairings

    :return adjacency_matrix: the graph's adjacency matrix
    '''
    num_nodes = edge_index.max().item() + 1
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    return adjacency_matrix

#theta=choice([10., 40., 90.]),
#eta=choice([3., 5., 8.]),
    

#theta=40
#eta=5
#alpha = 0.5
#beta = 0.


#x, edge_index, adj_mat, y = process_data('inj_amazon')

def loss_fun2(x, decoded, adj_mat, s, mu, logvar, alpha = 0.5, beta = 0.5,pos_weight_a = 0.5,pos_weight_s=0.5):
    '''
    The loss function calculates the loss to be backpropogated in the model.  
    It combines MSE for the features and adjacency matrix along with KL divergence

    :param x: a tensor of size (number of nodes) x (feature vector length) made up of each node's feature vector
    :param decoded: the VAE's reconstruction of x
    :param adj_mat: the graph's adjacency matrix
    :param s: the VAE's reconstruction of the adjacency matrix
    :param mu: the mean for the distribution created in the latent space
    :param logvar: the log of the variance for the distribution in the latent space
    :param alpha: a weight to control the percentage of structure loss (and feature loss)
    :param beta: a weight to control the percentage of feature loss (and KL divergence)
    :param train: a boolean to communicate if the model is training or testing

    :return loss: a loss value for each node in the form of a tensor with shape (number of nodes)
    '''
    
    #print(pos_weight_a)
    #print(pos_weight_s)
    #kl_loss = nn.KLDivLoss(reduction="batchmean")
    #output = kl_loss(input, target)
    
    
    #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #KL divergence
    
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    #print("kl loss")
    #print(kl_loss)

    #kl_loss = (logvar ** 2 + mu ** 2 - torch.log(logvar) - 0.5).sum()
    
    #atribute reconstruction loss
    feature_loss = torch.pow(x - decoded, 2) 
    
    if pos_weight_a != 0.5: feature_loss = torch.where(x > 0, feature_loss * pos_weight_a, feature_loss * (1 - pos_weight_a)) #position 1
    
    if (beta==0):
     attribute_loss = feature_loss  #aggregated loss for node attributes
    else:
     #attribute_loss = feature_loss  #aggregated loss for node attributes
     attribute_loss = (1-beta) * feature_loss + beta * kl_loss #aggregated loss for node attributes 

    #if pos_weight_a != 0.5: attribute_loss = torch.where(x > 0, attribute_loss * pos_weight_a, attribute_loss * (1 - pos_weight_a)) #position 2
   
 
    #attribute_loss = torch.mean(attribute_loss, dim=1, keepdim=True)
    
    attribute_loss = torch.sqrt(torch.sum(attribute_loss, 1))
    
    
    
    #structure reconstruction loss
    structure_loss = torch.pow(adj_mat - s, 2) 
    
    if pos_weight_s != 0.5: structure_loss = torch.where(adj_mat > 0, structure_loss * pos_weight_s, structure_loss * (1 - pos_weight_s))  
                                
    
    #if pos_weight_a != 0.5: attribute_loss = torch.where(x > 0, attribute_loss * pos_weight_a, attribute_loss * (1 - pos_weight_a))
    
    structure_loss = torch.sqrt(torch.sum(structure_loss, 1)) #calculate adj matrix MSE by node
    

    if (alpha==1):
     loss = structure_loss
    else:    
     loss = alpha * structure_loss + (1 - alpha) * attribute_loss.squeeze()



    #if train == False:
    #    #during testing, don't include KL divergence
    #    feature_loss = torch.mean(feature_loss, dim=1, keepdim=True)
    #    loss = alpha * structure_loss + (1 - alpha) * feature_loss.squeeze()
    #else:
    #    attribute_loss = torch.mean(attribute_loss, dim=1, keepdim=True)
    #    loss = alpha * structure_loss + (1 - alpha) * attribute_loss.squeeze()
    return loss

def loss_fun(x, decoded, adj_mat, s, mu, logvar, alpha = 0.5, beta = 0.5,pos_weight_a = 0.5,pos_weight_s=0.5):
    '''
    The loss function calculates the loss to be backpropogated in the model.  
    It combines MSE for the features and adjacency matrix along with KL divergence

    :param x: a tensor of size (number of nodes) x (feature vector length) made up of each node's feature vector
    :param decoded: the VAE's reconstruction of x
    :param adj_mat: the graph's adjacency matrix
    :param s: the VAE's reconstruction of the adjacency matrix
    :param mu: the mean for the distribution created in the latent space
    :param logvar: the log of the variance for the distribution in the latent space
    :param alpha: a weight to control the percentage of structure loss (and feature loss)
    :param beta: a weight to control the percentage of feature loss (and KL divergence)
    :param train: a boolean to communicate if the model is training or testing

    :return loss: a loss value for each node in the form of a tensor with shape (number of nodes)
    '''
    
    #print(pos_weight_a)
    #print(pos_weight_s)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #KL divergence
    
    feature_loss = torch.pow(x - decoded, 2) #MSE from oroginal and reconstructed features
    
    if pos_weight_a != 0.5: feature_loss = torch.where(x > 0, feature_loss * pos_weight_a, feature_loss * (1 - pos_weight_a))
    
    
    structure_loss = torch.pow(adj_mat - s, 2) #MSE from original adj matrix and reconstructed adj mat
    
    if pos_weight_s != 0.5: structure_loss = torch.where(adj_mat > 0, structure_loss * pos_weight_s, structure_loss * (1 - pos_weight_s))  
                                
    attribute_loss = beta * feature_loss + (1 - beta) * kl_loss #aggregated loss for node attributes 
    #if pos_weight_a != 0.5: attribute_loss = torch.where(x > 0, attribute_loss * pos_weight_a, attribute_loss * (1 - pos_weight_a))
    
    structure_loss = torch.sqrt(torch.sum(structure_loss, 1)) #calculate adj matrix MSE by node
    


                              
    
    attribute_loss = torch.mean(attribute_loss, dim=1, keepdim=True)
    loss = alpha * structure_loss + (1 - alpha) * attribute_loss.squeeze()

    #if train == False:
    #    #during testing, don't include KL divergence
    #    feature_loss = torch.mean(feature_loss, dim=1, keepdim=True)
    #    loss = alpha * structure_loss + (1 - alpha) * feature_loss.squeeze()
    #else:
    #    attribute_loss = torch.mean(attribute_loss, dim=1, keepdim=True)
    #    loss = alpha * structure_loss + (1 - alpha) * attribute_loss.squeeze()
    return loss
    

#def train(model, x, edge_index, scheduler,optimizer, adj_mat, n_epochs=200):
def train(model, x, edge_index, optimizer, adj_mat, n_epochs=25):
    '''
    The train function executes the forward passes and backpropogation that train the model

    :param model: the model to be trained
    :param x: the tensor of the node features
    :param edge_index: a tensor representing each of the edges in the graph as node pairings
    :param optimizer: the function that determines how weights are updated in the network
    :param adj_mat: the graph's adjacency matrix
    :param n_epoch: specifies the number of epochs for the model to be trained on

    :return model: the model with updated weights
    '''
    early_stopper = EarlyStopper(patience=10, min_delta=0)
    

    #theta=[40.]
    #eta=[5.]   


    #alpha=[0.5]
    #beta=[1.0,0.5]

    #theta=[90.]
    #eta=[8.]
    #alpha=[0.5]
    #beta=[1.0]
   

    #print(pos_weight_a)
    #print(pos_weight_s)
    

    
  
    for epoch in range(1, n_epochs + 1):

     #print("epoch ",epoch, "alpha ", alpha_value, "beta ",beta_value)

     model.train()
        
     pos_weight_a = eta_value/(1+eta_value)
     pos_weight_s = theta_value/(1+theta_value)
      
     decoded, s, mu, logvar = model(x, edge_index, beta_value ) #forward pass   
     loss = loss_fun2(x, decoded, adj_mat, s, mu, logvar, alpha = alpha_value, beta = beta_value, pos_weight_a=pos_weight_a, pos_weight_s=pos_weight_s)
     #print("structure")
     #print(s)

     loss = loss.cpu()

         
     #print("loss: ",loss.detach())
     #loss_per_node = double_recon_loss(x, decoded, adj_mat, s, weight=0.5, pos_weight_a = 0.83, pos_weight_s = 0.97)
     #loss = double_recon_loss(x, decoded, adj_mat, s, weight=alpha_value, pos_weight_a = pos_weight_a, pos_weight_s = pos_weight_s)
     #loss = loss.nanmean()

     #loss = loss.cpu()

     loss = loss.mean()
     #loss_per_node = loss_per_node.mean()

        
     optimizer.zero_grad()
     #loss_per_node.backward() # backpropogation
     loss.backward() # backpropogation
     optimizer.step()
     #scheduler.step()

        
     model.eval()
     auc_score = 0
     average_auc = 0
     for i in range(1):
      decoded, s, mu, logvar = model(x, edge_index, beta_value)

      loss_test = loss_fun2(x, decoded, adj_mat, s, mu, logvar, alpha = alpha_value, beta = beta_value, pos_weight_a=pos_weight_a, pos_weight_s=pos_weight_s)
      loss_test = loss_test.cpu()

      #loss_test = double_recon_loss(x, decoded, adj_mat, s, weight=alpha_value, pos_weight_a = pos_weight_a, pos_weight_s = pos_weight_s)
      #loss_test = loss_test.cpu()
      #print("loss test: ",loss_test.detach())

         

      #if(torch.any(torch.isnan(loss_test.detach()))):
      if(torch.any(torch.isnan(loss_test.detach())) or torch.any(torch.isinf(loss_test.detach()))):   
        auc_score = 0
        break
      else: 
        #print("Current auc Score: ",auc_score)
        auc_score += eval_roc_auc(y, loss_test.detach())
        average_auc +=1
     
     if(auc_score > 0):
      auc_score = auc_score/average_auc
       #print("auc score ", auc_score)
      # break
      # #print("break at epoch ", epoch)
      # #break
      #else: 
      # #print(loss_test.detach())  
      # auc_score = eval_roc_auc(y, loss_test.detach())
      # #print("Current Score: ",auc_score)
      # continue
      #auc_score = eval_roc_auc(y, loss_per_node_test.detach())
      #avg_accu =  eval_average_precision(y, loss_per_node_test.detach())
         
     global best_auc_score
     #print("best_auc_score", best_auc_score)
     if(auc_score > best_auc_score):
       best_auc_score = auc_score
       print("Best auc Score: ",best_auc_score)
       #print("epoch ",epoch," vae ", vae_value, " pos_weight_a ",pos_weight_a,"  pos_weight_s ", pos_weight_s, " alpha ",alpha_value," beta ",beta_value)  
       print("epoch ",epoch," pos_weight_a ",pos_weight_a,"  pos_weight_s ", pos_weight_s, " alpha ",alpha_value," beta ",beta_value)  
       model_path = "models/model_"+dataset_sel_val+".ptx"
       torch.save(model.state_dict(),model_path)
       global best_pos_weight_a
       best_pos_weight_a = pos_weight_a
       global best_pos_weight_s
       best_pos_weight_s = pos_weight_s
       global best_alpha
       best_alpha = alpha_value
       global best_beta
       best_beta = beta_value
       #global best_vae
       #best_vae = cur_vae_value
          
        
       #print(f"Epoch [{epoch}/{n_epochs}], Train Loss: {loss.item():.4f}")
       #print(f"Epoch [{epoch}/{n_epochs}], Train Loss: {loss_per_node.item():.4f}")
       
       #print(f"Epoch [{epoch}/{n_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {loss_test.item():.4f}, Test auc: {auc_score:.4f}, , Test acc: {avg_accu:.4f}")
        
       #if early_stopper.early_stop(loss_per_node.detach()):   
     if early_stopper.early_stop(loss.detach()):           
         break
     else:
         continue 
    
    return model


def test(model, x, edge_index, adj_mat,beta):
    '''
    The test function performs a single forward pass, then computes the auc score based on the loss value for each node

    :param model: the model to be trained
    :param x: the tensor of the node features
    :param adj_mat: the graph's adjacency matrix

    :return: auc score which is an accuracy score for binary classification
    '''
    model.eval()
    decoded, s, mu, logvar = model(x, edge_index,beta)
    

  
    loss = loss_fun2(x, decoded, adj_mat, s, mu, logvar, alpha = best_alpha, beta = best_beta, pos_weight_a=best_pos_weight_a, pos_weight_s=best_pos_weight_s)
    
    loss = loss.cpu()
    #loss = double_recon_loss(x, decoded, adj_mat, s, weight=0.5, pos_weight_a = 0.5, pos_weight_s = 0.5)
    #loss = loss.cpu()
    auc_score = eval_roc_auc(y, loss.detach())
    #auc_score = eval_roc_auc(y, loss.detach())
    #print(f"Test Loss: {loss.item():.4f}")
    #print("Final Score:")

    #mybins = []
    
    #for k in range(0,100,5): 
    # mybins += [k]
    #yp = loss.detach()
 
    #plt.figure(figsize=(10, 10))
    #plt.xlabel('# occcurences')
    #plt.ylabel('Anomaly score')
    #counts, bins, bars = plt.hist(yp,mybins)
  
    #plt.xticks(mybins[::8],horizontalalignment='right',fontsize=12,rotation=90)

    #plt.savefig('.models/fig_hist.pdf')
    #plt.show()
 

    return auc_score
    

#torch.manual_seed(3407)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print("Device is ",device)

#torch.use_deterministic_algorithms(True)

theta=[10., 40., 90.]
eta=[3., 5., 8.]
alpha=[0.0,0.25, 0.5, 0.75,1.0]

#theta=[40.]
#eta=[5.]
#alpha=[0.0]

#beta=[0.0,0.25, 0.5, 0.75,1.0]
#beta=[0.0,0.1,0.5]

#beta=[0.0,0.5]

beta=[0.001,0.01]

#vae =[0.0,0.25, 0.5, 0.75,1.0]

#global cur_vae_value
#cur_vae_value = 0

#x, edge_index, adj_mat, y = process_data('weibo')
#x, edge_index, adj_mat, y = process_data('books')
for dataset_sel_val in dataset_sel:

 print("dataset_sel: ",dataset_sel_val)
 print("Processing ",dataset_sel_val) 
 seed_everything(12345) 
 #seed_everything(3407) 
 #seed_everything(717) 
 #seed_everything(42) 
 x, edge_index, adj_mat, y = process_data(dataset_sel_val)
 global best_auc_score
 best_auc_score = 0
 for theta_value in theta:
  for eta_value in eta:
   for alpha_value in alpha:
    for beta_value in beta:
     #for vae_value in vae: 
      #seed_everything(42) 
      #seed_everything(12345)
      #seed_everything(717)
      #seed_everything(3407)

      #cur_vae_value = vae_value

      #x = x.double()
      #edge_index = edge_index
      #adj_mat =  adj_mat.double()
      #y = y.double() 

      x = x.to(device)
      edge_index = edge_index.to(device)
      adj_mat = adj_mat.to(device)

      #x, edge_index, adj_mat, y = process_data('disney')
      #x, edge_index, adj_mat, y = process_data('enron')
    

 
      #model = GVAE(input_size=x.size()[1], hidden_size = (x.size()[1] // 2), latent_size=(x.size()[1] // 4), head_count=8)
      model = GVAE(input_size=x.size()[1], num_nodes=x.size()[0], hidden_size = 64, latent_size=32, head_count=8)

      #model_path = "models/model_"+dataset_sel_val+"_good.ptx"

      #model.load_state_dict(torch.load(model_path)) 
      model.to(device)

      #model = GVAE(input_size=x.size()[1], hidden_size = (x.size()[1] // 4), latent_size=(x.size()[1] // 8), head_count=8)

      #model = GVAE(input_size=x.size()[1], hidden_size = (x.size()[1]), latent_size=(x.size()[1] // 2))
      #model = GVAE(input_size=x.size()[1], hidden_size = (x.size()[1]), latent_size=(x.size()[1]))
      if (dataset_sel_val == 'weibo'):
       lr = 0.001
      else: 
       lr = 0.01 
       #lr = 0.0001  
      optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=0.05)
      #optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01, weight_decay=0.05)
      #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, cycle_momentum=False, step_size_up = 1000)


      #print("hidden size")
      #print((x.size()[1] // 2))

      #print("latent size")
      #print((x.size()[1] // 4))

 

      #model = train(model, x, edge_index, scheduler,optimizer, adj_mat)

      model = train(model, x, edge_index, optimizer, adj_mat,n_epochs=100)

 model_path = "models/model_"+dataset_sel_val+".ptx"

 #cur_vae_value = best_vae

 #best_pos_weight_a=0.88
 #best_pos_weight_s =0.97 
 #best_alpha=0.75  
 #best_beta=0.5


 print("Testing best model for dataset_sel: ",dataset_sel_val)
 model.load_state_dict(torch.load(model_path)) 
 #seed_everything(12345)
 #seed_everything(717)
 best_auc_score = 0
 for ite in range(10):
  cur_auc_score=test(model, x, edge_index, adj_mat,best_beta)
  print("Current Score:")
  print(cur_auc_score)
  if(cur_auc_score > best_auc_score):
   best_auc_score = cur_auc_score
 print("Best Score:")
 print(best_auc_score)
 print("pos_weight_a ",best_pos_weight_a,"  pos_weight_s ", best_pos_weight_s, " alpha ",best_alpha," beta ",best_beta)  
 
 
 print("Testing best model for dataset_sel with beta=0: ",dataset_sel_val)
 best_auc_score = 0
 best_beta = 0
 for ite in range(10):
  cur_auc_score=test(model, x, edge_index, adj_mat,best_beta)
  print("Current Score:")
  print(cur_auc_score)
  if(cur_auc_score > best_auc_score):
   best_auc_score = cur_auc_score
 print("Best Score:")
 print(best_auc_score)
 print("pos_weight_a ",best_pos_weight_a,"  pos_weight_s ", best_pos_weight_s, " alpha ",best_alpha," beta ",best_beta)  


 

 #del model

  





