import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import softplus

import sys
sys.path.insert(0, 'MusicVAE/src/')

from data.dataloader import MidiDataset
from data.bar_transform import BarTransform
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split

from torch.autograd import Variable
import matplotlib.pyplot as plt
from IPython.display import Image, Audio, display, clear_output
import numpy as np
from sklearn.decomposition import PCA
%matplotlib nbagg
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette(sns.dark_palette("purple"))
from torch.nn.functional import binary_cross_entropy
from torch import optim
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.autograd import Variable
import time
import os
import math
from midi_builder import MidiBuilder

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_features,teacher_forcing, eps_i):
        super(VariationalAutoencoder, self).__init__()
        
        self.teacher_forcing = teacher_forcing
        self.eps_i = eps_i

        self.latent_features = latent_features

        #data goes into bidirectional encoder
        self.encoder = torch.nn.LSTM(
                batch_first = True,
                input_size = input_size,
                hidden_size = enc_hidden_size,
                num_layers = 1,
                bidirectional = True)
        
        #encoded data goes onto connect linear layer. inputs must be*2 because LSTM is bidirectional
        #output must be 2*latentspace because it needs to be split into miu and sigma right after.
        self.encoderOut = nn.Linear(in_features=enc_hidden_size*2, out_features=latent_features*2)
        
        #after being converted data goes through a fully connected layer
        self.linear_z = nn.Linear(in_features=latent_features, out_features=decoders_initial_size)
        
        self.dropout= nn.Dropout(p=dropout_rate)
        
        self.worddropout = nn.Dropout2d(p=dropout_rate)
        
        # Define the conductor and note decoder
        self.conductor = nn.LSTM(decoders_initial_size, decoders_initial_size, num_layers=1,batch_first=True)
        self.decoder = nn.LSTM(NUM_PITCHES+decoders_initial_size, decoders_initial_size, num_layers=1,batch_first=True)
        
        # Linear note to note type (classes/pitches)
        self.linear = nn.Linear(decoders_initial_size, NUM_PITCHES)

        
    #used to initialize the hidden layer of the encoder to zero before every batch
    def init_hidden(self, batch_size):
        #must be 2 x batch x hidden_size because its a bi-directional LSTM
        init = torch.zeros(2, batch_size, enc_hidden_size, device=device)
        c0 = torch.zeros(2, batch_size, enc_hidden_size, device=device)
    
        #2 because has 2 layers
        #n_layers_conductor
        init_conductor = torch.zeros(1, batch_size, decoders_initial_size, device=device)
        c_condunctor = torch.zeros(1, batch_size, decoders_initial_size, device=device)
        
        return init,c0,init_conductor,c_condunctor

    # Coin toss to determine whether to use teacher forcing on a note(Scheduled sampling)
    # Will always be True for eps_i = 1.
    def use_teacher_forcing(self):
        with torch.no_grad():
            tf = np.random.rand(1)[0] <= self.eps_i
        return tf
    
    def set_scheduled_sampling(self, eps_i):
        self.eps_i = eps_i

    def forward(self, x):
        batch_size = x.size(0)
        
        note = torch.zeros(batch_size, 1 , NUM_PITCHES,device=device)

        the_input = torch.cat([note,x],dim=1)
        
        outputs = {}
        
        #creates hidden layer values
        h0,c0,hconductor,cconductor = self.init_hidden(batch_size)
        
        x = self.worddropout(x)
        
        #resets encoder at the beginning of every batch and gives it x
        x, hidden = self.encoder(x, ( h0,c0))
        
        #x=self.dropout(x)
        
        #goes from 4096 to 1024
        x = self.encoderOut(x)      
        
        #x=self.dropout(x)
        
        # Split encoder outputs into a mean and variance vector 
        mu, log_var = torch.chunk(x, 2, dim=-1)
                
        # Make sure that the log variance is positive
        log_var = softplus(log_var)
               
        # :- Reparametrisation trick
        # a sample from N(mu, sigma) is mu + sigma * epsilon
        # where epsilon ~ N(0, 1)
                
        # Don't propagate gradients through randomness
        with torch.no_grad():
            batch_size = mu.size(0)
            epsilon = torch.randn(batch_size, 1, self.latent_features)
            
            if cuda:
                epsilon = epsilon.cuda()
        
        #setting sigma
        sigma = torch.exp(log_var*2)
        
        #generate z - latent space
        z = mu + epsilon * sigma
        
        #decrese space
        z = self.linear_z(z)
        
        #z=self.dropout(z)
        
        #make dimensions fit (NOT SURE IF THIS IS ENTIRELY CORRECT)
        #z = z.permute(1,0,2)

        #DECODER ##############
        
        conductor_hidden = (hconductor,cconductor)
        
        counter=0
        
        notes = torch.zeros(batch_size,TOTAL_NOTES,NUM_PITCHES,device=device)

        # For the first timestep the note is the embedding
        note = torch.zeros(batch_size, 1 , NUM_PITCHES,device=device)
        
        # Go through each element in the latent sequence
        for i in range(16):
            embedding, conductor_hidden = self.conductor(z[:,i,:].view(batch_size,1, -1), conductor_hidden)    
           
            if self.use_teacher_forcing():
                
                 # Reset the decoder state of each 16 bar sequence
                decoder_hidden = (torch.randn(1,batch_size, decoders_initial_size,device=device), torch.randn(1,batch_size, decoders_initial_size,device=device))
                
                embedding = embedding.expand(batch_size, NOTESPERBAR, embedding.shape[2])
                
                e = torch.cat([embedding,the_input[:,range(i*16,i*16+16),:]],dim=-1)
                
                notes2, decoder_hidden = self.decoder(e, decoder_hidden)
                
                aux = self.linear(notes2)
                aux = torch.softmax(aux, dim=2);
                    
                #generates 16 notes per batch at a time
                notes[:,range(i*16,i*16+16),:]=aux;
            else:           
                 # Reset the decoder state of each 16 bar sequence
                decoder_hidden = (torch.randn(1,batch_size, decoders_initial_size,device=device), torch.randn(1,batch_size, decoders_initial_size,device=device))
                
                for _ in range(sequence_length):
                    # Concat embedding with previous note
                    
                    e = torch.cat([embedding, note], dim=-1)
                    e = e.view(batch_size, 1, -1)

                    # Generate a single note (for each batch)
                    note, decoder_hidden = self.decoder(e, decoder_hidden)
                    
                    aux = self.linear(note)
                    aux = torch.softmax(aux, dim=2);
                    
                    notes[:,counter,:]=aux.squeeze();
                    
                    note=aux
                    
                    counter=counter+1


        outputs["x_hat"] = notes
        outputs["z"] = z
        outputs["mu"] = mu
        outputs["log_var"] = log_var
        
        return outputs

def ELBO_loss(y, t, mu, log_var, weight):
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    cuda = torch.cuda.is_available()
    likelihood = -binary_cross_entropy(y, t, reduction="none")
    likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

    # Regularization error: 
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    sigma = torch.exp(log_var*2)
    n_mu = torch.Tensor([0])
    n_sigma = torch.Tensor([1])
    if cuda:
        n_mu = n_mu.cuda()
        n_sigma = n_sigma.cuda()

    p = Normal(n_mu, n_sigma)
    q = Normal(mu, sigma)

    #The method signature is P and Q, but might need to be reversed to calculate divergence of Q with respect to P
    kl_div = kl_divergence(q, p)
    
    # In the case of the KL-divergence between diagonal covariance Gaussian and 
    # a standard Gaussian, an analytic solution exists. Using this excerts a lower
    # variance estimator of KL(q||p)
    #kl = -weight * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=(1,2))
    
    # Combining the two terms in the evidence lower bound objective (ELBO) 
    # mean over batch
    ELBO = torch.mean(likelihood) - (weight*torch.mean(kl_div)) # add a weight to the kl using warmup
    
    # notice minus sign as we want to maximise ELBO
    return -ELBO, kl_div.mean(),weight*kl_div.mean() # mean instead of sum


def lin_decay(train_loader, i, mineps=0):
    return np.max([mineps, 1 - (1/len(train_loader))*i])

def inv_sigmoid_decay(i, rate=40):
    return rate/(rate + np.exp(i/rate))

def train_VAE(model, optimizer, train_loader, test_loader, device, PATH = "model.pt", num_epochs = 100, warmup_epochs= 90, scheduled_decay_rate = 40,pre_warmup_epochs = 10, 
 loss_function = ELBO_loss, use_scheduled_sampling = False, totalbars = 16, tmp_img = "tmp_vae_out.png"):
    #optimizer = optim.Adam(net.parameters(), lr=0.001)
    warmup_lerp = 1/warmup_epochs
    warmup_w=0
    eps_i = 1
    train_loss, valid_loss = [], []
    train_kl, valid_kl,train_klw = [], [],[]
    start = time.time()
    print("Training epoch {}".format(0))
    #epochs loop
    for epoch in range(num_epochs):
        
        batch_loss, batch_kl,batch_klw = [], [],[]
        model.train()

        for i_batch, sample_batched in enumerate(train_loader):
            #if i_batch == 10:
            #    break
            x = sample_batched['piano_rolls']

            x = x.type('torch.FloatTensor')
            
            #if i_batch%10==0:
            #    print("batch:",i_batch)

            x = Variable(x)

            # This is an alternative way of putting
            # a tensor on the GPU
            x = x.to(device)
            
            ## Calc the sched sampling rate:
            if epoch >= pre_warmup_epochs and use_scheduled_sampling:
                eps_i = inv_sigmoid_decay(i_batch, rate=scheduled_decay_rate)

            model.set_scheduled_sampling(eps_i)
            
            outputs = model(x)
            x_hat = outputs['x_hat']
            mu, log_var = outputs['mu'], outputs['log_var']

            elbo, kl,kl_w = loss_function(x_hat, x, mu, log_var, warmup_w)

            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

            batch_loss.append(elbo.item())
            batch_kl.append(kl.item())
            batch_klw.append(kl_w.item())
        train_loss.append(np.mean(batch_loss))
        train_kl.append(np.mean(batch_kl))
        train_klw.append(np.mean(batch_klw))
        torch.save(model.state_dict(),PATH)

        # Evaluate, do not propagate gradients
        with torch.no_grad():
            model.eval()

            # Just load a single batch from the test loader
            x = next(iter(test_loader))
            x = Variable(x['piano_rolls'].type('torch.FloatTensor'))

            x = x.to(device)

            model.set_scheduled_sampling(1.) # Please use teacher forcing for validations
            outputs = model(x)
            x_hat = outputs['x_hat']
            mu, log_var = outputs['mu'], outputs['log_var']
            z = outputs["z"]

            elbo, kl,klw = loss_function(x_hat, x, mu, log_var, warmup_w)

            # We save the latent variable and reconstruction for later use
            # we will need them on the CPU to plot
            x = x.to("cpu")
            x_hat = x_hat.to("cpu")
            z = z.detach().to("cpu").numpy()

            valid_loss.append(elbo.item())
            valid_kl.append(kl.item())
        
        if epoch >= pre_warmup_epochs:
            warmup_w = warmup_w + warmup_lerp
            if warmup_w > 1:
                warmup_w=1.
        
        if epoch == 0:
            continue
                
        # -- Plotting --
        f, axarr = plt.subplots(2, 1, figsize=(10, 10))
        
        
        # Loss
        ax = axarr[0]
        ax.set_title("ELBO")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error')

        ax.plot(np.arange(epoch+1), train_loss, color="black")
        ax.plot(np.arange(epoch+1), valid_loss, color="gray", linestyle="--")
        ax.legend(['Training', 'Validation'])
        
        
        # KL / reconstruction
        ax = axarr[1]
        
        ax.set_title("Kullback-Leibler Divergence")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL divergence')


        ax.plot(np.arange(epoch+1), train_kl, color="black")
        ax.plot(np.arange(epoch+1), valid_kl, color="gray", linestyle="--")
        ax.plot(np.arange(epoch+1), train_klw, color="blue", linestyle="--")
        ax.legend(['Training', 'Validation','Weighted'])
        
        print("Epoch: {}, {} seconds elapsed".format(epoch, time.time() - start))
        
        plt.savefig(tmp_img)
        plt.close(f)
        display(Image(filename=tmp_img))
        
        clear_output(wait=True)

        os.remove(tmp_img)

    end_time = time.time() - start
    print("Finished. Time elapsed: {} seconds".format(end_time))

def decode(x, z_gen,model, NOTESPERBAR = 16,decoders_initial_size = 16, gen_batch = 32, TOTAL_NOTES = 256, NUM_PITCHES = 61, device = 'cuda', totalbars = 16, ):
  z_gen = z_gen.to(device)
  # Sample from latent space
  h_gen,c_gen,hconductor_gen,cconductor_gen = model.init_hidden(gen_batch)
  conductor_hidden_gen = (hconductor_gen,cconductor_gen)
  notes_gen = torch.zeros(gen_batch,TOTAL_NOTES,NUM_PITCHES,device=device)
  # For the first timestep the note is the embedding
  note_gen = torch.zeros(gen_batch, 1 , NUM_PITCHES,device=device)
  counter=0
  the_input = torch.cat([note_gen,x],dim=1)
  for i in range(totalbars):#totalbars = 16
      decoder_hidden_gen = (torch.randn(1,gen_batch, decoders_initial_size,device=device), torch.randn(1,gen_batch, decoders_initial_size,device=device))
      embedding_gen, conductor_hidden_gen = model.conductor(z_gen[:,i,:].view(gen_batch,1, -1), conductor_hidden_gen)
      embedding_gen = embedding_gen.expand(gen_batch, NOTESPERBAR, embedding_gen.shape[2])
      e = torch.cat([embedding_gen,the_input[:,range(i*16,i*16+16),:]],dim=-1)
      notes2, decoder_hidden_gen = model.decoder(e, decoder_hidden_gen)
      aux = model.linear(notes2)
      aux = torch.softmax(aux, dim=2)
      notes_gen[:,range(i*16,i*16+16),:]=aux
  return(notes_gen) 
