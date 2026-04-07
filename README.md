# VGAE\_SGRACE



Graph Variational Autoencoder for anomaly detection on graphs





Combines GAT/GCN layers and injects variability into the model defined by the beta parameter. 


<img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/112b5c09-0860-4e3e-bf37-e9269ed7c16e" />




Based on https://link.springer.com/article/10.1007/s00521-025-11357-5



You need to install PYGOD libraries:



https://github.com/pygod-team/pygod



Then, for a quick run just do:



python3 gae.py 



Currently the variational tests (beta > 0) do not outperform the non-variational (beta = 0)



Best AUC results measured with beta = 0 for considered datasets:



inj\_cora	weibo	enron	reddit	amazon	disney	books

0.86             0.94    0.85   0.64    0.90     0.75   0.64              

