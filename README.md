# VGAE\_SGRACE



Graph Variational Autoencoder for anomaly detection on graphs with SGRACE layers (https://ieeexplore.ieee.org/document/11108959/https://github.com/eejlny/SGRACEx1) for hardware acceleration. 





Combines GAT/GCN layers and injects variability into the model defined by the beta parameter. The following picture shows a possible GNN configuration.


<img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/112b5c09-0860-4e3e-bf37-e9269ed7c16e" />




Based on https://link.springer.com/article/10.1007/s00521-025-11357-5



To test the model without hardware acceleration you need to install PYGOD libraries:



https://github.com/pygod-team/pygod



Then, for a quick run just do:



python3 gae.py 



Currently the variational tests (beta > 0) do not outperform the non-variational (beta = 0). 

Hardware support is work in progress but initial results demonstrate good anomaly scores with 1-bit precision using the SGRACE accelerator (blue normal/red abnormal) as illustrated in the following picture.

<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/6be14681-fdb5-4fbe-8183-998b506e0d67" />



Best AUC results measured with beta = 0 for considered datasets using the software model:



inj\_cora	weibo	enron	reddit	amazon	disney	books

0.86             0.94    0.85   0.64    0.90     0.75   0.64              

