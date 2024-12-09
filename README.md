# Neur2BiLO


Implementation of Neur2BiLO, an efficient learning-based algorithm for mixed-integer (non-)linear bilevel optimization.  Implementation coming soon.  Reference below.
 - \[1\] Dumouchelle, J., Julien, E., Kurtz, J., & Khalil, E. B. Neur2BiLO: Neural Bilevel Optimization.  *The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2024. [\[Paper\]](https://openreview.net/pdf?id=T5Xb0iGCCv)
   

# Reference

Please cite our work if you find our code/paper useful to your work. 

```
@inproceedings{
  dumouchelle2024neurbilo,
  title={Neur2Bi{LO}: Neural Bilevel Optimization},
  author={Justin Dumouchelle and Esther Julien and Jannis Kurtz and Elias Boutros Khalil},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
}
```


## Benchmark Instances and References

If using any of the benchmark problems/instances from our paper, please cite the appropriate references.  

### Knapsack Interdiction Problem
- Reference: Yen Tang, Jean-Philippe P Richard, and J Cole Smith. A class of algorithms for mixed-integer bilevel min–max optimization. *Journal of Global Optimization*, 66:225–262, 2016.
- Link to instances: [https://web.archive.org/web/20220121032905/http://jcsmith.people.clemson.edu/Test_Instances_files/BKPIns.zip](https://web.archive.org/web/20220121032905/http://jcsmith.people.clemson.edu/Test_Instances_files/BKPIns.zip)
- Note we provide these instances and instances with 100 items in the MibS input file format in `data/kp/solver_instances/`.  

### Critical Node Problem/Game
- Reference: Gabriele Dragotto, Amine Boukhtouta, Andrea Lodi, and Mehdi Taobane. The critical
node game, 2023.
- Link to instances: [https://github.com/ds4dm/CNG-Instances](https://github.com/ds4dm/CNG-Instances)
- Note that the instances used in this work are contained in the data directory that differ from those at the above link but were randomly generated using the same procedure.  We provide the MibS input file format instances used in our experiments in `data/cng/solver_instances/`.  

### Donor Recipient Problem
- Reference: Shraddha Ghatkar, Ashwin Arulselvan, and Alec Morton. Solution techniques for bi-level knapsack problems. *Computers & Operations Research*, 159:106343, 2023.
- Link to instances: [https://github.com/ashwin-1983/DR-BKP/](https://github.com/ashwin-1983/DR-BKP/)
- Note that if using these instances with our code, they will need to be downloaded, unzipped, and moved to `data/dr/DR-BKP-main/`.

### Discrete Network Design Problem
- Reference: David Rey. Computational benchmarking of exact methods for the bilevel discrete network design problem. *Transportation Research Procedia*, 47:11–18, 2020.
- Link to instances: [https://github.com/davidrey123/DNDP/](https://github.com/davidrey123/DNDP/)


