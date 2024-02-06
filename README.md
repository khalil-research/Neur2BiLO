# Neur2BiLO


Implementation of "Neur2BiLO: Neural Bilevel Optimization" [https://arxiv.org/pdf/2205.12006.pdf](https://arxiv.org/pdf/2402.02552.pdf). Note that this repository only contains the data. The remainder of the code will be added at a later date.


# Reference

Please cite our work if you find our code/paper useful to your work. 

```
  @article{dumouchelle2024neur2bilo,
    title={Neur2RO: Neural Bilevel Optimization},
    author={Dumouchelle, Justin and Julien, Esther and Kurtz, Jannis and Khalil, Elias B},
    journal={arXiv preprint arXiv:2402.02552},
    year={2024}
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


