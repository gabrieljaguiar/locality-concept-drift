# A comprehensive analysis of concept drift locality in data streams


## Usage

### Single-Class

<!-- ADD here how to use Single-Class part -->
Single-Class drifting data streams are generated using the file ``generators/single_class.py``

The following code serves as an illustrative example of generating streams featuring single-class drifts. It facilitates the generation of streams with Local Drifts exhibiting 3, 5, and 10 classes, encompassing 2 and 5 features. Additionally, it includes only Sudden Drifts, denoted by a ``drift_width`` value of 1.

```python 
from generators.single_class import generate_streams

streams = generate_streams(
    n_classes = [3, 5, 10],
    n_features = [2, 5],
    drift_width = [1],
    locality = ["local"],
):

```


<table>
  <tr>
    <td>Local</td>
    <td>Global</td>
  </tr>
  <tr>
    <td valign="top"><img src="figures/single_local_drift.gif"></td>
    <td valign="top"><img src="figures/single_global_drift.gif"></td>
  </tr>
 </table>


### Multi-Class

Multi-Class drifting data streams are generated using the file ``generators/multi_class.py``

The following code serves as an illustrative example of generating streams featuring multi-class drifts. It facilitates the generation of streams with Local Drifts exhibiting 3, 5, and 10 classes, encompassing 2 and 5 features. Additionally, this code allows the specification of the number of affected classes in each scenario. For instance, with 3 classes, only 1 will be affected, while for 5 classes, there will be streams with 2 and 3 affected classes, and so on. 

```python 
from generators.single_class import generate_streams

streams = generate_streams(
    n_classes = [3, 5, 10],
    n_features = [2, 5],
    classes_affected = [[1], [2, 3], [3, 5]]
    drift_width = [1],
    locality = ["local"],
):

```


<table>
  <tr>
    <td>Local</td>
    <td>Global</td>
  </tr>
  <tr>
    <td valign="top"><img src="figures/multi_local_drift.gif"></td>
    <td valign="top"><img src="figures/multi_global_drift.gif"></td>
  </tr>
 </table>


## Citation
```
@misc{aguiar2023local,
  author={Aguiar, Gabriel and Cano, Alberto},
  title={A comprehensive analysis of concept drift locality in data streams},
  year={2023},
  eprint={2204.03719},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```