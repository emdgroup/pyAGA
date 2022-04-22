pyAGA - Find Approximate Graph Automorphisms
============================================

## Installing Required Python Packages
Please use Python version 3.6 or later.
Required packages are listed in `requirements.txt` and can be installed with `pip install -r requirements.txt`.

## Installing SCIP
To run pyAGA, please install the SCIP optimization suite version 7.0.3 from https://www.scipopt.org/, 
see [here](https://www.scipopt.org/index.php#license) for licensing details. 


## Running the program
### Example calculation
On your command line, just invoke
```
python example_calculation.py
```
to run an example calculation with fixed parameters. 

An interactive matplotlib window is opened, containing a histogram of the edge weights and the bins. 
The calculation starts after you close this window.
It calculates the automorphism group for a world with 20x10-pixels and translational invariance. After running for 5-6 minutes, this yields the expected 400 permutations.

### Different parameters
If you want to run multiple tests with different parameters, consider running the script `calculate_automorphisms.py`
instead by running
```
python calculate_automorphisms.py <testcase>
```
where `<testcase>` corresponds to a configuration
file inside the subdirectory `calculations/` called `<testcase>.ini`. 
An example configuration file for the `calculate_automorphisms`-module is given by:

    [global_timeout]
    # Time until the entire run is terminated
    global_timeout = 1000
    
    [main]
    # Maximum time any one iteration can process until it is terminated
    time_per_iteration = 1000
    # The prefix of the matrices used (decide which world/dataset to consider)
    world_name = two_letter_words_20x10
    
    error_value_limits = 0.01
    percentages = 50.0, 60.0
    kde_bandwidths = 3.3e-4
    fault_tolerance_ratios = 0.2


The bottom four parameters `error_value_limits`, `percentages`, `kde_bandwidths`, and `fault_tolerance_ratios` can be
simple floats or tuples of floats, in which case the cartesian product of all values will be calculated and the algorithm will be run
for every element of this cartesian product. The results are then saved to 
`calculations/results/<testcase>_results_<uuid4_string>.xlsx`. The random
`uuid4_string` is added to prevent accidental overwriting of previously calculated results.

As all matrices are stored in the subdirectory data/, this will load the adjacency matrices
- data/two_letter_words_20x10_concurrence_matrix_50.0.pickle  and
- data/two_letter_words_20x10_concurrence_matrix_60.0.pickle

and perform the automorphism calculation. 
If you want to try a different world or a different percentage of unique observations, change the parameter `world_name` and `percentages` accordingly.

For a detailed explanation into the consequence of these parameters, we refer to the publication mentioned above.