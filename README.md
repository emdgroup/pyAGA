pyAGA - Find Approximate Graph Automorphisms
============================================

## Installing Required Python Packages
Please use Python version 3.6 or later.
Required packages are listed in `requirements.txt` and can be installed using with `pip install -r requirements.txt`.

## Running the program
On your command line, just invoke `python main_automorphism_finder.py` to run an example calculation. The
parameters are set inside the python file. An interactive matplotlib window is opened,
containing a histogram of the edge weights and the bins. The calculation starts after 
you close this window.

To run the test suite, use `python -m unittest test.py`. Note that the last argument is simply `test.py`, *not* 
`.\test.py` or `./test.py`.

If you want to run multiple tests with different parameters, consider running the script `parameter_study.py`
instead by running `python parameter_study.py <testcase>`, where `<testcase>` corresponds to a configuration
file inside the subdirectory `parameter_study/` called `parameter_study_<testcase>`. 
An example input file for the `parameter_study`-module is given by:

    [global_timeout]
    # Time until the entire run is terminated
    global_timeout = 1000
    
    [main]
    # Maximum time any one iteration can process until it is terminated
    time_per_iteration = 1000
    # The prefix of the matrices used in 
    world_name = two_letter_words_20x10
    integer_matrices = False
    trafo_round_decimals = None
    use_integer_programming = True
    quiet = False
    norm = Norm.L_INFINITY
    
    error_value_limit = 0.01,
    percentages = 50.0,
    kde_bandwidths = 3.3e-4,
    fault_tolerance_ratios = 0.2,

The bottom four parameters `error_value_limit`, `percentages`, `kde_bandwidths`, and `fault_tolerance_ratios` can be
simple floats or tuples of floats, in which case the cartesian product of all values will be calculated and the algorithm will be run
for every element of this cartesian product. The results are then saved to 
`parameter_study/results/<testcase>_results_<uuid4_string>.xlsx`. The random
`uuid4_string` is added to prevent accidental overwriting of previously calculated results.
For a detailed explanation into the consequence of these parameters, we refer to the publication mentioned above.
