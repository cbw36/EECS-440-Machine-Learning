Working implementation of decision tree algorithms for EECS 440.

Wolfe contributions commit 1:
    Design first attempt at ID3 algorithm with following pseudocode
        Check base cases
        If dont apply, find best attribute to split on (IG/IGR not implemented yet)
        Set this attribute as root of new tree
        Iterate over the attributes values
            Add a branch for each value
            Determine examples which have this value
            If none do, create leaf
            Else Recurse through ID3 again

Wolfe week 2:
    Design the tree structure we will use.
    Lastly, implement some useful helper functions like find_best_attribute(), most_common_label(), etc...

Wolfe Week 3:
    Edit handling of attributes to be list of features, and calculate threshold on each iteration
    Bin thresholds to improve complexity
    Implement command line inputs
    Impelement functions to determine number of positive and negative for each attribute value to send to IG calculations for continuous
    Modify ID3 to better handle nominal vs continuous


Yau contributions commit 1:
	Minor changes to code.
	Working on implementation of IG and IGR, procedure written.
	
Yau contributions commit 2:
	Finished implementing IG and IGR.

Yau contributions commit 3:
    Reimplementation of various functions
    Debugging
    Testing and data gathering for writeup
