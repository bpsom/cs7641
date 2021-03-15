Randomized Optimization - Project2
Bommegowda P. Somashekharagowda
GTID: 903387896 
CS7641
Spring-2021

1) Code github Repo link -
	https://github.gatech.edu/bps6/CS7641-Project1/tree/master/project1
	The repo consists of all source files to execute the project, datasets, and env.yml to install the python anaconda environment to run the project
	github commit id: 2ac74cca469aa8284cca544ee423386e724d64aa

2) File Details
	The code has below source files -
		RandomOptProblems.py
		ROTestOptimization.py
	Dataset files -
		Data\winequality-white.csv

	results directory consists of the plots and the result.txt file when the code is executed -
		results\
		result\Results.txt

	
3) Required Libraries to execute the code
    
    Requires mlrose package and python 3.6 to execute.
	
	
	# Acknowledgements - Referred to works from 
	1. Rollings, A. (2020). mlrose: Machine Learning, Randomized Optimization and SEarch pack-age for Python, hiive extended remix. https://github.com/hiive/mlrose. Accessed: 14 March 2021.
	2. Hayes, G. (2019). mlrose: Machine Learning, Randomized Optimization and SEarch package for Python. https://github.com/gkhayes/mlrose. Accessed: 14 March 2021.
	3. David S. Park for the MIMIC enhancements (from https://github.com/parkds/mlrose)


4) Working environment where this code was used (System Setup)

    On a Windows 10 machine Pycharm editor with Anaconda python 3.6 as project interpreter was used to implement the code. 
	

5) Code execution

    On a python3.6 console execute command with all required libraries installed, activate conda environment as explained above.
    Then execute -

	For Linux environment
		$ python ROTestOptimization.py -s 1

	For windows enviroment
		$ python SUTestLearners.py -s 0
	
    Note: Code execution takes I think around 4-5 hours to produce the results. 
    
6) Output

    The code generates plots and a Results.txt file with data. They are stored into `results` directory.
	
	Note: During execution of neural networks it takes hours to produce results.

