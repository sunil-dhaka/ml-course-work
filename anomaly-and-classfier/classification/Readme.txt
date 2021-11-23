-----------------Classifier README.txt------------------


Plugins-
Software: python3 jupyter notebook (.sh files run python3 jupyter notebooks)

Environments: conda 4.10.3, python 3.8.5

Dependencies-
Python Libraries: pandas 1.1.3, numpy 1.19.2, sklearn 1.0.1, matplotlib 3.3.2, seaborn 0.11.2 


Programs-

.sh files:

classifier-s110.sh 
top level script that runs the test part
It takes <test-file> as input along with command
Like: 
```sh
bash classifier-s110.sh <test-file-path>
```

Makefile
Check requirements needed for the project. And installs them.
Simple makefile to run the optimization classification.ipynb file using classification.sh ONLY. It does not run classifier-s110.sh. For that .sh file is there. 
I have to do it because a Makefile can not read input in order from terminal.

How to use:
In order to run all programs sequentially, run the following command from the terminal-
bash classifier-s110.sh 

One could use `time` command before bash execution to see how much time script takes to run.

How to run particular shell script:
--------------------------------------------------------------------------------------------
classifier-s110.sh
This script runs the test part of the assignment. Runs only when there is a test-file input.

classification.sh
runs classification.ipynb as
jupyter nbconvert --to notebook --execute classification.ipynb

------------------------------------------------------------------------------------------------
- NOTE: Every detail about the procedure to solve the question is given in markdowns and comments of notebooks.

Jupyter Notebook and .py script files points:

classifier-s110.sh
Auto Input: training-s110.dat
User Input: <test-file>

Output: answer-s110.csv

Note: it is just a notebook file that explains I approached the question and decided on the model I am finally using for training.
classification.ipynb
Input: training-s110.dat

Output: It is the optimization part of classification