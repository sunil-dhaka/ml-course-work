-----------------anomaly README.txt------------------


Plugins-
Software: python3 jupyter notebook (.sh files run python3 jupyter notebooks)

Environments: conda 4.10.3, python 3.8.5


Dependencies-
Python Libraries: pandas 1.1.3, numpy 1.19.2, matplotlib 3.3.2 

Programs-

.sh files:
anomaly.sh 
top level script to run the anomaly.ipynb file

Makefile
to run entire project after checking requirements

How to use:
In order to run all programs sequentially, run the following command from the terminal-
bash anomaly-s110.sh 

One could use `time` command before bash execution to see how much time script takes to run.

How to run particular shell script:
--------------------------------------------------------------------------------------------
anomaly.sh
runs anomaly.ipynb as
jupyter nbconvert --to notebook --execute anomaly.ipynb

------------------------------------------------------------------------------------------------
- NOTE: Every detail about the procedure to solve the question is given in markdowns and comments of notebooks.

Jupyter Notebook files points:

anomaly.ipynb
Input: anomaly-s110.dat

Output: answer-s110.dat

Note: Output file is seperated by ' ' like answer-sample.dat file