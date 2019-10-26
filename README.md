deepAI
Implementation of a Deep Active Inference agent, accompagnying the article “Deep Active Inference” ([Biological Cybernetics, 2018](https://link.springer.com/article/10.1007/s00422-018-0785-7); [preprint](https://arxiv.org/abs/1709.02341)). The code depends on Python 2.7 and Theano for tensor operations.

To install the required components on Ubuntu, you can follow: http://deeplearning.net/software/theano/install_ubuntu.html

To sample from the trained model saved in deepAI_demo_best.pkl, just run
python sample_deepAI_demo.py

To train a new agent, run
python deepAI_paper.py

To plot the progress, run
python plot_log_deepAI_paper.py

To sample from the trained model, run
python sample_deepAI_paper.py

The scripts to directly reproduce all figures in the paper are in the ./figures/ subfolder.
