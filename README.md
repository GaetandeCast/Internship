# Shapley value comparison
Folder containing the necessary files to run experiments reproducing Louppe 2021 "From global to local MDI variable importances for random forests and when they are Shapley values".
# Profilings and benchmarks 
Folder containing the CPU and memory profiling results done to ensure the controlled computation cost of the added methods. Contains a benchmark on a conditionally dependant dataset in attempt to favor a method over the other
# Theory
Some proofs and elements can be found at : https://www.overleaf.com/read/wqsvxqqrffyw#090a1b
# To-do
- write on latex the draft notes on the entropy version of the methods and why extending mdi_oob naturally does not coincide with mdi on train data.
- Try to find a way to extend mdi_oob for entropy in a way that matches mdi on train
- Extend the shapley value comparison : Use a noised version of the led dataset and study the effects of increasing the training and testing sizes.
- Add a tests to ensure that increasing sample size makes biased and unbiased feature importance converge 
- Add sample weights support
- Add sparse data input support
