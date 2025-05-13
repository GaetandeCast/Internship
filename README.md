# Shapley value comparison
Folder containing the necessary files to run experiments reproducing Louppe 2021 "From global to local MDI variable importances for random forests and when they are Shapley values".
# Profilings and benchmarks 
Folder containing the CPU and memory profiling results done to ensure the controlled computation cost of the added methods. Contains a benchmark on a conditionally dependant dataset in attempt to favor a method over the other
# Theory
Some proofs and elements can be found at : https://www.overleaf.com/read/wqsvxqqrffyw#090a1b
# To-do
- [x] write on latex the draft notes on the entropy version of the methods and why extending mdi_oob naturally does not coincide with mdi on train data.
- [x] make a test that verifies that one-hot classification w/ gini and reg with MSE coincide for biased and unbiased MDI.
- [ ] Try to find a way to extend mdi_oob for entropy in a way that matches mdi on train
- [ ] Extend the shapley value comparison : Use a noised version of the led dataset and study the effects of increasing the training and testing sizes.
- [x] Add a tests to ensure that increasing sample size makes biased and unbiased feature importance converge.
- [x] Add sample weights support
- [ ] Add sparse data input support
- [ ] Do a theoretical comparison of MDI, uMDIs, Sage/shapley, permutation importance, CPI, LOCO