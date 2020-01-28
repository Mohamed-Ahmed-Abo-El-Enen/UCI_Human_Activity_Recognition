import Model_1_1 as Model

if __name__ == "__main__":
    standardize_list = [False, True]
    n_filters = [8, 16, 32, 64, 128, 256]
    n_kernels = [2, 3, 5, 7, 11]
    repeats = 10
    Model.run_experiment(standardize_list, n_filters, n_kernels, repeats)
