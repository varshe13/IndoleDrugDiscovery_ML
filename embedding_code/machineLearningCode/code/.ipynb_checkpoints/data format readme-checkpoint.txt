The molecule structures in numpy arrays can be found in- "112723_all_GP_molecules_4D.pickle"
Their respective activities can be found in- "112723_all_moleducles_GP_activities.pickle"
In total this dataset contains 191 molecules, for which I generated 15 conformers for each of them and embedded them in a 30x30x30 grid. In total there should be 2865 molecules in the list in total.
Please let me know if you have any questions.

Two lists in the dataset.
all_GP_moledules_4D is a list containing 2865 items. Each item is a 2x30x30x30 volume.
GP_activities is a list of 2865 targets. each item is one of three numbers (0.5, 1.0, 1.5) means low, med, high