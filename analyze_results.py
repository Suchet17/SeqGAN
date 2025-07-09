import matplotlib.pyplot as plt
import numpy as np

path = "SeqGAN/try4_realDataset_embeddedMotif_noGaps1/Output.fasta"
motif = "CCACGAGGGGGCGC" #TODO: Motif matrix
seqs = []

with open(path) as f:
    seqs = np.array([i.strip() for i in f.readlines()[1::2]])

index = np.zeros(len(seqs), dtype=np.int16)-1
for i, seq in enumerate(seqs):
    try:
        index[i] = seq.index(motif)
    except ValueError:
        pass

plt.hist(index, bins = np.arange(-1, 101))
plt.show()
