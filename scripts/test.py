data_list = []
target_list = []

import encoding as enc

peptides = ["AACG"]
embedded = enc.esm_1b(peptides, model="MSA")

print(embedded)
print(embedded[0].shape)
print(embedded)