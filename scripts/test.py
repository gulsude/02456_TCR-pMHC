data_list = []
target_list = []

import encoding as enc

peptides = ["AGAG"]
#embedded = enc.esm_1b(peptides, model="MSA")


a = enc.esm_ASM(peptides, pooling=False)

print(len(a))
print(a)

