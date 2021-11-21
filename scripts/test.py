data_list = []
target_list = []

import encoding as enc

peptides = ["AGAG"]
embedded = enc.esm_1b(peptides, pooling=False)


#a = enc.esm_ASM(peptides, pooling=False)

print(embedded[0].shape)
print(embedded )

