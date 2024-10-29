from bosonic import BoseWord, BoseSentence, normal_order


w1 = BoseWord({(0, 0): '+', (1, 1): '-'})

w2 = BoseWord({(0, 0): '-', (1, 1): '-', (2, 1): '+'})

bs = BoseSentence({w1:1.0, w2:1.0})

print(w1, w2, bs)
print(normal_order(bs))
