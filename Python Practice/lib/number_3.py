x = "We can only see a short distance ahead, but we can see plenty there that needs to be done."
y = x.split(" s")
z = []
for chunk in y:
    z.append(str(len(chunk) - 1))

phrase = "_".join(z)

for word in phrase.split('3_3'):
    if "0" in word:
        print(word, word[0:2], z[0], sep="..", end=".")