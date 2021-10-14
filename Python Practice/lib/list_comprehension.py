numList = [x for x in range(0, 10)]
numListSquared = [x**2 for x in numList]
# print(numListSquared)

strList = ['Arnav', 'Gauri', 'Fiona', 'Kada', 'Nivi']
allStrList = [name if name != "Arnav" else "not Arnav" for name in strList]
strListWithI = [name for name in strList if "i" in name]
print(strListWithI)