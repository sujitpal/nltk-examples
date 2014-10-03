import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
fin = open("../../data/word_som_online.txt", 'rb')
for line in fin:
  word, x, y = line.strip().split("\t")
  ax.text(int(x), int(y), word)
fin.close()
ax.axis([0, 50, 0, 50])
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.title("Word Clusters (Online Training)")
plt.grid()
plt.show()
