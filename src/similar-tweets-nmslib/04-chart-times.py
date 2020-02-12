
#%%
import matplotlib.pyplot as plt

num_recs = [1000, 5000, 10000, 20000, 40000, 63111]
index_load_times = [0.163, 1.471, 4.655, 11.402, 28.614, 53.969]
# query_times = [0.018, 0.013, 0.020, 0.051, 0.033, 0.054]
query_times = [0.018, 0.020, 0.028, 0.045, 0.051, 0.054]

plt.plot(num_recs, index_load_times, marker="o", color="r")
plt.title("Index Load times")
plt.xlabel("number of records")
plt.ylabel("load time (s)")
plt.show()

# plt.plot(num_recs, query_times, marker="o", color="r")
# plt.title("Query times")
# plt.xlabel("number of records")
# plt.ylabel("query time (s)")
# plt.show()


# %%
