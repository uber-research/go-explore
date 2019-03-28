
from tensorflow import HistogramProto
from collections import Counter

def makeHistProto(dist : Counter, bins, keys=None) -> HistogramProto:
	if keys == None:
		keys = sorted(dist.keys())
	hist = HistogramProto()
	hist.min = -0.5
	hist.max = bins-0.5
	hist.num = sum(dist.values())

	hist.sum = sum(i * dist[key] for i, key in enumerate(keys))
	hist.sum_squares = sum((i * dist[key]) ** 2 for i, key in enumerate(keys))
	for i in range(bins):
		hist.bucket_limit.append(i+0.5)
	for key in keys:
		hist.bucket.append(dist[key] * (30/bins))
	for _ in range(bins - len(keys)):
		hist.bucket.append(0)

	return hist