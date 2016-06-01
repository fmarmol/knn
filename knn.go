package knn

import (
	"sync"
)

type KNNClassifier struct {
	*BaseKNN
}

func NewKNNClassifier(k int, x [][]float64, y []float64, d DistanceFunction) *KNNClassifier {
	return &KNNClassifier{NewKNN(k, x, y, d)}
}

type KNNRegressor struct {
	*BaseKNN
}

func NewKNNRegressor(k int, x [][]float64, y []float64, d DistanceFunction) *KNNRegressor {
	return &KNNRegressor{NewKNN(k, x, y, d)}
}

func (k *KNNClassifier) Predict(x [][]float64) []float64 {
	wg := sync.WaitGroup{}
	ret := make([]float64, len(x))
	wg.Add(len(x))
	for i, xx := range x {
		go func(index int, l []float64) {
			c, _ := k.NNeighboors(l).PredictProba()
			ret[index] = c
			wg.Done()
		}(i, xx)
	}
	wg.Wait()
	return ret
}

func (k *KNNRegressor) Predict(x [][]float64) []float64 {
	wg := sync.WaitGroup{}
	ret := make([]float64, len(x))
	wg.Add(len(x))
	for i, xx := range x {
		go func(index int, l []float64) {
			c := k.NNeighboors(l).Predict()
			ret[index] = c
			wg.Done()
		}(i, xx)
	}
	wg.Wait()
	return ret
}
