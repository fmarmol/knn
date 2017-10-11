package knn

import (
	"sync"
)

//KNN struct
type KNN struct {
	*base
}

//New creates a new classifier
func New(k int, x matrix, y vector, d distanceFunction) *KNN {
	return &KNN{newKNN(k, x, y, d)}
}

//PredictProba calculates probabilities for each row
//to be in nearests neighboors labels
func (knn *KNN) PredictProba(x matrix) vector {
	wg := sync.WaitGroup{}
	ret := make(vector, len(x))
	wg.Add(len(x))
	for i, xx := range x {
		go func(index int, l []float64) {
			c, _ := knn.nearestNeighboors(l).predictProba()
			ret[index] = c
			wg.Done()
		}(i, xx)
	}
	wg.Wait()
	return ret
}

//Predict calculates the value for each row
//from nearests neighboors
func (knn *KNN) Predict(x matrix) vector {
	wg := sync.WaitGroup{}
	ret := make(vector, len(x))
	wg.Add(len(x))
	for i, xx := range x {
		go func(index int, l []float64) {
			c := knn.nearestNeighboors(l).predict()
			ret[index] = c
			wg.Done()
		}(i, xx)
	}
	wg.Wait()
	return ret
}
