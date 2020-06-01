package knn

import (
	"sync"
)

//KNN struct
type KNN struct {
	*base
}

//New creates a new classifier
func New(k int, x Matrix, y Vector, d DistanceFunction) *KNN {
	return &KNN{newKNN(k, x, y, d)}
}

//PredictProba calculates probabilities for each row
//to be in nearests neighboors labels
func (knn *KNN) PredictProba(x Matrix) Vector {
	wg := sync.WaitGroup{}
	ret := make(Vector, len(x))
	wg.Add(len(x))
	for i, xx := range x {
		go func(index int, l []float64) {
			c, _ := knn.NearestNeighbors(l).predictProba()
			ret[index] = c
			wg.Done()
		}(i, xx)
	}
	wg.Wait()
	return ret
}

//Predict calculates the value for each row
//from nearests neighboors
func (knn *KNN) Predict(x Matrix) Vector {
	wg := sync.WaitGroup{}
	ret := make(Vector, len(x))
	wg.Add(len(x))
	for i, xx := range x {
		go func(index int, l []float64) {
			c := knn.NearestNeighbors(l).predict()
			ret[index] = c
			wg.Done()
		}(i, xx)
	}
	wg.Wait()
	return ret
}
