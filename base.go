package knn

import (
	"fmt"
	"math"
	"sort"
)

type DistanceFunction func([]float64, []float64) float64

type BaseKNN struct {
	K        int
	X        [][]float64
	Y        []float64
	Distance DistanceFunction
}

//Neighboor struct with Label Y and it's Index in Training and Distance with a point
type NN struct {
	Index    int
	Distance float64
	Y        float64
}

//Struct of Class for a KNN classifier
type Class struct {
	C     float64 //Class label
	Proba float64 //Probability to be in this class
}

type Classes []Class

//START SORT []Class
func (c Classes) Len() int {
	return len(c)
}

func (c Classes) Less(i, j int) bool {
	return c[i].Proba > c[j].Proba
}

func (c Classes) Swap(i, j int) {
	c[i], c[j] = c[j], c[i]
}

//END SORT []Class

//START SORT []NN
type NNs []NN

func (n NNs) Len() int {
	return len(n)
}

func (n NNs) Less(i, j int) bool {
	return n[i].Distance < n[j].Distance
}

func (n NNs) Swap(i, j int) {
	n[i], n[j] = n[j], n[i]
}

//END SORT []NN

func (nns NNs) PredictProba() (float64, Classes) {
	classes := Classes{}
	labels_classes := map[float64]float64{}
	wtot := 0.0
	for i := range nns {
		c := nns[i].Y
		_, ok := labels_classes[c]
		if ok == false {
			labels_classes[c] = 0.0
		}
		if nns[i].Distance == 0 {
			return nns[i].Y, []Class{Class{C: nns[i].Y, Proba: 1.0}}
		}
		w := 1.0 / nns[i].Distance
		labels_classes[c] += w
		wtot += w
	}
	for k, v := range labels_classes {
		classes = append(classes, Class{C: k, Proba: v / wtot})
	}
	sort.Sort(classes)
	return classes[0].C, classes
}

func (nns NNs) Predict() float64 {
	wtot := 0.0
	ret := 0.0
	for i := range nns {
		if nns[i].Distance == 0 {
			return nns[i].Y
		}
		wtot += 1.0 / nns[i].Distance
	}
	for i := range nns {
		ret += (1.0 / (nns[i].Distance * wtot)) * nns[i].Y
	}
	return ret
}

func NewKNN(k int, x [][]float64, y []float64, d DistanceFunction) *BaseKNN {
	if k > len(x) {
		panic("error k")
	}
	return &BaseKNN{K: k, X: x, Y: y, Distance: d}
}

func (k *BaseKNN) NNeighboors(x []float64) NNs {
	ret := make(NNs, len(k.X))
	for i := range k.X {
		d, err := distance(x, k.X[i], k.Distance)
		if err != nil {
			panic(err)
		}
		ret[i] = NN{Index: i, Distance: d, Y: k.Y[i]}
	}
	sort.Sort(ret)
	return ret[:k.K]
}

func Euclidian(x1 []float64, x2 []float64) float64 {
	sum := 0.0
	for i := range x1 {
		sum += (x1[i] - x2[i]) * (x1[i] - x2[i])
	}
	return math.Sqrt(sum)
}

func distance(x1 []float64, x2 []float64, f DistanceFunction) (float64, error) {
	if x1 == nil {
		return -1, fmt.Errorf("x1 is nil")
	}
	if x2 == nil {
		return -1, fmt.Errorf("x2 is nil")
	}
	if len(x1) != len(x2) {
		return -1, fmt.Errorf("len(x1) [%v] != len(x2) [%v]", len(x1), len(x2))
	}
	if len(x1) == 0 {
		return -1, fmt.Errorf("length x1 and x2 equal 0 ")
	}
	return f(x1, x2), nil
}
