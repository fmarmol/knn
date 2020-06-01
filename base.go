package knn

import (
	"fmt"
	"math"
	"sort"
)

// Vector is a Vector of float64
type Vector = []float64

// Matrix is an alias for [][]float64
type Matrix = []Vector

// DistanceFunction is an alias
type DistanceFunction = func(Vector, Vector) float64

// Base struct
type base struct {
	K        int              // number of nieghbors
	X        Matrix           // X matrix of features
	Y        Vector           // y vector of value or labels
	Distance DistanceFunction // distance function used
}

// Class for a KNN classifier
type Class struct {
	C     float64 //Class label
	Proba float64 //Probability to be in this class
}

type classes []Class

// Len returns the number of classes
func (c classes) Len() int {
	return len(c)
}

// Less to implement sort interface
func (c classes) Less(i, j int) bool {
	return c[i].Proba > c[j].Proba
}

// Swap to implement sort interface
func (c classes) Swap(i, j int) {
	c[i], c[j] = c[j], c[i]
}

//Neighbor struct with Label Y and it's Index in Training and Distance with a point
type Neighbor struct {
	Index    int     // index in X matrix
	Distance float64 // distance from a row
	Y        float64 // value or label
}

type neighbors []Neighbor

// Len returns the number of neighboors
func (nns neighbors) Len() int {
	return len(nns)
}

// Less to implement sort interface
func (nns neighbors) Less(i, j int) bool {
	return nns[i].Distance < nns[j].Distance
}

// Swap to implement sort interface
func (nns neighbors) Swap(i, j int) {
	nns[i], nns[j] = nns[j], nns[i]
}

// predictProba ...
func (nns neighbors) predictProba() (float64, classes) {
	cls := classes{}
	labelsClasses := map[float64]float64{}
	wtot := 0.0
	for i := range nns {
		c := nns[i].Y
		if _, ok := labelsClasses[c]; !ok {
			labelsClasses[c] = 0.0

		}
		if nns[i].Distance == 0 {
			return nns[i].Y, []Class{{C: nns[i].Y, Proba: 1.0}}
		}
		w := 1.0 / nns[i].Distance
		labelsClasses[c] += w
		wtot += w
	}
	for k, v := range labelsClasses {
		cls = append(cls, Class{C: k, Proba: v / wtot})
	}
	sort.Sort(cls)
	return cls[0].C, cls
}

// Predict
func (nns neighbors) predict() float64 {
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

// NewKNN creates a new base knn
func newKNN(k int, x Matrix, y Vector, d DistanceFunction) *base {
	if k > len(x) {
		panic("error k")
	}
	return &base{K: k, X: x, Y: y, Distance: d}
}

// NNeighbors returns the k nearests neighboors
func (b *base) NearestNeighbors(x Vector) neighbors {
	ret := make(neighbors, len(b.X))
	for i := range b.X {
		d, err := distance(x, b.X[i], b.Distance)
		if err != nil {
			panic(err)
		}
		ret[i] = Neighbor{Index: i, Distance: d, Y: b.Y[i]}
	}
	sort.Sort(ret)
	return ret[:b.K]
}

// Euclidian calculate the euclidian distance
func Euclidian(x1, x2 Vector) float64 {
	sum := 0.0
	for i := range x1 {
		sum += (x1[i] - x2[i]) * (x1[i] - x2[i])
	}
	return math.Sqrt(sum)
}

// distance apply a distanceFunction on two vectors
func distance(x1 Vector, x2 Vector, f DistanceFunction) (float64, error) {
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
