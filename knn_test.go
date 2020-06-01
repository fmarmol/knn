package knn

import "testing"

func TestPredict(t *testing.T) {
	tt := []struct {
		name string
		X    Matrix
		y    Vector
		k    int
		x    Matrix
		p    Vector
	}{
		{
			X: Matrix{Vector{1, 2, 3}, Vector{3, 2, 1}},
			y: Vector{1, 2},
			k: 1,
			x: Matrix{Vector{1, 2, 3}},
			p: Vector{1},
		},
	}
	for _, tc := range tt {
		for index, neighboor := range New(tc.k, tc.X, tc.y, Euclidian).Predict(tc.x) {
			if neighboor != tc.p[index] {
				t.Fatal("nop")
			}
		}
	}
}
