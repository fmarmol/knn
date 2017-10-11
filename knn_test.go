package knn

import "testing"

func TestPredict(t *testing.T) {
	tt := []struct {
		name string
		X    matrix
		y    vector
		k    int
		x    matrix
		p    vector
	}{
		{
			X: matrix{vector{1, 2, 3}, vector{3, 2, 1}},
			y: vector{1, 2},
			k: 1,
			x: matrix{vector{1, 2, 3}},
			p: vector{1},
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
