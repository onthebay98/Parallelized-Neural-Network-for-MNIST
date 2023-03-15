package scheduler

import (
	"math"
	"math/rand"
	"time"
)

// referenced https://www.youtube.com/watch?v=w8yWXqWQYmU for functions directly related to the neural network
// referenced ChatGPT for some functions related to matrix operations

type weightsAndBiases struct {
	w1 [][]float64
	b1 [][]float64
	w2 [][]float64
	b2 [][]float64
}

// initialize weights and biases
// weight1 is 10x784, bias1 is 10x1, weight2 is 10x10, bias2 is 10x1
// weights are initialized to random values between -0.5 and 0.5
// biases are initialized to 0
func Init_params() ([][]float64, [][]float64, [][]float64, [][]float64) {
	rand.Seed(time.Now().UnixNano()) // https://stackoverflow.com/questions/68203678/golang-rand-int-why-every-time-same-values
	w1 := make([][]float64, 10)
	b1 := make([][]float64, 10)
	w2 := make([][]float64, 10)
	b2 := make([][]float64, 10)
	for i := 0; i < 10; i++ {
		w1[i] = make([]float64, 784)
		b1[i] = make([]float64, 1)
		w2[i] = make([]float64, 10)
		b2[i] = make([]float64, 1)

		// biases set to 0
		b1[i][0] = rand.Float64() - 0.5
		b2[i][0] = rand.Float64() - 0.5

		// random weights for w1
		for j := 0; j < 784; j++ {
			w1[i][j] = rand.Float64() - 0.5
		}
		// random weights for w2
		for j := 0; j < 10; j++ {
			w2[i][j] = rand.Float64() - 0.5
		}
	}
	return w1, b1, w2, b2
}

// computes the dot product of two matrices
func Dot(a [][]float64, b [][]float64) [][]float64 {
	aRows := len(a)
	bRows := len(b)
	bCols := len(b[0])

	// create a new array to store the dot product
	// has the same number of rows as A and the same number of columns as B
	c := make([][]float64, aRows)
	for i := range c {
		c[i] = make([]float64, bCols)
	}

	// compute the dot product of each row of A and each column of B
	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			sum := 0.0
			for k := 0; k < bRows; k++ {
				sum += a[i][k] * b[k][j]
			}
			c[i][j] = sum
		}
	}
	return c
}

// adds matrices of equal size
func Add(a [][]float64, b [][]float64) [][]float64 {
	rows := len(a)
	cols := len(a[0])

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a[i][j] += b[i][j]
		}
	}
	return a
}

// subtracts matrices of equal size
func Subtract(a [][]float64, b [][]float64) [][]float64 {
	rows := len(a)
	cols := len(a[0])

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a[i][j] -= b[i][j]
		}
	}
	return a
}

// element-wise multiplication of matrices of equal size
func Multiply(a [][]float64, b [][]float64) [][]float64 {
	rows := len(a)
	cols := len(a[0])

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a[i][j] *= b[i][j]
		}
	}
	return a
}

// subtracts scalar from matrix
func ScalarSubtract(a [][]float64, scalar float64) [][]float64 {
	rows := len(a)
	cols := len(a[0])

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a[i][j] -= scalar
		}
	}
	return a
}

// multiples matrix by a scalar
func ScalarMultiply(scalar float64, a [][]float64) [][]float64 {
	rows := len(a)
	cols := len(a[0])

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a[i][j] *= scalar
		}
	}
	return a
}

// matrix multiply
func MatrixMultiply(a, b [][]float64) [][]float64 {
	aRows, aCols := len(a), len(a[0])
	_, bCols := len(b), len(b[0])

	// make a new array to store the product
	c := make([][]float64, aRows)
	for i := range c {
		c[i] = make([]float64, bCols)
	}

	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			var sum float64
			for k := 0; k < aCols; k++ {
				sum += a[i][k] * b[k][j]
			}
			c[i][j] = sum
		}
	}
	return c
}

// adds a vector to a matrix, row-wise
func AddVectorToMatrix(a [][]float64, b [][]float64) [][]float64 {
	rows := len(a)
	cols := len(a[0])

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a[i][j] += b[i][0]
		}
	}
	return a
}

// transpose a matrix
func Transpose(a [][]float64) [][]float64 {
	rows := len(a)
	cols := len(a[0])
	aT := make([][]float64, cols)

	for i := 0; i < cols; i++ {
		aT[i] = make([]float64, rows)
		for j := 0; j < rows; j++ {
			aT[i][j] = a[j][i]
		}
	}
	return aT
}

// ReLU activation function
// returns 0 if Z < 0, otherwise returns Z
func ReLU(a [][]float64) [][]float64 {
	rows := len(a)
	cols := len(a[0])

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if a[i][j] < 0 {
				a[i][j] = 0
			}
		}
	}

	return a
}

// derivative of ReLU
// returns 0 if Z < 0, otherwise returns 1
func DerivativeReLU(a [][]float64) [][]float64 {
	rows := len(a)
	cols := len(a[0])

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if a[i][j] <= 0 {
				a[i][j] = 0
			} else {
				a[i][j] = 1
			}
		}
	}

	return a
}

// softmax activation function
func Softmax(matrix [][]float64) [][]float64 {
	rows := len(matrix)
	cols := len(matrix[0])
	softmaxed := make([][]float64, rows)

	for i := 0; i < rows; i++ {
		softmaxed[i] = make([]float64, cols)
	}

	for j := 0; j < cols; j++ {
		colSum := 0.0
		for i := 0; i < rows; i++ {
			colSum += math.Exp(matrix[i][j])
		}
		for i := 0; i < rows; i++ {
			softmaxed[i][j] = math.Exp(matrix[i][j]) / colSum
			if math.IsNaN(softmaxed[i][j]) {
				softmaxed[i][j] = 0
			}
		}
	}

	return softmaxed
}

// one hot encoding for labels
func OneHot(labels []float64) [][]float64 {
	oneHotY := make([][]float64, len(labels))
	for i := range oneHotY {
		oneHotY[i] = make([]float64, 10)
		oneHotY[i][int(labels[i])] = 1
	}
	return Transpose(oneHotY)
}

// argmax function
func Argmax(a [][]float64) []float64 {
	max := make([]float64, len(a[0]))
	for i := 0; i < len(a[0]); i++ {
		max[i] = 0
		for j := 0; j < len(a); j++ {
			if a[j][i] > a[int(max[i])][i] {
				max[i] = float64(j)
			}
		}
	}
	return max
}

// Sum 2D matrix
func Sum(a [][]float64) float64 {
	sum := 0.0
	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a[i]); j++ {
			sum += a[i][j]
		}
	}
	return sum
}

// Sum 2D matrix row-wise; returns a 2D matrix with 1 column
func SumRows(matrix [][]float64) [][]float64 {
	sums := make([][]float64, 0)

	for i := 0; i < len(matrix); i++ {
		sum := 0.0
		for j := 0; j < len(matrix[i]); j++ {
			sum += matrix[i][j]
		}
		sums = append(sums, []float64{sum})
	}

	return sums
}

// forward propagation
func Forward_prop(w1 [][]float64, b1 [][]float64, w2 [][]float64, b2 [][]float64, x [][]float64) ([][]float64, [][]float64, [][]float64, [][]float64) {
	w1copy := Clone(w1)
	xcopy := Clone(x)
	b1copy := Clone(b1)
	z1 := AddVectorToMatrix(Dot(w1copy, xcopy), b1copy) // 10 x m

	z1copy := Clone(z1)
	a1 := ReLU(z1copy) // 10 x m

	w2copy := Clone(w2)
	a1copy := Clone(a1)
	b2copy := Clone(b2)
	z2 := AddVectorToMatrix(Dot(w2copy, a1copy), b2copy) // 10 x m

	z2copy := Clone(z2)
	a2 := Softmax(z2copy) // 10 x m
	return z1, a1, z2, a2
}

// back propagation
func Back_prop(z1 [][]float64, a1 [][]float64, z2 [][]float64, a2 [][]float64, w1 [][]float64, w2 [][]float64, x [][]float64, y []float64) ([][]float64, [][]float64, [][]float64, [][]float64) {
	ycopy := make([]float64, len(y))
	copy(ycopy, y)
	oneHotY := OneHot(ycopy)

	a2copy := Clone(a2)
	oneHotYcopy := Clone(oneHotY)
	dz2 := ScalarMultiply(2, Subtract(a2copy, oneHotYcopy))

	dz2copy := Clone(dz2)
	a1copy := Clone(a1)
	dw2 := ScalarMultiply(1/float64(len(y)), Dot(dz2copy, Transpose(a1copy)))

	dz2copy2 := Clone(dz2)
	db2 := ScalarMultiply(1.0/float64(len(y)), SumRows(dz2copy2))

	w2copy := Clone(w2)
	dz2copy3 := Clone(dz2)
	z1copy := Clone(z1)
	dz1 := Multiply(Dot(Transpose(w2copy), dz2copy3), DerivativeReLU(z1copy))

	dz1copy := Clone(dz1)
	xcopy := Clone(x)
	dw1 := ScalarMultiply(1/float64(len(y)), Dot(dz1copy, Transpose(xcopy)))

	dz1copy2 := Clone(dz1)
	db1 := ScalarMultiply(1/float64(len(y)), SumRows(dz1copy2))
	return dw1, db1, dw2, db2
}

// updates the parameters
func UpdateParameters(w1 [][]float64, b1 [][]float64, w2 [][]float64, b2 [][]float64, dw1 [][]float64, db1 [][]float64, dw2 [][]float64, db2 [][]float64, learningRate float64) ([][]float64, [][]float64, [][]float64, [][]float64) {
	w1copy := Clone(w1)
	dw1copy := Clone(dw1)
	w1 = Subtract(w1copy, ScalarMultiply(learningRate, dw1copy))

	b1copy := Clone(b1)
	db1copy := Clone(db1)
	b1 = Subtract(b1copy, ScalarMultiply(learningRate, db1copy))

	w2copy := Clone(w2)
	dw2copy := Clone(dw2)
	w2 = Subtract(w2copy, ScalarMultiply(learningRate, dw2copy))

	b2copy := Clone(b2)
	db2copy := Clone(db2)
	b2 = Subtract(b2copy, ScalarMultiply(learningRate, db2copy))
	return w1, b1, w2, b2
}

// get accuracy of the model
func GetAccuracy(yPred []float64, y []float64) float64 {
	accuracy := 0.0
	for i := 0; i < len(y); i++ {
		if yPred[i] == y[i] {
			accuracy++
		}
	}
	return accuracy / float64(len(y))
}

// forward prop => back prop => update params => repeat
func GradientDescent(x [][]float64, y []float64, learningRate float64, epochs int) weightsAndBiases {
	w1, b1, w2, b2 := Init_params() // these are coming out fine every epoch
	for i := 0; i < epochs; i++ {
		w1copy := Clone(w1)
		b1copy := Clone(b1)
		w2copy := Clone(w2)
		b2copy := Clone(b2)

		w1copy2 := Clone(w1)
		w2copy2 := Clone(w2)

		w1copy3 := Clone(w1)
		b1copy3 := Clone(b1)
		w2copy3 := Clone(w2)
		b2copy3 := Clone(b2)

		xcopy := Clone(x)
		xcopy2 := Clone(x)
		z1, a1, z2, a2 := Forward_prop(w1copy, b1copy, w2copy, b2copy, xcopy)

		a1copy := Clone(a1)
		a2copy := Clone(a2)
		//a2copy2 := Clone(a2)
		z1copy := Clone(z1)
		z2copy := Clone(z2)

		dw1, db1, dw2, db2 := Back_prop(z1copy, a1copy, z2copy, a2copy, w1copy2, w2copy2, xcopy2, y)

		dw1copy := Clone(dw1)
		db1copy := Clone(db1)
		dw2copy := Clone(dw2)
		db2copy := Clone(db2)

		w1, b1, w2, b2 = UpdateParameters(w1copy3, b1copy3, w2copy3, b2copy3, dw1copy, db1copy, dw2copy, db2copy, learningRate)

		// if i%10 == 0 {
		// 	fmt.Println("accuracy: ")
		// 	fmt.Printf("%f", GetAccuracy(Argmax(a2copy2), y))
		// 	fmt.Println("")
		// }
	}

	return weightsAndBiases{w1, b1, w2, b2}
}

func MakePredictions(x [][]float64, w1 [][]float64, b1 [][]float64, w2 [][]float64, b2 [][]float64) []float64 {
	_, _, _, a2 := Forward_prop(w1, b1, w2, b2, x)
	return Argmax(a2)
}

// this function averages all of our weights and biases and returns a new weightsAndBiases struct
func AggregateResults(allWeightsAndBiases [](weightsAndBiases)) weightsAndBiases {

	finalw1 := make([][]float64, 10)
	for i := range finalw1 {
		finalw1[i] = make([]float64, 784)
	}
	for i := 0; i < 10; i++ {
		for j := 0; j < 784; j++ {
			// for each weight and bias
			for k := 0; k < len(allWeightsAndBiases); k++ {
				finalw1[i][j] += allWeightsAndBiases[k].w1[i][j]
			}
			finalw1[i][j] /= float64(len(allWeightsAndBiases))
		}
	}

	finalw2 := make([][]float64, 10)
	for i := range finalw2 {
		finalw2[i] = make([]float64, 10)
	}
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			// for each weight and bias
			for k := 0; k < len(allWeightsAndBiases); k++ {
				finalw2[i][j] += allWeightsAndBiases[k].w2[i][j]
			}
			finalw2[i][j] /= float64(len(allWeightsAndBiases))
		}
	}

	finalb1 := make([][]float64, 10)
	finalb2 := make([][]float64, 10)
	for i := range finalb1 {
		finalb1[i] = make([]float64, 1)
		finalb2[i] = make([]float64, 1)
	}
	for i := 0; i < 10; i++ {
		for k := 0; k < len(allWeightsAndBiases); k++ {
			finalb1[i][0] += allWeightsAndBiases[k].b1[i][0]
			finalb2[i][0] += allWeightsAndBiases[k].b2[i][0]
		}
		finalb1[i][0] /= float64(len(allWeightsAndBiases))
		finalb2[i][0] /= float64(len(allWeightsAndBiases))
	}

	return weightsAndBiases{finalw1, finalb1, finalw2, finalb2}
}
