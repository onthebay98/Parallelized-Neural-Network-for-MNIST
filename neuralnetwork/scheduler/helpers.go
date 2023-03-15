package scheduler

import (
	"proj3/mnist"
)

// loads in data
func LoadData(config Config) ([][]float64, []float64, [][]float64, []float64) {
	// use mnist package to load in training and test data
	train, test, err := mnist.Load("../../proj3/mnist")
	if err != nil {
		panic(err)
	}

	// each image is represented as a 784-byte array
	// we convert the images and labels to vectors of float64s
	xTrain := Transpose(ImagesToVectors(train.Images)) // 784x60000
	yTrain := LabelsToVector(train.Labels)             // 60000x1
	xTest := Transpose(ImagesToVectors(test.Images))   // 784x10000
	yTest := LabelsToVector(test.Labels)               // 10000x1
	ScalarMultiply(1.0/255.0, xTrain)                  // normalize the data
	Clone(xTrain)
	ScalarMultiply(1.0/255.0, xTest) // normalize the data

	return xTrain, yTrain, xTest, yTest
}

// converts an array of Images to an array of vectors
func ImagesToVectors(images []*mnist.Image) [][]float64 {
	vectors := make([][]float64, len(images))
	for i := 0; i < len(images); i++ {
		vectors[i] = ImageToVector(images[i])
	}
	return vectors
}

// converts an Image to a vector
func ImageToVector(image *mnist.Image) []float64 {
	vector := make([]float64, 784)
	for i := 0; i < 784; i++ {
		vector[i] = float64(image[i])
	}
	return vector
}

// converts Labels to a vector
func LabelsToVector(labels []mnist.Label) []float64 {
	vectors := make([]float64, len(labels))
	for i := 0; i < len(labels); i++ {
		vectors[i] = float64(labels[i])
	}
	return vectors
}

// Clones a 2D array of floats
// https://stackoverflow.com/questions/68542702/clone-float-slice-in-go-without-affecting-the-original
func Clone(arr [][]float64) (res [][]float64) {
	res = make([][]float64, len(arr))
	for i := range arr {
		res[i] = append([]float64{}, arr[i]...)
	}
	return
}
