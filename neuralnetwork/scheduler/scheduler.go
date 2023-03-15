package scheduler

import (
	"proj3/concurrent"
)

type Config struct {
	Mode string // Represents which scheduler scheme to use
	// If Mode == "s" run the sequential version
	// If Mode == "p" run the parallel version
	// These are the only values for Version
	ThreadCount int // Runs the parallel version of the program with the
	// specified number of threads (i.e., goroutines)
	Epochs int // The number of epochs to run the neural network for
}

// Run the correct version based on the Mode field of the configuration value
func Schedule(config Config) {
	if config.Mode == "s" {
		RunSequential(config)
	} else if config.Mode == "ws" || config.Mode == "wb" {
		RunParallel(config)
	} else {
		panic("Invalid scheduling scheme given.")
	}
}

// we only need to have one global array to store all of our results
type SharedContext struct {
	AllWeightsAndBiases [](weightsAndBiases)
}

// a TrainingBatch consists of a shared context, a batch of training data, and a batch of training labels
type TrainingBatch struct {
	ctx    *SharedContext
	xTrain [][]float64
	yTrain []float64
	id     int
	epochs int
}

func NewSharedContext(ctx *SharedContext, xTrain [][]float64, yTrain []float64, id int, epochs int) concurrent.Runnable {
	return &TrainingBatch{ctx, xTrain, yTrain, id, epochs}
}

// our neural network will have 784 input nodes, 1 hidden layer with 10 nodes, and 10 output nodes (one for each digit)
// extensively referenced https://www.youtube.com/watch?v=w8yWXqWQYmU for the general structure of the neural network
func RunSequential(config Config) {
	xTrain, yTrain, xTest, yTest := LoadData(config)

	weightsAndBiases := GradientDescent(xTrain, yTrain, 0.1, config.Epochs) // returns final weights and biases

	// generates accuracy for test data
	// testPredictions := MakePredictions(xTest, weightsAndBiases.w1, weightsAndBiases.b1, weightsAndBiases.w2, weightsAndBiases.b2)
	// fmt.Println("test accuracy: ")
	// fmt.Println(GetAccuracy(testPredictions, yTest))
	GetAccuracy(MakePredictions(xTest, weightsAndBiases.w1, weightsAndBiases.b1, weightsAndBiases.w2, weightsAndBiases.b2), yTest)
}

// runs gradient descent on a batch of training data
// adds the final weights and biases to the shared context
// for parallel
func (task *TrainingBatch) Run() {
	weightsAndBiases := GradientDescent(task.xTrain, task.yTrain, 0.1, task.epochs) // returns final weights and biases for one training batch

	// add the weights and biases from one training batch to the shared context
	task.ctx.AllWeightsAndBiases = append(task.ctx.AllWeightsAndBiases, weightsAndBiases)
}

func RunParallel(config Config) {
	xTrain, yTrain, xTest, yTest := LoadData(config)

	// initialize executor and load it with tasks
	// we use a form a data parallelism + ensemble learning
	// in other words, we split up our training data, run each split through the neural network, and average the results

	// initialize SharedContext with an empty global array of weights and biases, size is our number of threads
	context := SharedContext{make([](weightsAndBiases), config.ThreadCount)}

	// initialize executor
	var executor concurrent.ExecutorService
	if config.Mode == "ws" {
		executor = concurrent.NewWorkStealingExecutor(config.ThreadCount, 10)
	} else if config.Mode == "wb" {
		executor = concurrent.NewWorkBalancingExecutor(config.ThreadCount, 10, 10)
	}

	width := 1000
	for i := 0; i < 60; i++ { // split training set into 60 chunks
		chunkCeil := width * i
		chunkFloor := chunkCeil + width

		// https://stackoverflow.com/questions/54507818/selecting-2d-sub-slice-of-a-2d-slice-using-ranges-in-go
		xTraincopy := Clone(xTrain)
		b := xTraincopy[0:784]
		for i := range b {
			b[i] = b[i][chunkCeil:chunkFloor]
		}

		// submit each chunk to the executor
		// we're sending a pointer to the shared context, the training and test data, and the id of the chunk
		// we send the id so that we can distribute the tasks evenly across threads
		executor.Submit(NewSharedContext(&context, b, yTrain[chunkCeil:chunkFloor], i, config.Epochs))
	}
	// blocks until all tasks are complete
	executor.Shutdown()
	// averages the weights and biases from all of the training batches
	weightsAndBiases := AggregateResults(context.AllWeightsAndBiases[config.ThreadCount:])

	// generates accuracy for test data
	// testPredictions := MakePredictions(xTest, weightsAndBiases.w1, weightsAndBiases.b1, weightsAndBiases.w2, weightsAndBiases.b2)
	// fmt.Println("test accuracy: ")
	// fmt.Println(GetAccuracy(testPredictions, yTest))
	GetAccuracy(MakePredictions(xTest, weightsAndBiases.w1, weightsAndBiases.b1, weightsAndBiases.w2, weightsAndBiases.b2), yTest)
}
