package concurrent

import (
	"math/rand"
	"sync"
)

type WorkStealingExecutor struct {
	capacity int
	tasks    int
	index    int
	shutdown bool
	wg       *sync.WaitGroup
	lock     *sync.Mutex
	deques   []DEQueue
}

// NewWorkStealingExecutor returns an ExecutorService that is implemented using the work-stealing algorithm.
// @param capacity - The number of goroutines in the pool
// @param threshold - The number of items that a goroutine in the pool can
// grab from the executor in one time period. For example, if threshold = 10
// this means that a goroutine can grab 10 items from the executor all at
// once to place into their local queue before grabbing more items. It's
// not required that you use this parameter in your implementation.
func NewWorkStealingExecutor(capacity, threshold int) ExecutorService {
	// create an array of deques - one for each thread
	deque := make([]DEQueue, capacity)
	for i := 0; i < capacity; i++ {
		deque[i] = NewUnBoundedDEQueue()
	}

	// create the executor
	executor := &WorkStealingExecutor{
		capacity: capacity,
		tasks:    0,
		index:    0,
		wg:       &sync.WaitGroup{},
		lock:     &sync.Mutex{},
		deques:   deque,
	}

	executor.BeginExecutor() // launch the threads

	return executor
}

// add goroutines and start executing tasks on the local deques
func (executor *WorkStealingExecutor) BeginExecutor() {
	for i := 0; i < executor.capacity; i++ {
		executor.wg.Add(1)
		go executor.StealingWorker(i)
	}
}

func (executor *WorkStealingExecutor) Submit(task interface{}) Future {
	executor.lock.Lock()
	defer executor.lock.Unlock()
	executor.index = executor.index % executor.capacity
	executor.deques[executor.index].PushBottom(task) // add task to the end of the deque
	executor.tasks++
	executor.index++
	return nil // nil future
}

func (executor *WorkStealingExecutor) Shutdown() {
	executor.lock.Lock()
	executor.shutdown = true
	executor.lock.Unlock()
	executor.wg.Wait() // wait until all goroutines are done
}

// run tasks; threadId refers to the local deque the thread will be operating on
func (executor *WorkStealingExecutor) StealingWorker(threadId int) {
	defer executor.wg.Done()
	// there will be a gap between when this loop starts iterating and tasks are submitted
	// our "shutdown" variable prevents this loop from terminating before all tasks have been submitted
	for {
		if executor.deques[threadId].IsEmpty() {
			if executor.shutdown && executor.tasks == 0 {
				return // no more work to do
			}

			// pick a random deque
			randDequeId := threadId
			for randDequeId == threadId { // make sure the random deque is not the same as the current thread
				randDequeId = rand.Intn(executor.capacity)
			}

			// lock here to prevent the random deque from becoming empty before we poptop
			executor.lock.Lock()
			if !executor.deques[randDequeId].IsEmpty() { // if not empty, then steal
				task, _ := executor.deques[randDequeId].PopTop().(Runnable) // https://go.dev/tour/methods/15
				task.Run()
				executor.tasks--
			}
			executor.lock.Unlock()

		} else {
			// there is work to do
			task, _ := executor.deques[threadId].PopTop().(Runnable)
			executor.lock.Lock()
			task.Run()
			executor.tasks--
			executor.lock.Unlock()
		}

	}
}
