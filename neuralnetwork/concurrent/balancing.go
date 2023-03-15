package concurrent

import (
	"math/rand"
	"sync"
)

type WorkBalancingExecutor struct {
	capacity int
	tasks    int
	index    int
	shutdown bool
	wg       *sync.WaitGroup
	lock     *sync.Mutex
	deques   []DEQueue
	balance  int
}

// NewWorkBalancingExecutor returns an ExecutorService that is implemented using the work-balancing algorithm.
// @param capacity - The number of goroutines in the pool
// @param thresholdQueue - The number of items that a goroutine in the pool can
// grab from the executor in one time period. For example, if threshold = 10
// this means that a goroutine can grab 10 items from the executor all at
// once to place into their local queue before grabbing more items. It's
// not required that you use this parameter in your implementation.
// @param thresholdBalance - The threshold used to know when to perform
// balancing. Remember, if two local queues are to be balanced the
// difference in the sizes of the queues must be greater than or equal to
// thresholdBalance. You must use this parameter in your implementation.
func NewWorkBalancingExecutor(capacity, thresholdQueue, thresholdBalance int) ExecutorService {
	// create an array of deques - one for each thread
	deque := make([]DEQueue, capacity)
	for i := 0; i < capacity; i++ {
		deque[i] = NewUnBoundedDEQueue()
	}

	// create the executor
	executor := &WorkBalancingExecutor{
		capacity: capacity,
		tasks:    0,
		index:    0,
		wg:       &sync.WaitGroup{},
		lock:     &sync.Mutex{},
		deques:   deque,
		balance:  thresholdBalance,
	}

	executor.BeginExecutor() // launch the threads

	return executor
}

// add goroutines and start executing tasks on the local deques
func (executor *WorkBalancingExecutor) BeginExecutor() {
	for i := 0; i < executor.capacity; i++ {
		executor.wg.Add(1)
		go executor.BalancingWorker(i)
	}
}

func (executor *WorkBalancingExecutor) Submit(task interface{}) Future {
	executor.lock.Lock()
	defer executor.lock.Unlock()
	executor.index = executor.index % executor.capacity
	executor.deques[executor.index].PushBottom(task) // add task to the end of the deque
	executor.tasks++
	executor.index++
	return nil // nil future
}

func (executor *WorkBalancingExecutor) Shutdown() {
	executor.lock.Lock()
	executor.shutdown = true
	executor.lock.Unlock()
	executor.wg.Wait() // wait until all goroutines are done
}

// this was taken pretty much straight from the art of multiprocessing textbook
// run tasks; threadId refers to the local deque the thread will be operating on
func (executor *WorkBalancingExecutor) BalancingWorker(threadId int) {
	defer executor.wg.Done()
	// there will be a gap between when this loop starts iterating and tasks are submitted
	// our "shutdown" variable prevents this loop from terminating before all tasks have been submitted
	for {
		size := executor.deques[threadId].Size()
		if executor.deques[threadId].IsEmpty() {
			if executor.shutdown && executor.tasks == 0 {
				return // no more work to do
			}
		} else if rand.Intn(executor.deques[threadId].Size()+1) == size {
			victim := rand.Intn(executor.capacity)
			var min int
			var max int
			if victim > threadId {
				min = threadId
				max = victim
			} else {
				min = victim
				max = threadId
			}

			var qMin DEQueue
			var qMax DEQueue
			if executor.deques[min].Size() > executor.deques[max].Size() {
				qMin = executor.deques[min]
				qMax = executor.deques[max]
			} else {
				qMin = executor.deques[min]
				qMax = executor.deques[max]
			}
			diff := qMax.Size() - qMin.Size()

			if diff > executor.balance {
				for qMax.Size() > qMin.Size() {
					qMin.PushBottom(qMax.PopTop())
				}
			}

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
