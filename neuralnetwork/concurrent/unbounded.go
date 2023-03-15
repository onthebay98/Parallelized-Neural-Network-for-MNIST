package concurrent

import "sync"

/**** YOU CANNOT MODIFY ANY OF THE FOLLOWING INTERFACES/TYPES ********/
type Task interface{}

type DEQueue interface {
	PushBottom(task Task)
	IsEmpty() bool //returns whether the queue is empty
	PopTop() Task
	PopBottom() Task
	Size() int
}

/******** DO NOT MODIFY ANY OF THE ABOVE INTERFACES/TYPES *********************/

// the following is a doubly ended queue that is unbounded
// it's a bit simpler than the book implementation, but it functions the same
// since we use a linked list, we don't need to worry about updating the size

func NewUnBoundedDEQueue() DEQueue {
	return &UnBoundedDEQueue{
		mutex: &sync.Mutex{},
	}
}

type UnBoundedDEQueue struct {
	front *Node
	back  *Node
	mutex *sync.Mutex
}

type Node struct { // holds the actual item being held in the dequeue
	task  Task
	front *Node
	back  *Node
}

// iterates through the list and returns the size
func (q *UnBoundedDEQueue) Size() int {
	q.mutex.Lock()
	defer q.mutex.Unlock()
	size := 0
	curr := q.front
	for curr != nil {
		size++
		curr = curr.back
	}
	return size
}

// pushes new task onto the end of the list
func (q *UnBoundedDEQueue) PushBottom(task Task) {
	q.mutex.Lock()
	defer q.mutex.Unlock()

	node := &Node{}
	node.task = task
	node.front = q.back // set the node front of the current back node to the new node
	node.back = nil     // the new node is the back node, so it's back pointer is nil

	if q.back == nil { // if list is empty, set front to new node
		q.front = node
	} else { // otherwise, set the node behind the current back node to the new node
		q.back.back = node
	}
	q.back = node
}

// checks if list is empty
func (q *UnBoundedDEQueue) IsEmpty() bool {
	q.mutex.Lock()
	defer q.mutex.Unlock()
	return q.front == nil // if front is nil, queue is empty
}

// pops the top task off
func (q *UnBoundedDEQueue) PopTop() Task {
	q.mutex.Lock()
	defer q.mutex.Unlock()
	task := q.front.task
	q.front = q.front.back // move front pointer to the node behind the current front node
	if q.front == nil {
		q.back = nil // if element is the last element, set back to nil
	} else {
		q.front.front = nil
	}
	return task
}

// pops the bottom task off
func (q *UnBoundedDEQueue) PopBottom() Task {
	q.mutex.Lock()
	defer q.mutex.Unlock()
	task := q.back.task
	q.back = q.back.front // move back pointer to the node ahead of the current back node
	if q.back == nil {
		q.front = nil // if element is the last element, set front to nil
	} else {
		q.back.back = nil
	}
	return task
}
