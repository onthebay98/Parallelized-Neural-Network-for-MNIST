package main

import (
	"fmt"
	"os"
	"proj3/scheduler"
	"strconv"
	"time"
)

const usage = "Usage: editor mode [number of threads]\n" +
	"mode     = (bsp) run the BSP mode, (pipeline) run the pipeline mode\n" +
	"number of epochs\n" +
	"[number of threads] = Runs the parallel version of the program with the specified number of threads.\n"

func main() {
	if len(os.Args) < 3 {
		fmt.Println(usage)
		return
	}

	config := scheduler.Config{Mode: "", Epochs: 0, ThreadCount: 0}
	if len(os.Args) >= 4 {
		config.Epochs, _ = strconv.Atoi(os.Args[1])
		config.Mode = os.Args[2]
		config.ThreadCount, _ = strconv.Atoi(os.Args[3])
	} else {
		config.Mode = "s"
		config.Epochs, _ = strconv.Atoi(os.Args[1])
	}

	start := time.Now()
	scheduler.Schedule(config)
	end := time.Since(start).Seconds()
	fmt.Printf("%.2f\n", end)

}
