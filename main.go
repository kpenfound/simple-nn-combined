package main

import "fmt"
import "time"
import "math"
import "math/rand"

func main() {
  var mutationRate float64 = 0.1 // 10% mutation rate, the first genetic experiment showed 10% as a strong rate
  var acceptableError float64 = 0.001 // Accuracy of 0.1%
  inputs := []float64{0.3, 1.4}
  target := 1.7

  rand.Seed(time.Now().UTC().UnixNano()) // Seed the random generator /once/

  geneticNet := SetupNeuralNetwork(rand.Float64(),rand.Float64())
  geneticSelector := Selector{geneticNet, 0.0, mutationRate}

  err := 1.0
  iterations := 0

  for err > acceptableError { // Mutate until we are within our acceptable error
    geneticNet = *geneticSelector.Mutate()
    geneticNet.inputs = inputs
    geneticNet.Update()

    output := geneticNet.outputs[0]
    // Run backprop network
    backpropNet := SetupNeuralNetwork(output, 1.0)
    propagator := BackPropagator{backpropNet, target, 1.0}
    for iterations = 0; iterations < 50; iterations++ { // Only do 50 iterations, should be more than enough
      propagator.nn.inputs = inputs
      propagator.nn.Update()
      propOutput := propagator.nn.outputs[0]
      if propOutput == target { // Hit our target exactly, stop here
        break
      }
      propagator.Propagate(propOutput)
    }
    err = math.Abs(propagator.nn.outputs[0] - target) / target // calculate % error from target
    score := 10.0 - err
    geneticSelector.Select(geneticNet, score)
    fmt.Println("Err:",err,"Out:", propagator.nn.outputs[0])
  }

  fmt.Println("Final error:", (err * 100), "% in ", iterations, " iterations")
}
