package main

import "fmt"
import "time"
import "math"
import "math/rand"

func main() {
  var mutationRate float64 = 0.1
  var acceptableError float64 = 0.001
  inputs := []float64{0.3, 1.4}
  target := 1.7

  rand.Seed(time.Now().UTC().UnixNano()) // Seed the random generator *once*

  geneticNet := SetupNeuralNetwork(rand.Float64(),rand.Float64())
  geneticSelector := Selector{geneticNet, 0.0, mutationRate}

  err := 1.0
  iterations := 0

  for err > acceptableError {
    geneticNet = *geneticSelector.Mutate()
    geneticNet.inputs = inputs
    geneticNet.Update()

    output := geneticNet.outputs[0]
    // Run backprop network
    backpropNet := SetupNeuralNetwork(output, 1.0)
    propagator := BackPropogator{backpropNet, target, 1.0}
    for iterations = 0; iterations < 50; iterations++ {
      propagator.nn.inputs = inputs
      propagator.nn.Update()
      propOutput := propagator.nn.outputs[0]
      if propOutput == target {
        break
      }
      propagator.Propogate(propOutput)
    }
    err = math.Abs(propagator.nn.outputs[0] - target) / target
    score := 10.0 - err
    geneticSelector.Select(geneticNet, score)
    fmt.Println("Err:",err,"Out:", propagator.nn.outputs[0])
  }

  fmt.Println("Final error:", (err * 100), "% in ", iterations, " iterations")
}
