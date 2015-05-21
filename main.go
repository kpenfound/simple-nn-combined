package main

import "fmt"
import "math"
import "time"
import "os"
import "strconv"

func main() {
  var iterations int
  var mutationRate float64
  var e1, e2 error
  mutationRate, e1 = strconv.ParseFloat(os.Args[len(os.Args) -2], 64)
  iterations, e2 = strconv.Atoi(os.Args[len(os.Args) -1])
  if e1 != nil || e2 != nil {
    fmt.Println("Arguments parsing error")
  }
  inputs := []float64{1.1, 2.1}
  target := 3.4
  tests := 10

  sum := 0.0

  for i := 0; i < tests; i++ {
    res := geneticSimulation(mutationRate, SetupNeuralNetwork(), iterations, inputs, target)

    /*
      Lets ignore the zeroes.  They're going to happen unless we have some additional logic in our network or mutation algorithms.
      Our non-linear function is centered at zero and if any bias gets a 0 at random our output will be zero.
      For the sake of having the most simple example, we'll ignore this case.
    */
    if res == 0.0 {
      i--
    } else {
      sum += math.Abs(res - target) / target
      fmt.Println("Result:",res)
    }
    time.Sleep(time.Second * 1) // Sleeping for one second to ensure that our random seed is different from the previous test
  }
  avgErr := sum / float64(tests) // In the future, std deviation may be more useful

  fmt.Println(iterations, "iterations completed with", (avgErr * 100), "% error")
}



func geneticSimulation(mutation float64, nn NeuronNetwork, iterations int, inputs []float64, target float64) float64 {
  gen := Selector{nn,0.0, mutation} // Initialize genetic model with a 0 score

  for i := 0; i < iterations; i++ {
    nn = *gen.Mutate() // Mutate network
    nn.inputs = inputs

    nn.Update() // Execute network
    score := 1 - math.Abs(target - nn.outputs[0]) // Score to guage accuracy comapred to target
    gen.Select(nn, score) // Select network with the best score
  }

  return gen.nn.outputs[0]
}
