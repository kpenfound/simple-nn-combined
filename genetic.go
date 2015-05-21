package main

import "math/rand"

type Selector struct {
  nn NeuronNetwork // Our highest scoring network
  score float64 // Holder for the score associated with high scoring network
  chanceToMutate float64 // Mutation Rate
}

func (s *Selector) Select(newNetwork NeuronNetwork, newScore float64) { // Stores the network with the best score
  if newScore > s.score {
    s.nn = newNetwork
    s.score = newScore
  }
}

func (s *Selector) Mutate() *NeuronNetwork { // Randomly mutates each weight and threshold based on mutation rate
  newNet := s.nn

  for i := 0; i < len(newNet.neuronLayers); i++ {
    for j := 0; j < len(newNet.neuronLayers[i].neurons); j++ {
      for k := 0; k < len(newNet.neuronLayers[i].neurons[j].weights); k++ {
        f1 := rand.Float64()
        if f1 < s.chanceToMutate {
          r1 := rand.Float64() - 0.5 // Between -0.5 and 0.5
          newNet.neuronLayers[i].neurons[j].weights[k] += r1
        }
      }
      f2 := rand.Float64()
      if f2 < s.chanceToMutate {
        r2 := rand.Float64() - 0.5 // Between -0.5 and 0.5
        newNet.neuronLayers[i].neurons[j].threshold += r2
      }
    }
  }

  return &newNet
}
