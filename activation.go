package main

import (
	"math"
)

// gelu
func Gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

// Sigmoid
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Swish
func Swish(x float64) float64 {
	return x * Sigmoid(x)
}

// ReLU (Rectified Linear Unit)
func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// LeakyReLU (Leaky Rectified Linear Unit)
func LeakyReLU(x, alpha float64) float64 {
	if x > 0 {
		return x
	}
	return alpha * x
}

// Tanh (Hyperbolic Tangent)
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

// SwiGLU layer
type SwiGLU struct {
	bias  bool
	dim   int
	dense *Tensor
}

// NewSwiGLU creates a new SwiGLU layer
func NewSwiGLU(bias bool, dim int) *SwiGLU {
	swiglu := &SwiGLU{
		bias:  bias,
		dim:   dim,
		dense: &Tensor{data: make([]float64, dim), shape: []int{2, dim}}, // Adjust the shape as needed
	}
	return swiglu
}

// Forward computes the SwiGLU activation
func (s *SwiGLU) Forward(x *Tensor) *Tensor {
	rows := x.shape[0]
	cols := x.shape[1]

	// Split the input into two parts
	outData := make([]float64, rows*cols/2)
	gateData := make([]float64, rows*cols/2)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols/2; j++ {
			idx := i*cols + j
			outData[i*cols/2+j] = x.data[idx]
			gateData[i*cols/2+j] = x.data[idx+cols/2]
		}
	}

	out := &Tensor{data: outData, shape: []int{rows, cols / 2}}
	gate := &Tensor{data: gateData, shape: []int{rows, cols / 2}}

	// Apply Swish activation to the gate
	for i := range gate.data {
		gate.data[i] *= Sigmoid(gate.data[i])
	}

	// Multiply out and gate
	for i := range out.data {
		out.data[i] *= gate.data[i]
	}

	return out
}
