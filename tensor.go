package main

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"strings"
	"sync"
)

// Tensor structure
type Tensor struct {
	data  []float64
	shape []int
}

// Shape method returns the shape of the tensor
func (t *Tensor) Shape() []int {
	return t.shape
}

// NumElements method returns thenumber of element in the tensor
func (t *Tensor) NumElements() int64 {
	n := int64(1)
	for _, d := range t.shape {
		n *= int64(d)
	}
	return n
}

func (t *Tensor) Equal(tensorB Tensor) bool {
	// compare the shape
	if len(t.shape) != len(tensorB.shape) {
		return false
	}
	for i, _ := range t.shape {
		if t.shape[i] != tensorB.shape[i] {
			return false
		}
	}

	// compare each value
	for i, _ := range t.data {
		if t.data[i] != tensorB.data[i] {
			return false
		}
	}
	return true
}

// Strides method returns the strides of the tensor
func (t *Tensor) Strides() ([]int, error) {
	if t == nil {
		return nil, errors.New("tensor is nil")
	}
	// Calculate the strides of the tensor
	strides := make([]int, len(t.shape))
	stride := 1
	for i := len(t.shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= t.shape[i]
	}
	return strides, nil
}

// Reshape method modifies the dimensions of the tensor
func (t *Tensor) Reshape(newShape []int) error {
	// Check if the product of new dimensions is equal to the size of the underlying data array
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}
	if newSize != len(t.data) {
		return errors.New("new shape size does not match data size")
	}
	t.shape = newShape
	return nil
}

// At method returns the value at the specified indices in the tensor
func (t *Tensor) At(indices ...int) (float64, error) {
	// Calculate the index in the data array based on the indices
	index := 0
	for i, dim := range t.shape {
		if len(indices) <= i || indices[i] >= dim || indices[i] < 0 {
			return 0.0, errors.New("index out of bounds")
		}
		index = index*dim + indices[i]
	}
	return t.data[index], nil
}

// SetAt method set the value at the specified indices in the tensor
func (t *Tensor) SetAt(value float64, indices ...int) error {
	// Calculate the index in the data array based on the indices
	index := 0
	for i, dim := range t.shape {
		if len(indices) <= i || indices[i] >= dim || indices[i] < 0 {
			return errors.New("index out of bounds")
		}
		index = index*dim + indices[i]
	}
	t.data[index] = value
	return nil
}

// CopyTo method copies the data from the current tensor to the target tensor
func (t *Tensor) CopyTo(target *Tensor) error {
	if !equalShapes(t.shape, target.shape) {
		return errors.New("shapes are not compatible for copying")
	}
	copy(target.data, t.data)
	return nil
}

// Transpose method transposes the dimensions of the tensor in-place
func (t *Tensor) Transpose() {
	// Check if the tensor has at least 2 dimensions
	if len(t.shape) < 2 {
		return // No change needed for 1D tensors
	}

	// Calculate the new shape for transposition
	newShape := make([]int, len(t.shape))
	for i := range newShape {
		newShape[i] = t.shape[len(t.shape)-1-i]
	}

	transposedStrides := make([]int, len(newShape))
	transposedStrides[0] = 1
	for i := 1; i < len(newShape); i++ {
		transposedStrides[i] = transposedStrides[i-1] * newShape[i-1]
	}

	// Create a copy of the data for swapping
	dataCopy := make([]float64, len(t.data))
	copy(dataCopy, t.data)

	// Perform in-place transposition by swapping values
	for i := 0; i < len(t.data); i++ {
		indicesOriginal := make([]int, len(t.shape))
		indexOriginal := i
		indicesTransposed := make([]int, len(newShape))

		// Calculate indices for the original shape
		for j := 0; j < len(t.shape); j++ {
			indicesOriginal[j] = indexOriginal % t.shape[j]
			indexOriginal /= t.shape[j]
		}

		// Calculate indices for the transposed shape
		for j := 0; j < len(newShape); j++ {
			indicesTransposed[j] = indicesOriginal[len(newShape)-1-j]
		}

		// Calculate the corresponding indices in the dataCopy
		indexCopy := 0
		for j := 0; j < len(newShape); j++ {
			indexCopy += indicesTransposed[j] * transposedStrides[j]
		}

		// Swap values between data and dataCopy
		t.data[i], dataCopy[indexCopy] = dataCopy[indexCopy], t.data[i]
	}

	// Update the shape of the tensor
	t.shape = newShape
}

// Clone method creates a deep copy of the tensor
func (t *Tensor) Clone() *Tensor {
	// Create a new tensor with the same shape
	clone := &Tensor{
		data:  make([]float64, len(t.data)),
		shape: make([]int, len(t.shape)),
	}

	// Copy the data and shape from the original tensor
	copy(clone.data, t.data)
	copy(clone.shape, t.shape)

	return clone
}

// TransposeNew method returns a new tensor with transposed dimensions
func (t *Tensor) TransposeNew() *Tensor {
	// Check if the tensor has at least 2 dimensions
	if len(t.shape) < 2 {
		return t.Clone() // No change needed for 1D tensors, return the same tensor
	}

	// Calculate the new shape for transposition
	newShape := make([]int, len(t.shape))
	for i := range newShape {
		newShape[i] = t.shape[len(t.shape)-1-i]
	}

	// Create a new tensor with the transposed shape
	transposedTensor := &Tensor{
		data:  make([]float64, len(t.data)),
		shape: newShape,
	}

	copy(transposedTensor.data, t.data)
	transposedTensor.Transpose()

	return transposedTensor
}

// SoftMax applies the softmax function to a Tensor along a specified axis and returns a new Tensor.
func (t *Tensor) SoftMax(axis int) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("input tensor is nil")
	}

	if len(t.data) == 0 {
		return nil, errors.New("input tensor is empty")
	}

	if axis < 0 || axis >= len(t.shape) {
		return nil, errors.New("invalid axis")
	}

	// Calculate the number of columns (elements along the specified axis)
	numCols := 1
	for i := 0; i < len(t.shape); i++ {
		if i != axis {
			numCols *= t.shape[i]
		}
	}

	// Calculate the size of each column (number of elements to consider for softmax)
	colSize := t.shape[axis]

	// Create a new tensor to store the softmax result
	outputData := make([]float64, len(t.data))
	copy(outputData, t.data) // Copy the data to the output tensor

	// Apply the softmax function to each column using a worker pool pattern
	var wg sync.WaitGroup
	workerPoolSize := 10 // Set the number of workers in the pool
	jobs := make(chan int, workerPoolSize)
	results := make(chan int, numCols)

	// Start worker goroutines
	for i := 0; i < workerPoolSize; i++ {
		go func() {
			for i := range jobs {
				startIdx := i * colSize
				endIdx := startIdx + colSize

				// Find the maximum value in the column
				maxVal := math.Inf(-1)
				for j := startIdx; j < endIdx; j++ {
					if t.data[j] > maxVal {
						maxVal = t.data[j]
					}
				}

				// Apply the softmax function to the column
				var sum float64
				for j := startIdx; j < endIdx; j++ {
					outputData[j] = math.Exp(t.data[j] - maxVal)
					sum += outputData[j]
				}

				// Normalize the column by dividing each element by the sum
				for j := startIdx; j < endIdx; j++ {
					outputData[j] /= sum
				}

				results <- i
			}
		}()
	}

	// Send jobs to the worker goroutines
	for i := 0; i < numCols; i++ {
		jobs <- i
	}
	close(jobs)

	// Wait for all jobs to be completed
	for i := 0; i < numCols; i++ {
		<-results
	}
	wg.Wait()

	return &Tensor{data: outputData, shape: t.shape}, nil
}

/**
 * LayerNorm performs layer normalization on a Tensor object.
 *
 * Layer normalization is a technique used in machine learning to normalize the values of a tensor along a specified axis.
 *
 * Inputs:
 * - axis (int): The axis along which to perform layer normalization.
 * - g (float64): The scaling factor.
 * - b (float64): The bias factor.
 * - eps (float64): A small value added to the variance to avoid division by zero.
 *
 * Outputs:
 * - normalized (*Tensor): A new Tensor object with the layer-normalized values.
 *
 * Example Usage:
 * t := &Tensor{
 *     data:  []float64{1.0, 2.0, 3.0, 4.0},
 *     shape: []int{2, 2},
 * }
 * normalized, err := t.LayerNorm(1, 1.0, 0.0, 1e-8)
 * if err != nil {
 *     fmt.Println("Error:", err)
 * } else {
 *     fmt.Println("Normalized Tensor:", normalized)
 * }
 */
func (t *Tensor) LayerNorm(axis int, g, b float64, eps float64) (*Tensor, error) {
	/* pseudo-code
	func layerssNorm(x []float64, g, b float64, eps float64) []float64 {
		mean := mean(x)
	   variance := variance(x)
	    x = normalize(x, mean, variance, eps)
	    return scale(x, g, b)
	}

	 func normalize(x []float64, mean, variance, eps float64) []float64 {
	   	for i, v := range x {
	   			x[i] = (v - mean) / math.Sqrt(variance+eps)
	   	}
	   	return x
	   }

	   func scale(x []float64, g, b float64) []float64 {
	   	for i, v := range x {
	   			x[i] = g*v + b
	   	}
	   	return x
	   } */

	var axisSize int
	// validation
	if axis < 0 {
		axis = len(t.shape) + axis
	}
	if axis < 0 || axis >= len(t.shape) {
		return nil, errors.New("invalid axis")
	}

	axisSize = t.shape[axis]

	// Calculate mean and variance along the specified axis
	mean := make([]float64, axisSize)
	variance := make([]float64, axisSize)

	// Calculate mean along the specified axis
	for i := 0; i < axisSize; i++ {
		sum := 0.0
		count := 0
		indices := make([]int, len(t.shape))
		copy(indices, t.shape)

		indices[axis] = i

		for j := range t.data {
			match := true
			for k := range indices {
				if k == axis {
					continue
				}
				index := j / int(math.Pow(float64(t.shape[k]), float64(len(t.shape)-k-1))) % t.shape[k]
				if index != indices[k] {
					match = false
					break
				}
			}
			if match {
				sum += t.data[j]
				count++
			}
		}

		mean[i] = sum / float64(count)
	}

	// Calculate variance along the specified axis
	for i := 0; i < axisSize; i++ {
		sum := 0.0
		count := 0
		indices := make([]int, len(t.shape))
		copy(indices, t.shape)

		indices[axis] = i

		for j := range t.data {
			match := true
			for k := range indices {
				if k == axis {
					continue
				}
				index := j / int(math.Pow(float64(t.shape[k]), float64(len(t.shape)-k-1))) % t.shape[k]
				if index != indices[k] {
					match = false
					break
				}
			}
			if match {
				sum += math.Pow(t.data[j]-mean[i], 2)
				count++
			}
		}

		variance[i] = sum / float64(count)
	}

	// Apply layer normalization
	shapeMultiplier := 1
	for i := 0; i < axis; i++ {
		shapeMultiplier *= t.shape[i]
	}

	resultData := make([]float64, len(t.data))
	for i := 0; i < len(t.data); i++ {
		axisIndex := (i / shapeMultiplier) % axisSize
		resultData[i] = g*(t.data[i]-mean[axisIndex])/math.Sqrt(variance[axisIndex]+eps) + b
	}

	return &Tensor{
		data:  resultData,
		shape: t.shape,
	}, nil
}

// ArgMax returns the index that has the highest value along the specified axis.
// If axis is -1, it computes the global maximum.
//
// Inputs:
// - axis (int): The axis along which to find the maximum value. If -1, it computes the global maximum.
//
// Outputs:
// - maxIndices ([]int): The indices that have the highest values along the specified axis.
// - error: An error indicating the reason for failure, if any.
//
// Example Usage:
//
//	t := &Tensor{
//	    data: []float64{1.2, 3.4, 2.1, 5.6, 4.3, 0.9},
//	    shape: []int{2, 3},
//	}
//
// indices, err := t.ArgMax(1)
//
//	if err != nil {
//	    fmt.Println("Error:", err)
//	} else {
//
//	    fmt.Println("Indices:", indices)
//	}
//
// Expected Output:
// Indices: [1 0]
func (t *Tensor) ArgMax(axis int) ([]int, error) {
	if len(t.data) == 0 {
		return []int{-1}, errors.New("input tensor is empty")
	}

	if axis == -1 {
		maxIndex := 0
		maxValue := t.data[0]

		for i, value := range t.data {
			if value > maxValue {
				maxValue = value
				maxIndex = i
			}
		}

		return []int{maxIndex}, nil
	}

	if axis < 0 || axis >= len(t.shape) {
		return []int{-1}, errors.New("invalid axis")
	}

	axisSize := t.shape[axis]
	if len(t.data)%axisSize != 0 {
		return []int{-1}, errors.New("length of t.data is not a multiple of axisSize")
	}

	numSlices := len(t.data) / axisSize

	maxIndices := make([]int, numSlices)
	maxValues := make([]float64, numSlices)

	for i := 0; i < numSlices; i++ {
		start := i * axisSize
		end := (i + 1) * axisSize

		maxIndex := 0
		maxValue := t.data[start]

		for j := start + 1; j < end; j++ {
			if t.data[j] > maxValue {
				maxValue = t.data[j]
				maxIndex = j - start
			}
		}

		maxIndices[i] = maxIndex
		maxValues[i] = maxValue
	}

	return maxIndices, nil
}

// String return a Matrix representation in a human-friendly format.
func (t *Tensor) String() string {
	var sb strings.Builder
	for i := 0; i < len(t.data); i++ {
		sb.WriteString(fmt.Sprintf("%.2f\t", t.data[i]))
		if (i+1)%t.shape[len(t.shape)-1] == 0 {
			sb.WriteString("\n")
		}
	}
	return sb.String()
}

func equalShapes(shape1, shape2 []int) bool {
	if len(shape1) != len(shape2) {
		return false
	}
	for i, dim := range shape1 {
		if shape2[i] != dim {
			return false
		}
	}
	return true
}

// Concat function concatenates multiple tensors into a new tensor
func Concat(tensors ...*Tensor) (*Tensor, error) {
	// Check if there are any tensors to concatenate
	if len(tensors) == 0 {
		return nil, errors.New("no tensors to concatenate")
	}

	// Create a copy of the shape of the first tensor as the target shape
	targetShape := append([]int(nil), tensors[0].shape...)

	// Get the concatenation axis
	concatDim := targetShape[len(targetShape)-1]

	// Iterate over the remaining tensors and compare their shapes with the target shape
	for _, tensor := range tensors[1:] {
		// Exclude the last dimension for comparison
		if !equalShapes(targetShape[:len(targetShape)-1], tensor.shape[:len(tensor.shape)-1]) {
			return nil, errors.New("tensors have incompatible shapes for concatenation")
		}
		// Calculate the new size of the concatenation axis
		concatDim += tensor.shape[len(tensor.shape)-1]
	}

	// Create a new tensor with an empty data slice and the target shape
	result := &Tensor{make([]float64, 0, len(tensors)*concatDim), targetShape}

	// Copy data from input tensors to the result tensor
	for _, tensor := range tensors {
		result.data = append(result.data, tensor.data...)
	}

	return result, nil
}

// DotProduct calculates the dot product of two tensors.
// It checks if the shapes of the tensors are compatible and returns an error if they are not.
//
// Example Usage:
//
//	a := &Tensor{
//	  data:  []float64{1, 2, 3},
//	  shape: []int{3},
//	}
//	b := &Tensor{
//	  data:  []float64{4, 5, 6},
//	  shape: []int{3},
//	}
//	result, err := DotProduct(a, b)
//	if err != nil {
//	  fmt.Println(err)
//	  return
//	}
//	fmt.Println(result.data) // Output: [32]
//
// Inputs:
//
//	a: A pointer to a Tensor struct representing the first tensor.
//	b: A pointer to a Tensor struct representing the second tensor.
//
// Outputs:
//
//	result: A pointer to a Tensor struct representing the dot product result.
//	err: An error indicating if the shapes of the tensors are incompatible.
func DotProduct(a, b *Tensor) (*Tensor, error) {
	// Check if the tensors are nil
	if a == nil || b == nil {
		return nil, fmt.Errorf("input tensors cannot be nil")
	}

	// Check if the shapes of both tensors are compatible for dot product
	if !compatibleShapes(a.shape, b.shape) {
		return nil, fmt.Errorf("incompatible shapes for dot product: %v and %v", a.shape, b.shape)
	}

	// Calculate the dot product
	resultShape := []int{1}
	result := &Tensor{
		data:  make([]float64, 1), // Dot product results in a single scalar
		shape: resultShape,
	}

	// Compute the dot product
	dotProduct := new(big.Float)
	for i := 0; i < len(a.data); i++ {
		aFloat := new(big.Float).SetFloat64(a.data[i])
		bFloat := new(big.Float).SetFloat64(b.data[i])
		dotProduct.Add(dotProduct, new(big.Float).Mul(aFloat, bFloat))
	}

	result.data[0], _ = dotProduct.Float64()
	return result, nil
}

// Helper function to check if shapes are compatible
func compatibleShapes(shapeA, shapeB []int) bool {
	if len(shapeA) != len(shapeB) {
		return false
	}
	for i := range shapeA {
		if shapeA[i] != shapeB[i] {
			return false
		}
	}
	return true
}
