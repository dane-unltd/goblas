package goblas

import (
	"errors"
	"github.com/gonum/blas"
)

type General struct {
	Data   []float64
	M, N   int
	Stride int
}

func NewGeneral(m, n int, data []float64) General {
	var A General
	if order == blas.RowMajor {
		A = General{data, m, n, n}
	} else {
		A = General{data, m, n, m}
	}
	must(A.Check())
	return A
}

func (A General) Check() error {
	if A.N < 0 {
		return errors.New("blas: n < 0")
	}
	if A.M < 0 {
		return errors.New("blas: m < 0")
	}
	if A.Stride < 1 {
		return errors.New("blas: illegal stride")
	}
	if order == blas.ColMajor {
		if A.Stride < A.M {
			return errors.New("blas: illegal stride")
		}
		if (A.N-1)*A.Stride+A.M > len(A.Data) {
			return errors.New("blas: insufficient amount of data")
		}
	} else if order == blas.RowMajor {
		if A.Stride < A.N {
			return errors.New("blas: illegal stride")
		}
		if (A.M-1)*A.Stride+A.N > len(A.Data) {
			return errors.New("blas: insufficient amount of data")
		}
	}
	return nil
}

func (A General) Row(i int) Vector {
	if i >= A.M || i < 0 {
		panic("blas: index out of range")
	}
	if order == blas.RowMajor {
		return Vector{A.Data[A.Stride*i:], A.N, 1}
	} else {
		return Vector{A.Data[i:], A.N, A.Stride}
	}
	panic("unreachable")
}

func (A General) Col(i int) Vector {
	if i >= A.N || i < 0 {
		panic("blas: index out of range")
	}
	if order == blas.RowMajor {
		return Vector{A.Data[i:], A.M, A.Stride}
	} else {
		return Vector{A.Data[A.Stride*i:], A.M, 1}
	}
	panic("unreachable")
}

func (A General) Sub(i, j, r, c int) General {
	must(A.Check())
	if i >= A.M || i < 0 {
		panic("blas: index out of range")
	}
	if j >= A.N || i < 0 {
		panic("blas: index out of range")
	}
	if r < 0 || c < 0 {
		panic("blas: r < 0 or c < 0")
	}
	return General{A.Data[index(i, j, A.Stride):], r, c, A.Stride}
}

type GeneralBanded struct {
	General
	KL, KU int
}

type Triangular struct {
	Data   []float64
	N      int
	Stride int
	Uplo   blas.Uplo
	Diag   blas.Diag
}

type TriangularBanded struct {
	Data   []float64
	N, K   int
	Stride int
	Uplo   blas.Uplo
	Diag   blas.Diag
}

type TriangularPacked struct {
	Data []float64
	N    int
	Uplo blas.Uplo
	Diag blas.Diag
}

type Symmetric struct {
	Data      []float64
	N, Stride int
	Uplo      blas.Uplo
}

type SymmetricBanded struct {
	Data         []float64
	N, K, Stride int
	Uplo         blas.Uplo
}

type SymmetricPacked struct {
	Data []float64
	N    int
	Uplo blas.Uplo
}

type Vector struct {
	Data []float64
	N    int
	Inc  int
}

func NewVector(v []float64) Vector {
	return Vector{v, len(v), 1}
}

func (v Vector) Check() error {
	if v.N < 0 {
		return errors.New("blas: n < 0")
	}
	if v.Inc == 0 {
		return errors.New("blas: zero x index increment")
	}
	if (v.N-1)*v.Inc >= len(v.Data) {
		return errors.New("blas: index out of range")
	}
	return nil
}

func Ge2Tr(A General, d blas.Diag, ul blas.Uplo) Triangular {
	n := A.M
	if A.N < n {
		n = A.N
	}
	return Triangular{A.Data, n, A.Stride, ul, d}
}

func Ge2Sy(A General, ul blas.Uplo) Symmetric {
	n := A.M
	if A.N < n {
		n = A.N
	}
	return Symmetric{A.Data, n, A.Stride, ul}
}
func must(err error) {
	if err != nil {
		panic(err)
	}
}

func index(i, j, ld int) int {
	if order == blas.RowMajor {
		return i*ld + j
	} else {
		return i + j*ld
	}
}
