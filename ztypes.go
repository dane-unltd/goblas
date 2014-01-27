package goblas

import (
	"errors"
	"github.com/gonum/blas"
)

type GeneralCmplx struct {
	Data   []complex128
	M, N   int
	Stride int
}

func NewGeneralCmplx(m, n int, data []complex128) GeneralCmplx {
	var A GeneralCmplx
	if order == blas.RowMajor {
		A = GeneralCmplx{data, m, n, n}
	} else {
		A = GeneralCmplx{data, m, n, m}
	}
	must(A.Check())
	return A
}

func (A GeneralCmplx) Check() error {
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

func (A GeneralCmplx) Row(i int) VectorCmplx {
	if i >= A.M || i < 0 {
		panic("blas: index out of range")
	}
	if order == blas.RowMajor {
		return VectorCmplx{A.Data[A.Stride*i:], A.N, 1}
	} else {
		return VectorCmplx{A.Data[i:], A.N, A.Stride}
	}
	panic("unreachable")
}

func (A GeneralCmplx) Col(i int) VectorCmplx {
	if i >= A.N || i < 0 {
		panic("blas: index out of range")
	}
	if order == blas.RowMajor {
		return VectorCmplx{A.Data[i:], A.M, A.Stride}
	} else {
		return VectorCmplx{A.Data[A.Stride*i:], A.M, 1}
	}
	panic("unreachable")
}

func (A GeneralCmplx) Sub(i, j, r, c int) GeneralCmplx {
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
	return GeneralCmplx{A.Data[index(i, j, A.Stride):], r, c, A.Stride}
}

type GeneralCmplxBand struct {
	GeneralCmplx
	KL, KU int
}

type TriangularCmplx struct {
	Data   []complex128
	N      int
	Stride int
	Uplo   blas.Uplo
	Diag   blas.Diag
}

type TriangularCmplxBand struct {
	Data   []complex128
	N, K   int
	Stride int
	Uplo   blas.Uplo
	Diag   blas.Diag
}

type TriangularCmplxPacked struct {
	Data []complex128
	N    int
	Uplo blas.Uplo
	Diag blas.Diag
}

type SymmetricCmplx struct {
	Data      []complex128
	N, Stride int
	Uplo      blas.Uplo
}

type Hermitian struct {
	Data      []complex128
	N, Stride int
	Uplo      blas.Uplo
}

type HermitianBand struct {
	Data         []complex128
	N, K, Stride int
	Uplo         blas.Uplo
}

type HermitianPacked struct {
	Data []complex128
	N    int
	Uplo blas.Uplo
}

type VectorCmplx struct {
	Data []complex128
	N    int
	Inc  int
}

func NewVectorCmplx(v []complex128) VectorCmplx {
	return VectorCmplx{v, len(v), 1}
}

func (v VectorCmplx) Check() error {
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

func Gec2Trc(A GeneralCmplx, d blas.Diag, ul blas.Uplo) TriangularCmplx {
	n := A.M
	if A.N < n {
		n = A.N
	}
	return TriangularCmplx{A.Data, n, A.Stride, ul, d}
}

func Gec2He(A GeneralCmplx, ul blas.Uplo) Hermitian {
	n := A.M
	if A.N < n {
		n = A.N
	}
	return Hermitian{A.Data, n, A.Stride, ul}
}
