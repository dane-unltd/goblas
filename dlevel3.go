package blasw

import "github.com/gonum/blas"

func Dgemm(tA, tB blas.Transpose, alpha float64, A, B General, beta float64, C General) {
	var m, n, k int
	if tA == blas.NoTrans {
		m, k = A.M, A.N
	} else {
		m, k = A.N, A.M
	}
	if tB == blas.NoTrans {
		n = B.N
	} else {
		n = B.M
	}
	impl.Dgemm(order, tA, tB, m, n, k, alpha, A.Data, A.Stride,
		B.Data, B.Stride, beta, C.Data, C.Stride)
}

func Dsymm(s blas.Side, alpha float64, A Symmetric, B General, beta float64, C General) {
	var m, n int
	if s == blas.Left {
		m = A.N
		n = B.N
	} else {
		m = B.M
		n = A.N
	}
	impl.Dsymm(order, s, A.Uplo, m, n, alpha, A.Data, A.Stride,
		B.Data, B.Stride, beta, C.Data, C.Stride)
}

func Dsyrk(t blas.Transpose, alpha float64, A General, beta float64, C Symmetric) {
	var n, k int
	if t == blas.NoTrans {
		n, k = A.M, A.N
	} else {
		n, k = A.N, A.M
	}
	impl.Dsyrk(order, C.Uplo, t, n, k, alpha, A.Data, A.Stride, beta,
		C.Data, C.Stride)
}

func Dsyr2k(t blas.Transpose, alpha float64, A, B General, beta float64, C Symmetric) {
	var n, k int
	if t == blas.NoTrans {
		n, k = A.M, A.N
	} else {
		n, k = A.N, A.M
	}
	impl.Dsyr2k(order, C.Uplo, t, n, k, alpha, A.Data, A.Stride,
		B.Data, B.Stride, beta, C.Data, C.Stride)
}

func Dtrmm(s blas.Side, tA blas.Transpose, alpha float64, A Triangular, B General) {
	impl.Dtrmm(order, s, A.Uplo, tA, A.Diag, B.M, B.N, alpha, A.Data, A.Stride,
		B.Data, B.Stride)
}

func Dtrsm(s blas.Side, tA blas.Transpose, alpha float64, A Triangular, B General) {
	impl.Dtrsm(order, s, A.Uplo, tA, A.Diag, B.M, B.N, alpha, A.Data, A.Stride,
		B.Data, B.Stride)
}
