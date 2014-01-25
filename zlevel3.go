package blasw

import "github.com/gonum/blas"

func Zgemm(tA, tB blas.Transpose, alpha complex128, A, B GeneralCmplx, beta complex128, C GeneralCmplx) {
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
	implCmplx.Zgemm(order, tA, tB, m, n, k, alpha, A.Data, A.Stride,
		B.Data, B.Stride, beta, C.Data, C.Stride)
}

func Zsymm(s blas.Side, alpha complex128, A SymmetricCmplx, B GeneralCmplx, beta complex128, C GeneralCmplx) {
	var m, n int
	if s == blas.Left {
		m = A.N
		n = B.N
	} else {
		m = B.M
		n = A.N
	}
	implCmplx.Zsymm(order, s, A.Uplo, m, n, alpha, A.Data, A.Stride,
		B.Data, B.Stride, beta, C.Data, C.Stride)
}

func Zsyrk(t blas.Transpose, alpha complex128, A GeneralCmplx, beta complex128, C SymmetricCmplx) {
	var n, k int
	if t == blas.NoTrans {
		n, k = A.M, A.N
	} else {
		n, k = A.N, A.M
	}
	implCmplx.Zsyrk(order, C.Uplo, t, n, k, alpha, A.Data, A.Stride, beta,
		C.Data, C.Stride)
}

func Zsyr2k(t blas.Transpose, alpha complex128, A, B GeneralCmplx, beta complex128, C SymmetricCmplx) {
	var n, k int
	if t == blas.NoTrans {
		n, k = A.M, A.N
	} else {
		n, k = A.N, A.M
	}
	implCmplx.Zsyr2k(order, C.Uplo, t, n, k, alpha, A.Data, A.Stride,
		B.Data, B.Stride, beta, C.Data, C.Stride)
}

func Ztrmm(s blas.Side, tA blas.Transpose, alpha complex128, A TriangularCmplx, B GeneralCmplx) {
	implCmplx.Ztrmm(order, s, A.Uplo, tA, A.Diag, B.M, B.N, alpha, A.Data, A.Stride,
		B.Data, B.Stride)
}

func Ztrsm(s blas.Side, tA blas.Transpose, alpha complex128, A TriangularCmplx, B GeneralCmplx) {
	implCmplx.Ztrsm(order, s, A.Uplo, tA, A.Diag, B.M, B.N, alpha, A.Data, A.Stride,
		B.Data, B.Stride)
}

func Zhemm(s blas.Side, alpha complex128, A Hermitian, B GeneralCmplx, beta complex128, C GeneralCmplx) {
	var m, n int
	if s == blas.Left {
		m = A.N
		n = B.N
	} else {
		m = B.M
		n = A.N
	}
	implCmplx.Zhemm(order, s, A.Uplo, m, n, alpha, A.Data, A.Stride,
		B.Data, B.Stride, beta, C.Data, C.Stride)
}

func Zherk(t blas.Transpose, alpha float64, A GeneralCmplx, beta float64, C Hermitian) {
	var n, k int
	if t == blas.NoTrans {
		n, k = A.M, A.N
	} else {
		n, k = A.N, A.M
	}
	implCmplx.Zherk(order, C.Uplo, t, n, k, alpha, A.Data, A.Stride, beta,
		C.Data, C.Stride)
}

func Zher2k(t blas.Transpose, alpha complex128, A, B GeneralCmplx, beta float64, C Hermitian) {
	var n, k int
	if t == blas.NoTrans {
		n, k = A.M, A.N
	} else {
		n, k = A.N, A.M
	}
	implCmplx.Zher2k(order, C.Uplo, t, n, k, alpha, A.Data, A.Stride,
		B.Data, B.Stride, beta, C.Data, C.Stride)
}
