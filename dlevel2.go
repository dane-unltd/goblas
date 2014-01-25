package goblas

import "github.com/gonum/blas"

func Dgemv(tA blas.Transpose, alpha float64, A General, x Vector, beta float64, y Vector) {
	must(A.Check())
	must(x.Check())
	if tA == blas.NoTrans {
		if x.N != A.N {
			panic("blas: dimension mismatch")
		}
	} else if tA == blas.Trans {
		if x.N != A.M {
			panic("blas: dimension mismatch")
		}
	} else {
		panic("blas: illegal value for tA")
	}

	impl.Dgemv(order, tA, A.M, A.N, alpha, A.Data, A.Stride, x.Data, x.Inc, beta, y.Data, y.Inc)
}

func Dgbmv(tA blas.Transpose, alpha float64, A GeneralBanded, x Vector, beta float64, y Vector) {
	impl.Dgbmv(order, tA, A.M, A.N, A.KL, A.KU, alpha, A.Data,
		A.Stride, x.Data, x.Inc, beta, y.Data, y.Inc)
}

func Dtrmv(tA blas.Transpose, A Triangular, x Vector) {
	impl.Dtrmv(order, A.Uplo, tA, A.Diag, A.N, A.Data, A.Stride, x.Data, x.Inc)
}

func Dtbmv(tA blas.Transpose, A TriangularBanded, x Vector) {
	impl.Dtbmv(order, A.Uplo, tA, A.Diag, A.N, A.K, A.Data, A.Stride, x.Data, x.Inc)
}

func Dtpmv(tA blas.Transpose, A TriangularPacked, x Vector) {
	impl.Dtpmv(order, A.Uplo, tA, A.Diag, A.N, A.Data, x.Data, x.Inc)
}

func Dtrsv(tA blas.Transpose, A Triangular, x Vector) {
	impl.Dtrsv(order, A.Uplo, tA, A.Diag, A.N, A.Data, A.Stride, x.Data, x.Inc)
}

func Dtbsv(tA blas.Transpose, A TriangularBanded, x Vector) {
	impl.Dtbsv(order, A.Uplo, tA, A.Diag, A.N, A.K, A.Data, A.Stride, x.Data, x.Inc)
}
func Dtpsv(tA blas.Transpose, A TriangularPacked, x Vector) {
	impl.Dtpsv(order, A.Uplo, tA, A.Diag, A.N, A.Data, x.Data, x.Inc)
}

func Dsymv(alpha float64, A Symmetric, x Vector, beta float64, y Vector) {
	impl.Dsymv(order, A.Uplo, A.N, alpha, A.Data, A.Stride, x.Data, x.Inc, beta, y.Data, y.Inc)
}

func Dsbmv(alpha float64, A SymmetricBanded, x Vector, beta float64, y Vector) {
	impl.Dsbmv(order, A.Uplo, A.N, A.K, alpha, A.Data, A.Stride, x.Data,
		x.Inc, beta, y.Data, y.Inc)
}

func Dspmv(alpha float64, A SymmetricPacked, x Vector, beta float64, y Vector) {
	impl.Dspmv(order, A.Uplo, A.N, alpha, A.Data, x.Data, x.Inc, beta, y.Data, y.Inc)
}

func Dger(alpha float64, x Vector, y Vector, A General) {
	impl.Dger(order, A.M, A.N, alpha, x.Data, x.Inc, y.Data, y.Inc, A.Data, A.Stride)
}

func Dsyr(alpha float64, x Vector, A Symmetric) {
	impl.Dsyr(order, A.Uplo, A.N, alpha, x.Data, x.Inc, A.Data, A.Stride)
}

func Dspr(alpha float64, x Vector, A SymmetricPacked) {
	impl.Dspr(order, A.Uplo, A.N, alpha, x.Data, x.Inc, A.Data)
}

func Dsyr2(alpha float64, x Vector, y Vector, A Symmetric) {
	impl.Dsyr2(order, A.Uplo, A.N, alpha, x.Data, x.Inc, y.Data, y.Inc, A.Data, A.Stride)
}

func Dspr2(alpha float64, x Vector, y Vector, A SymmetricPacked) {
	impl.Dspr2(order, A.Uplo, A.N, alpha, x.Data, x.Inc, y.Data, y.Inc, A.Data)
}
